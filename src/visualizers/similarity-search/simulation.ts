import { clamp, cosine, dot, euclidean, norm } from "../../lib/math";
import type { SimulationStep } from "../../types/simulation";

export type Vec2 = [number, number];
export type Item = { id: string; v: Vec2; source: string };
export type Metric = "cosine" | "dot" | "l2";
export type DatasetId = "default" | "ties" | "dupes";

/** Hand-placed 2-D teaching vectors. Lengths differ on purpose so the three metrics disagree on raw vectors. */
export const DATASETS: Record<DatasetId, { name: string; query: Vec2; k: number; items: Item[]; note: string }> = {
  default: {
    name: "Mixed lengths", query: [1.6, 1.2], k: 3,
    note: "Vectors of very different lengths: dot product favours the long ones, cosine ignores length, L2 favours whatever is physically closest.",
    items: [
      ["A", [2.0, 1.0]], ["B", [1.0, 1.8]], ["C", [3.8, 2.6]], ["D", [0.6, 0.5]], ["E", [-1.5, 2.0]], ["F", [2.6, -0.6]],
      ["G", [-2.4, -1.0]], ["H", [0.4, -2.2]], ["I", [-0.8, -3.0]], ["J", [1.4, 2.6]], ["K", [-3.2, 1.0]], ["L", [2.2, 2.0]],
    ].map(([id, v]) => ({ id: id as string, v: v as Vec2, source: id as string })),
  },
  ties: {
    name: "Tie at the cutoff", query: [2, 0], k: 3,
    note: "C and D are mirror images around the query: identical distance, cosine and dot product. Only a tie-break rule decides which one makes top-3.",
    items: [
      ["A", [2.5, 0.3]], ["B", [1.6, -0.4]], ["C", [2, 1.5]], ["D", [2, -1.5]], ["E", [-1, 2.5]], ["F", [-2, -1]], ["G", [0.5, -3]], ["H", [-2.8, 1.2]],
    ].map(([id, v]) => ({ id: id as string, v: v as Vec2, source: id as string })),
  },
  dupes: {
    name: "Near-duplicate crowding", query: [1.5, 1.5], k: 3,
    note: "Four copies of the same FAQ (indexed from four mirrors) sit together. Exact kNN happily returns all of them and crowds out other sources.",
    items: ([
      ["F1", [1.8, 1.4], "faq"], ["F2", [1.82, 1.43], "faq"], ["F3", [1.78, 1.37], "faq"], ["F4", [1.85, 1.46], "faq"],
      ["G", [1.1, 2.0], "guide"], ["P", [2.2, 0.8], "policy"], ["X", [-1.6, 1.8], "x"], ["Y", [-2.2, -1.2], "y"], ["Z", [0.8, -2.6], "z"],
    ] as [string, Vec2, string][]).map(([id, v, source]) => ({ id, v, source })),
  },
};

export const METRIC_NAME: Record<Metric, string> = { cosine: "cosine", dot: "dot product", l2: "L2 distance" };
export const unit = (v: Vec2): Vec2 => { const n = norm(v); return n === 0 ? [0, 0] : [v[0] / n, v[1] / n]; };

/** Higher is better for every metric; L2 is negated so all lists sort the same way. */
export function score(metric: Metric, q: Vec2, v: Vec2) {
  return metric === "cosine" ? cosine(q, v) : metric === "dot" ? dot(q, v) : -euclidean(q, v);
}

export type Ranked = { item: Item; score: number; rank: number; tiedWithNext: boolean };
const EPS = 1e-9;

/** Exact (brute-force) search: score every item, sort descending, break ties by id so the result is deterministic. */
export function rank(q: Vec2, items: Item[], metric: Metric, normalized = false): Ranked[] {
  const qq = normalized ? unit(q) : q;
  const sorted = items
    .map((item) => ({ item, score: score(metric, qq, normalized ? unit(item.v) : item.v) }))
    .sort((a, b) => (Math.abs(b.score - a.score) > EPS ? b.score - a.score : a.item.id.localeCompare(b.item.id)));
  return sorted.map((s, i) => ({ ...s, rank: i + 1, tiedWithNext: i + 1 < sorted.length && Math.abs(sorted[i + 1].score - s.score) <= EPS }));
}

/** Top-k, optionally keeping only the best item per source (a simple near-duplicate collapse). */
export function topK(ranked: Ranked[], k: number, onePerSource: boolean) {
  const out: Ranked[] = [];
  const seen = new Set<string>();
  for (const r of ranked) {
    if (out.length >= k) break;
    if (onePerSource && seen.has(r.item.source)) continue;
    seen.add(r.item.source);
    out.push(r);
  }
  return out;
}

export type SimConfig = { dataset: DatasetId; query: Vec2; k: number; metric: Metric; normalize: boolean; onePerSource: boolean };
export type SSPhase = "scene" | "scan" | "capture" | "metrics" | "normalize" | "cutoff";
export type SSState = { phase: SSPhase; captured: number };

export function simulate(cfgIn: SimConfig) {
  const ds = DATASETS[cfgIn.dataset];
  const items = ds.items;
  const k = clamp(Math.round(cfgIn.k), 1, Math.min(6, items.length));
  const q: Vec2 = norm(cfgIn.query) === 0 ? ds.query : cfgIn.query; // a zero query has no direction; cosine is undefined
  const byL2 = rank(q, items, "l2");
  const raw = { cosine: rank(q, items, "cosine"), dot: rank(q, items, "dot"), l2: byL2 };
  const unitRanks = { cosine: rank(q, items, "cosine", true), dot: rank(q, items, "dot", true), l2: rank(q, items, "l2", true) };
  const finalRanked = (cfgIn.normalize ? unitRanks : raw)[cfgIn.metric];
  const final = topK(finalRanked, k, cfgIn.onePerSource);
  const sources = new Set(final.map((r) => r.item.source)).size;
  const kth = finalRanked[k - 1];
  const tieAtCutoff = !cfgIn.onePerSource && kth?.tiedWithNext ? finalRanked[k] : undefined;

  const steps: SimulationStep<SSState>[] = [];
  const dist = (r: Ranked) => (-r.score).toFixed(3);
  steps.push({ id: "scene", title: "A query among stored vectors", description: `${items.length} stored vectors and the query q = [${q.join(", ")}]. Exact search compares q with every one of them.`, state: { phase: "scene", captured: 0 }, events: [] });
  steps.push({
    id: "scan", title: "Measure the distance to every vector", description: `Brute force: ${items.length} distance computations, no index, no shortcuts. That is why exact kNN is O(N·d) per query.`,
    state: { phase: "scan", captured: 0 },
    events: byL2.map((r) => ({ id: `d-${r.item.id}`, type: "distance", message: `‖q − ${r.item.id}‖ = ${dist(r)}`, tex: `\\lVert q - ${r.item.id} \\rVert = ${dist(r)}` })),
  });
  for (let i = 1; i <= k; i++) {
    const r = byL2[i - 1];
    steps.push({
      id: `capture-${i}`, title: `Ring grows to neighbour #${i}`, description: `The ring expands to radius ${dist(r)} and captures ${r.item.id}${r.tiedWithNext ? ` (tied with ${byL2[i].item.id} at the same distance)` : ""}.`,
      state: { phase: "capture", captured: i },
      events: [{ id: `c-${i}`, type: "capture", message: `#${i}: ${r.item.id} at distance ${dist(r)}`, severity: r.tiedWithNext ? "warning" : "info" }],
    });
  }
  const agree = (m: Record<Metric, Ranked[]>) => m.cosine.slice(0, k).map((r) => r.item.id).join() === m.dot.slice(0, k).map((r) => r.item.id).join() && m.dot.slice(0, k).map((r) => r.item.id).join() === m.l2.slice(0, k).map((r) => r.item.id).join();
  steps.push({
    id: "metrics", title: "Same vectors, three metrics", description: agree(raw) ? `On these raw vectors all three metrics happen to agree on the top-${k}.` : `On raw vectors the metrics disagree: top-1 is ${raw.cosine[0].item.id} by cosine, ${raw.dot[0].item.id} by dot, ${raw.l2[0].item.id} by L2.`,
    state: { phase: "metrics", captured: k }, events: [],
  });
  steps.push({
    id: "normalize", title: "Normalize every vector to length 1", description: "Each vector is divided by its length and slides onto the unit circle. Now dot = cosine and L2² = 2 − 2·cosine, so all three rankings become identical.",
    state: { phase: "normalize", captured: k }, events: [{ id: "norm", type: "identity", message: "‖a − b‖² = 2 − 2 cos(a, b) for unit vectors", tex: "\\lVert a - b \\rVert^2 = 2 - 2\\cos(a, b)" }],
  });
  steps.push({
    id: "cutoff", title: `Cut at k = ${k}`, description: `Top-${k} by ${METRIC_NAME[cfgIn.metric]}${cfgIn.normalize ? " on normalized vectors" : " on raw vectors"}: ${final.map((r) => r.item.id).join(", ")}. ${sources} distinct source${sources === 1 ? "" : "s"}.`,
    state: { phase: "cutoff", captured: k },
    events: [
      ...(tieAtCutoff ? [{ id: "tie", type: "tie", severity: "warning" as const, message: `${kth.item.id} and ${tieAtCutoff.item.id} tie at the cutoff; the id tie-break keeps ${kth.item.id}. Another engine may keep the other.` }] : []),
      ...(sources < final.length ? [{ id: "dup", type: "crowding", severity: "warning" as const, message: `Only ${sources} distinct source(s) in top-${k}: near-duplicates are crowding the results.` }] : []),
    ],
  });
  return { steps, q, k, items, byL2, raw, unitRanks, final, finalRanked, sources, tieAtCutoff };
}
