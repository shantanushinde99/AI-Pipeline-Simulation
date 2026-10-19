import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Pt = [number, number];
export type IndexType = "flat" | "ivf" | "hnsw";
export const INDEX_INFO: Record<IndexType, { name: string; short: string }> = {
  flat: { name: "Flat (brute force)", short: "Compare the query with every stored vector. Exact, cost grows linearly with N." },
  ivf: { name: "IVF (inverted file)", short: "Cluster vectors into cells with k-means; at query time scan only the nprobe nearest cells." },
  hnsw: { name: "HNSW-lite (graph)", short: "Navigate a layered proximity graph greedily, from a sparse top layer down to the full bottom layer." },
};

export type VdbParams = { index: IndexType; k: number; nlist: number; nprobe: number; m: number; ef: number; query: Pt };
export const QUERIES: { id: string; label: string; q: Pt }[] = [
  { id: "cluster", label: "Inside a cluster", q: [27, 70] },
  { id: "border", label: "Near a cell border", q: [38, 58] },
  { id: "sparse", label: "Sparse region", q: [86, 88] },
];
export const DEFAULT_VDB: VdbParams = { index: "ivf", k: 5, nlist: 8, nprobe: 2, m: 5, ef: 10, query: QUERIES[1].q };

export function sanitize(p: VdbParams): VdbParams {
  const nlist = Math.max(2, Math.min(16, Math.round(p.nlist)));
  const k = Math.max(1, Math.min(20, Math.round(p.k)));
  return {
    ...p, k, nlist, nprobe: Math.max(1, Math.min(nlist, Math.round(p.nprobe))), m: Math.max(2, Math.min(10, Math.round(p.m))),
    ef: Math.max(k, Math.min(100, Math.round(p.ef))), query: [Math.max(0, Math.min(100, p.query[0])), Math.max(0, Math.min(100, p.query[1]))],
  };
}

export const d2 = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
export const dist = (a: Pt, b: Pt) => Math.sqrt(d2(a, b));

/** 200 seeded 2-D teaching vectors: six Gaussian clusters plus uniform background noise. */
export function makePoints(seed = 7, n = 200): Pt[] {
  const r = rng(seed);
  const centers: Pt[] = [[25, 72], [70, 75], [50, 40], [20, 25], [80, 30], [55, 90]];
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    if (i % 10 === 9) { pts.push([2 + r.next() * 96, 2 + r.next() * 96]); continue; }
    const c = centers[i % centers.length];
    pts.push([Math.max(1, Math.min(99, c[0] + r.gaussian() * 7)), Math.max(1, Math.min(99, c[1] + r.gaussian() * 7))]);
  }
  return pts.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
}

const byDist = (q: Pt, pts: Pt[]) => (a: number, b: number) => d2(q, pts[a]) - d2(q, pts[b]) || a - b;

export type SearchResult = { ids: number[]; computations: number; recall: number };
export const recallAt = (ids: number[], truth: number[]) => (truth.length ? ids.filter((i) => truth.includes(i)).length / truth.length : 1);

// ---------- flat ----------
export function flatSearch(pts: Pt[], q: Pt, k: number): SearchResult {
  const ids = pts.map((_, i) => i).sort(byDist(q, pts)).slice(0, k);
  return { ids, computations: pts.length, recall: 1 };
}

// ---------- IVF ----------
export type Ivf = { history: Pt[][]; centroids: Pt[]; assign: number[] };
export function trainIvf(pts: Pt[], nlist: number, iters = 8, seed = 3): Ivf {
  const r = rng(seed);
  const picks: number[] = [];
  while (picks.length < Math.min(nlist, pts.length)) {
    const i = Math.floor(r.next() * pts.length);
    if (!picks.includes(i)) picks.push(i);
  }
  let cents: Pt[] = picks.map((i) => [...pts[i]] as Pt);
  const history: Pt[][] = [cents];
  const nearest = (p: Pt, cs: Pt[]) => cs.reduce((b, c, j) => (d2(p, c) < d2(p, cs[b]) ? j : b), 0);
  for (let it = 0; it < iters; it++) {
    const assign = pts.map((p) => nearest(p, cents));
    cents = cents.map((c, j) => {
      const mine = pts.filter((_, i) => assign[i] === j);
      return mine.length ? [mine.reduce((s, p) => s + p[0], 0) / mine.length, mine.reduce((s, p) => s + p[1], 0) / mine.length] as Pt : c;
    });
    history.push(cents);
  }
  return { history, centroids: cents, assign: pts.map((p) => nearest(p, cents)) };
}

export type IvfSearch = SearchResult & { centroidOrder: number[]; probed: number[]; scanned: number[] };
export function ivfSearch(pts: Pt[], ivf: Ivf, q: Pt, k: number, nprobe: number): IvfSearch {
  const centroidOrder = ivf.centroids.map((_, j) => j).sort((a, b) => d2(q, ivf.centroids[a]) - d2(q, ivf.centroids[b]) || a - b);
  const probed = centroidOrder.slice(0, nprobe);
  const scanned = pts.map((_, i) => i).filter((i) => probed.includes(ivf.assign[i]));
  const ids = [...scanned].sort(byDist(q, pts)).slice(0, k);
  return { ids, computations: ivf.centroids.length + scanned.length, recall: 0, centroidOrder, probed, scanned };
}

/** Convex Voronoi cell of centroid i, clipped to the [0,100]² box (half-plane intersection). */
export function voronoiCell(cents: Pt[], i: number): Pt[] {
  let poly: Pt[] = [[0, 0], [100, 0], [100, 100], [0, 100]];
  const c = cents[i];
  cents.forEach((o, j) => {
    if (j === i || (o[0] === c[0] && o[1] === c[1])) return;
    const n: Pt = [o[0] - c[0], o[1] - c[1]];
    const lim = (d2(o, [0, 0]) - d2(c, [0, 0])) / 2; // keep p·n <= lim (closer to c than to o)
    const f = (p: Pt) => p[0] * n[0] + p[1] * n[1] - lim;
    const out: Pt[] = [];
    poly.forEach((p, a) => {
      const q = poly[(a + 1) % poly.length];
      const fp = f(p), fq = f(q);
      if (fp <= 0) out.push(p);
      if ((fp <= 0) !== (fq <= 0)) { const t = fp / (fp - fq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
    });
    poly = out;
  });
  return poly;
}

// ---------- HNSW-lite ----------
export type Hnsw = { levels: number[]; maxLevel: number; entry: number; /** adjacency per layer: layer → node → neighbours */ adj: Map<number, number[]>[] };
/**
 * Levels drawn like HNSW (P(level ≥ l) = (1/M)^l, capped at 2). Nodes are inserted in storage order; on each of its
 * layers a node links (both ways) to its M nearest already-inserted nodes, so early nodes get long-range links.
 * Real HNSW finds those neighbours with a search (ef_construction) and prunes links with a diversity heuristic.
 */
export function buildHnsw(pts: Pt[], m: number, seed = 11): Hnsw {
  const r = rng(seed);
  const mL = 1 / Math.log(Math.max(2, m));
  const levels = pts.map(() => Math.min(2, Math.floor(-Math.log(Math.max(1e-12, r.next())) * mL)));
  const maxLevel = Math.max(...levels);
  const adj: Map<number, number[]>[] = [];
  for (let l = 0; l <= maxLevel; l++) {
    const g = new Map<number, Set<number>>();
    for (let i = 0; i < pts.length; i++) {
      if (levels[i] < l) continue;
      const earlier = [...g.keys()].sort(byDist(pts[i], pts)).slice(0, m);
      g.set(i, new Set(earlier));
      for (const j of earlier) g.get(j)!.add(i);
    }
    adj.push(new Map([...g].map(([i, set]) => [i, [...set].sort((a, b) => a - b)])));
  }
  const entry = levels.findIndex((l) => l === maxLevel);
  return { levels, maxLevel, entry, adj };
}

export type Hop = { from: number; to: number };
export type LayerTrace = { layer: number; hops: Hop[]; evaluated: number[]; start: number; end: number };
export type HnswSearch = SearchResult & { layers: LayerTrace[] };

export function hnswSearch(pts: Pt[], h: Hnsw, q: Pt, k: number, ef: number): HnswSearch {
  const seen = new Set<number>();
  let computations = 0;
  const dq = (i: number) => { if (!seen.has(i)) { seen.add(i); computations++; } return d2(q, pts[i]); };
  let ep = h.entry;
  dq(ep);
  const layers: LayerTrace[] = [];
  for (let l = h.maxLevel; l >= 1; l--) {
    const hops: Hop[] = [], evaluated: number[] = [];
    const start = ep;
    let improved = true;
    while (improved) {
      improved = false;
      let best = ep;
      for (const nb of h.adj[l].get(ep) ?? []) {
        if (!seen.has(nb)) evaluated.push(nb);
        if (dq(nb) < dq(best)) best = nb;
      }
      if (best !== ep) { hops.push({ from: ep, to: best }); ep = best; improved = true; }
    }
    layers.push({ layer: l, hops, evaluated, start, end: ep });
  }
  // Layer 0: best-first beam search keeping the `ef` closest found so far.
  const visited = new Set<number>([ep]);
  const cand = [ep];
  let res = [ep];
  const hops: Hop[] = [], evaluated: number[] = [];
  const parent = new Map<number, number>();
  const start = ep;
  while (cand.length) {
    cand.sort((a, b) => dq(a) - dq(b));
    const c = cand.shift()!;
    const worst = res.reduce((w, i) => (dq(i) > dq(w) ? i : w), res[0]);
    if (res.length >= ef && dq(c) > dq(worst)) break;
    if (parent.has(c)) hops.push({ from: parent.get(c)!, to: c });
    for (const nb of h.adj[0].get(c) ?? []) {
      if (visited.has(nb)) continue;
      visited.add(nb);
      if (!seen.has(nb)) evaluated.push(nb);
      const w = res.reduce((x, i) => (dq(i) > dq(x) ? i : x), res[0]);
      if (res.length < ef || dq(nb) < dq(w)) {
        cand.push(nb);
        parent.set(nb, c);
        res.push(nb);
        if (res.length > ef) {
          const far = res.reduce((x, j) => (dq(j) > dq(x) ? j : x), res[0]);
          res = res.filter((i) => i !== far);
        }
      }
    }
  }
  const ids = [...res].sort(byDist(q, pts)).slice(0, k);
  layers.push({ layer: 0, hops, evaluated, start, end: ids[0] ?? ep });
  return { ids, computations, recall: 0, layers };
}

/** Where a new vector goes in each index. */
export type Insert = { p: Pt; ivfCell: number; ivfComputations: number; level: number; links: { layer: number; to: number[] }[]; hnswComputations: number };
export function insertPoint(pts: Pt[], ivf: Ivf, h: Hnsw, p: Pt, m: number): Insert {
  const order = ivf.centroids.map((_, j) => j).sort((a, b) => d2(p, ivf.centroids[a]) - d2(p, ivf.centroids[b]));
  const level = 1; // fixed so the new node visibly joins two layers
  const links = [];
  let comps = 0;
  for (let l = Math.min(level, h.maxLevel); l >= 0; l--) {
    const nodes = pts.map((_, i) => i).filter((i) => h.levels[i] >= l);
    comps += nodes.length; // ponytail: exhaustive neighbour pick for clarity; real HNSW finds them with the same greedy search (ef_construction)
    links.push({ layer: l, to: nodes.sort(byDist(p, pts)).slice(0, m) });
  }
  return { p, ivfCell: order[0], ivfComputations: ivf.centroids.length, level, links, hnswComputations: comps };
}

export type VdbPhase = "data" | "scan" | "train" | "assign" | "probe" | "scan-cells" | "build" | "layer" | "topk" | "compare" | "insert";
export type VdbState = { phase: VdbPhase; layer?: number };

export function runAll(pIn: VdbParams) {
  const p = sanitize(pIn);
  const pts = makePoints();
  const truth = flatSearch(pts, p.query, p.k);
  const ivf = trainIvf(pts, p.nlist);
  const iv = ivfSearch(pts, ivf, p.query, p.k, p.nprobe);
  iv.recall = recallAt(iv.ids, truth.ids);
  const h = buildHnsw(pts, p.m);
  const hs = hnswSearch(pts, h, p.query, p.k, p.ef);
  hs.recall = recallAt(hs.ids, truth.ids);
  return { p, pts, truth, ivf, iv, h, hs, insert: insertPoint(pts, ivf, h, [62, 58], p.m) };
}
export type VdbRun = ReturnType<typeof runAll>;

/** Recall/cost trade-off curves for the current query: IVF over nprobe, HNSW over ef. */
export function tradeoff(run: VdbRun) {
  const { pts, ivf, h, p, truth } = run;
  const ivfCurve = Array.from({ length: p.nlist }, (_, i) => { const r = ivfSearch(pts, ivf, p.query, p.k, i + 1); return { x: r.computations, y: recallAt(r.ids, truth.ids), label: `nprobe ${i + 1}` }; });
  const efs = [...new Set([p.k, 8, 12, 16, 24, 32, 48, 64].filter((e) => e >= p.k))].sort((a, b) => a - b);
  const hnswCurve = efs.map((ef) => { const r = hnswSearch(pts, h, p.query, p.k, ef); return { x: r.computations, y: recallAt(r.ids, truth.ids), label: `ef ${ef}` }; });
  return { ivfCurve, hnswCurve, flat: { x: pts.length, y: 1, label: "flat" } };
}

export function simulate(pIn: VdbParams) {
  const run = runAll(pIn);
  const { p, pts, truth, ivf, iv, h, hs, insert } = run;
  let e = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id: `e${e++}`, type: "log", message, severity, tex });
  const S = (id: string, title: string, state: VdbState, description: string, events: SimulationEvent[]): SimulationStep<VdbState> => ({ id, title, state, description, events });
  const res = p.index === "flat" ? truth : p.index === "ivf" ? iv : hs;
  const common = {
    data: S("data", "The stored vectors and a query", { phase: "data" }, `${pts.length} vectors are stored (2-D here so you can see them). The query (✕) wants its ${p.k} nearest neighbours.`, [ev(`${pts.length} vectors, k = ${p.k}, query (${p.query[0]}, ${p.query[1]})`)]),
    topk: S("topk", `Top-${p.k} and recall`, { phase: "topk" }, `The index returns its ${p.k} best candidates. Recall@${p.k} compares them with the exact answer from a flat scan.`,
      [ev(`returned ${res.ids.join(", ")}`), ev(`recall@${p.k} = ${res.recall.toFixed(2)} with ${res.computations} distance computations`, res.recall < 1 ? "warning" : "info", `\\text{recall@}${p.k} = ${res.recall.toFixed(2)}`)]),
    compare: S("compare", "Recall vs. cost", { phase: "compare" }, "Every index is a trade-off between distance computations (latency) and recall. Flat sits at 100% recall and maximum cost.", [ev(`flat ${truth.computations} · IVF ${iv.computations} (recall ${iv.recall.toFixed(2)}) · HNSW ${hs.computations} (recall ${hs.recall.toFixed(2)})`)]),
  };
  let steps: SimulationStep<VdbState>[];
  if (p.index === "flat") {
    steps = [
      common.data,
      S("scan", "Compute every distance", { phase: "scan" }, "A flat index has no structure: it computes the distance from the query to all stored vectors, one by one.", [ev(`${pts.length} distance computations`)]),
      common.topk, common.compare,
      S("insert", "Insert a new vector", { phase: "insert" }, "Inserting into a flat index is just appending a row: no training, no links. Every future query pays one more distance.", [ev(`appended vector ${pts.length}; N = ${pts.length + 1}`)]),
    ];
  } else if (p.index === "ivf") {
    steps = [
      common.data,
      S("train", "Train k-means centroids", { phase: "train" }, `Offline, k-means places ${p.nlist} centroids: assign each vector to its nearest centroid, move each centroid to the mean of its vectors, repeat.`, [ev(`${ivf.history.length - 1} k-means iterations`)]),
      S("assign", "Each vector joins a cell", { phase: "assign" }, "Every vector is filed under its nearest centroid. The regions closest to each centroid (Voronoi cells) partition the space.", ivf.centroids.map((_, j) => ev(`cell ${j}: ${ivf.assign.filter((a) => a === j).length} vectors`))),
      S("probe", `Probe the ${p.nprobe} nearest cell${p.nprobe > 1 ? "s" : ""}`, { phase: "probe" }, `At query time, compare the query with the ${p.nlist} centroids only, and pick the nprobe = ${p.nprobe} closest cells.`, [ev(`${p.nlist} centroid distances; probing cells ${iv.probed.join(", ")}`)]),
      S("scan-cells", "Scan only those cells", { phase: "scan-cells" }, `Only vectors filed in the probed cells are compared: ${iv.scanned.length} of ${pts.length}.`, [ev(`${iv.scanned.length} vector distances (+${p.nlist} centroid distances)`)]),
      common.topk, common.compare,
      S("insert", "Insert a new vector", { phase: "insert" }, "A new vector is compared with the centroids and appended to the nearest cell's list. Centroids are not retrained, so heavy inserts slowly degrade the clustering.", [ev(`new vector → cell ${insert.ivfCell} (${insert.ivfComputations} distances)`)]),
    ];
  } else {
    const upper = hs.layers.filter((l) => l.layer > 0);
    steps = [
      common.data,
      S("build", "Build a layered graph", { phase: "build" }, `Each vector gets a random level (most stay on layer 0). On every layer it links to its M = ${p.m} nearest neighbours. Upper layers are sparse highways.`, [0, 1, 2].filter((l) => l <= h.maxLevel).map((l) => ev(`layer ${l}: ${h.levels.filter((x) => x >= l).length} nodes`))),
      ...upper.map((l) => S(`layer-${l.layer}`, `Greedy hops on layer ${l.layer}`, { phase: "layer", layer: l.layer }, `Start at ${l.start}; repeatedly move to whichever neighbour is closer to the query. Stop when no neighbour is closer: that node is the entry to layer ${l.layer - 1}.`, [ev(`${l.hops.length} hops, ${l.evaluated.length} new distances, ends at ${l.end}`)])),
      S("layer-0", "Beam search on layer 0", { phase: "layer", layer: 0 }, `On the full bottom layer, keep the ef = ${p.ef} best candidates found so far and expand them best-first until no candidate can improve the list.`, [ev(`${hs.layers.at(-1)!.hops.length} expansions, ${hs.layers.at(-1)!.evaluated.length} new distances`)]),
      common.topk, common.compare,
      S("insert", "Insert a new vector", { phase: "insert" }, `The new vector draws level ${insert.level}, so it joins layers 0–${insert.level}. On each it links to its M nearest neighbours, which also link back.`, insert.links.map((l) => ev(`layer ${l.layer}: link to ${l.to.join(", ")}`))),
    ];
  }
  return { steps, run };
}
