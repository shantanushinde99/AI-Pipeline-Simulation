import { clamp, euclidean } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Vec = [number, number];
export type Pt = { x: Vec; label: number };
export type DatasetId = "aligned" | "mismatch" | "overlap";
export type SuConfig = { dataset: DatasetId; k: number; dataSeed: number; initSeed: number };

export const N = 60, MAX_K = 6, MAX_ITERS = 12;
export const DATASETS: { value: DatasetId; label: string; classes: number }[] = [
  { value: "aligned", label: "Clusters match classes (3 blobs)", classes: 3 },
  { value: "mismatch", label: "Clusters ≠ classes (top/bottom blobs, left/right labels)", classes: 2 },
  { value: "overlap", label: "Overlapping classes", classes: 3 },
];

/** Gaussian blobs in [0, 10]². In "mismatch" the label is the side (x₁ > 5), not the blob. */
export function generate(dataset: DatasetId, seed: number): Pt[] {
  const r = rng(seed);
  const spec: Record<DatasetId, { c: Vec; s: Vec }[]> = {
    aligned: [{ c: [2.5, 7], s: [0.9, 0.9] }, { c: [7.5, 7], s: [0.9, 0.9] }, { c: [5, 2.6], s: [0.9, 0.9] }],
    mismatch: [{ c: [5, 7.6], s: [2.3, 0.7] }, { c: [5, 2.4], s: [2.3, 0.7] }],
    overlap: [{ c: [4, 6], s: [1.3, 1.3] }, { c: [6, 6], s: [1.3, 1.3] }, { c: [5, 4.2], s: [1.3, 1.3] }],
  };
  const blobs = spec[dataset];
  return Array.from({ length: N }, (_, i) => {
    const b = i % blobs.length;
    const x: Vec = [clamp(blobs[b].c[0] + blobs[b].s[0] * r.gaussian(), 0.2, 9.8), clamp(blobs[b].c[1] + blobs[b].s[1] * r.gaussian(), 0.2, 9.8)];
    const rounded: Vec = [Math.round(x[0] * 100) / 100, Math.round(x[1] * 100) / 100];
    return { x: rounded, label: dataset === "mismatch" ? (rounded[0] > 5 ? 1 : 0) : b };
  });
}

export const meanOf = (pts: Vec[]): Vec => [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
export const nearest = (x: Vec, centers: Vec[]) => centers.reduce((best, c, j) => (euclidean(x, c) < euclidean(x, centers[best]) ? j : best), 0);

/** Nearest-centroid classifier: one mean per class; predict the class whose mean is closest. */
export function fitNearestCentroid(pts: Pt[], classes: number) {
  const centroids = Array.from({ length: classes }, (_, c) => meanOf(pts.filter((p) => p.label === c).map((p) => p.x)));
  const predict = (x: Vec) => nearest(x, centroids);
  const acc = pts.filter((p) => predict(p.x) === p.label).length / pts.length;
  return { centroids, predict, acc, counts: centroids.map((_, c) => pts.filter((p) => p.label === c).length) };
}

export type Iteration = { assign: number[]; before: Vec[]; after: Vec[]; inertia: number; changed: number; moved: number[] };

export const inertiaOf = (xs: Vec[], assign: number[], centers: Vec[]) => xs.reduce((s, x, i) => s + euclidean(x, centers[assign[i]]) ** 2, 0);

/** Lloyd's k-means with Forgy initialisation (k distinct random points). Empty clusters keep their old centroid. */
export function kMeans(xs: Vec[], k: number, seed: number) {
  const r = rng(seed);
  const idx = xs.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const initIdx = idx.slice(0, k);
  let centers: Vec[] = initIdx.map((i) => [...xs[i]] as Vec);
  let prev: number[] | null = null;
  const iters: Iteration[] = [];
  for (let it = 0; it < MAX_ITERS; it++) {
    const assign = xs.map((x) => nearest(x, centers));
    const changed = prev ? assign.filter((a, i) => a !== prev![i]).length : xs.length;
    if (prev && changed === 0) break;
    const after = centers.map((c, j) => {
      const members = xs.filter((_, i) => assign[i] === j);
      return members.length ? meanOf(members) : c;
    });
    iters.push({ assign, before: centers, after, inertia: inertiaOf(xs, assign, after), changed, moved: after.map((a, j) => euclidean(a, centers[j])) });
    centers = after;
    prev = assign;
  }
  return { initIdx, iters, final: centers, assign: iters.at(-1)!.assign, converged: iters.length < MAX_ITERS };
}

/** contingency[cluster][class] and purity = Σ_cluster max_class count / n. */
export function contingency(assign: number[], labels: number[], k: number, classes: number) {
  const table = Array.from({ length: k }, () => new Array<number>(classes).fill(0));
  assign.forEach((a, i) => table[a][labels[i]]++);
  const purity = table.reduce((s, row) => s + Math.max(...row), 0) / assign.length;
  const majority = table.map((row) => row.indexOf(Math.max(...row)));
  return { table, purity, majority };
}

export type SuPhase = "data" | "sup-means" | "sup-boundary" | "km-init" | "km-iter" | "compare";
export type SuState = { phase: SuPhase; headline: string; sub: string; iter?: number };

export function simulate(raw: SuConfig) {
  const cfg: SuConfig = { ...raw, k: clamp(Math.round(raw.k), 1, MAX_K) };
  const classes = DATASETS.find((d) => d.value === cfg.dataset)!.classes;
  const pts = generate(cfg.dataset, cfg.dataSeed);
  const xs = pts.map((p) => p.x);
  const sup = fitNearestCentroid(pts, classes);
  const km = kMeans(xs, cfg.k, cfg.initSeed);
  const ct = contingency(km.assign, pts.map((p) => p.label), cfg.k, classes);
  const f2 = (v: number) => v.toFixed(2);
  const steps: SimulationStep<SuState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "compute", message, severity, tex });

  steps.push({ id: "data", title: "Same points, with and without labels", description: `${N} points with two features. On the left each point keeps its class label (${classes} classes). On the right the labels are hidden: the algorithm only sees coordinates.`,
    state: { phase: "data", headline: `The same ${N} points, two ways`, sub: "Left: supervised (labels visible). Right: unsupervised (labels hidden, every point looks the same)." },
    events: [ev("d", `${N} points, ${classes} classes; class sizes ${sup.counts.join(" / ")}`)] });
  steps.push({ id: "sup-means", title: "Supervised: average each class", description: `Because the labels are known, training is simple: average the points of each class. Class means: ${sup.centroids.map((c, i) => `class ${i + 1} (${f2(c[0])}, ${f2(c[1])})`).join(", ")}.`,
    state: { phase: "sup-means", headline: "Supervised: the labels tell us which points belong together", sub: "Each class's points send their coordinates to the class average (the star)." },
    events: sup.centroids.map((c, i) => ev(`m${i}`, `μ_${i + 1} = mean of ${sup.counts[i]} points = (${f2(c[0])}, ${f2(c[1])})`, undefined, `\\mu_{${i + 1}} = \\tfrac{1}{${sup.counts[i]}}\\textstyle\\sum x_i = (${f2(c[0])},\\ ${f2(c[1])})`)) });
  steps.push({ id: "sup-boundary", title: "Supervised: draw the decision boundary", description: `Every location is assigned to the nearest class mean, which carves the plane into regions. ${Math.round(sup.acc * 100)}% of the training points land in their own class's region.`,
    state: { phase: "sup-boundary", headline: `A boundary appears: nearest class mean wins (${Math.round(sup.acc * 100)}% correct)`, sub: "The shading shows what the classifier would predict for a new point anywhere on the plane." },
    events: [ev("acc", `training accuracy = ${(sup.acc * 100).toFixed(1)}%`, sup.acc < 0.8 ? "warning" : undefined)] });
  steps.push({ id: "km-init", title: `Unsupervised: drop ${cfg.k} starting centroids`, description: `k-means needs k up front (k = ${cfg.k}). It starts from ${cfg.k} randomly chosen points (init seed ${cfg.initSeed}). No labels are used anywhere.`,
    state: { phase: "km-init", headline: `k-means starts: ${cfg.k} centroid${cfg.k > 1 ? "s" : ""} dropped on random points`, sub: "Without labels, the only thing to go on is distance: which points sit close together." },
    events: [ev("init", `initial centroids = points #${km.initIdx.join(", #")}`)] });
  km.iters.forEach((it, i) => {
    steps.push({ id: `iter-${i}`, title: `k-means iteration ${i + 1}`, description: `Assign: every point joins its nearest centroid (${it.changed} point${it.changed === 1 ? "" : "s"} ${i ? "changed cluster" : "assigned"}). Update: each centroid moves to the mean of its points. Inertia (total squared distance) = ${it.inertia.toFixed(1)}.`,
      state: { phase: "km-iter", iter: i, headline: `Iteration ${i + 1}: recolour by nearest centroid, then move centroids to the mean`, sub: `${it.changed} point${it.changed === 1 ? "" : "s"} ${i ? "switched cluster" : "assigned"}; inertia ${it.inertia.toFixed(1)}. It stops when nothing switches.` },
      events: [
        ev(`a${i}`, `${it.changed} assignment change(s)`),
        ev(`j${i}`, `inertia J = ${it.inertia.toFixed(2)}`, undefined, `J = \\textstyle\\sum_i \\lVert x_i - \\mu_{c_i} \\rVert^2 = ${it.inertia.toFixed(2)}`),
        ...(it.after.some((_, j) => !it.assign.includes(j)) ? [ev(`e${i}`, "a centroid has no points (empty cluster); it stays where it is", "warning")] : []),
      ] });
  });
  const matches = cfg.k === classes && ct.purity >= 0.9;
  steps.push({ id: "compare", title: "Compare clusters with the hidden classes", description: `Revealing the labels: cluster purity is ${(ct.purity * 100).toFixed(0)}% (share of points that belong to their cluster's majority class). ${matches ? "Here the natural groups coincide with the classes." : "The clusters k-means found are not the classes: it groups by distance, and nothing tells it which grouping you care about."}`,
    state: { phase: "compare", headline: matches ? `Clusters line up with classes (purity ${(ct.purity * 100).toFixed(0)}%)` : `Clusters ≠ classes: purity ${(ct.purity * 100).toFixed(0)}%`, sub: "Shapes now show each point's true class; colours still show its cluster. A ring marks a point outside its cluster's majority class." },
    events: [
      ev("conv", km.converged ? `converged after ${km.iters.length} iteration(s)` : `stopped at the ${MAX_ITERS}-iteration cap`, km.converged ? undefined : "warning"),
      ev("pur", `purity = ${(ct.purity * 100).toFixed(1)}%`, matches ? undefined : "warning"),
    ] });
  return { cfg, classes, pts, sup, km, ct, steps };
}

export type SuSim = ReturnType<typeof simulate>;
