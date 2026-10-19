import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Label = 0 | 1;
export type Pt = { x: [number, number]; label: Label };
export type DatasetId = "blocks" | "diagonal";
export type TreeConfig = { dataset: DatasetId; maxDepth: number; minSamples: number; noise: number; seed: number };
export type Region = { x0: number; x1: number; y0: number; y1: number };
export type Candidate = { t: number; g: number; nL: number; nR: number };
export type Split = { feature: 0 | 1; threshold: number; g: number; giniL: number; giniR: number; nL: number; nR: number; left: number; right: number };
export type TreeNode = {
  id: number; depth: number; region: Region; idx: number[]; counts: [number, number]; gini: number; prediction: Label;
  split?: Split; leafReason?: string; parent?: number;
};

export const LO = 0, HI = 10, N_TRAIN = 40, N_TEST = 40, MAX_DEPTH = 8;
export const FEATURE = ["x₁", "x₂"] as const;
export const CLASS_NAME = ["A", "B"] as const;

const truth: Record<DatasetId, (x: number, y: number) => Label> = {
  blocks: (x, y) => ((x > 4 && y > 5) || x > 7.5 ? 1 : 0),
  diagonal: (x, y) => (y > x ? 1 : 0),
};
export const DATASETS: { value: DatasetId; label: string }[] = [
  { value: "blocks", label: "Blocks (axis-aligned rule)" },
  { value: "diagonal", label: "Diagonal (x₂ > x₁)" },
];

/** Uniform points in [0, 10]², labelled by the dataset's rule, then each label flipped with probability `noise`. */
export function generate(dataset: DatasetId, n: number, noise: number, seed: number): Pt[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => {
    const x = Math.round(r.next() * 100) / 10, y = Math.round(r.next() * 100) / 10;
    const clean = truth[dataset](x, y);
    const flip = r.next() < noise;
    return { x: [x, y] as [number, number], label: (flip ? 1 - clean : clean) as Label };
  });
}

export function gini(counts: readonly number[]) {
  const n = counts.reduce((s, v) => s + v, 0);
  return n === 0 ? 0 : 1 - counts.reduce((s, c) => s + (c / n) ** 2, 0);
}

const countOf = (pts: Pt[], idx: number[]): [number, number] => {
  const c: [number, number] = [0, 0];
  for (const i of idx) c[pts[i].label]++;
  return c;
};

/** Every midpoint between consecutive distinct values of one feature, with the weighted child Gini it would give. */
export function candidates(pts: Pt[], idx: number[], f: 0 | 1): Candidate[] {
  const vals = [...new Set(idx.map((i) => pts[i].x[f]))].sort((a, b) => a - b);
  const out: Candidate[] = [];
  for (let k = 0; k < vals.length - 1; k++) {
    const t = (vals[k] + vals[k + 1]) / 2;
    const L = idx.filter((i) => pts[i].x[f] <= t), R = idx.filter((i) => pts[i].x[f] > t);
    const g = (L.length * gini(countOf(pts, L)) + R.length * gini(countOf(pts, R))) / idx.length;
    out.push({ t, g, nL: L.length, nR: R.length });
  }
  return out;
}

/** Lowest weighted Gini across both features; ties go to x₁, then to the smaller threshold. */
export function bestSplit(cands: [Candidate[], Candidate[]]) {
  let best: { feature: 0 | 1; c: Candidate } | null = null;
  for (const f of [0, 1] as const) for (const c of cands[f]) if (!best || c.g < best.c.g - 1e-12) best = { feature: f, c };
  return best;
}

export type SearchInfo = { nodeId: number; cands: [Candidate[], Candidate[]]; bestPerFeature: [Candidate | null, Candidate | null] };

/** Greedy CART, breadth-first, so the tree grows level by level exactly as the steps show it. */
export function buildTree(pts: Pt[], maxDepth: number, minSamples: number) {
  const nodes: TreeNode[] = [];
  const order: { kind: "split"; search: SearchInfo; nodeId: number } [] = [];
  const leafNotes: { nodeId: number; reason: string; afterSplit: number }[] = [];
  const make = (idx: number[], depth: number, region: Region, parent?: number): TreeNode => {
    const counts = countOf(pts, idx);
    const node: TreeNode = { id: nodes.length, depth, region, idx, counts, gini: gini(counts), prediction: counts[1] > counts[0] ? 1 : 0, parent };
    nodes.push(node);
    return node;
  };
  const queue = [make(pts.map((_, i) => i), 0, { x0: LO, x1: HI, y0: LO, y1: HI })];
  while (queue.length) {
    const node = queue.shift()!;
    const leaf = (reason: string) => { node.leafReason = reason; leafNotes.push({ nodeId: node.id, reason, afterSplit: order.length }); };
    if (node.gini === 0) { leaf("pure: every sample has the same class"); continue; }
    if (node.depth >= maxDepth) { leaf(`max depth ${maxDepth} reached`); continue; }
    if (node.idx.length < minSamples) { leaf(`only ${node.idx.length} samples (< min ${minSamples} to split)`); continue; }
    const cands: [Candidate[], Candidate[]] = [candidates(pts, node.idx, 0), candidates(pts, node.idx, 1)];
    const best = bestSplit(cands);
    if (!best || best.c.g >= node.gini - 1e-12) { leaf("no split lowers the impurity"); continue; }
    const f = best.feature, t = best.c.t;
    const Li = node.idx.filter((i) => pts[i].x[f] <= t), Ri = node.idx.filter((i) => pts[i].x[f] > t);
    const r = node.region;
    const left = make(Li, node.depth + 1, f === 0 ? { ...r, x1: t } : { ...r, y1: t }, node.id);
    const right = make(Ri, node.depth + 1, f === 0 ? { ...r, x0: t } : { ...r, y0: t }, node.id);
    node.split = { feature: f, threshold: t, g: best.c.g, giniL: left.gini, giniR: right.gini, nL: Li.length, nR: Ri.length, left: left.id, right: right.id };
    const bestOf = (fi: 0 | 1) => cands[fi].reduce<Candidate | null>((a, c) => (!a || c.g < a.g - 1e-12 ? c : a), null);
    order.push({ kind: "split", nodeId: node.id, search: { nodeId: node.id, cands, bestPerFeature: [bestOf(0), bestOf(1)] } });
    queue.push(left, right);
  }
  return { nodes, order, leafNotes };
}

export function predict(nodes: TreeNode[], x: [number, number]): Label {
  let n = nodes[0];
  while (n.split) n = nodes[x[n.split.feature] <= n.split.threshold ? n.split.left : n.split.right];
  return n.prediction;
}

export const accuracy = (nodes: TreeNode[], pts: Pt[]) => (pts.length ? pts.filter((p) => predict(nodes, p.x) === p.label).length / pts.length : Number.NaN);

export type TreeState = {
  phase: "data" | "split" | "evaluate";
  headline: string;
  sub: string;
  /** Nodes on screen once this step's animation finishes. */
  visible: number[];
  /** Children that appear during this step. */
  added: number[];
  search?: SearchInfo;
};

const f3 = (v: number) => v.toFixed(3);

export function simulate(raw: TreeConfig) {
  const cfg: TreeConfig = {
    ...raw,
    maxDepth: clamp(Math.round(raw.maxDepth), 1, MAX_DEPTH),
    minSamples: clamp(Math.round(raw.minSamples), 2, N_TRAIN),
    noise: clamp(raw.noise, 0, 0.45),
  };
  const train = generate(cfg.dataset, N_TRAIN, cfg.noise, cfg.seed);
  // The test set comes from a different seed stream of the same distribution: never seen during growth.
  const test = generate(cfg.dataset, N_TEST, cfg.noise, cfg.seed + 7919);
  const { nodes, order, leafNotes } = buildTree(train, cfg.maxDepth, cfg.minSamples);
  const trainAcc = accuracy(nodes, train), testAcc = accuracy(nodes, test);
  const depthSweep = Array.from({ length: MAX_DEPTH }, (_, i) => {
    const t = buildTree(train, i + 1, cfg.minSamples).nodes;
    return { depth: i + 1, train: accuracy(t, train), test: accuracy(t, test), leaves: t.filter((n) => !n.split).length };
  });
  const root = nodes[0];
  const steps: SimulationStep<TreeState>[] = [];
  const visible = [0];
  const notesAfter = (k: number): SimulationEvent[] =>
    leafNotes.filter((l) => l.afterSplit === k).map((l) => ({ id: `leaf-${l.nodeId}`, type: "leaf", message: `Node #${l.nodeId} becomes a leaf (predicts ${CLASS_NAME[nodes[l.nodeId].prediction]}): ${l.reason}.` }));

  steps.push({
    id: "data",
    title: "Training data and the root node",
    description: `${N_TRAIN} labelled points: ${root.counts[0]} of class A and ${root.counts[1]} of class B. Before any split, the whole plane is one node with Gini impurity ${f3(root.gini)}.`,
    state: { phase: "data", headline: `${N_TRAIN} labelled points, one big node`, sub: `Gini impurity = ${f3(root.gini)}: how mixed the classes are (0 = pure, 0.5 = 50/50).`, visible: [0], added: [] },
    events: [{ id: "root", type: "gini", message: `Gini(root) = 1 − (${root.counts[0]}/${N_TRAIN})² − (${root.counts[1]}/${N_TRAIN})² = ${f3(root.gini)}`, tex: `G_{\\text{root}} = 1 - \\left(\\tfrac{${root.counts[0]}}{${N_TRAIN}}\\right)^2 - \\left(\\tfrac{${root.counts[1]}}{${N_TRAIN}}\\right)^2 = ${f3(root.gini)}` }],
  });

  order.forEach((o, k) => {
    const n = nodes[o.nodeId], s = n.split!;
    visible.push(s.left, s.right);
    const fn = FEATURE[s.feature];
    const tried = o.search.cands[0].length + o.search.cands[1].length;
    steps.push({
      id: `split-${n.id}`,
      title: `Split node #${n.id}: ${fn} ≤ ${s.threshold.toFixed(2)}`,
      description: `The algorithm tries all ${tried} candidate thresholds (midpoints between neighbouring values) on both features, and keeps the one with the lowest weighted Gini: ${fn} ≤ ${s.threshold.toFixed(2)} gives ${f3(s.g)} (down from ${f3(n.gini)}).`,
      state: {
        phase: "split",
        headline: `Node #${n.id}: try every threshold, keep the purest split`,
        sub: `A line sweeps across ${FEATURE[0]}, then ${FEATURE[1]}; the curve below/right is the weighted Gini for each position. Lowest wins.`,
        visible: [...visible], added: [s.left, s.right], search: o.search,
      },
      events: [
        ...notesAfter(k),
        { id: `best-${n.id}`, type: "split", message: `best: ${fn} ≤ ${s.threshold.toFixed(2)}, weighted Gini = ${s.nL}/${n.idx.length}·${f3(s.giniL)} + ${s.nR}/${n.idx.length}·${f3(s.giniR)} = ${f3(s.g)}`,
          tex: `G_{\\text{split}} = \\tfrac{${s.nL}}{${n.idx.length}}\\cdot ${f3(s.giniL)} + \\tfrac{${s.nR}}{${n.idx.length}}\\cdot ${f3(s.giniR)} = ${f3(s.g)}` },
        { id: `gain-${n.id}`, type: "gain", message: `impurity decrease = ${f3(n.gini)} − ${f3(s.g)} = ${f3(n.gini - s.g)}` },
      ],
    });
  });

  const leaves = nodes.filter((n) => !n.split).length;
  const best = depthSweep.reduce((a, b) => (b.test > a.test ? b : a));
  const gap = trainAcc - testAcc;
  steps.push({
    id: "evaluate",
    title: "Evaluate on unseen test points",
    description: `The finished tree has ${leaves} leaves and depth ${Math.max(...nodes.map((n) => n.depth))}. Training accuracy ${(trainAcc * 100).toFixed(0)}%, test accuracy ${(testAcc * 100).toFixed(0)}% on ${N_TEST} fresh points.`,
    state: {
      phase: "evaluate",
      headline: gap > 0.1 ? `Train ${(trainAcc * 100).toFixed(0)}% vs test ${(testAcc * 100).toFixed(0)}%: the tree memorised noise` : `Train ${(trainAcc * 100).toFixed(0)}% vs test ${(testAcc * 100).toFixed(0)}%`,
      sub: "Hollow markers are new test points. A red ring marks a test point the tree gets wrong.",
      visible: [...visible], added: [],
    },
    events: [
      ...notesAfter(order.length),
      { id: "acc", type: "metric", message: `train accuracy ${(trainAcc * 100).toFixed(1)}%, test accuracy ${(testAcc * 100).toFixed(1)}%`, severity: gap > 0.1 ? "warning" : "info" },
      { id: "sweep", type: "sweep", message: `best test accuracy over depths 1–${MAX_DEPTH}: ${(best.test * 100).toFixed(1)}% at depth ${best.depth}` },
    ],
  });

  return { cfg, train, test, nodes, steps, trainAcc, testAcc, depthSweep, leaves };
}

export type TreeSim = ReturnType<typeof simulate>;
