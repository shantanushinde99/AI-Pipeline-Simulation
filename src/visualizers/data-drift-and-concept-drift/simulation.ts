import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import { trainLogistic, type Model } from "../../scenarios/ml-model";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type DriftKind = "none" | "covariate" | "concept" | "both";
export const DRIFTS: { id: DriftKind; name: string }[] = [
  { id: "covariate", name: "Covariate shift: P(x) moves" },
  { id: "concept", name: "Concept drift: P(y|x) changes" },
  { id: "both", name: "Both at once" },
  { id: "none", name: "No drift (control)" },
];

export const BATCHES = 8;
export const N_REF = 240;
export const N_BATCH = 120;
export const MAX_SHIFT: [number, number] = [2.0, 1.4]; // where the inputs end up under full covariate shift
export const MAX_ANGLE = 55; // degrees the true boundary rotates under full concept drift
export const BINS = 5;

export type Pt = { x: [number, number]; y: 0 | 1 };
export type ScoredPt = Pt & { p: number; pred: 0 | 1 };

/** The real-world rule: a gently curved boundary x2 = 0.4·x1 + 0.25·x1², rotated by `angleDeg` about the origin. */
export function truth(x: [number, number], angleDeg: number): 0 | 1 {
  const a = (-angleDeg * Math.PI) / 180;
  const u = Math.cos(a) * x[0] - Math.sin(a) * x[1];
  const v = Math.sin(a) * x[0] + Math.cos(a) * x[1];
  return v - 0.4 * u - 0.25 * u * u > 0 ? 1 : 0;
}

/** Points along the true boundary (same count for every angle, so the SVG path can morph). */
export function truthCurve(angleDeg: number, n = 41): [number, number][] {
  const a = (angleDeg * Math.PI) / 180;
  return Array.from({ length: n }, (_, i) => {
    const u = -3 + (6 * i) / (n - 1);
    const v = 0.4 * u + 0.25 * u * u;
    return [Math.cos(a) * u - Math.sin(a) * v, Math.sin(a) * u + Math.cos(a) * v];
  });
}

export function drift(kind: DriftKind, t: number, magnitude: number) {
  const f = clamp(magnitude, 0, 1) * (t / BATCHES);
  const cov = kind === "covariate" || kind === "both";
  const con = kind === "concept" || kind === "both";
  return { shift: (cov ? [MAX_SHIFT[0] * f, MAX_SHIFT[1] * f] : [0, 0]) as [number, number], angle: con ? MAX_ANGLE * f : 0 };
}

export function referenceData(seed: number): Pt[] {
  const r = rng(seed);
  return Array.from({ length: N_REF }, () => {
    const x: [number, number] = [r.gaussian() * 0.7, r.gaussian() * 0.7];
    return { x, y: truth(x, 0) };
  });
}

/** Batch t: the same N_BATCH "users" (fixed base positions) with fresh jitter, moved by the covariate shift and labelled by the current concept. */
export function batchData(seed: number, t: number, kind: DriftKind, magnitude: number): Pt[] {
  const base = rng(seed + 1000);
  const jitter = rng(seed + 2000 + t);
  const { shift, angle } = drift(kind, t, magnitude);
  return Array.from({ length: N_BATCH }, () => {
    const x: [number, number] = [base.gaussian() * 0.7 + jitter.gaussian() * 0.12 + shift[0], base.gaussian() * 0.7 + jitter.gaussian() * 0.12 + shift[1]];
    return { x, y: truth(x, angle) };
  });
}

/** Bin edges at the reference quantiles, so each bin holds 1/BINS of the reference data. */
export function quantileEdges(values: number[], bins = BINS): number[] {
  const s = [...values].sort((a, b) => a - b);
  return Array.from({ length: bins - 1 }, (_, i) => s[Math.floor(((i + 1) * s.length) / bins)]);
}

export function binShares(values: number[], edges: number[]): number[] {
  const counts = new Array<number>(edges.length + 1).fill(0);
  for (const v of values) {
    let b = 0;
    while (b < edges.length && v >= edges[b]) b++;
    counts[b]++;
  }
  return counts.map((c) => (values.length ? c / values.length : 0));
}

/** Population Stability Index: Σ (a − e)·ln(a / e) over bins; empty bins are floored at 1e-4. */
export function psi(expected: number[], actual: number[]): number {
  return expected.reduce((s, e, i) => {
    const ee = Math.max(e, 1e-4), aa = Math.max(actual[i], 1e-4);
    return s + (aa - ee) * Math.log(aa / ee);
  }, 0);
}

/** Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two empirical CDFs. */
export function ksStat(a: number[], b: number[]): number {
  const xs = [...new Set([...a, ...b])].sort((p, q) => p - q);
  const sa = [...a].sort((p, q) => p - q), sb = [...b].sort((p, q) => p - q);
  let i = 0, j = 0, d = 0;
  for (const x of xs) {
    while (i < sa.length && sa[i] <= x) i++;
    while (j < sb.length && sb[j] <= x) j++;
    d = Math.max(d, Math.abs(i / sa.length - j / sb.length));
  }
  return d;
}
/** KS critical value at α = 0.05. */
export const ksCritical = (n: number, m: number) => 1.36 * Math.sqrt((n + m) / (n * m));

export type DriftInput = { kind: DriftKind; magnitude: number; labelDelay: number; psiThreshold: number; accThreshold: number; seed: number };
export const DEFAULT_INPUT: DriftInput = { kind: "covariate", magnitude: 1, labelDelay: 2, psiThreshold: 0.2, accThreshold: 0.85, seed: 11 };

export type BatchStats = { t: number; acc: number; psi: [number, number]; ks: [number, number]; shares: [number[], number[]]; shift: [number, number]; angle: number };

export type DriftState = {
  phase: "reference" | "batch" | "summary";
  t: number;
  points: ScoredPt[];
  stats: BatchStats[]; // batches 1..t (t = 0 has none)
  measured: { t: number; acc: number }[]; // accuracies whose labels have arrived by now
  featureAlert: boolean;
  labelAlert: boolean;
  headline: string;
  sub: string;
};

export type DriftRun = { model: Model; reference: Pt[]; refAcc: number; edges: [number[], number[]]; refShares: [number[], number[]]; steps: SimulationStep<DriftState>[]; firstFeatureAlert?: number; firstLabelAlert?: number };

const score = (m: Model, pts: Pt[]): ScoredPt[] => pts.map((p) => { const pr = m.prob(p.x); return { ...p, p: pr, pred: pr >= 0.5 ? 1 : 0 }; });
const accuracy = (pts: ScoredPt[]) => (pts.length ? pts.filter((p) => p.pred === p.y).length / pts.length : 0);

export function simulate(raw: DriftInput): DriftRun {
  const input = { ...raw, magnitude: clamp(raw.magnitude, 0, 1), labelDelay: Math.round(clamp(raw.labelDelay, 0, BATCHES)) };
  const { kind, magnitude, labelDelay, psiThreshold, accThreshold, seed } = input;
  const reference = referenceData(seed);
  const model = trainLogistic({ features: ["x1", "x2"], rows: reference.map((p) => ({ x: p.x, y: p.y })) }, { epochs: 600, lr: 0.5 });
  const refScored = score(model, reference);
  const refAcc = accuracy(refScored);
  const col = (pts: Pt[], j: 0 | 1) => pts.map((p) => p.x[j]);
  const edges: [number[], number[]] = [quantileEdges(col(reference, 0)), quantileEdges(col(reference, 1))];
  const refShares: [number[], number[]] = [binShares(col(reference, 0), edges[0]), binShares(col(reference, 1), edges[1])];

  const stats: BatchStats[] = [];
  const steps: SimulationStep<DriftState>[] = [];
  let firstFeatureAlert: number | undefined, firstLabelAlert: number | undefined;

  steps.push({
    id: "reference",
    title: "Train on reference data",
    description: `The model (logistic regression) is trained on ${N_REF} reference points. Its straight boundary matches the slightly curved true boundary well where the data lives: reference accuracy ${(refAcc * 100).toFixed(1)}%.`,
    state: { phase: "reference", t: 0, points: refScored, stats: [], measured: [], featureAlert: false, labelAlert: false,
      headline: "The model learns a boundary from the reference data", sub: "Filled dots are class 1, hollow dots class 0. Dashed line = model, curve = the real rule." },
    events: [{ id: "train", type: "train", message: `trained on ${N_REF} points, reference accuracy ${(refAcc * 100).toFixed(1)}%` }],
  });

  for (let t = 1; t <= BATCHES; t++) {
    const pts = score(model, batchData(seed, t, kind, magnitude));
    const { shift, angle } = drift(kind, t, magnitude);
    const shares: [number[], number[]] = [binShares(col(pts, 0), edges[0]), binShares(col(pts, 1), edges[1])];
    const s: BatchStats = {
      t, acc: accuracy(pts), shift, angle, shares,
      psi: [psi(refShares[0], shares[0]), psi(refShares[1], shares[1])],
      ks: [ksStat(col(reference, 0), col(pts, 0)), ksStat(col(reference, 1), col(pts, 1))],
    };
    stats.push(s);
    const measured = stats.filter((b) => b.t + labelDelay <= t).map((b) => ({ t: b.t, acc: b.acc }));
    const featureAlert = Math.max(...s.psi) >= psiThreshold;
    const labelAlert = measured.some((m) => m.acc < accThreshold);
    const newFeature = featureAlert && firstFeatureAlert === undefined;
    const newLabel = labelAlert && firstLabelAlert === undefined;
    if (newFeature) firstFeatureAlert = t;
    if (newLabel) firstLabelAlert = t;
    const arrived = measured.at(-1)?.t === t - labelDelay ? measured.at(-1) : undefined;
    const events: SimulationEvent[] = [
      { id: `psi-${t}`, type: "feature-test", severity: featureAlert ? "warning" : "info", message: `PSI x1=${s.psi[0].toFixed(3)}, x2=${s.psi[1].toFixed(3)} (alert ≥ ${psiThreshold})`,
        tex: `\\text{PSI}_{x_1} = ${s.psi[0].toFixed(3)},\\; \\text{PSI}_{x_2} = ${s.psi[1].toFixed(3)}` },
      arrived
        ? { id: `lab-${t}`, type: "label-monitor", severity: arrived.acc < accThreshold ? "error" : "info", message: `labels for batch ${arrived.t} arrived: accuracy ${(arrived.acc * 100).toFixed(1)}%` }
        : { id: `lab-${t}`, type: "label-monitor", message: `labels for batch ${t} are still on their way (${labelDelay}-batch delay)` },
    ];
    const what = kind === "none" ? "Nothing changed in the world" : kind === "covariate" ? "Inputs slide to a new region" : kind === "concept" ? "Same inputs, but some true labels flip" : "Inputs slide and labels flip";
    steps.push({
      id: `batch-${t}`,
      title: `Batch ${t}`,
      description: `${N_BATCH} new production requests. Covariate shift moved the input mean by (${shift[0].toFixed(2)}, ${shift[1].toFixed(2)}); the true boundary is rotated ${angle.toFixed(1)}°. True accuracy on this batch: ${(s.acc * 100).toFixed(1)}% (unknown until labels arrive).`,
      state: {
        phase: "batch", t, points: pts, stats: [...stats], measured, featureAlert, labelAlert,
        headline: `Batch ${t}: ${what}`,
        sub: `${featureAlert ? "Feature-distribution test: DRIFT" : "Feature-distribution test: no drift"} · ${measured.length ? `label monitor: ${labelAlert ? "ACCURACY ALERT" : "accuracy OK"}` : "label monitor: waiting for labels"}`,
      },
      events,
    });
  }

  const last = stats.at(-1)!;
  const caught = (v?: number) => (v === undefined ? "never" : `at batch ${v}`);
  steps.push({
    id: "summary",
    title: "Which detector caught it?",
    description: `Feature test (PSI ≥ ${psiThreshold}) alerted ${caught(firstFeatureAlert)}. Label-based monitor (accuracy < ${accThreshold * 100}%) alerted ${caught(firstLabelAlert)}, ${labelDelay} batch(es) after the data it judged. Final true accuracy: ${(last.acc * 100).toFixed(1)}%.`,
    state: {
      phase: "summary", t: BATCHES, points: steps.at(-1)!.state.points, stats: [...stats],
      measured: stats.filter((b) => b.t + labelDelay <= BATCHES).map((b) => ({ t: b.t, acc: b.acc })),
      featureAlert: firstFeatureAlert !== undefined, labelAlert: firstLabelAlert !== undefined,
      headline: kind === "concept" ? "PSI stayed quiet: the inputs never moved, only their meaning did"
        : kind === "covariate" ? "PSI saw the inputs move long before labels confirmed the damage"
        : kind === "both" ? "Both detectors fired, for different reasons" : "No drift: neither detector should fire",
      sub: `Feature test: ${caught(firstFeatureAlert)} · label monitor: ${caught(firstLabelAlert)} · true accuracy now ${(last.acc * 100).toFixed(0)}%`,
    },
    events: [{ id: "sum", type: "summary", severity: kind !== "none" && firstLabelAlert === undefined && firstFeatureAlert === undefined ? "error" : "info", message: `feature alert ${caught(firstFeatureAlert)}, label alert ${caught(firstLabelAlert)}` }],
  });

  return { model, reference, refAcc, edges, refShares, steps, firstFeatureAlert, firstLabelAlert };
}
