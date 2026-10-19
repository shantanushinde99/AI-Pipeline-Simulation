import { clamp, sigmoid } from "../../lib/math";
import { rng } from "../../lib/random";
import { trainLogistic, type Row } from "../../scenarios/ml-model";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { binShares, ksCritical, ksStat, psi } from "../data-drift-and-concept-drift/simulation";

export type DriftType = "none" | "covariate" | "concept" | "gradual";
export type DriftConfig = { kind: DriftType; magnitude: number; start: number; labelDelay: number; psiThreshold: number; accThreshold: number; retrain: boolean; seed: number };
export const DEFAULT_DRIFT: DriftConfig = { kind: "concept", magnitude: 1, start: 3, labelDelay: 2, psiThreshold: 0.2, accThreshold: 0.8, retrain: false, seed: 5 };

export const T = 12;
export const N_TRAIN = 400;
export const N_BATCH = 200;
/** Fixed equal-width bin edges on the feature; the two end bins are open-ended. */
export const EDGES = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5];
export const BOUNDARY0 = 0.5; // true P(y=1|x) = sigmoid(4(x − b)), b = 0.5 at training time
const SLOPE = 4;

/** Where the input mean and the true boundary sit at batch t (t = -1 is the training snapshot). */
export function driftAt(kind: DriftType, t: number, magnitude: number, start: number) {
  const on = t >= start;
  const ramp = clamp((t - start + 1) / (T - start), 0, 1);
  switch (kind) {
    case "covariate": return { mu: on ? 0.6 * magnitude : 0, b: BOUNDARY0 };
    case "concept": return { mu: 0, b: on ? BOUNDARY0 - magnitude : BOUNDARY0 };
    case "gradual": return { mu: 0.6 * magnitude * ramp, b: BOUNDARY0 - magnitude * ramp };
    default: return { mu: 0, b: BOUNDARY0 };
  }
}

export function sample(seed: number, n: number, mu: number, b: number): Row[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => {
    const x = Math.round((mu + r.gaussian()) * 1000) / 1000;
    return { x: [x], y: r.next() < sigmoid(SLOPE * (x - b)) ? 1 : 0 };
  });
}

export type Batch = {
  t: number; mu: number; b: number; xs: number[];
  shares: number[]; refShares: number[]; contrib: number[];
  psi: number; ks: number; ksCrit: number; ksAt: number; inputAlert: boolean;
  acc: number; version: number; // model version that scored it
  labelsAt: number; accAlert: boolean;
};
export type Retrain = { at: number; from: number[]; version: number; boundary: number };
export type DriftState = { t: number }; // -1 = training, T = summary
export type DriftSim = {
  cfg: DriftConfig; trainXs: number[]; refs: number[][]; trainAcc: number; boundaries: number[]; // model boundary per version
  batches: Batch[]; retrains: Retrain[]; firstInputAlert?: number; firstAccAlertKnown?: number;
  steps: SimulationStep<DriftState>[];
};

/** x where the 1-feature logistic model crosses p = 0.5. */
const boundaryOf = (m: ReturnType<typeof trainLogistic>) => m.mean[0] - (m.bias / (m.weights[0] || 1e-9)) * m.std[0];
const accuracy = (m: ReturnType<typeof trainLogistic>, rows: Row[]) => rows.filter((r) => (m.prob(r.x) >= 0.5 ? 1 : 0) === r.y).length / rows.length;

export function simulate(raw: DriftConfig): DriftSim {
  const cfg: DriftConfig = { ...raw, magnitude: clamp(raw.magnitude, 0, 2), start: Math.round(clamp(raw.start, 0, T - 1)), labelDelay: Math.round(clamp(raw.labelDelay, 0, 4)) };
  const L = cfg.labelDelay;
  const train = sample(cfg.seed, N_TRAIN, 0, BOUNDARY0);
  let model = trainLogistic({ features: ["x"], rows: train });
  const trainAcc = accuracy(model, train);
  let ref = train.map((r) => r.x[0]);
  let refShares = binShares(ref, EDGES);
  let version = 0;
  const boundaries = [boundaryOf(model)];
  const refs = [ref];
  const data: Row[][] = [];
  const batches: Batch[] = [];
  const retrains: Retrain[] = [];
  for (let t = 0; t < T; t++) {
    const { mu, b } = driftAt(cfg.kind, t, cfg.magnitude, cfg.start);
    const rows = sample(cfg.seed * 31 + 100 + t, N_BATCH, mu, b);
    data.push(rows);
    const xs = rows.map((r) => r.x[0]);
    const shares = binShares(xs, EDGES);
    const contrib = shares.map((a, i) => { const aa = Math.max(a, 1e-4), ee = Math.max(refShares[i], 1e-4); return (aa - ee) * Math.log(aa / ee); });
    // KS statistic and where along x the largest CDF gap sits (for drawing).
    const ks = ksStat(ref, xs);
    const grid = [...xs].sort((p, q) => p - q);
    const cdf = (arr: number[], v: number) => arr.filter((u) => u <= v).length / arr.length;
    let ksAt = grid[0], best = -1;
    for (const v of grid) { const g = Math.abs(cdf(ref, v) - cdf(xs, v)); if (g > best + 1e-12) { best = g; ksAt = v; } }
    const ksCrit = ksCritical(ref.length, xs.length);
    const p = psi(refShares, shares);
    batches.push({ t, mu, b, xs, shares, refShares, contrib, psi: p, ks, ksCrit, ksAt, inputAlert: p > cfg.psiThreshold || ks > ksCrit, acc: accuracy(model, rows), version, labelsAt: t + L, accAlert: false });
    // End of week t: labels for batch t − L arrive.
    const j = t - L;
    if (j >= 0) {
      const labelled = batches[j];
      labelled.accAlert = labelled.acc < cfg.accThreshold;
      if (cfg.retrain && labelled.accAlert && labelled.version === version) {
        const from = [j]; // ponytail: retrain on the latest labelled week only; a sliding window + holdout check is the real-world upgrade
        const rowsR = from.flatMap((k) => data[k]);
        model = trainLogistic({ features: ["x"], rows: rowsR });
        ref = rowsR.map((r) => r.x[0]);
        refShares = binShares(ref, EDGES);
        version++;
        boundaries.push(boundaryOf(model));
        refs.push(ref);
        retrains.push({ at: t, from, version, boundary: boundaryOf(model) });
      }
    }
  }
  const firstInputAlert = batches.find((x) => x.inputAlert)?.t;
  const firstAcc = batches.find((x) => x.accAlert);
  const firstAccAlertKnown = firstAcc ? firstAcc.labelsAt : undefined;

  const ev = (id: string, message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id, type: "monitor", message, severity, tex });
  const steps: SimulationStep<DriftState>[] = [{
    id: "train", title: "Train on a snapshot and keep its histogram as the reference",
    description: `${N_TRAIN} labelled rows. The model learns a boundary at x = ${boundaries[0].toFixed(2)} and gets ${(trainAcc * 100).toFixed(1)}% on its own data.`,
    state: { t: -1 }, events: [ev("tr", `reference histogram stored (${EDGES.length + 1} bins)`)],
  }];
  for (const x of batches) {
    const events: SimulationEvent[] = [
      ev(`b${x.t}`, `week ${x.t + 1}: PSI ${x.psi.toFixed(3)}, KS ${x.ks.toFixed(3)}`, x.inputAlert ? "warning" : "info", `\\text{PSI} = ${x.psi.toFixed(3)},\\; D = ${x.ks.toFixed(3)}`),
    ];
    if (x.inputAlert) events.push(ev(`a${x.t}`, `input-drift alert (${x.psi > cfg.psiThreshold ? "PSI" : "KS"} over threshold)`, "warning"));
    const lab = batches[x.t - L];
    if (lab) events.push(ev(`l${x.t}`, `labels for week ${lab.t + 1} arrived: accuracy ${(lab.acc * 100).toFixed(1)}%`, lab.accAlert ? "error" : "info"));
    const rt = retrains.find((r) => r.at === x.t);
    if (rt) events.push(ev(`r${x.t}`, `retrained v${rt.version} on weeks ${rt.from.map((v) => v + 1).join(", ")}: boundary → ${rt.boundary.toFixed(2)}`, "info"));
    const headline = rt ? "Labels confirm the drop, so the model is retrained on recent weeks"
      : lab?.accAlert ? `Labels for week ${lab.t + 1} arrive: accuracy has fallen`
      : x.inputAlert ? "The input histogram has moved: PSI/KS raise an alert"
      : "A new week of production traffic is scored";
    steps.push({ id: `w${x.t}`, title: `Week ${x.t + 1}: ${headline}`, description: `PSI ${x.psi.toFixed(3)} (threshold ${cfg.psiThreshold}), KS D ${x.ks.toFixed(3)} (critical ${x.ksCrit.toFixed(3)}). ${lab ? `Accuracy known up to week ${lab.t + 1}.` : `No labels yet (delay ${L} weeks).`}`, state: { t: x.t }, events });
  }
  steps.push({
    id: "summary", title: "Summary: what each monitor saw, and when",
    description: `First input alert: ${firstInputAlert !== undefined ? `week ${firstInputAlert + 1}` : "never"}. First accuracy alert known: ${firstAccAlertKnown !== undefined ? `week ${firstAccAlertKnown + 1}` : "never"}. Retrains: ${retrains.length}.`,
    state: { t: T }, events: [ev("s", `final-week accuracy ${(batches[T - 1].acc * 100).toFixed(1)}%`, batches[T - 1].acc < cfg.accThreshold ? "error" : "info")],
  });
  return { cfg, trainXs: train.map((r) => r.x[0]), refs, trainAcc, boundaries, batches, retrains, firstInputAlert, firstAccAlertKnown, steps };
}
