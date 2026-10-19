import { clamp, sigmoid } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Example = { id: number; score: number; y: 0 | 1 };
export type ClsConfig = { n: number; prevalence: number; separation: number; threshold: number; seed: number };
export const DEFAULT_CLS: ClsConfig = { n: 40, prevalence: 0.5, separation: 2.5, threshold: 0.5, seed: 4 };

/**
 * Synthetic model scores. Positives get sigmoid(+sep/2 + noise), negatives sigmoid(−sep/2 + noise), so
 * `separation` is how well the model ranks the classes. Scores are rounded to 3 decimals.
 */
export function makeExamples({ n, prevalence, separation, seed }: Omit<ClsConfig, "threshold">): Example[] {
  const N = Math.round(clamp(n, 4, 80));
  const nPos = clamp(Math.round(N * clamp(prevalence, 0, 1)), 1, N - 1);
  const r = rng(seed);
  const order = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order.map((o, id) => {
    const y = (o < nPos ? 1 : 0) as 0 | 1;
    const s = sigmoid((y ? 1 : -1) * (separation / 2) + 1.1 * r.gaussian());
    return { id, y, score: Math.round(s * 1000) / 1000 };
  });
}

/** Predicted positive when score ≥ threshold. Ratios with a zero denominator are NaN (undefined), not 0. */
export function metricsAt(ex: readonly Example[], t: number) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const e of ex) {
    const p = e.score >= t;
    if (e.y && p) tp++;
    else if (!e.y && p) fp++;
    else if (!e.y) tn++;
    else fn++;
  }
  const div = (a: number, b: number) => (b === 0 ? Number.NaN : a / b);
  const precision = div(tp, tp + fp);
  const recall = div(tp, tp + fn);
  const f1 = div(2 * tp, 2 * tp + fp + fn);
  return { tp, fp, tn, fn, n: ex.length, precision, recall, f1, accuracy: div(tp + tn, ex.length), specificity: div(tn, tn + fp), fpr: div(fp, fp + tn) };
}
export type Metrics = ReturnType<typeof metricsAt>;

/** Candidate thresholds from strict (nobody positive) to lenient (everybody positive). */
const thresholds = (ex: readonly Example[]) => [Number.POSITIVE_INFINITY, ...[...new Set(ex.map((e) => e.score))].sort((a, b) => b - a)];

export type RocPoint = { t: number; fpr: number; tpr: number };
export function rocCurve(ex: readonly Example[]): RocPoint[] {
  return thresholds(ex).map((t) => {
    const m = metricsAt(ex, t);
    return { t, fpr: m.fpr, tpr: m.recall };
  });
}

/** Trapezoidal area under the ROC points (they run from (0,0) to (1,1)). */
export function rocAuc(roc: readonly RocPoint[]) {
  let a = 0;
  for (let i = 1; i < roc.length; i++) a += (roc[i].fpr - roc[i - 1].fpr) * (roc[i].tpr + roc[i - 1].tpr) / 2;
  return a;
}

export type PrPoint = { t: number; recall: number; precision: number };
export function prCurve(ex: readonly Example[]): PrPoint[] {
  return thresholds(ex).slice(1).map((t) => {
    const m = metricsAt(ex, t);
    return { t, recall: m.recall, precision: m.precision };
  });
}

/** Average precision: Σ (Rₖ − Rₖ₋₁) · Pₖ, the step-wise area under the PR curve (sklearn's definition). */
export function averagePrecision(pr: readonly PrPoint[]) {
  let ap = 0, prev = 0;
  for (const p of pr) {
    ap += (p.recall - prev) * p.precision;
    prev = p.recall;
  }
  return ap;
}

export type ClsPhase = "scores" | "threshold" | "count" | "precision" | "recall" | "summary" | "roc" | "pr" | "baseline";
export type ClsState = { phase: ClsPhase };

const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "undefined");

export function simulate(cfg: ClsConfig) {
  const ex = makeExamples(cfg);
  const t = clamp(cfg.threshold, 0, 1.001);
  const m = metricsAt(ex, t);
  const roc = rocCurve(ex);
  const pr = prCurve(ex);
  const auc = rocAuc(roc);
  const ap = averagePrecision(pr);
  const pos = m.tp + m.fn, neg = m.fp + m.tn;
  const allNeg = metricsAt(ex, Number.POSITIVE_INFINITY);
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "metric", message, tex, severity });
  const s = (id: ClsPhase, title: string, description: string, events: SimulationEvent[]): SimulationStep<ClsState> => ({ id, title, description, state: { phase: id }, events });

  const steps: SimulationStep<ClsState>[] = [
    s("scores", "Model scores on a line", `${ex.length} examples: ${pos} truly positive, ${neg} truly negative. Each sits at its model score between 0 and 1. Nothing is decided yet.`,
      [ev("n", `${pos} positives, ${neg} negatives, prevalence ${(pos / ex.length).toFixed(2)}`, `\\text{prevalence} = \\tfrac{${pos}}{${ex.length}} = ${(pos / ex.length).toFixed(3)}`)]),
    s("threshold", "Apply the threshold", `Every example with score ≥ ${t.toFixed(2)} is predicted positive (${m.tp + m.fp}); the rest are predicted negative (${m.tn + m.fn}).`,
      [ev("t", `threshold = ${t.toFixed(2)} → ${m.tp + m.fp} predicted positive`)]),
    s("count", "Count the four outcomes", `TP = ${m.tp} (caught), FP = ${m.fp} (false alarm), FN = ${m.fn} (missed), TN = ${m.tn} (correctly ignored).`,
      [ev("c", `TP ${m.tp}, FP ${m.fp}, FN ${m.fn}, TN ${m.tn}`)]),
    s("precision", "Precision: how many alarms were real?", `Precision = TP / (TP + FP) = ${m.tp} / ${m.tp + m.fp} = ${f3(m.precision)}.`,
      [ev("p", `precision = ${f3(m.precision)}`, Number.isFinite(m.precision) ? `\\text{precision} = \\tfrac{${m.tp}}{${m.tp} + ${m.fp}} = ${f3(m.precision)}` : undefined, Number.isFinite(m.precision) ? undefined : "warning")]),
    s("recall", "Recall: how many positives were caught?", `Recall = TP / (TP + FN) = ${m.tp} / ${pos} = ${f3(m.recall)}.`,
      [ev("r", `recall = ${f3(m.recall)}`, `\\text{recall} = \\tfrac{${m.tp}}{${m.tp} + ${m.fn}} = ${f3(m.recall)}`)]),
    s("summary", "F1, accuracy and specificity", `F1 = ${f3(m.f1)}, accuracy = ${f3(m.accuracy)}, specificity = ${f3(m.specificity)}.`,
      [ev("f1", `F1 = ${f3(m.f1)}`, `F_1 = \\tfrac{2\\,TP}{2\\,TP + FP + FN} = \\tfrac{${2 * m.tp}}{${2 * m.tp + m.fp + m.fn}} = ${f3(m.f1)}`),
        ev("acc", `accuracy = ${f3(m.accuracy)}`, `\\text{accuracy} = \\tfrac{${m.tp} + ${m.tn}}{${ex.length}} = ${f3(m.accuracy)}`)]),
    s("roc", "Sweep every threshold: the ROC curve", `Lowering the threshold from 1 to 0 traces (FPR, TPR). Area under it (AUC) = ${auc.toFixed(3)}: the chance a random positive scores above a random negative.`,
      [ev("auc", `ROC AUC = ${auc.toFixed(3)} over ${roc.length} thresholds`, `\\text{AUC} = \\sum_k (FPR_k - FPR_{k-1})\\,\\tfrac{TPR_k + TPR_{k-1}}{2} = ${auc.toFixed(3)}`)]),
    s("pr", "The precision-recall curve", `Same sweep, plotted as (recall, precision). Average precision = ${ap.toFixed(3)}; a random model scores about the prevalence, ${(pos / ex.length).toFixed(3)}.`,
      [ev("ap", `AP = ${ap.toFixed(3)} vs random baseline ${(pos / ex.length).toFixed(3)}`, `AP = \\sum_k (R_k - R_{k-1})\\,P_k = ${ap.toFixed(3)}`)]),
    s("baseline", "Sanity check against a dumb baseline", `A "model" that always says negative scores accuracy ${f3(allNeg.accuracy)} with recall 0. Yours: accuracy ${f3(m.accuracy)}, recall ${f3(m.recall)}.`,
      [ev("b", `always-negative accuracy = ${f3(allNeg.accuracy)}`, undefined, allNeg.accuracy >= (m.accuracy || 0) - 0.03 ? "warning" : undefined)]),
  ];
  return { ex, t, m, roc, pr, auc, ap, allNeg, steps, prevalence: pos / ex.length };
}
export type ClsSim = ReturnType<typeof simulate>;
