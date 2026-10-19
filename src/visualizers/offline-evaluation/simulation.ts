import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type SliceId = "short" | "long" | "code" | "nonen";
export const SLICES: { id: SliceId; name: string; n: number }[] = [
  { id: "short", name: "Short questions", n: 30 },
  { id: "long", name: "Long questions", n: 25 },
  { id: "code", name: "Code questions", n: 15 },
  { id: "nonen", name: "Non-English", n: 10 },
];

export type Scenario = "trap" | "uniform" | "noise";
/** Correct answers per slice (same order as SLICES). B's Non-English count comes from the slider. */
export const SCENARIOS: Record<Scenario, { label: string; a: number[]; b: number[]; bNonEn: number }> = {
  trap: { label: "Simpson-style trap (B breaks Non-English)", a: [21, 16, 9, 8], b: [28, 22, 13, 3], bNonEn: 3 },
  uniform: { label: "Honest improvement (B better everywhere)", a: [21, 16, 9, 8], b: [24, 19, 11, 9], bNonEn: 9 },
  noise: { label: "No real difference (noise only)", a: [21, 16, 9, 8], b: [22, 15, 10, 7], bNonEn: 7 },
};

export type OfflineParams = { scenario: Scenario; bNonEn: number; resamples: number; prodShareNonEn: number; seed: number };
export const DEFAULT_OFFLINE: OfflineParams = { scenario: "trap", bNonEn: 3, resamples: 1000, prodShareNonEn: 0.35, seed: 7 };

export type Example = { id: number; slice: SliceId; a: boolean; b: boolean };
export type SliceResult = { id: SliceId; name: string; n: number; accA: number; accB: number; diff: number; ci: [number, number]; regress: boolean };
export type Phase = "dataset" | "runA" | "runB" | "aggregate" | "resample" | "bootstrap" | "slices" | "reweight";
export type OfflineState = { phase: Phase };

/** Linear-interpolated percentile of an ascending-sorted array (q in [0, 1]). */
export function percentile(sorted: readonly number[], q: number) {
  if (!sorted.length) return Number.NaN;
  const pos = clamp(q, 0, 1) * (sorted.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Labelled dataset: inside each slice, a seeded "difficulty" decides which examples each model gets right, so A and B are correlated (paired). */
export function buildDataset(p: OfflineParams): Example[] {
  const sc = SCENARIOS[p.scenario];
  const r = rng(p.seed);
  const out: Example[] = [];
  SLICES.forEach((s, si) => {
    const diff = Array.from({ length: s.n }, () => r.next());
    const shifted = diff.map((d) => d + 0.35 * (r.next() - 0.5));
    const rank = (xs: number[]) => xs.map((v, i) => ({ v, i })).sort((x, y) => x.v - y.v).map((o) => o.i);
    const aCount = sc.a[si];
    const bCount = s.id === "nonen" ? clamp(Math.round(p.bNonEn), 0, s.n) : sc.b[si];
    const aSet = new Set(rank(diff).slice(0, aCount));
    const bSet = new Set(rank(shifted).slice(0, bCount));
    for (let i = 0; i < s.n; i++) out.push({ id: out.length, slice: s.id, a: aSet.has(i), b: bSet.has(i) });
  });
  return out;
}

const acc = (xs: Example[], m: "a" | "b") => (xs.length ? xs.filter((e) => e[m]).length / xs.length : Number.NaN);

/** Paired bootstrap of acc(B) − acc(A): resample examples with replacement, keep each example's A and B outcome together. */
export function bootstrapDiff(xs: Example[], resamples: number, seed: number) {
  const r = rng(seed);
  const n = xs.length;
  const diffs: number[] = [];
  let firstPicks: number[] = [];
  for (let k = 0; k < resamples; k++) {
    const picks = new Array<number>(n).fill(0);
    let d = 0;
    for (let j = 0; j < n; j++) {
      const i = Math.floor(r.next() * n);
      picks[i]++;
      d += Number(xs[i].b) - Number(xs[i].a);
    }
    if (k === 0) firstPicks = picks;
    diffs.push(n ? d / n : 0);
  }
  const sorted = [...diffs].sort((a, b) => a - b);
  return { diffs, sorted, ci: [percentile(sorted, 0.025), percentile(sorted, 0.975)] as [number, number], firstPicks };
}

export type Hist = { x0: number; x1: number; counts: number[][] };
/** Histogram of bootstrap diffs plus snapshots after growing fractions of the resamples (drives the "forming" animation). */
export function histogram(diffs: number[], binWidth: number, fractions: number[]): Hist {
  const lo = Math.floor(Math.min(...diffs) / binWidth) * binWidth;
  const hi = Math.ceil(Math.max(...diffs) / binWidth + 1e-9) * binWidth + binWidth;
  const bins = Math.max(1, Math.round((hi - lo) / binWidth));
  const counts = fractions.map((f) => {
    const c = new Array<number>(bins).fill(0);
    diffs.slice(0, Math.round(f * diffs.length)).forEach((d) => c[clamp(Math.floor((d - lo) / binWidth + 1e-9), 0, bins - 1)]++);
    return c;
  });
  return { x0: lo, x1: lo + bins * binWidth, counts };
}

export const HIST_FRACTIONS = [0, 0.05, 0.15, 0.35, 0.65, 1];

export function simulate(input: OfflineParams) {
  const p: OfflineParams = {
    ...input,
    bNonEn: clamp(Math.round(input.bNonEn), 0, 10),
    resamples: clamp(Math.round(input.resamples), 50, 4000),
    prodShareNonEn: clamp(input.prodShareNonEn, 0, 0.95),
  };
  const data = buildDataset(p);
  const accA = acc(data, "a"), accB = acc(data, "b");
  const boot = bootstrapDiff(data, p.resamples, p.seed + 101);
  const hist = histogram(boot.diffs, 2 / data.length, HIST_FRACTIONS);
  const slices: SliceResult[] = SLICES.map((s, i) => {
    const xs = data.filter((e) => e.slice === s.id);
    const b = bootstrapDiff(xs, p.resamples, p.seed + 200 + i);
    const a1 = acc(xs, "a"), b1 = acc(xs, "b");
    return { id: s.id, name: s.name, n: xs.length, accA: a1, accB: b1, diff: b1 - a1, ci: b.ci, regress: b.ci[1] < 0 };
  });
  // Re-weight slice accuracies to the production traffic mix: Non-English gets prodShareNonEn, the rest keep their relative eval shares.
  const restN = data.length - SLICES[3].n;
  const prodWeights = SLICES.map((s) => (s.id === "nonen" ? p.prodShareNonEn : (1 - p.prodShareNonEn) * (s.n / restN)));
  const evalWeights = SLICES.map((s) => s.n / data.length);
  const prodA = slices.reduce((t, s, i) => t + prodWeights[i] * s.accA, 0);
  const prodB = slices.reduce((t, s, i) => t + prodWeights[i] * s.accB, 0);
  const worst = slices.reduce((w, s) => (s.diff < w.diff ? s : w));
  const nA = data.filter((e) => e.a).length, nB = data.filter((e) => e.b).length;
  const picks = boot.firstPicks;
  const neverPicked = picks.filter((c) => c === 0).length;
  const firstDiff = boot.diffs[0];

  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "eval", message, tex, severity });
  const f = (v: number, d = 4) => v.toFixed(d);
  const steps: SimulationStep<OfflineState>[] = [
    { id: "dataset", title: "The labelled dataset", state: { phase: "dataset" },
      description: `${data.length} labelled examples, frozen before either model is run, split into ${SLICES.length} slices (segments) so we can see where a model wins or loses.`,
      events: [ev("ds", `Loaded ${data.length} examples: ${SLICES.map((s) => `${s.name} ${s.n}`).join(", ")}`)] },
    { id: "runA", title: "Run model A (baseline)", state: { phase: "runA" },
      description: "Every example is sent to model A and its answer is graded against the label. Each cell is one example: correct or wrong.",
      events: [ev("a", `Model A: ${nA}/${data.length} correct`, `\\text{acc}_A = \\tfrac{${nA}}{${data.length}} = ${f(accA)}`)] },
    { id: "runB", title: "Run model B (candidate)", state: { phase: "runB" },
      description: "The same examples, in the same order, go to model B. Using identical inputs is what makes the comparison paired.",
      events: [ev("b", `Model B: ${nB}/${data.length} correct`, `\\text{acc}_B = \\tfrac{${nB}}{${data.length}} = ${f(accB)}`)] },
    { id: "aggregate", title: "Compare aggregate accuracy", state: { phase: "aggregate" },
      description: `Overall, B scores ${(accB * 100).toFixed(1)}% vs A's ${(accA * 100).toFixed(1)}%: a difference of ${((accB - accA) * 100).toFixed(1)} percentage points. A single number, but how sure are we?`,
      events: [ev("d", `Δ = acc_B − acc_A = ${f(accB - accA)}`, `\\Delta = ${f(accB)} - ${f(accA)} = ${f(accB - accA)}`, accB < accA ? "warning" : undefined)] },
    { id: "resample", title: "Bootstrap: one resample", state: { phase: "resample" },
      description: `Draw ${data.length} examples with replacement from the dataset. Some appear twice or more; ${neverPicked} never appear. Recompute Δ on this resample.`,
      events: [ev("r1", `Resample #1: ${neverPicked} examples never picked; Δ* = ${f(firstDiff)}`, `\\Delta^{*}_1 = ${f(firstDiff)}`)] },
    { id: "bootstrap", title: `Repeat ${p.resamples} times`, state: { phase: "bootstrap" },
      description: `The spread of Δ* across ${p.resamples} resamples estimates how much Δ would wobble on a different sample of data. The middle 95% is the confidence interval.`,
      events: [ev("ci", `95% CI for Δ: [${f(boot.ci[0])}, ${f(boot.ci[1])}]`, `\\text{CI}_{95} = [q_{0.025}, q_{0.975}] = [${f(boot.ci[0])}, ${f(boot.ci[1])}]`, boot.ci[0] <= 0 ? "warning" : undefined)] },
    { id: "slices", title: "Break it down by slice", state: { phase: "slices" },
      description: worst.regress
        ? `${worst.name} drops from ${(worst.accA * 100).toFixed(0)}% to ${(worst.accB * 100).toFixed(0)}% and its whole CI is below 0, while the aggregate still says B wins.`
        : "No slice has a CI entirely below 0: the improvement holds up per segment (or the evidence is too weak to say).",
      events: slices.map((s) => ev(`s-${s.id}`, `${s.name}: Δ = ${f(s.diff, 3)}, CI [${f(s.ci[0], 3)}, ${f(s.ci[1], 3)}]`, undefined, s.regress ? "error" : undefined)) },
    { id: "reweight", title: "Re-weight to production traffic", state: { phase: "reweight" },
      description: `The eval set is ${(evalWeights[3] * 100).toFixed(1)}% Non-English but production is ${(p.prodShareNonEn * 100).toFixed(0)}%. Re-weighting the slice accuracies to the production mix gives A ${(prodA * 100).toFixed(1)}% vs B ${(prodB * 100).toFixed(1)}%.`,
      events: [ev("pw", `Production-weighted: A ${f(prodA)}, B ${f(prodB)}`, `\\textstyle\\sum_s w_s\\,\\text{acc}_s:\\; A = ${f(prodA)},\\; B = ${f(prodB)}`, prodB < prodA ? "error" : undefined)] },
  ];
  return { params: p, data, accA, accB, nA, nB, boot, hist, slices, worst, evalWeights, prodWeights, prodA, prodB, picks, neverPicked, firstDiff, steps };
}
export type OfflineSim = ReturnType<typeof simulate>;
