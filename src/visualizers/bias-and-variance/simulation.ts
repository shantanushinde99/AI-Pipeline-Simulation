import { clamp, mean } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { fitPolynomial, trueFn, type Point } from "../model-evaluation/simulation";

export type BvConfig = { degree: number; noise: number; n: number; sets: number; seed: number };
export const MAX_DEGREE = 10, GRID = Array.from({ length: 81 }, (_, i) => -1 + i / 40);

/**
 * M independent training sets of n points, y = sin(3x) + N(0, σ²). x is stratified: one uniform draw per
 * equal-width bin of [−1, 1], so every set covers the whole range (no accidental extrapolation gaps).
 */
export function drawSets({ n, noise, sets, seed }: Pick<BvConfig, "n" | "noise" | "sets" | "seed">): Point[][] {
  const r = rng(seed);
  return Array.from({ length: sets }, () => Array.from({ length: n }, (_, i) => {
    const x = -1 + (2 * (i + r.next())) / n;
    return { x, y: trueFn(x) + noise * r.gaussian() };
  }));
}

/**
 * Bias–variance decomposition on the evaluation grid:
 *   bias²(x) = (mean_m f̂_m(x) − f(x))²,  variance(x) = mean_m (f̂_m(x) − f̄(x))², averaged over x.
 * Expected squared error on a new noisy label = bias² + variance + σ².
 */
export function decompose(sets: Point[][], degree: number, noise: number, grid = GRID) {
  const curves = sets.map((s) => { const m = fitPolynomial(s, degree, 0); return grid.map(m.predict); });
  const avg = grid.map((_, g) => mean(curves.map((c) => c[g])));
  const bias2x = grid.map((x, g) => (avg[g] - trueFn(x)) ** 2);
  const varx = grid.map((_, g) => mean(curves.map((c) => (c[g] - avg[g]) ** 2)));
  const bias2 = mean(bias2x), variance = mean(varx), noise2 = noise ** 2;
  const trainMse = mean(sets.map((s) => { const f = fitPolynomial(s, degree, 0); return mean(s.map((p) => (p.y - f.predict(p.x)) ** 2)); }));
  return { curves, avg, bias2x, varx, bias2, variance, noise2, total: bias2 + variance + noise2, trainMse };
}

/** Monte-Carlo check: average squared error of every fitted model against fresh noisy labels on the grid. */
export function measuredTestMse(curves: number[][], noise: number, seed: number, grid = GRID) {
  const r = rng(seed + 104729);
  let s = 0, c = 0;
  for (const curve of curves) for (let rep = 0; rep < 5; rep++) grid.forEach((x, g) => { s += (trueFn(x) + noise * r.gaussian() - curve[g]) ** 2; c++; });
  return s / c;
}

export type BvPhase = "truth" | "one" | "many" | "mean" | "variance" | "decompose" | "sweep";
export type BvState = { phase: BvPhase; headline: string; sub: string };

export function simulate(raw: BvConfig) {
  const cfg: BvConfig = {
    degree: clamp(Math.round(raw.degree), 1, MAX_DEGREE),
    noise: clamp(raw.noise, 0, 1),
    n: clamp(Math.round(raw.n), MAX_DEGREE + 2, 60),
    sets: clamp(Math.round(raw.sets), 2, 40),
    seed: raw.seed,
  };
  const sets = drawSets(cfg);
  const d = decompose(sets, cfg.degree, cfg.noise);
  const measured = measuredTestMse(d.curves, cfg.noise, cfg.seed);
  const fit1 = fitPolynomial(sets[0], cfg.degree, 0);
  const firstTrainMse = mean(sets[0].map((p) => (p.y - fit1.predict(p.x)) ** 2));
  const sweep = Array.from({ length: MAX_DEGREE }, (_, i) => {
    const r = decompose(sets, i + 1, cfg.noise);
    return { degree: i + 1, bias2: r.bias2, variance: r.variance, noise2: r.noise2, total: r.total, trainMse: r.trainMse };
  });
  const best = sweep.reduce((a, b) => (b.total < a.total ? b : a));
  const dominant = d.bias2 > d.variance ? "bias" : "variance";
  const f4 = (v: number) => (Math.abs(v) >= 100 ? v.toExponential(2) : v.toFixed(4));
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "compute", message, severity, tex });

  const steps: SimulationStep<BvState>[] = [
    { id: "truth", title: "The true function", description: "Data come from y = sin(3x) plus Gaussian noise. In real life you never see this curve; here it is drawn dashed so you can judge every model against it.",
      state: { phase: "truth", headline: "Start from the truth: y = sin(3x) + noise", sub: "Normally hidden. Shown here so we can measure how far models land from it." },
      events: [ev("t", `noise σ = ${cfg.noise}, so the irreducible error σ² = ${(cfg.noise ** 2).toFixed(4)}`)] },
    { id: "one", title: "Fit one training set", description: `Draw ${cfg.n} noisy points and fit a degree-${cfg.degree} polynomial by least squares. One curve tells you little: was it a good model or a lucky sample?`,
      state: { phase: "one", headline: `One training set of ${cfg.n} points → one degree-${cfg.degree} curve`, sub: "Least squares picks the polynomial closest to these particular noisy points." },
      events: [ev("o", `training MSE of set #1 = ${firstTrainMse.toFixed(4)}`)] },
    { id: "many", title: `Repeat with ${cfg.sets} training sets`, description: `Same recipe, ${cfg.sets} fresh samples, ${cfg.sets} curves. How much they disagree with each other is the model's variance.`,
      state: { phase: "many", headline: `Redraw the data ${cfg.sets} times: watch the curves ${cfg.degree >= 7 ? "fan out" : "pile up"}`, sub: "Each curve is the same model class trained on a different random sample." },
      events: [ev("m", `${cfg.sets} sets × ${cfg.n} points fitted`)] },
    { id: "mean", title: "Average model vs. truth = bias", description: `Averaging the ${cfg.sets} curves gives the model's 'expected' prediction. Its squared distance from the truth, averaged over x, is bias² = ${f4(d.bias2)}.`,
      state: { phase: "mean", headline: `The average curve misses the truth by bias² = ${f4(d.bias2)}`, sub: cfg.degree <= 2 ? "Too simple: even the average of many fits cannot bend like sin(3x)." : "The shaded gap is systematic error that more data of the same kind won't remove." },
      events: [ev("b", `bias² = mean_x (f̄(x) − f(x))² = ${f4(d.bias2)}`, undefined, `\\text{bias}^2 = \\tfrac{1}{|G|}\\textstyle\\sum_x (\\bar{f}(x) - f(x))^2 = ${f4(d.bias2)}`)] },
    { id: "variance", title: "Spread around the average = variance", description: `The band shows ±1 standard deviation of the ${cfg.sets} curves around their mean. Averaged over x, variance = ${f4(d.variance)}.`,
      state: { phase: "variance", headline: `The curves scatter around their mean: variance = ${f4(d.variance)}`, sub: "Dots on the three vertical lines are each model's prediction at that x." },
      events: [ev("v", `variance = mean_x mean_m (f̂_m(x) − f̄(x))² = ${f4(d.variance)}`, d.variance > 1 ? "warning" : undefined, `\\text{var} = \\tfrac{1}{|G|}\\textstyle\\sum_x \\tfrac{1}{M}\\sum_m (\\hat{f}_m(x) - \\bar{f}(x))^2 = ${f4(d.variance)}`)] },
    { id: "decompose", title: "Expected error = bias² + variance + noise", description: `bias² ${f4(d.bias2)} + variance ${f4(d.variance)} + σ² ${f4(d.noise2)} = ${f4(d.total)}. Measured directly against fresh noisy labels: ${f4(measured)}. ${dominant === "bias" ? "Bias dominates: the model is too simple." : "Variance dominates: the model is too flexible for this much data."}`,
      state: { phase: "decompose", headline: `Expected error ${f4(d.total)}: ${dominant === "bias" ? "mostly bias (underfitting)" : "mostly variance (overfitting)"}`, sub: "Bias² and variance are computed from the curves; σ² is the noise no model can remove." },
      events: [
        ev("sum", `bias² + variance + σ² = ${f4(d.total)}`, undefined, `\\mathbb{E}[(y - \\hat{f})^2] = ${f4(d.bias2)} + ${f4(d.variance)} + ${f4(d.noise2)} = ${f4(d.total)}`),
        ev("mc", `Monte-Carlo check on fresh labels: ${f4(measured)}`),
      ] },
    { id: "sweep", title: "Bias and variance vs. complexity", description: `Repeating the experiment for every degree 1–${MAX_DEGREE}: bias² falls, variance rises, and their sum is lowest at degree ${best.degree} (expected error ${f4(best.total)}).`,
      state: { phase: "sweep", headline: `The sweet spot here is degree ${best.degree}`, sub: "Bias² falls and variance rises as complexity grows; the total is U-shaped." },
      events: [ev("best", `lowest expected error at degree ${best.degree}: ${f4(best.total)}`, best.degree === cfg.degree ? undefined : "warning")] },
  ];
  return { cfg, sets, ...d, measured, firstTrainMse, sweep, best, steps };
}

export type BvSim = ReturnType<typeof simulate>;
