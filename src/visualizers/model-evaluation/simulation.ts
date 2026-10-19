import { solveLinear } from "../../lib/math";
import { mse, r2 } from "../../lib/metrics";
import { rng } from "../../lib/random";
import type { SimulationStep } from "../../types/simulation";

export type EvalConfig = { n: number; noise: number; degree: number; testFraction: number; lambda: number; seed: number };
export type Point = { x: number; y: number };

export const MAX_DEGREE = 15;
export const fmtErr = (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(4));
export const trueFn = (x: number) => Math.sin(3 * x);
/** Added to the diagonal when λ = 0 so the normal equations stay solvable; small enough not to regularize visibly. */
export const JITTER = 1e-10;

export function generateData({ n, noise, testFraction, seed }: Pick<EvalConfig, "n" | "noise" | "testFraction" | "seed">) {
  const r = rng(seed);
  const pts: Point[] = Array.from({ length: n }, () => {
    const x = r.next() * 2 - 1;
    return { x, y: trueFn(x) + noise * r.gaussian() };
  });
  // Fisher–Yates with the same seeded stream: the split is reproducible.
  const idx = pts.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.max(1, Math.round(n * testFraction));
  return { train: idx.slice(nTest).map((i) => pts[i]), test: idx.slice(0, nTest).map((i) => pts[i]) };
}

const features = (x: number, d: number) => Array.from({ length: d + 1 }, (_, k) => x ** k);

/** Ridge regression on polynomial features: (XᵀX + λI′)w = Xᵀy, intercept not penalized. */
export function fitPolynomial(train: Point[], degree: number, lambda: number) {
  const p = degree + 1;
  const A = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b = new Array<number>(p).fill(0);
  for (const { x, y } of train) {
    const f = features(x, degree);
    for (let i = 0; i < p; i++) {
      b[i] += f[i] * y;
      for (let j = 0; j < p; j++) A[i][j] += f[i] * f[j];
    }
  }
  for (let i = 1; i < p; i++) A[i][i] += Math.max(lambda, JITTER) * train.length;
  A[0][0] += JITTER;
  const w = solveLinear(A, b);
  return { w, predict: (x: number) => features(x, degree).reduce((s, v, k) => s + v * w[k], 0) };
}

export function evaluate(train: Point[], test: Point[], degree: number, lambda: number) {
  const model = fitPolynomial(train, degree, lambda);
  const pTrain = train.map((p) => model.predict(p.x));
  const pTest = test.map((p) => model.predict(p.x));
  return {
    model,
    trainMse: mse(train.map((p) => p.y), pTrain),
    testMse: mse(test.map((p) => p.y), pTest),
    trainR2: r2(train.map((p) => p.y), pTrain),
    testR2: r2(test.map((p) => p.y), pTest),
  };
}

export function sweep(train: Point[], test: Point[], lambda: number) {
  return Array.from({ length: MAX_DEGREE }, (_, i) => {
    const e = evaluate(train, test, i + 1, lambda);
    return { degree: i + 1, trainMse: e.trainMse, testMse: e.testMse };
  });
}

export type Verdict = "underfitting" | "good fit" | "overfitting";

/**
 * Heuristic, relative to this dataset: compare the chosen degree's test error with the best test error
 * found by the sweep. More than 25% worse on the simpler side = underfitting, on the complex side = overfitting.
 */
export function diagnose(degree: number, testMse: number, sw: ReturnType<typeof sweep>): { verdict: Verdict; bestDegree: number; bestTestMse: number } {
  const best = sw.reduce((a, b) => (b.testMse < a.testMse ? b : a));
  const verdict: Verdict = testMse <= best.testMse * 1.25 ? "good fit" : degree < best.degree ? "underfitting" : "overfitting";
  return { verdict, bestDegree: best.degree, bestTestMse: best.testMse };
}

export type EvalPhase = "data" | "split" | "fit" | "train-eval" | "test-eval" | "diagnose";
export type EvalState = { phase: EvalPhase };

export function simulate(cfg: EvalConfig) {
  const { train, test } = generateData(cfg);
  const e = evaluate(train, test, cfg.degree, cfg.lambda);
  const sw = sweep(train, test, cfg.lambda);
  const d = diagnose(cfg.degree, e.testMse, sw);
  const gap = e.testMse - e.trainMse;
  const lo = Math.min(...train.map((p) => p.x)), hi = Math.max(...train.map((p) => p.x));
  const outside = test.filter((p) => p.x < lo || p.x > hi).length;
  const steps: SimulationStep<EvalState>[] = [
    { id: "data", title: "Generate noisy data", description: `${cfg.n} points sampled from y = sin(3x) plus Gaussian noise with σ = ${cfg.noise}. The dashed curve is the true function the model never sees directly.`, state: { phase: "data" }, events: [{ id: "d", type: "data", message: `noise variance σ² = ${(cfg.noise ** 2).toFixed(4)}: the best possible expected MSE on new data` }] },
    { id: "split", title: "Split into train and test", description: `${train.length} points train the model; ${test.length} points (${Math.round(cfg.testFraction * 100)}%) are held out and never used for fitting.`, state: { phase: "split" }, events: [{ id: "s", type: "split", message: `train = ${train.length}, test = ${test.length}` }] },
    { id: "fit", title: `Fit degree-${cfg.degree} polynomial`, description: `Least squares finds ${cfg.degree + 1} coefficients that minimize training error${cfg.lambda > 0 ? ` plus a ridge penalty λ = ${cfg.lambda} on their size` : ""}.`, state: { phase: "fit" }, events: [{ id: "f", type: "fit", message: `w = [${e.model.w.slice(0, 4).map((v) => v.toFixed(2)).join(", ")}${e.model.w.length > 4 ? ", …" : ""}]`, severity: Math.max(...e.model.w.map(Math.abs)) > 50 ? "warning" : "info" }] },
    { id: "train-eval", title: "Measure training error", description: `Training MSE = ${e.trainMse.toFixed(4)}. Vertical lines are residuals: the gap between each training label and the curve.`, state: { phase: "train-eval" }, events: [{ id: "t", type: "metric", message: `train MSE ${e.trainMse.toFixed(4)}, R² ${e.trainR2.toFixed(3)}` }] },
    { id: "test-eval", title: "Measure test error", description: `Test MSE = ${fmtErr(e.testMse)} on points the model never saw. Gap (test − train) = ${fmtErr(gap)}.`, state: { phase: "test-eval" }, events: [
      { id: "v", type: "metric", message: `test MSE ${fmtErr(e.testMse)}, R² ${fmtErr(e.testR2)}`, severity: gap > e.trainMse * 2 && gap > 0.02 ? "warning" : "info" },
      ...(outside ? [{ id: "ext", type: "extrapolation", message: `${outside} test point(s) lie outside the training x-range [${lo.toFixed(2)}, ${hi.toFixed(2)}]. The model must extrapolate there, where high-degree polynomials diverge fastest.`, severity: "warning" as const }] : []),
    ] },
    { id: "diagnose", title: `Diagnosis: ${d.verdict}`, description: d.verdict === "good fit" ? `Test error is within 25% of the best degree in the sweep (degree ${d.bestDegree}).` : d.verdict === "underfitting" ? `The model is too simple: degree ${d.bestDegree} reaches test MSE ${d.bestTestMse.toFixed(4)}. Both errors are high.` : `The model memorizes noise: training error is low but test error is ${fmtErr(e.testMse / d.bestTestMse)}× the best (degree ${d.bestDegree}).`, state: { phase: "diagnose" }, events: [{ id: "x", type: "diagnosis", message: `verdict: ${d.verdict}`, severity: d.verdict === "good fit" ? "info" : "warning" }] },
  ];
  return { steps, train, test, ...e, sweep: sw, ...d };
}
