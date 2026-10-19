import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Pt = { x: number; y: number; outlier?: boolean };
export type LRConfig = { n: number; noise: number; seed: number; outlier: boolean };
export const DEFAULT_LR: LRConfig = { n: 8, noise: 1, seed: 3, outlier: false };
/** The hidden "true" line the synthetic data is drawn around. */
export const TRUE_LINE = { slope: 0.7, intercept: 2 };
export const OUTLIER: Pt = { x: 8.5, y: 0.5, outlier: true };

export type Phase = "data" | "mean-x" | "mean-y" | "products" | "slope" | "intercept" | "residuals" | "squares" | "r2";
export type LRState = { phase: Phase; headline: string; sub: string };

const r1 = (v: number) => Math.round(v * 10) / 10;

/** Seeded points around TRUE_LINE, rounded to one decimal so the arithmetic is easy to follow. */
export function makeData(cfg: LRConfig): Pt[] {
  const n = clamp(Math.round(cfg.n), 2, 20);
  const r = rng(cfg.seed);
  const pts: Pt[] = Array.from({ length: n }, (_, i) => {
    const x = r1(clamp(1 + (8 * i) / Math.max(1, n - 1) + (r.next() - 0.5) * 0.6, 0.5, 9.5));
    return { x, y: r1(TRUE_LINE.intercept + TRUE_LINE.slope * x + cfg.noise * r.gaussian()) };
  });
  return cfg.outlier ? [...pts, OUTLIER] : pts;
}

export type Fit = { n: number; xm: number; ym: number; sxx: number; sxy: number; slope: number; intercept: number; sse: number; sst: number; r2: number };

export const predict = (slope: number, intercept: number, x: number) => intercept + slope * x;
export const residuals = (pts: Pt[], slope: number, intercept: number) => pts.map((p) => p.y - predict(slope, intercept, p.x));
export const sse = (pts: Pt[], slope: number, intercept: number) => residuals(pts, slope, intercept).reduce((s, r) => s + r * r, 0);

/** Ordinary least squares. With no spread in x the slope is undefined; we fall back to a flat line through ȳ. */
export function ols(pts: Pt[]): Fit {
  const n = pts.length;
  if (!n) return { n, xm: 0, ym: 0, sxx: 0, sxy: 0, slope: 0, intercept: 0, sse: 0, sst: 0, r2: Number.NaN };
  const xm = pts.reduce((s, p) => s + p.x, 0) / n;
  const ym = pts.reduce((s, p) => s + p.y, 0) / n;
  const sxx = pts.reduce((s, p) => s + (p.x - xm) ** 2, 0);
  const sxy = pts.reduce((s, p) => s + (p.x - xm) * (p.y - ym), 0);
  const slope = sxx === 0 ? 0 : sxy / sxx;
  const intercept = ym - slope * xm;
  const e = sse(pts, slope, intercept);
  const sst = pts.reduce((s, p) => s + (p.y - ym) ** 2, 0);
  return { n, xm, ym, sxx, sxy, slope, intercept, sse: e, sst, r2: sst === 0 ? Number.NaN : 1 - e / sst };
}

const f = (v: number, d = 2) => v.toFixed(d);
const p = (v: number, d = 2) => (v < 0 ? `(${f(v, d)})` : f(v, d));

export function simulate(cfg: LRConfig) {
  const pts = makeData(cfg);
  const fit = ols(pts);
  const res = residuals(pts, fit.slope, fit.intercept);
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });
  const outlierRes = cfg.outlier ? res[res.length - 1] : 0;
  const steps: SimulationStep<LRState>[] = [
    {
      id: "data", title: "Plot the data",
      description: `${fit.n} points (x, y). We want the straight line ŷ = a + b·x that sits as close to them as possible.`,
      state: { phase: "data", headline: `${fit.n} points land on the plot`, sub: "Goal: one straight line that predicts y from x with the smallest total squared miss." },
      events: [ev("n", `n = ${fit.n} points`, `n = ${fit.n}`), ...(cfg.outlier ? [ev("out", `Outlier added at (${OUTLIER.x}, ${OUTLIER.y})`, undefined, "warning")] : [])],
    },
    {
      id: "mean-x", title: "Mean of x",
      description: `Average the x values: x̄ = ${f(fit.xm, 3)}. The best line always passes through the point of means.`,
      state: { phase: "mean-x", headline: `The average x is x̄ = ${f(fit.xm)}`, sub: "A vertical guide slides to the centre of the x values." },
      events: [ev("xm", `x̄ = Σx / n = ${f(fit.xm, 3)}`, `\\bar{x} = \\frac{1}{n}\\sum x_i = ${f(fit.xm, 3)}`)],
    },
    {
      id: "mean-y", title: "Mean of y",
      description: `Average the y values: ȳ = ${f(fit.ym, 3)}. The crosshair (x̄, ȳ) is the pivot the line will rotate around.`,
      state: { phase: "mean-y", headline: `The average y is ȳ = ${f(fit.ym)}: the line must pass through (x̄, ȳ)`, sub: "Least squares always pivots around this centre point." },
      events: [ev("ym", `ȳ = Σy / n = ${f(fit.ym, 3)}`, `\\bar{y} = \\frac{1}{n}\\sum y_i = ${f(fit.ym, 3)}`)],
    },
    {
      id: "products", title: "Covariance: how x and y move together",
      description: `For each point multiply its x-distance by its y-distance from the centre. Green rectangles (same sign) say "x up, y up"; red ones disagree. Sum: Sxy = ${f(fit.sxy, 3)}. Spread of x alone: Sxx = ${f(fit.sxx, 3)}.`,
      state: { phase: "products", headline: "Each point draws a rectangle to the centre: green agrees, red disagrees", sub: `Area sum Sxy = ${f(fit.sxy)} (co-movement), and Sxx = ${f(fit.sxx)} (spread of x).` },
      events: [
        ev("sxy", `Sxy = Σ(x−x̄)(y−ȳ) = ${f(fit.sxy, 3)}`, `S_{xy} = \\sum (x_i-\\bar{x})(y_i-\\bar{y}) = ${f(fit.sxy, 3)}`),
        ev("sxx", `Sxx = Σ(x−x̄)² = ${f(fit.sxx, 3)}`, `S_{xx} = \\sum (x_i-\\bar{x})^2 = ${f(fit.sxx, 3)}`),
      ],
    },
    {
      id: "slope", title: "Slope = Sxy / Sxx",
      description: `b = ${f(fit.sxy, 3)} / ${f(fit.sxx, 3)} = ${f(fit.slope, 4)}. The line rotates around (x̄, ȳ) until its steepness matches.`,
      state: { phase: "slope", headline: `Slope b = Sxy ÷ Sxx = ${f(fit.slope, 3)}`, sub: "The line swings around the centre point from flat to the least-squares tilt." },
      events: [ev("b", `b = ${f(fit.sxy, 3)} / ${f(fit.sxx, 3)} = ${f(fit.slope, 4)}`, `b = \\frac{S_{xy}}{S_{xx}} = \\frac{${f(fit.sxy, 3)}}{${f(fit.sxx, 3)}} = ${f(fit.slope, 4)}`,
        fit.sxx === 0 ? "warning" : undefined)],
    },
    {
      id: "intercept", title: "Intercept = ȳ − b·x̄",
      description: `a = ${f(fit.ym, 3)} − ${p(fit.slope, 4)}·${f(fit.xm, 3)} = ${f(fit.intercept, 4)}. Extend the line back to x = 0 to read it off.`,
      state: { phase: "intercept", headline: `Intercept a = ȳ − b·x̄ = ${f(fit.intercept, 3)}`, sub: "Slide the line back to x = 0: that height is the intercept." },
      events: [ev("a", `a = ȳ − b·x̄ = ${f(fit.intercept, 4)}`, `a = \\bar{y} - b\\,\\bar{x} = ${f(fit.ym, 3)} - ${p(fit.slope, 4)} \\cdot ${f(fit.xm, 3)} = ${f(fit.intercept, 4)}`)],
    },
    {
      id: "residuals", title: "Residuals",
      description: "Each residual is the vertical miss: eᵢ = yᵢ − ŷᵢ. Points above the line have positive residuals, points below negative. Their sum is 0 for an OLS line.",
      state: { phase: "residuals", headline: "Each point's miss (residual) grows as a vertical line", sub: "Residual = actual y − predicted y. They cancel out: their sum is 0." },
      events: [ev("sum", `Σ residuals = ${f(res.reduce((s, r) => s + r, 0), 6)}`, `\\sum e_i = ${f(Math.abs(res.reduce((s, r) => s + r, 0)) < 5e-7 ? 0 : res.reduce((s, r) => s + r, 0), 6)}`),
        ...(cfg.outlier ? [ev("ores", `Outlier residual = ${f(outlierRes)}`, `e_{\\text{outlier}} = ${f(outlierRes)}`, "warning")] : [])],
    },
    {
      id: "squares", title: "Squared residuals → SSE",
      description: `Square every residual: each becomes an actual square whose area is eᵢ². The total area SSE = ${f(fit.sse, 3)} is exactly what OLS made as small as possible.`,
      state: { phase: "squares", headline: `Squares of the misses: total area SSE = ${f(fit.sse)}`, sub: "Big misses make big squares. No other line has a smaller total area." },
      events: [ev("sse", `SSE = Σe² = ${f(fit.sse, 3)}`, `\\text{SSE} = \\sum e_i^2 = ${f(fit.sse, 3)}`),
        ...(cfg.outlier ? [ev("oshare", `The outlier alone contributes ${f((100 * outlierRes ** 2) / fit.sse, 0)}% of SSE`, undefined, "warning")] : [])],
    },
    {
      id: "r2", title: "R²: how much better than the mean?",
      description: `Compare with the flat line y = ȳ: its squared misses total SST = ${f(fit.sst, 3)}. R² = 1 − SSE/SST = ${f(fit.r2, 3)}.`,
      state: { phase: "r2", headline: `R² = ${f(fit.r2, 3)}: the line explains ${Number.isFinite(fit.r2) ? f(100 * fit.r2, 0) : "—"}% of the variation in y`, sub: "Baseline: the flat mean line. R² measures how much of its squared error the fitted line removes." },
      events: [ev("r2", `R² = 1 − ${f(fit.sse, 3)}/${f(fit.sst, 3)} = ${f(fit.r2, 4)}`, `R^2 = 1 - \\frac{\\text{SSE}}{\\text{SST}} = 1 - \\frac{${f(fit.sse, 3)}}{${f(fit.sst, 3)}} = ${f(fit.r2, 4)}`)],
    },
  ];
  return { pts, fit, res, steps };
}
