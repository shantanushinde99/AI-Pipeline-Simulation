import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type LossId = "mse" | "mae" | "huber";
export const LOSS_IDS: LossId[] = ["mse", "mae", "huber"];
export const LOSS_NAME: Record<LossId, string> = { mse: "MSE", mae: "MAE", huber: "Huber" };

/** Per-point loss of residual r and its derivative dℓ/dr. */
export function lossOf(id: LossId, r: number, delta: number) {
  const a = Math.abs(r);
  if (id === "mse") return r * r;
  if (id === "mae") return a;
  return a <= delta ? 0.5 * r * r : delta * (a - 0.5 * delta);
}
export function gradOf(id: LossId, r: number, delta: number) {
  if (id === "mse") return 2 * r;
  if (id === "mae") return Math.sign(r);
  return clamp(r, -delta, delta);
}

export type Pt = { x: number; y: number; outlier?: boolean };
export type Line = { slope: number; intercept: number };

export function makeData(seed: number, outlier: boolean, outlierDrop: number): Pt[] {
  const r = rng(seed);
  const pts: Pt[] = Array.from({ length: 8 }, (_, i) => {
    const x = Math.round((1 + i * 1.1 + r.next() * 0.4) * 10) / 10;
    return { x, y: Math.round((1 + 0.5 * x + 0.35 * r.gaussian()) * 10) / 10 };
  });
  return outlier ? [...pts, { x: 9, y: Math.round((1 + 0.5 * 9 - outlierDrop) * 10) / 10, outlier: true }] : pts;
}

/** Weighted least squares for a line (weights ≥ 0). */
export function wls(pts: Pt[], w: number[]): Line {
  const sw = w.reduce((s, v) => s + v, 0) || 1;
  const xm = pts.reduce((s, p, i) => s + w[i] * p.x, 0) / sw;
  const ym = pts.reduce((s, p, i) => s + w[i] * p.y, 0) / sw;
  const sxx = pts.reduce((s, p, i) => s + w[i] * (p.x - xm) ** 2, 0);
  const sxy = pts.reduce((s, p, i) => s + w[i] * (p.x - xm) * (p.y - ym), 0);
  const slope = sxx < 1e-12 ? 0 : sxy / sxx;
  return { slope, intercept: ym - slope * xm };
}

export const residuals = (pts: Pt[], l: Line) => pts.map((p) => p.y - (l.intercept + l.slope * p.x));
export const totalLoss = (id: LossId, pts: Pt[], l: Line, delta: number) => residuals(pts, l).reduce((s, r) => s + lossOf(id, r, delta), 0) / Math.max(1, pts.length);

/**
 * Best line under each loss. MSE is closed-form; MAE and Huber use iteratively reweighted least squares
 * (weight 1/|r| for MAE, min(1, δ/|r|) for Huber), which converges to their minimizers.
 */
export function fitLine(id: LossId, pts: Pt[], delta: number): Line {
  let l = wls(pts, pts.map(() => 1));
  if (id === "mse") return l;
  for (let it = 0; it < 200; it++) {
    const w = residuals(pts, l).map((r) => {
      const a = Math.max(Math.abs(r), 1e-6);
      return id === "mae" ? 1 / a : Math.min(1, delta / a);
    });
    l = wls(pts, w);
  }
  return l;
}

/* ---------------- classification ---------------- */

export type Example = { id: string; y: 0 | 1; p: number; wrong?: boolean };
export const BASE_EXAMPLES: Example[] = [
  { id: "A", y: 1, p: 0.95 }, { id: "B", y: 1, p: 0.7 }, { id: "C", y: 1, p: 0.4 },
  { id: "D", y: 0, p: 0.1 }, { id: "E", y: 0, p: 0.3 }, { id: "F", y: 0, p: 0.75 },
];
export const CONFIDENT_WRONG: Example = { id: "G", y: 1, p: 0.02, wrong: true };
/** Probability the model gave to the TRUE class. */
export const qOf = (e: Example) => (e.y ? e.p : 1 - e.p);
export const bce = (q: number) => -Math.log(Math.max(q, 1e-12));
export const mseProb = (q: number) => (1 - q) ** 2;
/** |∂loss/∂z| through a sigmoid output (z = logit): BCE → 1 − q, MSE → 2q(1 − q)². */
export const bceGrad = (q: number) => 1 - q;
export const mseGrad = (q: number) => 2 * q * (1 - q) ** 2;

/* ---------------- steps ---------------- */

export type Mode = "regression" | "classification";
export type LFConfig = { mode: Mode; outlier: boolean; outlierDrop: number; delta: number; seed: number; confidentWrong: boolean };
export const DEFAULT_LF: LFConfig = { mode: "regression", outlier: true, outlierDrop: 6, delta: 1, seed: 2, confidentWrong: true };

export type RegPhase = "data" | "candidate" | "curves" | "bars" | "grads" | "fit-mse" | "fit-mae" | "fit-huber" | "compare";
export type ClsPhase = "examples" | "bce" | "mse" | "cbars" | "cgrads" | "csummary";
export type LFState = { phase: RegPhase | ClsPhase; headline: string; sub: string };

const f = (v: number, d = 3) => v.toFixed(d);
const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "loss", message, tex, severity });

export function simulate(raw: LFConfig) {
  const cfg = { ...raw, delta: clamp(raw.delta, 0.1, 10), outlierDrop: clamp(raw.outlierDrop, 0, 15) };
  const pts = makeData(cfg.seed, cfg.outlier, cfg.outlierDrop);
  const clean = pts.filter((p) => !p.outlier);
  const candidate = wls(clean, clean.map(() => 1));
  const res = residuals(pts, candidate);
  const fits = Object.fromEntries(LOSS_IDS.map((id) => [id, fitLine(id, pts, cfg.delta)])) as Record<LossId, Line>;
  const share = (id: LossId) => {
    const ls = res.map((r) => lossOf(id, r, cfg.delta));
    const tot = ls.reduce((s, v) => s + v, 0);
    return cfg.outlier && tot > 0 ? ls[ls.length - 1] / tot : 0;
  };
  const oi = pts.length - 1;
  const reg: SimulationStep<LFState>[] = [
    { id: "data", title: "The data", description: `${clean.length} points follow a straight trend${cfg.outlier ? `, plus one outlier ${cfg.outlierDrop} units below it` : ""}.`,
      state: { phase: "data", headline: cfg.outlier ? "Eight well-behaved points and one outlier" : "Eight points with a straight-line trend", sub: "A loss function decides how much each miss costs, and therefore which line is 'best'." },
      events: [ev("n", `${pts.length} points`), ...(cfg.outlier ? [ev("o", `Outlier at (${pts[oi].x}, ${pts[oi].y})`, undefined, "warning")] : [])] },
    { id: "candidate", title: "Residuals of a candidate line", description: `Take a reasonable line (the least-squares fit to the clean points) and measure each vertical miss r = y − ŷ.`,
      state: { phase: "candidate", headline: "Measure each point's miss (residual) from a candidate line", sub: cfg.outlier ? `The outlier misses by ${f(res[oi], 2)}; the others by less than ${f(Math.max(...res.slice(0, oi).map(Math.abs)) + 0.005, 2)}.` : "Every miss is small." },
      events: [ev("r", `residuals: ${res.map((r) => f(r, 2)).join(", ")}`)] },
    { id: "curves", title: "Residual → loss", description: `Each residual is fed to three loss curves: MSE = r², MAE = |r|, Huber (δ = ${cfg.delta}) = ½r² near 0 and linear beyond δ.`,
      state: { phase: "curves", headline: "Each residual slides onto three loss curves", sub: "Same misses, three price lists: squared (MSE), absolute (MAE), and Huber (squared small, linear large)." },
      events: LOSS_IDS.map((id) => ev(id, `${LOSS_NAME[id]} mean = ${f(totalLoss(id, pts, candidate, cfg.delta))}`, `\\overline{\\ell}_{\\text{${LOSS_NAME[id]}}} = ${f(totalLoss(id, pts, candidate, cfg.delta))}`)) },
    { id: "bars", title: "Per-point loss", description: cfg.outlier ? `Share of total loss from the outlier: MSE ${f(100 * share("mse"), 0)}%, MAE ${f(100 * share("mae"), 0)}%, Huber ${f(100 * share("huber"), 0)}%.` : "Without an outlier all three losses spread their cost fairly evenly.",
      state: { phase: "bars", headline: cfg.outlier ? `Under MSE the outlier is ${f(100 * share("mse"), 0)}% of the total loss` : "Every point costs a similar amount", sub: cfg.outlier ? `Under MAE it's ${f(100 * share("mae"), 0)}%, under Huber ${f(100 * share("huber"), 0)}%. Squaring lets one point dominate.` : "Turn on the outlier to see squaring amplify one big miss." },
      events: cfg.outlier ? LOSS_IDS.map((id) => ev(`s${id}`, `${LOSS_NAME[id]}: outlier share ${f(100 * share(id), 1)}%`, undefined, id === "mse" ? "warning" : undefined)) : [] },
    { id: "grads", title: "Gradient per point", description: `The pull each point exerts on the fit is the loss's slope dℓ/dr: MSE 2r grows without limit, MAE is always ±1, Huber is r clipped to ±${cfg.delta}.`,
      state: { phase: "grads", headline: "How hard does each point pull on the line?", sub: "Gradient = slope of the loss curve. MSE's pull grows with the miss; MAE and Huber cap it." },
      events: cfg.outlier ? LOSS_IDS.map((id) => ev(`g${id}`, `${LOSS_NAME[id]} outlier pull |dℓ/dr| = ${f(Math.abs(gradOf(id, res[oi], cfg.delta)), 2)}`)) : [] },
    ...LOSS_IDS.map((id): SimulationStep<LFState> => ({
      id: `fit-${id}`, title: `Best line under ${LOSS_NAME[id]}`,
      description: `Minimizing mean ${LOSS_NAME[id]} gives ŷ = ${f(fits[id].intercept)} + ${f(fits[id].slope)}·x (${id === "mse" ? "closed form" : "iteratively reweighted least squares"}).`,
      state: { phase: `fit-${id}` as RegPhase, headline: `${LOSS_NAME[id]} fit: slope ${f(fits[id].slope, 2)}${cfg.outlier ? (Math.abs(fits[id].slope - candidate.slope) > 0.1 ? " (dragged by the outlier)" : " (barely moved by the outlier)") : ""}`,
        sub: id === "mse" ? "MSE's line tilts toward the outlier to shrink its huge squared miss." : id === "mae" ? "MAE's line follows the majority: like a median, one outlier can't drag it far." : `Huber behaves like MSE for misses under δ = ${cfg.delta} and like MAE beyond it.` },
      events: [ev(`f${id}`, `${LOSS_NAME[id]} fit: slope ${f(fits[id].slope, 4)}, intercept ${f(fits[id].intercept, 4)}`, `\\hat{y} = ${f(fits[id].intercept)} + ${f(fits[id].slope)}\\,x`)],
    })),
    { id: "compare", title: "Compare the fits", description: `True trend slope ≈ 0.5. MSE ${f(fits.mse.slope)}, MAE ${f(fits.mae.slope)}, Huber ${f(fits.huber.slope)}.`,
      state: { phase: "compare", headline: cfg.outlier ? "One outlier, three different 'best' lines" : "With clean data, all three losses agree", sub: cfg.outlier ? "Pick the loss that matches what you believe about your errors: rare big errors → MAE/Huber." : "Loss choice matters most when errors have heavy tails." },
      events: LOSS_IDS.map((id) => ev(`c${id}`, `${LOSS_NAME[id]}: slope ${f(fits[id].slope)}`)) },
  ];

  const examples = cfg.confidentWrong ? [...BASE_EXAMPLES, CONFIDENT_WRONG] : BASE_EXAMPLES;
  const qs = examples.map(qOf);
  const meanB = qs.reduce((s, q) => s + bce(q), 0) / qs.length, meanM = qs.reduce((s, q) => s + mseProb(q), 0) / qs.length;
  const g = cfg.confidentWrong ? qOf(CONFIDENT_WRONG) : 0;
  const cls: SimulationStep<LFState>[] = [
    { id: "examples", title: "Predictions to score", description: `${examples.length} predictions. The x-axis is q, the probability the model gave to the TRUE class (1 = perfectly right, 0 = completely wrong).`,
      state: { phase: "examples", headline: "Each prediction lands at q = probability given to the true class", sub: cfg.confidentWrong ? "Example G is confidently wrong: true class 1, predicted 0.02." : "Right side = right and confident, left side = wrong and confident." },
      events: examples.map((e) => ev(e.id, `${e.id}: y = ${e.y}, p = ${e.p} → q = ${f(qOf(e), 2)}`, undefined, e.wrong ? "warning" : undefined)) },
    { id: "bce", title: "Binary cross-entropy", description: `BCE = −log q. It is 0 for a perfect prediction and grows without bound as q → 0. Mean BCE = ${f(meanB)}.`,
      state: { phase: "bce", headline: "Cross-entropy: −log q explodes for confident mistakes", sub: cfg.confidentWrong ? `G costs −log 0.02 = ${f(bce(g), 2)}, more than all the others combined.` : "Being wrong costs a lot; being confidently wrong costs far more." },
      events: [ev("mb", `mean BCE = ${f(meanB, 4)}`, `\\text{BCE} = -\\tfrac{1}{n}\\sum \\log q_i = ${f(meanB, 4)}`)] },
    { id: "mse", title: "MSE on probabilities", description: `MSE = (1 − q)² is capped at 1, so even a totally confident mistake costs at most 1. Mean MSE = ${f(meanM)}.`,
      state: { phase: "mse", headline: "MSE on probabilities: the penalty can never exceed 1", sub: "A confident mistake costs barely more than a hesitant one." },
      events: [ev("mm", `mean MSE = ${f(meanM, 4)}`, `\\text{MSE} = \\tfrac{1}{n}\\sum (1 - q_i)^2 = ${f(meanM, 4)}`)] },
    { id: "cbars", title: "Per-example loss", description: "Bars compare each example's BCE and MSE.",
      state: { phase: "cbars", headline: "Per-example cost: BCE singles out the confident mistake", sub: "MSE spreads the blame evenly; BCE concentrates it where the model is most wrong." }, events: [] },
    { id: "cgrads", title: "Gradient through the sigmoid", description: `What training feels: |∂loss/∂z| at the logit. BCE: 1 − q. MSE: 2q(1 − q)², which vanishes as q → 0.${cfg.confidentWrong ? ` For G: BCE ${f(bceGrad(g))}, MSE ${f(mseGrad(g))}.` : ""}`,
      state: { phase: "cgrads", headline: cfg.confidentWrong ? `For G, BCE pulls ${f(bceGrad(g) / mseGrad(g), 0)}× harder than MSE` : "BCE's gradient is largest exactly where the model is most wrong", sub: "With MSE + sigmoid, a saturated wrong prediction gets almost no gradient, so it learns very slowly." },
      events: [ev("gb", "BCE gradient = 1 − q", "\\left|\\tfrac{\\partial\\, \\text{BCE}}{\\partial z}\\right| = 1 - q"), ev("gm", "MSE gradient = 2q(1 − q)²", "\\left|\\tfrac{\\partial\\, \\text{MSE}}{\\partial z}\\right| = 2q(1-q)^2", cfg.confidentWrong ? "warning" : undefined)] },
    { id: "csummary", title: "Why classifiers use cross-entropy", description: "Cross-entropy matches the probabilistic meaning of the output and keeps a strong gradient for confident mistakes.",
      state: { phase: "csummary", headline: "Use cross-entropy for probabilities, not MSE", sub: "It punishes confident mistakes and keeps their gradient alive." }, events: [] },
  ];
  return { cfg, pts, candidate, res, fits, examples, steps: cfg.mode === "regression" ? reg : cls, share };
}
