import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type DatasetId = "houses" | "energy";
export type RegConfig = { dataset: DatasetId; n: number; noise: number; outlier: boolean; altUnits: boolean; seed: number };
export const DEFAULT_REG: RegConfig = { dataset: "houses", n: 10, noise: 1, outlier: false, altUnits: false, seed: 7 };

/** LaTeX for a unit label. "$" is a math delimiter in LaTeX, so units like k$ must escape it inside \text{}. */
export const unitTex = (unit: string) => `\\text{${unit.replace(/\$/g, "\\$")}}`;

export const DATASETS: Record<DatasetId, { name: string; unit: string; altUnit: string; factor: number; what: string }> = {
  houses: { name: "House prices", unit: "k$", altUnit: "$", factor: 1000, what: "price" },
  energy: { name: "Hourly energy use (targets near zero)", unit: "kWh", altUnit: "Wh", factor: 1000, what: "energy" },
};

export type Point = { id: number; y: number; yhat: number };
/** Index of the example that gets the wild prediction when the outlier toggle is on. */
export const OUTLIER_AT = 6;

/** Deterministic toy data. `noise` scales the model's typical error; the outlier adds a single huge miss. */
export function makeData(cfg: RegConfig): Point[] {
  const n = Math.round(clamp(cfg.n, 4, 14));
  const r = rng(cfg.seed);
  const houses = cfg.dataset === "houses";
  const pts = Array.from({ length: n }, (_, i) => {
    const y = houses ? Math.round(150 + r.next() * 350) : Math.round((r.next() < 0.4 ? 0.02 + r.next() * 0.15 : 0.5 + r.next() * 2.5) * 100) / 100;
    const err = r.gaussian() * cfg.noise * (houses ? 22 : 0.18);
    let yhat = houses ? Math.round(y + err) : Math.max(0, Math.round((y + err) * 100) / 100);
    if (cfg.outlier && i === Math.min(OUTLIER_AT, n - 1)) yhat = houses ? y + 260 : Math.round((y + 2.6) * 100) / 100;
    return { id: i + 1, y, yhat };
  });
  const f = cfg.altUnits ? DATASETS[cfg.dataset].factor : 1;
  return pts.map((p) => ({ ...p, y: p.y * f, yhat: Math.round(p.yhat * f * 100) / 100 }));
}

/** All five metrics plus every per-example contribution, so the UI can animate the running sums. */
export function regressionMetrics(pts: readonly Point[]) {
  const n = pts.length;
  const res = pts.map((p) => p.y - p.yhat);
  const abs = res.map(Math.abs);
  const sq = res.map((e) => e * e);
  const ape = pts.map((p, i) => (p.y === 0 ? Number.POSITIVE_INFINITY : (abs[i] / Math.abs(p.y)) * 100));
  const ybar = pts.reduce((s, p) => s + p.y, 0) / n;
  const tot = pts.map((p) => (p.y - ybar) ** 2);
  const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);
  const mae = sum(abs) / n, mse = sum(sq) / n;
  const ssRes = sum(sq), ssTot = sum(tot);
  return { n, res, abs, sq, ape, tot, ybar, mae, mse, rmse: Math.sqrt(mse), ssRes, ssTot, r2: ssTot === 0 ? Number.NaN : 1 - ssRes / ssTot, mape: sum(ape) / n };
}
export type RegMetrics = ReturnType<typeof regressionMetrics>;

export const fmtNum = (v: number) => {
  if (!Number.isFinite(v)) return "∞";
  const a = Math.abs(v);
  if (a >= 1e6) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
};

export type RegPhase = "data" | "residuals" | "mae" | "mse" | "rmse" | "r2" | "mape" | "compare";
export type RegState = { phase: RegPhase };

export function simulate(cfg: RegConfig) {
  const ds = DATASETS[cfg.dataset];
  const unit = cfg.altUnits ? ds.altUnit : ds.unit;
  const pts = makeData(cfg);
  const m = regressionMetrics(pts);
  // Counterfactuals for the comparison step: flip the outlier, flip the units.
  const other = regressionMetrics(makeData({ ...cfg, outlier: !cfg.outlier }));
  const scaled = regressionMetrics(makeData({ ...cfg, altUnits: !cfg.altUnits }));
  const worst = m.abs.indexOf(Math.max(...m.abs));
  const worstApe = m.ape.indexOf(Math.max(...m.ape));
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "metric", message, tex, severity });
  const s = (phase: RegPhase, title: string, description: string, events: SimulationEvent[]): SimulationStep<RegState> => ({ id: phase, title, description, state: { phase }, events });
  const shareSq = m.ssRes ? m.sq[worst] / m.ssRes : 0;
  const u = unitTex(unit);

  const steps: SimulationStep<RegState>[] = [
    s("data", "Actual values vs predictions", `${m.n} examples of ${ds.what} in ${unit}. Filled dots are the true values y, hollow dots the model's predictions ŷ.`,
      [ev("d", `${m.n} pairs (y, ŷ), mean y = ${fmtNum(m.ybar)} ${unit}`)]),
    s("residuals", "Residuals: the miss for each example", `Each bar is a residual e = y − ŷ. The biggest miss is #${worst + 1}: ${fmtNum(m.res[worst])} ${unit}.`,
      [ev("r", `largest |e| = ${fmtNum(m.abs[worst])} at #${worst + 1}`, `e_{${worst + 1}} = ${fmtNum(pts[worst].y)} - ${fmtNum(pts[worst].yhat)} = ${fmtNum(m.res[worst])}`)]),
    s("mae", "MAE: average size of the miss", `Add up every |e| (each bar drops into the running total), then divide by ${m.n}. MAE = ${fmtNum(m.mae)} ${unit}.`,
      [ev("mae", `MAE = ${fmtNum(m.mae)}`, `\\text{MAE} = \\tfrac{1}{${m.n}}\\sum |e_i| = \\tfrac{${fmtNum(m.mae * m.n)}}{${m.n}} = ${fmtNum(m.mae)}\\ ${u}`)]),
    s("mse", "MSE: square first, then average", `Squaring makes big misses count much more: #${worst + 1} alone is ${(shareSq * 100).toFixed(0)}% of the sum. MSE = ${fmtNum(m.mse)} ${unit}².`,
      [ev("mse", `MSE = ${fmtNum(m.mse)}`, `\\text{MSE} = \\tfrac{1}{${m.n}}\\sum e_i^2 = \\tfrac{${fmtNum(m.ssRes)}}{${m.n}} = ${fmtNum(m.mse)}\\ ${u}^2`, shareSq > 0.5 ? "warning" : undefined)]),
    s("rmse", "RMSE: back to the original units", `√MSE = ${fmtNum(m.rmse)} ${unit}. RMSE ≥ MAE always; the gap (${fmtNum(m.rmse - m.mae)}) grows when a few errors are much larger than the rest.`,
      [ev("rmse", `RMSE = ${fmtNum(m.rmse)}`, `\\text{RMSE} = \\sqrt{${fmtNum(m.mse)}} = ${fmtNum(m.rmse)}\\ ${u}`)]),
    s("r2", "R²: better than predicting the mean?", `Compare the model's squared error (SS_res = ${fmtNum(m.ssRes)}) with always predicting the mean ${fmtNum(m.ybar)} (SS_tot = ${fmtNum(m.ssTot)}). R² = ${Number.isFinite(m.r2) ? m.r2.toFixed(3) : "undefined"}.`,
      [ev("r2", `R² = ${m.r2.toFixed(3)}`, `R^2 = 1 - \\tfrac{${fmtNum(m.ssRes)}}{${fmtNum(m.ssTot)}} = ${m.r2.toFixed(3)}`, m.r2 < 0 ? "warning" : undefined)]),
    s("mape", "MAPE: error as a percentage of the truth", `Each |e| is divided by its own |y|. #${worstApe + 1} has y = ${fmtNum(pts[worstApe].y)}, so its error is ${fmtNum(m.ape[worstApe])}% of the truth. MAPE = ${fmtNum(m.mape)}%.`,
      [ev("mape", `MAPE = ${fmtNum(m.mape)}%`, `\\text{MAPE} = \\tfrac{100\\%}{${m.n}}\\sum \\tfrac{|e_i|}{|y_i|} = ${fmtNum(m.mape)}\\%`, m.ape[worstApe] > 100 ? "warning" : undefined)]),
    s("compare", "Which metrics moved, and why", `${cfg.outlier ? "Removing" : "Adding"} the wild prediction changes MSE ×${fmtNum(other.mse / m.mse)} but MAE only ×${fmtNum(other.mae / m.mae)}. Changing units rescales MAE/RMSE ×${DATASETS[cfg.dataset].factor}, MSE ×${DATASETS[cfg.dataset].factor}², while R² and MAPE stay put.`,
      [ev("c", `outlier flip: MSE ×${fmtNum(other.mse / m.mse)}, MAE ×${fmtNum(other.mae / m.mae)}, R² ${m.r2.toFixed(3)} → ${other.r2.toFixed(3)}`)]),
  ];
  return { cfg, ds, unit, pts, m, other, scaled, worst, worstApe, steps };
}
export type RegSim = ReturnType<typeof simulate>;
