import { sigmoid } from "../lib/math";
import { classificationMetrics } from "../lib/metrics";
import { rng } from "../lib/random";

export type Row = { x: number[]; y: number };
export type Dataset = { features: string[]; rows: Row[] };

/** Logistic regression with standardization, trained by full-batch gradient descent. Deterministic. */
export function trainLogistic(ds: Dataset, opts: { epochs?: number; lr?: number; l2?: number; positiveWeight?: number } = {}) {
  const { epochs = 400, lr = 0.5, l2 = 1e-3, positiveWeight = 1 } = opts;
  const d = ds.features.length;
  const mu = Array.from({ length: d }, (_, j) => ds.rows.reduce((s, r) => s + r.x[j], 0) / ds.rows.length);
  const sd = Array.from({ length: d }, (_, j) => Math.sqrt(ds.rows.reduce((s, r) => s + (r.x[j] - mu[j]) ** 2, 0) / ds.rows.length) || 1);
  const z = (x: number[]) => x.map((v, j) => (v - mu[j]) / sd[j]);
  const Z = ds.rows.map((r) => z(r.x));
  let w = new Array<number>(d).fill(0);
  let b = 0;
  const wsum = ds.rows.reduce((s, r) => s + (r.y ? positiveWeight : 1), 0);
  for (let e = 0; e < epochs; e++) {
    const gw = new Array<number>(d).fill(0);
    let gb = 0;
    ds.rows.forEach((r, i) => {
      const p = sigmoid(Z[i].reduce((s, v, j) => s + v * w[j], b));
      const c = (r.y ? positiveWeight : 1) * (p - r.y);
      Z[i].forEach((v, j) => (gw[j] += c * v));
      gb += c;
    });
    w = w.map((wj, j) => wj - lr * (gw[j] / wsum + l2 * wj));
    b -= (lr * gb) / wsum;
  }
  return {
    features: ds.features, weights: w, bias: b, mean: mu, std: sd,
    prob: (x: number[]) => sigmoid(z(x).reduce((s, v, j) => s + v * w[j], b)),
  };
}
export type Model = ReturnType<typeof trainLogistic>;

export function evaluateModel(m: Model, rows: Row[], threshold = 0.5) {
  const probs = rows.map((r) => m.prob(r.x));
  const preds = probs.map((p) => (p >= threshold ? 1 : 0));
  return { ...classificationMetrics(rows.map((r) => r.y), preds), probs, n: rows.length, positives: rows.filter((r) => r.y).length };
}

/**
 * Customer churn data. `cancel_ticket_logged` is written by the billing system *after* a customer
 * cancels, so at prediction time in production it is always 0. Including it is label leakage.
 */
export function churnData(seed: number, n: number, production = false): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const tenure = Math.round(r.next() * 60);
    const spend = 20 + r.next() * 180;
    const tickets = Math.floor(r.next() * r.next() * 10);
    const p = sigmoid(-0.07 * tenure + 0.006 * spend + 0.45 * tickets - 0.6);
    const y = r.next() < p ? 1 : 0;
    const leak = production ? 0 : y === 1 ? (r.next() < 0.97 ? 1 : 0) : r.next() < 0.02 ? 1 : 0;
    return { x: [tenure, spend, tickets, leak], y };
  });
  return { features: ["tenure_months", "monthly_spend", "support_tickets", "cancel_ticket_logged"], rows };
}

export const dropFeature = (ds: Dataset, name: string): Dataset => {
  const j = ds.features.indexOf(name);
  return { features: ds.features.filter((_, k) => k !== j), rows: ds.rows.map((r) => ({ ...r, x: r.x.filter((_, k) => k !== j) })) };
};

/** Card-transaction fraud data with ~3% positives. */
export function fraudData(seed: number, n: number): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const fraud = r.next() < 0.03 ? 1 : 0;
    const amount = fraud ? 120 + r.gaussian() * 60 : 60 + r.gaussian() * 40;
    const foreign = r.next() < (fraud ? 0.55 : 0.12) ? 1 : 0;
    const nightHour = r.next() < (fraud ? 0.5 : 0.18) ? 1 : 0;
    const velocity = fraud ? 2.5 + r.gaussian() : 1 + r.gaussian() * 0.8;
    return { x: [amount, foreign, nightHour, velocity], y: fraud };
  });
  return { features: ["amount_usd", "is_foreign", "night_hour", "txn_velocity_1h"], rows };
}

/** Column reordering bug at the serving boundary. */
export const reorder = (rows: Row[], order: number[]) => rows.map((r) => ({ ...r, x: order.map((j) => r.x[j]) }));
