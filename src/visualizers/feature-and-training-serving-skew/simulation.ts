import { sigmoid } from "../../lib/math";
import type { SimulationStep } from "../../types/simulation";

/** Snapshot time for both paths: 2026-10-19 00:00 UTC. Order times are stored as hours before it. */
export const AS_OF = "2026-10-19 00:00 UTC";
export const STALE_HOURS = 48; // the online aggregates table was last refreshed 2 days before AS_OF
export const LOCAL_OFFSET = -5; // the online service's server clock: UTC−5
export const TRAIN_MEAN_RATING = 4.2;

export type Order = { hoursAgo: number; cents: number };
export type Entity = { id: string; orders: Order[]; ratings: number[]; tenureDays: number };

export const ENTITIES: Entity[] = [
  { id: "c_101", orders: [{ hoursAgo: 3, cents: 4250 }, { hoursAgo: 30, cents: 1800 }, { hoursAgo: 100, cents: 5600 }, { hoursAgo: 200, cents: 2100 }], ratings: [5, 4], tenureDays: 400 },
  { id: "c_102", orders: [{ hoursAgo: 50, cents: 12000 }, { hoursAgo: 150, cents: 8000 }], ratings: [], tenureDays: 30 },
  { id: "c_103", orders: [{ hoursAgo: 20, cents: 2500 }, { hoursAgo: 40, cents: 3100 }, { hoursAgo: 60, cents: 1900 }, { hoursAgo: 80, cents: 2700 }, { hoursAgo: 120, cents: 2200 }], ratings: [3, 4, 4], tenureDays: 800 },
  { id: "c_104", orders: [{ hoursAgo: 180, cents: 9900 }], ratings: [], tenureDays: 15 },
  { id: "c_105", orders: [{ hoursAgo: 5, cents: 3100 }, { hoursAgo: 100, cents: 2900 }], ratings: [5], tenureDays: 120 },
  { id: "c_106", orders: [{ hoursAgo: 70, cents: 1500 }, { hoursAgo: 90, cents: 1700 }, { hoursAgo: 140, cents: 1600 }], ratings: [4, 5], tenureDays: 260 },
];

export type FeatureId = "orders_7d" | "avg_order_value" | "last_order_hour" | "avg_rating" | "tenure_days";
export type BugId = "stale_join" | "unit" | "timezone" | "default_fill";
export const FEATURES: FeatureId[] = ["orders_7d", "avg_order_value", "last_order_hour", "avg_rating", "tenure_days"];
export const BUG_OF: Record<FeatureId, BugId | null> = { orders_7d: "stale_join", avg_order_value: "unit", last_order_hour: "timezone", avg_rating: "default_fill", tenure_days: null };
export const BUGS: { id: BugId; name: string }[] = [
  { id: "stale_join", name: "Stale join (aggregates table 2 days old)" },
  { id: "unit", name: "Unit mismatch (cents vs dollars)" },
  { id: "timezone", name: "Timezone offset (server local time)" },
  { id: "default_fill", name: "Default fill (null → 0 vs mean)" },
];

export type FeatureDef = { label: string; unit: string; batch: string; online: string; onlineBug: string; why: string };
export const DEFS: Record<FeatureId, FeatureDef> = {
  orders_7d: {
    label: "orders in last 7 days", unit: "count",
    batch: "SELECT COUNT(*) FROM orders\nWHERE ts >= :as_of - INTERVAL '7 days'\n  AND ts <  :as_of",
    online: "sum(1 for o in orders\n    if as_of - 7d <= o.ts < as_of)",
    onlineBug: "agg = aggregates.get(customer_id)\nreturn agg.orders_7d  # table refreshed 2 days ago",
    why: "The online path reads a pre-aggregated table that was last refreshed 2 days ago: it misses the newest orders and still counts ones that have aged out.",
  },
  avg_order_value: {
    label: "average order value", unit: "$",
    batch: "SELECT AVG(amount_cents) / 100.0\nFROM orders",
    online: "mean(o.amount_cents for o in orders) / 100",
    onlineBug: "mean(o.amount_cents for o in orders)\n# forgot / 100: returns cents",
    why: "Training saw dollars (e.g. 34.38); serving sends cents (3437.50). The model multiplies a 100× larger number by the same weight.",
  },
  last_order_hour: {
    label: "hour of last order", unit: "h",
    batch: "SELECT EXTRACT(HOUR FROM MAX(ts)\n  AT TIME ZONE 'UTC')",
    online: "last.ts.astimezone(timezone.utc).hour",
    onlineBug: "datetime.fromtimestamp(last.ts).hour\n# naive: server clock is UTC−5",
    why: "The warehouse stores and extracts UTC; the service converts with the server's local clock, shifting every hour by −5.",
  },
  avg_rating: {
    label: "average rating", unit: "★",
    batch: "SELECT COALESCE(AVG(rating), 4.2)\n-- 4.2 = training-set mean",
    online: "mean(ratings) if ratings else 4.2",
    onlineBug: "mean(ratings) if ratings else 0\n# new customers get 0 stars",
    why: "Customers without ratings were imputed with the mean (4.2) in training but get 0 online, which the model reads as 'hated everything'.",
  },
  tenure_days: {
    label: "tenure", unit: "days",
    batch: "DATE_DIFF(:as_of, signup_date, DAY)",
    online: "(as_of - signup_date).days",
    onlineBug: "(as_of - signup_date).days",
    why: "Both paths agree here: a control feature with no bug.",
  },
};

const utcHour = (hoursAgo: number) => (((24 - (hoursAgo % 24)) % 24) + 24) % 24;
const avg = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;

/** The training path's definition (batch SQL), which is the reference. */
export function batchFeature(e: Entity, f: FeatureId): number {
  if (f === "orders_7d") return e.orders.filter((o) => o.hoursAgo > 0 && o.hoursAgo <= 168).length;
  if (f === "avg_order_value") return e.orders.length ? avg(e.orders.map((o) => o.cents)) / 100 : 0;
  if (f === "last_order_hour") return e.orders.length ? utcHour(Math.min(...e.orders.map((o) => o.hoursAgo))) : 0;
  if (f === "avg_rating") return e.ratings.length ? avg(e.ratings) : TRAIN_MEAN_RATING;
  return e.tenureDays;
}

/** The serving path. With `shared` both paths call the same definition, so bugs cannot diverge. */
export function onlineFeature(e: Entity, f: FeatureId, bugs: ReadonlySet<BugId>, shared: boolean): number {
  const bug = BUG_OF[f];
  if (shared || !bug || !bugs.has(bug)) return batchFeature(e, f);
  if (f === "orders_7d") return e.orders.filter((o) => o.hoursAgo > STALE_HOURS && o.hoursAgo <= STALE_HOURS + 168).length;
  if (f === "avg_order_value") return e.orders.length ? avg(e.orders.map((o) => o.cents)) : 0;
  if (f === "last_order_hour") return (batchFeature(e, f) + LOCAL_OFFSET + 24) % 24;
  return e.ratings.length ? avg(e.ratings) : 0; // avg_rating
}

/** Hand-set logistic model ("will buy again in 7 days"), standing in for the model trained on batch features. */
export const WEIGHTS: Record<FeatureId, number> = { orders_7d: 0.45, avg_order_value: 0.02, last_order_hour: 0.06, avg_rating: 0.8, tenure_days: 0.002 };
export const BIAS = -2.0;
export const CENTER: Partial<Record<FeatureId, number>> = { last_order_hour: 12, avg_rating: 4 };
export type FeatureVec = Record<FeatureId, number>;
export function predict(x: FeatureVec) {
  const terms = FEATURES.map((f) => ({ f, v: WEIGHTS[f] * (x[f] - (CENTER[f] ?? 0)) }));
  const z = terms.reduce((s, t) => s + t.v, BIAS);
  return { terms, z, p: sigmoid(z) };
}

export type Row = { id: string; batch: Partial<FeatureVec>; online: Partial<FeatureVec>; pBatch?: number; pOnline?: number };
export type SkewState = { phase: "raw" | "feature" | "score" | "report"; feature?: FeatureId; rows: Row[]; headline: string; sub: string };
export type SkewInput = { bugs: BugId[]; shared: boolean };
export const DEFAULT_INPUT: SkewInput = { bugs: ["stale_join", "unit", "timezone", "default_fill"], shared: false };

export const differs = (a: number | undefined, b: number | undefined) => a !== undefined && b !== undefined && Math.abs(a - b) > 1e-9;

export function featureVectors(input: SkewInput) {
  const bugs = new Set(input.bugs);
  return ENTITIES.map((e) => {
    const b = Object.fromEntries(FEATURES.map((f) => [f, batchFeature(e, f)])) as FeatureVec;
    const o = Object.fromEntries(FEATURES.map((f) => [f, onlineFeature(e, f, bugs, input.shared)])) as FeatureVec;
    return { id: e.id, batch: b, online: o, pBatch: predict(b).p, pOnline: predict(o).p };
  });
}

export function skewReport(input: SkewInput) {
  const v = featureVectors(input);
  return {
    perFeature: FEATURES.map((f) => ({ f, mismatches: v.filter((r) => differs(r.batch[f], r.online[f])).length })),
    maxDp: Math.max(...v.map((r) => Math.abs(r.pOnline - r.pBatch))),
    flipped: v.filter((r) => (r.pBatch >= 0.5) !== (r.pOnline >= 0.5)).length,
  };
}

export function simulate(input: SkewInput): SimulationStep<SkewState>[] {
  const vecs = featureVectors(input);
  const steps: SimulationStep<SkewState>[] = [];
  const rows: Row[] = ENTITIES.map((e) => ({ id: e.id, batch: {}, online: {} }));
  const snap = () => rows.map((r) => ({ ...r, batch: { ...r.batch }, online: { ...r.online } }));

  steps.push({
    id: "raw", title: "Same raw events, two code paths",
    description: `${ENTITIES.length} customers' raw orders and ratings (as of ${AS_OF}) feed both the nightly batch SQL that built the training set and the online feature service that feeds the live model.`,
    state: { phase: "raw", rows: snap(), headline: "The same raw events flow into two separately written pipelines", sub: input.shared ? "Fix on: both paths call one shared feature definition." : "Training used batch SQL; serving re-implemented the features in Python." },
    events: [{ id: "raw", type: "data", message: `${ENTITIES.reduce((s, e) => s + e.orders.length, 0)} orders, ${ENTITIES.reduce((s, e) => s + e.ratings.length, 0)} ratings for ${ENTITIES.length} customers` }],
  });

  for (const f of FEATURES) {
    vecs.forEach((v, i) => { rows[i].batch[f] = v.batch[f]; rows[i].online[f] = v.online[f]; });
    const bad = vecs.filter((v) => differs(v.batch[f], v.online[f]));
    const bug = BUG_OF[f];
    const active = !!bug && input.bugs.includes(bug) && !input.shared;
    steps.push({
      id: `f-${f}`, title: `Feature: ${f}`,
      description: `${DEFS[f].label}. ${active ? DEFS[f].why : input.shared ? "Both paths call the same shared definition, so the values are identical by construction." : "This path's code matches the training definition here."}`,
      state: { phase: "feature", feature: f, rows: snap(),
        headline: bad.length ? `${f}: ${bad.length} of ${ENTITIES.length} customers get a different value at serving time` : `${f}: training and serving agree for every customer`,
        sub: active ? BUGS.find((b) => b.id === bug)!.name : "no skew on this feature" },
      events: bad.length
        ? bad.map((v) => ({ id: `${f}-${v.id}`, type: "skew", severity: "warning" as const, message: `${v.id}: train ${fmtF(f, v.batch[f])} vs serve ${fmtF(f, v.online[f])}` }))
        : [{ id: `${f}-ok`, type: "skew", message: `${f}: 0 mismatches` }],
    });
  }

  vecs.forEach((v, i) => { rows[i].pBatch = v.pBatch; rows[i].pOnline = v.pOnline; });
  const worst = vecs.reduce((a, b) => (Math.abs(b.pOnline - b.pBatch) > Math.abs(a.pOnline - a.pBatch) ? b : a));
  const dWorst = Math.abs(worst.pOnline - worst.pBatch);
  steps.push({
    id: "score", title: "Score both feature vectors",
    description: `The same model scores each customer twice: once with training-path features (what it was validated on) and once with serving-path features (what production actually sees). Largest gap: ${worst.id}, ${worst.pBatch.toFixed(3)} vs ${worst.pOnline.toFixed(3)}.`,
    state: { phase: "score", rows: snap(), headline: dWorst > 1e-9 ? `Same model, same customer, different answer: up to ${(dWorst * 100).toFixed(0)} points apart` : "Same model, same features: identical predictions", sub: "p = probability the customer orders again within 7 days" },
    events: vecs.map((v) => ({ id: `p-${v.id}`, type: "predict", severity: Math.abs(v.pOnline - v.pBatch) > 0.1 ? "warning" as const : undefined, message: `${v.id}: p_train=${v.pBatch.toFixed(3)}, p_serve=${v.pOnline.toFixed(3)}` })),
  });

  const rep = skewReport(input);
  const total = rep.perFeature.reduce((s, x) => s + x.mismatches, 0);
  steps.push({
    id: "report", title: "Skew report & the fix",
    description: `${total} mismatched feature values out of ${ENTITIES.length * FEATURES.length}; ${rep.flipped} customer(s) cross the 0.5 decision threshold. With one shared feature definition (a feature store), serving computes exactly what training computed, so every diff is 0.`,
    state: { phase: "report", rows: snap(), headline: total ? `${total} skewed values; a shared feature definition removes all of them` : "Zero skew: training and serving compute identical features", sub: `max |Δp| = ${rep.maxDp.toFixed(3)} · decisions flipped: ${rep.flipped}` },
    events: [{ id: "rep", type: "report", severity: total ? "error" : "info", message: `mismatches: ${rep.perFeature.map((x) => `${x.f}=${x.mismatches}`).join(", ")}` }],
  });
  return steps;
}

export function fmtF(f: FeatureId, v: number | undefined): string {
  if (v === undefined) return "—";
  if (f === "avg_order_value") return v.toFixed(2);
  if (f === "avg_rating") return v.toFixed(2);
  return String(Math.round(v * 100) / 100);
}
