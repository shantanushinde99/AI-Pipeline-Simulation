import { clamp } from "../../lib/math";
import { classificationMetrics } from "../../lib/metrics";
import { rng } from "../../lib/random";
import { churnData, dropFeature, trainLogistic, type Dataset, type Row } from "../../scenarios/ml-model";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type ModelKind = "logistic" | "knn";
export type LeakConfig = {
  leakTarget: boolean; // keep `cancel_ticket_logged`, written after the label is known
  scaleBeforeSplit: boolean; // fit the scaler on train + test rows
  duplicates: boolean; // copy 30% of rows before splitting, so twins land on both sides
  model: ModelKind;
  n: number;
  seed: number;
};
export const DEFAULT_LEAK: LeakConfig = { leakTarget: true, scaleBeforeSplit: false, duplicates: false, model: "logistic", n: 240, seed: 7 };
export const DUP_FRAC = 0.3;
export const TEST_FRAC = 0.3;
export const LEAK_FEATURE = "cancel_ticket_logged";

export type Phase = "load" | "features" | "split" | "scale" | "train" | "evaluate" | "production";
export const PHASES: Phase[] = ["load", "features", "split", "scale", "train", "evaluate", "production"];

/** One original row, tracked through the pipeline. `twinOf` marks a duplicate copy of another row. */
export type Tracked = Row & { id: number; twinOf?: number; split: "train" | "test" };
export type Scores = { accuracy: number; recall: number; precision: number; f1: number; correct: boolean[] };

export type LeakSim = {
  cfg: LeakConfig;
  features: string[];
  rows: Tracked[]; // after duplication + shuffle, with split assigned
  nTrain: number; nTest: number;
  leakedTwins: number; // test rows with an identical twin in train
  corr: { name: string; r: number }[]; // feature–label correlation on the training split
  scaler: { fitOn: "train" | "train + test"; mean: number[]; std: number[]; trainOnlyMean: number[]; trainOnlyStd: number[] };
  weights: { name: string; w: number }[]; // logistic only; empty for k-NN
  offline: Scores;
  production: Scores & { n: number };
  prodRows: Row[];
  steps: SimulationStep<{ phase: Phase }>[];
};

const pearson = (a: number[], b: number[]) => {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
};

export function fitScaler(rows: Row[], d: number) {
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r.x[j], 0) / rows.length);
  const std = Array.from({ length: d }, (_, j) => Math.sqrt(rows.reduce((s, r) => s + (r.x[j] - mean[j]) ** 2, 0) / rows.length) || 1);
  return { mean, std };
}

/** 1-nearest-neighbour: copies the label of the closest training row, so it memorizes exact twins. */
export function knnPredict(train: Row[], x: number[]): number {
  let best = Infinity, y = 0;
  for (const r of train) {
    const d = r.x.reduce((s, v, j) => s + (v - x[j]) ** 2, 0);
    if (d < best) { best = d; y = r.y; }
  }
  return y;
}

function score(labels: number[], preds: number[]): Scores {
  const m = classificationMetrics(labels, preds);
  return { accuracy: m.accuracy, recall: m.recall, precision: m.precision, f1: m.f1, correct: preds.map((p, i) => p === labels[i]) };
}

export function simulate(raw: LeakConfig): LeakSim {
  const cfg = { ...raw, n: Math.round(clamp(raw.n, 40, 600)) };
  const r = rng(cfg.seed * 7919 + 13);
  let ds: Dataset = churnData(cfg.seed, cfg.n);
  let prod: Dataset = churnData(cfg.seed + 1000, cfg.n, true); // fresh customers; the leak column is always 0 at prediction time
  if (!cfg.leakTarget) { ds = dropFeature(ds, LEAK_FEATURE); prod = dropFeature(prod, LEAK_FEATURE); }
  const d = ds.features.length;

  // Duplicate rows, then shuffle and split. Duplication before splitting is the bug.
  let rows: Omit<Tracked, "split">[] = ds.rows.map((row, id) => ({ ...row, id }));
  if (cfg.duplicates) {
    const copies = Math.round(cfg.n * DUP_FRAC);
    for (let c = 0; c < copies; c++) {
      const src = rows[Math.floor(r.next() * cfg.n)];
      rows.push({ x: [...src.x], y: src.y, id: rows.length, twinOf: src.id });
    }
  }
  rows = [...rows];
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  const nTest = Math.round(rows.length * TEST_FRAC);
  const tracked: Tracked[] = rows.map((row, i) => ({ ...row, split: i < rows.length - nTest ? "train" : "test" }));
  const train = tracked.filter((t) => t.split === "train"), test = tracked.filter((t) => t.split === "test");
  const family = (t: Tracked) => t.twinOf ?? t.id;
  const trainFamilies = new Set(train.map(family));
  const leakedTwins = test.filter((t) => trainFamilies.has(family(t))).length;

  // Scaler: fit on train only (correct) or on everything (leak), then apply the same transform everywhere.
  const tOnly = fitScaler(train, d);
  const sc = cfg.scaleBeforeSplit ? fitScaler(tracked, d) : tOnly;
  const z = (row: Row): Row => ({ y: row.y, x: row.x.map((v, j) => (v - sc.mean[j]) / sc.std[j]) });
  const zTrain = train.map(z), zTest = test.map(z), zProd = prod.rows.map(z);

  let predict: (x: number[]) => number;
  let weights: LeakSim["weights"] = [];
  if (cfg.model === "logistic") {
    const m = trainLogistic({ features: ds.features, rows: zTrain });
    // trainLogistic re-standardizes internally; report weights per original feature.
    weights = ds.features.map((name, j) => ({ name, w: m.weights[j] }));
    predict = (x) => (m.prob(x) >= 0.5 ? 1 : 0);
  } else {
    predict = (x) => knnPredict(zTrain, x);
  }
  const offline = score(zTest.map((t) => t.y), zTest.map((t) => predict(t.x)));
  const production = { ...score(zProd.map((t) => t.y), zProd.map((t) => predict(t.x))), n: zProd.length };
  const corr = ds.features.map((name, j) => ({ name, r: pearson(train.map((t) => t.x[j]), train.map((t) => t.y)) }));

  const leaks = [cfg.leakTarget && "target-leakage feature", cfg.scaleBeforeSplit && "scaler fit before split", cfg.duplicates && "duplicates across split"].filter(Boolean) as string[];
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const ev = (id: string, message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id, type: "pipeline", message, severity, tex });
  const leakCol = ds.features.indexOf(LEAK_FEATURE);
  const steps: SimulationStep<{ phase: Phase }>[] = [
    { id: "load", title: "Load the raw data", description: `${cfg.n} customers, each with ${d} feature columns and a churn label (${ds.rows.filter((x) => x.y).length} churned).`, state: { phase: "load" },
      events: [ev("l1", `loaded ${cfg.n} rows`)] },
    { id: "features", title: "Build the feature columns", description: cfg.leakTarget ? `${LEAK_FEATURE} is kept. It is filled in by billing after a customer cancels, so it encodes the label itself.` : `${LEAK_FEATURE} is dropped: it would not exist at prediction time.`, state: { phase: "features" },
      events: [cfg.leakTarget ? ev("f1", `${LEAK_FEATURE} kept: corr with label = ${corr[leakCol].r.toFixed(2)}`, "error", `r(\\text{cancel\\_ticket}, y) = ${corr[leakCol].r.toFixed(2)}`) : ev("f1", `features: ${ds.features.join(", ")}`)] },
    { id: "split", title: "Split into train and test", description: cfg.duplicates ? `${rows.length - cfg.n} duplicate rows were added before the split; ${leakedTwins} of ${nTest} test rows have an identical twin in train.` : `${train.length} rows train the model, ${nTest} are held out to estimate production performance.`, state: { phase: "split" },
      events: [ev("s1", `train ${train.length} · test ${nTest}`), ...(cfg.duplicates ? [ev("s2", `${leakedTwins} test rows also appear in train`, "error")] : [])] },
    { id: "scale", title: "Fit and apply the scaler", description: cfg.scaleBeforeSplit ? "The mean and standard deviation were computed on train + test rows, so test statistics leak into training." : "Mean and standard deviation come from training rows only, then the same numbers transform test and production rows.", state: { phase: "scale" },
      events: [ev("c1", `scaler fit on ${cfg.scaleBeforeSplit ? "train + test" : "train"}: mean(tenure) = ${sc.mean[0].toFixed(2)}`, cfg.scaleBeforeSplit ? "warning" : "info")] },
    { id: "train", title: cfg.model === "logistic" ? "Train logistic regression" : "Fit 1-nearest-neighbour", description: cfg.model === "logistic" ? "Gradient descent learns one weight per feature. A leaked column grabs most of the weight." : "1-NN stores every training row and copies the label of the closest one.", state: { phase: "train" },
      events: [ev("t1", cfg.model === "logistic" ? `weights: ${weights.map((w) => `${w.name}=${w.w.toFixed(2)}`).join(", ")}` : `stored ${train.length} training rows`)] },
    { id: "evaluate", title: "Evaluate on the held-out test set", description: `Offline accuracy ${pct(offline.accuracy)}, recall ${pct(offline.recall)}.`, state: { phase: "evaluate" },
      events: [ev("e1", `offline accuracy ${pct(offline.accuracy)}`, leaks.length ? "warning" : "info")] },
    { id: "production", title: "Score fresh production customers", description: `On ${prod.rows.length} new customers the same pipeline gets ${pct(production.accuracy)} accuracy, recall ${pct(production.recall)}.`, state: { phase: "production" },
      events: [ev("p1", `production accuracy ${pct(production.accuracy)} (gap ${((offline.accuracy - production.accuracy) * 100).toFixed(1)} pts)`, offline.accuracy - production.accuracy > 0.05 ? "error" : "info"),
        ...(leaks.length ? [ev("p2", `active leaks: ${leaks.join(", ")}`, "warning")] : [])] },
  ];

  return {
    cfg, features: ds.features, rows: tracked, nTrain: train.length, nTest, leakedTwins, corr,
    scaler: { fitOn: cfg.scaleBeforeSplit ? "train + test" : "train", mean: sc.mean, std: sc.std, trainOnlyMean: tOnly.mean, trainOnlyStd: tOnly.std },
    weights, offline, production, prodRows: prod.rows, steps,
  };
}
