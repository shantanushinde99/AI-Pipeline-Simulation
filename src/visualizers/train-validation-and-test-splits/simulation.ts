import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Row = { id: number; x: [number, number]; y: 0 | 1 };
export type Bucket = "train" | "val" | "test";
export const CANDIDATES = [1, 3, 5, 9, 15, 25, 41];

/** True rule: y = 1 above a wavy boundary, then a fraction `noise` of labels is flipped. */
export function makeData(seed: number, n: number, noise: number): Row[] {
  const r = rng(seed);
  return Array.from({ length: n }, (_, id) => {
    const x: [number, number] = [r.next(), r.next()];
    const clean = x[1] > 0.5 + 0.22 * Math.sin(2 * Math.PI * x[0]) ? 1 : 0;
    return { id, x, y: (r.next() < noise ? 1 - clean : clean) as 0 | 1 };
  });
}

export function shuffle<T>(xs: T[], seed: number): T[] {
  const r = rng(seed);
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** k-nearest-neighbour accuracy: each eval row takes the majority label of its k closest training rows. */
export function knnAccuracy(train: Row[], evalRows: Row[], k: number): number {
  if (!evalRows.length || !train.length) return Number.NaN;
  const kk = Math.min(k, train.length);
  let correct = 0;
  for (const e of evalRows) {
    const near = train.map((t) => ({ d: (t.x[0] - e.x[0]) ** 2 + (t.x[1] - e.x[1]) ** 2, y: t.y, id: t.id }))
      .sort((a, b) => a.d - b.d || a.id - b.id).slice(0, kk);
    const ones = near.reduce((s, v) => s + v.y, 0);
    correct += (ones * 2 > kk ? 1 : ones * 2 < kk ? 0 : near[0].y) === e.y ? 1 : 0;
  }
  return correct / evalRows.length;
}

export type Score = { k: number; train: number; val: number; test: number; folds: number[] };
export type SplitPhase = "data" | "shuffle" | "split" | "train" | "fold" | "validate" | "select" | "test";
export type SplitState = { phase: SplitPhase; fold: number | null; revealed: ("train" | "val" | "test")[]; winner: number | null };
export type SplitInput = { seed: number; n: number; noise: number; valFrac: number; testFrac: number; kfold: boolean; folds: number; tuneOnTest: boolean };
export const DEFAULTS: SplitInput = { seed: 4, n: 90, noise: 0.15, valFrac: 0.2, testFrac: 0.2, kfold: false, folds: 5, tuneOnTest: false };

export function simulate(raw: SplitInput) {
  const p = { ...raw, n: Math.round(clamp(raw.n, 20, 200)), noise: clamp(raw.noise, 0, 0.45), testFrac: clamp(raw.testFrac, 0.1, 0.4), folds: Math.round(clamp(raw.folds, 2, 10)) };
  p.valFrac = clamp(raw.valFrac, 0.1, 0.8 - p.testFrac);
  const data = makeData(p.seed, p.n, p.noise);
  const order = shuffle(data, p.seed + 1);
  const nTest = Math.round(p.n * p.testFrac);
  const nVal = p.kfold ? 0 : Math.round(p.n * p.valFrac);
  const bucket: Record<number, Bucket> = {};
  const fold: Record<number, number> = {};
  order.forEach((r, i) => { bucket[r.id] = i < p.n - nTest - nVal ? "train" : i < p.n - nTest ? "val" : "test"; });
  const dev = order.slice(0, p.n - nTest);
  dev.forEach((r, i) => { fold[r.id] = i % p.folds; });
  const of = (b: Bucket) => order.filter((r) => bucket[r.id] === b);
  const [train, val, test] = [of("train"), of("val"), of("test")];
  const fresh = makeData(p.seed + 999, 600, p.noise);

  const scores: Score[] = CANDIDATES.map((k) => {
    const folds = p.kfold ? Array.from({ length: p.folds }, (_, f) => knnAccuracy(dev.filter((r) => fold[r.id] !== f), dev.filter((r) => fold[r.id] === f), k)) : [];
    return {
      k, folds,
      train: knnAccuracy(p.kfold ? dev : train, p.kfold ? dev : train, k),
      val: p.kfold ? folds.reduce((s, v) => s + v, 0) / folds.length : knnAccuracy(train, val, k),
      // Final model is refit on all non-test rows before the single test evaluation.
      test: knnAccuracy(dev, test, k),
    };
  });
  const pick = (key: "val" | "test") => scores.reduce((b, s) => (s[key] > b[key] ? s : b)).k;
  const winner = p.tuneOnTest ? pick("test") : pick("val");
  const w = scores.find((s) => s.k === winner)!;
  const freshAcc = knnAccuracy(dev, fresh, winner);
  const bestFresh = Math.max(...CANDIDATES.map((k) => knnAccuracy(dev, fresh, k)));

  const steps: SimulationStep<SplitState>[] = [];
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "split", message, tex, severity });
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const push = (id: string, title: string, description: string, state: SplitState, events: SimulationEvent[] = []) => steps.push({ id, title, description, state, events });
  const base = { fold: null, revealed: [], winner: null } satisfies Omit<SplitState, "phase">;

  push("data", "Collect labelled rows", `${p.n} labelled rows, ${pct(p.noise)} of labels flipped by noise.`, { ...base, phase: "data" }, [ev("n", `${p.n} rows loaded`)]);
  push("shuffle", "Shuffle", "Shuffle first so every split is a random sample (here the data has no time order).", { ...base, phase: "shuffle" }, [ev("sh", `shuffled with seed ${p.seed + 1}`)]);
  push("split", p.kfold ? "Split off the test set, fold the rest" : "Split into train / validation / test",
    p.kfold ? `${nTest} rows are locked away as the test set. The other ${dev.length} rows are dealt into ${p.folds} folds.` : `${train.length} train, ${val.length} validation and ${nTest} test rows. The test set is locked away.`,
    { ...base, phase: "split" }, [ev("split", p.kfold ? `dev = ${dev.length}, test = ${nTest}, ${p.folds} folds` : `train = ${train.length}, val = ${val.length}, test = ${nTest}`)]);
  push("train", "Fit every candidate on the training rows", `Each candidate is k-nearest-neighbours with a different k. Training accuracy flatters small k: k = 1 scores ${pct(scores[0].train)} because each row is its own nearest neighbour.`,
    { ...base, phase: "train", revealed: ["train"] }, scores.map((s) => ev(`tr${s.k}`, `k=${s.k}: train acc ${pct(s.train)}`)));
  if (p.kfold) {
    for (let f = 0; f < p.folds; f++) {
      push(`fold-${f}`, `Fold ${f + 1} of ${p.folds} is the validation fold`, `Fit on the other ${p.folds - 1} folds, score on fold ${f + 1}. Every row gets one turn as validation.`,
        { ...base, phase: "fold", fold: f, revealed: ["train"] }, scores.map((s) => ev(`f${f}-${s.k}`, `k=${s.k}: fold ${f + 1} acc ${pct(s.folds[f])}`)));
    }
  }
  if (p.tuneOnTest) {
    push("validate", "Score every candidate on the TEST set (leak)", "Using the test set to choose k turns it into a second validation set. Its score is no longer an unbiased estimate.",
      { ...base, phase: "validate", revealed: ["train", "test"] }, [ev("leak", "Test set used for model selection", undefined, "warning")]);
  } else {
    push("validate", p.kfold ? "Average the fold scores" : "Score every candidate on validation", p.kfold ? "Each candidate's CV score is the mean of its fold accuracies; it uses every dev row for validation once." : "Validation rows were never used for fitting, so their accuracy is a fair way to compare candidates.",
      { ...base, phase: "validate", revealed: ["train", "val"] }, scores.map((s) => ev(`va${s.k}`, `k=${s.k}: ${p.kfold ? "CV" : "val"} acc ${pct(s.val)}`)));
  }
  push("select", `Pick k = ${winner}`, `k = ${winner} has the best ${p.tuneOnTest ? "test" : p.kfold ? "CV" : "validation"} accuracy. It is refit on all ${dev.length} non-test rows.`,
    { ...base, phase: "select", revealed: p.tuneOnTest ? ["train", "test"] : ["train", "val"], winner }, [ev("pick", `winner k = ${winner}`, `k^* = \\arg\\max_k \\text{acc}_{\\text{${p.tuneOnTest ? "test" : "val"}}}(k) = ${winner}`)]);
  const gap = w.test - freshAcc;
  push("test", "Unlock the test set once", `Reported test accuracy ${pct(w.test)}; on 600 fresh rows from the same source the same model scores ${pct(freshAcc)}.`,
    { ...base, phase: "test", revealed: ["train", "val", "test"], winner },
    [ev("test", `test acc ${pct(w.test)}, fresh ${pct(freshAcc)}`, `\\text{acc}_{\\text{test}} = ${(w.test * 100).toFixed(1)}\\% \\quad \\text{acc}_{\\text{fresh}} = ${(freshAcc * 100).toFixed(1)}\\%`, p.tuneOnTest && gap > 0.02 ? "warning" : undefined)]);
  return { steps, p, data, order, bucket, fold, train, val, test, dev, scores, winner, freshAcc, bestFresh, reported: w.test };
}
