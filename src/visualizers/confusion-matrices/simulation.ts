import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type CmMode = "binary" | "multi";
export type CmConfig = {
  mode: CmMode; n: number; quality: number; imbalanced: boolean; majorityFault: boolean;
  costMiss: number; costOther: number; seed: number;
};
export const DEFAULT_CM: CmConfig = { mode: "binary", n: 20, quality: 0.8, imbalanced: false, majorityFault: false, costMiss: 10, costOther: 1, seed: 3 };

/** Class 0 is always the critical class ("urgent"): missing it costs `costMiss`. */
export const CLASSES: Record<CmMode, string[]> = { binary: ["urgent", "not urgent"], multi: ["urgent", "routine", "self-care"] };
const PRIORS: Record<CmMode, { balanced: number[]; imbalanced: number[] }> = {
  binary: { balanced: [0.5, 0.5], imbalanced: [0.15, 0.85] },
  multi: { balanced: [1 / 3, 1 / 3, 1 / 3], imbalanced: [0.12, 0.28, 0.6] },
};

export type Example = { id: number; actual: number; predicted: number };

/** Exact class counts from the priors (largest remainder), shuffled; each prediction is right with probability `quality`. */
export function makeExamples(cfg: CmConfig): Example[] {
  const n = Math.round(clamp(cfg.n, 6, 36));
  const priors = PRIORS[cfg.mode][cfg.imbalanced ? "imbalanced" : "balanced"];
  const K = priors.length;
  const raw = priors.map((p) => p * n);
  const counts = raw.map(Math.floor);
  const order = raw.map((v, i) => [v - Math.floor(v), i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (let j = 0; counts.reduce((s, v) => s + v, 0) < n; j++) counts[order[j % K][1]]++;
  const actual = counts.flatMap((c, k) => Array<number>(c).fill(k));
  const r = rng(cfg.seed);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [actual[i], actual[j]] = [actual[j], actual[i]];
  }
  const majority = counts.indexOf(Math.max(...counts));
  return actual.map((a, id) => {
    const u = r.next(), w = r.next();
    let p = a;
    if (cfg.majorityFault) p = majority;
    else if (u >= cfg.quality) {
      // Wrong: pick one of the other classes; in 3 classes, neighbours are confused twice as often.
      const others = Array.from({ length: K }, (_, k) => k).filter((k) => k !== a);
      const weights = others.map((k) => (Math.abs(k - a) === 1 ? 2 : 1));
      let x = w * weights.reduce((s, v) => s + v, 0);
      p = others.find((_, i) => (x -= weights[i]) < 0) ?? others[0];
    }
    return { id: id + 1, actual: a, predicted: p };
  });
}

/** matrix[actual][predicted] */
export function confusionMatrix(ex: readonly Example[], K: number) {
  const m = Array.from({ length: K }, () => new Array<number>(K).fill(0));
  for (const e of ex) m[e.actual][e.predicted]++;
  return m;
}

const div = (a: number, b: number) => (b === 0 ? Number.NaN : a / b);

export function perClass(m: number[][]) {
  const K = m.length;
  return Array.from({ length: K }, (_, c) => {
    const col = m.reduce((s, row) => s + row[c], 0);
    const row = m[c].reduce((s, v) => s + v, 0);
    const precision = div(m[c][c], col), recall = div(m[c][c], row);
    return { tp: m[c][c], col, row, precision, recall, f1: div(2 * m[c][c], col + row) };
  });
}

export const rowNormalize = (m: number[][]) => m.map((row) => { const s = row.reduce((a, b) => a + b, 0); return row.map((v) => div(v, s)); });

export const costMatrix = (K: number, costMiss: number, costOther: number) =>
  Array.from({ length: K }, (_, a) => Array.from({ length: K }, (_, p) => (a === p ? 0 : a === 0 ? costMiss : costOther)));

export function totalCost(m: number[][], c: number[][]) {
  return m.reduce((s, row, a) => s + row.reduce((t, v, p) => t + v * c[a][p], 0), 0);
}

export type CmPhase = "list" | "first" | "rest" | "diagonal" | "precision" | "recall" | "normalize" | "cost";
export type CmState = { phase: CmPhase; landedBefore: number; landedAfter: number };

export const FIRST = 3;
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "undefined");

export function simulate(cfg: CmConfig) {
  const names = CLASSES[cfg.mode];
  const K = names.length;
  const ex = makeExamples(cfg);
  const n = ex.length;
  const m = confusionMatrix(ex, K);
  const pc = perClass(m);
  const correct = m.reduce((s, row, i) => s + row[i], 0);
  const accuracy = correct / n;
  const macroF1 = pc.reduce((s, c) => s + (Number.isFinite(c.f1) ? c.f1 : 0), 0) / K;
  const norm = rowNormalize(m);
  const cost = costMatrix(K, cfg.costMiss, cfg.costOther);
  const total = totalCost(m, cost);
  const missedCritical = m[0].reduce((s, v, p) => s + (p ? v : 0), 0);
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "metric", message, tex, severity });
  const first = ex.slice(0, FIRST);
  const s = (phase: CmPhase, title: string, description: string, events: SimulationEvent[], landedBefore = n, landedAfter = n): SimulationStep<CmState> =>
    ({ id: phase, title, description, state: { phase, landedBefore, landedAfter }, events });
  const t = (name: string) => `\\text{${name}}`;

  const steps: SimulationStep<CmState>[] = [
    s("list", "A labelled evaluation set", `${n} patient messages, each with a true class and the model's predicted class. The matrix starts empty.`,
      [ev("l", names.map((nm, k) => `${nm}: ${m[k].reduce((a, b) => a + b, 0)}`).join(", "))], 0, 0),
    s("first", "Drop the first examples into cells", `Each example goes to row = its actual class, column = its predicted class. ${first.map((e) => `#${e.id}: ${names[e.actual]} → ${names[e.predicted]}`).join("; ")}.`,
      first.map((e) => ev(`f${e.id}`, `#${e.id} actual ${names[e.actual]}, predicted ${names[e.predicted]} → cell [${names[e.actual]}, ${names[e.predicted]}]`, undefined, e.actual === e.predicted ? undefined : "warning")), 0, FIRST),
    s("rest", "Fill in the rest", `The remaining ${n - FIRST} examples land; each cell counter ticks up as they arrive. ${correct} of ${n} land on the diagonal.`,
      [ev("r", `matrix = [${m.map((r) => `[${r.join(", ")}]`).join(", ")}]`)], FIRST, n),
    s("diagonal", "The diagonal is what the model got right", `Accuracy = sum of the diagonal / total = ${correct} / ${n} = ${f3(accuracy)}. Everything off the diagonal is an error, and its position says which mistake.`,
      [ev("a", `accuracy = ${f3(accuracy)}`, `${t("accuracy")} = \\tfrac{${m.map((r, i) => r[i]).join(" + ")}}{${n}} = ${f3(accuracy)}`)]),
    s("precision", "Precision reads down a column", `For each predicted class: of everything the model labelled it, how much really was? ${names.map((nm, k) => `${nm} ${f3(pc[k].precision)}`).join(", ")}.`,
      pc.map((c, k) => ev(`p${k}`, `precision(${names[k]}) = ${c.tp}/${c.col} = ${f3(c.precision)}`, `P_{${t(names[k])}} = \\tfrac{${c.tp}}{${c.col}} = ${f3(c.precision)}`, Number.isFinite(c.precision) ? undefined : "warning"))),
    s("recall", "Recall reads across a row", `For each actual class: how much of it did the model find? ${names.map((nm, k) => `${nm} ${f3(pc[k].recall)}`).join(", ")}.`,
      pc.map((c, k) => ev(`r${k}`, `recall(${names[k]}) = ${c.tp}/${c.row} = ${f3(c.recall)}`, `R_{${t(names[k])}} = \\tfrac{${c.tp}}{${c.row}} = ${f3(c.recall)}`, k === 0 && c.recall < 0.7 ? "warning" : undefined))),
    s("normalize", "Row-normalize to compare classes fairly", `Dividing each row by its total turns counts into "what fraction of class X went where". The diagonal now shows recall per class: ${names.map((nm, k) => `${nm} ${f3(norm[k][k])}`).join(", ")}.`,
      [ev("n", "each row now sums to 1", `\\tilde{M}_{ij} = \\tfrac{M_{ij}}{\\sum_j M_{ij}}`)]),
    s("cost", "Not all mistakes cost the same", `Missing an urgent message costs ${cfg.costMiss}; any other error costs ${cfg.costOther}. Total cost = ${total} (${missedCritical} missed urgent × ${cfg.costMiss} + ${n - correct - missedCritical} other errors × ${cfg.costOther}).`,
      [ev("c", `total cost = ${total}`, `\\text{cost} = \\sum_{i,j} M_{ij}\\,C_{ij} = ${total}`, missedCritical > 0 && cfg.costMiss > cfg.costOther ? "warning" : undefined)]),
  ];
  return { names, K, ex, n, m, pc, correct, accuracy, macroF1, norm, cost, total, missedCritical, steps };
}
export type CmSim = ReturnType<typeof simulate>;
