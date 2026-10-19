import { clamp, mean } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Mode = "sweep" | "seeds";
export type ETConfig = { mode: Mode; runs: number; valSize: number; sweepSeed: number };
export const DEFAULT_ET: ETConfig = { mode: "sweep", runs: 12, valSize: 200, sweepSeed: 2 };
export const VAL_SIZES = [50, 100, 200, 500, 2000, 10000];
export const TEST_SIZE = 10000;
export const SEED_SD = 0.008; // spread caused by training randomness alone

/** The (hidden) true accuracy of a configuration: a smooth made-up surface peaking at lr ≈ 0.0063, depth 7. */
export const trueAccuracy = (lr: number, depth: number) => clamp(0.9 - 0.05 * (Math.log10(lr) + 2.2) ** 2 - 0.003 * (depth - 7) ** 2, 0.5, 0.99);

/** Accuracy measured on n examples whose true rate is p: the count of correct answers is rounded, so values are k/n. */
const measure = (p: number, n: number, z: number) => Math.round(clamp(p + z * Math.sqrt((p * (1 - p)) / n), 0, 1) * n) / n;
export const standardError = (p: number, n: number) => Math.sqrt((p * (1 - p)) / n);

export type Run = {
  id: number; lr: number; depth: number; seed: number;
  trueAcc: number; // what the config would score on infinite data, averaged over seeds
  modelAcc: number; // what this particular trained model really scores (true + seed effect)
  trainAcc: number; valAcc: number; testAcc: number;
};

export function makeRun(id: number, lr: number, depth: number, seed: number, valSize: number): Run {
  const r = rng(seed * 7919 + 13);
  const trueAcc = trueAccuracy(lr, depth);
  const modelAcc = clamp(trueAcc + SEED_SD * r.gaussian(), 0.5, 0.995);
  const valAcc = measure(modelAcc, valSize, r.gaussian());
  const testAcc = measure(modelAcc, TEST_SIZE, r.gaussian());
  const trainAcc = Math.min(0.999, modelAcc + 0.006 * depth);
  return { id, lr, depth, seed, trueAcc, modelAcc, trainAcc, valAcc, testAcc };
}

const sig3 = (v: number) => Number(v.toPrecision(2));

/** Random search: log-uniform learning rate, uniform integer depth. */
export function sampleSweep(cfg: ETConfig): Run[] {
  const n = clamp(Math.round(cfg.runs), 2, 20);
  const valSize = clamp(Math.round(cfg.valSize), 10, 100000);
  const r = rng(cfg.sweepSeed * 1013 + 1);
  return Array.from({ length: n }, (_, i) => {
    const lr = sig3(10 ** (-4 + 3.5 * r.next()));
    const depth = 2 + Math.floor(r.next() * 11);
    return makeRun(i + 1, lr, depth, 100 + i + cfg.sweepSeed * 50, valSize);
  });
}

export const bestBy = (runs: Run[], key: "valAcc" | "testAcc") =>
  runs.reduce<Run | undefined>((b, r) => (!b || r[key] > b[key] ? r : b), undefined);

export type ETPhase = "define" | "run" | "best" | "compare" | "test";
export type ETState = {
  phase: ETPhase;
  mode: Mode;
  runs: Run[]; // runs finished so far
  current?: Run;
  best?: Run;
  previousBest?: Run;
  compare?: [Run, Run];
  valSize: number;
};

export type ETResult = { steps: SimulationStep<ETState>[]; all: Run[]; best: Run; bestByTest: Run; seedStats?: { mean: number; sd: number; min: number; max: number } };

export function simulateTracking(cfg: ETConfig, compareIds?: [number, number]): ETResult {
  const valSize = clamp(Math.round(cfg.valSize), 10, 100000);
  const sweep = sampleSweep(cfg);
  let all = sweep;
  let seedStats: ETResult["seedStats"];
  if (cfg.mode === "seeds") {
    // Same config as the sweep's winner, retrained with different seeds.
    const w = bestBy(sweep, "valAcc")!;
    all = sweep.map((_, i) => makeRun(i + 1, w.lr, w.depth, 5000 + i, valSize));
    const vals = all.map((r) => r.valAcc);
    const m = mean(vals);
    seedStats = { mean: m, sd: Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, vals.length - 1)), min: Math.min(...vals), max: Math.max(...vals) };
  }
  const best = bestBy(all, "valAcc")!;
  const bestByTest = bestBy(all, "testAcc")!;
  const ranked = [...all].sort((a, b) => b.valAcc - a.valAcc);
  const pick = (id: number | undefined, fallback: Run) => all.find((r) => r.id === id) ?? fallback;
  const pair: [Run, Run] = [pick(compareIds?.[0], ranked[0]), pick(compareIds?.[1], ranked[1] ?? ranked[0])];

  const steps: SimulationStep<ETState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "tracking", message, severity, tex });
  const base = { mode: cfg.mode, valSize };
  const seeds = cfg.mode === "seeds";

  steps.push({
    id: "define", title: seeds ? "Fix one config, vary only the seed" : "Define the search space",
    description: seeds ? `Every run uses lr=${best.lr}, depth=${best.depth}. Only the random seed changes, so any spread is noise.`
      : `${all.length} runs: learning rate log-uniform in [1e-4, 0.3], depth uniform in 2–12. Each run is scored on a ${valSize}-example validation set.`,
    state: { ...base, phase: "define", runs: [] }, events: [ev("def", seeds ? `config fixed: lr=${best.lr}, depth=${best.depth}` : `sweep: ${all.length} random configs, val n=${valSize}`)],
  });
  let leader: Run | undefined;
  all.forEach((run, i) => {
    const prev = leader;
    if (!leader || run.valAcc > leader.valAcc) leader = run;
    const newBest = leader === run;
    steps.push({
      id: `run-${run.id}`, title: `Run ${run.id}: lr ${run.lr}, depth ${run.depth}`,
      description: `Run ${run.id} (seed ${run.seed}) scores ${run.valAcc.toFixed(3)} on validation (${Math.round(run.valAcc * valSize)}/${valSize} correct).${newBest ? " New best so far." : ""}`,
      state: { ...base, phase: "run", runs: all.slice(0, i + 1), current: run, best: leader, previousBest: prev },
      events: [ev(`r${run.id}`, `run ${run.id}: train ${run.trainAcc.toFixed(3)}, val ${run.valAcc.toFixed(3)}`, undefined, `\\text{val acc} = \\frac{${Math.round(run.valAcc * valSize)}}{${valSize}} = ${run.valAcc.toFixed(3)}`),
        ...(newBest ? [ev(`b${run.id}`, `new best: run ${run.id}${prev ? ` (was run ${prev.id}, ${prev.valAcc.toFixed(3)})` : ""}`)] : [])],
    });
  });
  steps.push({
    id: "best", title: `Best by validation: run ${best.id}`,
    description: seeds ? `Best seed reaches ${best.valAcc.toFixed(3)}, worst ${seedStats!.min.toFixed(3)}: a ${((seedStats!.max - seedStats!.min) * 100).toFixed(1)}-point spread from the same config.`
      : `Run ${best.id} (lr ${best.lr}, depth ${best.depth}) has the highest validation accuracy, ${best.valAcc.toFixed(3)}.`,
    state: { ...base, phase: "best", runs: all, best }, events: [ev("best", `argmax val acc → run ${best.id} (${best.valAcc.toFixed(3)})`)],
  });
  steps.push({
    id: "compare", title: `Compare run ${pair[0].id} vs run ${pair[1].id}`,
    description: `Side by side: the gap in validation accuracy is ${((pair[0].valAcc - pair[1].valAcc) * 100).toFixed(1)} pts, while one standard error at n=${valSize} is about ${(standardError(pair[0].valAcc, valSize) * 100).toFixed(1)} pts.`,
    state: { ...base, phase: "compare", runs: all, best, compare: pair },
    events: [ev("cmp", `Δval = ${(pair[0].valAcc - pair[1].valAcc).toFixed(3)}, SE ≈ ${standardError(pair[0].valAcc, valSize).toFixed(3)}`,
      Math.abs(pair[0].valAcc - pair[1].valAcc) < 2 * standardError(pair[0].valAcc, valSize) ? "warning" : undefined)],
  });
  const drop = best.valAcc - best.testAcc;
  steps.push({
    id: "test", title: "Held-out test: was the winner just lucky?",
    description: `On ${TEST_SIZE.toLocaleString("en-US")} fresh examples, run ${best.id} scores ${best.testAcc.toFixed(3)} (validation said ${best.valAcc.toFixed(3)}).${bestByTest.id !== best.id ? ` Run ${bestByTest.id} is actually better on test (${bestByTest.testAcc.toFixed(3)}).` : ""}`,
    state: { ...base, phase: "test", runs: all, best },
    events: [ev("t", `run ${best.id}: val ${best.valAcc.toFixed(3)} → test ${best.testAcc.toFixed(3)} (optimism ${drop.toFixed(3)})`, drop > 0.01 ? "warning" : undefined),
      ...(bestByTest.id !== best.id ? [ev("t2", `best by test is run ${bestByTest.id} (${bestByTest.testAcc.toFixed(3)}), not the validation winner`, "warning")] : [])],
  });
  return { steps, all, best, bestByTest, seedStats };
}
