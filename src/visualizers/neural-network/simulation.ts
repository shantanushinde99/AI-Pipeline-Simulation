import { sigmoid } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Activation = "relu" | "sigmoid" | "tanh" | "linear";
/** `name` and `tex` are LaTeX. */
export const ACTIVATIONS: Record<Activation, { fn: (z: number) => number; name: string; tex: string }> = {
  relu: { fn: (z) => Math.max(0, z), name: "\\operatorname{relu}", tex: "\\operatorname{relu}(z) = \\max(0, z)" },
  sigmoid: { fn: sigmoid, name: "\\sigma", tex: "\\sigma(z) = \\frac{1}{1 + e^{-z}}" },
  tanh: { fn: Math.tanh, name: "\\tanh", tex: "\\tanh(z) = \\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}" },
  linear: { fn: (z) => z, name: "\\operatorname{linear}", tex: "\\operatorname{linear}(z) = z" },
};

/** 2 inputs → 3 hidden → 1 sigmoid output. */
export type Weights = {
  w1: number[][]; // [hidden j][input i]
  b1: number[];
  w2: number[]; // [hidden j]
  b2: number;
};
export type NetworkInput = { x: [number, number]; weights: Weights; activation: Activation };

export const H = 3;
export const neuronIds = { input: ["x1", "x2"], hidden: ["h1", "h2", "h3"], output: "y" } as const;

export function forward({ x, weights: w, activation }: NetworkInput) {
  const act = ACTIVATIONS[activation].fn;
  const z1 = w.w1.map((row, j) => row[0] * x[0] + row[1] * x[1] + w.b1[j]);
  const h = z1.map(act);
  const z2 = w.w2.reduce((s, v, j) => s + v * h[j], w.b2);
  return { z1, h, z2, y: sigmoid(z2) };
}

export type NeuronValue = { z?: number; a?: number };
export type NNState = {
  active: string | null;
  phase: "input" | "sum" | "activate" | "predict";
  values: Record<string, NeuronValue>;
  activeEdges: string[];
};

export const edgeId = (from: string, to: string) => `${from}->${to}`;

/** Every frame is derived from one real forward pass, revealed neuron by neuron. */
export function simulate(input: NetworkInput): SimulationStep<NNState>[] {
  const { x, weights: w, activation } = input;
  const r = forward(input);
  const values: Record<string, NeuronValue> = { x1: { a: x[0] }, x2: { a: x[1] } };
  const steps: SimulationStep<NNState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "compute", message, severity, tex });
  const p = (v: number, d?: number) => { const s = d === undefined ? String(v) : v.toFixed(d); return v < 0 ? `(${s})` : s; };
  const h = (j: number) => `h_{${j + 1}}`;
  const push = (s: Omit<SimulationStep<NNState>, "state"> & { state: Omit<NNState, "values"> }) =>
    steps.push({ ...s, state: { ...s.state, values: structuredClone(values) } });

  push({
    id: "inputs",
    title: "Load inputs",
    description: `The input layer holds the raw features x1 = ${x[0]} and x2 = ${x[1]}. Input neurons do no computation.`,
    state: { active: null, phase: "input", activeEdges: [] },
    events: [ev("in", `Inputs loaded: x = [${x.join(", ")}]`)],
  });

  for (let j = 0; j < H; j++) {
    const id = neuronIds.hidden[j];
    values[id] = { z: r.z1[j] };
    push({
      id: `${id}-sum`,
      title: `${id}: weighted sum + bias`,
      description: `${id} multiplies each input by its connection weight, adds them, then adds its bias.`,
      state: { active: id, phase: "sum", activeEdges: neuronIds.input.map((i) => edgeId(i, id)) },
      events: [ev(`${id}-z`, `z_${id} = ${w.w1[j][0]}·${x[0]} + ${w.w1[j][1]}·${x[1]} + ${w.b1[j]} = ${r.z1[j].toFixed(4)}`, undefined,
        `z_{${h(j)}} = ${p(w.w1[j][0])} \\cdot ${p(x[0])} + ${p(w.w1[j][1])} \\cdot ${p(x[1])} + ${p(w.b1[j])} = ${r.z1[j].toFixed(4)}`)],
    });
    values[id] = { z: r.z1[j], a: r.h[j] };
    const dead = activation === "relu" && r.z1[j] <= 0;
    push({
      id: `${id}-act`,
      title: `${id}: activation`,
      description: `${id} passes z through ${activation} to produce its output (activation).`,
      state: { active: id, phase: "activate", activeEdges: [] },
      events: [
        ev(`${id}-a`, `a_${id} = ${activation}(${r.z1[j].toFixed(4)}) = ${r.h[j].toFixed(4)}`, undefined, `a_{${h(j)}} = ${ACTIVATIONS[activation].name}(${r.z1[j].toFixed(4)}) = ${r.h[j].toFixed(4)}`),
        ...(dead ? [ev(`${id}-dead`, `${id} outputs 0 because z ≤ 0 under ReLU, so its outgoing weight has no effect on this input.`, "warning")] : []),
      ],
    });
  }

  values.y = { z: r.z2 };
  push({
    id: "y-sum",
    title: "Output: weighted sum + bias",
    description: "The output neuron combines the three hidden activations with its own weights and bias.",
    state: { active: "y", phase: "sum", activeEdges: neuronIds.hidden.map((h) => edgeId(h, "y")) },
    events: [ev("y-z", `z_y = ${w.w2.map((v, j) => `${v}·${r.h[j].toFixed(4)}`).join(" + ")} + ${w.b2} = ${r.z2.toFixed(4)}`, undefined,
      `z_y = ${w.w2.map((v, j) => `${p(v)} \\cdot ${p(r.h[j], 4)}`).join(" + ")} + ${p(w.b2)} = ${r.z2.toFixed(4)}`)],
  });
  values.y = { z: r.z2, a: r.y };
  push({
    id: "y-act",
    title: "Output: sigmoid",
    description: "A sigmoid squashes z into (0, 1) so the output can be read as a probability.",
    state: { active: "y", phase: "activate", activeEdges: [] },
    events: [ev("y-a", `ŷ = σ(${r.z2.toFixed(4)}) = ${r.y.toFixed(4)}`, undefined, `\\hat{y} = \\sigma(${r.z2.toFixed(4)}) = ${r.y.toFixed(4)}`)],
  });
  push({
    id: "predict",
    title: "Prediction",
    description: `ŷ = ${r.y.toFixed(4)}. With a 0.5 threshold the predicted class is ${r.y >= 0.5 ? 1 : 0}.`,
    state: { active: null, phase: "predict", activeEdges: [] },
    events: [ev("cls", `class = ŷ ≥ 0.5 ? 1 : 0 → ${r.y >= 0.5 ? 1 : 0}`, undefined, `\\text{class} = [\\hat{y} \\ge 0.5] = ${r.y >= 0.5 ? 1 : 0}`)],
  });
  return steps;
}

export type Sample = { x: [number, number]; target: 0 | 1; label: string };
export type Dataset = { id: string; name: string; features: [string, string]; samples: Sample[]; preset: Weights; presetActivation: Activation };

export const DATASETS: Dataset[] = [
  {
    id: "xor",
    name: "XOR (not linearly separable)",
    features: ["x1", "x2"],
    samples: [
      { x: [0, 0], target: 0, label: "(0, 0)" },
      { x: [0, 1], target: 1, label: "(0, 1)" },
      { x: [1, 0], target: 1, label: "(1, 0)" },
      { x: [1, 1], target: 0, label: "(1, 1)" },
    ],
    // h1 = relu(x1 − x2), h2 = relu(x2 − x1): either fires only when exactly one input is on.
    preset: { w1: [[1, -1], [-1, 1], [0.5, 0.5]], b1: [0, 0, 0], w2: [6, 6, 0], b2: -3 },
    presetActivation: "relu",
  },
  {
    id: "exam",
    name: "Exam pass (hours studied, hours slept)",
    features: ["study h", "sleep h"],
    samples: [
      { x: [8, 7], target: 1, label: "8h / 7h" },
      { x: [2, 8], target: 0, label: "2h / 8h" },
      { x: [6, 4], target: 1, label: "6h / 4h" },
      { x: [3, 3], target: 0, label: "3h / 3h" },
      { x: [9, 2], target: 1, label: "9h / 2h" },
      { x: [4, 6], target: 0, label: "4h / 6h" },
    ],
    // margin m = 0.6·study + 0.4·sleep − 5; h1 = relu(m), h2 = relu(−m) → ŷ = σ(3m).
    preset: { w1: [[0.6, 0.4], [-0.6, -0.4], [0, 0]], b1: [-5, 5, 0], w2: [3, -3, 0], b2: 0 },
    presetActivation: "relu",
  },
];

export function randomWeights(seed: number): Weights {
  let s = seed;
  const r = () => {
    s = (s * 16807) % 2147483647;
    return Math.round(((s / 2147483647) * 4 - 2) * 10) / 10;
  };
  return { w1: [[r(), r()], [r(), r()], [r(), r()]], b1: [r(), r(), r()], w2: [r(), r(), r()], b2: r() };
}

export function evaluateDataset(ds: Dataset, weights: Weights, activation: Activation) {
  const rows = ds.samples.map((s) => {
    const y = forward({ x: s.x, weights, activation }).y;
    return { ...s, y, pred: (y >= 0.5 ? 1 : 0) as 0 | 1 };
  });
  const eps = 1e-12;
  const logLoss = -rows.reduce((t, r) => t + (r.target ? Math.log(Math.max(r.y, eps)) : Math.log(Math.max(1 - r.y, eps))), 0) / rows.length;
  const accuracy = rows.filter((r) => r.pred === r.target).length / rows.length;
  return { rows, logLoss, accuracy };
}
