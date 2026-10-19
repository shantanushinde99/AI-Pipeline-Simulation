import { clamp } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { forward, H, type Activation, type NetworkInput, type Weights } from "../neural-network/simulation";

/** Derivative of each hidden activation, written in terms of z. */
export const DERIV: Record<Activation, { fn: (z: number) => number; tex: string }> = {
  relu: { fn: (z) => (z > 0 ? 1 : 0), tex: "\\operatorname{relu}'(z) = [z > 0]" },
  sigmoid: { fn: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }, tex: "\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))" },
  tanh: { fn: (z) => 1 - Math.tanh(z) ** 2, tex: "\\tanh'(z) = 1 - \\tanh^2(z)" },
  linear: { fn: () => 1, tex: "\\operatorname{linear}'(z) = 1" },
};

const EPS = 1e-12;
/** Binary cross-entropy for one example. */
export const bce = (y: number, t: number) => -(t * Math.log(Math.max(y, EPS)) + (1 - t) * Math.log(Math.max(1 - y, EPS)));
export const lossOf = (input: NetworkInput, t: number) => bce(forward(input).y, t);

export type Grads = Weights;

/** Analytic gradients of the BCE loss for the 2-3-1 network (sigmoid output). */
export function backprop(input: NetworkInput, t: number) {
  const f = forward(input);
  const { x, weights: w, activation } = input;
  const dz2 = f.y - t; // ∂L/∂z_y for sigmoid + BCE
  const gw2 = f.h.map((h) => dz2 * h);
  const dh = w.w2.map((v) => dz2 * v);
  const dact = f.z1.map((z) => DERIV[activation].fn(z));
  const dz1 = dh.map((v, j) => v * dact[j]);
  const gw1 = dz1.map((d) => [d * x[0], d * x[1]]);
  const grads: Grads = { w1: gw1, b1: dz1, w2: gw2, b2: dz2 };
  return { f, loss: bce(f.y, t), dz2, dh, dact, dz1, grads };
}

export const applyUpdate = (w: Weights, g: Grads, lr: number): Weights => ({
  w1: w.w1.map((row, j) => row.map((v, i) => v - lr * g.w1[j][i])),
  b1: w.b1.map((v, j) => v - lr * g.b1[j]),
  w2: w.w2.map((v, j) => v - lr * g.w2[j]),
  b2: w.b2 - lr * g.b2,
});

export type BPPhase = "forward" | "loss" | "delta" | "grad-out" | "hidden" | "update" | "after";
export type BPState = { phase: BPPhase; hidden: number | null; shown: string[] };
export type BPInput = NetworkInput & { target: 0 | 1; lr: number };

/** Every frame comes from one real forward + backward pass and one update. `shown` lists params whose gradient is on screen. */
export function simulate(raw: BPInput) {
  const lr = clamp(raw.lr, 0, 10);
  const input: NetworkInput = { x: raw.x, weights: raw.weights, activation: raw.activation };
  const bp = backprop(input, raw.target);
  const next = applyUpdate(raw.weights, bp.grads, lr);
  const after = backprop({ ...input, weights: next }, raw.target);
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });
  const f4 = (v: number) => v.toFixed(4);
  const steps: SimulationStep<BPState>[] = [];
  const shown: string[] = [];
  const push = (s: Omit<SimulationStep<BPState>, "state">, state: Omit<BPState, "shown">) => steps.push({ ...s, state: { ...state, shown: [...shown] } });

  push({
    id: "forward", title: "Forward pass",
    description: `The input flows forward and the network predicts ŷ = ${f4(bp.f.y)} for an example whose true label is ${raw.target}.`,
    events: [ev("fwd", `ŷ = ${f4(bp.f.y)}`, `\\hat{y} = \\sigma(z_y) = \\sigma(${f4(bp.f.z2)}) = ${f4(bp.f.y)}`)],
  }, { phase: "forward", hidden: null });
  push({
    id: "loss", title: "Measure the loss",
    description: `Binary cross-entropy turns the miss into one number: L = ${f4(bp.loss)}. Lower is better.`,
    events: [ev("loss", `L = ${f4(bp.loss)}`, `L = -[t\\log\\hat{y} + (1-t)\\log(1-\\hat{y})] = ${f4(bp.loss)}`)],
  }, { phase: "loss", hidden: null });
  push({
    id: "delta", title: "Error signal at the output",
    description: `For sigmoid + cross-entropy the chain rule collapses to ∂L/∂z_y = ŷ − t = ${f4(bp.dz2)}. This is the signal that travels backward.`,
    events: [ev("dz2", `δ_y = ${f4(bp.dz2)}`, `\\delta_y = \\frac{\\partial L}{\\partial z_y} = \\hat{y} - t = ${f4(bp.dz2)}`)],
  }, { phase: "delta", hidden: null });
  shown.push("w2-0", "w2-1", "w2-2", "b2");
  push({
    id: "grad-out", title: "Gradients of the output weights",
    description: "Each output weight's gradient is δ_y times the activation that flowed along that weight in the forward pass.",
    events: bp.grads.w2.map((g, j) => ev(`gw2-${j}`, `∂L/∂w_h${j + 1}→y = ${f4(g)}`, `\\frac{\\partial L}{\\partial w_{h_${j + 1}\\to y}} = \\delta_y\\, h_${j + 1} = ${f4(bp.dz2)} \\cdot ${f4(bp.f.h[j])} = ${f4(g)}`)),
  }, { phase: "grad-out", hidden: null });
  for (let j = 0; j < H; j++) {
    shown.push(`w1-${j}-0`, `w1-${j}-1`, `b1-${j}`);
    const dead = bp.dact[j] === 0;
    push({
      id: `hidden-${j}`, title: `Back through h${j + 1}`,
      description: `δ_y flows back along w = ${raw.weights.w2[j]} to h${j + 1}, is multiplied by the activation's slope (${f4(bp.dact[j])}), then split onto h${j + 1}'s incoming weights.`,
      events: [
        ev(`dh${j}`, `∂L/∂h${j + 1} = ${f4(bp.dh[j])}`, `\\frac{\\partial L}{\\partial h_${j + 1}} = \\delta_y\\, w_{h_${j + 1}\\to y} = ${f4(bp.dh[j])}`),
        ev(`dz${j}`, `δ_h${j + 1} = ${f4(bp.dz1[j])}`, `\\delta_{h_${j + 1}} = \\frac{\\partial L}{\\partial h_${j + 1}}\\, f'(z_{h_${j + 1}}) = ${f4(bp.dh[j])} \\cdot ${f4(bp.dact[j])} = ${f4(bp.dz1[j])}`),
        ...(dead ? [ev(`dead${j}`, `h${j + 1} is inactive (ReLU, z ≤ 0): its incoming weights get zero gradient and will not learn from this example.`, undefined, "warning")] : []),
      ],
    }, { phase: "hidden", hidden: j });
  }
  push({
    id: "update", title: "Update every weight",
    description: `Each parameter steps against its gradient: w ← w − η·∂L/∂w with η = ${lr}.`,
    events: [ev("upd", `13 parameters updated with η = ${lr}`, `w \\leftarrow w - ${lr}\\,\\frac{\\partial L}{\\partial w}`)],
  }, { phase: "update", hidden: null });
  const better = after.loss < bp.loss;
  push({
    id: "after", title: "Loss after one step",
    description: `Re-running the forward pass with the new weights gives L = ${f4(after.loss)} (was ${f4(bp.loss)}).`,
    events: [ev("after", `L: ${f4(bp.loss)} → ${f4(after.loss)}`, `L_{\\text{before}} = ${f4(bp.loss)} \\;\\to\\; L_{\\text{after}} = ${f4(after.loss)}`, better ? undefined : "warning"),
      ...(better ? [] : [ev("overshoot", "Loss went up: the learning rate is too large and the step overshot.", undefined, "warning")])],
  }, { phase: "after", hidden: null });
  return { steps, bp, next, after, lr };
}

/** Central finite difference of the loss w.r.t. every parameter, used to verify backprop. */
export function numericGrads(input: NetworkInput, t: number, h = 1e-6): Grads {
  const d = (mut: (w: Weights, e: number) => void) => {
    const p = structuredClone(input.weights); mut(p, h);
    const m = structuredClone(input.weights); mut(m, -h);
    return (lossOf({ ...input, weights: p }, t) - lossOf({ ...input, weights: m }, t)) / (2 * h);
  };
  return {
    w1: input.weights.w1.map((row, j) => row.map((_, i) => d((w, e) => { w.w1[j][i] += e; }))),
    b1: input.weights.b1.map((_, j) => d((w, e) => { w.b1[j] += e; })),
    w2: input.weights.w2.map((_, j) => d((w, e) => { w.w2[j] += e; })),
    b2: d((w, e) => { w.b2 += e; }),
  };
}
