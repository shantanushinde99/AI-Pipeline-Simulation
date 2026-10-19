import { cosine, mean } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { add, attend, matVec, type Mat, type Vec } from "../self-attention/attention";

export const TOKENS = ["the", "cat", "sat"];
export const D = 4;
export const HID = 8;
export const EPS = 1e-5;

/** Hand-placed teaching-scale input vectors (3 tokens × d = 4). */
export const X0: Mat = [
  [1.0, -0.5, 0.3, 0.8],
  [-0.4, 1.2, 0.6, -0.2],
  [0.5, 0.2, -1.0, 0.9],
];

/** Hand-set (untrained) weights. Both blocks share them when stacked, to keep the page small. */
export const W = {
  Q: [[0.5, 0, 0.25, 0], [0, 0.5, 0, 0.25], [0.25, 0, 0.5, 0], [0, 0.25, 0, 0.5]],
  K: [[0.5, 0.25, 0, 0], [0, 0.5, 0.25, 0], [0, 0, 0.5, 0.25], [0.25, 0, 0, 0.5]],
  V: [[0.8, 0, 0, 0.2], [0, 0.8, 0.2, 0], [0.2, 0, 0.8, 0], [0, 0.2, 0, 0.8]],
  O: [[0.6, 0.2, 0, 0], [0, 0.6, 0.2, 0], [0, 0, 0.6, 0.2], [0.2, 0, 0, 0.6]],
  W1: [
    [0.5, -0.3, 0.2, 0.1], [-0.2, 0.6, 0.1, -0.3], [0.3, 0.2, -0.5, 0.4], [-0.4, -0.1, 0.3, 0.5],
    [0.6, 0.4, 0, -0.2], [0, -0.5, 0.6, 0.2], [-0.3, 0.3, 0.4, -0.6], [0.2, 0, -0.3, 0.7],
  ],
  b1: [0.1, 0, -0.1, 0.05, 0, 0.1, -0.05, 0],
  W2: [
    [0.4, -0.2, 0.3, 0, 0.5, -0.1, 0.2, 0.1],
    [-0.1, 0.5, 0, 0.3, 0.2, -0.4, 0.1, 0.2],
    [0.2, 0.1, -0.4, 0.3, 0, 0.5, 0.3, -0.2],
    [0, -0.3, 0.2, 0.4, -0.2, 0.1, -0.3, 0.5],
  ],
  b2: [0, 0.05, -0.05, 0],
};

export function layerNorm(x: Vec) {
  const mu = mean(x);
  const variance = mean(x.map((v) => (v - mu) ** 2));
  const sd = Math.sqrt(variance + EPS);
  return { mu, variance, sd, y: x.map((v) => (v - mu) / sd) };
}
/** GELU, tanh approximation (as in GPT-2). */
export const gelu = (z: number) => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)));

export type Phase = "input" | "ln1" | "attn" | "add1" | "ln2" | "up" | "gelu" | "down" | "add2" | "output";
export type BlockState = {
  phase: Phase;
  block: number; // 0-based; −1 for input/output
  before: Mat; // vectors entering this stage
  after: Mat; // vectors leaving it
  branch?: Mat; // sub-layer output being added (add phases)
  stats?: { mu: number; variance: number }[]; // LayerNorm statistics per token
  attn?: Mat; // attention weights (attn phase)
};
export type BlockParams = { blocks: 1 | 2; residual: boolean };
export const DEFAULT_PARAMS: BlockParams = { blocks: 1, residual: true };

/** Runs the blocks and records every intermediate stage. */
export function run(params: BlockParams) {
  const blocks = params.blocks === 2 ? 2 : 1;
  let x: Mat = X0.map((r) => [...r]);
  const input = x.map((r) => [...r]);
  const stages: BlockState[] = [{ phase: "input", block: -1, before: input, after: input }];
  for (let b = 0; b < blocks; b++) {
    const ln1 = x.map(layerNorm);
    const h = ln1.map((r) => r.y);
    stages.push({ phase: "ln1", block: b, before: x, after: h, stats: ln1.map(({ mu, variance }) => ({ mu, variance })) });
    const att = attend(h.map((v) => matVec(W.Q, v)), h.map((v) => matVec(W.K, v)), h.map((v) => matVec(W.V, v)));
    const a = att.out.map((v) => matVec(W.O, v));
    stages.push({ phase: "attn", block: b, before: h, after: a, attn: att.weights });
    const x1 = params.residual ? x.map((v, i) => add(v, a[i])) : a;
    stages.push({ phase: "add1", block: b, before: x, branch: a, after: x1 });
    const ln2 = x1.map(layerNorm);
    const h2 = ln2.map((r) => r.y);
    stages.push({ phase: "ln2", block: b, before: x1, after: h2, stats: ln2.map(({ mu, variance }) => ({ mu, variance })) });
    const up = h2.map((v) => add(matVec(W.W1, v), W.b1));
    stages.push({ phase: "up", block: b, before: h2, after: up });
    const act = up.map((v) => v.map(gelu));
    stages.push({ phase: "gelu", block: b, before: up, after: act });
    const down = act.map((v) => add(matVec(W.W2, v), W.b2));
    stages.push({ phase: "down", block: b, before: act, after: down });
    const x2 = params.residual ? x1.map((v, i) => add(v, down[i])) : down;
    stages.push({ phase: "add2", block: b, before: x1, branch: down, after: x2 });
    x = x2;
  }
  stages.push({ phase: "output", block: -1, before: input, after: x });
  return { stages, input, output: x, health: health(input, x) };
}

/** How much of the input survives, and how distinct the tokens stay. */
export function health(input: Mat, out: Mat) {
  const keep = out.map((v, i) => cosine(v, input[i]));
  const pairs = [[0, 1], [0, 2], [1, 2]].map(([i, j]) => cosine(out[i], out[j]));
  const inPairs = [[0, 1], [0, 2], [1, 2]].map(([i, j]) => cosine(input[i], input[j]));
  return { keep, meanKeep: mean(keep), tokenSimilarity: mean(pairs), inputSimilarity: mean(inPairs), outNorm: mean(out.map((v) => Math.hypot(...v))) };
}

const TITLES: Record<Phase, string> = {
  input: "Input vectors", ln1: "LayerNorm 1", attn: "Self-attention", add1: "Residual add 1", ln2: "LayerNorm 2",
  up: "MLP: expand to 8", gelu: "GELU", down: "MLP: project back to 4", add2: "Residual add 2", output: "Block output",
};
const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });
const v2 = (v: Vec) => `[${v.map((x) => x.toFixed(2)).join(", ")}]`;

export function simulate(params: BlockParams): SimulationStep<BlockState>[] {
  const r = run(params);
  const multi = params.blocks === 2;
  return r.stages.map((st, idx) => {
    const tag = multi && st.block >= 0 ? ` (block ${st.block + 1})` : "";
    const t0 = TOKENS[0];
    let description = "";
    let events: SimulationEvent[] = [];
    switch (st.phase) {
      case "input": description = "Three tokens enter as 4-number vectors: the residual stream."; events = [ev("x", `x_${t0} = ${v2(st.after[0])}`)]; break;
      case "ln1": case "ln2": {
        const s = st.stats![0];
        description = "LayerNorm subtracts each vector's own mean and divides by its standard deviation, so every token enters the sub-layer with mean 0 and variance 1.";
        events = [ev("ln", `"${t0}": mean ${s.mu.toFixed(3)}, var ${s.variance.toFixed(3)} → ${v2(st.after[0])}`, `\\mu = ${s.mu.toFixed(3)},\\ \\sigma^2 = ${s.variance.toFixed(3)}`)];
        break;
      }
      case "attn": description = "Each normalized token attends to all three (softmax(QKᵀ/√d)V), then W_O projects the mix back to the stream's size."; events = [ev("a", `attention row "${t0}": ${v2(st.attn![0])}`)]; break;
      case "add1": case "add2":
        description = params.residual ? "The sub-layer's output is added onto the untouched input: x + f(x). The stream keeps everything it had and gains an update." : "Residuals are off: the sub-layer output replaces the stream. Whatever the sub-layer dropped is gone.";
        events = [ev("add", `"${t0}": ${v2(st.before[0])} + ${v2(st.branch![0])} → ${v2(st.after[0])}`), ...(params.residual ? [] : [ev("nores", "Skip connection removed: output = f(x) only.", undefined, "warning")])];
        break;
      case "up": description = "The first MLP layer expands each token from 4 to 8 numbers (W₁·x + b₁)."; events = [ev("up", `"${t0}" hidden pre-activation: ${v2(st.after[0])}`)]; break;
      case "gelu": description = "GELU keeps positive values (almost) as they are and squashes negative values toward 0, smoothly."; events = [ev("g", `${st.before[0].filter((v) => v < 0).length} of 8 values of "${t0}" were negative and shrink toward 0`)]; break;
      case "down": description = "The second MLP layer projects the 8 numbers back to 4 (W₂·h + b₂), ready to be added to the stream."; events = [ev("d", `"${t0}" MLP output: ${v2(st.after[0])}`)]; break;
      case "output": {
        const h = r.health;
        description = `Mean cosine(output, input) = ${h.meanKeep.toFixed(3)}; mean similarity between tokens = ${h.tokenSimilarity.toFixed(3)} (input: ${h.inputSimilarity.toFixed(3)}).`;
        events = [ev("h", `cos(out, in) per token: ${h.keep.map((v) => v.toFixed(3)).join(", ")}`, undefined, h.meanKeep < 0.5 ? "warning" : undefined)];
        break;
      }
    }
    return { id: `${st.phase}-${st.block}-${idx}`, title: TITLES[st.phase] + tag, description, state: st, events };
  });
}
