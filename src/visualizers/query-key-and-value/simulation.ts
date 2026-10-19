import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { attend, matVec, type Mat, type Vec } from "../self-attention/attention";

export const TOKENS = ["Kim", "drank", "hot", "tea"];
/** Teaching-scale embedding features. */
export const DIMS = ["person", "action", "temperature", "drinkable"];
/** What each query/key dimension means: a query asks "I seek …", a key says "I am …". */
export const QK_DIMS = ["agent", "action", "modifier", "drinkable"];
export const D = 4;

export const EMBEDDINGS: Mat = [
  [1, 0, 0, 0], // Kim
  [0, 1, 0, 0.2], // drank
  [0, 0, 1, 0.3], // hot
  [0, 0, 0.2, 1], // tea
];

export type MatName = "WQ" | "WK" | "WV";
export type Weights = Record<MatName, Mat>;

/** Hand-designed: actions seek an agent and something drinkable; modifiers seek drinkables; nouns seek actions/modifiers. */
export const DEFAULT_WEIGHTS: Weights = {
  WQ: [
    [0, 1, 0, 0],
    [1.5, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 2, 1.5, 0],
  ],
  WK: [
    [2, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 1.5, 0],
    [0, 0, 0, 2],
  ],
  WV: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0.5, 1],
  ],
};

export type QKVParams = { query: number; weights: Weights; swapKeys: boolean };
export const DEFAULT_PARAMS: QKVParams = { query: 1, weights: DEFAULT_WEIGHTS, swapKeys: false };
/** The pair whose keys are exchanged by the "swap keys" control (values stay put). */
export const SWAP: [number, number] = [2, 3];

export function setEntry(w: Weights, m: MatName, r: number, c: number, v: number): Weights {
  const n = structuredClone(w);
  n[m][r][c] = Math.max(-5, Math.min(5, v));
  return n;
}

export function compute(params: QKVParams) {
  const i = Math.max(0, Math.min(TOKENS.length - 1, Math.round(params.query)));
  const { WQ, WK, WV } = params.weights;
  const X = EMBEDDINGS;
  const Q = X.map((x) => matVec(WQ, x));
  const K = X.map((x) => matVec(WK, x));
  if (params.swapKeys) [K[SWAP[0]], K[SWAP[1]]] = [K[SWAP[1]], K[SWAP[0]]];
  const V = X.map((x) => matVec(WV, x));
  const att = attend(Q, K, V);
  const weights = att.weights[i];
  const top = weights.reduce((b, w, j) => (w > weights[b] ? j : b), 0);
  // Running total of the output as each token's weighted value is added (used to animate the stacking).
  const partial: Mat = [];
  weights.reduce((acc, w, j) => {
    const next = acc.map((v, d) => v + w * V[j][d]);
    partial.push(next);
    return next;
  }, new Array<number>(D).fill(0));
  return { i, X, Q, K, V, q: Q[i], scores: att.scores[i], scaled: att.scaled[i], weights, out: att.out[i], top, partial };
}

export type Phase = "embed" | "project-q" | "project-k" | "project-v" | "dot" | "scale" | "softmax" | "mix";
export const PHASES: Phase[] = ["embed", "project-q", "project-k", "project-v", "dot", "scale", "softmax", "mix"];
export type QKVState = { phase: Phase; query: number; q: Vec; K: Mat; V: Mat; scores: Vec; scaled: Vec; weights: Vec; out: Vec; top: number };

const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });
const v2 = (v: Vec) => v.map((x) => x.toFixed(2)).join(", ");

export function simulate(params: QKVParams): SimulationStep<QKVState>[] {
  const c = compute(params);
  const tok = TOKENS[c.i];
  const base = { query: c.i, q: [...c.q], K: c.K.map((r) => [...r]), V: c.V.map((r) => [...r]), scores: [...c.scores], scaled: [...c.scaled], weights: [...c.weights], out: [...c.out], top: c.top };
  const st = (phase: Phase): QKVState => ({ ...base, phase });
  const swapNote = params.swapKeys ? [ev("swap", `Keys of "${TOKENS[SWAP[0]]}" and "${TOKENS[SWAP[1]]}" are swapped (values are not).`, undefined, "warning")] : [];
  return [
    { id: "embed", title: "Start from embeddings", state: st("embed"), description: `Each token is a 4-number embedding x. "${tok}" will be the query token.`, events: [ev("x", `x_${tok} = [${c.X[c.i].join(", ")}]`)] },
    { id: "project-q", title: "q = W_Q · x", state: st("project-q"), description: `Multiply W_Q by "${tok}"'s embedding, one row at a time: each row's dot product with x gives one entry of the query.`, events: [ev("q", `q = [${v2(c.q)}]`, `q = W_Q\\,x_{\\text{${tok}}} = [${v2(c.q)}]`)] },
    { id: "project-k", title: "K = W_K · x for every token", state: st("project-k"), description: "The same key matrix is applied to every token's embedding, giving each token a key: the label it is filed under.", events: [ev("k", `k_${TOKENS[c.top]} = [${v2(c.K[c.top])}]`), ...swapNote] },
    { id: "project-v", title: "V = W_V · x for every token", state: st("project-v"), description: "A third matrix gives each token a value: the content it hands over if it is attended to.", events: [ev("v", `v_${TOKENS[c.top]} = [${v2(c.V[c.top])}]`)] },
    { id: "dot", title: "q · k for each key", state: st("dot"), description: `The query of "${tok}" is compared with each key by a dot product. Highest: "${TOKENS[c.top]}" = ${c.scores[c.top].toFixed(2)}.`, events: c.scores.map((s, j) => ev(`s${j}`, `q·k_${TOKENS[j]} = ${s.toFixed(2)}`, `q \\cdot k_{\\text{${TOKENS[j]}}} = ${s.toFixed(2)}`)) },
    { id: "scale", title: "Scale by 1/√d", state: st("scale"), description: "Divide each score by √4 = 2 so the softmax is not over-confident.", events: [ev("sc", `${c.scores[c.top].toFixed(2)} / 2 = ${c.scaled[c.top].toFixed(2)}`)] },
    { id: "softmax", title: "Softmax → attention weights", state: st("softmax"), description: `Weights sum to 1. "${TOKENS[c.top]}" receives ${(c.weights[c.top] * 100).toFixed(1)}% of "${tok}"'s attention.`, events: [ev("w", `weights = [${c.weights.map((w) => w.toFixed(3)).join(", ")}]`)] },
    { id: "mix", title: "Mix the values", state: st("mix"), description: `out = Σ w_j v_j. The keys decided where to look; the values decide what comes back.`, events: [ev("out", `out = [${v2(c.out)}]`, `\\text{out} = \\sum_j w_j v_j = [${v2(c.out)}]`), ...swapNote] },
  ];
}
