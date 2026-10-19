import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { add, attend, matVec, maxAbsDiff, scale, type Mat, type Vec } from "../self-attention/attention";

export const D = 4;
/** Hand-placed teaching-scale embeddings. */
export const WORDS: Record<string, Vec> = {
  dog: [1, 0.2, 0, 0.5],
  bites: [0, 1, 0.3, 0],
  man: [0.8, 0, 0.5, 0.4],
  alice: [0.9, 0.1, 0.6, 0],
  sees: [0.1, 0.9, 0, 0.3],
  bob: [0.7, 0, 0.1, 0.8],
};
export type PresetId = "dog" | "alice";
export const PRESETS: { id: PresetId; a: string[]; b: string[] }[] = [
  { id: "dog", a: ["dog", "bites", "man"], b: ["man", "bites", "dog"] },
  { id: "alice", a: ["alice", "sees", "bob"], b: ["bob", "sees", "alice"] },
];

/** Fixed (hand-set, untrained) single-head projections. Not symmetric, so order effects are not hidden by symmetry. */
export const WQ: Mat = [[1, 0.5, 0, 0], [0, 1, 0.5, 0], [0.5, 0, 1, 0], [0, 0, 0.5, 1]];
export const WK: Mat = [[1, 0, 0, 0.5], [0.5, 1, 0, 0], [0, 0.5, 1, 0], [0, 0, 0.5, 1]];
export const WV: Mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

/** PE(pos, 2i) = sin(pos / base^(2i/d)), PE(pos, 2i+1) = cos(pos / base^(2i/d)). */
export function sinusoidalPE(pos: number, d = D, base = 10000): Vec {
  return Array.from({ length: d }, (_, j) => {
    const i = Math.floor(j / 2);
    const a = pos / base ** ((2 * i) / d);
    return j % 2 === 0 ? Math.sin(a) : Math.cos(a);
  });
}
export const peMatrix = (n: number, d = D) => Array.from({ length: n }, (_, p) => sinusoidalPE(p, d));

/** Bidirectional self-attention outputs for a token sequence, with PE added at strength `pe` (0 = no positions). */
export function encode(tokens: readonly string[], pe: number) {
  const X = tokens.map((t) => WORDS[t]);
  const P = peMatrix(tokens.length).map((v) => scale(v, pe));
  const Xp = X.map((x, i) => add(x, P[i]));
  const att = attend(Xp.map((x) => matVec(WQ, x)), Xp.map((x) => matVec(WK, x)), Xp.map((x) => matVec(WV, x)));
  return { X, P, Xp, out: att.out, weights: att.weights };
}

/** For each word, its output in order A vs order B. Without PE the difference is exactly 0 (permutation equivariance). */
export function orderCheck(preset: PresetId, pe: number) {
  const p = PRESETS.find((x) => x.id === preset) ?? PRESETS[0];
  const A = encode(p.a, pe), B = encode(p.b, pe);
  const rows = p.a.map((w, ia) => {
    const ib = p.b.indexOf(w);
    return { word: w, posA: ia, posB: ib, outA: A.out[ia], outB: B.out[ib], diff: maxAbsDiff(A.out[ia], B.out[ib]) };
  });
  return { preset: p, A, B, rows, maxDiff: Math.max(...rows.map((r) => r.diff)) };
}

// ---- RoPE (2-D) ----
export const rotate = ([x, y]: Vec, a: number): Vec => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
export const ROPE_Q: Vec = [1, 0.25];
export const ROPE_K: Vec = [0.7, 0.7];
export function ropeDot(m: number, n: number, theta: number, q = ROPE_Q, k = ROPE_K) {
  const qm = rotate(q, m * theta), kn = rotate(k, n * theta);
  return { qm, kn, dot: qm[0] * kn[0] + qm[1] * kn[1] };
}

export type PosParams = { preset: PresetId; pe: number; theta: number; m: number; n: number; shift: number };
export const DEFAULT_PARAMS: PosParams = { preset: "dog", pe: 1, theta: 0.5, m: 1, n: 2, shift: 2 };

export type Phase = "words" | "attn-a" | "attn-b" | "pe-matrix" | "pe-add" | "attn-pe" | "rope-rotate" | "rope-relative";
export type PosState = { phase: Phase };

const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });

export function simulate(params: PosParams): SimulationStep<PosState>[] {
  const none = orderCheck(params.preset, 0);
  const withPe = orderCheck(params.preset, params.pe);
  const { a, b } = none.preset;
  const r0 = ropeDot(params.m, params.n, params.theta);
  const r1 = ropeDot(params.m + params.shift, params.n + params.shift, params.theta);
  const st = (phase: Phase): PosState => ({ phase });
  return [
    { id: "words", title: "Same words, different order", state: st("words"), description: `"${a.join(" ")}" and "${b.join(" ")}" use identical words but mean different things.`, events: [ev("w", `Order A: ${a.join(" ")} · Order B: ${b.join(" ")}`)] },
    { id: "attn-a", title: "Attention on order A (no positions)", state: st("attn-a"), description: "Self-attention mixes every token with every other token using only their embeddings.", events: [ev("a", `out(${a[0]}) = [${none.rows[0].outA.map((v) => v.toFixed(3)).join(", ")}]`)] },
    { id: "attn-b", title: "Shuffle and recompute", state: st("attn-b"), description: `Reorder to "${b.join(" ")}" and recompute. Each word's output is identical: attention without positions is order-blind.`, events: [ev("b", `max |out_A − out_B| = ${none.maxDiff.toFixed(6)}`, `\\max |\\text{out}_A - \\text{out}_B| = ${none.maxDiff.toFixed(6)}`, none.maxDiff < 1e-9 ? "warning" : undefined)] },
    { id: "pe-matrix", title: "Sinusoidal positional encodings", state: st("pe-matrix"), description: "Each position gets a fixed vector of sines and cosines at different frequencies: low dimensions change fast, high dimensions slowly.", events: [ev("pe", `PE(1) = [${sinusoidalPE(1).map((v) => v.toFixed(3)).join(", ")}]`)] },
    { id: "pe-add", title: "Add PE to each embedding", state: st("pe-add"), description: `x + ${params.pe.toFixed(1)}·PE(pos): the same word now looks different at different positions.`, events: [ev("add", `x_${a[0]} + PE(0) = [${withPe.A.Xp[0].map((v) => v.toFixed(2)).join(", ")}]`)] },
    { id: "attn-pe", title: "Recompute both orders with positions", state: st("attn-pe"), description: withPe.maxDiff > 1e-9 ? `Now outputs depend on order: max difference ${withPe.maxDiff.toFixed(3)}.` : "PE strength is 0, so the model is order-blind again.", events: [ev("d", `max |out_A − out_B| = ${withPe.maxDiff.toFixed(4)}`, undefined, withPe.maxDiff < 1e-9 ? "warning" : undefined)] },
    { id: "rope-rotate", title: "RoPE: rotate q and k by position", state: st("rope-rotate"), description: `Rotary embeddings rotate the query by m·θ and the key by n·θ instead of adding a vector. q·k = ${r0.dot.toFixed(3)}.`, events: [ev("r", `q·k at m=${params.m}, n=${params.n}: ${r0.dot.toFixed(4)}`, `R(m\\theta)q \\cdot R(n\\theta)k = ${r0.dot.toFixed(4)}`)] },
    { id: "rope-relative", title: "RoPE depends only on distance", state: st("rope-relative"), description: `Shift both positions by ${params.shift}: both vectors turn by the same angle, so the angle between them and q·k stay the same (${r1.dot.toFixed(3)}).`, events: [ev("rr", `q·k at m=${params.m + params.shift}, n=${params.n + params.shift}: ${r1.dot.toFixed(4)} (Δ = ${Math.abs(r1.dot - r0.dot).toExponential(1)})`)] },
  ];
}
