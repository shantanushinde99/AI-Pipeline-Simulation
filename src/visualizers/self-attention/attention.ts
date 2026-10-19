/**
 * Small, pure linear-algebra helpers shared by the transformer visualizers
 * (self-attention, query-key-and-value, positional-information, transformer-blocks).
 * Matrices are row-major: W[i][j] is the weight from input j to output i, so y = W·x means y_i = Σ_j W[i][j]·x_j.
 */
import { dot } from "../../lib/math";

export type Vec = number[];
export type Mat = number[][];

export const matVec = (W: Mat, x: Vec): Vec => W.map((row) => dot(row, x));
export const add = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i]);
export const scale = (a: Vec, s: number): Vec => a.map((v) => v * s);
export const transpose = (A: Mat): Mat => (A.length ? A[0].map((_, j) => A.map((r) => r[j])) : []);
export const matMul = (A: Mat, B: Mat): Mat => A.map((row) => transpose(B).map((col) => dot(row, col)));
export const identity = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
export const maxAbsDiff = (a: Vec, b: Vec) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

/** Numerically stable softmax. −Infinity entries (masked) get weight 0; an all-masked row returns zeros. */
export function softmax(xs: Vec): Vec {
  const m = Math.max(...xs);
  if (!Number.isFinite(m)) return xs.map(() => 0);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export type AttentionResult = { scores: Mat; scaled: Mat; weights: Mat; out: Mat };

/** Single-head scaled dot-product attention over row vectors Q, K, V (one row per token). */
export function attend(Q: Mat, K: Mat, V: Mat, opts: { scaled?: boolean; causal?: boolean } = {}): AttentionResult {
  const { scaled = true, causal = false } = opts;
  const d = Q[0]?.length ?? 1;
  const f = scaled ? 1 / Math.sqrt(d) : 1;
  const scores = Q.map((q) => K.map((k) => dot(q, k)));
  const scaledM = scores.map((row, i) => row.map((s, j) => (causal && j > i ? -Infinity : s * f)));
  const weights = scaledM.map(softmax);
  const out = weights.map((w) => V[0].map((_, c) => w.reduce((s, wj, j) => s + wj * V[j][c], 0)));
  return { scores, scaled: scaledM, weights, out };
}

/** Number → LaTeX-safe string; negatives get parentheses so substituted formulas read correctly. */
export const tn = (v: number, d = 2) => (v < 0 ? `(${v.toFixed(d)})` : v.toFixed(d));
export const texVec = (v: Vec, d = 2) => `\\begin{bmatrix} ${v.map((x) => x.toFixed(d)).join(" & ")} \\end{bmatrix}`;
export const texMat = (M: Mat, d = 1) => `\\begin{bmatrix} ${M.map((r) => r.map((x) => x.toFixed(d)).join(" & ")).join(" \\\\ ")} \\end{bmatrix}`;
