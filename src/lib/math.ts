export const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
export const norm = (a: readonly number[]) => Math.sqrt(dot(a, a));
export const euclidean = (a: readonly number[], b: readonly number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

/** Returns NaN when either vector has zero length: cosine similarity is undefined there. */
export function cosine(a: readonly number[], b: readonly number[]) {
  const d = norm(a) * norm(b);
  return d === 0 ? Number.NaN : dot(a, b) / d;
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : Number.NaN);
export const round = (v: number, digits = 3) => Number(v.toFixed(digits));
export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : "undefined");

/** Solve A x = b with Gaussian elimination and partial pivoting. Throws on a singular system. */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) throw new Error("singular system");
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}
