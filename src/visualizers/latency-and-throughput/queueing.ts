/** Small, React-free helpers shared by the production-engineering visualizers (model-serving, caching, rate limits…). */

/** Nearest-rank percentile (p in 0..100). NaN for an empty list. */
export function percentile(xs: readonly number[], p: number): number {
  if (!xs.length) return Number.NaN;
  const s = [...xs].sort((a, b) => a - b);
  const rank = Math.ceil((p / 100) * s.length);
  return s[Math.min(s.length, Math.max(1, rank)) - 1];
}

/** Exponential sample with the given mean from a uniform draw u in [0, 1). */
export const expSample = (u: number, mean: number) => -mean * Math.log(1 - u);

/** Time-average of a step function N(t) built from +1/−1 events, over [0, until]. */
export function timeAverage(events: { t: number; d: number }[], until: number): number {
  if (until <= 0) return 0;
  const ev = [...events].sort((a, b) => a.t - b.t || a.d - b.d);
  let n = 0, last = 0, area = 0;
  for (const e of ev) {
    const t = Math.min(e.t, until);
    area += n * (t - last);
    last = t;
    n += e.d;
  }
  area += n * Math.max(0, until - last);
  return area / until;
}

/** Fixed-width histogram counts. */
export function histogram(xs: readonly number[], lo: number, hi: number, bins: number): number[] {
  const out = new Array<number>(bins).fill(0);
  const w = (hi - lo) / bins || 1;
  for (const x of xs) out[Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / w)))]++;
  return out;
}
