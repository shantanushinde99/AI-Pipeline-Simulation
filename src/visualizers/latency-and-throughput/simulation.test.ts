import { describe, expect, it } from "vitest";
import { DEFAULT_LT, HORIZON_MS, runQueue, serviceSample, simulate, summarize, sweep } from "./simulation";
import { histogram, percentile, timeAverage } from "./queueing";

describe("queueing helpers", () => {
  it("nearest-rank percentiles match a hand calculation", () => {
    const xs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    expect(percentile(xs, 50)).toBe(50);
    expect(percentile(xs, 95)).toBe(100);
    expect(percentile(xs, 10)).toBe(10);
    expect(percentile([], 50)).toBeNaN();
  });
  it("time average of N(t) and histogram counts", () => {
    // one request in [0,2], another in [1,3] over [0,4]: area = 2 + 2 = 4 → L = 1
    expect(timeAverage([{ t: 0, d: 1 }, { t: 2, d: -1 }, { t: 1, d: 1 }, { t: 3, d: -1 }], 4)).toBeCloseTo(1, 12);
    expect(histogram([0, 1, 5, 9.9, 10], 0, 10, 2)).toEqual([2, 3]);
  });
});

describe("M/G/c simulation", () => {
  it("constant service with one worker: hand-checkable FIFO", () => {
    const reqs = runQueue({ ...DEFAULT_LT, workers: 1, dist: "constant", serviceMs: 100 });
    for (let i = 1; i < reqs.length; i++) expect(reqs[i].start).toBeCloseTo(Math.max(reqs[i].arrive, reqs[i - 1].end), 9);
    expect(reqs.every((q) => Math.abs(q.end - q.start - 100) < 1e-9)).toBe(true);
  });
  it("is deterministic for a seed and changes with the seed", () => {
    expect(runQueue(DEFAULT_LT)).toEqual(runQueue(DEFAULT_LT));
    expect(runQueue({ ...DEFAULT_LT, seed: 8 })[0].arrive).not.toBe(runQueue(DEFAULT_LT)[0].arrive);
  });
  it("Little's law holds exactly on the sample path", () => {
    const s = summarize(runQueue(DEFAULT_LT), DEFAULT_LT);
    expect(s.L).toBeCloseTo(s.lambdaW, 9);
    expect(s.capacity).toBeCloseTo(12, 9);
  });
  it("throughput saturates at capacity and p99 explodes when overloaded", () => {
    const pts = sweep(DEFAULT_LT);
    const over = pts.at(-1)!;
    expect(over.rho).toBeGreaterThan(1.2);
    expect(over.throughput).toBeLessThan(12.5);
    expect(over.p50).toBeGreaterThan(pts[0].p50 * 10);
    expect(over.p99).toBeGreaterThan(pts[0].p99 * 3);
  });
  it("burst raises the arrival count and the max queue", () => {
    const a = simulate({ ...DEFAULT_LT, arrivalRate: 6 });
    const b = simulate({ ...DEFAULT_LT, arrivalRate: 6, burst: true });
    expect(b.sum.n).toBeGreaterThan(a.sum.n);
    expect(b.sum.maxQueue).toBeGreaterThan(a.sum.maxQueue);
  });
  it("produces one step per simulated second plus a summary, and the heavy tail keeps the mean", () => {
    const { steps } = simulate(DEFAULT_LT);
    expect(steps).toHaveLength(HORIZON_MS / 1000 + 1);
    expect(steps.at(-1)!.id).toBe("summary");
    // heavy tail mixture: 0.9·m/1.9 + 0.1·10m/1.9 = m (with u = 0.5 the jitter factor is 1)
    expect(0.9 * serviceSample("heavy-tail", 190, 0.5, 0) + 0.1 * serviceSample("heavy-tail", 190, 0.5, 0.95)).toBeCloseTo(190, 9);
  });
});
