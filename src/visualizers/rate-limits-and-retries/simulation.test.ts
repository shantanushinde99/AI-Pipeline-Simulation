import { describe, expect, it } from "vitest";
import { DEFAULT_RL, horizonOf, backoffDelay, refill, runRateLimit, simulate, tokensAt } from "./simulation";

describe("token bucket and backoff math", () => {
  it("refills at r per second up to b", () => {
    expect(refill(0, 0, 200, 5, 5)).toBeCloseTo(1, 12);
    expect(refill(2, 1000, 1400, 5, 5)).toBeCloseTo(4, 12);
    expect(refill(4, 0, 10_000, 5, 5)).toBe(5);
  });
  it("backoff delays per policy", () => {
    expect([1, 2, 3].map((k) => backoffDelay("fixed", 500, k, 0.5))).toEqual([500, 500, 500]);
    expect([1, 2, 3, 4].map((k) => backoffDelay("exponential", 500, k, 0.5))).toEqual([500, 1000, 2000, 4000]);
    expect(backoffDelay("exp-jitter", 500, 3, 0.25)).toBe(500);
    expect(backoffDelay("exponential", 500, 10, 0)).toBe(8000);
  });
});

describe("rate-limit simulation", () => {
  it("the first wave admits exactly the burst, and the bucket replays consistently", () => {
    const { attempts, cfg } = runRateLimit(DEFAULT_RL);
    const first = attempts.filter((a) => a.n === 1);
    expect(first).toHaveLength(24);
    expect(first.filter((a) => a.ok)).toHaveLength(5);
    for (const a of attempts) expect(tokensAt(attempts, cfg, a.t)).toBeCloseTo(a.tokensAfter, 9);
  });
  it("fixed delays retry in synchronized waves", () => {
    const { attempts } = runRateLimit(DEFAULT_RL);
    // every attempt lands within the first 100 ms of a 500 ms slot
    expect(attempts.every((a) => a.t % 500 < 100)).toBe(true);
  });
  it("jitter removes retry spikes; no-jitter policies either give up or finish late", () => {
    const fixed = simulate(DEFAULT_RL).sum;
    const expo = simulate({ ...DEFAULT_RL, policy: "exponential" }).sum;
    const jitter = simulate({ ...DEFAULT_RL, policy: "exp-jitter" }).sum;
    expect(fixed.gaveUp).toBeGreaterThan(0);
    expect(jitter.succeeded).toBe(24);
    expect(jitter.retrySpike).toBeLessThan(fixed.retrySpike * 0.7);
    expect(expo.retrySpike).toBe(fixed.retrySpike);
    expect(jitter.doneAt).toBeLessThan(expo.doneAt);
  });
  it("respecting Retry-After never retries before the advertised time", () => {
    const { attempts } = runRateLimit({ ...DEFAULT_RL, policy: "exp-jitter", baseMs: 100, retryAfter: true });
    for (const a of attempts.filter((x) => x.nextAt !== undefined)) expect(a.nextAt! - a.t).toBeGreaterThanOrEqual(a.retryAfterMs!);
    expect(simulate({ ...DEFAULT_RL, policy: "exp-jitter", baseMs: 100, retryAfter: true }).sum.early).toBe(0);
    expect(simulate({ ...DEFAULT_RL, policy: "exp-jitter", baseMs: 100 }).sum.early).toBeGreaterThan(0);
  });
  it("is deterministic, has one step per second plus a summary, and gives up after max attempts", () => {
    expect(runRateLimit(DEFAULT_RL)).toEqual(runRateLimit(DEFAULT_RL));
    const run = simulate(DEFAULT_RL);
    expect(run.steps).toHaveLength(horizonOf(run.attempts) / run.windowMs + 1);
    expect(run.steps.length).toBeLessThanOrEqual(9);
    expect(run.attempts.at(-1)!.t).toBeLessThan(run.horizon);
    const harsh = simulate({ ...DEFAULT_RL, clients: 40, baseMs: 300, maxAttempts: 5 });
    expect(harsh.sum.gaveUp).toBeGreaterThan(0);
    expect(harsh.attempts.filter((a) => a.gaveUp).every((a) => a.n === 5)).toBe(true);
  });
});
