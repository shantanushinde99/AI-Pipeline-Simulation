import { describe, expect, it } from "vitest";
import { DEFAULT_RETRY, jittered, reliability, runRetries, simulate } from "./simulation";

describe("retries and timeouts", () => {
  it("flaky service: exponential backoff timeline matches a hand calculation", () => {
    const run = runRetries(DEFAULT_RETRY);
    // 300 (503) + 200 wait + 350 (503) + 400 wait + 280 (OK) = 1530
    expect(run.segments.map((s) => [s.kind, s.start, s.end])).toEqual([
      ["attempt", 0, 300], ["backoff", 300, 500], ["attempt", 500, 850], ["backoff", 850, 1250], ["attempt", 1250, 1530],
    ]);
    expect(run.status).toBe("success");
    expect(run.endMs).toBe(1530);
  });

  it("timeouts cut attempts at the timeout and a non-retryable error stops immediately", () => {
    const slow = runRetries({ ...DEFAULT_RETRY, scenario: "slow" });
    expect(slow.segments[0]).toMatchObject({ kind: "attempt", outcome: "timeout", end: 2000, serverEnd: 2600 });
    expect(slow.status).toBe("success");
    const bad = runRetries({ ...DEFAULT_RETRY, scenario: "badRequest" });
    expect(bad.status).toBe("non-retryable");
    expect(bad.attempts).toBe(1);
    // Misconfigured policy keeps retrying a permanent error until attempts run out.
    const wasteful = runRetries({ ...DEFAULT_RETRY, scenario: "badRequest", retryAllErrors: true });
    expect(wasteful.attempts).toBe(4);
    expect(wasteful.status).toBe("exhausted");
  });

  it("a retried non-idempotent call double-charges; an idempotency key prevents it", () => {
    const noKey = runRetries({ ...DEFAULT_RETRY, scenario: "payment" });
    expect(noKey.status).toBe("success");
    expect(noKey.charges).toBe(2);
    const withKey = runRetries({ ...DEFAULT_RETRY, scenario: "payment", idempotencyKey: true });
    expect(withKey.charges).toBe(1);
    expect(withKey.segments.at(-1)).toMatchObject({ kind: "attempt", charge: "replayed" });
  });

  it("the deadline bounds total time", () => {
    const run = runRetries({ ...DEFAULT_RETRY, scenario: "slow", deadlineMs: 3000 });
    expect(run.status).toBe("deadline");
    expect(run.endMs).toBeLessThanOrEqual(3000);
  });

  it("jitter is seeded, bounded by the ceiling and changes with the seed", () => {
    expect(jittered(400, "full", 0.5)).toBe(200);
    expect(jittered(400, "equal", 0.5)).toBe(300);
    const a = runRetries({ ...DEFAULT_RETRY, scenario: "outage", jitter: "full", seed: 1 });
    const b = runRetries({ ...DEFAULT_RETRY, scenario: "outage", jitter: "full", seed: 1 });
    const c = runRetries({ ...DEFAULT_RETRY, scenario: "outage", jitter: "full", seed: 2 });
    expect(a).toEqual(b);
    expect(a.endMs).not.toBe(c.endMs);
    for (const s of a.segments) if (s.kind === "backoff") expect(s.end - s.start).toBeLessThanOrEqual(s.ceiling);
  });

  it("reliability model: P(success) = 1 - f^n and expected latency by hand", () => {
    const cfg = { ...DEFAULT_RETRY, timeoutMs: 1000, baseBackoffMs: 100 };
    const r = reliability(0.5, 2, cfg, 200);
    expect(r.pSuccess).toBeCloseTo(0.75, 12);
    // attempt 1: 0.5*200 + 0.5*1000 = 600; attempt 2 (prob 0.5): 0.5*(100 + 600) = 350
    expect(r.expectedMs).toBeCloseTo(950, 9);
  });

  it("steps reveal one segment at a time and end with the run status", () => {
    const { run, steps } = simulate(DEFAULT_RETRY);
    expect(steps).toHaveLength(run.segments.length + 2);
    expect(steps.map((s) => s.state.visible)).toEqual([0, 1, 2, 3, 4, 5, 5]);
    expect(steps.at(-1)!.state.status).toBe("success");
  });
});
