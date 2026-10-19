import { describe, expect, it } from "vitest";
import { COLD_START_MS, DEFAULT_SERVING, HORIZON_MS, SCALE_AT_MS, WINDOW_MS, batchMs, batchReplica, metricsAt, runServing, simulate, sweepBatch } from "./simulation";

const arr = (ts: number[]) => ts.map((arrive, id) => ({ id, arrive }));

describe("dynamic batcher", () => {
  it("closes on size when requests arrive fast", () => {
    const b = batchReplica(arr([0, 1, 2, 3]), 0, 4, 10);
    expect(b).toHaveLength(1);
    expect(b[0]).toMatchObject({ start: 3, size: 4, reason: "full", end: 3 + batchMs(4) });
  });
  it("closes on max wait when traffic is light", () => {
    const b = batchReplica(arr([0, 5, 30]), 0, 8, 10);
    expect(b.map((x) => [x.start, x.size, x.reason])).toEqual([[10, 2, "max wait"], [40, 1, "max wait"]]);
  });
  it("waits for a busy GPU and then takes everything queued", () => {
    // first batch runs 0→22.5 (size 1, wait 0); 1, 2, 3 queue meanwhile and go together at 22.5
    const b = batchReplica(arr([0, 1, 2, 3]), 0, 8, 0);
    expect(b[0]).toMatchObject({ start: 0, size: 1 });
    expect(b[1]).toMatchObject({ start: batchMs(1), size: 3, reason: "GPU was busy" });
  });
  it("a cold replica cannot start before it is ready", () => {
    const b = batchReplica(arr([0]), 450, 8, 0);
    expect(b[0].start).toBe(450);
  });
});

describe("serving simulation", () => {
  it("is deterministic and produces one step per window", () => {
    const a = simulate(DEFAULT_SERVING);
    expect(a.reqs).toEqual(simulate(DEFAULT_SERVING).reqs);
    expect(a.steps).toHaveLength(HORIZON_MS / WINDOW_MS);
    expect(a.reqs.every((q) => q.end > q.start && q.start >= q.arrive)).toBe(true);
  });
  it("batching beats batch size 1 under load", () => {
    const pts = sweepBatch(DEFAULT_SERVING);
    const one = pts.find((p) => p.maxBatch === 1)!, eight = pts.find((p) => p.maxBatch === 8)!;
    expect(eight.p95).toBeLessThan(one.p95 / 3);
    expect(eight.throughput).toBeGreaterThan(one.throughput);
  });
  it("routing before readiness makes requests wait through the cold start", () => {
    const cfg = { ...DEFAULT_SERVING, replicas: 1, arrivalRate: 260, scaleOut: true };
    const safe = runServing(cfg), unsafe = runServing({ ...cfg, routeBeforeReady: true });
    const newId = 1;
    expect(safe.reqs.filter((q) => q.replica === newId).every((q) => q.arrive >= SCALE_AT_MS + COLD_START_MS)).toBe(true);
    const stuck = unsafe.reqs.filter((q) => q.replica === newId && q.arrive < SCALE_AT_MS + COLD_START_MS);
    expect(stuck.length).toBeGreaterThan(0);
    expect(stuck.every((q) => q.start >= SCALE_AT_MS + COLD_START_MS)).toBe(true);
  });
  it("GPU utilization is busy time over available time", () => {
    const run = { reqs: [], replicas: [{ id: 0, createdAt: 0, readyAt: 0 }], batches: [{ id: 0, replica: 0, size: 1, start: 10, end: 30, reason: "full" as const, reqIds: [], firstArrive: 0 }] };
    expect(metricsAt(run, 100).gpuUtil).toBeCloseTo(0.2, 12);
    expect(metricsAt(run, 20).gpuUtil).toBeCloseTo(0.5, 12);
  });
});
