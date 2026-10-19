import { describe, expect, it } from "vitest";
import type { TraceSpan } from "../../types/simulation";
import { DEFAULT_INPUT, buildTrace, criticalPath, samplePopulation, simulate, spanEnd, traceparent } from "./simulation";

const span = (id: string, parentId: string | undefined, startMs: number, durationMs: number): TraceSpan => ({ id, parentId, name: id, component: "x", startMs, durationMs, status: "ok", events: [] });

describe("distributed tracing", () => {
  it("critical path picks the slowest parallel child and self times sum to the total", () => {
    // root 0–100; A 0–10 then parallel B 10–40 and C 10–90; D 90–100
    const spans = [span("root", undefined, 0, 100), span("A", "root", 0, 10), span("B", "root", 10, 30), span("C", "root", 10, 80), span("D", "root", 90, 10)];
    const cp = criticalPath(spans);
    expect(cp.map((c) => c.id)).toEqual(["root", "A", "C", "D"]);
    expect(cp.reduce((t, c) => t + c.self, 0)).toBe(100);
    expect(cp[0].self).toBe(0);
  });

  it("formats a W3C traceparent header", () => {
    expect(traceparent("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", true)).toBe("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
    expect(traceparent("a".repeat(32), "b".repeat(16), false).endsWith("-00")).toBe(true);
  });

  it("parallel fan-out waits for the slowest shard; sequential adds them up", () => {
    const par = buildTrace(3, { slowShard: true, toolError: false, parallel: true }).spans;
    const seq = buildTrace(3, { slowShard: true, toolError: false, parallel: false }).spans;
    const shards = (s: TraceSpan[]) => s.filter((x) => x.component === "vector-db");
    const retr = (s: TraceSpan[]) => s.find((x) => x.name === "retrieve")!;
    expect(spanEnd(retr(par))).toBe(Math.max(...shards(par).map(spanEnd)) + 3);
    const sumShards = shards(seq).reduce((t, x) => t + x.durationMs, 0);
    expect(retr(seq).durationMs).toBeGreaterThan(sumShards);
    expect(seq[0].durationMs).toBeGreaterThan(par[0].durationMs);
  });

  it("every span shares one trace and children start inside their parent", () => {
    const run = simulate(DEFAULT_INPUT);
    const byId = new Map(run.spans.map((s) => [s.id, s]));
    for (const s of run.spans.filter((x) => x.parentId)) {
      const p = byId.get(s.parentId!)!;
      expect(s.startMs).toBeGreaterThanOrEqual(p.startMs);
      expect(spanEnd(s)).toBeLessThanOrEqual(spanEnd(p));
    }
    const hops = run.steps.flatMap((s) => s.state.hops);
    hops.forEach((h) => expect(h.header).toContain(run.traceId));
    expect(run.critical.reduce((t, c) => t + c.self, 0)).toBe(run.total);
    expect(run.steps).toHaveLength(9);
  });

  it("the slow-shard preset makes the shard the bottleneck", () => {
    const run = simulate({ ...DEFAULT_INPUT, preset: "slow-shard" });
    const top = run.critical.slice(1).reduce((a, b) => (b.self > a.self ? b : a));
    expect(run.spans.find((s) => s.id === top.id)!.name).toBe("ann_search shard-2");
    const healthy = simulate({ ...DEFAULT_INPUT, preset: "healthy" });
    const topH = healthy.critical.slice(1).reduce((a, b) => (b.self > a.self ? b : a));
    expect(healthy.spans.find((s) => s.id === topH.id)!.component).toBe("llm");
  });

  it("tail sampling keeps every slow or failed trace; head sampling is blind to them", () => {
    const head = samplePopulation({ ...DEFAULT_INPUT, samplingRate: 0.2, tailSampling: false });
    const tail = samplePopulation({ ...DEFAULT_INPUT, samplingRate: 0.2, tailSampling: true });
    expect(tail.filter((p) => p.slow || p.error).every((p) => p.kept)).toBe(true);
    expect(head.filter((p) => p.kept).every((p) => p.head)).toBe(true);
    expect(samplePopulation({ ...DEFAULT_INPUT, samplingRate: 1 }).every((p) => p.kept)).toBe(true);
    expect(samplePopulation({ ...DEFAULT_INPUT, samplingRate: 0 }).some((p) => p.kept)).toBe(false);
    expect(samplePopulation(DEFAULT_INPUT)).toEqual(samplePopulation(DEFAULT_INPUT));
  });
});
