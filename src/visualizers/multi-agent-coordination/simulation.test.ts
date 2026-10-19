import { describe, expect, it } from "vitest";
import { DEFAULT_COORD, NET_MS, applyWrite, simulate, type CoordConfig } from "./simulation";

const run = (p: Partial<CoordConfig>) => simulate({ ...DEFAULT_COORD, ...p });

describe("multi-agent coordination", () => {
  it("orchestrator-worker schedule matches hand-computed times and runs workers in parallel", () => {
    const { msgs } = run({});
    const at = (label: string) => msgs.find((m) => m.label.startsWith(label))!;
    // the user's task arrives at 50 ms; the orchestrator plans for 300 ms
    expect(at("Find the API").sent).toBe(350);
    expect(at("Draft the client").sent).toBe(450);
    // researcher ready at 400, works 1500 → 1900; coder ready at 500, works 2000 → 2500
    expect(at("rate limit = 100").sent).toBe(1900);
    expect(at("client.py v1").sent).toBe(2500);
    const final = msgs.at(-1)!;
    expect(final.kind).toBe("final");
    // wall clock is far below the sum of all work because research and coding overlap
    const work = msgs.reduce((a, m) => a + m.work, 0);
    expect(final.arrive).toBeLessThan(work + msgs.length * NET_MS);
  });

  it("messages are shown in time order", () => {
    for (const pattern of ["orchestrator", "pipeline", "debate"] as const)
      for (const failure of ["none", "pingpong", "conflict", "timeout"] as const) {
        const { msgs } = run({ pattern, failure });
        for (let i = 1; i < msgs.length; i++) expect(msgs[i].sent).toBeGreaterThanOrEqual(msgs[i - 1].sent);
      }
  });

  it("ping-pong hand-offs stop at the hop limit; the pipeline fails, the orchestrator recovers", () => {
    const orch = run({ failure: "pingpong", maxHandoffs: 6 });
    expect(orch.msgs.filter((m) => m.kind === "handoff")).toHaveLength(6);
    expect(orch.outcome.status).toBe("success");
    const pipe = run({ pattern: "pipeline", failure: "pingpong", maxHandoffs: 6 });
    expect(pipe.outcome.status).toBe("failed");
    expect(pipe.steps.at(-1)!.state.totals.handoffs).toBe(6);
  });

  it("conflicting outputs: arbitration picks the cited value, last-write-wins ships the wrong one", () => {
    const arb = run({ failure: "conflict" });
    expect(arb.outcome.limitUsed).toBe(100);
    expect(arb.steps.at(-1)!.state.bb.find((e) => e.key === "rate_limit" && e.status === "resolved")!.value).toBe("100/min");
    const lww = run({ failure: "conflict", arbitration: false });
    expect(lww.outcome).toMatchObject({ limitUsed: 60, status: "degraded" });
    expect(lww.steps.at(-1)!.state.bb.filter((e) => e.status === "conflict")).toHaveLength(2);
  });

  it("a hung worker times out at dispatch + timeout; a long enough timeout waits instead", () => {
    const t = run({ failure: "timeout", workerTimeoutMs: 4000 });
    const timeout = t.msgs.find((m) => m.kind === "timeout")!;
    expect(timeout.sent).toBe(350 + 4000);
    expect(t.outcome.status).toBe("degraded");
    const patient = run({ failure: "timeout", workerTimeoutMs: 12000 });
    expect(patient.msgs.some((m) => m.kind === "timeout")).toBe(false);
    expect(patient.outcome.status).toBe("success");
    expect(patient.msgs.at(-1)!.arrive).toBeGreaterThan(t.msgs.find((m) => m.kind === "final")!.arrive);
  });

  it("blackboard writes: new key ok, differing value conflicts, re-assertion resolves", () => {
    let bb = applyWrite([], { key: "k", value: "1", cited: true }, "research", 1);
    bb = applyWrite(bb, { key: "k", value: "2", cited: false }, "coder", 2);
    expect(bb.map((e) => e.status)).toEqual(["conflict", "conflict"]);
    bb = applyWrite(bb, { key: "k", value: "1", cited: true }, "review", 3);
    expect(bb.map((e) => e.status)).toEqual(["resolved", "superseded"]);
  });

  it("steps = intro + one per message + summary", () => {
    const r = run({ pattern: "debate" });
    expect(r.steps).toHaveLength(r.msgs.length + 2);
    expect(r.steps.at(-1)!.state.done).toBe(true);
  });
});
