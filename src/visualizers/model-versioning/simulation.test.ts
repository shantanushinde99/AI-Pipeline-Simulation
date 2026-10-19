import { describe, expect, it } from "vitest";
import { DEFAULT_MV, SNAPSHOTS, fnv, simulateRegistry, train } from "./simulation";

describe("model versioning", () => {
  it("fnv matches the published FNV-1a test vectors", () => {
    expect(fnv("")).toBe("811c9dc5");
    expect(fnv("a")).toBe("e40c292c");
    expect(fnv("foobar")).toBe("bf9cf968");
  });

  it("training is a pure function of its lineage", () => {
    const a = train(SNAPSHOTS.s2.hash, 15000, "b7e41c9", { lr: 0.01, depth: 6 }, 7);
    expect(train(SNAPSHOTS.s2.hash, 15000, "b7e41c9", { lr: 0.01, depth: 6 }, 7)).toEqual(a);
    expect(train(SNAPSHOTS.s2.hash, 15000, "b7e41c9", { lr: 0.01, depth: 6 }, 8).hash).not.toBe(a.hash);
    // response surface at its optimum: 0.84 + 0.03·log2(15000/12000) = 0.84966 (± 0.005 seed jitter)
    expect(Math.abs(a.accuracy - (0.84 + 0.03 * Math.log2(1.25)))).toBeLessThanOrEqual(0.0051);
  });

  it("produces the registry story in order and ends with production on v2", () => {
    const steps = simulateRegistry(DEFAULT_MV);
    expect(steps.map((s) => s.id)).toEqual(["snap-1", "train-1", "stage-1", "prod-1", "snap-2", "train-2", "stage-2", "promote", "repro", "audit"]);
    expect(steps[3].state.aliases).toEqual({ staging: 1, production: 1 });
    expect(steps[6].state.aliases).toEqual({ staging: 2, production: 1 });
    expect(steps[7].state.gate?.pass).toBe(true);
    expect(steps[7].state.aliases.production).toBe(2);
    expect(steps[0].state.models).toHaveLength(0); // earlier snapshots are not mutated by later steps
  });

  it("a fully recorded lineage reproduces the same hash", () => {
    const repro = simulateRegistry(DEFAULT_MV).find((s) => s.id === "repro")!.state.repro!;
    expect(repro.match).toBe(true);
    expect(repro.diffs).toEqual([]);
  });

  it("each lineage gap breaks reproducibility for its own reason", () => {
    const diffs = (gap: "unpinned-data" | "unlogged-seed" | "dirty-code") => simulateRegistry({ ...DEFAULT_MV, gap }).find((s) => s.id === "repro")!.state.repro!;
    expect(diffs("unpinned-data")).toMatchObject({ match: false, diffs: ["dataHash"] });
    expect(diffs("unlogged-seed")).toMatchObject({ match: false, diffs: ["seed"] });
    expect(diffs("dirty-code")).toMatchObject({ match: false, diffs: ["codeHash"] });
  });

  it("a strict gate keeps production on v1", () => {
    const steps = simulateRegistry({ ...DEFAULT_MV, minGain: 0.08 });
    const promote = steps.find((s) => s.id === "promote")!.state;
    expect(promote.gate!.pass).toBe(false);
    expect(promote.aliases.production).toBe(1);
    // poor hyperparameters also fail the default gate
    const bad = simulateRegistry({ ...DEFAULT_MV, v2: { lr: 0.1, depth: 2 } }).find((s) => s.id === "promote")!.state;
    expect(bad.gate!.v2).toBeLessThan(bad.gate!.v1);
    expect(bad.aliases.production).toBe(1);
  });
});
