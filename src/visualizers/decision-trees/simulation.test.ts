import { describe, expect, it } from "vitest";
import { accuracy, buildTree, candidates, generate, gini, predict, simulate, type Pt } from "./simulation";

const p = (x: number, y: number, label: 0 | 1): Pt => ({ x: [x, y], label });

describe("decision tree (CART, Gini)", () => {
  it("computes Gini impurity by hand", () => {
    expect(gini([5, 5])).toBeCloseTo(0.5, 12);
    expect(gini([10, 0])).toBe(0);
    // 1 − (3/4)² − (1/4)² = 1 − 0.5625 − 0.0625 = 0.375
    expect(gini([3, 1])).toBeCloseTo(0.375, 12);
  });

  it("scores candidate thresholds with the weighted child Gini", () => {
    const pts = [p(1, 0, 0), p(2, 0, 0), p(3, 0, 1), p(4, 0, 1)];
    const c = candidates(pts, [0, 1, 2, 3], 0);
    expect(c.map((v) => v.t)).toEqual([1.5, 2.5, 3.5]);
    // t = 1.5: left {A}, right {A,B,B} → 1/4·0 + 3/4·(1 − 1/9 − 4/9) = 1/3
    expect(c[0].g).toBeCloseTo(1 / 3, 12);
    expect(c[1].g).toBe(0); // perfect split
  });

  it("finds the perfect split and predicts correctly", () => {
    const pts = [p(1, 9, 0), p(2, 1, 0), p(3, 5, 0), p(6, 2, 1), p(7, 8, 1), p(9, 4, 1)];
    const { nodes } = buildTree(pts, 3, 2);
    expect(nodes[0].split?.feature).toBe(0);
    expect(nodes[0].split?.threshold).toBe(4.5);
    expect(nodes).toHaveLength(3);
    expect(predict(nodes, [0, 0])).toBe(0);
    expect(predict(nodes, [8, 0])).toBe(1);
    expect(accuracy(nodes, pts)).toBe(1);
  });

  it("respects max depth and min samples", () => {
    const pts = generate("diagonal", 40, 0.1, 3);
    for (const d of [1, 2, 4]) expect(Math.max(...buildTree(pts, d, 2).nodes.map((n) => n.depth))).toBeLessThanOrEqual(d);
    const { nodes } = buildTree(pts, 8, 15);
    for (const n of nodes) if (n.split) expect(n.idx.length).toBeGreaterThanOrEqual(15);
  });

  it("is deterministic and produces data → splits → evaluate steps", () => {
    const cfg = { dataset: "blocks" as const, maxDepth: 3, minSamples: 2, noise: 0.05, seed: 4 };
    const a = simulate(cfg), b = simulate(cfg);
    expect(a.steps.map((s) => s.id)).toEqual(b.steps.map((s) => s.id));
    expect(a.steps[0].id).toBe("data");
    expect(a.steps.at(-1)!.id).toBe("evaluate");
    expect(a.steps.length).toBe(a.nodes.filter((n) => n.split).length + 2);
    // every split step reveals exactly its two children
    for (const s of a.steps.filter((s) => s.state.phase === "split")) expect(s.state.added).toHaveLength(2);
  });

  it("a deep tree on noisy data fits training better than it generalises", () => {
    const deep = simulate({ dataset: "blocks", maxDepth: 8, minSamples: 2, noise: 0.2, seed: 9 });
    expect(deep.trainAcc).toBe(1); // memorises every training point
    expect(deep.testAcc).toBeLessThan(deep.trainAcc - 0.1);
    const shallow = simulate({ dataset: "blocks", maxDepth: 1, minSamples: 2, noise: 0.2, seed: 9 });
    expect(shallow.steps.length).toBeLessThan(deep.steps.length);
  });
});
