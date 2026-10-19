import { describe, expect, it } from "vitest";
import { sigmoid } from "../../lib/math";
import { DEFAULT_LOG, INIT, boundary, gradient, loss, makeData, prob, simulate, type Params } from "./simulation";

describe("logistic regression", () => {
  it("matches a hand calculation of probability and loss", () => {
    const w = { w1: 1, w2: -1, b: 0.5 };
    expect(prob(w, [2, 1])).toBeCloseTo(sigmoid(1.5), 12);
    // one point of each class: −[ln σ(1.5) + ln(1 − σ(−0.5))] / 2
    const l = loss([{ x: [2, 1], y: 1 }, { x: [0, 1], y: 0 }], w, 0);
    expect(l).toBeCloseTo(-(Math.log(sigmoid(1.5)) + Math.log(1 - sigmoid(-0.5))) / 2, 12);
  });

  it("analytic gradient matches finite differences (with regularization)", () => {
    const pts = makeData("overlap", 3);
    const w: Params = { w1: 0.4, w2: -0.7, b: 0.2 };
    const g = gradient(pts, w, 0.3);
    const h = 1e-6;
    for (const key of ["w1", "w2", "b"] as const) {
      const num = (loss(pts, { ...w, [key]: w[key] + h }, 0.3) - loss(pts, { ...w, [key]: w[key] - h }, 0.3)) / (2 * h);
      expect(g[key]).toBeCloseTo(num, 6);
    }
  });

  it("learns the separable preset and the loss keeps falling", () => {
    const { history } = simulate(DEFAULT_LOG);
    const last = history.at(-1)!;
    expect(last.acc).toBeGreaterThanOrEqual(0.95);
    expect(last.loss).toBeLessThan(history[0].loss / 2);
    for (let i = 1; i < history.length; i++) expect(history[i].loss).toBeLessThanOrEqual(history[i - 1].loss + 1e-12);
  });

  it("cannot solve XOR: loss stays near ln 2 and accuracy stays low", () => {
    const { history } = simulate({ ...DEFAULT_LOG, preset: "xor", epochs: 200 });
    const last = history.at(-1)!;
    expect(last.acc).toBeLessThanOrEqual(0.75);
    expect(last.loss).toBeGreaterThan(0.6);
  });

  it("regularization shrinks the weights; zero learning rate changes nothing", () => {
    const a = simulate({ ...DEFAULT_LOG, lambda: 0 }).history.at(-1)!;
    const b = simulate({ ...DEFAULT_LOG, lambda: 0.5 }).history.at(-1)!;
    expect(Math.hypot(b.w1, b.w2)).toBeLessThan(Math.hypot(a.w1, a.w2));
    const still = simulate({ ...DEFAULT_LOG, lr: 0 }).history.at(-1)!;
    expect([still.w1, still.w2, still.b]).toEqual([INIT.w1, INIT.w2, INIT.b]);
  });

  it("is deterministic and steps are data → init → epochs → result", () => {
    const s = simulate(DEFAULT_LOG);
    expect(simulate(DEFAULT_LOG).pts).toEqual(s.pts);
    expect(s.steps[0].id).toBe("data");
    expect(s.steps[1].id).toBe("init");
    expect(s.steps.at(-1)!.id).toBe("result");
    expect(s.steps.at(-2)!.id).toBe(`epoch-${DEFAULT_LOG.epochs}`);
  });

  it("clips the boundary to the box", () => {
    const seg = boundary({ w1: 1, w2: -1, b: 0 })!; // y = x
    expect(seg[0][0]).toBeCloseTo(-3);
    expect(seg[1][1]).toBeCloseTo(3);
    expect(boundary({ w1: 0, w2: 1, b: -10 })).toBeNull(); // y = 10 is outside
  });
});
