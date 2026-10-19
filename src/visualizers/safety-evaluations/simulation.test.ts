import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, PROMPTS, evaluate, rates, score, simulate } from "./simulation";
import { sigmoid } from "../../lib/math";

const get = (id: string) => PROMPTS.find((p) => p.id === id)!;

describe("safety evaluation", () => {
  it("scores a prompt by hand: bias + risk words + context words → sigmoid", () => {
    // L2: -2 + shoot 2.2 + low 0.7 + portraits -0.6 = 0.3
    const s = score(get("L2"), true);
    expect(s.risk).toEqual(["shoot", "low"]);
    expect(s.context).toEqual(["portraits"]);
    expect(s.z).toBeCloseTo(0.3, 12);
    expect(s.score).toBeCloseTo(sigmoid(0.3), 12);
    // Without context rules the portrait penalty disappears.
    expect(score(get("L2"), false).z).toBeCloseTo(0.9, 12);
  });

  it("computes the confusion matrix and refusal rates at the default threshold", () => {
    const { overall } = evaluate(DEFAULT_CONFIG);
    expect(overall).toMatchObject({ tp: 8, fn: 1, fp: 1, tn: 14 });
    expect(overall.overRefusal).toBeCloseTo(1 / 15, 12);
    expect(overall.underRefusal).toBeCloseTo(1 / 9, 12);
  });

  it("the threshold trades over-refusal against under-refusal", () => {
    const items = PROMPTS.map((p) => score(p, true));
    expect(rates(items, 0)).toMatchObject({ overRefusal: 1, underRefusal: 0 });
    expect(rates(items, 1)).toMatchObject({ overRefusal: 0, underRefusal: 1 });
    const { sweep } = evaluate(DEFAULT_CONFIG);
    for (let i = 1; i < sweep.length; i++) {
      expect(sweep[i].overRefusal).toBeLessThanOrEqual(sweep[i - 1].overRefusal);
      expect(sweep[i].underRefusal).toBeGreaterThanOrEqual(sweep[i - 1].underRefusal);
    }
  });

  it("dropping benign lookalikes hides over-refusal from naive keyword rules", () => {
    const naive = evaluate({ ...DEFAULT_CONFIG, contextAware: false });
    expect(naive.overall.fp).toBe(3); // kill / shoot / attack lookalikes blocked
    expect(naive.overall.overRefusal).toBeCloseTo(3 / 15, 12);
    const hidden = evaluate({ ...DEFAULT_CONFIG, contextAware: false, includeLookalikes: false });
    expect(hidden.overall.overRefusal).toBe(0);
    expect(hidden.set).toHaveLength(18);
  });

  it("builds deterministic steps: set, 4 scoring steps, then the analysis", () => {
    const run = simulate(DEFAULT_CONFIG);
    expect(run.steps.map((s) => s.state.phase)).toEqual(["set", "score", "score", "score", "score", "threshold", "confusion", "category", "tradeoff", "lookalikes"]);
    expect(simulate(DEFAULT_CONFIG).overall).toEqual(run.overall);
    expect(simulate({ ...DEFAULT_CONFIG, threshold: 0.3 }).overall).not.toEqual(run.overall);
  });
});
