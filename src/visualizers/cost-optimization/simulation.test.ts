import { describe, expect, it } from "vitest";
import { BASELINE, DEFAULT_CONFIG, LEVERS, evaluate, frontier, requestCost, simulate, waterfall } from "./simulation";

describe("cost optimization", () => {
  it("prices one request by hand: in × in-price + out × out-price", () => {
    expect(requestCost("large", 1800, 1100)).toBeCloseTo((1800 * 3 + 1100 * 15) / 1e6, 12); // $0.0219
    expect(requestCost("small", 1000, 200)).toBeCloseTo((1000 * 0.25 + 200 * 1.25) / 1e6, 12);
  });

  it("baseline: 2M requests on the large model = $24,000/month at quality 0.932", () => {
    const b = evaluate(BASELINE);
    // avg output = 0.6·250 + 0.25·500 + 0.15·1100 = 440; per request = (1800·3 + 440·15)/1e6 = $0.012
    expect(b.cost).toBeCloseTo(24000, 6);
    expect(b.quality).toBeCloseTo(0.6 * 0.95 + 0.25 * 0.92 + 0.15 * 0.88, 12);
  });

  it("each lever moves cost (and quality) the way it should", () => {
    expect(evaluate({ ...BASELINE, hitRate: 0.2 }).cost).toBeCloseTo(24000 * 0.8, 6);
    expect(evaluate({ ...BASELINE, batching: true }).cost).toBeCloseTo(24000 * (1 - 0.5 * 0.3), 6);
    const allSmall = evaluate({ ...BASELINE, route: 3 });
    expect(allSmall.quality).toBeCloseTo(0.6 * 0.93 + 0.25 * 0.78 + 0.15 * 0.5, 12);
    expect(allSmall.cost).toBeLessThan(24000 / 10);
    const capped = evaluate({ ...BASELINE, cap: 550 });
    expect(capped.rows[2].outTok).toBe(550);
    expect(capped.rows[2].quality).toBeCloseTo(0.88 * 0.75, 12); // half the answer → capF 0.75
    expect(evaluate({ ...BASELINE, trim: 0.3 }).quality).toBeCloseTo(evaluate(BASELINE).quality, 12); // free up to 30%
  });

  it("waterfall savings add up from baseline to the final config", () => {
    const w = waterfall(DEFAULT_CONFIG);
    expect(w.stages.map((s) => s.id)).toEqual(LEVERS.map((l) => l.id));
    const total = w.stages.reduce((s, x) => s + x.saving, 0);
    expect(w.baseline.cost - total).toBeCloseTo(w.final.cost, 6);
    expect(w.final.cost).toBeCloseTo(evaluate(DEFAULT_CONFIG).cost, 6);
    expect(w.stages.every((s) => s.saving >= 0)).toBe(true);
  });

  it("frontier points are not dominated by any configuration", () => {
    const { pts, front } = frontier(DEFAULT_CONFIG);
    expect(pts).toHaveLength(100);
    for (const f of front) expect(pts.some((p) => p.cost < f.cost - 1e-9 && p.quality > f.quality + 1e-9)).toBe(false);
    for (let i = 1; i < front.length; i++) expect(front[i].quality).toBeGreaterThan(front[i - 1].quality);
  });

  it("builds deterministic steps and reacts to parameters", () => {
    const run = simulate(DEFAULT_CONFIG);
    expect(run.steps.map((s) => s.id)).toEqual(["request", "baseline", "lever-trim", "lever-cache", "lever-route", "lever-batch", "lever-cap", "total", "frontier"]);
    expect(simulate(DEFAULT_CONFIG)).toEqual(run);
    expect(simulate({ ...DEFAULT_CONFIG, hitRate: 0.5 }).current.cost).toBeLessThan(run.current.cost);
    expect(evaluate({ ...DEFAULT_CONFIG, trim: 5, route: 9 }).config).toMatchObject({ trim: 0.6, route: 3 });
  });
});
