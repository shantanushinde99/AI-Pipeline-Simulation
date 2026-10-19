import { describe, expect, it } from "vitest";
import { sigmoid } from "../../lib/math";
import { DEFAULT_INPUT, ENTITIES, FEATURES, batchFeature, onlineFeature, predict, simulate, skewReport, type BugId } from "./simulation";

const [c101, c102, , c104] = ENTITIES;
const all = new Set<BugId>(["stale_join", "unit", "timezone", "default_fill"]);

describe("training-serving skew", () => {
  it("batch features match hand calculations", () => {
    expect(batchFeature(c101, "orders_7d")).toBe(3); // 3h, 30h, 100h ago; 200h is outside 7 days
    expect(batchFeature(c101, "avg_order_value")).toBeCloseTo(13750 / 4 / 100, 10); // $34.375
    expect(batchFeature(c101, "last_order_hour")).toBe(21); // 3 hours before midnight UTC
    expect(batchFeature(c102, "avg_rating")).toBe(4.2); // no ratings → training mean
  });

  it("each bug skews exactly its own feature", () => {
    expect(onlineFeature(c101, "orders_7d", all, false)).toBe(2); // stale window (48h, 216h] → 100h, 200h
    expect(onlineFeature(c104, "orders_7d", all, false)).toBe(1); // aged-out order still counted
    expect(onlineFeature(c101, "avg_order_value", all, false)).toBeCloseTo(3437.5, 10); // cents
    expect(onlineFeature(c101, "last_order_hour", all, false)).toBe(16); // UTC−5
    expect(onlineFeature(c102, "avg_rating", all, false)).toBe(0); // null → 0
    expect(onlineFeature(c101, "tenure_days", all, false)).toBe(400);
    expect(onlineFeature(c101, "avg_order_value", new Set<BugId>(["timezone"]), false)).toBeCloseTo(34.375, 10);
  });

  it("the shared definition removes every diff", () => {
    for (const e of ENTITIES) for (const f of FEATURES) expect(onlineFeature(e, f, all, true)).toBe(batchFeature(e, f));
    const rep = skewReport({ ...DEFAULT_INPUT, shared: true });
    expect(rep.perFeature.every((x) => x.mismatches === 0)).toBe(true);
    expect(rep.maxDp).toBe(0);
  });

  it("predict matches a hand-computed logit", () => {
    const x = { orders_7d: 2, avg_order_value: 50, last_order_hour: 18, avg_rating: 4.5, tenure_days: 100 };
    // -2 + 0.9 + 1.0 + 0.36 + 0.4 + 0.2 = 0.86
    expect(predict(x).z).toBeCloseTo(0.86, 10);
    expect(predict(x).p).toBeCloseTo(sigmoid(0.86), 12);
  });

  it("the unit bug saturates the prediction and flips decisions", () => {
    const only = skewReport({ bugs: ["unit"], shared: false });
    expect(only.perFeature.find((x) => x.f === "avg_order_value")!.mismatches).toBe(6);
    expect(only.maxDp).toBeGreaterThan(0.5);
    expect(skewReport({ bugs: [], shared: false }).maxDp).toBe(0);
  });

  it("produces raw + 5 feature + score + report steps, revealing columns progressively", () => {
    const steps = simulate(DEFAULT_INPUT);
    expect(steps.map((s) => s.id)).toEqual(["raw", "f-orders_7d", "f-avg_order_value", "f-last_order_hour", "f-avg_rating", "f-tenure_days", "score", "report"]);
    expect(steps[1].state.rows[0].batch.avg_order_value).toBeUndefined();
    expect(steps[2].state.rows[0].online.avg_order_value).toBeCloseTo(3437.5, 10);
    expect(simulate(DEFAULT_INPUT)).toEqual(steps);
  });
});
