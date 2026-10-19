import { describe, expect, it } from "vitest";
import { DEFAULT_GOLDEN, MIN_SUPPORT, cohenKappa, dedupe, gate, jaccard, normalize, simulate, type Item } from "./simulation";

describe("golden datasets", () => {
  it("normalizes and measures near-duplicates with token Jaccard", () => {
    expect(normalize("Where is my order 1211?? please")).toBe("where is my order 1211");
    expect(jaccard("How do I reset my password?", "how do i reset my password!! thanks")).toBe(1);
    // different order numbers: {where,is,my,order,X} vs {…,Y} → 4/6
    expect(jaccard("Where is my order 1211?", "Where is my order 3321?")).toBeCloseTo(4 / 6, 12);
    const d = dedupe([{ id: 1, text: "Do you ship to Canada?" }, { id: 2, text: "do you ship to canada asap" }, { id: 3, text: "Where is my order 1?" }]);
    expect(d.map((x) => x.dupOf)).toEqual([null, 1, null]);
  });

  it("Cohen's kappa matches a hand calculation", () => {
    // 10 items: 8 agree. r1: 6 answer, 4 refuse; r2: 6 answer, 4 refuse; pe = .6·.6 + .4·.4 = .52
    const a = ["answer", "answer", "answer", "answer", "answer", "answer", "refuse", "refuse", "refuse", "refuse"] as const;
    const b = ["answer", "answer", "answer", "answer", "answer", "refuse", "answer", "refuse", "refuse", "refuse"] as const;
    const k = cohenKappa([...a], [...b]);
    expect(k.po).toBeCloseTo(0.8, 12);
    expect(k.pe).toBeCloseTo(0.52, 12);
    expect(k.kappa).toBeCloseTo((0.8 - 0.52) / 0.48, 12);
  });

  it("the gate ignores categories below minimum support", () => {
    const mk = (cat: Item["cat"], candPass: boolean): Item => ({ id: 0, cat, text: "", truth: "answer", dupOf: null, source: "v1", basePass: true, candPass });
    const items = [...Array.from({ length: 20 }, () => mk("billing", true)), mk("nonen", false)];
    expect(gate(items).pass).toBe(true); // 1 failing non-EN example < MIN_SUPPORT
    const more = [...items, ...Array.from({ length: MIN_SUPPORT - 1 }, () => mk("nonen", false))];
    expect(gate(more).pass).toBe(false);
  });

  it("coverage-gap default: the regression passes v1 unnoticed and v2 catches it", () => {
    const s = simulate(DEFAULT_GOLDEN);
    expect(s.coverage.find((c) => c.cat === "nonen")!.n).toBeLessThan(MIN_SUPPORT);
    expect(s.g1.pass).toBe(true);
    expect(s.g2.pass).toBe(false);
    expect(s.coverage2.every((c) => c.n >= DEFAULT_GOLDEN.target)).toBe(true);
    expect(s.v1.length + s.dups.length).toBe(DEFAULT_GOLDEN.sampleSize);
    expect(s.v2.length).toBe(s.v1.length + s.added.length);
    expect(s.hashV1).not.toBe(s.hashV2);
  });

  it("is deterministic, and stratified sampling catches the regression already in v1", () => {
    expect(simulate(DEFAULT_GOLDEN)).toEqual(simulate(DEFAULT_GOLDEN));
    const st = simulate({ ...DEFAULT_GOLDEN, strategy: "stratified" });
    expect(st.coverage.find((c) => c.cat === "nonen")!.n).toBeGreaterThanOrEqual(MIN_SUPPORT);
    expect(st.g1.pass).toBe(false);
    expect(simulate({ ...DEFAULT_GOLDEN, regression: "none" }).g2.pass).toBe(true);
  });

  it("sloppier reviewers lower kappa", () => {
    const good = simulate({ ...DEFAULT_GOLDEN, reviewerAcc: 0.98 }).kappa.kappa;
    const bad = simulate({ ...DEFAULT_GOLDEN, reviewerAcc: 0.6 }).kappa.kappa;
    expect(bad).toBeLessThan(good);
    expect(simulate(DEFAULT_GOLDEN).steps).toHaveLength(10);
  });
});
