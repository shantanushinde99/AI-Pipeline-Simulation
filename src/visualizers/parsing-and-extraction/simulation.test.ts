import { describe, expect, it } from "vitest";
import { buildPage, diffWords, factFound, parse, referenceText, score, simulate } from "./simulation";

describe("parsing and extraction", () => {
  it("naive extraction interleaves the columns, keeps furniture and drops the table", () => {
    const r = parse(buildPage(24), "naive");
    expect(r.text).toContain("SSO is available on the Request logs are retained for Enterprise plan only.");
    expect(r.text.startsWith("Northwind Cloud Admin Guide")).toBe(true);
    expect(r.text).toContain("Page 12 of 40");
    expect(r.text).not.toContain("600");
    expect(r.dropped).toHaveLength(12);
  });

  it("layout-aware reads columns in order but flattens table cells", () => {
    const r = parse(buildPage(24), "layout");
    expect(r.columnsDetected).toBe(true);
    expect(r.text).toContain("SSO is available on the Enterprise plan only.");
    expect(r.text).not.toContain("Page 12");
    expect(r.text).toContain("Plan Requests/min Seats Starter 60 1 Pro 600 10");
  });

  it("table-aware linearizes each row with its headers", () => {
    const r = parse(buildPage(24), "table");
    expect(r.text).toContain("Pro plan: Requests/min 600; Seats 10.");
    expect(r.text).toBe(referenceText());
  });

  it("scores facts per strategy: 1/6, 5/6, 6/6", () => {
    const s = (st: "naive" | "layout" | "table") => score(parse(buildPage(24), st).text).recovered;
    expect([s("naive"), s("layout"), s("table")]).toEqual([1, 5, 6]);
    expect(factFound("Pro plan: Requests/min 600;", "pro plan requests min 600")).toBe(true);
    expect(factFound("approve plan", "pro plan")).toBe(false);
  });

  it("a narrow gutter defeats column detection", () => {
    const r = parse(buildPage(8), "layout");
    expect(r.columnsDetected).toBe(false);
    expect(score(r.text).recovered).toBeLessThan(score(parse(buildPage(24), "layout").text).recovered);
  });

  it("word diff is an LCS: same + missing = reference length", () => {
    const d = diffWords("a b x c", "a b c d");
    expect(d).toEqual([{ op: "same", word: "a" }, { op: "same", word: "b" }, { op: "extra", word: "x" }, { op: "same", word: "c" }, { op: "missing", word: "d" }]);
    const sc = score(parse(buildPage(24), "table").text);
    expect(sc.wordRecall).toBe(1);
    expect(sc.extraWords).toBe(0);
  });

  it("steps reveal every segment exactly once, then diff and facts", () => {
    const { steps, result } = simulate("layout");
    expect(steps[0].id).toBe("page");
    expect(steps.at(-2)!.id).toBe("diff");
    expect(steps.at(-1)!.id).toBe("facts");
    const reads = steps.filter((s) => s.state.phase === "read");
    expect(reads[0].state.from).toBe(0);
    expect(reads.at(-1)!.state.upto).toBe(result.segments.length);
    for (let i = 1; i < reads.length; i++) expect(reads[i].state.from).toBe(reads[i - 1].state.upto);
  });
});
