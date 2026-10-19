import { describe, expect, it } from "vitest";
import { CATALOGUES, PHASES, REQUESTS, WEIGHTS, accuracyOnSamples, extract, scoreTools, simulateSelection, tokenize } from "./simulation";

describe("tool selection scoring", () => {
  it("tokenizes into stemmed content words", () => {
    expect(tokenize("How many vacation days do I get per year?")).toEqual(["vacation", "day", "year"]);
    expect(tokenize("What's the weather in Paris tomorrow?")).toEqual(["weather", "paris", "tomorrow"]);
  });

  it("matches a hand calculation for the well-described catalogue", () => {
    const r = scoreTools(REQUESTS[0].text, CATALOGUES.clear.tools);
    const hb = r.scores.find((s) => s.name === "search_handbook")!;
    // vacation, day, year all appear in the handbook description → overlap 3/3; query is fillable → argFit 1
    expect(hb.matched).toEqual(["vacation", "day", "year"]);
    expect(hb.total).toBeCloseTo(WEIGHTS.overlap * 1 + WEIGHTS.args * 1, 10);
    const web = r.scores.find((s) => s.name === "web_search")!;
    // "Not for: ... vacation ..." → penalty 1: 0 + 0.3 − 1
    expect(web.exclusionHits).toEqual(["vacation"]);
    expect(web.total).toBeCloseTo(-0.7, 10);
    expect(r.winner).toBe("search_handbook");
  });

  it("identical descriptions tie and list order picks the wrong tool; clarifying fixes it", () => {
    const bad = scoreTools(REQUESTS[0].text, CATALOGUES.overlapping.tools);
    expect(bad.tie).toBe(true);
    expect(bad.winner).toBe("web_search");
    expect(accuracyOnSamples("overlapping", 160).correct).toBe(3);
    expect(accuracyOnSamples("clear", 160).correct).toBe(4);
  });

  it("too many tools overflow the budget, truncate descriptions and degrade selection", () => {
    const r = scoreTools(REQUESTS[0].text, CATALOGUES.many.tools, 160);
    expect(CATALOGUES.many.tools).toHaveLength(24);
    expect(r.wordsPerTool).toBe(Math.floor(160 / 24));
    expect(r.winner).toBe("get_calendar");
    // margins collapse: weather is now a tie decided by alphabetical order
    expect(scoreTools(REQUESTS[1].text, CATALOGUES.many.tools, 160).tie).toBe(true);
    // a large enough budget removes truncation and restores the right pick
    const roomy = scoreTools(REQUESTS[0].text, CATALOGUES.many.tools, 400);
    expect(roomy.wordsPerTool).toBeNull();
    expect(roomy.winner).toBe("search_handbook");
  });

  it("extracts arguments with deterministic rules", () => {
    expect(extract("email", "Email the Q3 report to dana@acme.com")).toBe("dana@acme.com");
    expect(extract("currency_from", "Convert 250 USD to EUR")).toBe("USD");
    expect(extract("currency_to", "Convert 250 USD to EUR")).toBe("EUR");
    expect(extract("city", "weather in tokyo")).toBe("Tokyo");
    expect(extract("expression", "no numbers here")).toBeNull();
    const r = scoreTools("Convert 250 USD to EUR", CATALOGUES.clear.tools);
    expect(r.call).toEqual({ name: "convert_currency", arguments: { amount: "250", from: "USD", to: "EUR" } });
  });

  it("produces one deterministic step per phase", () => {
    const cfg = { request: REQUESTS[2].text, catalogue: "clear" as const, budget: 160 };
    const a = simulateSelection(cfg);
    expect(a.steps.map((s) => s.state.phase)).toEqual(PHASES);
    expect(simulateSelection(cfg)).toEqual(a);
    expect(a.steps.at(-1)!.title).toBe("Picked send_email: correct");
  });
});
