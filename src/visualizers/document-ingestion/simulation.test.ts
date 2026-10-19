import { describe, expect, it } from "vitest";
import { DEFAULT_INGEST, clean, contentHash, extract, ingest, simulate, tally } from "./simulation";
import { fnv1aHex } from "./kit";

const outcome = (cfg: Partial<typeof DEFAULT_INGEST>, id: string) => ingest({ ...DEFAULT_INGEST, ...cfg }).find((d) => d.source.id === id)!.outcome;

describe("document ingestion", () => {
  it("FNV-1a matches the published test vectors", () => {
    expect(fnv1aHex("")).toBe("811c9dc5");
    expect(fnv1aHex("a")).toBe("e40c292c");
    expect(fnv1aHex("foobar")).toBe("bf9cf968");
  });

  it("extracts text from each format and cleans boilerplate", () => {
    expect(extract("html", "<html><head><title>x</title></head><body><nav>Home | Docs</nav><p>Hello <b>there</b></p><footer>© Co</footer></body></html>")).toBe("Home | Docs\nHello there\n© Co");
    expect(extract("markdown", "# Title\n\nSome **bold** text\n- item")).toBe("Title\nSome bold text\nitem");
    expect(extract("db-rows", '[{"q":"Q1?","a":"A1."}]')).toBe("Q: Q1? A: A1.");
    const c = clean("Home | Docs\nHello   there\r\nPage 2 of 9\n© Co");
    expect(c.text).toBe("Hello there");
    expect(c.removed).toEqual(["Home | Docs", "Page 2 of 9", "© Co"]);
  });

  it("drops the re-formatted copy only when hashing normalized text", () => {
    const docs = ingest(DEFAULT_INGEST);
    const a = docs.find((d) => d.source.id === "pricing")!;
    const b = docs.find((d) => d.source.id === "pricing-copy")!;
    expect(a.rawHash).not.toBe(b.rawHash);
    expect(a.normHash).toBe(b.normHash);
    expect(b.outcome).toBe("duplicate");
    expect(b.dupOf).toBe("pricing.pdf");
    expect(outcome({ hashOn: "raw" }, "pricing-copy")).toBe("queued");
  });

  it("incremental mode only re-ingests documents whose hash changed", () => {
    const docs = ingest({ ...DEFAULT_INGEST, mode: "incremental" });
    const queued = docs.filter((d) => d.outcome === "queued").map((d) => d.source.id);
    expect(queued).toEqual(["rate-limits"]);
    expect(docs.filter((d) => d.outcome === "unchanged").map((d) => d.source.id)).toEqual(["pricing", "sso", "refunds", "faq"]);
    expect(outcome({ mode: "incremental" }, "pricing-copy")).toBe("duplicate");
  });

  it("silently skipped files break the accounting identity; logged ones do not", () => {
    const logged = tally(ingest(DEFAULT_INGEST));
    expect(logged).toMatchObject({ received: 7, queued: 5, duplicate: 1, rejected: 1, lost: 0 });
    expect(logged.accounted).toBe(logged.received);
    const silent = tally(ingest({ ...DEFAULT_INGEST, unsupported: "silent" }));
    expect(silent.lost).toBe(1);
    expect(silent.accounted).toBe(silent.received - 1);
  });

  it("steps advance cards stage by stage and stop dropped ones", () => {
    const { steps, docs } = simulate(DEFAULT_INGEST);
    expect(steps.map((s) => s.id)).toEqual(["inbox", "fetch", "detect", "extract", "clean", "hash", "metadata", "queue"]);
    const dup = (i: number) => steps[i].state.docs.find((d) => d.id === "pricing-copy")!;
    expect(dup(4).col).toBe(4);
    expect(dup(7).col).toBe(5);
    expect(dup(7).stopped).toBe(true);
    expect(steps[7].state.docs.filter((d) => d.outcome === "queued")).toHaveLength(5);
    expect(docs.find((d) => d.source.id === "rate-limits")!.metadata!.content_hash).toBe(contentHash(docs.find((d) => d.source.id === "rate-limits")!.cleaned));
    // Deterministic.
    expect(JSON.stringify(simulate(DEFAULT_INGEST).steps)).toBe(JSON.stringify(steps));
  });
});
