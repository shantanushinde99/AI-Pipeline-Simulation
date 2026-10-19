import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { fnv1aHex } from "./kit";

export type Kind = "pdf" | "html" | "markdown" | "db-rows" | "unsupported";
export type Source = { id: string; uri: string; name: string; raw: string; version: string; updatedAt: string; acl: string; connector: "file" | "web" | "db" };
export type Outcome = "queued" | "duplicate" | "unchanged" | "rejected" | "lost";

export type IngestConfig = {
  /** full = re-ingest everything; incremental = skip documents whose content hash matches the last run. */
  mode: "full" | "incremental";
  /** What happens to a file type no extractor supports. */
  unsupported: "log" | "silent";
  /** Hash the cleaned, normalized text (correct) or the raw bytes (misses re-formatted duplicates). */
  hashOn: "normalized" | "raw";
};
export const DEFAULT_INGEST: IngestConfig = { mode: "full", unsupported: "log", hashOn: "normalized" };

export const STAGES = ["inbox", "fetch", "detect", "extract", "clean", "hash", "metadata", "queue"] as const;
export type StageId = (typeof STAGES)[number];
export const STAGE_LABEL: Record<StageId, string> = {
  inbox: "sources", fetch: "fetch", detect: "detect type", extract: "extract", clean: "clean", hash: "hash+dedupe", metadata: "metadata", queue: "index queue",
};

const PRICING_RAW = "%PDF-1.7\n[page 1]\nPlans and pricing\nNorthwind Cloud offers three plans. Every plan includes unlimited projects.\nThe Pro plan costs $49 per month.\nPage 1 of 1 · © Northwind Cloud";

/** A small, fictional batch. Each source exists to exercise one branch of the pipeline. */
export const SOURCES: Source[] = [
  { id: "pricing", uri: "s3://kb/pricing.pdf", name: "pricing.pdf", raw: PRICING_RAW, version: "v3", updatedAt: "2026-09-01", acl: "public", connector: "file" },
  {
    id: "sso", uri: "https://docs.northwind.example/sso", name: "sso.html", version: "v1", updatedAt: "2026-03-30", acl: "public", connector: "web",
    raw: "<!doctype html><html><head><title>SSO</title><script>track('view')</script></head><body><nav>Home | Docs | Pricing</nav><h1>Single sign-on setup</h1><p>SSO is available on the Enterprise plan only.</p><p>Upload the identity provider metadata XML file.</p><footer>© Northwind Cloud</footer></body></html>",
  },
  { id: "refunds", uri: "git://kb/refund-policy.md", name: "refund-policy.md", version: "v2", updatedAt: "2026-06-12", acl: "public", connector: "file", raw: "# Refund policy\n\nAnnual plans can be refunded within **30 days** of purchase.\n\n- Monthly plans: *14 days*, if usage stayed under the free tier." },
  {
    id: "faq", uri: "postgres://support/faq?rows=2", name: "faq (2 rows)", version: "row-max-id 812", updatedAt: "2026-10-10", acl: "support-team", connector: "db",
    raw: '[{"q":"Can I change my plan?","a":"Yes, from Settings > Billing at any time."},{"q":"Is there a free tier?","a":"Yes, the Starter plan is $0 per month."}]',
  },
  { id: "pricing-copy", uri: "s3://kb/old/pricing (copy).pdf", name: "pricing (copy).pdf", version: "v3", updatedAt: "2026-10-12", acl: "public", connector: "file", raw: "%PDF-1.4\r\n[page 1]\r\nPlans  and pricing\r\nNorthwind Cloud offers   three plans.  Every plan includes unlimited projects.\r\nThe Pro plan costs $49 per month.\r\nPage 1 of 1 · © Northwind Cloud" },
  { id: "rate-limits", uri: "git://kb/api-rate-limits.md", name: "api-rate-limits.md", version: "v3", updatedAt: "2026-10-01", acl: "public", connector: "file", raw: "# API rate limits\n\nThe Pro plan allows **600** requests per minute per API key.\n\nAbove the limit you get HTTP 429." },
  { id: "diagram", uri: "s3://kb/architecture.vsdx", name: "architecture.vsdx", version: "v1", updatedAt: "2026-08-20", acl: "engineering", connector: "file", raw: "PK\u0003\u0004 [binary Visio drawing, 214 KB]" },
];

/** What the last ingestion run stored (uri → normalized content hash). The rate-limit page has changed since. */
const PREVIOUS_RAW: Record<string, { raw: string; version: string }> = {
  "s3://kb/pricing.pdf": { raw: PRICING_RAW, version: "v3" },
  "https://docs.northwind.example/sso": { raw: SOURCES[1].raw, version: "v1" },
  "git://kb/refund-policy.md": { raw: SOURCES[2].raw, version: "v2" },
  "postgres://support/faq?rows=2": { raw: SOURCES[3].raw, version: "row-max-id 812" },
  "git://kb/api-rate-limits.md": { raw: "# API rate limits\n\nThe Pro plan allows **300** requests per minute per API key.\n\nAbove the limit you get HTTP 429.", version: "v2" },
};

export const bytesOf = (s: string) => new TextEncoder().encode(s).length;

// ---------- the stages, as pure functions ----------
export function detect(s: Source): { kind: Kind; evidence: string } {
  if (s.raw.startsWith("%PDF")) return { kind: "pdf", evidence: "magic bytes %PDF" };
  if (/^<!doctype html|^<html/i.test(s.raw)) return { kind: "html", evidence: "<!doctype html>" };
  if (s.connector === "db") return { kind: "db-rows", evidence: "database connector" };
  if (s.raw.startsWith("PK\u0003\u0004")) return { kind: "unsupported", evidence: "zip header + .vsdx" };
  if (s.name.endsWith(".md")) return { kind: "markdown", evidence: ".md extension" };
  return { kind: "unsupported", evidence: "unknown format" };
}

/** Pull plain text out of each format. Structure is kept as line breaks. */
export function extract(kind: Kind, raw: string): string {
  switch (kind) {
    case "pdf":
      return raw.split(/\r?\n/).filter((l) => !/^%PDF|^\[page \d+\]$/.test(l)).join("\n");
    case "html":
      return raw
        .replace(/<head>[\s\S]*?<\/head>/gi, "")
        .replace(/<(script|style)[\s\S]*?<\/\1>/gi, "")
        .replace(/<\/(h\d|p|nav|footer|li|div)>/gi, "\n")
        .replace(/<[^>]+>/g, "")
        .split("\n").filter((l) => l.trim()).join("\n");
    case "markdown":
      return raw.split("\n").filter((l) => l.trim()).map((l) => l.replace(/^#+\s*|^[-*]\s+/, "").replace(/\*\*?([^*]+)\*\*?/g, "$1")).join("\n");
    case "db-rows":
      return (JSON.parse(raw) as { q: string; a: string }[]).map((r) => `Q: ${r.q} A: ${r.a}`).join("\n");
    case "unsupported":
      return "";
  }
}

export const BOILERPLATE = [/^page \d+ of \d+/i, /^©/, /^home \| /i];
/** Drop boilerplate lines (page footers, nav bars, copyright) and normalize whitespace and line endings. */
export function clean(text: string): { text: string; removed: string[] } {
  const lines = text.replace(/\r/g, "").split("\n").map((l) => l.replace(/\s+/g, " ").trim()).filter(Boolean);
  const removed = lines.filter((l) => BOILERPLATE.some((re) => re.test(l)));
  return { text: lines.filter((l) => !removed.includes(l)).join("\n"), removed };
}

/** The dedupe key: hash of the cleaned text, lower-cased, so formatting-only differences collide. */
export const contentHash = (cleaned: string) => fnv1aHex(cleaned.toLowerCase());

export type DocResult = {
  source: Source; bytes: number; kind: Kind; evidence: string;
  extracted: string; cleaned: string; removed: string[];
  rawHash: string; normHash: string; hash: string;
  outcome: Outcome; reason: string; dupOf?: string;
  /** Index into STAGES of the last stage this document reaches. */
  stopAt: number;
  metadata?: Record<string, string>;
};

const STAGE_IX = (s: StageId) => STAGES.indexOf(s);

export function ingest(cfg: IngestConfig): DocResult[] {
  const seen = new Map<string, string>();
  const prev = new Map<string, string>();
  if (cfg.mode === "incremental") {
    for (const [uri, p] of Object.entries(PREVIOUS_RAW)) {
      const s = SOURCES.find((x) => x.uri === uri)!;
      const cleaned = clean(extract(detect(s).kind, p.raw)).text;
      const h = cfg.hashOn === "raw" ? fnv1aHex(p.raw) : contentHash(cleaned);
      prev.set(uri, h);
      seen.set(h, s.name);
    }
  }
  return SOURCES.map((source) => {
    const { kind, evidence } = detect(source);
    const extracted = extract(kind, source.raw);
    const { text: cleaned, removed } = clean(extracted);
    const rawHash = fnv1aHex(source.raw);
    const normHash = contentHash(cleaned);
    const hash = cfg.hashOn === "raw" ? rawHash : normHash;
    const base = { source, bytes: bytesOf(source.raw), kind, evidence, extracted, cleaned, removed, rawHash, normHash, hash };
    if (kind === "unsupported") {
      return cfg.unsupported === "log"
        ? { ...base, outcome: "rejected" as const, reason: "no extractor for .vsdx: sent to dead-letter queue", stopAt: STAGE_IX("detect") }
        : { ...base, outcome: "lost" as const, reason: "no extractor; dropped without a log line", stopAt: STAGE_IX("detect") };
    }
    if (prev.get(source.uri) === hash) return { ...base, outcome: "unchanged" as const, reason: "same hash as last run: skip", stopAt: STAGE_IX("hash") };
    const dupOf = seen.get(hash);
    if (dupOf && dupOf !== source.name) return { ...base, outcome: "duplicate" as const, reason: `same content as ${dupOf}`, dupOf, stopAt: STAGE_IX("hash") };
    seen.set(hash, source.name);
    const metadata = {
      doc_id: source.id, source: source.uri, type: kind, version: source.version, updated_at: source.updatedAt, acl: source.acl, content_hash: hash,
    };
    const changed = prev.has(source.uri);
    return { ...base, outcome: "queued" as const, reason: changed ? "content changed since last run" : "new document", stopAt: STAGE_IX("queue"), metadata };
  });
}

export type Tally = Record<Outcome, number> & { received: number; bytes: number; accounted: number };
export function tally(docs: DocResult[]): Tally {
  const c = (o: Outcome) => docs.filter((d) => d.outcome === o).length;
  const t = { received: docs.length, bytes: docs.reduce((s, d) => s + d.bytes, 0), queued: c("queued"), duplicate: c("duplicate"), unchanged: c("unchanged"), rejected: c("rejected"), lost: c("lost") };
  return { ...t, accounted: t.queued + t.duplicate + t.unchanged + t.rejected };
}

export type DocView = { id: string; col: number; stopped: boolean; outcome: Outcome | "moving" };
export type IngestState = { stage: StageId; stageIndex: number; docs: DocView[] };

const DESCRIPTIONS: Record<StageId, string> = {
  inbox: "Seven sources are waiting: files from object storage and git, a web page, rows from a database, plus a stray copy and a binary diagram.",
  fetch: "Each connector downloads the raw bytes. Nothing is interpreted yet; the bytes counter just climbs.",
  detect: "The pipeline sniffs each file's first bytes (and the connector) to decide which extractor to use. A file no extractor supports stops here.",
  extract: "Each extractor turns its format into plain text: PDF text layers, HTML without tags and scripts, Markdown without symbols, rows as Q/A lines.",
  clean: "Boilerplate lines (page footers, nav bars, copyright) are removed and whitespace and line endings are normalized.",
  hash: "A content hash of the cleaned text is compared with the hashes already seen. Same hash = same content, so the document is dropped or skipped.",
  metadata: "Survivors get the metadata retrieval will need later: source, version, updated_at, access control list and the hash itself.",
  queue: "Documents land in the indexing queue for chunking and embedding. Every received document should be accounted for.",
};

export function simulate(cfgIn: IngestConfig): { steps: SimulationStep<IngestState>[]; docs: DocResult[]; tally: Tally } {
  const cfg = { ...DEFAULT_INGEST, ...cfgIn };
  const docs = ingest(cfg);
  const t = tally(docs);
  let n = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id: `e${n++}`, type: "log", message, severity, tex });
  const steps = STAGES.map((stage, si): SimulationStep<IngestState> => {
    const views: DocView[] = docs.map((d) => ({ id: d.source.id, col: Math.min(si, d.stopAt), stopped: si >= d.stopAt && d.outcome !== "queued", outcome: si >= d.stopAt ? d.outcome : "moving" }));
    const events: SimulationEvent[] = [];
    if (stage === "inbox") events.push(ev(`${docs.length} sources discovered (mode: ${cfg.mode})`));
    if (stage === "fetch") events.push(ev(`fetched ${t.bytes} bytes from ${docs.length} sources`));
    if (stage === "detect") {
      for (const d of docs) {
        if (d.outcome === "rejected") events.push(ev(`${d.source.name}: unsupported type (${d.evidence}) → dead-letter queue`, "warning"));
        // A silent skip produces no event at all: that absence is the failure.
      }
      events.push(ev(`${docs.filter((d) => d.kind !== "unsupported").length} documents routed to an extractor`));
    }
    if (stage === "extract") events.push(ev(`extracted ${docs.reduce((s, d) => s + d.extracted.length, 0)} characters of text`));
    if (stage === "clean") events.push(ev(`removed ${docs.reduce((s, d) => s + d.removed.length, 0)} boilerplate lines`));
    if (stage === "hash") {
      for (const d of docs) {
        if (d.outcome === "duplicate") events.push(ev(`${d.source.name}: hash ${d.hash} already seen (${d.dupOf}) → dropped`, "warning"));
        if (d.outcome === "unchanged") events.push(ev(`${d.source.name}: hash ${d.hash} unchanged → skipped`));
      }
      if (cfg.hashOn === "raw" && docs.some((d) => d.source.id === "pricing-copy" && d.outcome !== "duplicate")) events.push(ev("raw-bytes hash: pricing (copy).pdf differs only in whitespace, but its hash is different, so it slips through", "warning"));
    }
    if (stage === "metadata") events.push(ev(`attached metadata to ${t.queued} documents`));
    if (stage === "queue") {
      events.push(ev(`queued ${t.queued} documents for indexing`));
      events.push(ev(`received ${t.received} = queued ${t.queued} + duplicate ${t.duplicate} + unchanged ${t.unchanged} + rejected ${t.rejected}${t.lost ? ` + unaccounted ${t.lost}` : ""}`, t.lost ? "error" : "info"));
    }
    return { id: stage, title: si === 0 ? "Sources arrive" : `${STAGE_LABEL[stage][0].toUpperCase()}${STAGE_LABEL[stage].slice(1)}`, description: DESCRIPTIONS[stage], state: { stage, stageIndex: si, docs: views }, events };
  });
  return { steps, docs, tally: t };
}
