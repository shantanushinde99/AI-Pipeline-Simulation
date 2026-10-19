import { dot } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { CORPUS, RATE_LIMITS_V2, type SourceDoc } from "../rag/corpus";
import { chunk, embed, parse, tokens, type EmbedModel } from "../rag/simulation";

export type CaseId = "vocab" | "split" | "stale" | "dup" | "filter" | "embed";
export type PoolChunk = { id: string; docId: string; text: string; lang: string };
export type Row = { key: string; id: string; text: string; score: number; relevant: boolean };
export type Ranking = {
  label: string; query: string; model: EmbedModel; poolSize: number;
  rows: Row[]; // ranked, best first (excluded chunks are not in here)
  excluded: { key: string; id: string; text: string; reason: string; relevant: boolean }[];
  relRank: number | null; relStatus: "ranked" | "missing" | "excluded"; relScore: number | null;
};
export type Diagnostic = { facts: [string, string, boolean][]; tokens?: { query: string[]; chunk: string[]; shared: string[]; chunkId: string } };
export type FailureCase = {
  id: CaseId; title: string; fact: string; fault: string; fixLabel: string;
  symptom: string; diagnosticText: string; fixText: string;
  healthy: Ranking; broken: Ranking; fixed: Ranking; diag: Diagnostic;
};

const CHUNK = 30, OVERLAP = 10;
/** Every doc is ingested with lang "en-US", except data-retention.md, which an older pipeline tagged "en". */
const LANG: Record<string, string> = { retention: "en" };
const poolOf = (docs: SourceDoc[] = CORPUS, size = CHUNK, overlap = OVERLAP): PoolChunk[] =>
  docs.flatMap((d) => chunk(parse(d, "linearize"), size, overlap).map((c) => ({ id: c.id, docId: d.id, text: c.text, lang: LANG[d.id] ?? "en-US" })));

/** Rank a pool by dot product of unit vectors (= cosine). Ties keep pool order. Index vectors are always embed-v1. */
export function rank(label: string, query: string, pool: PoolChunk[], fact: string, opts: { model?: EmbedModel; filter?: (c: PoolChunk) => string | null; dedupe?: number } = {}): Ranking {
  const model = opts.model ?? "embed-v1";
  const q = embed(query, model);
  const excluded: Ranking["excluded"] = [];
  const key = (c: PoolChunk) => `${c.id}::${c.text.slice(0, 24)}`;
  const isRel = (c: PoolChunk) => c.text.includes(fact);
  const kept = pool.filter((c) => {
    const reason = opts.filter?.(c) ?? null;
    if (reason) excluded.push({ key: key(c), id: c.id, text: c.text, reason, relevant: isRel(c) });
    return !reason;
  });
  let scored = kept.map((c, i) => ({ c, i, v: embed(c.text, "embed-v1") })).map((x) => ({ ...x, s: dot(q, x.v) })).sort((a, b) => b.s - a.s || a.i - b.i);
  if (opts.dedupe !== undefined) {
    const out: typeof scored = [];
    for (const x of scored) {
      const dupOf = out.find((y) => dot(x.v, y.v) >= opts.dedupe!);
      if (dupOf) excluded.push({ key: key(x.c), id: x.c.id, text: x.c.text, reason: `near-duplicate of ${dupOf.c.id} (cos ${dot(x.v, dupOf.v).toFixed(2)})`, relevant: isRel(x.c) });
      else out.push(x);
    }
    scored = out;
  }
  const rows: Row[] = scored.map(({ c, s }) => ({ key: key(c), id: c.id, text: c.text, score: s, relevant: isRel(c) }));
  const r = rows.findIndex((x) => x.relevant);
  const ex = excluded.find((x) => x.relevant);
  return {
    label, query, model, poolSize: pool.length, rows, excluded,
    relRank: r >= 0 ? r + 1 : null, relStatus: r >= 0 ? "ranked" : ex ? "excluded" : "missing", relScore: r >= 0 ? rows[r].score : null,
  };
}

const DUP_FILES = ["rate-limits (copy).md", "kb-export/rate-limits.md", "rate-limits-2025.md"];

export function buildCases(): FailureCase[] {
  const P = poolOf();
  const cases: FailureCase[] = [];

  {
    const fact = "30 days of purchase";
    const user = "Can I get my money back on a yearly plan after three weeks?";
    const rewritten = "Can I get a refund on an annual plan after three weeks?";
    const broken = rank("user's wording", user, P, fact);
    const relChunk = P.find((c) => c.text.includes(fact))!;
    const qt = [...new Set(tokens(user))], ct = [...new Set(tokens(relChunk.text))];
    cases.push({
      id: "vocab", title: "Vocabulary mismatch", fact, fault: "user says 'money back' / 'yearly'", fixLabel: "query rewrite with a domain synonym map",
      symptom: "Users get answers about pricing when they ask about refunds in their own words.",
      diagnosticText: "The query and the refund chunk share almost no terms: 'money back' ≠ 'refund', 'yearly' ≠ 'annual'.",
      fixText: "Rewrite/expand the query (synonyms, LLM rewrite) or add hybrid keyword + dense retrieval with a domain thesaurus.",
      healthy: rank("docs' wording", rewritten, P, fact), broken, fixed: rank("rewritten query", rewritten, P, fact),
      diag: { facts: [["user query", user, true], ["rewritten", rewritten, false]], tokens: { query: qt, chunk: ct, shared: qt.filter((t) => ct.includes(t)), chunkId: relChunk.id } },
    });
  }
  {
    const fact = "metadata XML";
    const q = "How do I enable SSO?";
    const small = poolOf(CORPUS, 15, 0);
    const brokenRel = small.find((c) => c.text.includes(fact))!;
    const prev = small.find((c) => c.docId === brokenRel.docId && small.indexOf(c) === small.indexOf(brokenRel) - 1)!;
    cases.push({
      id: "split", title: "Chunk boundary split", fact, fault: "chunk size 15, no overlap", fixLabel: "chunk size 30 with 10-word overlap",
      symptom: "'How do I enable SSO?' returns the plan-availability sentence but never the setup steps.",
      diagnosticText: "The instruction sentence is cut in two: 'enable SSO' ends one chunk, 'metadata XML file' starts the next, which shares no words with the query.",
      fixText: "Chunk on sentence boundaries, add overlap, or retrieve neighbouring chunks (parent/child retrieval).",
      healthy: rank("30 words, overlap 10", q, P, fact), broken: rank("15 words, no overlap", q, small, fact), fixed: rank("30 words, overlap 10", q, P, fact),
      diag: { facts: [["chunk before the cut", `${prev.id}: “…${prev.text.split(" ").slice(-8).join(" ")}”`, false], ["chunk with the fact", `${brokenRel.id}: “${brokenRel.text.split(" ").slice(0, 10).join(" ")}…”`, true]], tokens: { query: [...new Set(tokens(q))], chunk: [...new Set(tokens(brokenRel.text))], shared: [...new Set(tokens(q))].filter((t) => tokens(brokenRel.text).includes(t)), chunkId: brokenRel.id } },
    });
  }
  {
    const fact = "600 requests";
    const q = "What is the API rate limit on the Pro plan?";
    const stale = poolOf(CORPUS.map((d) => (d.id === RATE_LIMITS_V2.id ? RATE_LIMITS_V2 : d)));
    cases.push({
      id: "stale", title: "Stale index", fact, fault: "index snapshot built 2026-04-20 (rate-limits v2)", fixLabel: "re-index from current docs (v3)",
      symptom: "Answers quote 300 requests/minute; the docs say 600.",
      diagnosticText: "The top hit looks right (same title, same words) but comes from v2. The v3 chunk with '600' was never indexed.",
      fixText: "Re-index on document change; store doc version/updatedAt with every chunk and alert when the index is older than the source.",
      healthy: rank("index from v3", q, P, fact), broken: rank("stale index (v2)", q, stale, fact), fixed: rank("re-indexed (v3)", q, P, fact),
      diag: { facts: [["index built", "2026-04-20", true], ["api-rate-limits.md in index", `${RATE_LIMITS_V2.version} (${RATE_LIMITS_V2.updatedAt}) → “300 requests per minute”`, true], ["api-rate-limits.md in source", `${CORPUS[4].version} (${CORPUS[4].updatedAt}) → “600 requests per minute”`, false]] },
    });
  }
  {
    const fact = "$49";
    const q = "What does the Pro plan cost per month and what is the API rate limit?";
    const src = P.find((c) => c.id === "rate-limits#0")!;
    const dups: PoolChunk[] = DUP_FILES.map((f, i) => ({ id: `${f}#0`, docId: f, text: i === 0 ? src.text : `${src.text} (mirrored)`, lang: "en-US" }));
    const withDups = [...P, ...dups];
    const broken = rank("index with 3 mirrored copies", q, withDups, fact);
    const top = broken.rows.slice(0, 3);
    cases.push({
      id: "dup", title: "Near-duplicate crowding", fact, fault: "3 mirrored copies of the rate-limit page indexed", fixLabel: "drop results with cosine ≥ 0.95 to a better hit",
      symptom: "Two-part questions get only half answered: the context is the same page four times.",
      diagnosticText: `The top ${top.length} results are copies of one page (pairwise cosine ≈ ${dot(embed(top[0].text, "embed-v1"), embed(top[1].text, "embed-v1")).toFixed(2)}); distinct evidence is pushed below k.`,
      fixText: "Deduplicate at ingestion (hash/MinHash), and diversify results at query time (near-duplicate filter or MMR).",
      healthy: rank("deduplicated index", q, P, fact), broken, fixed: rank("near-duplicate filter", q, withDups, fact, { dedupe: 0.95 }),
      diag: { facts: [...top.map((r, i): [string, string, boolean] => [`top-${i + 1}`, r.id, r.id !== "rate-limits#0"]), ["distinct source pages in top-3", String(new Set(top.map((r) => r.text.replace(" (mirrored)", ""))).size), true]] },
    });
  }
  {
    const fact = "365 days";
    const q = "How long are request logs kept on the Enterprise plan?";
    cases.push({
      id: "filter", title: "Wrong metadata filter", fact, fault: "filter lang == \"en-US\"", fixLabel: "normalize lang tags; filter lang starts with \"en\"",
      symptom: "Retention questions return 'I couldn't find this' even though the page exists.",
      diagnosticText: "The filter runs before ranking. data-retention.md was ingested with lang = \"en\", so its chunks never reach the scorer.",
      fixText: "Validate metadata at ingestion (enums, normalization), log how many candidates a filter removes, and alert when a filter empties a doc set.",
      healthy: rank("no filter", q, P, fact), broken: rank("filter lang == en-US", q, P, fact, { filter: (c) => (c.lang === "en-US" ? null : `lang = "${c.lang}" ≠ "en-US"`) }),
      fixed: rank("filter lang ^= en", q, P, fact, { filter: (c) => (c.lang.startsWith("en") ? null : `lang = "${c.lang}"`) }),
      diag: { facts: [["filter", "lang == \"en-US\"", true], ...CORPUS.map((d): [string, string, boolean] => [d.file, `lang = "${LANG[d.id] ?? "en-US"}" → ${(LANG[d.id] ?? "en-US") === "en-US" ? "kept" : "filtered out"}`, (LANG[d.id] ?? "en-US") !== "en-US"])] },
    });
  }
  {
    const fact = "365 days";
    const q = "How long are request logs kept on the Enterprise plan?";
    const healthy = rank("query on embed-v1", q, P, fact);
    const broken = rank("query on embed-v2", q, P, fact, { model: "embed-v2" });
    cases.push({
      id: "embed", title: "Embedding model mismatch", fact, fault: "query encoder upgraded to embed-v2, index still embed-v1", fixLabel: "embed queries with the index's model (or re-index with v2)",
      symptom: "Every query suddenly returns loosely related chunks with low similarity scores.",
      diagnosticText: "Query and chunk vectors live in different spaces: the same words hash to different dimensions, so similarity is noise.",
      fixText: "Store the embedding model id with the index and assert it matches the query encoder; re-index before switching models.",
      healthy, broken, fixed: rank("query back on embed-v1", q, P, fact),
      diag: { facts: [["query model", "embed-v2", true], ["index model", "embed-v1", false], ["top-1 similarity", `${healthy.rows[0].score.toFixed(2)} → ${broken.rows[0].score.toFixed(2)}`, true], ["similarity to the needed chunk", `${healthy.relScore!.toFixed(2)} → ${broken.relScore!.toFixed(2)}`, true]] },
    });
  }
  return cases;
}

export const CASES = buildCases();

export type RFPhase = "healthy" | "broken" | "diagnose" | "fixed" | "summary";
export type RFState = { phase: RFPhase; ranking: "healthy" | "broken" | "fixed" };

export function simulate(caseId: CaseId, topK: number) {
  const k = Math.max(1, Math.min(5, Math.round(topK)));
  const c = CASES.find((x) => x.id === caseId) ?? CASES[0];
  const where = (r: Ranking) => (r.relStatus === "ranked" ? `#${r.relRank} of ${r.rows.length}` : r.relStatus === "excluded" ? "removed before ranking" : "not in the index");
  const inK = (r: Ranking) => r.relRank !== null && r.relRank <= k;
  let n = 0;
  const ev = (message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id: `e${n++}`, type: "retrieval", message, severity, tex });
  const scoreTex = (r: Ranking) => (r.relScore === null ? undefined : `s(q, c^{*}) = \\hat q \\cdot \\hat c^{*} = ${r.relScore.toFixed(3)}`);
  const steps: SimulationStep<RFState>[] = [
    {
      id: "healthy", title: "Healthy ranking",
      description: `Query: “${c.healthy.query}”. The chunk containing “${c.fact}” ranks ${where(c.healthy)}, ${inK(c.healthy) ? "inside" : "outside"} the top-${k}.`,
      state: { phase: "healthy", ranking: "healthy" },
      events: [ev(`${c.healthy.rows.length} chunks scored by cosine similarity`), ev(`needed chunk at ${where(c.healthy)}`, undefined, scoreTex(c.healthy))],
    },
    {
      id: "broken", title: `Fault: ${c.title}`,
      description: `${c.fault}. The needed chunk is now ${where(c.broken)}${inK(c.broken) ? "" : `, so it is out of the top-${k} and the generator never sees it`}.`,
      state: { phase: "broken", ranking: "broken" },
      events: [ev(`fault: ${c.fault}`, "warning"), ev(`needed chunk: ${where(c.healthy)} → ${where(c.broken)}`, inK(c.broken) ? "warning" : "error", scoreTex(c.broken)),
        ...c.broken.excluded.filter((x) => x.relevant).map((x) => ev(`${x.id} excluded: ${x.reason}`, "error"))],
    },
    {
      id: "diagnose", title: "Diagnose",
      description: c.diagnosticText,
      state: { phase: "diagnose", ranking: "broken" },
      events: c.diag.facts.map(([kk, v, bad]) => ev(`${kk}: ${v}`, bad ? "warning" : undefined)),
    },
    {
      id: "fixed", title: `Fix: ${c.fixLabel}`,
      description: `After the fix the needed chunk is back at ${where(c.fixed)}${inK(c.fixed) ? `, inside the top-${k}` : `, still outside the top-${k}: raise k or combine fixes`}.`,
      state: { phase: "fixed", ranking: "fixed" },
      events: [ev(`fix: ${c.fixLabel}`), ev(`needed chunk: ${where(c.broken)} → ${where(c.fixed)}`, inK(c.fixed) ? undefined : "warning", scoreTex(c.fixed))],
    },
    {
      id: "summary", title: "Symptom → diagnostic → fix",
      description: `${c.symptom} Diagnostic: ${c.diagnosticText} Fix: ${c.fixText}`,
      state: { phase: "summary", ranking: "fixed" },
      events: [ev(`rank path: ${where(c.healthy)} → ${where(c.broken)} → ${where(c.fixed)}`)],
    },
  ];
  return { c, k, steps, inK, where };
}
