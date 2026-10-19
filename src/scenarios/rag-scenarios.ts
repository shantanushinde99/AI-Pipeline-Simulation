import { CORPUS, QUESTIONS } from "../visualizers/rag/corpus";
import { DEFAULT_RAG, TEMPLATES, parse, runGoldenSet, runRag, splitSentences, type RagConfig, type RagRun } from "../visualizers/rag/simulation";
import { RAG_EDGES, STAGES } from "../visualizers/rag/stages";
import type { Evidence, HealthMetric, Outcome, Scenario, SystemType } from "./types";
import type { Difficulty } from "../types/topic";

const pct = (v: number) => `${Math.round(v * 100)}%`;

export const RAG_ARCH: Scenario["architecture"] = {
  nodes: STAGES.map((s) => ({
    id: s.id, label: s.label, Icon: s.Icon, x: s.x, y: s.y,
    match: (span) => {
      const ids: Record<string, (id: string) => boolean> = {
        docs: () => false, parse: (id) => id.startsWith("parse:"), chunk: (id) => id === "chunk", embed: (id) => id === "embed.index", index: (id) => id === "index.write",
        query: (id) => id === "rag.query", embedq: (id) => id === "embed.query", retrieve: (id) => id === "retrieve", rerank: (id) => id === "rerank",
        context: (id) => id === "context", prompt: (id) => id === "prompt", generate: (id) => id === "generate", answer: () => false,
      };
      return ids[s.id](span.id);
    },
  })),
  edges: RAG_EDGES,
};

type Golden = ReturnType<typeof runGoldenSet>;
export type RagAction = "golden" | "answer" | "retrieval" | "context" | "parser" | "chunks" | "index" | "models" | "prompt" | "citation" | "faithfulness" | "dashboards" | "deploys";

const ACTION_LABEL: Record<RagAction, string> = {
  golden: "Run the golden-set evaluation",
  answer: "Compare the answer with the source of truth",
  retrieval: "Inspect retrieval results",
  context: "Inspect the assembled context",
  parser: "Check parser output for the source document",
  chunks: "Inspect chunk boundaries in the source document",
  index: "Check index metadata and document versions",
  models: "Compare query and index embedding models",
  prompt: "Inspect the prompt template",
  citation: "Verify citations against passages",
  faithfulness: "Run a faithfulness check",
  dashboards: "Check latency and error dashboards",
  deploys: "Read the deploy log",
};

function evidenceFor(a: RagAction, r: RagRun, g: Golden, deploys: string[]): Omit<Evidence, "relevant"> {
  const q = r.question;
  const fact = q.expect.toLowerCase();
  const src = CORPUS.find((d) => d.id === q.sourceDoc)!;
  const lines = ((): string[] => {
    switch (a) {
      case "golden":
        return [`answer accuracy ${pct(g.accuracy)}, retrieval recall@k ${pct(g.retrievalRecall)}, faithfulness ${pct(g.faithfulness)}, citation accuracy ${pct(g.citationAccuracy)}`,
          ...g.runs.map((x) => `${x.evaluation.ok ? "PASS" : "FAIL"} · ${x.question.text}`)];
      case "answer": {
        const truth = splitSentences(parse(src, "linearize").text).find((s) => s.toLowerCase().includes(fact)) ?? "";
        return [`Answer given: “${r.answer.text}”`, `Source of truth (${src.file} ${src.version}, ${src.updatedAt}): “${truth}”`];
      }
      case "retrieval":
        return [`query model ${r.cfg.queryModel}; top-${r.cfg.topK}:`, ...r.retrieved.map((h) => `#${h.rank} ${h.entry.id} cos=${h.score.toFixed(3)}${h.entry.text.toLowerCase().includes(fact) ? "  ← contains reference fact" : ""}`),
          r.evaluation.retrievalHit ? "A retrieved chunk contains the reference fact." : `No retrieved chunk contains “${q.expect}”.`];
      case "context":
        return [`${r.contextTokens}/${r.cfg.contextBudget} tokens, ${r.context.length} passage(s):`, ...r.context.map((c) => `[${c.label}] ${c.entry.id} · ${c.tokens} tokens${c.truncated ? ` · TRUNCATED (${c.droppedTokens} cut)` : ""}${c.text.toLowerCase().includes(fact) ? " · contains reference fact" : ""}`),
          r.evaluation.contextHit ? "The reference fact is present in the context." : "The reference fact is NOT present in the context."];
      case "parser": {
        const p = r.index.parsed.find((x) => x.docId === src.id)!;
        const ev = r.spans.find((s) => s.id === `parse:${src.id}`)?.events ?? [];
        return [`${p.file}: ${p.blocks} source blocks, ${p.tablesFound} table(s) found, ${p.tablesDropped} dropped, mode=${r.cfg.tables}`, ...ev.map((e) => `log: ${e.message}`), `Parsed text: “${p.text.slice(0, 220)}${p.text.length > 220 ? "…" : ""}”`,
          p.text.toLowerCase().includes(fact) ? "Reference fact present in parsed text." : `Reference fact “${q.expect}” is absent from parsed text.`];
      }
      case "chunks":
        return [`chunk_size=${r.cfg.chunkSize}, overlap=${r.cfg.overlap}`, ...r.index.chunks.filter((c) => c.docId === src.id).map((c) => `${c.id} [${c.startWord}–${c.endWord}]: “${c.text}”`)];
      case "index":
        return [`index model ${r.index.model}, built ${r.index.builtAt}`, ...r.index.sources.map((s, i) => `${s.file}: indexed ${s.version} (${s.updatedAt}) · source of truth ${CORPUS[i].version} (${CORPUS[i].updatedAt})${s.version !== CORPUS[i].version ? "  ← OUT OF DATE" : ""}`)];
      case "models":
        return [`query encoder: ${r.cfg.queryModel}`, `index vectors: ${r.index.model}`, `mean top-1 similarity on golden set: ${g.meanTop1.toFixed(3)} (healthy baseline ${runGoldenSet(DEFAULT_RAG).meanTop1.toFixed(3)})`];
      case "prompt":
        return [`template: ${r.cfg.promptTemplate}`, `system text: “${TEMPLATES[r.cfg.promptTemplate]}”`];
      case "citation":
        return r.answer.citations.length === 0 ? ["The answer has no citations."] : r.answer.citations.map((c) => {
          const item = r.context.find((x) => x.label === c.label);
          return `[${c.label}] → ${c.chunkId ?? "no such passage"}: ${item?.text.includes(r.answer.sentence) ? "contains the cited sentence" : "does NOT contain the cited sentence"}`;
        }).concat(r.context.filter((x) => x.text.includes(r.answer.sentence) && r.answer.sentence).map((x) => `The cited sentence actually appears in passage [${x.label}] (${x.entry.id}).`));
      case "faithfulness":
        return [r.answer.source === "none" ? "No answer was given, nothing to check." : r.evaluation.faithful ? "Answer sentence appears verbatim in the context: faithful." : "Answer sentence does NOT appear in the context: unfaithful (unsupported claim).", `Golden-set faithfulness: ${pct(g.faithfulness)}`];
      case "dashboards":
        return [`request success rate ${pct(g.successRate)}, HTTP 5xx 0%`, `p95 latency ${g.p95LatencyMs.toFixed(0)} ms (SLO 1500 ms)`, "No alerts fired in the last 24 h."];
      case "deploys":
        return deploys;
    }
  })();
  const spanId = { retrieval: "retrieve", context: "context", parser: `parse:${q.sourceDoc}`, prompt: "prompt", models: "embed.query", chunks: "chunk" }[a as string];
  return { id: a, action: ACTION_LABEL[a], title: ACTION_LABEL[a], lines, spanId };
}

type RagScenarioDef = {
  id: string; title: string; difficulty: Difficulty; system?: SystemType; symptom: string; objective: string; incident: string;
  questionId: string; fault: Partial<RagConfig>; actions: RagAction[]; relevant: RagAction[]; deploys: string[];
  hypotheses: Scenario["hypotheses"]; fixes: (Scenario["fixes"][number] & { cfg: Partial<RagConfig> })[]; verifications: Scenario["verifications"]; rootCause: string;
};

export function ragScenario(d: RagScenarioDef): Scenario {
  const run = (fixId?: string): Outcome => {
    const fix = d.fixes.find((f) => f.id === fixId);
    const cfg: RagConfig = { ...DEFAULT_RAG, ...d.fault, ...(fix?.cfg ?? {}) };
    const r = runRag(cfg, d.questionId);
    const g = runGoldenSet(cfg);
    const metrics: HealthMetric[] = [
      { label: "Success rate", value: pct(g.successRate), status: "ok", kind: "health" },
      { label: "p95 latency", value: `${g.p95LatencyMs.toFixed(0)} ms`, status: g.p95LatencyMs < 1500 ? "ok" : "warning", kind: "health" },
      { label: "Error rate", value: "0%", status: "ok", kind: "health" },
      { label: "Golden-set accuracy", value: pct(g.accuracy), status: g.accuracy === 1 ? "ok" : g.accuracy >= 0.8 ? "warning" : "error", kind: "quality" },
      { label: "Faithfulness", value: pct(g.faithfulness), status: g.faithfulness === 1 ? "ok" : "error", kind: "quality" },
      { label: "Citation accuracy", value: pct(g.citationAccuracy), status: g.citationAccuracy === 1 ? "ok" : "error", kind: "quality" },
    ];
    return {
      spans: r.spans,
      metrics,
      transcript: [
        { label: "User", text: r.question.text },
        { label: "Assistant", text: r.answer.text, tone: r.evaluation.ok ? "ok" : "err" },
      ],
      resolved: r.evaluation.ok && g.accuracy === 1 && g.faithfulness === 1 && g.citationAccuracy === 1,
      evidence: d.actions.map((a) => ({ ...evidenceFor(a, r, g, d.deploys), relevant: d.relevant.includes(a) })),
    };
  };
  return {
    id: d.id, title: d.title, system: d.system ?? "RAG", difficulty: d.difficulty, symptom: d.symptom, objective: d.objective, incident: d.incident,
    relatedTopic: "rag-pipeline", architecture: RAG_ARCH, run, hypotheses: d.hypotheses, verifications: d.verifications, rootCause: d.rootCause,
    fixes: d.fixes.map(({ id, label, explanation }) => ({ id, label, explanation })),
  };
}

const q = (id: string) => QUESTIONS.find((x) => x.id === id)!.text;

export const RAG_SCENARIOS: Scenario[] = [
  ragScenario({
    id: "rag-parser-table", title: "Pricing answers contain no prices", difficulty: "Beginner",
    symptom: "Plan-price questions get vague answers", objective: "Trace a missing fact back to the ingestion stage that dropped it.",
    incident: `Support escalated 14 chats this week: asked “${q("q-price")}”, the assistant replies with something about annual billing and never states a price. The pricing PDF was re-ingested on Monday.`,
    questionId: "q-price", fault: { tables: "drop" },
    actions: ["golden", "answer", "retrieval", "context", "parser", "chunks", "index", "dashboards", "deploys"],
    relevant: ["parser", "chunks", "retrieval", "deploys"],
    deploys: ["Mon 09:12 ingestion-service 2.0.0: PDF parser switched to 'fast text-only' mode (3× faster)", "Mon 09:40 ingestion job re-ran for pricing.pdf: success", "Tue 15:05 web-chat 4.2: new typing indicator"],
    hypotheses: [
      { id: "h-parser", text: "The parser dropped the pricing table during ingestion, so prices never reached the index.", isRootCause: true, evidenceIds: ["parser", "chunks", "deploys"], explanation: "Parser output shows 1 table found, 1 dropped, and no chunk contains “$49”. The ingestion upgrade switched to text-only mode." },
      { id: "h-topk", text: "top-k is too small, so the chunk with prices isn't retrieved.", isRootCause: false, evidenceIds: ["retrieval", "chunks"], explanation: "pricing#0 is already ranked #1. No chunk anywhere in the index contains a price, so a larger k cannot help." },
      { id: "h-llm", text: "The generator is ignoring the context and hallucinating.", isRootCause: false, evidenceIds: ["context"], explanation: "The context contains no price at all; the generator answered from the best sentence it had. It's a symptom of missing input." },
      { id: "h-stale", text: "The index holds an old version of the pricing PDF.", isRootCause: false, evidenceIds: ["index"], explanation: "Index metadata shows pricing.pdf v3, the current version." },
    ],
    fixes: [
      { id: "f-parser", label: "Switch the parser to table-aware mode (linearize tables) and re-index", explanation: "Restores the table rows as text, so the price is chunked, embedded and retrievable.", cfg: { tables: "linearize" } },
      { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "Retrieves more of the same content; none of it contains prices.", cfg: { topK: 8, contextChunks: 5 } },
      { id: "f-reindex", label: "Re-run the ingestion job for pricing.pdf", explanation: "Re-running the same text-only parser reproduces the same loss.", cfg: {} },
    ],
    verifications: [
      { id: "v-golden", label: "Re-run the golden set and add an ingestion test asserting known table values (e.g. “$49”) exist in the index", sound: true, explanation: "Checks both the mechanism (value indexed) and the outcome, and guards against the next parser change." },
      { id: "v-once", label: "Ask the pricing question once in staging", sound: false, explanation: "One manual check doesn't cover other tables or prevent regressions." },
      { id: "v-job", label: "Confirm the ingestion job finishes without errors", sound: false, explanation: "It already finished without errors. Dropping tables was silent." },
    ],
    rootCause: "The ingestion upgrade switched the PDF parser to text-only mode, which silently skips tables. The pricing table, the only place prices appear, never reached the index.",
  }),
  ragScenario({
    id: "rag-chunk-split", title: "SSO instructions stop mid-sentence", difficulty: "Intermediate",
    symptom: "Answer ends with “…uploads the identity”", objective: "See how fixed-size chunking can cut an instruction in half.",
    incident: `Enterprise admins report that for “${q("q-sso")}” the assistant's instructions stop halfway through a sentence and never mention which file to upload.`,
    questionId: "q-sso", fault: { chunkSize: 20, overlap: 0 },
    actions: ["golden", "answer", "retrieval", "context", "chunks", "parser", "index", "deploys"],
    relevant: ["chunks", "retrieval", "deploys"],
    deploys: ["Thu 10:20 indexer-config: chunk_size 80 → 20, overlap 15 → 0 (cut embedding cost by 40%)", "Thu 10:45 full re-index: success", "Fri 08:00 docs: no changes to sso-setup.md since March"],
    hypotheses: [
      { id: "h-chunk", text: "A chunk boundary splits the instruction; the half naming the metadata file isn't retrieved.", isRootCause: true, evidenceIds: ["chunks", "retrieval", "deploys"], explanation: "With 20-word chunks and no overlap, sso#1 ends at “…uploads the identity” and sso#2 starts “provider metadata XML file”. sso#2 shares no query terms and isn't retrieved." },
      { id: "h-parser", text: "The parser truncated the SSO document.", isRootCause: false, evidenceIds: ["parser"], explanation: "Parsed text contains the full instruction including “metadata XML file”." },
      { id: "h-budget", text: "The context budget cuts the answer short.", isRootCause: false, evidenceIds: ["context"], explanation: "Context uses far less than the 300-token budget; nothing is truncated." },
      { id: "h-gen", text: "The generator has an output length limit.", isRootCause: false, evidenceIds: ["context", "answer"], explanation: "The answer ends exactly where the chunk text ends: the generator reproduced its whole input sentence." },
    ],
    fixes: [
      { id: "f-chunk", label: "Restore chunk size 80 with 15-word overlap and re-index", explanation: "The instruction fits in one chunk again; overlap protects sentences near boundaries.", cfg: { chunkSize: 80, overlap: 15 } },
      { id: "f-budget", label: "Raise the context budget to 400 tokens", explanation: "The budget wasn't the constraint; the missing half was never retrieved.", cfg: { contextBudget: 400 } },
      { id: "f-topk", label: "Raise top-k to 8", explanation: "sso#2 may be retrieved but the reranker and generator still prefer the chunk matching “enable SSO”.", cfg: { topK: 8 } },
    ],
    verifications: [
      { id: "v-golden", label: "Re-run the golden set and add a check that no reference answer sentence spans a chunk boundary", sound: true, explanation: "Directly tests the failure mechanism across all documents." },
      { id: "v-count", label: "Confirm the number of chunks went down", sound: false, explanation: "Chunk count says nothing about whether answers are intact." },
      { id: "v-cost", label: "Check the embedding cost dashboard", sound: false, explanation: "Cost is why the change was made, not evidence of correctness." },
    ],
    rootCause: "A cost change cut chunk size to 20 words with no overlap. Fixed windows ignore sentence boundaries, so the SSO instruction was split and the half naming the file had nothing in common with the query.",
  }),
  ragScenario({
    id: "rag-stale-index", title: "Assistant quotes last year's rate limit", difficulty: "Beginner",
    symptom: "Says 300 req/min; docs say 600", objective: "Tell a faithful-but-wrong answer apart from a hallucination.",
    incident: `Pro customers asking “${q("q-rate")}” are told 300 requests per minute. The public docs say 600. Several customers throttled their own clients unnecessarily.`,
    questionId: "q-rate", fault: { indexSnapshot: "stale" },
    actions: ["golden", "answer", "retrieval", "context", "faithfulness", "index", "prompt", "dashboards", "deploys"],
    relevant: ["answer", "index", "faithfulness", "deploys"],
    deploys: ["Oct 01 docs-site: published api-rate-limits.md v3 (Pro 300 → 600 req/min)", "Sep 12 platform: migrated cron jobs to new scheduler (index-rebuild job not migrated)", "Apr 20 index-rebuild: last successful run"],
    hypotheses: [
      { id: "h-stale", text: "The index is a stale snapshot built before the rate-limit page was updated.", isRootCause: true, evidenceIds: ["index", "answer", "deploys"], explanation: "Index metadata shows api-rate-limits.md v2 (April) while the source is v3 (October). The rebuild job wasn't migrated." },
      { id: "h-halluc", text: "The model hallucinated the number 300.", isRootCause: false, evidenceIds: ["faithfulness", "context"], explanation: "The answer sentence appears verbatim in the context. It is faithful to what was retrieved." },
      { id: "h-retrieval", text: "Retrieval picked the wrong document.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "Top-1 is the rate-limits chunk, the right document, just an old version of it." },
      { id: "h-prompt", text: "A prompt template change altered answers.", isRootCause: false, evidenceIds: ["prompt"], explanation: "Template is grounded-v1, unchanged." },
    ],
    fixes: [
      { id: "f-reindex", label: "Rebuild the index from current sources and restore the scheduled rebuild", explanation: "The index gets v3 and stays fresh.", cfg: { indexSnapshot: "current" } },
      { id: "f-prompt", label: "Add “prefer the most recent information” to the prompt", explanation: "The context only contains the old version; there is nothing more recent to prefer.", cfg: {} },
      { id: "f-topk", label: "Raise top-k to 8", explanation: "More chunks from the same stale index.", cfg: { topK: 8 } },
    ],
    verifications: [
      { id: "v-fresh", label: "Re-run the golden set, diff indexed versions against sources, and alert when index lag exceeds 24 h", sound: true, explanation: "Confirms the fix and detects the next staleness before users do." },
      { id: "v-once", label: "Ask the rate-limit question once", sound: false, explanation: "Doesn't cover other documents or future drift." },
      { id: "v-job", label: "Check the rebuild job reports success", sound: false, explanation: "Its last success was in April; 'success' says nothing about freshness." },
    ],
    rootCause: "The index-rebuild job was lost in a scheduler migration, so the index still holds April's v2 rate-limit page. Retrieval and generation are working correctly on outdated data.",
  }),
  ragScenario({
    id: "rag-context-truncation", title: "Annual customers told they have 14 days", difficulty: "Intermediate",
    symptom: "Right document retrieved, wrong answer", objective: "Locate evidence lost between retrieval and generation.",
    incident: `Customers on annual plans asking “${q("q-refund")}” are told refunds are possible within 14 days. Policy says 30 days for annual plans. Refund disputes are up. A latency-reduction change shipped last week.`,
    questionId: "q-refund", fault: { contextBudget: 50 },
    actions: ["golden", "answer", "retrieval", "context", "chunks", "prompt", "faithfulness", "dashboards", "deploys"],
    relevant: ["retrieval", "context", "deploys"],
    deploys: ["Wed 16:30 context-builder 1.9: max_context_tokens 300 → 50 (p95 latency −180 ms)", "Wed 11:00 reranker 2.1: minor scoring refactor", "Mon 09:00 docs: refund-policy.md unchanged since June"],
    hypotheses: [
      { id: "h-budget", text: "The context budget truncates the top chunk before the annual-plan sentence.", isRootCause: true, evidenceIds: ["retrieval", "context", "deploys"], explanation: "refunds#0 is retrieved at rank 1 and contains “30 days of purchase”, but the context keeps only 50 tokens of it and the annual sentence is cut." },
      { id: "h-retrieval", text: "Retrieval doesn't find the refund policy.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "refunds#0 is rank 1." },
      { id: "h-gen", text: "The generator confuses monthly and annual plans.", isRootCause: false, evidenceIds: ["context", "faithfulness"], explanation: "The answer is faithful to the context it received; the annual sentence simply wasn't in it." },
      { id: "h-doc", text: "The refund policy document is outdated.", isRootCause: false, evidenceIds: ["answer"], explanation: "The source of truth says 30 days for annual plans and the index holds the current version." },
    ],
    fixes: [
      { id: "f-budget", label: "Restore the context budget to 300 tokens", explanation: "The full chunk fits again.", cfg: { contextBudget: 300 } },
      { id: "f-topk", label: "Raise top-k to 8", explanation: "Retrieval was already correct; the budget still truncates the first chunk.", cfg: { topK: 8 } },
      { id: "f-small", label: "Use 20-word chunks so more of them fit in 50 tokens", explanation: "May help this question but splits other documents (try it: SSO breaks).", cfg: { chunkSize: 20, overlap: 0 } },
    ],
    verifications: [
      { id: "v-golden", label: "Re-run the golden set and track “fact retrieved but missing from context” as a metric", sound: true, explanation: "That metric isolates exactly this failure and would have caught the change." },
      { id: "v-latency", label: "Confirm p95 latency is still improved", sound: false, explanation: "Latency was never the problem; quality was." },
      { id: "v-rank", label: "Confirm the refund chunk still ranks first", sound: false, explanation: "It ranked first throughout the incident." },
    ],
    rootCause: "A latency change cut the context budget from 300 to 50 tokens. The correct chunk is retrieved first but truncated before the annual-plan sentence, so the generator answers from the monthly-plan sentence.",
  }),
  ragScenario({
    id: "rag-citation", title: "Correct answers cite the wrong source", difficulty: "Intermediate",
    symptom: "Citations open passages that don't support the answer", objective: "Separate answer correctness from attribution correctness.",
    incident: `Users clicking citations land on passages that don't contain what the answer says. Example: “${q("q-price")}”. The answer text itself looks right.`,
    questionId: "q-price", fault: { citationBug: true },
    actions: ["golden", "answer", "citation", "context", "retrieval", "prompt", "deploys"],
    relevant: ["citation", "golden", "deploys"],
    deploys: ["Tue 14:10 answer-formatter 3.1: citation labels now 1-based to match the UI", "Tue 09:30 web-chat 4.3: citation hover cards", "Mon 17:00 reranker 2.1"],
    hypotheses: [
      { id: "h-offby1", text: "Citation label mapping is off by one.", isRootCause: true, evidenceIds: ["citation", "golden", "deploys"], explanation: "Every citation points to the passage after the one containing the sentence, on every golden question (citation accuracy 0%) while answer accuracy is unaffected." },
      { id: "h-retrieval", text: "Retrieval returns the wrong chunks.", isRootCause: false, evidenceIds: ["answer", "context"], explanation: "The answer is correct and its sentence is in the context; the right chunk was retrieved." },
      { id: "h-invent", text: "The model invents citations at random.", isRootCause: false, evidenceIds: ["citation"], explanation: "The error is systematic (always +1), which points to deterministic code, not random invention." },
      { id: "h-prompt", text: "Passages are numbered inconsistently in the prompt.", isRootCause: false, evidenceIds: ["prompt", "context"], explanation: "The prompt numbers passages [1]..[n] in order." },
    ],
    fixes: [
      { id: "f-map", label: "Fix the label mapping so the cited label is the passage the sentence came from", explanation: "Citations point to supporting passages again.", cfg: { citationBug: false } },
      { id: "f-reindex", label: "Re-index the corpus", explanation: "The index isn't involved.", cfg: {} },
      { id: "f-one", label: "Put only 1 chunk in context so there's only one passage to cite", explanation: "The off-by-one now points at a passage that doesn't exist.", cfg: { contextChunks: 1 } },
    ],
    verifications: [
      { id: "v-support", label: "Add a citation-support check (cited passage must contain the cited sentence) to the golden set and require 100%", sound: true, explanation: "Automates the exact property that broke." },
      { id: "v-manual", label: "Click a few citations manually", sound: false, explanation: "Unrepeatable and easy to miss off-by-one cases." },
      { id: "v-acc", label: "Confirm answer accuracy is unchanged", sound: false, explanation: "Answer accuracy was never affected; it can't detect this bug." },
    ],
    rootCause: "The answer formatter switched to 1-based labels while the mapping code already added 1, so every citation points to the next passage. Answer text is unaffected, which is why accuracy metrics stayed green.",
  }),
  ragScenario({
    id: "rag-unfaithful", title: "Answers contradict the retrieved evidence", difficulty: "Advanced",
    symptom: "Wrong numbers despite fresh docs", objective: "Use faithfulness checks to tell a grounding failure from a data failure.",
    incident: `QA flagged that the assistant says the Pro rate limit is 300 requests/minute and that Pro costs $39. Both are wrong. Index-freshness checks are green.`,
    questionId: "q-rate", fault: { promptTemplate: "concise-v2" },
    actions: ["golden", "answer", "retrieval", "context", "faithfulness", "index", "prompt", "citation", "deploys"],
    relevant: ["context", "faithfulness", "prompt", "deploys"],
    deploys: ["Thu 13:00 prompt-registry: answer template grounded-v1 → concise-v2 (“friendlier, shorter answers”)", "Thu 09:00 index-rebuild: success, all docs current", "Wed 18:20 llm-gateway: connection pool size 20 → 40"],
    hypotheses: [
      { id: "h-ground", text: "The new prompt dropped the grounding instruction, so the model answers from (outdated) memory.", isRootCause: true, evidenceIds: ["context", "faithfulness", "prompt", "deploys"], explanation: "The context contains “600 requests per minute”, but the answer sentence appears nowhere in it. The template lost “use ONLY the context”." },
      { id: "h-stale", text: "The index is stale.", isRootCause: false, evidenceIds: ["index", "context"], explanation: "Index holds v3 and the context contains the correct value." },
      { id: "h-retrieval", text: "Retrieval found the wrong document.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "Top-1 is rate-limits#0, which contains the right answer." },
      { id: "h-cite", text: "Citation mapping is broken.", isRootCause: false, evidenceIds: ["citation", "faithfulness"], explanation: "The sentence isn't in any passage, so no mapping could make it supported. This is an unfaithful answer, not a mislabeled one." },
    ],
    fixes: [
      { id: "f-prompt", label: "Restore the grounded-v1 template", explanation: "The generator is constrained to the context again.", cfg: { promptTemplate: "grounded-v1" } },
      { id: "f-reindex", label: "Rebuild the index", explanation: "The index is already current.", cfg: { indexSnapshot: "current" } },
      { id: "f-topk", label: "Raise top-k to 8", explanation: "The right passage is already in context; the model ignores it.", cfg: { topK: 8 } },
    ],
    verifications: [
      { id: "v-faith", label: "Add a faithfulness evaluator to the golden set and block prompt changes that lower it", sound: true, explanation: "Prompt changes are code changes; gate them on the property they can break." },
      { id: "v-short", label: "Confirm answers are shorter, as intended", sound: false, explanation: "Measures the goal of the change, not the regression." },
      { id: "v-fresh", label: "Re-check index freshness", sound: false, explanation: "It was already green." },
    ],
    rootCause: "A prompt change removed the grounding instruction. With the right passage in context, the (simulated) model still answered from memorized, outdated facts. The answer is unfaithful to its evidence.",
  }),
  ragScenario({
    id: "prod-everything-green", system: "Production", title: "Everything is green, but the answers are wrong", difficulty: "Advanced",
    symptom: "Thumbs-down tripled; no alerts", objective: "Investigate a silent quality regression that health checks cannot see.",
    incident: `Since this afternoon, thumbs-down feedback on assistant answers has tripled across all topics. Uptime is 100%, error rate 0%, p95 latency normal. No alerts fired. Example: “${q("q-retention")}”.`,
    questionId: "q-retention", fault: { queryModel: "embed-v2" },
    actions: ["dashboards", "golden", "answer", "retrieval", "context", "models", "index", "prompt", "faithfulness", "deploys"],
    relevant: ["retrieval", "models", "golden", "deploys"],
    deploys: ["14:02 query-service 2.8.0: bump embeddings client 3.x → 4.0 (default model now embed-v2)", "13:40 web-chat 4.4: new chat widget theme", "11:15 llm-gateway: raise upstream rate limit", "Yesterday 22:00 index-rebuild: success (embed-v1)"],
    hypotheses: [
      { id: "h-embed", text: "The query encoder was upgraded to embed-v2 while the index still holds embed-v1 vectors.", isRootCause: true, evidenceIds: ["models", "retrieval", "golden", "deploys"], explanation: "Query and index models differ; top-1 similarity collapsed to near zero and retrieval returns unrelated chunks for every topic." },
      { id: "h-llm", text: "The LLM provider silently changed model versions.", isRootCause: false, evidenceIds: ["context", "faithfulness"], explanation: "Answers are faithful to their context; the context itself is irrelevant. The generator is working on bad input." },
      { id: "h-stale", text: "The index is stale.", isRootCause: false, evidenceIds: ["index"], explanation: "All indexed documents are current versions." },
      { id: "h-prompt", text: "A prompt template change degraded answers.", isRootCause: false, evidenceIds: ["prompt"], explanation: "Template is grounded-v1, unchanged." },
      { id: "h-load", text: "A traffic spike is degrading the service.", isRootCause: false, evidenceIds: ["dashboards"], explanation: "Latency and error rate are normal. Health metrics can't see this failure." },
    ],
    fixes: [
      { id: "f-pin", label: "Pin the query encoder to embed-v1 (match the index)", explanation: "Query and chunk vectors are comparable again.", cfg: { queryModel: "embed-v1" } },
      { id: "f-reembed", label: "Re-embed the whole corpus with embed-v2", explanation: "Also valid: index and queries share embed-v2.", cfg: { indexModel: "embed-v2" } },
      { id: "f-llm", label: "Roll back the llm-gateway change", explanation: "That change doesn't touch retrieval.", cfg: {} },
      { id: "f-topk", label: "Raise top-k to 8", explanation: "More near-random chunks.", cfg: { topK: 8 } },
    ],
    verifications: [
      { id: "v-monitor", label: "Re-run the golden set, monitor mean top-1 similarity, and fail startup if query and index model ids differ", sound: true, explanation: "Restores quality and turns this silent failure into a loud one." },
      { id: "v-errors", label: "Confirm the error rate stays at 0%", sound: false, explanation: "It was 0% during the incident." },
      { id: "v-one", label: "Check that one example question now answers correctly", sound: false, explanation: "One question can pass by chance; this failure is systemic." },
    ],
    rootCause: "A client-library upgrade changed the query encoder's default model to embed-v2. The index still contains embed-v1 vectors, so similarities are meaningless and retrieval returns unrelated chunks. Every request succeeds quickly, so health dashboards stay green.",
  }),
];
