import type { ReactNode } from "react";
import { motion } from "motion/react";
import {
  Binary, Bot, Database, FileCode, FileText, Layers, ListOrdered, MessageSquare, Quote, ScanText, Scissors, Search, Waypoints,
  type LucideIcon,
} from "lucide-react";
import { Badge, Callout, KV, cx } from "../../components/ui";
import type { PipeEdge, PipeLabel, PipeStatus } from "../../components/pipeline/PipelineCanvas";
import { CORPUS } from "./corpus";
import { DIM, TEMPLATES, splitSentences, tokens, wordCount, type RagRun } from "./simulation";

export type StageId = "docs" | "parse" | "chunk" | "embed" | "index" | "query" | "embedq" | "retrieve" | "rerank" | "context" | "prompt" | "generate" | "answer";

type Stage = { id: StageId; label: string; Icon: LucideIcon; x: number; y: number; flow: "Indexing" | "Query"; what: string; why: string; failures: string[]; spanIds: (run: RagRun) => string[] };

export const STAGES: Stage[] = [
  { id: "docs", label: "Documents", Icon: FileText, x: 0, y: 30, flow: "Indexing", what: "The raw knowledge base: PDFs and Markdown pages, each with a version and last-updated date.", why: "The system can only ever answer from what is in here.", failures: ["Document missing from the corpus", "Outdated version published"], spanIds: () => ["index.build"] },
  { id: "parse", label: "Parse", Icon: ScanText, x: 170, y: 30, flow: "Indexing", what: "Turns each file into plain text. Tables are linearized into sentences ('Pro plan: Price $49 per month; …').", why: "Anything the parser drops is invisible to every later stage.", failures: ["Tables or images silently skipped", "Headers/footers mixed into body text", "Wrong reading order in multi-column PDFs"], spanIds: (r) => r.index.parsed.map((p) => `parse:${p.docId}`) },
  { id: "chunk", label: "Chunk", Icon: Scissors, x: 340, y: 30, flow: "Indexing", what: "Splits each document into fixed-size word windows with overlap between neighbours.", why: "Chunks are the unit of retrieval: too small loses context, too large dilutes relevance and wastes the context budget.", failures: ["Instruction split across chunk boundary", "Chunks too large to fit the context budget"], spanIds: () => ["chunk"] },
  { id: "embed", label: "Embed chunks", Icon: Binary, x: 510, y: 30, flow: "Indexing", what: `Maps each chunk to a ${DIM}-dimensional unit vector (simulated: hashed bag-of-words, not a neural model).`, why: "Query and chunks must be embedded by the same model to be comparable.", failures: ["Embedding model changed without re-indexing", "Text truncated by model input limit"], spanIds: () => ["embed.index"] },
  { id: "index", label: "Vector index", Icon: Database, x: 680, y: 30, flow: "Indexing", what: "Stores chunk vectors with metadata: source doc, version, embedding model, build date.", why: "The index is a snapshot. If it isn't rebuilt when docs change, answers go stale.", failures: ["Stale snapshot", "Missing documents", "Model/version metadata mismatch"], spanIds: () => ["index.write"] },
  { id: "query", label: "User query", Icon: MessageSquare, x: 0, y: 160, flow: "Query", what: "The user's question as typed.", why: "Wording matters: vector search matches meaning approximately, not intent exactly.", failures: ["Ambiguous or multi-part questions"], spanIds: () => ["rag.query"] },
  { id: "embedq", label: "Embed query", Icon: Waypoints, x: 170, y: 160, flow: "Query", what: "Embeds the question with the query encoder.", why: "Must use the same model (and version) as the index.", failures: ["Query encoder upgraded but index still on old model"], spanIds: () => ["embed.query"] },
  { id: "retrieve", label: "Retrieve top-k", Icon: Search, x: 340, y: 160, flow: "Query", what: "Scores every chunk vector by cosine similarity (dot product of unit vectors) and keeps the top-k.", why: "If the right chunk is not in top-k, nothing downstream can recover it.", failures: ["Relevant chunk ranked below k", "Near-duplicate chunks crowd out diverse evidence"], spanIds: () => ["retrieve"] },
  { id: "rerank", label: "Rerank", Icon: ListOrdered, x: 510, y: 160, flow: "Query", what: "Re-orders the top-k with a simulated reranker: query-term overlap + 0.5 × vector score.", why: "A second, more careful pass decides which chunks get the limited context slots.", failures: ["Reranker promotes keyword-stuffed chunks", "Added latency"], spanIds: (r) => (r.cfg.rerank ? ["rerank"] : []) },
  { id: "context", label: "Context", Icon: Layers, x: 680, y: 160, flow: "Query", what: "Takes the first N chunks in order until the token budget is used. The last one may be truncated.", why: "The model only sees what fits. Truncation silently removes evidence.", failures: ["Relevant sentence truncated", "Low-ranked noise fills the budget"], spanIds: () => ["context"] },
  { id: "prompt", label: "Build prompt", Icon: FileCode, x: 340, y: 290, flow: "Query", what: "Combines system instructions, numbered context passages and the question.", why: "Grounding instructions ('use only the context') and citation format are set here.", failures: ["Grounding instruction removed", "Passage numbering inconsistent with citation mapping"], spanIds: () => ["prompt"] },
  { id: "generate", label: "Generate (sim)", Icon: Bot, x: 510, y: 290, flow: "Query", what: "Rule-based stand-in for an LLM: picks the context sentence with most query-term overlap (≥ 2) and cites its passage. Not a language model.", why: "Shows how the answer depends entirely on what the context contains.", failures: ["Answer from model memory instead of context", "Citation points to the wrong passage"], spanIds: () => ["generate"] },
  { id: "answer", label: "Answer + cites", Icon: Quote, x: 680, y: 290, flow: "Query", what: "Final answer with citations mapped back to source chunks, checked against the golden answer.", why: "Correct text with a wrong citation still fails users who verify sources.", failures: ["Correct but uncited", "Cited but unfaithful"], spanIds: () => [] },
];

export const RAG_EDGES: PipeEdge[] = [
  { source: "docs", target: "parse" }, { source: "parse", target: "chunk" }, { source: "chunk", target: "embed" }, { source: "embed", target: "index" },
  { source: "index", target: "retrieve", sourceHandle: "b", targetHandle: "t" },
  { source: "query", target: "embedq" }, { source: "embedq", target: "retrieve" }, { source: "retrieve", target: "rerank" }, { source: "rerank", target: "context" },
  { source: "context", target: "prompt", sourceHandle: "b", targetHandle: "t" }, { source: "prompt", target: "generate" }, { source: "generate", target: "answer" },
];
export const RAG_LABELS: PipeLabel[] = [
  { id: "l1", text: "Indexing flow · runs offline when docs change", x: 0, y: 0 },
  { id: "l2", text: "Query flow · runs per request", x: 0, y: 130 },
];

const custom = (r: RagRun) => r.question.id === "custom";

/** Health as the visualizer can see it (it knows the injected fault and the reference answer). */
export function stageHealth(id: StageId, r: RagRun): PipeStatus {
  const e = r.evaluation;
  switch (id) {
    case "parse": return r.index.parsed.some((p) => p.tablesDropped) ? "warning" : "ok";
    case "index": return r.index.sources.some((s, i) => s.version !== CORPUS[i].version) ? "warning" : "ok";
    case "embedq": return r.cfg.queryModel !== r.index.model ? "warning" : "ok";
    case "retrieve": return !custom(r) && !e.retrievalHit ? "warning" : "ok";
    case "rerank": return r.cfg.rerank ? "ok" : "skipped";
    case "context": return r.context.some((c) => c.truncated) || (!custom(r) && e.retrievalHit && !e.contextHit) ? "warning" : "ok";
    case "generate": return r.answer.source === "prior" ? "warning" : "ok";
    case "answer": return custom(r) ? "ok" : e.ok ? "ok" : "error";
    default: return "ok";
  }
}

export function stageSub(id: StageId, r: RagRun): string {
  switch (id) {
    case "docs": return `${r.index.sources.length} files`;
    case "parse": return `${r.index.parsed.reduce((s, p) => s + p.tablesDropped, 0)} tables dropped`;
    case "chunk": return `${r.index.chunks.length} chunks`;
    case "embed": return `${r.index.model} · ${DIM}d`;
    case "index": return `built ${r.index.builtAt}`;
    case "query": return `${r.qTokens.length} terms`;
    case "embedq": return r.cfg.queryModel;
    case "retrieve": return `top-1 ${r.retrieved[0]?.score.toFixed(2) ?? "—"}`;
    case "rerank": return r.cfg.rerank ? `${r.reranked.length} reordered` : "disabled";
    case "context": return `${r.contextTokens}/${r.cfg.contextBudget} tokens`;
    case "prompt": return `${r.promptTokens} tokens`;
    case "generate": return r.answer.source === "none" ? "no answer" : `source: ${r.answer.source}`;
    case "answer": return custom(r) ? "no reference" : r.evaluation.ok ? "passes eval" : "fails eval";
  }
}

type Rows = [ReactNode, ReactNode][];
export function stageIO(id: StageId, r: RagRun): { input: Rows; output: Rows; config: Rows } {
  const c = r.cfg;
  switch (id) {
    case "docs": return { input: [["corpus", "Northwind Cloud docs"]], output: r.index.sources.map((s) => [s.file, `${s.version} · ${s.updatedAt}`]), config: [["snapshot", c.indexSnapshot]] };
    case "parse": return { input: [["files", r.index.parsed.length]], output: r.index.parsed.map((p) => [p.file, `${wordCount(p.text)} words${p.tablesDropped ? `, ${p.tablesDropped} table dropped` : ""}`]), config: [["table handling", c.tables]] };
    case "chunk": return { input: [["parsed words", r.index.parsed.reduce((s, p) => s + wordCount(p.text), 0)]], output: [["chunks", r.index.chunks.length], ["avg words", (r.index.chunks.reduce((s, x) => s + wordCount(x.text), 0) / r.index.chunks.length).toFixed(1)]], config: [["chunk size", `${c.chunkSize} words`], ["overlap", `${c.overlap} words`]] };
    case "embed": return { input: [["chunks", r.index.chunks.length]], output: [["vectors", r.index.entries.length], ["dimensions", DIM]], config: [["model", r.index.model]] };
    case "index": return { input: [["vectors", r.index.entries.length]], output: [["entries", r.index.entries.length], ["built", r.index.builtAt]], config: [["model", r.index.model], ["snapshot", c.indexSnapshot]] };
    case "query": return { input: [["text", r.question.text]], output: [["normalized terms", r.qTokens.join(", ")]], config: [] };
    case "embedq": return { input: [["terms", r.qTokens.join(", ")]], output: [["vector", `${DIM}d, unit length`]], config: [["query model", c.queryModel], ["index model", r.index.model]] };
    case "retrieve": return { input: [["candidates", r.index.entries.length]], output: r.retrieved.map((h) => [`#${h.rank} ${h.entry.id}`, h.score.toFixed(3)]), config: [["top-k", c.topK], ["metric", "cosine"]] };
    case "rerank": return { input: r.retrieved.map((h) => [h.entry.id, `rank ${h.rank}`]), output: c.rerank ? r.reranked.map((h, i) => [`#${i + 1} ${h.entry.id}`, `overlap ${h.overlap}, score ${h.rerankScore?.toFixed(2)}`]) : [["status", "disabled: retrieval order kept"]], config: [["enabled", String(c.rerank)]] };
    case "context": return { input: [["candidates", r.reranked.length]], output: r.context.map((x) => [`[${x.label}] ${x.entry.id}`, `${x.tokens} tok${x.truncated ? ` (cut ${x.droppedTokens})` : ""}`]), config: [["max chunks", c.contextChunks], ["budget", `${c.contextBudget} tokens`]] };
    case "prompt": return { input: [["passages", r.context.length], ["question", r.question.text]], output: [["prompt tokens", r.promptTokens]], config: [["template", c.promptTemplate]] };
    case "generate": return { input: [["prompt tokens", r.promptTokens]], output: [["answer", r.answer.text]], config: [["generator", "sim-gen-1 (rule-based)"]] };
    case "answer": return { input: [["answer", r.answer.text]], output: [["citations", r.answer.citations.map((x) => `[${x.label}] → ${x.chunkId ?? "nothing"}`).join(", ") || "none"]], config: [["reference", custom(r) ? "none" : r.question.expect]] };
  }
}

const Check = ({ ok, label }: { ok: boolean; label: string }) => <Badge tone={ok ? "ok" : "err"}>{ok ? "✓" : "✗"} {label}</Badge>;

function Hl({ text, terms }: { text: string; terms: string[] }) {
  return <>{text.split(/(\s+)/).map((w, i) => (tokens(w).some((t) => terms.includes(t)) ? <mark key={i} className="rounded bg-accent-soft px-0.5 text-fg">{w}</mark> : w))}</>;
}

/** The artifact produced by a stage: what you'd see if you logged its output. */
export function StageView({ id, r, onCite }: { id: StageId; r: RagRun; onCite: (chunkId: string) => void }) {
  const expect = custom(r) ? null : r.question.expect.toLowerCase();
  const hasFact = (t: string) => !!expect && t.toLowerCase().includes(expect);
  switch (id) {
    case "docs":
      return (
        <div className="grid gap-2 md:grid-cols-2">
          {r.index.sources.map((s) => (
            <div key={s.id} className="rounded-md border border-line p-2 text-xs">
              <div className="flex items-center gap-2"><span className="font-mono font-medium">{s.file}</span><Badge>{s.version}</Badge><span className="text-muted">{s.updatedAt}</span></div>
              <div className="mt-1 text-muted">{s.blocks.map((b) => b.type).join(" · ")}</div>
              {s.blocks.filter((b) => b.type === "table").map((b, i) => b.type === "table" && (
                <table key={i} className="mt-2 w-full border border-line text-[11px]"><thead><tr>{b.headers.map((h) => <th key={h} className="border-b border-line px-1 text-left">{h}</th>)}</tr></thead>
                  <tbody>{b.rows.map((row) => <tr key={row[0]}>{row.map((c, j) => <td key={j} className="px-1">{c}</td>)}</tr>)}</tbody></table>
              ))}
            </div>
          ))}
        </div>
      );
    case "parse":
      return (
        <div className="space-y-2">
          {r.index.parsed.map((p) => (
            <div key={p.docId} className="rounded-md border border-line p-2 text-xs">
              <div className="mb-1 flex items-center gap-2"><span className="font-mono">{p.file}</span>{p.tablesDropped > 0 && <Badge tone="warn">{p.tablesDropped} table dropped</Badge>}{p.tablesFound > 0 && !p.tablesDropped && <Badge tone="ok">table linearized</Badge>}</div>
              <p className="leading-relaxed text-muted">{p.text}</p>
            </div>
          ))}
        </div>
      );
    case "chunk": {
      const doc = r.question.sourceDoc;
      return (
        <div className="space-y-2">
          <p className="text-xs text-muted">Chunk boundaries for every document. Highlighted: chunks containing the reference answer “{expect ?? "n/a"}”.</p>
          {r.index.parsed.map((p) => {
            const cs = r.index.chunks.filter((c) => c.docId === p.docId);
            return (
              <details key={p.docId} open={p.docId === doc} className="rounded-md border border-line p-2 text-xs">
                <summary className="cursor-pointer font-mono">{p.file} · {cs.length} chunks</summary>
                <ol className="mt-2 space-y-1">
                  {cs.map((c) => (
                    <li key={c.id} className={cx("rounded border px-2 py-1 leading-relaxed", hasFact(c.text) ? "border-ok/50 bg-ok-soft" : "border-line")}>
                      <span className="mr-1 font-mono text-muted">{c.id} [{c.startWord}–{c.endWord}]</span>{c.text}
                    </li>
                  ))}
                </ol>
              </details>
            );
          })}
        </div>
      );
    }
    case "embed":
    case "embedq": {
      const vecs = id === "embedq" ? [{ id: "query", v: r.qVector }] : r.index.entries.slice(0, 4).map((e) => ({ id: e.id, v: e.vector }));
      return (
        <div className="space-y-2">
          {vecs.map(({ id: vid, v }) => (
            <div key={vid} className="text-xs">
              <div className="mb-0.5 font-mono text-muted">{vid} · first 64 of {DIM} dims</div>
              <svg viewBox="0 0 256 30" className="h-8 w-full" aria-label={`Vector preview for ${vid}`}>
                {v.slice(0, 64).map((x, i) => <rect key={i} x={i * 4} y={x >= 0 ? 15 - x * 30 : 15} width={3} height={Math.abs(x) * 30} fill={x >= 0 ? "var(--accent)" : "var(--warn)"} />)}
                <line x1={0} x2={256} y1={15} y2={15} stroke="var(--line)" />
              </svg>
            </div>
          ))}
          {id === "embedq" && r.cfg.queryModel !== r.index.model && <Callout tone="warn" title="Model mismatch">The query is embedded with {r.cfg.queryModel} but the index holds {r.index.model} vectors. Their dimensions mean different things, so similarities are close to random.</Callout>}
        </div>
      );
    }
    case "index":
      return (
        <div className="overflow-x-auto"><table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th>source</th><th>version in index</th><th>current version</th><th>chunks</th></tr></thead>
          <tbody>{r.index.sources.map((s, i) => (
            <tr key={s.id} className="border-t border-line"><td className="py-1 font-mono">{s.file}</td><td>{s.version} ({s.updatedAt})</td><td>{CORPUS[i].version}</td><td>{r.index.chunks.filter((c) => c.docId === s.id).length}</td></tr>
          ))}</tbody></table>
          <p className="mt-2 text-xs text-muted">Index model {r.index.model}, built {r.index.builtAt}.</p></div>
      );
    case "query":
      return <div className="text-sm"><p>“{r.question.text}”</p><p className="mt-2 text-xs text-muted">Normalized terms (stop-words removed, simple synonyms): <span className="font-mono">{r.qTokens.join(", ")}</span></p></div>;
    case "retrieve":
      return (
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th>rank</th><th>chunk</th><th>preview</th><th className="text-right">cosine</th></tr></thead>
          <tbody>{r.scored.slice(0, Math.max(8, r.cfg.topK)).map((h) => (
            <tr key={h.entry.id} className={cx("border-t border-line", h.rank <= r.cfg.topK && "bg-accent-soft/60")}>
              <td className="py-1 font-mono">{h.rank}{h.rank <= r.cfg.topK && " ✓"}</td>
              <td className="font-mono">{h.entry.id}</td>
              <td className="max-w-[320px] truncate">{hasFact(h.entry.text) && <Badge tone="ok">has answer</Badge>} {h.entry.text}</td>
              <td className="text-right font-mono">{h.score.toFixed(3)}</td>
            </tr>
          ))}</tbody>
        </table>
      );
    case "rerank":
      return !r.cfg.rerank ? <p className="text-xs text-muted">Reranking is disabled; retrieval order is passed straight to context assembly.</p> : (
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th>new</th><th>was</th><th>chunk</th><th>term overlap</th><th className="text-right">rerank score</th></tr></thead>
          <tbody>{r.reranked.map((h, i) => (
            <tr key={h.entry.id} className="border-t border-line"><td className="py-1 font-mono">{i + 1}</td><td className="font-mono text-muted">{h.rank}{h.rank !== i + 1 && (h.rank > i + 1 ? " ↑" : " ↓")}</td><td className="font-mono">{h.entry.id} {hasFact(h.entry.text) && <Badge tone="ok">has answer</Badge>}</td><td>{h.overlap}</td><td className="text-right font-mono">{h.rerankScore?.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
      );
    case "context":
      return (
        <div className="space-y-2">
          <div className="flex h-6 overflow-hidden rounded border border-line" aria-label={`Context window: ${r.contextTokens} of ${r.cfg.contextBudget} tokens used`}>
            {r.context.map((c) => (
              <motion.div key={c.label} initial={{ width: 0 }} animate={{ width: `${(c.tokens / r.cfg.contextBudget) * 100}%` }} transition={{ delay: c.label * 0.25, duration: 0.5, ease: "easeOut" }} className={cx("flex items-center justify-center overflow-hidden border-r border-surface text-[10px] text-white", c.truncated ? "bg-warn" : "bg-accent")} title={c.entry.id}>[{c.label}]</motion.div>
            ))}
          </div>
          <p className="text-[11px] text-muted">{r.contextTokens} / {r.cfg.contextBudget} tokens used (1 token ≈ 1 word in this simulation).</p>
          {r.context.map((c) => (
            <div key={c.label} className={cx("rounded-md border p-2 text-xs leading-relaxed", hasFact(c.text) ? "border-ok/50" : "border-line")}>
              <div className="mb-1 flex flex-wrap gap-1"><span className="font-mono">[{c.label}] {c.entry.id}</span>{c.truncated && <Badge tone="warn">truncated: {c.droppedTokens} tokens cut</Badge>}{hasFact(c.text) && <Badge tone="ok">has answer</Badge>}</div>
              <Hl text={c.text} terms={r.qTokens} />
            </div>
          ))}
        </div>
      );
    case "prompt":
      return <pre className="max-h-80 overflow-auto rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{r.prompt}</pre>;
    case "generate": {
      const cands = r.context.flatMap((c) => splitSentences(c.text).filter((s) => wordCount(s) >= 5).map((s) => ({ s, label: c.label, score: new Set(tokens(s).filter((t) => r.qTokens.includes(t))).size })));
      return (
        <div className="space-y-2 text-xs">
          <Callout tone="warn" title="Simulated generator">Not an LLM. It extracts the context sentence sharing the most query terms (needs ≥ 2), or says it doesn't know.{r.cfg.promptTemplate === "concise-v2" && " With the 'concise-v2' prompt (no grounding rule) it answers from a fixed, outdated memory when it has one."}</Callout>
          <p className="text-muted">System instruction: “{TEMPLATES[r.cfg.promptTemplate]}”</p>
          <table className="w-full"><thead className="text-left text-muted"><tr><th>passage</th><th>candidate sentence</th><th>overlap</th></tr></thead>
            <tbody>{cands.sort((a, b) => b.score - a.score).slice(0, 6).map((c, i) => (
              <tr key={i} className={cx("border-t border-line", c.s === r.answer.sentence && "bg-accent-soft")}><td className="py-1 font-mono">[{c.label}]</td><td>{c.s}</td><td className="font-mono">{c.score}</td></tr>
            ))}</tbody></table>
          {r.answer.source === "prior" && <Callout tone="err" title="Answered from memory">The answer sentence does not appear anywhere in the context.</Callout>}
        </div>
      );
    }
    case "answer":
      return (
        <div className="space-y-3">
          <p className="text-sm leading-relaxed" data-testid="rag-answer">
            {r.answer.text.replace(/\s*\[\d+\]$/, "")}{" "}
            {r.answer.citations.map((c) => (
              <button key={c.label} onClick={() => c.chunkId && onCite(c.chunkId)} disabled={!c.chunkId} className="rounded bg-accent-soft px-1 font-mono text-xs text-accent hover:underline disabled:opacity-50">[{c.label}] {c.chunkId ?? "missing"}</button>
            ))}
          </p>
          {custom(r) ? <p className="text-xs text-muted">Custom question: no reference answer, so correctness cannot be scored.</p> : (
            <div className="flex flex-wrap gap-1.5">
              <Check ok={r.evaluation.correct} label={`contains “${r.question.expect}”`} />
              <Check ok={r.evaluation.retrievalHit} label="answer chunk retrieved" />
              <Check ok={r.evaluation.contextHit} label="answer in context" />
              <Check ok={r.evaluation.faithful} label="faithful to context" />
              <Check ok={r.evaluation.citationCorrect} label="citation supports answer" />
            </div>
          )}
          <KV rows={[["reference", custom(r) ? "—" : r.question.expect], ["source doc", r.question.sourceDoc]]} />
        </div>
      );
  }
}
