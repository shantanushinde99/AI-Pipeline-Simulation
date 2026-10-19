import { useCallback, useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { TraceInspector } from "../../components/visualizer/TraceInspector";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { PipelineCanvas, type PipeNode } from "../../components/pipeline/PipelineCanvas";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle } from "../../components/ui";
import type { VisualizerTab } from "../types";
import { QUESTIONS, type Question } from "./corpus";
import { DEFAULT_RAG, answerQuestion, buildIndex, runGoldenSet, type RagConfig } from "./simulation";
import { RAG_EDGES, RAG_LABELS, STAGES, StageView, stageHealth, stageIO, stageSub, type StageId } from "./stages";

type Params = Pick<RagConfig, "chunkSize" | "overlap" | "topK" | "rerank" | "contextChunks" | "contextBudget">;
const DEFAULT_PARAMS: Params = { chunkSize: DEFAULT_RAG.chunkSize, overlap: DEFAULT_RAG.overlap, topK: DEFAULT_RAG.topK, rerank: true, contextChunks: 3, contextBudget: 300 };

export const FAULTS = {
  none: { label: "No fault (healthy)", cfg: {}, note: "" },
  tables: { label: "Parser drops tables", cfg: { tables: "drop" }, note: "Ask about the Pro plan price: the table holding it never reaches the index." },
  stale: { label: "Stale index snapshot", cfg: { indexSnapshot: "stale" }, note: "Ask about the API rate limit: the index still holds April's v2 page." },
  embed: { label: "Query encoder upgraded to embed-v2", cfg: { queryModel: "embed-v2" }, note: "Every question: similarities collapse because query and index vectors live in different spaces." },
  citation: { label: "Citation mapping off by one", cfg: { citationBug: true }, note: "Text stays correct; each citation points to the next passage." },
  prompt: { label: "Grounding instruction removed", cfg: { promptTemplate: "concise-v2" }, note: "Ask about the rate limit or price: the simulated model answers from outdated memory." },
} satisfies Record<string, { label: string; cfg: Partial<RagConfig>; note: string }>;
type FaultId = keyof typeof FAULTS;

export default function RagVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [fault, setFault] = useState<FaultId>("none");
  const [qid, setQid] = useState("q-refund");
  const [customText, setCustomText] = useState("");
  const [picked, setPicked] = useState<{ id: StageId; at: number } | null>(null);

  const cfg: RagConfig = useMemo(() => ({ ...DEFAULT_RAG, ...params, ...FAULTS[fault].cfg }), [params, fault]);
  const index = useMemo(() => buildIndex(cfg), [cfg]);
  const question: Question = useMemo(() => (qid === "custom" ? { id: "custom", text: customText || "…", expect: "", sourceDoc: "—" } : QUESTIONS.find((q) => q.id === qid)!), [qid, customText]);
  const run = useMemo(() => answerQuestion(index, cfg, question), [index, cfg, question]);
  const golden = useMemo(() => (tab === "metrics" ? runGoldenSet(cfg) : null), [cfg, tab]);

  const pb = usePlayback(STAGES.length, 1300);
  const stage = STAGES[pb.index];
  const selectedId: StageId = picked && picked.at === pb.index ? picked.id : stage.id;
  const selected = STAGES.find((s) => s.id === selectedId)!;
  const onSelect = useCallback((id: string) => setPicked({ id: id as StageId, at: pb.index }), [pb.index]);
  const set = (p: Partial<Params>) => setParams((v) => ({ ...v, ...p }));

  const nodes: PipeNode[] = useMemo(
    () => STAGES.map((s, i) => ({
      id: s.id, label: s.label, Icon: s.Icon, x: s.x, y: s.y, sub: i <= pb.index ? stageSub(s.id, run) : undefined,
      status: i < pb.index ? stageHealth(s.id, run) : i === pb.index ? (stageHealth(s.id, run) === "skipped" ? "skipped" : "active") : "pending",
    })),
    [pb.index, run],
  );
  const edges = useMemo(() => RAG_EDGES.map((e) => ({ ...e, active: e.target === stage.id })), [stage.id]);
  const events = selected.spanIds(run).flatMap((id) => run.spans.find((s) => s.id === id)?.events ?? []);
  const io = stageIO(selectedId, run);

  if (tab === "metrics") {
    return (
      <div className="space-y-3">
        <Callout title="Golden-set evaluation for the current configuration">Every question below is run through the same pipeline you configured on the Visualize tab{fault !== "none" && ` (fault injected: ${FAULTS[fault].label})`}.</Callout>
        {golden && (
          <>
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
              {[
                ["Answer accuracy", golden.accuracy], ["Retrieval recall@k", golden.retrievalRecall], ["Faithfulness", golden.faithfulness], ["Citation accuracy", golden.citationAccuracy],
              ].map(([l, v]) => (
                <Panel key={l as string} title={l as string}><div className="font-mono text-xl">{((v as number) * 100).toFixed(0)}%</div></Panel>
              ))}
              <Panel title="Success rate"><div className="font-mono text-xl">100%</div><p className="text-[11px] text-muted">no request errored</p></Panel>
              <Panel title="p95 latency"><div className="font-mono text-xl">{golden.p95LatencyMs.toFixed(0)} ms</div><p className="text-[11px] text-muted">simulated</p></Panel>
            </div>
            <Panel title="Per-question results">
              <div className="overflow-x-auto"><table className="w-full text-xs">
                <thead className="text-left text-muted"><tr><th className="py-1">question</th><th>answer</th><th>correct</th><th>retrieved</th><th>in context</th><th>faithful</th><th>citation</th></tr></thead>
                <tbody>{golden.runs.map((r) => (
                  <tr key={r.question.id} className="border-t border-line align-top">
                    <td className="py-1 pr-2">{r.question.text}</td><td className="max-w-xs pr-2">{r.answer.text}</td>
                    {[r.evaluation.correct, r.evaluation.retrievalHit, r.evaluation.contextHit, r.evaluation.faithful, r.evaluation.citationCorrect].map((v, i) => <td key={i}><StatusBadge status={v ? "ok" : "error"} label={v ? "yes" : "no"} /></td>)}
                  </tr>
                ))}</tbody></table></div>
            </Panel>
          </>
        )}
        <Panel title={`Trace for: “${run.question.text}”`}><TraceInspector spans={run.spans} initial="retrieve" /></Panel>
      </div>
    );
  }

  return (
    <Workspace
      main={
        <>
          <Panel title="Pipeline" actions={<span className="text-[11px] text-muted">Click any stage to inspect it</span>}>
            <PipelineCanvas nodes={nodes} edges={edges} labels={RAG_LABELS} selected={selectedId} onSelect={onSelect} height={380} ariaLabel="RAG pipeline: indexing flow and query flow" />
          </Panel>
          <PlaybackControls pb={pb} stepTitles={STAGES.map((s) => `${s.flow}: ${s.label}`)} />
          <Panel title={`Output of: ${selected.label}`} actions={<StatusBadge status={stageHealth(selectedId, run) === "skipped" ? "pending" : (stageHealth(selectedId, run) as "ok" | "warning" | "error")} label={stageHealth(selectedId, run)} />}>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={selectedId} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.22 }}>
                <StageView id={selectedId} r={run} onCite={() => setPicked({ id: "context", at: pb.index })} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <Panel title="Question & parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_PARAMS); setFault("none"); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 md:grid-cols-2">
              <Select label="Question" value={qid} onChange={setQid} options={[...QUESTIONS.map((q) => ({ value: q.id, label: q.text })), { value: "custom", label: "Custom question…" }]} />
              {qid === "custom" && (
                <div className="space-y-1"><label htmlFor="rag-custom" className="block text-xs font-medium">Your question</label>
                  <input id="rag-custom" value={customText} onChange={(e) => setCustomText(e.target.value)} placeholder="e.g. Can deleted projects be recovered?" className="h-8 w-full rounded-md border border-line bg-surface px-2 text-sm" /></div>
              )}
            </div>
            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Slider label="Chunk size (words)" value={params.chunkSize} min={10} max={150} step={5} onChange={(v) => set({ chunkSize: v, overlap: Math.min(params.overlap, v - 1) })} />
              <Slider label="Chunk overlap (words)" value={params.overlap} min={0} max={Math.min(60, params.chunkSize - 1)} onChange={(v) => set({ overlap: v })} />
              <Slider label="top-k retrieved" value={params.topK} min={1} max={8} onChange={(v) => set({ topK: v, contextChunks: Math.min(params.contextChunks, v) })} />
              <Slider label="Chunks placed in context" value={params.contextChunks} min={1} max={params.topK} onChange={(v) => set({ contextChunks: v })} />
              <Slider label="Context budget (tokens)" value={params.contextBudget} min={20} max={400} step={10} onChange={(v) => set({ contextBudget: v })} />
              <div className="flex items-end"><Toggle label="Rerank retrieved chunks" checked={params.rerank} onChange={(v) => set({ rerank: v })} hint="Simulated reranker (term overlap)" /></div>
            </div>
          </Panel>
          <Panel title="Inject a fault" actions={<SimulatedLabel>Failure injection</SimulatedLabel>}>
            <Select label="Fault" value={fault} onChange={setFault} options={(Object.keys(FAULTS) as FaultId[]).map((f) => ({ value: f, label: FAULTS[f].label }))} />
            {fault !== "none" && <p className="mt-2 text-xs text-muted">{FAULTS[fault].note} The fault changes the simulation itself; follow it through every stage.</p>}
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={`${stage.flow} flow: ${stage.label}`} what={stage.what} why={stage.why} />
          <Panel title={`Inspector: ${selected.label}`} actions={<Badge>{selected.flow}</Badge>}>
            <p className="mb-2 text-xs leading-relaxed">{selected.what}</p>
            <h4 className="mt-3 mb-1 text-[11px] font-semibold uppercase text-muted">Input</h4><KV rows={io.input} />
            <h4 className="mt-3 mb-1 text-[11px] font-semibold uppercase text-muted">Output</h4><KV rows={io.output} />
            {io.config.length > 0 && <><h4 className="mt-3 mb-1 text-[11px] font-semibold uppercase text-muted">Configuration</h4><KV rows={io.config} /></>}
            <h4 className="mt-3 mb-1 text-[11px] font-semibold uppercase text-muted">Can fail by</h4>
            <ul className="list-disc space-y-0.5 pl-4 text-xs">{selected.failures.map((f) => <li key={f}>{f}</li>)}</ul>
          </Panel>
          <Panel title="Answer so far">
            {pb.index >= STAGES.length - 2 ? <p className="text-xs leading-relaxed">{run.answer.text}</p> : <p className="text-xs text-muted">Generation hasn't run yet. Step forward or press Play.</p>}
          </Panel>
          <Panel title="Events"><EventLog events={events} /></Panel>
        </>
      }
    />
  );
}
