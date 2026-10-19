import { useMemo, useState, type FormEvent } from "react";
import { motion } from "motion/react";
import { CheckCircle2, RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../top-k-retrieval/StepCaption";
import { CONCEPTS, DEFAULT_INPUT, DOCS, LEXICON, QUERIES, embed, simulate, type FusionMode, type HybridInput, type Phase, type Ranked, type Run } from "./simulation";

const ROW = 38;
/** Shared schedule (seconds at 1×). */
const T = {
  chip: (i: number) => 0.1 + i * 0.12,
  idf: (i: number) => 0.25 + i * 0.3,
  bar: (i: number) => 0.2 + i * 0.1,
  sort: 1.4,
  ghost: (i: number) => 0.3 + i * 0.4,
  fly: 0.8,
};
const DUR: Record<Phase, (r: Run) => number> = {
  tokenize: (r) => (T.chip(r.qTokens.length) + 1.4) * 1000,
  idf: (r) => (T.idf(r.bm.qTerms.length) + 1.2) * 1000,
  bm25: () => 3000,
  vector: () => 3000,
  lists: () => 2400,
  fuse: (r) => (T.ghost(r.fused.length) + T.fly + 1) * 1000,
  compare: () => 2600,
};
const colorOf = { bm25: "var(--accent)", vector: "var(--violet)", fused: "var(--ok)" } as const;

export default function HybridVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<HybridInput>(DEFAULT_INPUT);
  const [text, setText] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const [replay, setReplay] = useState(0);
  const { run, steps } = useMemo(() => simulate(input), [input]);
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "lists"](run));
  const step = steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${run.query}`;
  const set = (p: Partial<HybridInput>) => setInput((v) => ({ ...v, ...p }));
  const selected = picked ?? (phase === "vector" ? run.vecList[0]?.id : run.bmList[0]?.id) ?? "D1";

  if (tab === "metrics") return null;

  const submit = (e: FormEvent) => { e.preventDefault(); if (text.trim()) { set({ customQuery: text.trim() }); pb.restart(); } };
  const story = narrate(phase, run);
  const order = (phase === "tokenize" || phase === "idf") ? 0 : ["bm25", "vector", "lists", "fuse", "compare"].indexOf(phase) + 1;

  return (
    <Workspace
      main={
        <>
          <Panel title="Keyword (BM25) vs vector vs fused" actions={<SimulatedLabel>Toy concept embedding</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <QueryLanes run={run} phase={phase} k={k} animKey={animKey} />
            <div className="mt-3 grid grid-cols-3">
              <Column title="BM25" color={colorOf.bm25} list={run.bmList} shown={order >= 1} sorted={order >= 1} phaseActive={phase === "bm25"} run={run} max={Math.max(1e-9, ...run.bmList.map((r) => r.score))} digits={2}
                k={k} animKey={animKey} selected={selected} onPick={setPicked} showRecall={phase === "compare"} recall={run.recall.bm25} dimTop={phase === "lists" || phase === "compare"} />
              <Column title="Vector" color={colorOf.vector} list={run.vecList} shown={order >= 2} sorted={order >= 2} phaseActive={phase === "vector"} run={run} max={1} digits={2}
                k={k} animKey={animKey} selected={selected} onPick={setPicked} showRecall={phase === "compare"} recall={run.recall.vector} dimTop={phase === "lists" || phase === "compare"} />
              <FusedColumn run={run} shown={order >= 4} flying={phase === "fuse"} k={k} animKey={animKey} selected={selected} onPick={setPicked} showRecall={phase === "compare"} />
            </div>
            <DocKey run={run} selected={selected} onPick={setPicked} />
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <StepCard run={run} phase={phase} selected={selected} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Query & parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(DEFAULT_INPUT); setText(""); setPicked(null); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="flex flex-wrap gap-1.5">
              {QUERIES.map((q) => (
                <Button key={q.id} size="sm" variant={!input.customQuery && input.queryId === q.id ? "primary" : "secondary"} onClick={() => { set({ queryId: q.id, customQuery: undefined }); setPicked(null); pb.restart(); }}>{q.label}</Button>
              ))}
            </div>
            <form onSubmit={submit} className="mt-2 flex gap-2">
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Or type a query, e.g. certificate expired on my domain" aria-label="Custom query" className="h-8 min-w-0 flex-1 rounded-md border border-line bg-surface px-2.5 text-sm" />
              <Button type="submit">Search</Button>
            </form>
            <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Slider label={<>BM25 <Tex>{"k_1"}</Tex> (tf saturation)</>} value={input.k1} min={0} max={3} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ k1: v })} />
              <Slider label={<>BM25 <Tex>{"b"}</Tex> (length normalization)</>} value={input.b} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ b: v })} />
              <Select label="Fusion" value={input.mode} onChange={(v: FusionMode) => set({ mode: v })} options={[{ value: "rrf", label: "Reciprocal Rank Fusion" }, { value: "weighted", label: "Weighted α blend" }]} />
              {input.mode === "rrf"
                ? <Slider label={<>RRF constant <Tex>{"k"}</Tex></>} value={input.kRrf} min={1} max={100} onChange={(v) => set({ kRrf: v })} hint="Small k: top ranks dominate. Large k: ranks count more evenly." />
                : <Slider label={<>α (weight on BM25)</>} value={input.alpha} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ alpha: v })} hint="α = 1 is keyword-only, α = 0 is vector-only." />}
              <Slider label="Evaluate top-N" value={input.topN} min={1} max={4} onChange={(v) => set({ topN: v })} />
            </div>
            <p className="mt-2 text-xs text-muted">{input.customQuery ? `Custom query: “${input.customQuery}” (no relevance labels).` : run.preset.expect}</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Keyword search nails exact identifiers; embeddings catch paraphrases. Production systems run both and fuse." />
          <Panel title="Calculation"><Calc run={run} phase={phase} selected={selected} /></Panel>
          <Panel title="Key values">
            <KV rows={[["documents N", run.bm.N], ["avg length avgdl", run.bm.avgdl.toFixed(2)], [<Tex>{"k_1,\\ b"}</Tex>, `${run.params.k1}, ${run.params.b}`], ["fusion", run.mode === "rrf" ? `RRF k = ${run.kRrf}` : `α = ${run.alpha}`], ["selected doc", selected]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(phase: Phase, run: Run) {
  const n = run.topN;
  switch (phase) {
    case "tokenize": return { headline: "The same query is read two different ways", sub: run.oov.length ? `Keyword search keeps every term, even ${run.oov.join(", ")}. The embedding has no meaning for ${run.oov.length === 1 ? "it" : "them"}.` : "Keyword search keeps exact strings; the embedding maps words to concepts." };
    case "idf": return { headline: "BM25 first asks: how rare is each term?", sub: "Terms that appear in few documents get a big IDF weight; common words barely count." };
    case "bm25": return { headline: "BM25 scores each document, then the column sorts", sub: "Bar = sum over matching terms of idf × saturated term frequency, adjusted for document length." };
    case "vector": return { headline: "Vector search scores the same documents by meaning", sub: "Bar = cosine similarity between concept vectors. No shared words needed." };
    case "lists": return { headline: `Two rankings disagree: compare their top ${n}`, sub: "✓ marks the documents a human labelled relevant for this query." };
    case "fuse": return run.mode === "rrf"
      ? { headline: "Each document flies in with 1/(k + rank) from each list", sub: "Reciprocal Rank Fusion only looks at positions, so differently scaled scores can be combined." }
      : { headline: "Each document flies in with α·BM25 + (1 − α)·cosine", sub: "A weighted blend needs both scores on the same scale, so BM25 is divided by its max." };
    case "compare": return run.relevant.length
      ? { headline: `recall@${n}: BM25 ${pctS(run.recall.bm25)} · vector ${pctS(run.recall.vector)} · fused ${pctS(run.recall.fused)}`, sub: run.preset.expect }
      : { headline: "Compare the three lists", sub: "Custom query: there are no relevance labels to score against." };
  }
}
const pctS = (v: number) => (Number.isNaN(v) ? "n/a" : `${Math.round(v * 100)}%`);

function QueryLanes({ run, phase, k, animKey }: { run: Run; phase: Phase; k: number; animKey: string }) {
  const toks = run.qTokens;
  const qv = run.qVec;
  const intro = phase === "tokenize";
  const idfPhase = phase === "idf";
  const maxIdf = Math.log(1 + (run.bm.N - 0 + 0.5) / 0.5);
  return (
    <div className="grid gap-2 rounded-lg bg-surface-2 p-2 sm:grid-cols-2">
      <div>
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide" style={{ color: colorOf.bm25 }}>BM25 sees exact terms</div>
        <div className="flex flex-wrap gap-1">
          {run.bm.qTerms.map((t, i) => {
            const df = run.bm.dfs.get(t)!;
            const iv = Math.log(1 + (run.bm.N - df + 0.5) / (df + 0.5));
            return (
              <motion.span key={`${t}-${intro || idfPhase ? animKey : ""}`} className="relative overflow-hidden rounded-md border border-accent/40 bg-surface px-1.5 py-0.5 font-mono text-[11px]"
                initial={intro ? { opacity: 0, y: -14 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: T.chip(i) * k, duration: 0.35 * k, ease: EASE }}>
                {idfPhase && <motion.span className="absolute inset-y-0 left-0 bg-accent/20" initial={{ width: 0 }} animate={{ width: `${(iv / maxIdf) * 100}%` }} transition={{ delay: T.idf(i) * k, duration: 0.5 * k, ease: EASE }} />}
                <span className="relative">{t}</span>
                {(idfPhase || !intro) && <span className="relative ml-1 text-muted">df {df}{idfPhase && <> · idf <AnimatedNumber value={iv} from={0} delay={T.idf(i) * k} duration={0.5 * k} /></>}</span>}
              </motion.span>
            );
          })}
        </div>
      </div>
      <div>
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide" style={{ color: colorOf.vector }}>Embedding sees concepts</div>
        <div className="mb-1 flex flex-wrap gap-1">
          {toks.map((t, i) => (
            <motion.span key={`${t}-${i}-${intro ? animKey : ""}`} className={cx("rounded-md border px-1.5 py-0.5 font-mono text-[11px]", LEXICON[t] ? "border-violet/40 bg-surface" : "border-line text-muted line-through")}
              initial={intro ? { opacity: 0, y: -14 } : false} animate={{ opacity: LEXICON[t] ? 1 : 0.6, y: 0 }} transition={{ delay: T.chip(i) * k, duration: 0.35 * k, ease: EASE }} title={LEXICON[t] ? Object.keys(LEXICON[t]).join(", ") : "out of vocabulary"}>
              {t}{!LEXICON[t] && <span className="sr-only"> (out of vocabulary)</span>}
            </motion.span>
          ))}
        </div>
        <div className="grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-0.5">
          {CONCEPTS.map((c, i) => qv[i] > 0 && (
            <div key={c} className="contents">
              <span className="text-[10px] text-muted">{c}</span>
              <div className="h-2 rounded bg-surface"><motion.div key={intro ? animKey : c} className="h-full rounded bg-violet/70" initial={{ width: intro ? 0 : `${qv[i] * 100}%` }} animate={{ width: `${qv[i] * 100}%` }} transition={{ delay: (intro ? T.chip(toks.length) + 0.2 : 0) * k, duration: 0.6 * k, ease: EASE }} /></div>
            </div>
          ))}
          {qv.every((x) => x === 0) && <span className="col-span-2 text-[11px] text-warn">Empty vector: no known words</span>}
        </div>
      </div>
    </div>
  );
}

function Column({ title, color, list, shown, sorted, phaseActive, run, max, digits, k, animKey, selected, onPick, showRecall, recall, dimTop }: {
  title: string; color: string; list: Ranked[]; shown: boolean; sorted: boolean; phaseActive: boolean; run: Run; max: number; digits: number; k: number; animKey: string;
  selected: string; onPick: (id: string) => void; showRecall: boolean; recall: number; dimTop: boolean;
}) {
  const missing = DOCS.filter((d) => !list.some((r) => r.id === d.id));
  // During its own phase rows start in document order, grow their bars, then slide into rank order.
  const slotOf = (id: string) => { const r = list.find((x) => x.id === id); return r ? r.rank - 1 : list.length + missing.findIndex((d) => d.id === id); };
  return (
    <div className="px-0.5 sm:px-1">
      <Header title={title} color={color} showRecall={showRecall} recall={recall} n={run.topN} />
      <div className="relative" style={{ height: DOCS.length * ROW }}>
        {dimTop && <motion.div className="absolute inset-x-0 top-0 rounded-md border-2 border-dashed" style={{ borderColor: color }} initial={{ opacity: 0 }} animate={{ opacity: 1, height: run.topN * ROW }} transition={{ duration: 0.4 * k }} />}
        {shown && DOCS.map((d, i) => {
          const r = list.find((x) => x.id === d.id);
          const slot = slotOf(d.id);
          const rel = run.relevant.includes(d.id);
          return (
            <motion.button key={`${d.id}-${phaseActive ? animKey : ""}`} onClick={() => onPick(d.id)} aria-label={`${d.id} ${d.title}: ${title} ${r ? `rank ${r.rank}, score ${r.score.toFixed(3)}` : "no match"}`}
              className={cx("absolute inset-x-0 flex items-center gap-1 rounded-md border bg-surface px-1 text-left text-[11px]", selected === d.id ? "border-fg" : "border-line", !r && "opacity-45")}
              style={{ height: ROW - 6 }}
              initial={phaseActive ? { y: i * ROW + 3, opacity: 0 } : false}
              animate={{ y: (sorted ? slot : i) * ROW + 3, opacity: r ? 1 : 0.45 }}
              transition={{ y: { delay: (phaseActive ? T.sort : 0) * k, duration: 0.7 * k, ease: EASE }, opacity: { duration: 0.25 * k, delay: phaseActive ? T.bar(i) * 0.5 * k : 0 } }}>
              <span className="w-5 shrink-0 font-mono font-semibold">{d.id}</span>
              <span className="relative h-3 min-w-0 flex-1 rounded-sm bg-surface-2">
                <motion.span className="absolute inset-y-0 left-0 rounded-sm" style={{ background: color }} initial={phaseActive ? { width: 0 } : false} animate={{ width: `${((r?.score ?? 0) / max) * 100}%` }}
                  transition={{ delay: (phaseActive ? T.bar(i) : 0) * k, duration: 0.6 * k, ease: EASE }} />
              </span>
              <span className="w-8 shrink-0 text-right font-mono text-[10px]">{r ? (phaseActive ? <AnimatedNumber value={r.score} from={0} digits={digits} delay={T.bar(i) * k} duration={0.6 * k} /> : r.score.toFixed(digits)) : "—"}</span>
              {rel && dimTop && <CheckCircle2 size={11} className="shrink-0 text-ok" aria-label="relevant" />}
            </motion.button>
          );
        })}
        {!shown && <div className="flex h-full items-center justify-center rounded-md border border-dashed border-line text-[11px] text-muted">waiting…</div>}
      </div>
    </div>
  );
}

function Header({ title, color, showRecall, recall, n }: { title: string; color: string; showRecall: boolean; recall: number; n: number }) {
  return (
    <div className="mb-1 flex min-h-6 flex-wrap items-center justify-between gap-1">
      <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color }}>{title}</span>
      {showRecall && !Number.isNaN(recall) && (
        <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 300, damping: 16 }}>
          <Badge tone={recall >= 1 ? "ok" : recall > 0 ? "warn" : "err"}>R@{n} {Math.round(recall * 100)}%</Badge>
        </motion.span>
      )}
    </div>
  );
}

function FusedColumn({ run, shown, flying, k, animKey, selected, onPick, showRecall }: { run: Run; shown: boolean; flying: boolean; k: number; animKey: string; selected: string; onPick: (id: string) => void; showRecall: boolean }) {
  const max = Math.max(1e-9, ...run.fused.map((f) => f.score));
  return (
    <div className="px-0.5 sm:px-1">
      <Header title="Fused" color={colorOf.fused} showRecall={showRecall} recall={run.recall.fused} n={run.topN} />
      <div className="relative" style={{ height: DOCS.length * ROW }}>
        {showRecall && <motion.div className="absolute inset-x-0 top-0 rounded-md border-2 border-dashed border-ok" initial={{ opacity: 0 }} animate={{ opacity: 1, height: run.topN * ROW }} transition={{ duration: 0.4 * k }} />}
        {!shown && <div className="flex h-full items-center justify-center rounded-md border border-dashed border-line text-[11px] text-muted">waiting…</div>}
        {shown && run.fused.map((f, i) => {
          const at = T.ghost(i) * k;
          return (
            <div key={f.id}>
              {/* Ghost contributions fly in from the BM25 column (two columns left) and the vector column (one column left). */}
              {flying && ([["bm25", f.bm25Rank, "-200%", f.fromBm25], ["vector", f.vecRank, "-100%", f.fromVec]] as const).map(([src, rank, x, v]) => rank && (
                <motion.div key={`${animKey}-${f.id}-${src}`} className="pointer-events-none absolute inset-x-0 z-10 flex items-center justify-center rounded-md px-1 font-mono text-[10px] text-white shadow dark:text-black"
                  style={{ height: ROW - 6, background: colorOf[src] }}
                  initial={{ x, y: (rank - 1) * ROW + 3, opacity: 0 }} animate={{ x: "0%", y: i * ROW + 3, opacity: [0, 1, 1, 0] }}
                  transition={{ delay: at + (src === "vector" ? 0.08 : 0) * k, duration: T.fly * k, ease: EASE, opacity: { delay: at, duration: T.fly * k, times: [0, 0.1, 0.8, 1] } }}>
                  {f.id} +{v.toFixed(run.mode === "rrf" ? 4 : 2)}
                </motion.div>
              ))}
              <motion.button key={flying ? `${animKey}-${f.id}` : f.id} onClick={() => onPick(f.id)} aria-label={`${f.id}: fused rank ${f.rank}, score ${f.score.toFixed(4)}`}
                className={cx("absolute inset-x-0 flex items-center gap-1 rounded-md border bg-surface px-1 text-left text-[11px]", selected === f.id ? "border-fg" : "border-line")}
                style={{ height: ROW - 6, top: i * ROW + 3 }}
                initial={flying ? { opacity: 0, scale: 0.9 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: flying ? at + T.fly * 0.8 * k : 0, duration: 0.3 * k }}>
                <span className="w-5 shrink-0 font-mono font-semibold">{f.id}</span>
                <span className="relative flex h-3 min-w-0 flex-1 overflow-hidden rounded-sm bg-surface-2">
                  <span className="h-full" style={{ width: `${(f.fromBm25 / max) * 100}%`, background: colorOf.bm25 }} />
                  <span className="h-full" style={{ width: `${(f.fromVec / max) * 100}%`, background: colorOf.vector }} />
                </span>
                <span className="w-9 shrink-0 text-right font-mono text-[10px]">{f.score.toFixed(run.mode === "rrf" ? 4 : 2)}</span>
                {run.relevant.includes(f.id) && showRecall && <CheckCircle2 size={11} className="shrink-0 text-ok" aria-label="relevant" />}
              </motion.button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DocKey({ run, selected, onPick }: { run: Run; selected: string; onPick: (id: string) => void }) {
  return (
    <div className="mt-2 grid gap-x-3 gap-y-0.5 text-[11px] sm:grid-cols-2">
      {DOCS.map((d) => (
        <button key={d.id} onClick={() => onPick(d.id)} className={cx("truncate rounded px-1 text-left hover:bg-surface-2", selected === d.id && "bg-surface-2 font-medium")} title={d.text}>
          <span className="font-mono">{d.id}</span> {d.title}{run.relevant.includes(d.id) && <span className="text-ok"> ✓ relevant</span>}
        </button>
      ))}
    </div>
  );
}

function StepCard({ run, phase, selected, k }: { run: Run; phase: Phase; selected: string; k: number }) {
  const doc = run.bm.scored.find((d) => d.id === selected)!;
  if (phase === "tokenize") {
    return (
      <div className="space-y-1 text-sm">
        <p>Keyword search treats <span className="font-mono">{run.bm.qTerms.join(", ")}</span> as strings to match exactly. The embedding converts known words into {CONCEPTS.length} concept axes and ignores {run.oov.length ? <span className="font-mono">{run.oov.join(", ")}</span> : "nothing"}.</p>
        <p className="text-xs text-muted">That difference is the whole story of hybrid search: identifiers like error codes are invisible to a model that never learned them, and paraphrases are invisible to exact matching.</p>
      </div>
    );
  }
  if (phase === "idf") {
    const maxI = Math.log(1 + (run.bm.N + 0.5) / 0.5);
    return (
      <div className="space-y-1.5">
        {run.bm.qTerms.map((t, i) => {
          const df = run.bm.dfs.get(t)!;
          const iv = Math.log(1 + (run.bm.N - df + 0.5) / (df + 0.5));
          return (
            <div key={t} className="grid grid-cols-[84px_minmax(0,1fr)_92px] items-center gap-2 text-xs">
              <span className="truncate font-mono">{t}</span>
              <div className="h-4 rounded bg-surface-2"><motion.div className="h-full rounded bg-accent/70" initial={{ width: 0 }} animate={{ width: `${(iv / maxI) * 100}%` }} transition={{ delay: T.idf(i) * k, duration: 0.5 * k, ease: EASE }} /></div>
              <span className="text-right font-mono">in {df}/{run.bm.N} · <AnimatedNumber value={iv} from={0} delay={T.idf(i) * k} duration={0.5 * k} /></span>
            </div>
          );
        })}
      </div>
    );
  }
  if (phase === "lists") {
    const topB = run.bmList.slice(0, run.topN).map((r) => r.id), topV = run.vecList.slice(0, run.topN).map((r) => r.id);
    const groups = [
      ["Only BM25 top " + run.topN, topB.filter((id) => !topV.includes(id)), colorOf.bm25],
      ["In both", topB.filter((id) => topV.includes(id)), "var(--fg)"],
      ["Only vector top " + run.topN, topV.filter((id) => !topB.includes(id)), colorOf.vector],
    ] as const;
    return (
      <div className="grid gap-2 sm:grid-cols-3">
        {groups.map(([label, ids, c], gi) => (
          <div key={label} className="rounded-md border border-line p-2">
            <div className="mb-1 text-[11px] font-semibold" style={{ color: c }}>{label}</div>
            <div className="flex flex-wrap gap-1">
              {ids.length ? ids.map((id, j) => (
                <motion.span key={id} className="rounded border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-xs" initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (0.3 + gi * 0.35 + j * 0.15) * k, type: "spring", stiffness: 300, damping: 18 }}>
                  {id}{run.relevant.includes(id) && <span className="text-ok"> ✓</span>}
                </motion.span>
              )) : <span className="text-xs text-muted">none</span>}
            </div>
          </div>
        ))}
        <p className="text-xs text-muted sm:col-span-3">Fusion will reward documents that appear high in both lists, and still keep strong documents that only one list found.</p>
      </div>
    );
  }
  if (phase === "bm25") {
    const max = Math.max(1e-9, ...doc.terms.map((t) => t.part));
    const norm = 1 - run.params.b + (run.params.b * doc.dl) / run.bm.avgdl;
    return (
      <div>
        <p className="mb-2 text-xs"><span className="font-mono font-semibold">{doc.id}</span> has {doc.dl} terms (avg {run.bm.avgdl.toFixed(1)}), so its length factor is <span className="font-mono">{norm.toFixed(2)}</span>{norm > 1 ? " (longer than average, matches count a bit less)" : " (shorter than average, matches count a bit more)"}. Click any row to inspect another document.</p>
        <div className="space-y-1">
          {doc.terms.map((t, i) => (
            <div key={t.term} className="grid grid-cols-[84px_minmax(0,1fr)_110px] items-center gap-2 text-xs">
              <span className={cx("truncate font-mono", !t.tf && "text-muted")}>{t.term}</span>
              <div className="h-4 rounded bg-surface-2"><motion.div className="h-full rounded bg-accent/70" initial={{ width: 0 }} animate={{ width: `${(t.part / max) * 100}%` }} transition={{ delay: T.bar(i) * k, duration: 0.5 * k, ease: EASE }} /></div>
              <span className="text-right font-mono text-[11px]">tf {t.tf} → <AnimatedNumber value={t.part} from={0} delay={T.bar(i) * k} duration={0.5 * k} /></span>
            </div>
          ))}
        </div>
        <p className="mt-2 border-t border-line pt-1 text-right font-mono text-xs">BM25({doc.id}) = {doc.score.toFixed(3)}</p>
      </div>
    );
  }
  if (phase === "vector") {
    const dv = embed(`${DOCS.find((d) => d.id === selected)!.title} ${DOCS.find((d) => d.id === selected)!.text}`);
    return (
      <div>
        <p className="mb-2 text-xs">Query (violet) vs <span className="font-mono font-semibold">{selected}</span> (grey) on each concept axis. Cosine = sum of the products, since both vectors have length 1.</p>
        <div className="space-y-1">
          {CONCEPTS.map((c, i) => (run.qVec[i] > 0 || dv[i] > 0) && (
            <div key={c} className="grid grid-cols-[96px_minmax(0,1fr)_56px] items-center gap-2 text-[11px]">
              <span className="truncate text-muted">{c}</span>
              <div className="space-y-0.5">
                <div className="h-1.5 rounded bg-surface-2"><motion.div className="h-full rounded bg-violet/70" initial={{ width: 0 }} animate={{ width: `${run.qVec[i] * 100}%` }} transition={{ delay: T.bar(i) * k, duration: 0.5 * k }} /></div>
                <div className="h-1.5 rounded bg-surface-2"><motion.div className="h-full rounded bg-muted/60" initial={{ width: 0 }} animate={{ width: `${dv[i] * 100}%` }} transition={{ delay: T.bar(i) * k, duration: 0.5 * k }} /></div>
              </div>
              <span className="text-right font-mono">{(run.qVec[i] * dv[i]).toFixed(3)}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 border-t border-line pt-1 text-right font-mono text-xs">cos = <AnimatedNumber value={run.vec.find((v) => v.id === selected)!.score} from={0} delay={1 * k} duration={0.6 * k} digits={3} /></p>
      </div>
    );
  }
  if (phase === "fuse") {
    return (
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">doc</th><th>BM25 rank</th><th>vector rank</th><th className="text-right">from BM25</th><th className="text-right">from vector</th><th className="text-right">fused</th></tr></thead>
        <tbody className="font-mono">
          {run.fused.map((f, i) => (
            <motion.tr key={f.id} className="border-t border-line" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (T.ghost(i) + T.fly * 0.8) * k, duration: 0.3 * k }}>
              <td className="py-0.5">{f.id}</td><td>{f.bm25Rank ?? "—"}</td><td>{f.vecRank ?? "—"}</td>
              <td className="text-right">{f.fromBm25.toFixed(run.mode === "rrf" ? 4 : 3)}</td><td className="text-right">{f.fromVec.toFixed(run.mode === "rrf" ? 4 : 3)}</td><td className="text-right font-semibold">{f.score.toFixed(run.mode === "rrf" ? 4 : 3)}</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    );
  }
  const rows = [["BM25", run.recall.bm25, colorOf.bm25], ["Vector", run.recall.vector, colorOf.vector], ["Fused", run.recall.fused, colorOf.fused]] as const;
  return run.relevant.length ? (
    <div className="space-y-1.5">
      {rows.map(([l, v, c], i) => (
        <div key={l} className="grid grid-cols-[64px_minmax(0,1fr)_48px] items-center gap-2 text-xs">
          <span>{l}</span>
          <div className="h-4 rounded bg-surface-2"><motion.div className="h-full rounded" style={{ background: c }} initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: (0.2 + i * 0.3) * k, duration: 0.6 * k, ease: EASE }} /></div>
          <span className="text-right font-mono">{Math.round(v * 100)}%</span>
        </div>
      ))}
      <p className="text-xs text-muted">recall@{run.topN} = share of the {run.relevant.length} relevant document(s) found in each list's top {run.topN}. Try the other presets: each method wins a different one.</p>
    </div>
  ) : <p className="text-sm text-muted">No labels for a custom query. Look at where your expected document lands in each column.</p>;
}

function Calc({ run, phase, selected }: { run: Run; phase: Phase; selected: string }) {
  const { k1, b } = run.params;
  if (phase === "tokenize") return <TexBlock>{`\\mathbf{q} = \\frac{\\sum_{w \\in q} \\mathbf{e}(w)}{\\lVert \\cdot \\rVert} = [${run.qVec.map((x) => x.toFixed(2)).join(",\\,")}]`}</TexBlock>;
  if (phase === "idf") {
    return <TexBlock>{`\\begin{aligned} \\text{idf}(t) &= \\ln\\!\\left(1 + \\frac{N - n_t + 0.5}{n_t + 0.5}\\right) \\\\ ${run.bm.qTerms.slice(0, 6).map((t) => { const df = run.bm.dfs.get(t)!; return `\\text{idf}(\\text{${t}}) &= \\ln\\!\\left(1 + \\tfrac{${run.bm.N} - ${df} + 0.5}{${df} + 0.5}\\right) = ${Math.log(1 + (run.bm.N - df + 0.5) / (df + 0.5)).toFixed(3)}`; }).join(" \\\\ ")} \\end{aligned}`}</TexBlock>;
  }
  if (phase === "bm25" || phase === "lists") {
    const doc = run.bm.scored.find((d) => d.id === selected)!;
    const hit = doc.terms.find((t) => t.tf > 0);
    return (
      <div className="space-y-2">
        <TexBlock>{"\\text{BM25}(D,Q) = \\sum_{t \\in Q} \\text{idf}(t)\\cdot\\frac{f(t,D)\\,(k_1+1)}{f(t,D) + k_1\\left(1 - b + b\\,\\frac{|D|}{\\text{avgdl}}\\right)}"}</TexBlock>
        {hit ? (
          <TexBlock>{`\\begin{aligned} &\\text{${doc.id}, term “${hit.term}”:} \\\\ &${hit.idf.toFixed(3)} \\cdot \\frac{${hit.tf}\\,(${k1}+1)}{${hit.tf} + ${k1}\\left(1 - ${b} + ${b}\\cdot\\frac{${doc.dl}}{${run.bm.avgdl.toFixed(2)}}\\right)} = ${hit.part.toFixed(3)} \\\\ &\\text{BM25}(${doc.id}) = ${doc.terms.filter((t) => t.tf).map((t) => t.part.toFixed(3)).join(" + ")} = ${doc.score.toFixed(3)} \\end{aligned}`}</TexBlock>
        ) : <p className="text-xs text-muted">{doc.id} shares no query term, so its BM25 score is 0 and it is not in the keyword list.</p>}
      </div>
    );
  }
  if (phase === "vector") {
    const d = DOCS.find((x) => x.id === selected)!;
    const dv = embed(`${d.title} ${d.text}`);
    const terms = CONCEPTS.map((_, i) => [run.qVec[i], dv[i]]).filter(([a, c]) => a > 0 && c > 0);
    return <TexBlock>{`\\cos(\\mathbf{q}, \\mathbf{d}_{${d.id.slice(1)}}) = ${terms.length ? terms.map(([a, c]) => `${a.toFixed(2)}\\cdot${c.toFixed(2)}`).join(" + ") : "0"} = ${run.vec.find((v) => v.id === d.id)!.score.toFixed(3)}`}</TexBlock>;
  }
  if (phase === "fuse" || phase === "compare") {
    const top = run.fused.slice(0, 3);
    return run.mode === "rrf" ? (
      <TexBlock>{`\\begin{aligned} \\text{RRF}(d) &= \\sum_{\\text{lists}} \\frac{1}{k + \\text{rank}(d)} \\\\ ${top.map((f) => `\\text{${f.id}} &= ${f.bm25Rank ? `\\tfrac{1}{${run.kRrf}+${f.bm25Rank}}` : "0"} + ${f.vecRank ? `\\tfrac{1}{${run.kRrf}+${f.vecRank}}` : "0"} = ${f.score.toFixed(4)}`).join(" \\\\ ")} \\end{aligned}`}</TexBlock>
    ) : (
      <TexBlock>{`\\begin{aligned} s(d) &= \\alpha\\,\\frac{\\text{BM25}(d)}{\\max \\text{BM25}} + (1-\\alpha)\\cos(\\mathbf{q},\\mathbf{d}) \\\\ ${top.map((f) => `\\text{${f.id}} &= ${f.fromBm25.toFixed(3)} + ${f.fromVec.toFixed(3)} = ${f.score.toFixed(3)}`).join(" \\\\ ")} \\end{aligned}`}</TexBlock>
    );
  }
  return null;
}
