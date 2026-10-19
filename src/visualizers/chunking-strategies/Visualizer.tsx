import { useCallback, useMemo, useState } from "react";
import { motion } from "motion/react";
import { CheckCircle2, RotateCcw, Scissors, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../document-ingestion/kit";
import { DEFAULT_PARAMS, STRATEGIES, adjacentSimilarity, compareAll, simulate, type Chunk, type ChunkParams, type ChunkPhase, type Fact, type Strategy } from "./simulation";

// ---------- one schedule ----------
const CUT = 0.3;
const cutAt = (i: number) => 0.4 + i * CUT;
const cardAt = (i: number) => 0.3 + i * 0.12;
const factAt = (i: number) => 0.3 + i * 0.45;

const TINTS = ["color-mix(in srgb, var(--accent) 16%, transparent)", "color-mix(in srgb, var(--violet) 18%, transparent)", "color-mix(in srgb, var(--ok) 16%, transparent)"];
const SOLID = ["var(--accent)", "var(--violet)", "var(--ok)"];

type Sim = ReturnType<typeof simulate>;

function narrate(phase: ChunkPhase, sim: Sim) {
  const { stats: st, params: p, chunks } = sim;
  switch (phase) {
    case "doc": return { headline: `One handbook: ${sim.doc.words.length} words, 4 sections, 5 facts to protect`, sub: "Underlined sentences are what users will ask about. A chunk must hold a whole fact for retrieval to return it." };
    case "candidates": return { headline: p.strategy === "semantic" ? "Embed every sentence and look for similarity drops" : p.strategy === "fixed" || p.strategy === "overlap" ? `Candidate cuts every ${p.strategy === "overlap" ? p.size - p.overlap : p.size} words, blind to meaning` : p.strategy === "heading" ? "Headings mark natural cut points" : "Only sentence ends are allowed as cut points", sub: STRATEGIES[p.strategy].rule };
    case "cut": return { headline: `${chunks.length - 1} cut lines slice the text into ${chunks.length} chunks`, sub: p.strategy === "fixed" ? "Watch where they land: some fall in the middle of a sentence." : "Each cut lands on a boundary the strategy allows." };
    case "overlap": return p.strategy === "overlap" ? { headline: `${st.overlapWords} words are stored twice`, sub: `Each chunk repeats the previous chunk's last ${p.overlap} words, so a fact near a cut has a second chance.` } : { headline: "No overlap: every word lives in exactly one chunk", sub: "Overlap trades extra storage and duplicate hits for fewer facts cut in half." };
    case "cards": return { headline: `${chunks.length} chunks become ${chunks.length} retrievable cards`, sub: "Each card is embedded and retrieved on its own. It no longer knows what came before it." };
    case "sizes": return { headline: `Sizes from ${st.min} to ${st.max} words (mean ${st.mean.toFixed(1)})`, sub: p.strategy === "fixed" || p.strategy === "overlap" ? "Fixed-size chunks are uniform except the tail." : "Respecting structure makes sizes uneven: short sections give short chunks." };
    case "facts": return { headline: st.split.length ? `${st.split.length} of ${sim.facts.length} facts are cut by a boundary` : "Every fact sits whole inside one chunk", sub: st.split.length ? "A question about a split fact retrieves half an answer, or the wrong half." : "This strategy and size keep every protected fact intact." };
    case "compare": return { headline: "Five strategies, same document, same size budget", sub: "Compare chunk counts, size spread and how many facts each one cuts." };
  }
}

export default function ChunkingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<ChunkParams>(DEFAULT_PARAMS);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(params), [params]);
  const comparison = useMemo(() => compareAll(sim.params), [sim.params]);
  const sentenceStarts = useMemo(() => new Set(sim.doc.sentences.map((s) => s.start)), [sim.doc]);
  const duration = useCallback((i: number) => {
    const ph = sim.steps[i]?.state.phase;
    const n = sim.chunks.length;
    const ms: Record<ChunkPhase, number> = { doc: 2400, candidates: 3000, cut: (cutAt(n) + 1) * 1000, overlap: 2800, cards: (cardAt(n) + 1.4) * 1000, sizes: 3200, facts: (factAt(5) + 1.2) * 1000, compare: 3800 };
    return ph ? ms[ph] : 3000;
  }, [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${JSON.stringify(sim.params)}-${step.id}-${replay}`;
  const story = narrate(phase, sim);
  const set = (p: Partial<ChunkParams>) => setParams((v) => ({ ...v, ...p }));
  const cardView = phase === "cards" || phase === "sizes" || phase === "compare";
  const midStarts = sim.chunks.filter((c) => !sentenceStarts.has(c.start)).length;

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title={`Chunking: ${STRATEGIES[sim.params.strategy].name}`} actions={sim.params.strategy === "semantic" ? <SimulatedLabel>Hashed bag-of-words embedding</SimulatedLabel> : undefined}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={`${sim.params.strategy}-${step.id}`} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            {cardView ? <Cards sim={sim} k={k} animKey={animKey} sentenceStarts={sentenceStarts} /> : <TextView sim={sim} phase={phase} k={k} animKey={animKey} />}
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
              <StageDetail phase={phase} sim={sim} comparison={comparison} k={k} midStarts={midStarts} sentenceStarts={sentenceStarts} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Chunker settings" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_PARAMS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Strategy" value={params.strategy} onChange={(v) => set({ strategy: v })} options={(Object.keys(STRATEGIES) as Strategy[]).map((s) => ({ value: s, label: STRATEGIES[s].name }))} />
              <Slider label="Chunk size (max words)" value={params.size} min={12} max={80} onChange={(v) => set({ size: v })} hint="Try 15: even structure-aware chunkers start cutting facts." />
              <Slider label={`Overlap (words)${params.strategy === "overlap" ? "" : " · only for fixed + overlap"}`} value={params.overlap} min={0} max={20} onChange={(v) => set({ overlap: v })} />
              <Slider label={`Similarity threshold τ${params.strategy === "semantic" ? "" : " · only for semantic"}`} value={params.threshold} min={0} max={0.6} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ threshold: v })} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why="The chunk is the unit of retrieval. Whatever a chunk cuts apart, the retriever can only ever return in pieces." />
          <Panel title="Calculation"><Calculation sim={sim} phase={phase} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["chunks", sim.stats.count], ["mean size", `${sim.stats.mean.toFixed(1)} words`], ["min / max", `${sim.stats.min} / ${sim.stats.max}`],
              ["words stored twice", sim.stats.overlapWords], ["chunks starting mid-sentence", midStarts], ["facts split", `${sim.stats.split.length}/${sim.facts.length} (${(sim.stats.splitPct * 100).toFixed(0)}%)`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/** The document as flowing text, with chunk tints, cut lines, overlap and fact highlights. */
function TextView({ sim, phase, k, animKey }: { sim: Sim; phase: ChunkPhase; k: number; animKey: string }) {
  const { doc, chunks, facts, candidates, stats: st } = sim;
  const showChunks = phase !== "doc" && phase !== "candidates";
  const chunkOf = (w: number) => { let c = -1; chunks.forEach((ch, i) => { if (ch.start <= w && w < ch.end) c = i; }); return c; };
  const inTwo = (w: number) => chunks.filter((ch) => ch.start <= w && w < ch.end).length > 1;
  const factOf = (w: number) => facts.find((f) => f.start <= w && w < f.end);
  const cutsAt = new Map(chunks.slice(1).map((c, i) => [c.start, i]));
  const endsAt = sim.params.strategy === "overlap" ? new Map(chunks.slice(0, -1).map((c, i) => [c.end, i])) : new Map<number, number>();
  const candSet = new Set(candidates);
  const blocks: { heading: boolean; words: number[] }[] = [];
  doc.words.forEach((w, i) => {
    const last = blocks[blocks.length - 1];
    if (w.heading) { if (!last || !last.heading || doc.words[i - 1]?.sentence !== w.sentence) blocks.push({ heading: true, words: [i] }); else last.words.push(i); }
    else if (!last || last.heading) blocks.push({ heading: false, words: [i] });
    else last.words.push(i);
  });
  const bar = (i: number, color: string, dashed = false, delay = 0) => (
    <span className="relative inline-block w-0 align-middle" aria-hidden>
      <motion.span key={`${animKey}-bar-${i}-${dashed}`} className="absolute -top-[0.95em] left-[-1.5px] block h-[1.5em] w-[3px] rounded-full" style={{ background: dashed ? "transparent" : color, borderLeft: dashed ? `2px dashed ${color}` : undefined, transformOrigin: "top" }}
        initial={phase === "cut" ? { scaleY: 0, opacity: 0 } : false} animate={{ scaleY: 1, opacity: 1 }} transition={{ delay: phase === "cut" ? delay * k : 0, duration: 0.3 * k, ease }} />
    </span>
  );
  return (
    <div className="space-y-2 rounded-md border border-line bg-surface p-3 text-[13px] leading-[1.9]">
      {blocks.map((b, bi) => (
        <p key={bi} className={cx(b.heading && "text-sm font-semibold text-violet")}>
          {b.words.map((wi) => {
            const w = doc.words[wi];
            const c = chunkOf(wi);
            const f = factOf(wi);
            const split = f && st.split.includes(f.id);
            const cutIx = cutsAt.get(wi);
            const endIx = endsAt.get(wi);
            const dup = showChunks && phase === "overlap" && inTwo(wi);
            return (
              <span key={wi}>
                {phase === "candidates" && candSet.has(wi) && (
                  <span className="relative inline-block w-0 align-middle" aria-hidden>
                    <motion.span key={`${animKey}-cand-${wi}`} className="absolute -top-[0.95em] left-[-1px] block h-[1.5em] w-[2px] bg-muted" initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 0.45] }} transition={{ delay: (0.3 + candidates.indexOf(wi) * 0.12) * k, duration: 0.6 * k }} />
                  </span>
                )}
                {showChunks && endIx !== undefined && bar(wi, SOLID[endIx % 3], true, cutAt(endIx) + 0.15)}
                {showChunks && cutIx !== undefined && bar(wi, SOLID[(cutIx + 1) % 3], false, cutAt(cutIx))}
                <motion.span
                  className={cx("rounded-sm px-[1px]", f && "underline decoration-dotted underline-offset-4", dup && "outline outline-1 outline-warn")}
                  initial={false}
                  animate={{ backgroundColor: phase === "facts" && f ? (split ? "color-mix(in srgb, var(--err) 22%, transparent)" : "color-mix(in srgb, var(--ok) 22%, transparent)") : showChunks && c >= 0 ? TINTS[c % 3] : "rgba(0,0,0,0)" }}
                  transition={{ delay: phase === "cut" ? cutAt(Math.max(0, c - 1)) * k : phase === "facts" && f ? factAt(facts.indexOf(f)) * k : 0, duration: 0.3 * k }}
                >{w.text}</motion.span>{" "}
              </span>
            );
          })}
        </p>
      ))}
      <div className="flex flex-wrap gap-3 border-t border-line pt-2 text-[11px] text-muted">
        <span className="flex items-center gap-1"><Scissors size={12} /> solid bar = chunk starts here</span>
        {sim.params.strategy === "overlap" && <span>dashed bar = previous chunk ends here</span>}
        <span><span className="underline decoration-dotted underline-offset-4">dotted underline</span> = protected fact</span>
        {phase === "overlap" && <span className="rounded-sm px-1 outline outline-1 outline-warn">outlined</span>}{phase === "overlap" && <span>= stored in two chunks</span>}
      </div>
    </div>
  );
}

function Cards({ sim, k, animKey, sentenceStarts }: { sim: Sim; k: number; animKey: string; sentenceStarts: Set<number> }) {
  const { chunks, doc, facts } = sim;
  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {chunks.map((c, i) => {
        const inside = facts.filter((f) => c.start <= f.start && c.end >= f.end);
        const partial = facts.filter((f) => !inside.includes(f) && f.start < c.end && f.end > c.start);
        const text = doc.words.slice(c.start, c.end).map((w) => w.text).join(" ");
        return (
          <motion.div key={`${animKey}-${i}`} initial={{ opacity: 0, y: 24, scale: 0.92 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ delay: cardAt(i) * k, type: "spring", stiffness: 220, damping: 22 }}
            className="rounded-md border-l-4 border border-line bg-surface p-2" style={{ borderLeftColor: SOLID[i % 3] }}>
            <div className="mb-1 flex flex-wrap items-center gap-1 text-[11px]">
              <span className="font-mono font-semibold">#{i + 1}</span><Badge>{c.end - c.start} words</Badge>
              {!sentenceStarts.has(c.start) && <Badge tone="warn">starts mid-sentence</Badge>}
              {inside.map((f) => <Badge key={f.id} tone="ok"><CheckCircle2 size={10} /> {f.id}</Badge>)}
              {partial.map((f) => <Badge key={f.id} tone="err"><XCircle size={10} /> part of {f.id}</Badge>)}
            </div>
            <p className="line-clamp-4 text-[11.5px] leading-snug text-muted">{text}</p>
          </motion.div>
        );
      })}
    </div>
  );
}

/** A horizontal map of the document: x = word position. Used to show candidates, chunks as lanes and overlaps. */
function Ruler({ sim, k, mode }: { sim: Sim; k: number; mode: "plain" | "candidates" | "lanes" }) {
  const N = sim.doc.words.length, W = 600, x = (w: number) => 10 + (w / N) * (W - 20);
  const lanes = mode === "lanes" ? 2 : 0;
  const H = 46 + lanes * 18;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Document ruler showing sentences, cut candidates and chunks">
      {sim.doc.sentences.map((s) => <rect key={s.index} x={x(s.start)} y={12} width={Math.max(1, x(s.end) - x(s.start) - 1)} height={10} rx={2} fill={s.heading ? "var(--violet)" : "var(--line)"} />)}
      {sim.facts.map((f) => <rect key={f.id} x={x(f.start)} y={24} width={x(f.end) - x(f.start)} height={3} fill={sim.stats.split.includes(f.id) ? "var(--err)" : "var(--ok)"} />)}
      <text x={10} y={8} fontSize={9} fill="var(--muted)">word 0</text><text x={W - 10} y={8} fontSize={9} textAnchor="end" fill="var(--muted)">word {N}</text>
      {mode === "candidates" && sim.candidates.map((c, i) => (
        <motion.line key={c} x1={x(c)} x2={x(c)} y1={4} y2={34} stroke="var(--fg)" strokeWidth={1.5} initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: (0.3 + i * 0.12) * k, duration: 0.25 * k }} />
      ))}
      {mode === "lanes" && sim.chunks.map((c, i) => (
        <motion.rect key={i} x={x(c.start)} y={34 + (i % 2) * 18} height={14} rx={3} fill={SOLID[i % 3]} fillOpacity={0.7}
          initial={{ width: 0 }} animate={{ width: Math.max(2, x(c.end) - x(c.start)) }} transition={{ delay: cutAt(i) * k, duration: 0.35 * k, ease }} />
      ))}
      {mode === "lanes" && sim.chunks.map((c, i) => <text key={`t${i}`} x={x(c.start) + 3} y={44 + (i % 2) * 18} fontSize={9} fill="var(--surface)">#{i + 1}</text>)}
    </svg>
  );
}

function StageDetail({ phase, sim, comparison, k, midStarts, sentenceStarts }: { phase: ChunkPhase; sim: Sim; comparison: ReturnType<typeof compareAll>; k: number; midStarts: number; sentenceStarts: Set<number> }) {
  const { stats: st, params: p } = sim;
  switch (phase) {
    case "doc":
      return (
        <div className="space-y-2">
          <Ruler sim={sim} k={k} mode="plain" />
          <ul className="grid gap-1 text-xs sm:grid-cols-2">
            {sim.facts.map((f, i) => <motion.li key={f.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.2) * k }}><span className="font-mono text-accent">{f.id}</span> {f.label} <span className="text-muted">(words {f.start}–{f.end})</span></motion.li>)}
          </ul>
          <p className="text-[11px] text-muted">Ruler: grey = sentences, violet = headings, the thin bar below = fact spans.</p>
        </div>
      );
    case "candidates":
      return p.strategy === "semantic" ? <SimilarityBars sim={sim} k={k} /> : (
        <div className="space-y-1">
          <Ruler sim={sim} k={k} mode="candidates" />
          <p className="text-xs text-muted">{sim.candidates.length} allowed cut positions (black ticks). {p.strategy === "fixed" || p.strategy === "overlap" ? "They ignore sentences and headings entirely." : p.strategy === "sentence" ? `Packing then picks the last sentence end that keeps each chunk ≤ ${p.size} words.` : `Inside a section longer than ${p.size} words, sentence ends are used too.`}</p>
        </div>
      );
    case "cut":
    case "overlap":
      return (
        <div className="space-y-1">
          <Ruler sim={sim} k={k} mode="lanes" />
          <p className="text-xs text-muted">Each coloured bar is one chunk, alternating lanes so overlaps show. {phase === "overlap" && (st.overlapWords ? `Where bars overlap, ${st.overlapWords} words are stored in both chunks.` : "The bars only touch: no overlap.")}</p>
        </div>
      );
    case "cards":
      return (
        <div className="grid gap-3 sm:grid-cols-3">
          {[["chunks", st.count, ""], ["start mid-sentence", midStarts, midStarts ? "text-warn" : "text-ok"], ["words stored twice", st.overlapWords, ""]].map(([l, v, cls], i) => (
            <div key={l as string} className="rounded-md bg-surface-2 p-2 text-center">
              <div className={cx("font-mono text-2xl", cls as string)}><AnimatedNumber value={v as number} from={0} digits={0} delay={(0.2 + i * 0.3) * k} duration={0.8 * k} /></div>
              <div className="text-[11px] text-muted">{l}</div>
            </div>
          ))}
          <p className="text-xs text-muted sm:col-span-3">A chunk that starts mid-sentence ("{sim.chunks.find((c) => !sentenceStarts.has(c.start)) ? sim.doc.words.slice(sim.chunks.find((c) => !sentenceStarts.has(c.start))!.start, sim.chunks.find((c) => !sentenceStarts.has(c.start))!.start + 4).map((w) => w.text).join(" ") + " …" : "none here"}") embeds poorly and reads oddly when quoted.</p>
        </div>
      );
    case "sizes": {
      const max = Math.max(p.size, ...st.sizes);
      return (
        <div className="space-y-2">
          <div className="relative flex h-28 items-end gap-1 rounded bg-surface-2 px-2 pt-2">
            <div className="absolute inset-x-2 border-t border-dashed border-err" style={{ bottom: `${(p.size / max) * 100 * 0.85 + 4}%` }}><span className="absolute -top-4 right-0 text-[10px] text-err">max {p.size}</span></div>
            {st.sizes.map((s, i) => (
              <motion.div key={i} className="relative flex-1 rounded-t" style={{ background: SOLID[i % 3], opacity: 0.75 }} initial={{ height: 0 }} animate={{ height: `${(s / max) * 85}%` }} transition={{ delay: (0.2 + i * 0.1) * k, duration: 0.5 * k, ease }}>
                <span className="absolute -top-4 inset-x-0 text-center font-mono text-[9px] text-fg">{s}</span>
              </motion.div>
            ))}
          </div>
          <KV rows={[["mean", st.mean.toFixed(1)], ["std dev", st.std.toFixed(1)], ["min / max", `${st.min} / ${st.max}`]]} />
        </div>
      );
    }
    case "facts":
      return (
        <div className="space-y-1">
          {sim.facts.map((f, i) => <FactRow key={f.id} f={f} split={st.split.includes(f.id)} chunks={sim.chunks} delay={factAt(i) * k} />)}
          <p className="pt-1 font-mono text-sm font-semibold">facts split: <AnimatedNumber value={st.splitPct * 100} from={0} digits={0} delay={0.3 * k} duration={factAt(4) * k} />%</p>
        </div>
      );
    case "compare":
      return (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">strategy</th><th className="text-right">chunks</th><th className="text-right">mean ± sd</th><th className="w-[38%] pl-3">facts split</th></tr></thead>
            <tbody>
              {comparison.map((r, i) => (
                <tr key={r.strategy} className={cx("border-t border-line", r.strategy === p.strategy && "bg-accent-soft/60 font-medium")}>
                  <td className="py-1">{STRATEGIES[r.strategy].name}</td>
                  <td className="text-right font-mono">{r.stats.count}</td>
                  <td className="text-right font-mono">{r.stats.mean.toFixed(0)} ± {r.stats.std.toFixed(0)}</td>
                  <td className="pl-3">
                    <div className="flex items-center gap-2">
                      <div className="h-3 flex-1 rounded bg-surface-2"><motion.div className={cx("h-full rounded", r.stats.split.length ? "bg-err/70" : "bg-ok/70")} initial={{ width: 0 }} animate={{ width: `${Math.max(2, r.stats.splitPct * 100)}%` }} transition={{ delay: (0.3 + i * 0.3) * k, duration: 0.5 * k, ease }} /></div>
                      <span className="w-9 text-right font-mono">{(r.stats.splitPct * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
  }
}

function FactRow({ f, split, chunks, delay }: { f: Fact; split: boolean; chunks: Chunk[]; delay: number }) {
  const holders = chunks.filter((c) => c.start < f.end && c.end > f.start).map((c) => `#${c.index + 1}`);
  return (
    <motion.div initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay }} className={cx("flex items-center gap-2 rounded px-2 py-1 text-xs", split ? "bg-err-soft" : "bg-ok-soft")}>
      {split ? <XCircle size={13} className="shrink-0 text-err" /> : <CheckCircle2 size={13} className="shrink-0 text-ok" />}
      <span className="font-mono text-accent">{f.id}</span><span className="min-w-0">{f.label}</span>
      <span className="ml-auto shrink-0 text-[10px] text-muted">{split ? `split over ${holders.join(" + ")}` : `whole in ${chunks.filter((c) => c.start <= f.start && c.end >= f.end).map((c) => `#${c.index + 1}`).join(", ")}`}</span>
    </motion.div>
  );
}

function SimilarityBars({ sim, k }: { sim: Sim; k: number }) {
  const sims = adjacentSimilarity(sim.doc.sentences);
  const tau = sim.params.threshold;
  const H = 120, W = 600, bw = (W - 40) / sims.length, y = (v: number) => H - 20 - v * (H - 34) / 0.7;
  return (
    <div className="space-y-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Similarity between adjacent sentences with threshold line">
        {sims.map((s, i) => {
          const cut = s.sim < tau;
          return (
            <g key={i}>
              <motion.rect x={30 + i * bw + 3} width={bw - 6} rx={2} fill={cut ? "var(--err)" : "var(--accent)"} fillOpacity={0.75}
                initial={{ y: H - 20, height: 0 }} animate={{ y: y(s.sim), height: H - 20 - y(s.sim) }} transition={{ delay: (0.2 + i * 0.12) * k, duration: 0.4 * k, ease }} />
              <text x={30 + i * bw + bw / 2} y={H - 8} textAnchor="middle" fontSize={9} fill={cut ? "var(--err)" : "var(--muted)"}>{cut ? "✂" : i + 1}</text>
              <motion.text x={30 + i * bw + bw / 2} y={y(s.sim) - 3} textAnchor="middle" fontSize={8.5} fontFamily="var(--font-mono)" fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.5 + i * 0.12) * k }}>{s.sim.toFixed(2)}</motion.text>
            </g>
          );
        })}
        <line x1={26} x2={W - 6} y1={y(tau)} y2={y(tau)} stroke="var(--err)" strokeDasharray="5 3" />
        <text x={W - 8} y={y(tau) - 4} textAnchor="end" fontSize={10} fill="var(--err)">τ = {tau.toFixed(2)}</text>
        <text x={4} y={14} fontSize={9} fill="var(--muted)">cos</text>
      </svg>
      <p className="text-xs text-muted">Bar i = similarity between sentence i and the one before it (headings are merged into their first sentence). Red bars fall below τ: a cut goes there. Section changes share no words, so they score 0.</p>
    </div>
  );
}

function Calculation({ sim, phase }: { sim: Sim; phase: ChunkPhase }) {
  const { params: p, stats: st, doc } = sim;
  const N = doc.words.length;
  const step = p.strategy === "overlap" ? p.size - p.overlap : p.size;
  return (
    <div className="space-y-2">
      {(p.strategy === "fixed" || p.strategy === "overlap") && (
        <TexBlock>{`\\begin{aligned} \\text{step} &= ${p.strategy === "overlap" ? `\\text{size} - \\text{overlap} = ${p.size} - ${p.overlap}` : "\\text{size}"} = ${step} \\\\ n_{\\text{chunks}} &= \\left\\lceil \\tfrac{N - \\text{size}}{\\text{step}} \\right\\rceil + 1 = \\left\\lceil \\tfrac{${N} - ${p.size}}{${step}} \\right\\rceil + 1 = ${Math.max(0, Math.ceil((N - p.size) / step)) + 1} \\end{aligned}`}</TexBlock>
      )}
      {p.strategy === "semantic" && <TexBlock>{`\\text{cut before } s_i \\iff \\cos(\\mathbf{e}_{i-1}, \\mathbf{e}_i) < \\tau = ${p.threshold.toFixed(2)}`}</TexBlock>}
      {(p.strategy === "sentence" || p.strategy === "heading") && <TexBlock>{`\\text{add } s_i \\text{ to chunk} \\iff |\\text{chunk}| + |s_i| \\le ${p.size}${p.strategy === "heading" ? " \\wedge s_i \\ne \\text{heading}" : ""}`}</TexBlock>}
      {phase === "sizes" && <TexBlock>{`\\bar{s} = \\tfrac{1}{n}\\textstyle\\sum_j s_j = \\tfrac{${st.sizes.reduce((a, b) => a + b, 0)}}{${st.count}} = ${st.mean.toFixed(1)}`}</TexBlock>}
      <TexBlock>{`\\text{split} = \\frac{|\\{f : \\nexists\\, c,\\ f \\subseteq c\\}|}{|F|} = \\frac{${st.split.length}}{${sim.facts.length}} = ${(st.splitPct * 100).toFixed(0)}\\%`}</TexBlock>
    </div>
  );
}
