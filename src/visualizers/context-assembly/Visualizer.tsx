import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, RotateCcw, Scissors, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../top-k-retrieval/StepCaption";
import { CHUNKS, DEFAULT_INPUT, FACT, QUERY, SIM, simulate, type Assembly, type AsmState, type AssemblyInput, type Decision, type Ordering, type Overflow, type Strategy } from "./simulation";

/** Shared schedule (seconds at 1×). MMR first shows its score table, then commits the pick. */
const T = { pulse: 0.1, seg: (mmr: boolean) => (mmr ? 1.5 : 0.45), segDur: 0.7, mmrRow: (i: number) => 0.15 + i * 0.1 };
const DUR = { candidates: 2600, consider: 2400, considerMmr: 3300, order: 2600, result: 2800 };
const PALETTE = ["var(--accent)", "var(--violet)", "var(--ok)", "var(--warn)"];
const colorOf = (id: string) => PALETTE[(Number(id.slice(1)) - 1) % PALETTE.length];
/** Illustrative U-shaped "attention by position" (Liu et al., 2023 found mid-context facts are used least). Not measured. */
const uCurve = (pos: number, n: number) => (n <= 1 ? 1 : 0.45 + 0.55 * (2 * (pos / (n - 1)) - 1) ** 2);

export default function ContextAssemblyVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<AssemblyInput>(DEFAULT_INPUT);
  const [replay, setReplay] = useState(0);
  const { asm, steps } = useMemo(() => simulate(input), [input]);
  const mmr = asm.input.strategy === "mmr";
  const pb = usePlayback(steps.length, (i) => { const p = steps[i]?.state.phase ?? "consider"; return p === "consider" ? (mmr ? DUR.considerMmr : DUR.consider) : DUR[p]; });
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<AssemblyInput>) => setInput((v) => ({ ...v, ...p }));
  const prevUsed = pb.index > 0 ? steps[pb.index - 1].state.used : 0;

  if (tab === "metrics") return null;
  const story = narrate(s, asm);

  return (
    <Workspace
      main={
        <>
          <Panel title="Filling the context window" actions={<SimulatedLabel>Toy token counts</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <BudgetBar s={s} asm={asm} prevUsed={prevUsed} k={k} animKey={animKey} />
            <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
              <CandidateList s={s} k={k} animKey={animKey} />
              <DroppedPanel s={s} k={k} />
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <StepCard s={s} asm={asm} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Strategy & parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(DEFAULT_INPUT); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Selection strategy" value={input.strategy} onChange={(v: Strategy) => { set({ strategy: v }); pb.restart(); }}
                options={[{ value: "greedy", label: "Greedy by rank" }, { value: "dedupe", label: "Dedupe near-duplicates, then greedy" }, { value: "mmr", label: "MMR (relevance vs redundancy)" }]} />
              <Slider label="Token budget" value={input.budget} min={60} max={260} step={5} onChange={(v) => set({ budget: v })} />
              <Select label="When a chunk does not fit" value={input.overflow} onChange={(v: Overflow) => set({ overflow: v })} options={[{ value: "truncate", label: "Truncate it and stop" }, { value: "drop", label: "Drop it, try smaller ones" }]} />
              <Select label="Ordering in the prompt" value={input.ordering} onChange={(v: Ordering) => set({ ordering: v })} options={[{ value: "relevance", label: "Most relevant first" }, { value: "ends", label: "Best at both ends (lost in the middle)" }]} />
              {input.strategy === "mmr" && <Slider label="MMR λ (1 = pure relevance)" value={input.lambda} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ lambda: v })} hint="Low λ chases diversity and can pick off-topic filler." />}
              {input.strategy === "dedupe" && <Slider label="Duplicate threshold (Jaccard)" value={input.dupThreshold} min={0.2} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ dupThreshold: v })} hint="Too low and distinct chunks get thrown away." />}
            </div>
            <p className="mt-2 text-xs text-muted">Query: “{QUERY}” · the answer needs the fact “{FACT}” (only in C5).</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="The model only sees what fits in the window. Wasting budget on duplicates or cutting the wrong chunk loses the answer before generation starts." />
          <Panel title="Calculation"><Calc s={s} asm={asm} /></Panel>
          <Panel title="Key values">
            <KV rows={[["budget B", asm.input.budget], ["used", s.used], ["chunks kept", s.picked.length], ["dropped / skipped", s.dropped.length], ["answer fact", asm.factChunk ? `in ${asm.factChunk.id}` : "lost"]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(s: AsmState, asm: Assembly) {
  const B = asm.input.budget;
  if (s.phase === "candidates") return { headline: `8 chunks, ${CHUNKS.reduce((a, c) => a + c.tokens, 0)} tokens, ${B} tokens of room`, sub: CHUNKS.reduce((a, c) => a + c.tokens, 0) > B ? "They cannot all fit. The strategy decides which ones reach the model." : "At this budget everything fits; shrink it to see the strategies differ." };
  if (s.phase === "consider" && s.decision) {
    const d = s.decision, c = CHUNKS.find((x) => x.id === d.id)!;
    const head = asm.input.strategy === "mmr" ? `MMR picks ${d.id}` : `${d.id} (${c.tokens} tokens)`;
    if (d.action === "add") return { headline: `${head} slides into the budget`, sub: `${d.used} + ${c.tokens} = ${d.used + c.tokens} ≤ ${B}.` };
    if (d.action === "truncate") return { headline: `${head} is too big: it gets cut to fit`, sub: `Only ${d.room} tokens left. Everything after the cut is lost${CHUNKS.find((x) => x.id === d.id)!.text.includes(FACT) ? ", which may include the answer" : ""}.` };
    if (d.action === "drop-dup") return { headline: `${head} is a near-duplicate: skipped`, sub: `Jaccard similarity with ${d.maxSimWith} is ${d.maxSim!.toFixed(2)}. It would spend tokens on nothing new.` };
    return { headline: `${head} does not fit: dropped`, sub: `${c.tokens} > ${d.room} tokens left. A smaller later chunk may still fit.` };
  }
  if (s.phase === "order") return { headline: asm.input.ordering === "ends" ? "Strongest chunks move to the start and the end" : "Chunks are laid out best-first", sub: asm.middle ? "The answer chunk sits in the middle, where long-context models use information least." : "The answer chunk is not stuck in the middle." };
  return asm.factChunk
    ? { headline: `The answer fact survived in ${asm.factChunk.id}`, sub: `${asm.used}/${B} tokens used. ${asm.middle ? "It sits mid-context: a risk, not a guarantee of failure." : "Its position is favourable."}` }
    : { headline: "The answer fact was lost before the model saw anything", sub: "Try dedupe, a larger budget, or “drop” instead of “truncate”." };
}

function BudgetBar({ s, asm, prevUsed, k, animKey }: { s: AsmState; asm: Assembly; prevUsed: number; k: number; animKey: string }) {
  const B = asm.input.budget;
  const ordered = s.order ? s.order.map((id) => s.picked.find((p) => p.id === id)!) : s.picked;
  const newest = s.phase === "consider" && s.decision && (s.decision.action === "add" || s.decision.action === "truncate") ? s.decision.id : null;
  const delay = T.seg(asm.input.strategy === "mmr") * k;
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="font-medium">Context budget</span>
        <span className="font-mono"><AnimatedNumber key={animKey} value={s.used} from={s.phase === "consider" ? prevUsed : s.used} digits={0} delay={delay} duration={T.segDur * k} /> / {B} tokens</span>
      </div>
      <div className="flex h-10 overflow-hidden rounded-md border border-line bg-surface-2">
        {ordered.map((p) => {
          const isNew = p.id === newest;
          return (
            <motion.div key={isNew ? `${p.id}-${animKey}` : p.id} layout="position" className="relative flex h-full shrink-0 items-center justify-center overflow-hidden border-r-2 border-surface text-[10px] font-semibold text-white dark:text-black"
              style={{ background: colorOf(p.id) }}
              initial={isNew ? { width: 0 } : false} animate={{ width: `${(p.tokens / B) * 100}%` }}
              transition={{ width: { delay: isNew ? delay : 0, duration: T.segDur * k, ease: EASE }, layout: { duration: 0.9 * k, ease: EASE } }}>
              <span className="truncate px-0.5">{p.id}<span className="hidden sm:inline"> · {p.tokens}</span></span>
              {p.truncated && <span className="absolute inset-y-0 right-0 w-1.5 bg-err" title="truncated" />}
            </motion.div>
          );
        })}
      </div>
      {(s.phase === "order" || s.phase === "result") && ordered.length > 0 && (
        <div className="relative mt-1 h-10">
          <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden>
            {(() => {
              let x = 0;
              const pts = ordered.map((p, i) => { const cx = ((x + p.tokens / 2) / B) * 100; x += p.tokens; return [cx, 28 - uCurve(i, ordered.length) * 24] as const; });
              return <motion.polyline key={animKey} points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--muted)" strokeWidth={0.6} strokeDasharray="1.5 1" vectorEffect="non-scaling-stroke"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.9 * k, duration: 0.8 * k }} />;
            })()}
          </svg>
          <div className="absolute inset-x-0 bottom-0 flex">
            {ordered.map((p, i) => (
              <span key={p.id} className={cx("shrink-0 truncate text-center text-[10px]", asm.factChunk?.id === p.id ? "font-semibold text-ok" : "text-muted")} style={{ width: `${(p.tokens / B) * 100}%` }}>
                {i + 1}{asm.factChunk?.id === p.id ? " ★" : ""}
              </span>
            ))}
          </div>
          <span className="absolute right-0 top-0 text-[9px] text-muted">position-sensitivity (illustrative)</span>
        </div>
      )}
    </div>
  );
}

function CandidateList({ s, k, animKey }: { s: AsmState; k: number; animKey: string }) {
  const maxTok = Math.max(...CHUNKS.map((c) => c.tokens));
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Candidates (reranked)</div>
      <div className="space-y-1">
        {CHUNKS.map((c, i) => {
          const picked = s.picked.find((p) => p.id === c.id);
          const dropped = s.dropped.find((d) => d.id === c.id);
          const current = s.decision?.id === c.id;
          const hasFact = c.text.includes(FACT);
          return (
            <motion.div key={current ? `${c.id}-${animKey}` : c.id}
              className={cx("flex items-center gap-2 rounded-md border px-2 py-1 text-xs", current ? "border-accent ring-2 ring-accent/30" : "border-line", dropped && !picked && "opacity-50")}
              initial={s.phase === "candidates" ? { opacity: 0, x: -16 } : current ? { scale: 1 } : false}
              animate={current ? { scale: [1, 1.03, 1], x: dropped && !picked ? [0, -6, 6, -4, 0] : 0, opacity: 1 } : { opacity: dropped && !picked ? 0.5 : 1, x: 0 }}
              transition={{ delay: s.phase === "candidates" ? i * 0.12 * k : T.pulse * k, duration: 0.5 * k }}>
              <span className="h-3 w-1 shrink-0 rounded" style={{ background: colorOf(c.id) }} />
              <span className="w-6 shrink-0 font-mono font-semibold">{c.id}</span>
              <span className="min-w-0 flex-1 truncate" title={c.text}>{c.title}{hasFact && <span className="text-ok"> ★</span>}</span>
              <span className="hidden w-16 shrink-0 sm:block"><span className="block h-1.5 rounded bg-surface-2"><span className="block h-full rounded bg-muted/60" style={{ width: `${(c.tokens / maxTok) * 100}%` }} /></span></span>
              <span className="w-9 shrink-0 text-right font-mono text-[10px]">{c.tokens}t</span>
              <span className="w-12 shrink-0 text-right">
                {picked ? (picked.truncated ? <Badge tone="warn"><Scissors size={10} aria-hidden />cut</Badge> : <Badge tone="ok"><CheckCircle2 size={10} aria-hidden />in</Badge>) : dropped ? <Badge tone="err"><XCircle size={10} aria-hidden />out</Badge> : <span className="font-mono text-[10px] text-muted">{c.rel.toFixed(2)}</span>}
              </span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function DroppedPanel({ s, k }: { s: AsmState; k: number }) {
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Dropped or cut</div>
      <div className="min-h-16 space-y-1 rounded-md border border-dashed border-line p-1.5">
        <AnimatePresence initial={false}>
          {s.dropped.map((d) => (
            <motion.div key={d.id} layout initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ delay: 0.5 * k, duration: 0.4 * k }}
              className={cx("rounded border px-1.5 py-1 text-[11px]", d.kind === "duplicate" ? "border-warn/40 bg-warn-soft" : "border-err/30 bg-err-soft")}>
              <span className="font-mono font-semibold">{d.id}</span> {d.reason}{CHUNKS.find((c) => c.id === d.id)!.text.includes(FACT) && <strong className="text-err"> (had the answer)</strong>}
            </motion.div>
          ))}
        </AnimatePresence>
        {!s.dropped.length && <p className="p-1 text-[11px] text-muted">Nothing dropped yet.</p>}
      </div>
    </div>
  );
}

function StepCard({ s, asm, k }: { s: AsmState; asm: Assembly; k: number }) {
  if (s.phase === "candidates") return <SimMatrix k={k} />;
  if (s.phase === "consider" && s.decision) return s.decision.mmr ? <MmrTable d={s.decision} lambda={asm.input.lambda} k={k} /> : <FitCheck d={s.decision} asm={asm} k={k} />;
  if (s.phase === "order") {
    return (
      <div className="space-y-1">
        {asm.order.map((id, i) => {
          const a = uCurve(i, asm.order.length);
          return (
            <motion.div key={id} layout className="grid grid-cols-[70px_minmax(0,1fr)_80px] items-center gap-2 text-xs" transition={{ duration: 0.6 * k }}>
              <span className="font-mono">pos {i + 1}: {id}</span>
              <div className="h-3 rounded bg-surface-2"><motion.div className="h-full rounded bg-muted/60" initial={{ width: 0 }} animate={{ width: `${a * 100}%` }} transition={{ delay: (0.9 + i * 0.1) * k, duration: 0.5 * k }} /></div>
              <span className={cx("text-right", asm.factChunk?.id === id && "font-semibold text-ok")}>{asm.factChunk?.id === id ? "★ answer" : `rank ${[...asm.picked].sort((x, y) => CHUNKS.find((c) => c.id === y.id)!.rel - CHUNKS.find((c) => c.id === x.id)!.rel).findIndex((p) => p.id === id) + 1}`}</span>
            </motion.div>
          );
        })}
        <p className="pt-1 text-[11px] text-muted">Grey bars: an illustrative U-shaped position sensitivity. Models tend to use the start and end of a long context best; this curve is a teaching sketch, not a measurement.</p>
      </div>
    );
  }
  const fact = asm.factChunk;
  return (
    <div className="space-y-2">
      <Callout tone={fact ? (asm.middle ? "warn" : "ok") : "err"} title={fact ? `Answer fact present in ${fact.id}${fact.truncated ? " (truncated, but the fact made it)" : ""}` : "Answer fact lost"}>
        {fact ? `Position ${asm.factPos + 1} of ${asm.order.length}${asm.middle ? " (middle of the context)" : ""}.` : `“${FACT}” only appears in C5, and C5 ${asm.dropped.find((d) => d.id === "C5")?.reason ?? "was not selected"}.`}
      </Callout>
      <div className="max-h-48 space-y-1 overflow-y-auto rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">
        {asm.order.map((id, i) => {
          const p = asm.picked.find((x) => x.id === id)!;
          const parts = p.text.split(FACT);
          return (
            <motion.p key={id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + i * 0.15) * k }}>
              [{i + 1}] {parts.length > 1 ? <>{parts[0]}<mark className="rounded bg-ok-soft px-0.5 text-ok">{FACT}</mark>{parts[1]}</> : p.text}{p.truncated && <span className="text-err"> ✂ …</span>}
            </motion.p>
          );
        })}
      </div>
      <p className="text-xs text-muted">Wasted on duplicates: {asm.picked.filter((p, i) => asm.picked.slice(0, i).some((q) => SIM[Number(q.id.slice(1)) - 1][Number(p.id.slice(1)) - 1] >= 0.6)).reduce((a, p) => a + p.tokens, 0)} tokens. Unused budget: {asm.input.budget - asm.used} tokens.</p>
    </div>
  );
}

function SimMatrix({ k }: { k: number }) {
  return (
    <div className="grid gap-3 sm:grid-cols-[auto_minmax(0,1fr)] sm:items-center">
      <div className="overflow-x-auto">
        <table className="text-[10px]">
          <thead><tr><th />{CHUNKS.map((c) => <th key={c.id} className="px-0.5 font-mono font-normal text-muted">{c.id}</th>)}</tr></thead>
          <tbody>
            {SIM.map((row, i) => (
              <tr key={i}>
                <td className="pr-1 font-mono text-muted">C{i + 1}</td>
                {row.map((v, j) => (
                  <td key={j} className="p-px">
                    <motion.div className={cx("flex h-6 w-7 items-center justify-center rounded-sm font-mono", i !== j && v >= 0.6 && "ring-1 ring-err")} style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(v * 70)}%, var(--surface-2))`, color: v > 0.55 ? "white" : "var(--fg)" }}
                      initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (i + j) * 0.05 * k, duration: 0.25 * k }}>
                      {i === j ? "" : v.toFixed(1)}
                    </motion.div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs leading-relaxed">Pairwise Jaccard similarity of the chunks' word sets. <strong>C1/C2</strong> (two copies of the refund intro) and <strong>C3/C4</strong> (page and FAQ copy) are near-duplicates (red outline). Greedy selection will pay for both copies. ★ C5 holds the answer, but its fact is in the last sentence.</p>
    </div>
  );
}

function FitCheck({ d, asm, k }: { d: Decision; asm: Assembly; k: number }) {
  const c = CHUNKS.find((x) => x.id === d.id)!;
  const B = asm.input.budget;
  const scale = Math.max(B, d.used + c.tokens);
  return (
    <div className="space-y-2 text-xs">
      {asm.input.strategy === "dedupe" && (
        <div>
          <div className="mb-0.5 flex justify-between"><span>max similarity to kept chunks{d.maxSimWith ? ` (${d.maxSimWith})` : ""}</span><span className="font-mono">{(d.maxSim ?? 0).toFixed(2)} vs threshold {asm.input.dupThreshold}</span></div>
          <div className="relative h-3 rounded bg-surface-2">
            <motion.div className={cx("h-full rounded", (d.maxSim ?? 0) >= asm.input.dupThreshold ? "bg-err/70" : "bg-ok/60")} initial={{ width: 0 }} animate={{ width: `${(d.maxSim ?? 0) * 100}%` }} transition={{ delay: 0.2 * k, duration: 0.5 * k }} />
            <div className="absolute inset-y-[-3px] w-0.5 bg-fg" style={{ left: `${asm.input.dupThreshold * 100}%` }} />
          </div>
        </div>
      )}
      {d.action !== "drop-dup" && (
        <div>
          <div className="mb-0.5 flex justify-between"><span>already used + this chunk vs budget</span><span className="font-mono">{d.used} + {c.tokens} = {d.used + c.tokens} {d.used + c.tokens <= B ? "≤" : ">"} {B}</span></div>
          <div className="relative flex h-4 rounded bg-surface-2">
            <div className="h-full rounded-l bg-muted/50" style={{ width: `${(d.used / scale) * 100}%` }} />
            <motion.div className={cx("h-full", d.used + c.tokens <= B ? "bg-ok/70" : "bg-err/70")} initial={{ width: 0 }} animate={{ width: `${(c.tokens / scale) * 100}%` }} transition={{ delay: T.seg(false) * k, duration: T.segDur * k, ease: EASE }} />
            <div className="absolute inset-y-[-3px] w-0.5 bg-fg" style={{ left: `${(B / scale) * 100}%` }} title="budget" />
          </div>
        </div>
      )}
      <p className="text-muted">{d.action === "add" ? "It fits, so it goes in whole." : d.action === "truncate" ? "Truncate mode keeps the first part and stops. Text near the end of the chunk is gone." : d.action === "drop-dup" ? "Skipping it saves its tokens for something new." : "Drop mode discards it and keeps looking for smaller chunks that fit."}</p>
    </div>
  );
}

function MmrTable({ d, lambda, k }: { d: Decision; lambda: number; k: number }) {
  const rows = d.mmr!;
  return (
    <div className="space-y-1 text-xs">
      <div className="grid grid-cols-[34px_minmax(0,1fr)_minmax(0,1fr)_54px] gap-2 text-[10px] text-muted"><span /><span>λ · relevance</span><span>(1 − λ) · max sim to picked</span><span className="text-right">MMR</span></div>
      {rows.map((r, i) => {
        const win = r.id === d.id;
        return (
          <motion.div key={r.id} className={cx("grid grid-cols-[34px_minmax(0,1fr)_minmax(0,1fr)_54px] items-center gap-2 rounded px-0.5", win && "bg-accent-soft")}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.mmrRow(i) * k }}>
            <span className="font-mono font-semibold">{r.id}</span>
            <div className="h-2.5 rounded bg-surface-2"><motion.div className="h-full rounded bg-ok/70" initial={{ width: 0 }} animate={{ width: `${lambda * r.rel * 100}%` }} transition={{ delay: T.mmrRow(i) * k, duration: 0.4 * k }} /></div>
            <div className="h-2.5 rounded bg-surface-2"><motion.div className="h-full rounded bg-err/70" initial={{ width: 0 }} animate={{ width: `${(1 - lambda) * r.maxSim * 100}%` }} transition={{ delay: T.mmrRow(i) * k, duration: 0.4 * k }} /></div>
            <span className="text-right font-mono">{r.score.toFixed(3)}{win && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.seg(true) - 0.3) * k }}> ◀</motion.span>}</span>
          </motion.div>
        );
      })}
      <p className="pt-1 text-muted">Highest MMR wins this round{d.action !== "add" ? `, but it ${d.action === "truncate" ? "has to be truncated" : "does not fit and is dropped"}` : " and goes into the budget"}. Already-picked chunks raise the red penalty of anything similar.</p>
    </div>
  );
}

function Calc({ s, asm }: { s: AsmState; asm: Assembly }) {
  const B = asm.input.budget;
  const d = s.decision;
  if (s.phase === "consider" && d) {
    const c = CHUNKS.find((x) => x.id === d.id)!;
    const fit = `${d.used} + ${c.tokens} = ${d.used + c.tokens} ${d.used + c.tokens <= B ? "\\le" : ">"} ${B}`;
    if (d.mmr) {
      const r = d.mmr.find((m) => m.id === d.id)!;
      return <TexBlock>{`\\begin{aligned} \\text{MMR}(c) &= \\lambda\\,\\text{rel}(c) - (1-\\lambda)\\max_{s \\in S} \\text{sim}(c, s) \\\\ \\text{MMR}(${d.id}) &= ${asm.input.lambda}\\cdot${c.rel} - ${(1 - asm.input.lambda).toFixed(2)}\\cdot${r.maxSim.toFixed(2)} = ${r.score.toFixed(3)} \\\\ \\text{fit: } & ${fit} \\end{aligned}`}</TexBlock>;
    }
    return <TexBlock>{`\\begin{aligned} \\text{sim}(a,b) &= \\frac{|A \\cap B|}{|A \\cup B|} \\\\ \\max_{s \\in S}\\text{sim}(${d.id}, s) &= ${(d.maxSim ?? 0).toFixed(2)}${d.maxSimWith ? `\\;(${d.maxSimWith})` : ""} \\\\ \\text{fit: } & ${fit} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "order" || s.phase === "result") {
    return <TexBlock>{`\\begin{aligned} \\textstyle\\sum_i t_i &= ${s.picked.map((p) => p.tokens).join(" + ") || "0"} = ${s.used} \\le ${B} \\\\ \\text{order} &= (${asm.order.join(",\\,")}) \\end{aligned}`}</TexBlock>;
  }
  return <TexBlock>{`\\textstyle\\sum_{i=1}^{8} t_i = ${CHUNKS.map((c) => c.tokens).join(" + ")} = ${CHUNKS.reduce((a, c) => a + c.tokens, 0)} ${CHUNKS.reduce((a, c) => a + c.tokens, 0) > B ? ">" : "\\le"} B = ${B}`}</TexBlock>;
}
