import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { CheckCircle2, CircleDashed, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "./StepCaption";
import { PRESETS, gain, metricsAt, simulate, totalRelevant, type Candidate, type Grade, type KMetrics, type Threshold, type TopKState } from "./simulation";

/** One schedule (seconds at 1×) for the slide, the counters, the bars and the chart segment. */
const T = { slide: 0.15, slideDur: 0.8, count: 0.9, countDur: 0.7, listStagger: 0.12 };
const DUR = { list: 2700, grow: 2500, pick: 3200 };
const ROW = 46;

const GRADE: Record<Grade, { label: string; tone: "ok" | "warn" | "neutral"; Icon: typeof CheckCircle2 }> = {
  2: { label: "answer", tone: "ok", Icon: CheckCircle2 },
  1: { label: "partial", tone: "warn", Icon: CircleDashed },
  0: { label: "not relevant", tone: "neutral", Icon: XCircle },
};

const METRICS = [
  { key: "precision", label: "precision@k", color: "var(--accent)" },
  { key: "recall", label: "recall@k", color: "var(--ok)" },
  { key: "rr", label: "RR@k (MRR, 1 query)", color: "var(--violet)" },
  { key: "ndcg", label: "nDCG@k", color: "var(--warn)" },
] as const;

const DEFAULTS = { presetId: "rank7", threshold: 2 as Threshold, chosenK: 4 };

export default function TopKVisualizer({ tab }: { tab: VisualizerTab }) {
  const [presetId, setPresetId] = useState(DEFAULTS.presetId);
  const [threshold, setThreshold] = useState<Threshold>(DEFAULTS.threshold);
  const [chosenK, setChosenK] = useState(DEFAULTS.chosenK);
  const [replay, setReplay] = useState(0);
  const preset = PRESETS.find((p) => p.id === presetId)!;
  const steps = useMemo(() => simulate({ presetId, threshold, chosenK }), [presetId, threshold, chosenK]);
  const pb = usePlayback(steps.length, (i) => (i === 0 ? DUR.list : i === steps.length - 1 ? DUR.pick : DUR.grow));
  const step = steps[pb.index];
  const s = step.state;
  const prev = pb.index > 0 ? steps[pb.index - 1].state.m : s.m;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const curve = useMemo(() => preset.candidates.map((_, i) => metricsAt(preset, i + 1, threshold)), [preset, threshold]);
  const R = totalRelevant(preset, threshold);

  if (tab === "metrics") return null;

  const headline = s.phase === "list"
    ? `The retriever hands back ${preset.candidates.length} ranked candidates`
    : s.phase === "grow"
      ? `k = ${s.k}: rank ${s.k} slides into the retrieved set`
      : `You chose k = ${s.k}: recall ${s.m.recall.toFixed(2)} for ${s.m.tokens} context tokens`;
  const sub = s.phase === "list"
    ? "Each row has a retriever score and a human relevance label. k decides how many rows the LLM gets to read."
    : s.phase === "grow"
      ? s.added!.grade >= threshold ? "A relevant chunk: recall and nDCG climb." : "Not relevant (under this threshold): precision drops, but you still pay for its tokens."
      : "Bigger k buys recall with tokens and noise. Move the “Your k” slider to compare.";

  return (
    <Workspace
      main={
        <>
          <Panel title="Ranked candidates → retrieved set" actions={<SimulatedLabel>Hand-set scores</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={headline} sub={sub} onReplay={() => setReplay((r) => r + 1)} />
            <p className="mb-2 text-xs text-muted">Query: <span className="font-medium text-fg">“{preset.query}”</span></p>
            <Lanes candidates={preset.candidates} state={s} threshold={threshold} k={k} animKey={animKey} isList={s.phase === "list"} />
            <TokenBar used={s.m.tokens} prev={prev.tokens} total={curve.at(-1)!.tokens} k={k} animKey={animKey} />
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {METRICS.map((m) => (
                  <MetricTile key={m.key} label={m.label} color={m.color} value={s.m[m.key]} from={s.phase === "grow" ? prev[m.key] : 0} k={k} />
                ))}
              </div>
              <KChart curve={curve} upTo={s.phase === "pick" ? curve.length : s.k} marker={s.phase === "pick" ? s.k : undefined} k={k} animKey={animKey} />
              <Legend />
              <p className="mt-1 text-xs text-muted">
                {s.phase === "list" ? "Press → to grow k one rank at a time. The chart traces each metric as k grows; grey bars are context tokens." :
                  s.phase === "grow" ? `${s.m.relInK} of ${R} relevant chunk(s) are inside the top ${s.k}.${preset.missed.length ? " One relevant chunk is not in the list at all." : ""}` :
                    preset.note}
              </p>
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Query & parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setPresetId(DEFAULTS.presetId); setThreshold(DEFAULTS.threshold); setChosenK(DEFAULTS.chosenK); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Query preset" value={presetId} onChange={(v) => { setPresetId(v); pb.restart(); }} options={PRESETS.map((p) => ({ value: p.id, label: p.name }))} />
              <Select label="Counts as relevant" value={String(threshold) as "1" | "2"} onChange={(v) => setThreshold(Number(v) as Threshold)}
                options={[{ value: "2", label: "Only chunks with the answer" }, { value: "1", label: "Answer or partial (lenient)" }]} />
              <Slider label="Your k (last step)" value={chosenK} min={1} max={preset.candidates.length} onChange={setChosenK} />
            </div>
            <p className="mt-2 text-xs text-muted">{preset.note}</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Anything below the k cutoff never reaches the LLM, no matter how good the prompt or model is." />
          <Panel title="Calculation"><Calc state={s} candidates={preset.candidates} pool={[...preset.candidates.map((c) => c.grade), ...preset.missed.map((m) => m.grade)]} R={R} /></Panel>
          <Panel title="Key values">
            <KV rows={[["k", s.k], ["relevant chunks (R)", R], ["relevant in top-k", s.m.relInK], ["context tokens", s.m.tokens], ["threshold", threshold === 2 ? "grade 2 only" : "grade ≥ 1"]]} />
            {preset.missed.length > 0 && <p className="mt-2 text-xs text-warn">Missed by the retriever: {preset.missed[0].title}</p>}
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Lanes({ candidates, state, threshold, k, animKey, isList }: { candidates: Candidate[]; state: TopKState; threshold: Threshold; k: number; animKey: string; isList: boolean }) {
  const n = candidates.length;
  return (
    <div>
      <div>
        <div className="mb-1 grid grid-cols-2 gap-[4%] text-[10px] font-semibold uppercase tracking-wide text-muted sm:text-[11px]">
          <span>Candidates (ranked by score)</span>
          <span>Retrieved: top-k → LLM context</span>
        </div>
        <div className="relative" style={{ height: n * ROW }}>
          {/* Retrieved zone grows to cover k rows; its bottom edge is the cutoff. */}
          <motion.div className="absolute right-0 top-0 w-[48%] rounded-lg border-2 border-accent/50 bg-accent-soft/60" initial={false}
            animate={{ height: Math.max(state.k * ROW, 4) }} transition={{ duration: 0.6 * k, delay: (state.phase === "grow" ? T.slide : 0) * k, ease: EASE }} />
          <motion.div className="absolute right-0 w-[48%] border-t-2 border-dashed border-accent" initial={false}
            animate={{ y: state.k * ROW }} transition={{ duration: 0.6 * k, delay: (state.phase === "grow" ? T.slide : 0) * k, ease: EASE }}>
            <span className="absolute top-1 right-1 rounded bg-accent px-1.5 font-mono text-[10px] text-white dark:text-black">k = {state.k}</span>
          </motion.div>
          {candidates.map((c, i) => (
            <div key={`slot-${c.id}`} className="absolute left-0 w-[48%] rounded-lg border border-dashed border-line" style={{ top: i * ROW + 3, height: ROW - 6 }} />
          ))}
          {candidates.map((c, i) => {
            const inK = i < state.k;
            const isNew = state.phase === "grow" && state.added?.id === c.id;
            const rel = c.grade >= threshold;
            const g = GRADE[c.grade];
            return (
              <motion.div key={isNew || isList ? `${c.id}-${animKey}` : c.id}
                className={cx("absolute left-0 flex w-[48%] items-center gap-1.5 rounded-lg border bg-surface px-1.5 text-xs shadow-sm sm:gap-2 sm:px-2", inK ? (rel ? "border-ok" : "border-line") : "border-line", isNew && "ring-2 ring-accent")}
                style={{ top: i * ROW + 3, height: ROW - 6 }}
                initial={isNew ? { x: "0%" } : isList ? { opacity: 0, y: -12 } : false}
                animate={{ x: inK ? "108.333%" : "0%", opacity: 1, y: 0 }}
                transition={{ x: { duration: T.slideDur * k, delay: isNew ? T.slide * k : 0, ease: EASE }, opacity: { delay: isList ? i * T.listStagger * k : 0, duration: 0.3 * k }, y: { delay: isList ? i * T.listStagger * k : 0, duration: 0.3 * k } }}>
                <span className="w-6 shrink-0 text-center font-mono text-[10px] text-muted sm:text-xs">#{i + 1}</span>
                <span className="min-w-0 flex-1 truncate font-medium" title={c.text}>{c.title}</span>
                <span className="hidden shrink-0 font-mono text-[10px] text-muted sm:inline">{c.score.toFixed(2)}</span>
                <Badge tone={c.grade >= threshold ? g.tone : "neutral"}><g.Icon size={11} aria-hidden /><span className="sr-only sm:not-sr-only">{g.label}</span></Badge>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function TokenBar({ used, prev, total, k, animKey }: { used: number; prev: number; total: number; k: number; animKey: string }) {
  return (
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-xs"><span className="font-medium">Context tokens sent to the LLM</span>
        <span className="font-mono"><AnimatedNumber key={animKey} value={used} from={prev} digits={0} delay={T.count * k} duration={T.countDur * k} /> / {total} (all 10)</span></div>
      <div className="h-3 overflow-hidden rounded bg-surface-2">
        <motion.div className="h-full rounded bg-accent/70" initial={{ width: `${(prev / total) * 100}%` }} animate={{ width: `${(used / total) * 100}%` }} key={animKey}
          transition={{ delay: T.count * k, duration: T.countDur * k, ease: EASE }} />
      </div>
    </div>
  );
}

function MetricTile({ label, color, value, from, k }: { label: string; color: string; value: number; from: number; k: number }) {
  const delta = value - from;
  return (
    <div className="rounded-md border border-line p-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="flex items-baseline gap-1.5 font-mono text-lg">
        <AnimatedNumber value={value} from={from} digits={2} delay={T.count * k} duration={T.countDur * k} />
        {Math.abs(delta) > 1e-9 && <motion.span className={cx("text-[11px]", delta > 0 ? "text-ok" : "text-err")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.count + T.countDur) * k }}>{delta > 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(2)}</motion.span>}
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded bg-surface-2">
        <motion.div className="h-full rounded" style={{ background: color }} initial={{ width: `${from * 100}%` }} animate={{ width: `${value * 100}%` }} transition={{ delay: T.count * k, duration: T.countDur * k, ease: EASE }} />
      </div>
    </div>
  );
}

function KChart({ curve, upTo, marker, k, animKey }: { curve: KMetrics[]; upTo: number; marker?: number; k: number; animKey: string }) {
  const W = 440, H = 160, L = 30, Rr = 36, Tp = 10, B = 26, n = curve.length;
  const sx = (x: number) => L + ((x - 1) / (n - 1)) * (W - L - Rr);
  const sy = (y: number) => Tp + (1 - y) * (H - Tp - B);
  const maxTok = curve.at(-1)!.tokens;
  const bw = (W - L - Rr) / (n - 1) * 0.5;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto mt-3 h-auto w-full max-w-[640px]" role="img" aria-label="Precision, recall, reciprocal rank and nDCG plotted against k, with context tokens as bars">
      {[0, 0.5, 1].map((t) => (
        <g key={t}><line x1={L} x2={W - Rr} y1={sy(t)} y2={sy(t)} stroke="var(--line)" /><text x={L - 5} y={sy(t) + 3} fontSize={10} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{t.toFixed(1)}</text></g>
      ))}
      <text x={W - Rr + 4} y={Tp + 8} fontSize={9} fill="var(--muted)">tokens</text>
      <text x={W - Rr + 4} y={Tp + 20} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{maxTok}</text>
      {curve.map((c) => (
        <g key={c.k}>
          <text x={sx(c.k)} y={H - 10} fontSize={10} textAnchor="middle" fill={c.k === upTo && marker === undefined ? "var(--accent)" : "var(--muted)"} fontFamily="var(--font-mono)">{c.k}</text>
          {c.k <= upTo && (
            <motion.rect key={c.k === upTo ? `${animKey}-bar` : c.k} x={sx(c.k) - bw / 2} width={bw} fill="var(--muted)" fillOpacity={0.18}
              initial={c.k === upTo ? { y: sy(0), height: 0 } : false} animate={{ y: sy(c.tokens / maxTok), height: sy(0) - sy(c.tokens / maxTok) }}
              transition={{ delay: T.count * k, duration: T.countDur * k, ease: EASE }} />
          )}
        </g>
      ))}
      <text x={(L + W - Rr) / 2} y={H} fontSize={10} textAnchor="middle" fill="var(--muted)">k</text>
      {marker !== undefined && <line x1={sx(marker)} x2={sx(marker)} y1={Tp} y2={sy(0)} stroke="var(--accent)" strokeDasharray="4 3" strokeWidth={1.5} />}
      {METRICS.map((m) => {
        const pts = curve.slice(0, upTo).map((c) => [sx(c.k), sy(c[m.key])] as const);
        const last = pts.at(-1), before = pts.at(-2);
        const drawLast = marker === undefined && last && before;
        return (
          <g key={m.key}>
            <polyline fill="none" stroke={m.color} strokeWidth={2} points={(drawLast ? pts.slice(0, -1) : pts).map((p) => p.join(",")).join(" ")} />
            {drawLast && (
              <motion.line key={animKey} x1={before[0]} y1={before[1]} stroke={m.color} strokeWidth={2} initial={{ x2: before[0], y2: before[1] }} animate={{ x2: last[0], y2: last[1] }}
                transition={{ delay: T.count * k, duration: T.countDur * k, ease: EASE }} />
            )}
            {pts.map((p, i) => (
              <motion.circle key={i === pts.length - 1 ? `${animKey}-${i}` : i} cx={p[0]} cy={p[1]} r={3} fill={m.color}
                initial={i === pts.length - 1 && marker === undefined ? { scale: 0 } : false} animate={{ scale: 1 }}
                transition={{ delay: (T.count + T.countDur) * k, type: "spring", stiffness: 300, damping: 15 }} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function Legend() {
  return (
    <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
      {METRICS.map((m) => (
        <span key={m.key} className="flex items-center gap-1"><svg width="18" height="6" aria-hidden><line x1="0" y1="3" x2="18" y2="3" stroke={m.color} strokeWidth="2" /></svg>{m.label}</span>
      ))}
      <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-muted/30" />context tokens</span>
    </div>
  );
}

function Calc({ state, candidates, pool, R }: { state: TopKState; candidates: Candidate[]; pool: Grade[]; R: number }) {
  const { k: kk, m } = state;
  if (kk === 0) return <TexBlock>{"P@k = \\frac{\\#\\text{relevant in top-}k}{k} \\quad R@k = \\frac{\\#\\text{relevant in top-}k}{R} \\\\[4pt] \\text{nDCG@}k = \\frac{\\sum_{i=1}^{k} \\frac{2^{g_i}-1}{\\log_2(i+1)}}{\\text{IDCG@}k}"}</TexBlock>;
  const terms = candidates.slice(0, kk).map((c, i) => ({ g: c.grade, i: i + 1 })).filter((t) => t.g > 0);
  const dcgTex = terms.length ? terms.map((t) => `\\frac{${gain(t.g)}}{\\log_2 ${t.i + 1}}`).join(" + ") : "0";
  const ideal = [...pool].sort((a, b) => b - a).slice(0, kk).map((g, i) => ({ g, i: i + 1 })).filter((t) => t.g > 0);
  const idcgTex = ideal.length ? ideal.map((t) => `\\frac{${gain(t.g)}}{\\log_2 ${t.i + 1}}`).join(" + ") : "0";
  return (
    <TexBlock>{`\\begin{aligned}
P@${kk} &= \\tfrac{${m.relInK}}{${kk}} = ${m.precision.toFixed(3)} \\\\
R@${kk} &= \\tfrac{${m.relInK}}{${R}} = ${m.recall.toFixed(3)} \\\\
\\text{RR}@${kk} &= ${m.rr > 0 ? `\\tfrac{1}{${Math.round(1 / m.rr)}} = ${m.rr.toFixed(3)}` : "0 \\;(\\text{no relevant hit})"} \\\\
\\text{DCG}@${kk} &= ${dcgTex} = ${m.dcg.toFixed(3)} \\\\
\\text{IDCG}@${kk} &= ${idcgTex} = ${m.idcg.toFixed(3)} \\\\
\\text{nDCG}@${kk} &= \\tfrac{${m.dcg.toFixed(3)}}{${m.idcg.toFixed(3)}} = ${m.ndcg.toFixed(3)}
\\end{aligned}`}</TexBlock>
  );
}
