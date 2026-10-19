import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { AlertTriangle, CheckCircle2, CircleDashed, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../top-k-retrieval/StepCaption";
import { gain } from "../top-k-retrieval/simulation";
import { LATENCY, PRESETS, WEIGHTS, simulate, tradeoff, type Phase, type RerankInput, type Scored } from "./simulation";

/** Shared schedule (seconds at 1×). */
const T = { sweep: 0.9, chip: (j: number) => 0.25 + j * 0.22, bar: 1.3, barDur: 0.6 };
const DUR: Record<Phase, number> = { first: 2400, score: 2800, sort: 2600, compare: 3000 };
const DEFAULTS: RerankInput = { presetId: "refund", n: 8, useNegation: true };

export default function RerankingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<RerankInput>(DEFAULTS);
  const [replay, setReplay] = useState(0);
  const { run, steps } = useMemo(() => simulate(input), [input]);
  const curve = useMemo(() => tradeoff(input.presetId, input.useNegation), [input.presetId, input.useNegation]);
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "score"]);
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<RerankInput>) => setInput((v) => ({ ...v, ...p }));
  const prevElapsed = pb.index > 0 ? steps[pb.index - 1].state.elapsedMs : s.elapsedMs;

  if (tab === "metrics") return null;

  const sorted = s.phase === "sort" || s.phase === "compare";
  const cutIds = run.preset.passages.slice(run.N).map((p) => p.id);
  const order = sorted ? [...run.reranked.map((r) => r.id), ...cutIds] : run.preset.passages.map((p) => p.id);
  const active = s.phase === "score" ? run.scored[s.i] : undefined;
  const story = narrate(s.phase, run, active);

  return (
    <Workspace
      main={
        <>
          <Panel title="Candidates → cross-encoder → new order" actions={<SimulatedLabel>Simulated cross-encoder</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <p className="mb-2 text-xs text-muted">Query: <span className="font-medium text-fg">“{run.preset.query}”</span> · intent: <span className="font-mono">{run.intent}</span></p>
            <div className="space-y-1">
              {order.map((id, pos) => {
                const p = run.preset.passages.find((x) => x.id === id)!;
                const sc = run.scored.find((x) => x.id === id);
                const scored = !!sc && (sorted || (s.phase === "score" && sc.firstRank - 1 <= s.i));
                const isActive = active?.id === id;
                const cut = !sc;
                return (
                  <div key={id}>
                    {pos === run.N && <div className="my-1 flex items-center gap-2 text-[10px] text-muted"><span className="h-px flex-1 border-t-2 border-dashed border-err/60" />cut at N = {run.N}: never reranked<span className="h-px flex-1 border-t-2 border-dashed border-err/60" /></div>}
                    <motion.div layout transition={{ layout: { duration: 0.9 * k, ease: EASE } }}
                      className={cx("relative overflow-hidden rounded-md border bg-surface px-2 py-1.5", isActive ? "border-accent ring-2 ring-accent/40" : "border-line", cut && "opacity-45")}>
                      {isActive && (
                        <motion.div key={animKey} className="pointer-events-none absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-accent/20 to-transparent"
                          initial={{ left: "-33%" }} animate={{ left: "100%" }} transition={{ duration: T.sweep * k, ease: "easeInOut" }} />
                      )}
                      <div className="relative flex items-start gap-2">
                        <span className="w-7 shrink-0 font-mono text-xs font-semibold">#{pos + 1}</span>
                        <div className="min-w-0 flex-1">
                          <p className="line-clamp-2 text-xs leading-snug sm:line-clamp-1">{p.text}</p>
                          <div className="mt-1 flex flex-wrap items-center gap-1">
                            <GradeBadge g={p.grade} />
                            {sorted && sc && sc.firstRank !== pos + 1 && <Badge tone="accent">was #{sc.firstRank}</Badge>}
                            {cut && <Badge>not reranked</Badge>}
                            {scored && sc && <Chips sc={sc} animate={isActive} k={k} animKey={animKey} />}
                          </div>
                        </div>
                        <ScoreBars first={p.firstScore} cross={scored ? sc!.p : undefined} grow={isActive} k={k} animKey={animKey} />
                      </div>
                    </motion.div>
                  </div>
                );
              })}
            </div>
            <LatencyMeter run={run} elapsed={s.elapsedMs} prev={s.phase === "score" ? prevElapsed : s.elapsedMs} k={k} animKey={animKey} upTo={s.phase === "first" ? -1 : s.phase === "score" ? s.i : run.N - 1} />
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <StepCard phase={s.phase} run={run} active={active} curve={curve} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Query & parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(DEFAULTS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Query preset" value={input.presetId} onChange={(v) => { set({ presetId: v }); pb.restart(); }} options={PRESETS.map((p) => ({ value: p.id, label: p.name }))} />
              <Slider label="N candidates to rerank" value={input.n} min={1} max={10} onChange={(v) => set({ n: v })} hint="More candidates: better recall into the reranker, more latency." />
              <div className="flex items-end"><Toggle label="Negation feature" checked={input.useNegation} onChange={(v) => set({ useNegation: v })} hint="Break it: turn off and “not available” looks like a match." /></div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="A cross-encoder reads query and passage together, so it catches negation and intent a vector score misses, but it costs one model call per candidate." />
          <Panel title="Calculation"><Calc phase={s.phase} run={run} active={active} /></Panel>
          <Panel title="Key values">
            <KV rows={[["N reranked", `${run.N} of ${run.preset.passages.length}`], ["reranker latency", `${s.elapsedMs.toFixed(1)} ms`], ["nDCG@3 before", run.before.toFixed(3)], ["nDCG@3 after", run.after.toFixed(3)], ["cost per pair", `${LATENCY.perPairMs} ms + ${LATENCY.perTokenMs} ms/token`]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

type Run = ReturnType<typeof simulate>["run"];

function narrate(phase: Phase, run: Run, active?: Scored) {
  if (phase === "first") {
    const ans = run.preset.passages.find((p) => p.grade === 2)!;
    return { headline: `The first stage returns 10 candidates; the top ${run.N} go to the reranker`, sub: `The answer sits at first-stage rank ${ans.firstRank}. ${ans.firstRank > run.N ? "It is below the cut, so reranking cannot save it." : "It is inside the cut, so the reranker gets a chance to lift it."}` };
  }
  if (phase === "score" && active) {
    return { headline: `Cross-encoder reads (query, ${active.id}) together → ${active.p.toFixed(2)}`, sub: active.f.negation ? `“${active.f.negation}” flips the meaning: the negation feature pulls the score down.` : active.f.intent ? "The passage contains what the question asks for, so the intent feature fires." : "Feature chips pop in as the pair is read; the bar shows the new relevance score." };
  }
  if (phase === "sort") return { headline: "Rows re-sort by the cross-encoder score", sub: `${run.reranked[0].id} moves from rank ${run.reranked[0].firstRank} to rank 1. Cut rows stay at the bottom.` };
  return { headline: `nDCG@3: ${run.before.toFixed(2)} → ${run.after.toFixed(2)} for ${run.latencyMs.toFixed(0)} ms`, sub: "Drag N: more candidates can only help quality if the answer was cut, but every pair adds latency." };
}

function GradeBadge({ g }: { g: 0 | 1 | 2 }) {
  return g === 2 ? <Badge tone="ok"><CheckCircle2 size={11} aria-hidden />answer</Badge> : g === 1 ? <Badge tone="warn"><CircleDashed size={11} aria-hidden />partial</Badge> : <Badge><XCircle size={11} aria-hidden />not relevant</Badge>;
}

function Chips({ sc, animate, k, animKey }: { sc: Scored; animate: boolean; k: number; animKey: string }) {
  const chips: { label: string; tone: "ok" | "neutral" | "err" }[] = [
    { label: `overlap ${sc.f.matched.length}/${sc.f.qTerms.length}`, tone: sc.f.overlap >= 0.99 ? "ok" : "neutral" },
    { label: sc.f.phrase ? `phrase “${sc.f.phrase}”` : "no phrase", tone: sc.f.phrase ? "ok" : "neutral" },
    { label: sc.f.intent ? "intent ✓" : "intent ✗", tone: sc.f.intent ? "ok" : "neutral" },
    ...(sc.f.negation ? [{ label: `negation “${sc.f.negation}”`, tone: "err" as const }] : []),
  ];
  return (
    <>
      {chips.map((c, j) => (
        <motion.span key={`${c.label}-${animate ? animKey : ""}`} initial={animate ? { opacity: 0, scale: 0.5 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: T.chip(j) * k, type: "spring", stiffness: 320, damping: 18 }}>
          <Badge tone={c.tone}>{c.tone === "err" && <AlertTriangle size={10} aria-hidden />}{c.label}</Badge>
        </motion.span>
      ))}
    </>
  );
}

function ScoreBars({ first, cross, grow, k, animKey }: { first: number; cross?: number; grow: boolean; k: number; animKey: string }) {
  return (
    <div className="w-20 shrink-0 space-y-1 sm:w-32">
      <div className="flex items-center gap-1" title="first-stage (vector) score">
        <div className="h-1.5 flex-1 rounded bg-surface-2"><div className="h-full rounded bg-muted/50" style={{ width: `${first * 100}%` }} /></div>
        <span className="w-7 text-right font-mono text-[10px] text-muted">{first.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-1" title="cross-encoder relevance">
        <div className="h-2.5 flex-1 rounded bg-surface-2">
          {cross !== undefined && <motion.div key={grow ? animKey : "s"} className="h-full rounded bg-accent" initial={grow ? { width: 0 } : false} animate={{ width: `${cross * 100}%` }} transition={{ delay: (grow ? T.bar : 0) * k, duration: T.barDur * k, ease: EASE }} />}
        </div>
        <span className="w-7 text-right font-mono text-[10px]">{cross === undefined ? "…" : grow ? <AnimatedNumber key={animKey} value={cross} from={0} delay={T.bar * k} duration={T.barDur * k} /> : cross.toFixed(2)}</span>
      </div>
    </div>
  );
}

function LatencyMeter({ run, elapsed, prev, k, animKey, upTo }: { run: Run; elapsed: number; prev: number; k: number; animKey: string; upTo: number }) {
  const total = LATENCY.fixedMs + run.preset.passages.reduce((s, p) => s + LATENCY.perPairMs + LATENCY.perTokenMs * p.tokens, 0);
  return (
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-xs"><span className="font-medium">Reranker latency (simulated)</span>
        <span className="font-mono"><AnimatedNumber key={animKey} value={elapsed} from={prev} digits={1} delay={T.bar * k} duration={T.barDur * k} /> ms <span className="text-muted">/ {total.toFixed(0)} ms if all 10</span></span></div>
      <div className="flex h-3 overflow-hidden rounded bg-surface-2">
        <div className="h-full bg-muted/50" style={{ width: `${(LATENCY.fixedMs / total) * 100}%` }} title="fixed overhead" />
        {run.scored.map((sc, i) => i <= upTo && (
          <motion.div key={i === upTo ? `${animKey}-${sc.id}` : sc.id} className={cx("h-full border-r border-surface", i % 2 ? "bg-accent/60" : "bg-accent/80")}
            initial={i === upTo && upTo < run.N && prev !== elapsed ? { width: 0 } : false} animate={{ width: `${(sc.latency / total) * 100}%` }} transition={{ delay: T.bar * k, duration: T.barDur * k, ease: EASE }} title={`${sc.id}: ${sc.latency.toFixed(1)} ms`} />
        ))}
      </div>
    </div>
  );
}

function StepCard({ phase, run, active, curve, k }: { phase: Phase; run: Run; active?: Scored; curve: ReturnType<typeof tradeoff>; k: number }) {
  if (phase === "first") {
    return (
      <div>
        <div className="flex gap-1">
          {run.preset.passages.map((p, i) => (
            <motion.div key={p.id} className={cx("flex h-12 flex-1 flex-col items-center justify-end rounded border text-[10px]", i < run.N ? "border-accent bg-accent-soft" : "border-dashed border-line opacity-60")}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: i < run.N ? 1 : 0.6, y: 0 }} transition={{ delay: i * 0.1 * k, duration: 0.3 * k }}>
              <span>{p.grade === 2 ? "★" : p.grade === 1 ? "◐" : "·"}</span>
              <span className="font-mono">#{i + 1}</span>
            </motion.div>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted">Shaded boxes go to the reranker (N = {run.N}). ★ = answer, ◐ = partial. The first stage is cheap and runs over the whole index; the reranker is accurate but expensive, so it only sees a short list.</p>
      </div>
    );
  }
  if (phase === "score" && active) {
    const rows = [
      ["bias", active.parts.bias], [`overlap × ${WEIGHTS.overlap}`, active.parts.overlap], [`phrase × ${WEIGHTS.phrase}`, active.parts.phrase],
      [`intent × ${WEIGHTS.intent}`, active.parts.intent], [`negation × ${WEIGHTS.negation}`, active.parts.negation], ["z (sum)", active.z],
    ] as const;
    const scale = Math.max(...rows.map(([, v]) => Math.abs(v)), 1e-9);
    return (
      <div>
        <div className="space-y-1">
          {rows.map(([label, v], i) => (
            <div key={label} className={cx("grid grid-cols-[110px_minmax(0,1fr)_52px] items-center gap-2 text-xs", i === rows.length - 1 && "border-t border-line pt-1 font-semibold")}>
              <span className="font-mono">{label}</span>
              <div className="relative h-4 rounded bg-surface-2">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <motion.div className={cx("absolute inset-y-0.5 rounded", i === rows.length - 1 ? "bg-accent" : v < 0 ? "bg-err/70" : "bg-ok/70")} style={v < 0 ? { right: "50%" } : { left: "50%" }}
                  initial={{ width: 0 }} animate={{ width: `${(Math.abs(v) / scale) * 50}%` }} transition={{ delay: (i === rows.length - 1 ? T.bar : T.chip(Math.max(0, i - 1))) * k, duration: 0.4 * k, ease: EASE }} />
              </div>
              <span className="text-right font-mono"><AnimatedNumber value={v} from={0} delay={(i === rows.length - 1 ? T.bar : T.chip(Math.max(0, i - 1))) * k} duration={0.4 * k} /></span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-sm">σ(<span className="font-mono">{active.z.toFixed(2)}</span>) = <strong className="font-mono"><AnimatedNumber value={active.p} from={0} delay={T.bar * k} duration={T.barDur * k} digits={3} /></strong> relevance, versus a first-stage score of <span className="font-mono">{active.firstScore.toFixed(2)}</span>. Cost: {active.latency.toFixed(1)} ms ({active.tokens} tokens).</p>
      </div>
    );
  }
  if (phase === "sort") {
    return (
      <div className="flex flex-wrap gap-1.5">
        {run.reranked.map((r, i) => (
          <motion.div key={r.id} className="flex items-center gap-1 rounded border border-line px-1.5 py-0.5 font-mono text-xs" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.9 + i * 0.1) * k }}>
            {r.id} <span className="text-muted">#{r.firstRank}</span> → <span className={cx(r.firstRank > i + 1 ? "text-ok" : r.firstRank < i + 1 ? "text-err" : "")}>#{i + 1}{r.firstRank > i + 1 ? " ▲" : r.firstRank < i + 1 ? " ▼" : ""}</span>
          </motion.div>
        ))}
      </div>
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-[200px_minmax(0,1fr)]">
      <div className="space-y-2">
        {([["before (first stage)", run.before, "bg-muted/60"], ["after reranking", run.after, "bg-accent"]] as const).map(([l, v, c], i) => (
          <div key={l}>
            <div className="flex justify-between text-xs"><span>nDCG@3 {l}</span><span className="font-mono"><AnimatedNumber value={v} from={0} delay={(0.2 + i * 0.4) * k} digits={3} /></span></div>
            <div className="h-3 rounded bg-surface-2"><motion.div className={cx("h-full rounded", c)} initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: (0.2 + i * 0.4) * k, duration: 0.6 * k, ease: EASE }} /></div>
          </div>
        ))}
      </div>
      <TradeoffChart curve={curve} n={run.N} k={k} />
    </div>
  );
}

function TradeoffChart({ curve, n, k }: { curve: ReturnType<typeof tradeoff>; n: number; k: number }) {
  const W = 360, H = 150, L = 28, R = 34, Tp = 8, B = 22;
  const maxLat = Math.max(...curve.map((c) => c.latency));
  const sx = (x: number) => L + ((x - 1) / (curve.length - 1)) * (W - L - R);
  const sy = (y: number) => Tp + (1 - y) * (H - Tp - B);
  const d = curve.map((c, i) => `${i ? "L" : "M"}${sx(c.n).toFixed(1)} ${sy(c.ndcg).toFixed(1)}`).join(" ");
  const bw = ((W - L - R) / (curve.length - 1)) * 0.5;
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="nDCG@3 after reranking and reranker latency for each N">
        {[0, 0.5, 1].map((t) => <g key={t}><line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" /><text x={L - 4} y={sy(t) + 3} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{t}</text></g>)}
        {curve.map((c, i) => (
          <g key={c.n}>
            <motion.rect x={sx(c.n) - bw / 2} width={bw} fill={c.n <= n ? "var(--warn)" : "var(--muted)"} fillOpacity={c.n <= n ? 0.35 : 0.15}
              initial={{ y: sy(0), height: 0 }} animate={{ y: sy(c.latency / maxLat), height: sy(0) - sy(c.latency / maxLat) }} transition={{ delay: (0.2 + i * 0.05) * k, duration: 0.5 * k }} />
            <text x={sx(c.n)} y={H - 8} fontSize={9} textAnchor="middle" fill={c.n === n ? "var(--accent)" : "var(--muted)"} fontFamily="var(--font-mono)">{c.n}</text>
          </g>
        ))}
        <text x={W - R + 3} y={Tp + 8} fontSize={8} fill="var(--muted)">{maxLat.toFixed(0)} ms</text>
        <motion.path d={d} fill="none" stroke="var(--accent)" strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.4 * k, duration: 1 * k, ease: EASE }} />
        <line x1={sx(n)} x2={sx(n)} y1={Tp} y2={sy(0)} stroke="var(--accent)" strokeDasharray="3 3" />
        <motion.circle cx={sx(n)} cy={sy(curve[n - 1].ndcg)} r={4} fill="var(--accent)" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 1.3 * k, type: "spring", stiffness: 300, damping: 15 }} />
        <text x={(L + W - R) / 2} y={H} fontSize={9} textAnchor="middle" fill="var(--muted)">N reranked</text>
      </svg>
      <figcaption className="flex flex-wrap gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1"><svg width="16" height="6" aria-hidden><line x1="0" y1="3" x2="16" y2="3" stroke="var(--accent)" strokeWidth="2" /></svg>nDCG@3 after reranking</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-warn/40" />latency (bars)</span>
      </figcaption>
    </figure>
  );
}

function Calc({ phase, run, active }: { phase: Phase; run: Run; active?: Scored }) {
  if (phase === "score" && active) {
    return (
      <TexBlock>{`\\begin{aligned} z &= b + w_o\\,o + w_p\\,p + w_i\\,i + w_n\\,n \\\\ &= ${WEIGHTS.bias} + ${WEIGHTS.overlap}\\cdot\\tfrac{${active.f.matched.length}}{${active.f.qTerms.length}} + ${WEIGHTS.phrase}\\cdot${active.f.phrase ? 1 : 0} + ${WEIGHTS.intent}\\cdot${active.f.intent ? 1 : 0} + (${WEIGHTS.negation})\\cdot${active.f.negation ? 1 : 0} \\\\ &= ${active.z.toFixed(3)} \\\\ s &= \\sigma(z) = \\frac{1}{1+e^{-z}} = ${active.p.toFixed(3)} \\\\ t &= ${LATENCY.perPairMs} + ${LATENCY.perTokenMs}\\cdot ${active.tokens} = ${active.latency.toFixed(1)}\\,\\text{ms} \\end{aligned}`}</TexBlock>
    );
  }
  if (phase === "compare" || phase === "sort") {
    const g = run.reranked.slice(0, 3).map((r) => r.grade);
    const b = run.preset.passages.slice(0, 3).map((p) => p.grade);
    const term = (gs: number[]) => gs.map((x, i) => `\\tfrac{${gain(x)}}{\\log_2 ${i + 2}}`).join(" + ");
    return (
      <TexBlock>{`\\begin{aligned} \\text{DCG}_{\\text{before}} &= ${term(b)} \\\\ \\text{DCG}_{\\text{after}} &= ${term(g)} \\\\ \\text{nDCG@3} &: ${run.before.toFixed(3)} \\to ${run.after.toFixed(3)} \\\\ \\text{latency} &= ${LATENCY.fixedMs} + \\textstyle\\sum_{j=1}^{${run.N}} t_j = ${run.latencyMs.toFixed(1)}\\,\\text{ms} \\end{aligned}`}</TexBlock>
    );
  }
  return (
    <div className="space-y-2 text-xs">
      <TexBlock>{"s(q, d) = \\sigma\\big(b + w_o\\,\\text{overlap} + w_p\\,\\text{phrase} + w_i\\,\\text{intent} + w_n\\,\\text{negation}\\big)"}</TexBlock>
      <p className="text-muted">A real cross-encoder learns these signals inside a transformer. Here they are explicit, hand-weighted features so you can see each one. <Tex>{`w = (${WEIGHTS.overlap}, ${WEIGHTS.phrase}, ${WEIGHTS.intent}, ${WEIGHTS.negation}),\\ b = ${WEIGHTS.bias}`}</Tex></p>
    </div>
  );
}
