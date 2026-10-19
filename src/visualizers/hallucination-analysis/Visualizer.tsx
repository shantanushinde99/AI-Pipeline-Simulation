import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, CircleHelp, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, bezier, ease, pct } from "./kit";
import { ANSWERS, DEFAULT_CONFIG, SOURCES, simulate, summarize, checkClaim, claimsOf, type Check, type CheckConfig, type HState, type SourceId, type Verdict } from "./simulation";

/** One timing schedule (seconds at 1×) shared by the canvas, the step card and the counters. */
const T = { partGap: 0.12, peel: (i: number) => 0.3 + i * 0.45, line: 0, lineDur: 0.7, dot: 0.1, dotDur: 0.8, hit: 0.9, stamp: 1.2, bar: 0.2, barDur: 1.2, compare: 1.6 };
const duration = (s: HState, n: number) => (s.phase === "answer" ? 2600 : s.phase === "decompose" ? (T.peel(n) + 1.2) * 1000 : s.phase === "check" ? 3200 : 3800);

const V: Record<Verdict, { tone: "ok" | "err" | "warn"; color: string; soft: string; Icon: typeof CheckCircle2; mark: string }> = {
  supported: { tone: "ok", color: "var(--ok)", soft: "var(--ok-soft)", Icon: CheckCircle2, mark: "✓" },
  contradicted: { tone: "err", color: "var(--err)", soft: "var(--err-soft)", Icon: XCircle, mark: "✗" },
  unverifiable: { tone: "warn", color: "var(--warn)", soft: "var(--warn-soft)", Icon: CircleHelp, mark: "?" },
};

function narrate(s: HState, checks: Check[], rate: number) {
  if (s.phase === "answer") return { headline: "A fluent answer arrives, citations and all", sub: "It reads well. The question is which parts of it are actually backed by the sources." };
  if (s.phase === "decompose") return { headline: `The answer peels apart into ${checks.length} atomic claims`, sub: "Each claim states one attribute with one value, so it can be checked on its own." };
  if (s.phase === "rate") return { headline: `Hallucination rate: ${(rate * 100).toFixed(0)}% of claims`, sub: "Contradicted claims (and, if you count them, unverifiable ones) divided by all claims." };
  const ch = checks[s.current!];
  const verb = ch.verdict === "supported" ? "is backed by" : ch.verdict === "contradicted" ? "is contradicted by" : "cannot be confirmed by";
  return { headline: `Claim C${s.current! + 1} ${verb} the sources${ch.type ? `: ${ch.type}` : ""}`, sub: ch.reason };
}

export default function HallucinationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [answerId, setAnswerId] = useState(ANSWERS[1].id);
  const [cfg, setCfg] = useState<CheckConfig>(DEFAULT_CONFIG);
  const [replay, setReplay] = useState(0);
  const answer = ANSWERS.find((a) => a.id === answerId)!;
  const run = useMemo(() => simulate(answer, cfg), [answer, cfg]);
  const n = run.claims.length;
  const pb = usePlayback(run.steps.length, (i) => duration(run.steps[i].state, n));
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${answerId}-${replay}`;
  const story = narrate(s, run.checks, run.summary.rate);
  const sofar = summarize(run.checks.slice(0, s.checked), cfg.countUnverifiable);
  const set = (p: Partial<CheckConfig>) => setCfg((c) => ({ ...c, ...p }));
  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Claim-by-claim check" actions={<SimulatedLabel>Rule-based checker</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <Paragraph key={`p-${animKey}`} parts={answer.parts} phase={s.phase} checks={run.checks} checked={s.checked} k={k} />
            <div className="mt-3 overflow-x-auto">
              <div className="min-w-[680px]">
                <ClaimMap checks={run.checks} state={s} missing={cfg.missing} k={k} animKey={animKey} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted">
              {(Object.keys(V) as Verdict[]).map((v) => { const I = V[v].Icon; return <span key={v} className="flex items-center gap-1"><I size={12} style={{ color: V[v].color }} aria-hidden />{v}</span>; })}
              <span>· line = the evidence the checker matched</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard state={s} run={run} cfg={cfg} k={k} answerId={answerId} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Answer & checker settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_CONFIG); setAnswerId(ANSWERS[1].id); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Answer to check" value={answerId} options={ANSWERS.map((a) => ({ value: a.id, label: a.name }))} onChange={(v) => { setAnswerId(v); pb.restart(); }} />
              <Select<SourceId | "none"> label="Break it: drop a source from retrieval" value={cfg.missing}
                options={[{ value: "none", label: "All sources retrieved" }, ...SOURCES.map((x) => ({ value: x.id, label: `${x.id} missing` }))]} onChange={(v) => set({ missing: v })} />
              <Slider label="Numeric tolerance" value={cfg.tolerance} min={0} max={0.3} step={0.01} format={(v) => pct(v)} onChange={(v) => set({ tolerance: v })}
                hint="How far a measurement may drift before it counts as contradicted. Years always need an exact match." />
              <Toggle label="Count unverifiable claims as hallucinations" checked={cfg.countUnverifiable} onChange={(v) => set({ countUnverifiable: v })}
                hint="Strict graders count anything the sources cannot back. Lenient graders only count contradictions." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.phase === "check" ? "One unsupported number inside an otherwise correct paragraph is exactly what readers miss." : "Rates per claim are comparable across answers of different lengths."} />
          <Panel title="Calculation">
            <Calculation state={s} checks={run.checks} cfg={cfg} />
            <p className="mt-2 text-[11px] text-muted">C = contradicted, U = unverifiable, N = claims checked so far; d = relative difference.</p>
          </Panel>
          <Panel title="Tally so far">
            <KV rows={[
              ["claims checked", `${s.checked} / ${n}`],
              [<span key="s" className="text-ok">✓ supported</span>, sofar.supported],
              [<span key="c" className="text-err">✗ contradicted</span>, sofar.contradicted],
              [<span key="u" className="text-warn">? unverifiable</span>, sofar.unverifiable],
              ["hallucination rate", s.checked ? pct(sofar.rate) : "—"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Paragraph({ parts, phase, checks, checked, k }: { parts: (string | Check["claim"])[]; phase: HState["phase"]; checks: Check[]; checked: number; k: number }) {
  let ci = -1;
  return (
    <p className="rounded-md border border-line bg-surface-2 p-3 text-sm leading-7">
      {parts.map((p, i) => {
        if (typeof p === "string") return <motion.span key={i} initial={phase === "answer" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: i * T.partGap * k }}>{p}</motion.span>;
        ci++;
        const idx = ci;
        const verdict = idx < checked && !(phase === "check" && idx === checked - 1) ? checks[idx].verdict : null;
        return (
          <motion.span key={i} className={cx("rounded px-0.5", phase !== "answer" && "border-b-2", verdict ? "" : phase !== "answer" && "border-accent bg-accent-soft")}
            style={verdict ? { background: V[verdict].soft, borderColor: V[verdict].color } : undefined}
            initial={phase === "answer" ? { opacity: 0 } : false}
            animate={phase === "decompose" ? { opacity: [1, 1, 0.45], scale: [1, 1.04, 1] } : { opacity: 1 }}
            transition={phase === "decompose" ? { delay: T.peel(idx) * k, duration: 0.6 * k } : { delay: i * T.partGap * k }}>
            {phase !== "answer" && <sup className="mr-0.5 font-mono text-[9px] text-accent">C{idx + 1}</sup>}{p.span}
          </motion.span>
        );
      })}
    </p>
  );
}

const CHIP = { x: 8, w: 350, h: 44, gap: 54, top: 10 };
const CARD = { x: 440, w: 312 };

function layoutSources(missing: string) {
  let y = 6;
  const facts: Record<string, number> = {};
  const cards = SOURCES.map((s) => {
    const h = 26 + s.facts.length * 22 + 6;
    const card = { s, y, h, gone: s.id === missing };
    s.facts.forEach((f, i) => (facts[f.id] = y + 26 + i * 22 + 9));
    y += h + 10;
    return card;
  });
  return { cards, facts, noEvidence: y + 14, notFound: y + 52, height: y + 76 };
}

function ClaimMap({ checks, state, missing, k, animKey }: { checks: Check[]; state: HState; missing: string; k: number; animKey: string }) {
  const L = layoutSources(missing);
  const H = Math.max(L.height, CHIP.top + checks.length * CHIP.gap + 6);
  const target = (ch: Check) => (ch.fact ? L.facts[ch.fact.id] : ch.type === "fabricated citation" ? L.notFound : L.noEvidence);
  const chipY = (i: number) => CHIP.top + i * CHIP.gap;
  const showChips = state.phase !== "answer";
  const cur = state.current;
  return (
    <svg viewBox={`0 0 760 ${H}`} className="h-auto w-full" style={{ overflow: "visible" }} role="img" aria-label="Claims on the left, matched against source facts on the right">
      {/* source cards */}
      {L.cards.map(({ s, y, h, gone }) => (
        <g key={s.id} opacity={gone ? 0.45 : 1}>
          <rect x={CARD.x} y={y} width={CARD.w} height={h} rx={8} fill="var(--surface-2)" stroke="var(--line)" strokeDasharray={gone ? "5 4" : undefined} />
          <text x={CARD.x + 10} y={y + 17} fontSize={11} fontWeight={600} fill="var(--fg)">{s.title}{gone ? "  (not retrieved)" : ""}</text>
          {s.facts.map((f, i) => {
            const hit = !gone && cur !== null && checks[cur].fact?.id === f.id;
            return (
              <g key={f.id}>
                {hit && <motion.rect key={animKey} x={CARD.x + 4} y={y + 26 + i * 22 - 2} width={CARD.w - 8} height={20} rx={4} fill="var(--accent-soft)" stroke="var(--accent)"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.hit * k, duration: 0.3 * k }} />}
                <text x={CARD.x + 12} y={y + 26 + i * 22 + 13} fontSize={11} fill={gone ? "var(--muted)" : "var(--fg)"}>{f.text}</text>
              </g>
            );
          })}
        </g>
      ))}
      <rect x={CARD.x} y={L.noEvidence - 14} width={CARD.w} height={28} rx={8} fill="none" stroke="var(--warn)" strokeDasharray="4 4" />
      <text x={CARD.x + 12} y={L.noEvidence + 4} fontSize={11} fill="var(--warn)">? no retrieved evidence for this attribute</text>
      <rect x={CARD.x} y={L.notFound - 14} width={CARD.w} height={28} rx={8} fill="none" stroke="var(--err)" strokeDasharray="4 4" />
      <text x={CARD.x + 12} y={L.notFound + 4} fontSize={11} fill="var(--err)">✗ cited source does not exist</text>

      {/* settled connectors for claims already checked */}
      {showChips && checks.slice(0, state.checked).map((ch, i) => i === cur ? null : (
        <path key={`old-${i}`} d={bezier(CHIP.x + CHIP.w, chipY(i) + CHIP.h / 2, CARD.x, target(ch)).d} fill="none" stroke={V[ch.verdict].color} strokeOpacity={0.3} strokeWidth={1.5} />
      ))}
      {/* the connector being drawn right now, with a dot riding it */}
      {cur !== null && (() => {
        const ch = checks[cur];
        const b = bezier(CHIP.x + CHIP.w, chipY(cur) + CHIP.h / 2, CARD.x, target(ch));
        return (
          <g key={`cur-${animKey}`}>
            <motion.path d={b.d} fill="none" stroke={V[ch.verdict].color} strokeWidth={2.5} strokeDasharray={ch.fact ? undefined : "6 4"}
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: T.line * k, duration: T.lineDur * k, ease }} />
            <motion.circle r={6} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} initial={{ cx: b.xs[0], cy: b.ys[0], opacity: 0 }}
              animate={{ cx: b.xs, cy: b.ys, opacity: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0] }} transition={{ delay: T.dot * k, duration: T.dotDur * k, ease: "easeInOut" }} />
          </g>
        );
      })()}

      {/* claim chips: they drop in from the paragraph during "decompose" */}
      {showChips && checks.map((ch, i) => {
        const y = chipY(i);
        const done = i < state.checked;
        const isCur = i === cur;
        const v = V[ch.verdict];
        return (
          <motion.g key={`chip-${i}-${state.phase === "decompose" ? animKey : "static"}`}
            initial={state.phase === "decompose" ? { opacity: 0, y: -70, scale: 0.9 } : false} animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: state.phase === "decompose" ? T.peel(i) * k : 0, duration: 0.6 * k, ease }}>
            {isCur && <motion.rect x={CHIP.x - 4} y={y - 4} width={CHIP.w + 8} height={CHIP.h + 8} rx={10} fill="none" stroke="var(--accent)" strokeWidth={2}
              initial={{ opacity: 0.9 }} animate={{ opacity: [0.9, 0.2, 0.9] }} transition={{ duration: 1.6, repeat: Infinity }} />}
            <rect x={CHIP.x} y={y} width={CHIP.w} height={CHIP.h} rx={8} fill={done && !isCur ? v.soft : "var(--surface)"} stroke={done ? v.color : "var(--line)"} strokeWidth={isCur ? 2 : 1} />
            <text x={CHIP.x + 10} y={y + 17} fontSize={12} fill="var(--fg)"><tspan fontFamily="var(--font-mono)" fill="var(--accent)" fontWeight={600}>C{i + 1} </tspan>{ch.claim.text}</text>
            <text x={CHIP.x + 10} y={y + 34} fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">{ch.claim.attr} = {String(ch.claim.value)}{ch.claim.cite ? `  cites [${ch.claim.cite}]` : "  (no citation)"}</text>
            {done && (
              <motion.g key={isCur ? animKey : "settled"} initial={isCur ? { scale: 0, opacity: 0 } : false} animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: isCur ? T.stamp * k : 0, type: "spring", stiffness: 260, damping: 16 }}>
                <rect x={CHIP.x + CHIP.w - 118} y={y + 23} width={112} height={16} rx={8} fill={v.color} />
                <text x={CHIP.x + CHIP.w - 62} y={y + 35} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--surface)">{v.mark} {ch.type ?? ch.verdict}</text>
              </motion.g>
            )}
          </motion.g>
        );
      })}
    </svg>
  );
}

function StepCard({ state, run, cfg, k, answerId }: { state: HState; run: ReturnType<typeof simulate>; cfg: CheckConfig; k: number; answerId: string }) {
  const { claims, checks, summary } = run;
  if (state.phase === "answer") {
    const cites = claims.filter((c) => c.cite).length;
    const tiles: [string, number][] = [["factual statements", claims.length], ["citations shown", cites], ["claims checked so far", 0]];
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          {tiles.map(([label, v], i) => (
            <motion.div key={label} className="rounded-md border border-line bg-surface-2 p-2 text-center" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + i * 0.3) * k }}>
              <div className="font-mono text-2xl"><AnimatedNumber value={v} from={0} digits={0} delay={(0.3 + i * 0.3) * k} duration={0.6 * k} /></div>
              <div className="text-[11px] text-muted">{label}</div>
            </motion.div>
          ))}
        </div>
        <Callout tone="warn" title="Citations are not evidence until you check them">A bracketed [S#] only says where the answer claims to come from. The checker will follow each one.</Callout>
      </div>
    );
  }
  if (state.phase === "decompose") {
    return (
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">claim</th><th>attribute</th><th>value</th><th>citation</th></tr></thead>
          <tbody className="font-mono">
            {claims.map((c, i) => (
              <motion.tr key={c.id} className="border-t border-line" initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (T.peel(i) + 0.4) * k, duration: 0.4 * k }}>
                <td className="py-1 text-accent">C{i + 1}</td><td>{c.attr}</td><td>{String(c.value)}</td><td>{c.cite ?? <span className="text-muted">none</span>}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-muted">One attribute, one value per claim: a sentence like &quot;its 3.6 m mirror sits at 2,150 m&quot; holds two claims that can be true or false independently.</p>
      </div>
    );
  }
  if (state.phase === "check") {
    const ch = checks[state.current!];
    const v = V[ch.verdict];
    const numeric = ch.relDiff !== undefined && !ch.fact?.exact;
    const scale = Math.max(cfg.tolerance * 2, (ch.relDiff ?? 0) * 1.15, 0.05);
    const rows: [string, string][] = [["claim says", `${ch.claim.attr} = ${ch.claim.value}`], ["source says", ch.fact ? `${ch.fact.attr} = ${ch.fact.value}${ch.fact.quantifier ? ` (${ch.fact.quantifier})` : ""}  [${ch.fact.source}]` : "nothing matching"]];
    return (
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_170px]">
        <div className="space-y-2">
          {rows.map(([l, val], i) => (
            <motion.div key={l} className="grid grid-cols-[90px_minmax(0,1fr)] gap-2 text-sm" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.35) * k }}>
              <span className="text-xs text-muted">{l}</span><span className="font-mono text-xs">{val}</span>
            </motion.div>
          ))}
          {numeric && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 * k }}>
              <div className="mb-1 flex justify-between text-[11px] text-muted"><span>relative difference</span><span className="font-mono"><AnimatedNumber value={ch.relDiff! * 100} from={0} digits={1} delay={T.hit * k} duration={0.5 * k} />%</span></div>
              <div className="relative h-4 rounded bg-surface-2">
                <motion.div className="absolute inset-y-0.5 left-0 rounded" style={{ background: v.color }} initial={{ width: 0 }} animate={{ width: `${(ch.relDiff! / scale) * 100}%` }} transition={{ delay: T.hit * k, duration: 0.5 * k, ease }} />
                <div className="absolute inset-y-[-3px] w-0.5 bg-fg" style={{ left: `${(cfg.tolerance / scale) * 100}%` }} />
              </div>
              <div className="mt-0.5 text-[10px] text-muted" style={{ paddingLeft: `${Math.min(80, (cfg.tolerance / scale) * 100)}%` }}>↑ tolerance {pct(cfg.tolerance)}</div>
            </motion.div>
          )}
          <motion.p className="text-xs leading-relaxed" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.stamp * k }}>{ch.reason}</motion.p>
        </div>
        <motion.div className="flex flex-col items-center justify-center gap-1 rounded-lg border-2 p-3 text-center" style={{ borderColor: v.color, background: v.soft }}
          initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: T.stamp * k, type: "spring", stiffness: 240, damping: 15 }}>
          <v.Icon size={26} style={{ color: v.color }} aria-hidden />
          <div className="text-sm font-semibold">{ch.verdict}</div>
          {ch.type && <Badge tone={v.tone}>{ch.type}</Badge>}
        </motion.div>
      </div>
    );
  }
  const segs: [Verdict, number][] = [["supported", summary.supported], ["contradicted", summary.contradicted], ["unverifiable", summary.unverifiable]];
  const others = ANSWERS.map((a) => ({ a, r: summarize(claimsOf(a).map((c) => checkClaim(c, cfg)), cfg.countUnverifiable).rate }));
  let acc = 0;
  return (
    <div className="space-y-3">
      <div className="relative h-8 overflow-hidden rounded-md bg-surface-2">
        {segs.map(([v, count]) => {
          const left = acc; acc += count / summary.n;
          return count ? (
            <motion.div key={v} className="absolute inset-y-0 flex items-center justify-center text-[11px] font-semibold text-white dark:text-black" style={{ left: `${left * 100}%`, background: V[v].color }}
              initial={{ width: 0 }} animate={{ width: `${(count / summary.n) * 100}%` }} transition={{ delay: (T.bar + left * T.barDur) * k, duration: (count / summary.n) * T.barDur * k, ease: "linear" }}>
              {V[v].mark} {count}
            </motion.div>
          ) : null;
        })}
      </div>
      <div className="flex flex-wrap items-baseline gap-2 text-sm">
        <span>hallucination rate =</span>
        <span className="font-mono text-2xl font-semibold"><AnimatedNumber value={summary.rate * 100} from={0} digits={0} delay={T.bar * k} duration={T.barDur * k} />%</span>
        {Object.entries(summary.byType).map(([t, c]) => <Badge key={t} tone="warn">{t} × {c}</Badge>)}
      </div>
      <div>
        <p className="mb-1 text-xs font-semibold">Same checker, every preset answer</p>
        <div className="space-y-1">
          {others.map(({ a, r }, i) => (
            <div key={a.id} className="grid grid-cols-[130px_minmax(0,1fr)_44px] items-center gap-2 text-xs">
              <span className={cx(a.id === answerId && "font-semibold")}>{a.name}</span>
              <div className="h-3 rounded bg-surface-2"><motion.div className="h-3 rounded bg-err/70" initial={{ width: 0 }} animate={{ width: `${r * 100}%` }} transition={{ delay: (T.compare + i * 0.15) * k, duration: 0.5 * k, ease }} /></div>
              <span className="text-right font-mono">{pct(r)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Calculation({ state, checks, cfg }: { state: HState; checks: Check[]; cfg: CheckConfig }) {
  const s = summarize(checks.slice(0, state.checked), cfg.countUnverifiable);
  const rate = `\\text{rate} = \\frac{C${cfg.countUnverifiable ? " + U" : ""}}{N}`;
  if (state.phase === "check") {
    const ch = checks[state.current!];
    return (
      <div className="space-y-2">
        {ch.relDiff !== undefined && ch.fact ? (
          <TexBlock>{ch.fact.exact
            ? `${ch.claim.value} ${ch.claim.value === ch.fact.value ? "=" : "\\neq"} ${ch.fact.value} \\;\\Rightarrow\\; \\text{${ch.verdict}}`
            : `d = \\frac{|${ch.claim.value} - ${ch.fact.value}|}{${ch.fact.value}} = ${ch.relDiff.toFixed(3)} \\;${ch.relDiff <= cfg.tolerance + 1e-12 ? "\\le" : ">"}\\; ${cfg.tolerance.toFixed(2)}`}</TexBlock>
        ) : <p className="text-xs">{ch.reason}</p>}
        <TexBlock>{`${rate} = \\frac{${s.hallucinated}}{${s.n}} = ${s.rate.toFixed(2)}`}</TexBlock>
      </div>
    );
  }
  return <TexBlock>{state.checked ? `${rate} = \\frac{${s.contradicted}${cfg.countUnverifiable ? ` + ${s.unverifiable}` : ""}}{${s.n}} = ${s.rate.toFixed(2)}` : rate}</TexBlock>;
}

