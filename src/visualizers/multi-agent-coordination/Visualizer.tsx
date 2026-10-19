import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, CheckCircle2, ClipboardList, RotateCcw, Timer } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { clamp } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../agent-termination/kit";
import {
  AGENTS, DEFAULT_COORD, FAILURES, NET_MS, PATTERNS, STUCK_MS, sanitize, simulate,
  type AgentId, type BBEntry, type CoordConfig, type CoordState, type Failure, type Kind, type Message, type Outcome, type Pattern,
} from "./simulation";

/** Schedule (seconds at 1×): arrow draws while the envelope travels, then the receiver lights up and the blackboard updates. */
const T = { start: 0.2, travel: 0.9, land: 1.1, bb: 1.2 };
const stepMs = (i: number, total: number) => (i === 0 ? 2200 : i === total - 1 ? 2800 : 2400);

const KIND_COLOR: Record<Kind, string> = { task: "var(--accent)", result: "var(--ok)", handoff: "var(--warn)", critique: "var(--violet)", timeout: "var(--err)", late: "var(--muted)", arbitrate: "var(--violet)", final: "var(--fg)" };
const name = (a: AgentId) => AGENTS.find((x) => x.id === a)!.name;

export default function MultiAgentVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<CoordConfig>(DEFAULT_COORD);
  const [replay, setReplay] = useState(0);
  const c = sanitize(cfg);
  const { msgs, outcome, steps } = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(steps.length, (i) => stepMs(i, steps.length));
  const step = steps[pb.index];
  const s = step.state;
  const prev = pb.index > 0 ? steps[pb.index - 1].state : s;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const current = !s.done && s.shown > 0 ? msgs[s.shown - 1] : undefined;
  const set = (p: Partial<CoordConfig>) => setCfg((x) => ({ ...x, ...p }));

  if (tab === "metrics") return null;
  const story = narrate(step.id, current, s, outcome, c);

  return (
    <Workspace
      main={
        <>
          <Panel title="Agents & messages" actions={<SimulatedLabel>Scripted agents, simulated latency</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Counter label="messages" value={s.totals.messages} from={prev.totals.messages} k={k} animKey={animKey} />
              <Counter label={`hand-offs (limit ${c.maxHandoffs})`} value={s.totals.handoffs} from={prev.totals.handoffs} k={k} animKey={animKey} warn={s.totals.handoffs >= c.maxHandoffs} />
              <Counter label="wall clock (s)" value={s.totals.clock / 1000} from={prev.totals.clock / 1000} digits={2} k={k} animKey={animKey} />
              <Counter label="agent work (s)" value={s.totals.work / 1000} from={prev.totals.work / 1000} digits={1} k={k} animKey={animKey} />
            </div>
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_220px]">
              <div className="overflow-x-auto">
                <div className="min-w-[540px]"><SequenceDiagram msgs={msgs} shown={s.shown} current={current} k={k} animKey={animKey} /></div>
              </div>
              <Blackboard entries={s.bb} prev={prev.bb} k={k} animKey={animKey} />
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              {(["task", "result", "handoff", "critique", "timeout", "late"] as Kind[]).map((kd) => (
                <span key={kd} className="flex items-center gap-1"><svg width="20" height="6" aria-hidden><line x1="0" y1="3" x2="20" y2="3" stroke={KIND_COLOR[kd]} strokeWidth="2" strokeDasharray={kd === "timeout" || kd === "late" ? "4 3" : undefined} /></svg>{kd}</span>
              ))}
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <StepCard current={current} s={s} msgs={msgs} outcome={outcome} cfg={c} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Pattern & failures" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_COORD); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="Coordination pattern" value={cfg.pattern} onChange={(v: Pattern) => set({ pattern: v })} options={(Object.keys(PATTERNS) as Pattern[]).map((p) => ({ value: p, label: PATTERNS[p] }))} />
              <Select label="Inject failure" value={cfg.failure} onChange={(v: Failure) => set({ failure: v })} options={(Object.keys(FAILURES) as Failure[]).map((f) => ({ value: f, label: FAILURES[f] }))} />
              <Slider label="Hop / round limit" value={cfg.maxHandoffs} min={2} max={10} onChange={(v) => set({ maxHandoffs: v })} hint="Max hand-offs (or debate rounds) before someone must intervene." />
              <Slider label="Worker timeout" value={cfg.workerTimeoutMs} min={1000} max={15000} step={500} format={(v) => `${(v / 1000).toFixed(1)} s`} onChange={(v) => set({ workerTimeoutMs: v })} hint={`The hung worker would answer after ${STUCK_MS / 1000} s.`} />
              <Toggle label="Arbitrate conflicting blackboard values" checked={cfg.arbitration} onChange={(v) => set({ arbitration: v })} hint="Off = last write wins." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={current?.kind === "handoff" ? "Hand-offs without ownership can bounce forever. A hop limit and an owner who decides stop the loop." : current?.kind === "timeout" ? "A coordinator that waits forever inherits its slowest worker's failures." : s.bb.some((e) => e.status === "conflict") ? "Agents disagreeing is normal. Silently keeping the last write is the bug." : undefined} />
          <Panel title="Calculation"><Calc current={current} s={s} /></Panel>
          <Panel title="Totals so far">
            <KV rows={[
              ["messages", s.totals.messages],
              ["hand-offs", `${s.totals.handoffs} / ${c.maxHandoffs}`],
              ["wall clock", `${(s.totals.clock / 1000).toFixed(2)} s`],
              ["agent work", `${(s.totals.work / 1000).toFixed(2)} s`],
              ["blackboard conflicts", s.bb.filter((e) => e.status === "conflict").length],
              ...(s.done ? [["outcome", <StatusBadge key="o" status={outcome.status === "success" ? "ok" : outcome.status === "degraded" ? "warning" : "error"} label={outcome.status} />] as [string, ReactNode]] : []),
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(id: string, m: Message | undefined, s: CoordState, outcome: Outcome, c: CoordConfig) {
  if (id === "intro") return { headline: PATTERNS[c.pattern], sub: c.failure === "none" ? "Watch who sends what to whom, and how long it takes." : `Injected failure: ${FAILURES[c.failure].toLowerCase()}.` };
  if (s.done) return { headline: outcome.status === "success" ? "The team delivered" : outcome.status === "degraded" ? "Delivered, but degraded" : "The run failed", sub: outcome.reason };
  if (!m) return { headline: "", sub: "" };
  const head = m.kind === "timeout" ? `${name(m.to)} gives up waiting for ${name(m.from)}`
    : m.kind === "late" ? `${name(m.from)} answers too late`
    : m.kind === "handoff" ? `${name(m.from)} hands the task back to ${name(m.to)} (hand-off ${s.totals.handoffs})`
    : m.kind === "final" ? `${name(m.from)} answers the user`
    : `${name(m.from)} → ${name(m.to)}: ${m.kind}`;
  return { headline: head, sub: `"${m.label}"${m.bb ? ` · blackboard: ${m.bb.key} = ${m.bb.value}` : ""}` };
}

function Counter({ label, value, from, digits = 0, k, animKey, warn }: { label: string; value: number; from: number; digits?: number; k: number; animKey: string; warn?: boolean }) {
  return (
    <div className={cx("rounded-md border px-2 py-1", warn ? "border-warn bg-warn-soft" : "border-line bg-surface-2")}>
      <div className="font-mono text-lg leading-tight"><AnimatedNumber key={animKey} value={value} from={from} digits={digits} delay={T.land * k} duration={0.5 * k} /></div>
      <div className="text-[10px] text-muted">{label}</div>
    </div>
  );
}

const W = 620, X0 = 95, LANE = 115, HEAD_Y = 6, ROW0 = 52;
const laneX = (a: AgentId) => X0 + AGENTS.findIndex((x) => x.id === a) * LANE;

function SequenceDiagram({ msgs, shown, current, k, animKey }: { msgs: Message[]; shown: number; current?: Message; k: number; animKey: string }) {
  const rowH = clamp(440 / Math.max(msgs.length, 1), 22, 34);
  const H = ROW0 + msgs.length * rowH + 10;
  const rowY = (i: number) => ROW0 + i * rowH + rowH / 2 + 4;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Sequence diagram: ${shown} of ${msgs.length} messages`}>
      <defs>
        {(Object.keys(KIND_COLOR) as Kind[]).map((kd) => (
          <marker key={kd} id={`arrow-${kd}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill={KIND_COLOR[kd]} /></marker>
        ))}
      </defs>
      <text x={4} y={HEAD_Y + 19} fontSize={9} fill="var(--muted)">sent at</text>
      {AGENTS.map((a) => {
        const x = laneX(a.id);
        const isFrom = current?.from === a.id, isTo = current?.to === a.id;
        return (
          <g key={a.id}>
            <line x1={x} x2={x} y1={HEAD_Y + 30} y2={H - 4} stroke="var(--line)" strokeDasharray="3 4" />
            {isTo && (
              <motion.rect key={`pulse-${animKey}`} x={x - 50} y={HEAD_Y} width={100} height={28} rx={8} fill="none" stroke={KIND_COLOR[current!.kind]} strokeWidth={2}
                initial={{ opacity: 0, scale: 1 }} animate={{ opacity: [0, 0.9, 0], scale: [1, 1.12, 1.2] }} transition={{ delay: T.land * k, duration: 0.8 * k }} style={{ transformOrigin: `${x}px ${HEAD_Y + 14}px` }} />
            )}
            <motion.rect x={x - 50} y={HEAD_Y} width={100} height={28} rx={8} initial={false}
              animate={{ fill: isFrom || isTo ? "var(--accent-soft)" : "var(--surface-2)", stroke: isFrom || isTo ? "var(--accent)" : "var(--line)" }} transition={{ duration: 0.3 }} />
            <text x={x} y={HEAD_Y + 18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--fg)">{a.name}</text>
          </g>
        );
      })}
      {msgs.slice(0, shown).map((m, i) => {
        const x1 = laneX(m.from), x2 = laneX(m.to), y = rowY(i);
        const isCur = m === current;
        const dir = x2 > x1 ? 1 : -1;
        const maxChars = Math.floor((Math.abs(x2 - x1) - 10) / 5.4);
        const label = m.label.length > maxChars ? `${m.label.slice(0, Math.max(4, maxChars - 1))}…` : m.label;
        const dashed = m.kind === "timeout" || m.kind === "late";
        return (
          <g key={isCur ? `m${i}-${animKey}` : `m${i}`} opacity={m.kind === "late" ? 0.7 : 1}>
            <title>{`${(m.sent / 1000).toFixed(2)} s · ${name(m.from)} → ${name(m.to)} [${m.kind}]: ${m.label}`}</title>
            <text x={4} y={y + 3} fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{(m.sent / 1000).toFixed(2)}s</text>
            <motion.line x1={x1} y1={y} x2={x2 - dir * 4} y2={y} stroke={KIND_COLOR[m.kind]} strokeWidth={isCur ? 2.5 : 1.6} strokeDasharray={dashed ? "5 4" : undefined} markerEnd={`url(#arrow-${m.kind})`}
              initial={isCur ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={{ delay: T.start * k, duration: T.travel * k, ease: EASE }} />
            <motion.text x={(x1 + x2) / 2} y={y - 5} textAnchor="middle" fontSize={10} fill="var(--fg)" initial={isCur ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: (T.start + 0.2) * k }}>
              {m.kind === "timeout" ? "⏱ " : m.kind === "handoff" ? "↩ " : ""}{label}
            </motion.text>
            {isCur && (
              <motion.g initial={{ x: x1, opacity: 0 }} animate={{ x: [x1, x2], opacity: [0, 1, 1, 0] }} transition={{ delay: T.start * k, duration: T.travel * k, ease: EASE, opacity: { delay: T.start * k, duration: (T.travel + 0.15) * k, times: [0, 0.1, 0.85, 1] } }}>
                <rect x={-9} y={y - 7} width={18} height={14} rx={3} fill={KIND_COLOR[m.kind]} stroke="var(--surface)" strokeWidth={1.5} />
                <path d={`M-9 ${y - 7} L0 ${y} L9 ${y - 7}`} fill="none" stroke="var(--surface)" strokeWidth={1.2} />
              </motion.g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

const STATUS_BADGE: Record<BBEntry["status"], { tone: "ok" | "err" | "neutral"; text: string }> = {
  ok: { tone: "ok", text: "✓ ok" }, conflict: { tone: "err", text: "⚠ conflict" }, resolved: { tone: "ok", text: "✓ resolved" }, superseded: { tone: "neutral", text: "superseded" },
};

function Blackboard({ entries, prev, k, animKey }: { entries: BBEntry[]; prev: BBEntry[]; k: number; animKey: string }) {
  return (
    <div className="rounded-md border border-line bg-surface-2 p-2">
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold"><ClipboardList size={13} /> Shared blackboard</div>
      {entries.length === 0 && <p className="text-[11px] text-muted">Empty. Agents post findings and artifacts here.</p>}
      <ul className="space-y-1">
        {entries.map((e, i) => {
          const isNew = i >= prev.length;
          const changed = !isNew && prev[i].status !== e.status;
          return (
            <motion.li key={isNew || changed ? `${i}-${animKey}` : i} initial={isNew ? { opacity: 0, x: 40 } : changed ? { scale: 1.06 } : false} animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ delay: T.bb * k, duration: 0.4 * k, ease: EASE }}
              className={cx("rounded border bg-surface px-1.5 py-1 text-[11px]", e.status === "conflict" ? "border-err" : e.status === "resolved" ? "border-ok" : "border-line", e.status === "superseded" && "opacity-50")}>
              <div className="flex items-center justify-between gap-1">
                <span className={cx("font-mono", e.status === "superseded" && "line-through")}>{e.key} = {e.value}</span>
                <Badge tone={STATUS_BADGE[e.status].tone}>{STATUS_BADGE[e.status].text}</Badge>
              </div>
              <div className="text-[10px] text-muted">by {name(e.by)} · {(e.at / 1000).toFixed(2)} s{e.cited ? " · cited" : " · uncited"}</div>
            </motion.li>
          );
        })}
      </ul>
    </div>
  );
}

function StepCard({ current, s, msgs, outcome, cfg, k }: { current?: Message; s: CoordState; msgs: Message[]; outcome: Outcome; cfg: CoordConfig; k: number }) {
  if (!current && !s.done) {
    const edges = [...new Set(msgs.filter((m) => m.kind !== "late" && m.kind !== "timeout").map((m) => `${name(m.from)} → ${name(m.to)}`))];
    return (
      <div className="space-y-2 text-sm">
        <p>Who talks to whom in this run:</p>
        <div className="flex flex-wrap gap-1.5">
          {edges.map((e, i) => <motion.span key={e} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.12) * k }} className="rounded border border-line bg-surface-2 px-2 py-0.5 text-xs">{e}</motion.span>)}
        </div>
        <p className="text-xs text-muted">Task for every pattern: build a Weather API client that respects the API's rate limit (truth: 100/min, documented in §4).</p>
      </div>
    );
  }
  if (s.done) {
    const parallel = s.totals.work / Math.max(1, s.totals.clock);
    return (
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap items-center gap-2"><StatusBadge status={outcome.status === "success" ? "ok" : outcome.status === "degraded" ? "warning" : "error"} label={outcome.status} /><span>{outcome.reason}</span></div>
        <div className="grid gap-2 sm:grid-cols-2">
          <BarRow label="wall clock" v={s.totals.clock} max={Math.max(s.totals.clock, s.totals.work)} cls="bg-accent/70" k={k} i={0} />
          <BarRow label="sum of agent work" v={s.totals.work} max={Math.max(s.totals.clock, s.totals.work)} cls="bg-violet/60" k={k} i={1} />
        </div>
        <p className="text-xs text-muted">Average agents busy at once: {parallel.toFixed(2)}. {parallel > 1.05 ? "Work overlapped (parallel workers)." : "Mostly sequential: each agent waited for the previous one."} Rate limit shipped: {outcome.limitUsed === null ? "nothing shipped" : `${outcome.limitUsed}/min ${outcome.limitUsed === 100 ? "(correct)" : "(wrong: docs say 100/min)"}`}.</p>
      </div>
    );
  }
  const m = current!;
  if (m.kind === "handoff") {
    return (
      <div className="space-y-2 text-sm">
        <p>Hop counter: each dot is one hand-off. When it reaches {cfg.maxHandoffs}, {cfg.pattern === "pipeline" ? "the run is killed (nobody owns the task)" : "the coordinator steps in"}.</p>
        <div className="flex flex-wrap gap-1.5">
          {Array.from({ length: cfg.maxHandoffs }, (_, i) => (
            <motion.span key={i} initial={false} animate={{ scale: i === s.totals.handoffs - 1 ? [1, 1.4, 1] : 1 }} transition={{ delay: T.land * k, duration: 0.4 * k }}
              className={cx("inline-block h-4 w-4 rounded-full border", i < s.totals.handoffs ? "border-warn bg-warn" : "border-line bg-surface-2")} />
          ))}
        </div>
        <p className="text-xs text-muted">Both agents are "right" locally: each thinks the other should go first. Without an owner (or a rule like "the receiver must act"), the task bounces.</p>
      </div>
    );
  }
  if (m.kind === "timeout" && m.timer) {
    const max = Math.max(STUCK_MS, m.timer.after) * 1.05;
    return (
      <div className="space-y-2 text-sm">
        <div className="grid grid-cols-[110px_minmax(0,1fr)] items-center gap-2 text-xs">
          <span className="text-muted">{name(m.to)} waits</span>
          <div className="relative h-4 rounded bg-surface-2"><motion.div className="absolute inset-y-0 left-0 rounded bg-err/70" initial={{ width: 0 }} animate={{ width: `${(m.timer.after / max) * 100}%` }} transition={{ delay: 0.1 * k, duration: 1 * k, ease: "linear" }} /></div>
          <span className="text-muted">{name(m.from)} needs</span>
          <div className="relative h-4 rounded bg-surface-2"><motion.div className="absolute inset-y-0 left-0 rounded border border-dashed border-muted bg-muted/20" initial={{ width: 0 }} animate={{ width: `${(STUCK_MS / max) * 100}%` }} transition={{ delay: 0.1 * k, duration: (STUCK_MS / m.timer.after) * k, ease: "linear" }} /></div>
        </div>
        <p className="flex items-center gap-1 text-xs"><Timer size={13} /> Timeout {(m.timer.after / 1000).toFixed(1)} s &lt; {STUCK_MS / 1000} s needed, so the {cfg.pattern === "pipeline" ? "next stage has nothing to work on and the pipeline fails" : "coordinator moves on without this worker"}.</p>
      </div>
    );
  }
  const conflicts = s.bb.filter((e) => e.status === "conflict");
  const t0 = m.sent - m.work;
  const scale = Math.max(1, m.arrive);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-2 text-sm">
        <div className="flex flex-wrap items-center gap-2"><Badge>{name(m.from)}</Badge>→<Badge>{name(m.to)}</Badge><Badge tone={m.kind === "late" ? "warn" : "accent"}>{m.kind}</Badge></div>
        <p className="rounded border border-line bg-surface-2 px-2 py-1 font-mono text-xs">{m.label}</p>
        <div className="space-y-1 text-[11px]">
          <div className="relative h-3 rounded bg-surface-2">
            <motion.div className="absolute inset-y-0 rounded bg-violet/50" style={{ left: `${(t0 / scale) * 100}%` }} initial={{ width: 0 }} animate={{ width: `${(m.work / scale) * 100}%` }} transition={{ delay: 0.1 * k, duration: 0.6 * k, ease: EASE }} />
            <motion.div className="absolute inset-y-0 rounded bg-accent" style={{ left: `${(m.sent / scale) * 100}%` }} initial={{ width: 0 }} animate={{ width: `${Math.max(0.8, (NET_MS / scale) * 100)}%` }} transition={{ delay: 0.7 * k, duration: 0.3 * k }} />
          </div>
          <div className="flex justify-between font-mono text-muted"><span>0 s</span><span>{name(m.from)} works {(m.work / 1000).toFixed(1)} s → sends at {(m.sent / 1000).toFixed(2)} s → arrives {(m.arrive / 1000).toFixed(2)} s</span></div>
        </div>
      </div>
      <div>
        <AnimatePresence>
          {conflicts.length > 1 ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.bb * k }} className="space-y-2">
              <div className="flex items-center gap-1 text-sm font-semibold text-err"><AlertTriangle size={14} /> Conflict on the blackboard</div>
              <div className="grid grid-cols-2 gap-2">
                {conflicts.map((e, i) => (
                  <div key={i} className="rounded border border-err/50 bg-err-soft p-1.5 text-xs"><div className="font-mono">{e.value}</div><div className="text-[10px] text-muted">{name(e.by)} · {e.cited ? "cited source" : "assumption"}</div></div>
                ))}
              </div>
              <p className="text-xs text-muted">{cfg.arbitration ? "Arbitration is on: someone must compare the evidence before work continues." : "Arbitration is off: whichever value is read last will be used."}</p>
            </motion.div>
          ) : m.kind === "arbitrate" || s.bb.some((e) => e.status === "resolved") ? (
            <Callout tone="ok" title="Resolved">The cited value is kept; the other is marked superseded so later readers cannot pick it up by accident.</Callout>
          ) : m.kind === "late" ? (
            <Callout tone="warn" title="Wasted work">The answer arrived after the decision. Cancel abandoned work (propagate a cancellation) to save cost.</Callout>
          ) : m.bb ? (
            <p className="flex items-center gap-1 text-xs"><CheckCircle2 size={13} className="text-ok" /> Posted to the blackboard: <span className="font-mono">{m.bb.key} = {m.bb.value}</span>{m.bb.cited ? " (with a citation)" : ""}.</p>
          ) : <p className="text-xs text-muted">No blackboard change. Messages carry instructions; the blackboard carries shared state.</p>}
        </AnimatePresence>
      </div>
    </div>
  );
}

function BarRow({ label, v, max, cls, k, i }: { label: string; v: number; max: number; cls: string; k: number; i: number }) {
  return (
    <div className="text-xs">
      <div className="flex justify-between"><span>{label}</span><span className="font-mono">{(v / 1000).toFixed(2)} s</span></div>
      <div className="h-3 rounded bg-surface-2"><motion.div className={cx("h-3 rounded", cls)} initial={{ width: 0 }} animate={{ width: `${(v / max) * 100}%` }} transition={{ delay: (0.2 + i * 0.5) * k, duration: 0.7 * k, ease: EASE }} /></div>
    </div>
  );
}

function Calc({ current, s }: { current?: Message; s: CoordState }) {
  if (current?.timer) return <TexBlock>{`t_{\\text{give up}} = t_{\\text{dispatch}} + \\text{timeout} = ${((current.sent - current.timer.after) / 1000).toFixed(2)} + ${(current.timer.after / 1000).toFixed(1)} = ${(current.sent / 1000).toFixed(2)}\\,\\text{s}`}</TexBlock>;
  if (current) return <TexBlock>{`\\begin{aligned} t_{\\text{sent}} &= t_{\\text{ready}} + t_{\\text{work}} = ${((current.sent - current.work) / 1000).toFixed(2)} + ${(current.work / 1000).toFixed(2)} = ${(current.sent / 1000).toFixed(2)}\\,\\text{s} \\\\ t_{\\text{arrive}} &= t_{\\text{sent}} + ${NET_MS / 1000} = ${(current.arrive / 1000).toFixed(2)}\\,\\text{s} \\end{aligned}`}</TexBlock>;
  if (s.done) return <TexBlock>{`\\text{parallelism} = \\frac{\\sum t_{\\text{work}}}{t_{\\text{wall}}} = \\frac{${(s.totals.work / 1000).toFixed(2)}}{${(s.totals.clock / 1000).toFixed(2)}} = ${(s.totals.work / Math.max(1, s.totals.clock)).toFixed(2)}`}</TexBlock>;
  return <TexBlock>{`t_{\\text{sent}} = t_{\\text{ready}}(\\text{sender}) + t_{\\text{work}}, \\quad t_{\\text{arrive}} = t_{\\text{sent}} + ${NET_MS}\\,\\text{ms}`}</TexBlock>;
}
