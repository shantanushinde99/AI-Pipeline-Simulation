import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, NumberField, Panel, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, pct } from "../offline-evaluation/kit";
import { DEFAULT_ONLINE, LATENCY_MARGIN, MDE, NULL_EXPERIMENTS, simulate, type OnlineParams, type OnlineSim, type OnlineState } from "./simulation";

/** One schedule (seconds at 1×). */
const DOT = { start: 0.15, gap: 0.06, travel: 1.0 };
const DAY_END = DOT.start + 29 * DOT.gap + DOT.travel; // last dot lands
const POINT_AT = 2.3;
const PEEK = { start: 0.3, perDay: 0.22 };
const DUR = { setup: 3000, day: 3400, guardrail: 3600, decision: 3600, peeking: 0 };
const duration = (s: OnlineState, days: number) => (s.phase === "peeking" ? (PEEK.start + days * PEEK.perDay + 1.4) * 1000 : DUR[s.phase]);

const PRESETS: { label: string; p: Partial<OnlineParams> }[] = [
  { label: "Real +1.5 pt lift", p: { trueLift: 0.015, peeking: false, addedLatency: 40, seed: 3 } },
  { label: "A/A test + peeking", p: { trueLift: 0, peeking: true, addedLatency: 40, seed: 8 } },
  { label: "Lift, but slow (guardrail)", p: { trueLift: 0.015, peeking: false, addedLatency: 180, seed: 3 } },
];

const sp = (v: number, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}`;

function narrate(sim: OnlineSim, s: OnlineState) {
  const p = sim.params;
  if (s.phase === "setup") return { headline: "Plan first: metric, guardrail, sample size, horizon", sub: `To detect +${(MDE * 100).toFixed(1)} pts you need ${sim.nNeeded.toLocaleString()} users per arm: about ${sim.daysNeeded} days. The plan runs ${p.days}.` };
  if (s.phase === "day") {
    const r = sim.rows[s.day - 1];
    const stop = sim.stoppedEarly && s.day === sim.final.day;
    return {
      headline: stop ? `Day ${s.day}: p = ${r.test.p.toFixed(3)} < 0.05, the peeker stops and ships` : `Day ${s.day}: users split into A and B; conversions accumulate`,
      sub: stop ? "Stopping at the first significant look is exactly what inflates false positives." : `Δ = ${sp(r.test.diff)} pts, 95% CI ${sp(r.test.ci[0])} to ${sp(r.test.ci[1])}, p = ${r.test.p.toFixed(3)}. The interval shrinks as n grows.`,
    };
  }
  if (s.phase === "guardrail") return { headline: sim.guard.pass ? "Guardrail holds: B is not meaningfully slower" : "Guardrail fails: B is too slow", sub: `p95 latency increase: ${sim.guard.mean.toFixed(0)} ms on average, upper bound ${sim.guard.upper.toFixed(0)} ms vs a ${LATENCY_MARGIN} ms limit.` };
  if (s.phase === "decision") return {
    headline: sim.decision === "ship" ? "Decision: ship B" : sim.decision === "blocked" ? "Decision: launch blocked by the guardrail" : "Decision: no significant difference, keep A",
    sub: sim.stoppedEarly ? `Ghost points show what the test would have said had it run all ${p.days} days: p = ${sim.fullFinal.test.p.toFixed(3)}.` : "The rule was written before the test started, so the data cannot bend it.",
  };
  return { headline: `Peeking every day turns 5% false positives into ${pct(sim.nulls.peekFpr, 0)}`, sub: `${NULL_EXPERIMENTS} A/A tests (no real difference). Red = stopped on a false "win" at some daily look.` };
}

export default function OnlineEvaluationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<OnlineParams>(DEFAULT_ONLINE);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(params), [params]);
  const pb = usePlayback(sim.steps.length, (i) => duration(sim.steps[i]?.state ?? { phase: "setup", day: 0 }, sim.params.days));
  const step = sim.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<OnlineParams>) => setParams((c) => ({ ...c, ...p }));
  if (tab === "metrics") return null;
  const story = narrate(sim, s);
  const row = s.day > 0 ? sim.rows[Math.min(s.day, sim.rows.length) - 1] : null;

  return (
    <Workspace
      main={
        <>
          <Panel title="A/B experiment" actions={<SimulatedLabel>Simulated users</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[580px]"><Canvas sim={sim} s={s} k={k} animKey={animKey} /></div></div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} s={s} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_ONLINE); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="mb-3 flex flex-wrap gap-1.5">
              <span className="self-center text-xs text-muted">Presets:</span>
              {PRESETS.map((x) => <Button key={x.label} size="sm" onClick={() => { set(x.p); pb.restart(); }}>{x.label}</Button>)}
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Slider label="Baseline conversion (A)" value={params.baseRate} min={0.02} max={0.3} step={0.01} format={(v) => pct(v, 0)} onChange={(v) => set({ baseRate: v })} />
              <Slider label="True lift of B (unknown in real life)" value={params.trueLift} min={-0.02} max={0.04} step={0.0025} format={(v) => `${sp(v)} pts`} onChange={(v) => set({ trueLift: v })} />
              <Slider label="Users per day (both arms)" value={params.usersPerDay} min={200} max={3000} step={100} onChange={(v) => set({ usersPerDay: v })} />
              <Slider label="Planned days" value={params.days} min={7} max={28} onChange={(v) => set({ days: v })} />
              <Slider label="Break it: latency B adds (p95)" value={params.addedLatency} min={0} max={300} step={10} format={(v) => `${v} ms`} onChange={(v) => set({ addedLatency: v })} hint={`Guardrail: ≤ ${LATENCY_MARGIN} ms increase`} />
              <div className="space-y-2">
                <Toggle label="Peeking: stop at the first daily p < 0.05" checked={params.peeking} onChange={(v) => set({ peeking: v })} hint="The classic mistake. Try it with the A/A preset." />
                <NumberField label="Seed (which users convert)" value={params.seed} step={1} min={1} max={9999} onChange={(v) => set({ seed: Math.round(v) })} />
              </div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.phase === "peeking" ? "A p-value is only valid at the sample size you fixed in advance. Every extra look is another chance for noise to cross 0.05." : "Offline evals predict; an online test measures real user behaviour under randomization, so the difference is causal."} />
          <Panel title="Calculation"><Calculation sim={sim} s={s} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["users per arm needed", sim.nNeeded.toLocaleString()],
              ["users so far (A / B)", row ? `${row.nA.toLocaleString()} / ${row.nB.toLocaleString()}` : "0 / 0"],
              ["conversion A / B", row ? `${pct(row.test.pA, 2)} / ${pct(row.test.pB, 2)}` : "—"],
              ["Δ, 95% CI (pts)", row ? `${sp(row.test.diff)} [${sp(row.test.ci[0])}, ${sp(row.test.ci[1])}]` : "—"],
              ["p-value", row ? row.test.p.toFixed(4) : "—"],
              ["false-positive rate (peek)", s.phase === "peeking" ? pct(sim.nulls.peekFpr) : "see last step"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- canvas ---------- */
const W = 680, H = 400, CX0 = 330, CX1 = 668;

function Canvas({ sim, s, k, animKey }: { sim: OnlineSim; s: OnlineState; k: number; animKey: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`A/B test, ${s.phase === "day" ? `day ${s.day}` : s.phase}`}>
      <g key={`left-${animKey}`}>
        {s.phase === "setup" && <PlanBars sim={sim} k={k} />}
        {s.phase === "day" && <Traffic sim={sim} day={s.day} k={k} />}
        {s.phase === "guardrail" && <Latency sim={sim} k={k} />}
        {s.phase === "decision" && <Rules sim={sim} k={k} />}
        {s.phase === "peeking" && <NullGrid sim={sim} k={k} />}
      </g>
      <line x1={300} x2={300} y1={10} y2={H - 10} stroke="var(--line)" />
      {s.phase === "peeking" ? <FprChart sim={sim} k={k} animKey={animKey} /> : <Charts sim={sim} s={s} k={k} animKey={animKey} />}
    </svg>
  );
}

function PlanBars({ sim, k }: { sim: OnlineSim; k: number }) {
  const planned = (sim.params.usersPerDay / 2) * sim.params.days;
  const max = Math.max(planned, sim.nNeeded);
  const rows = [{ l: "needed per arm", v: sim.nNeeded, c: "var(--violet)" }, { l: "planned per arm", v: planned, c: planned >= sim.nNeeded ? "var(--ok)" : "var(--warn)" }];
  return (
    <g transform="translate(14 40)">
      <text fontSize={12} fontWeight={600} fill="var(--fg)">Sample size plan (α = 0.05, power 80%)</text>
      {rows.map((r, i) => (
        <g key={r.l} transform={`translate(0 ${34 + i * 56})`}>
          <text fontSize={11} fill="var(--muted)">{r.l}</text>
          <rect y={8} width={260} height={20} rx={3} fill="var(--surface-2)" />
          <motion.rect y={8} height={20} rx={3} fill={r.c} initial={{ width: 0 }} animate={{ width: (r.v / max) * 260 }} transition={{ delay: (0.3 + i * 0.7) * k, duration: 0.8 * k, ease }} />
          <text x={258} y={0} textAnchor="end" fontSize={12} fontFamily="var(--font-mono)" fontWeight={600} fill="var(--fg)"><AnimatedNumber svg value={r.v} from={0} digits={0} delay={(0.3 + i * 0.7) * k} duration={0.8 * k} /></text>
        </g>
      ))}
      <motion.text y={170} fontSize={11} fill={planned >= sim.nNeeded ? "var(--ok)" : "var(--warn)"} fontWeight={600} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.9 * k }}>
        {planned >= sim.nNeeded ? `✓ ${sim.params.days} days is enough` : `⚠ underpowered: needs ~${sim.daysNeeded} days`}
      </motion.text>
      <motion.text y={195} fontSize={10.5} fill="var(--muted)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.1 * k }}>Guardrail: p95 latency may rise ≤ {LATENCY_MARGIN} ms.</motion.text>
      <motion.text y={213} fontSize={10.5} fill="var(--muted)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.1 * k }}>Decide only at day {sim.params.days}. Write it down now.</motion.text>
    </g>
  );
}

function Traffic({ sim, day, k }: { sim: OnlineSim; day: number; k: number }) {
  const r = sim.rows[day - 1];
  const prev = day > 1 ? sim.rows[day - 2] : { nA: 0, cA: 0, nB: 0, cB: 0 };
  const src: [number, number] = [34, 200];
  const arms = { A: { y: 95, n: r.nA, c: r.cA, n0: prev.nA, c0: prev.cA, color: "var(--accent)" }, B: { y: 305, n: r.nB, c: r.cB, n0: prev.nB, c0: prev.cB, color: "var(--violet)" } };
  return (
    <g>
      <text x={8} y={18} fontSize={11} fontWeight={600} fill="var(--fg)">Day {day}: {sim.params.usersPerDay.toLocaleString()} users randomized 50/50</text>
      <circle cx={src[0]} cy={src[1]} r={22} fill="var(--surface-2)" stroke="var(--line)" />
      <text x={src[0]} y={src[1] + 4} textAnchor="middle" fontSize={10} fill="var(--fg)">users</text>
      {(["A", "B"] as const).map((a) => {
        const m = arms[a];
        return (
          <g key={a} transform={`translate(150 ${m.y - 62})`}>
            <path d={`M ${src[0] - 150 + 22} ${src[1] - m.y + 62} C ${-40} ${src[1] - m.y + 62}, ${-60} 62, 0 62`} fill="none" stroke="var(--line)" strokeDasharray="3 4" />
            <rect width={136} height={124} rx={10} fill="var(--surface)" stroke={m.color} strokeWidth={1.5} />
            <text x={10} y={20} fontSize={12} fontWeight={700} fill={m.color}>Arm {a}{a === "A" ? " (control)" : " (new)"}</text>
            <text x={10} y={44} fontSize={10} fill="var(--muted)">users</text>
            <text x={126} y={44} textAnchor="end" fontSize={12} fontFamily="var(--font-mono)" fill="var(--fg)"><AnimatedNumber svg value={m.n} from={m.n0} digits={0} delay={DOT.start * k} duration={(DAY_END - DOT.start) * k} /></text>
            <text x={10} y={64} fontSize={10} fill="var(--muted)">✓ converted</text>
            <text x={126} y={64} textAnchor="end" fontSize={12} fontFamily="var(--font-mono)" fill="var(--ok)"><AnimatedNumber svg value={m.c} from={m.c0} digits={0} delay={DOT.start * k} duration={(DAY_END - DOT.start) * k} /></text>
            <text x={10} y={88} fontSize={10} fill="var(--muted)">rate</text>
            <text x={126} y={88} textAnchor="end" fontSize={13} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">
              <AnimatedNumber svg value={(m.c / Math.max(m.n, 1)) * 100} from={(m.c0 / Math.max(m.n0, 1)) * 100} digits={2} delay={DOT.start * k} duration={(DAY_END - DOT.start) * k} />%
            </text>
            <rect x={10} y={100} width={116} height={8} rx={4} fill="var(--surface-2)" />
            <motion.rect x={10} y={100} height={8} rx={4} fill="var(--ok)" initial={{ width: Math.min(116, (m.c0 / Math.max(m.n0, 1)) * 116 * 4) }} animate={{ width: Math.min(116, (m.c / Math.max(m.n, 1)) * 116 * 4) }} transition={{ delay: DOT.start * k, duration: (DAY_END - DOT.start) * k }} />
          </g>
        );
      })}
      {r.sample.map((u, i) => {
        const ty = arms[u.arm].y + (i % 5) * 6 - 12;
        return (
          <motion.g key={i} initial={{ x: src[0], y: src[1], opacity: 0 }} animate={{ x: [src[0], 100, 146], y: [src[1], (src[1] + ty) / 2, ty], opacity: [0, 1, 1, 0] }}
            transition={{ delay: (DOT.start + i * DOT.gap) * k, duration: DOT.travel * k, ease: "easeInOut", opacity: { delay: (DOT.start + i * DOT.gap) * k, duration: DOT.travel * k, times: [0, 0.1, 0.85, 1] } }}>
            <circle r={4.5} fill={u.conv ? "var(--ok)" : "var(--muted)"} stroke="var(--surface)" strokeWidth={1} />
            {u.conv && <text y={-7} textAnchor="middle" fontSize={8} fill="var(--ok)">✓</text>}
          </motion.g>
        );
      })}
      <text x={8} y={H - 12} fontSize={9.5} fill="var(--muted)">dots: today's first 30 users (green ✓ = converted)</text>
    </g>
  );
}

function Latency({ sim, k }: { sim: OnlineSim; k: number }) {
  const d = sim.rows.map((r) => r.latB - r.latA);
  const max = Math.max(LATENCY_MARGIN * 1.6, ...d.map(Math.abs), sim.guard.upper + 20);
  const x0 = 40, x1 = 280, base = 300, top = 60;
  const sy = (v: number) => base - (Math.max(0, v) / max) * (base - top);
  const bw = (x1 - x0) / d.length;
  return (
    <g>
      <text x={8} y={18} fontSize={11} fontWeight={600} fill="var(--fg)">Guardrail: daily p95 latency, B − A (ms)</text>
      <line x1={x0} x2={x1} y1={base} y2={base} stroke="var(--line)" />
      {d.map((v, i) => (
        <motion.rect key={i} x={x0 + i * bw + 1} width={Math.max(1, bw - 2)} fill={v > LATENCY_MARGIN ? "var(--err)" : "var(--accent)"} fillOpacity={0.7}
          initial={{ y: base, height: 0 }} animate={{ y: sy(v), height: base - sy(v) }} transition={{ delay: (0.2 + i * 0.08) * k, duration: 0.4 * k }} />
      ))}
      <line x1={x0} x2={x1} y1={sy(LATENCY_MARGIN)} y2={sy(LATENCY_MARGIN)} stroke="var(--err)" strokeDasharray="5 4" />
      <text x={x0} y={sy(LATENCY_MARGIN) - 5} fontSize={10} fill="var(--err)">limit +{LATENCY_MARGIN} ms</text>
      <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.4 + d.length * 0.08) * k }}>
        <line x1={x0} x2={x1} y1={sy(sim.guard.mean)} y2={sy(sim.guard.mean)} stroke="var(--fg)" strokeWidth={1.5} />
        <line x1={x0} x2={x1} y1={sy(sim.guard.upper)} y2={sy(sim.guard.upper)} stroke="var(--violet)" strokeWidth={1.5} strokeDasharray="2 3" />
        <text x={x1} y={sy(sim.guard.upper) - 4} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)" fill="var(--violet)">upper 95%: {sim.guard.upper.toFixed(0)}</text>
        <text x={x1} y={sy(sim.guard.mean) + 12} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">mean {sim.guard.mean.toFixed(0)}</text>
      </motion.g>
      <motion.text x={8} y={340} fontSize={12} fontWeight={700} fill={sim.guard.pass ? "var(--ok)" : "var(--err)"} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (0.8 + d.length * 0.08) * k }}>
        {sim.guard.pass ? "✓ Guardrail passes" : "✗ Guardrail fails: launch blocked"}
      </motion.text>
      <text x={x0} y={base + 14} fontSize={9.5} fill="var(--muted)">day 1 → {d.length}</text>
    </g>
  );
}

function Rules({ sim, k }: { sim: OnlineSim; k: number }) {
  const t = sim.final.test;
  const rules = [
    { l: "Primary: p < 0.05", ok: t.p < 0.05, v: `p = ${t.p.toFixed(3)}` },
    { l: "Primary: Δ > 0", ok: t.diff > 0, v: `${sp(t.diff)} pts` },
    { l: `Guardrail: p95 ≤ +${LATENCY_MARGIN} ms`, ok: sim.guard.pass, v: `≤ ${sim.guard.upper.toFixed(0)} ms` },
    { l: "Ran the planned horizon", ok: !sim.stoppedEarly, v: `${sim.final.day}/${sim.params.days} days` },
  ];
  const label = sim.decision === "ship" ? (sim.stoppedEarly ? "SHIP (peeked!)" : "SHIP") : sim.decision === "blocked" ? "BLOCKED" : "NO SHIP";
  const tone = sim.decision === "ship" && !sim.stoppedEarly ? "var(--ok)" : sim.decision === "no-ship" ? "var(--warn)" : "var(--err)";
  return (
    <g>
      <text x={8} y={18} fontSize={11} fontWeight={600} fill="var(--fg)">Pre-registered launch rule</text>
      {rules.map((r, i) => (
        <g key={r.l} transform={`translate(8 ${44 + i * 40})`}><motion.g initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.45) * k, duration: 0.35 * k }}>
          <rect width={280} height={32} rx={6} fill={r.ok ? "var(--ok-soft)" : "var(--err-soft)"} stroke={r.ok ? "var(--ok)" : "var(--err)"} strokeOpacity={0.5} />
          <text x={10} y={20} fontSize={11} fill="var(--fg)">{r.ok ? "✓" : "✗"} {r.l}</text>
          <text x={272} y={20} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">{r.v}</text>
        </motion.g></g>
      ))}
      <motion.g initial={{ opacity: 0, scale: 1.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 2.3 * k, type: "spring", stiffness: 200, damping: 14 }} style={{ transformOrigin: "148px 250px" }}>
        <rect x={38} y={222} width={220} height={52} rx={8} fill="none" stroke={tone} strokeWidth={3} />
        <text x={148} y={256} textAnchor="middle" fontSize={20} fontWeight={800} fill={tone}>{label}</text>
      </motion.g>
      {sim.stoppedEarly && <motion.text x={8} y={302} fontSize={10.5} fill="var(--err)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.8 * k }}>Full horizon would give p = {sim.fullFinal.test.p.toFixed(3)} ({sim.fullFinal.test.p < 0.05 ? "still significant" : "not significant"}).</motion.text>}
    </g>
  );
}

function NullGrid({ sim, k }: { sim: OnlineSim; k: number }) {
  const cols = 20, size = 11, pitch = 13.5;
  return (
    <g>
      <text x={8} y={18} fontSize={11} fontWeight={600} fill="var(--fg)">{NULL_EXPERIMENTS} A/A experiments, checked daily</text>
      {sim.nulls.firstSig.map((f, i) => {
        const x = 8 + (i % cols) * pitch, y = 34 + Math.floor(i / cols) * pitch;
        return (
          <g key={i}>
            <rect x={x} y={y} width={size} height={size} rx={2} fill="var(--surface-2)" stroke={sim.nulls.finalSig[i] ? "var(--fg)" : "none"} strokeWidth={1.5} />
            {f !== null && (
              <motion.rect x={x} y={y} width={size} height={size} rx={2} fill="var(--err)" initial={{ opacity: 0, scale: 0.2 }} animate={{ opacity: 0.85, scale: 1 }}
                transition={{ delay: (PEEK.start + (f - 0.5) * PEEK.perDay) * k, duration: 0.25 * k }} />
            )}
          </g>
        );
      })}
      <g transform="translate(8 186)">
        <text fontSize={11} fill="var(--fg)">day <tspan fontFamily="var(--font-mono)" fontWeight={700}><AnimatedNumber svg value={sim.params.days} from={0} digits={0} delay={PEEK.start * k} duration={sim.params.days * PEEK.perDay * k} /></tspan></text>
        <text y={22} fontSize={11} fill="var(--err)">■ false "win" at some look: <tspan fontFamily="var(--font-mono)" fontWeight={700}>{sim.nulls.firstSig.filter((f) => f !== null).length}</tspan></text>
        <text y={42} fontSize={11} fill="var(--fg)">□ outlined = significant at the final day only check: <tspan fontFamily="var(--font-mono)" fontWeight={700}>{sim.nulls.finalSig.filter(Boolean).length}</tspan></text>
        <text y={70} fontSize={10} fill="var(--muted)">Each square: {sim.params.days} days × {sim.params.usersPerDay} users, A and B</text>
        <text y={84} fontSize={10} fill="var(--muted)">have the same true rate ({pct(sim.params.baseRate, 0)}). Any "win" is noise.</text>
      </g>
    </g>
  );
}

function Charts({ sim, s, k, animKey }: { sim: OnlineSim; s: OnlineState; k: number; animKey: string }) {
  const D = sim.params.days;
  const sx = (d: number) => CX0 + ((d - 1) / Math.max(1, D - 1)) * (CX1 - CX0);
  const lo = Math.min(-0.01, ...sim.all.slice(1).map((r) => r.test.ci[0]), sim.params.trueLift), hi = Math.max(0.01, ...sim.all.slice(1).map((r) => r.test.ci[1]), sim.params.trueLift);
  const T1 = 34, B1 = 180;
  const sy = (v: number) => B1 - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (B1 - T1);
  const T2 = 226, B2 = 372;
  const py = (p: number) => T2 + (-Math.log10(Math.max(p, 1e-4)) / 4) * (B2 - T2);
  const shown = s.phase === "setup" ? [] : sim.rows.filter((r) => r.day <= s.day);
  const ghosts = s.phase === "decision" && sim.stoppedEarly ? sim.all.filter((r) => r.day > sim.final.day) : [];
  const newest = s.phase === "day" ? s.day : -1;
  const line = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const pointDelay = POINT_AT * k;
  return (
    <g>
      <text x={CX0 - 20} y={18} fontSize={11} fontWeight={600} fill="var(--fg)">Δ conversion (B − A), pts, with 95% CI</text>
      <line x1={CX0} x2={CX1} y1={sy(0)} y2={sy(0)} stroke="var(--fg)" strokeOpacity={0.6} />
      <text x={CX0 - 6} y={sy(0) + 3} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">0</text>
      <text x={CX0 - 6} y={T1 + 4} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{sp(hi, 1)}</text>
      <text x={CX0 - 6} y={B1} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{sp(lo, 1)}</text>
      <line x1={CX0} x2={CX1} y1={sy(sim.params.trueLift)} y2={sy(sim.params.trueLift)} stroke="var(--ok)" strokeDasharray="5 4" strokeOpacity={0.8} />
      <text x={CX1} y={sy(sim.params.trueLift) - 4} textAnchor="end" fontSize={9} fill="var(--ok)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">true effect {sp(sim.params.trueLift, 1)} (only the simulation knows)</text>
      {[...shown, ...ghosts].map((r) => {
        const ghost = r.day > sim.final.day && ghosts.length > 0;
        const isNew = r.day === newest;
        const x = sx(r.day), y = sy(r.test.diff);
        const color = ghost ? "var(--muted)" : "var(--accent)";
        return (
          <g key={`${r.day}-${isNew ? animKey : "s"}`} opacity={ghost ? 0.45 : 1}>
            <motion.line x1={x} x2={x} stroke={color} strokeWidth={2} initial={isNew ? { y1: y, y2: y } : false} animate={{ y1: sy(r.test.ci[1]), y2: sy(r.test.ci[0]) }} transition={{ delay: pointDelay + 0.25 * k, duration: 0.5 * k, ease }} />
            <motion.circle cx={x} cy={y} r={3.5} fill={color} initial={isNew ? { scale: 0 } : false} animate={{ scale: 1 }} transition={{ delay: pointDelay, duration: 0.3 * k }} />
          </g>
        );
      })}
      {shown.length > 1 && <path d={line(shown.map((r) => [sx(r.day), sy(r.test.diff)]))} fill="none" stroke="var(--accent)" strokeOpacity={0.5} />}

      <text x={CX0 - 20} y={210} fontSize={11} fontWeight={600} fill="var(--fg)">p-value by day (log scale)</text>
      {[1, 0.1, 0.01, 0.001].map((t) => (
        <g key={t}><line x1={CX0} x2={CX1} y1={py(t)} y2={py(t)} stroke="var(--line)" strokeOpacity={0.6} />
          <text x={CX0 - 6} y={py(t) + 3} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{t}</text></g>
      ))}
      <line x1={CX0} x2={CX1} y1={py(0.05)} y2={py(0.05)} stroke="var(--err)" strokeDasharray="5 4" />
      <text x={CX1} y={py(0.05) - 4} textAnchor="end" fontSize={9} fill="var(--err)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">α = 0.05 (below = "significant")</text>
      {shown.length > 1 && <path d={line(shown.map((r) => [sx(r.day), py(r.test.p)]))} fill="none" stroke="var(--violet)" strokeOpacity={0.6} />}
      {ghosts.length > 0 && <path d={line([sim.final, ...ghosts].map((r) => [sx(r.day), py(r.test.p)]))} fill="none" stroke="var(--muted)" strokeDasharray="3 3" />}
      {[...shown, ...ghosts].map((r) => {
        const isNew = r.day === newest, ghost = ghosts.includes(r);
        return <motion.circle key={`p${r.day}-${isNew ? animKey : "s"}`} cx={sx(r.day)} cy={py(r.test.p)} r={3.5} fill={ghost ? "var(--muted)" : r.test.p < 0.05 ? "var(--err)" : "var(--violet)"}
          initial={isNew ? { scale: 0, opacity: 0 } : false} animate={{ scale: 1, opacity: 1 }} transition={{ delay: pointDelay + 0.2 * k, duration: 0.3 * k }} />;
      })}
      {sim.stoppedEarly && s.phase !== "setup" && (s.phase !== "day" || s.day === sim.final.day) && (
        <g><line x1={sx(sim.final.day)} x2={sx(sim.final.day)} y1={T1} y2={B2} stroke="var(--err)" strokeWidth={1.5} />
          <text x={sx(sim.final.day) + 4} y={B2 - 4} fontSize={9.5} fontWeight={600} fill="var(--err)">peeker stops</text></g>
      )}
      {Array.from({ length: D }, (_, i) => i + 1).filter((d) => d === 1 || d === D || d % 7 === 0).map((d) => (
        <text key={d} x={sx(d)} y={B2 + 14} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">d{d}</text>
      ))}
    </g>
  );
}

function FprChart({ sim, k, animKey }: { sim: OnlineSim; k: number; animKey: string }) {
  const D = sim.params.days;
  const sx = (d: number) => CX0 + ((d - 1) / Math.max(1, D - 1)) * (CX1 - CX0);
  const top = Math.max(0.15, sim.nulls.peekFpr + 0.05);
  const T = 40, B = 300;
  const sy = (v: number) => B - (v / top) * (B - T);
  const d = sim.nulls.cumulative.map((v, i) => `${i ? "L" : "M"}${sx(i + 1).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  const end = (PEEK.start + D * PEEK.perDay) * k;
  return (
    <g key={animKey}>
      <text x={CX0 - 20} y={18} fontSize={11} fontWeight={600} fill="var(--fg)">Share of A/A tests that "found a winner"</text>
      {[0, 0.05, 0.1, 0.2, 0.3, 0.4].filter((t) => t <= top).map((t) => (
        <g key={t}><line x1={CX0} x2={CX1} y1={sy(t)} y2={sy(t)} stroke="var(--line)" strokeOpacity={0.6} />
          <text x={CX0 - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{(t * 100).toFixed(0)}%</text></g>
      ))}
      <line x1={CX0} x2={CX1} y1={sy(0.05)} y2={sy(0.05)} stroke="var(--ok)" strokeDasharray="5 4" />
      <text x={CX0 + 4} y={sy(0.05) + 12} fontSize={9.5} fill="var(--ok)">nominal 5% (what α promises)</text>
      <motion.path d={d} fill="none" stroke="var(--err)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: PEEK.start * k, duration: D * PEEK.perDay * k, ease: "linear" }} />
      <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: end }}>
        <text x={CX1} y={sy(sim.nulls.peekFpr) - 8} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke" textAnchor="end" fontSize={11} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--err)">peeking: {pct(sim.nulls.peekFpr)}</text>
        <circle cx={sx(D)} cy={sy(sim.nulls.fixedFpr)} r={5} fill="var(--fg)" />
        <text x={CX1 - 10} y={sy(sim.nulls.fixedFpr) - 10} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke" textAnchor="end" fontSize={11} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">one look at day {D}: {pct(sim.nulls.fixedFpr)}</text>
      </motion.g>
      {Array.from({ length: D }, (_, i) => i + 1).filter((x) => x === 1 || x === D || x % 7 === 0).map((x) => (
        <text key={x} x={sx(x)} y={B + 14} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">d{x}</text>
      ))}
      <text x={CX0} y={B + 40} fontSize={10} fill="var(--muted)">Each daily look is another chance for noise to dip below 0.05.</text>
      <text x={CX0} y={B + 54} fontSize={10} fill="var(--muted)">Fixes: a fixed horizon, or sequential tests (alpha spending).</text>
    </g>
  );
}

/* ---------- card ---------- */
function StepCard({ sim, s, k }: { sim: OnlineSim; s: OnlineState; k: number }) {
  if (s.phase === "setup") {
    const planned = (sim.params.usersPerDay / 2) * sim.params.days;
    return (
      <div className="grid gap-2 text-xs sm:grid-cols-2">
        <KV rows={[["primary metric", "conversion rate"], ["guardrail", `p95 latency ≤ +${LATENCY_MARGIN} ms`], ["α (two-sided)", "0.05"], ["power", "80%"], ["min. detectable effect", `+${(MDE * 100).toFixed(1)} pts`]]} />
        <div className="space-y-2">
          <KV rows={[["users needed / arm", sim.nNeeded.toLocaleString()], ["planned / arm", planned.toLocaleString()], ["days needed", sim.daysNeeded]]} />
          {planned < sim.nNeeded ? <Callout tone="warn">Underpowered: even a real +{(MDE * 100).toFixed(1)} pt lift will often look insignificant.</Callout> : <Callout tone="ok">Enough traffic to detect the planned effect most of the time.</Callout>}
        </div>
      </div>
    );
  }
  if (s.phase === "day") {
    const recent = sim.rows.filter((r) => r.day <= s.day).slice(-5);
    const first = sim.rows[0].test, cur = sim.rows[s.day - 1].test;
    return (
      <div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">day</th><th>A conv / users</th><th>B conv / users</th><th>Δ pts</th><th>95% CI</th><th>p</th></tr></thead>
            <tbody className="font-mono">
              <AnimatePresence initial={false}>
                {recent.map((r) => (
                  <motion.tr key={r.day} layout className={cx("border-t border-line", r.day === s.day && "bg-accent-soft")} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: r.day === s.day ? POINT_AT * k : 0, duration: 0.3 }}>
                    <td className="py-1">{r.day}</td><td>{r.cA}/{r.nA}</td><td>{r.cB}/{r.nB}</td><td>{sp(r.test.diff)}</td><td>[{sp(r.test.ci[0])}, {sp(r.test.ci[1])}]</td>
                    <td className={r.test.p < 0.05 ? "font-semibold text-err" : undefined}>{r.test.p.toFixed(3)}{r.test.p < 0.05 ? " *" : ""}</td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">CI width: {((first.ci[1] - first.ci[0]) * 100).toFixed(2)} pts on day 1 → {((cur.ci[1] - cur.ci[0]) * 100).toFixed(2)} pts now. It shrinks like 1/√n: 4× the users, half the width. * = p &lt; 0.05 at this look.</p>
      </div>
    );
  }
  if (s.phase === "guardrail") {
    return (
      <div className="grid gap-3 text-xs sm:grid-cols-2">
        <KV rows={[["mean daily Δ p95", `${sim.guard.mean.toFixed(1)} ms`], ["std. dev. of daily Δ", `${sim.guard.sd.toFixed(1)} ms`], ["days", sim.rows.length], ["upper 95% bound", `${sim.guard.upper.toFixed(1)} ms`], ["limit", `${LATENCY_MARGIN} ms`]]} />
        <Callout tone={sim.guard.pass ? "ok" : "err"} title={sim.guard.pass ? "Not meaningfully slower" : "Blocks the launch"}>
          A guardrail is a metric the change must not hurt, tested as non-inferiority: the worst plausible increase (upper bound) must stay under the limit.
        </Callout>
      </div>
    );
  }
  if (s.phase === "decision") {
    return (
      <div className="space-y-2 text-xs">
        <pre className="overflow-x-auto rounded-md bg-surface-2 p-2 font-mono text-[11px]">{`ship_B = (p < 0.05 and delta > 0)        # primary metric
     and (p95_increase_upper <= ${LATENCY_MARGIN} ms)   # guardrail
     # evaluated once, at day ${sim.params.days} (pre-registered)`}</pre>
        <div className="flex flex-wrap items-center gap-2">
          <span>Outcome:</span>
          {sim.decision === "ship" ? <Badge tone="ok"><CheckCircle2 size={11} /> ship B</Badge> : sim.decision === "blocked" ? <Badge tone="err"><XCircle size={11} /> blocked</Badge> : <Badge tone="warn">keep A</Badge>}
          {sim.stoppedEarly && <Badge tone="err">decided on day {sim.final.day} by peeking</Badge>}
        </div>
      </div>
    );
  }
  const bars = [{ l: "promised by α", v: 0.05, c: "bg-ok/70" }, { l: `one look at day ${sim.params.days}`, v: sim.nulls.fixedFpr, c: "bg-fg/50" }, { l: "peek daily, stop at first p < 0.05", v: sim.nulls.peekFpr, c: "bg-err/70" }];
  const max = Math.max(...bars.map((b) => b.v), 0.1);
  return (
    <div className="space-y-1.5">
      {bars.map((b, i) => (
        <div key={b.l} className="grid grid-cols-[minmax(0,190px)_minmax(0,1fr)_52px] items-center gap-2 text-xs">
          <span>{b.l}</span>
          <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-4 rounded", b.c)} initial={{ width: 0 }} animate={{ width: `${(b.v / max) * 100}%` }} transition={{ delay: (0.3 + i * 0.5) * k, duration: 0.6 * k, ease }} /></div>
          <span className="text-right font-mono">{pct(b.v)}</span>
        </div>
      ))}
      <p className="text-xs text-muted">False-positive rate = share of A/A tests that declared a winner. The fixed-horizon rate sits near 5% (sampling noise over {NULL_EXPERIMENTS} tests); peeking multiplies it.</p>
    </div>
  );
}

function Calculation({ sim, s }: { sim: OnlineSim; s: OnlineState }) {
  const f = (v: number, d = 4) => v.toFixed(d);
  if (s.phase === "setup") {
    const p1 = sim.params.baseRate, p2 = p1 + MDE;
    return <TexBlock>{`\\begin{aligned} n &= \\frac{(z_{0.975} + z_{0.80})^2\\,[p_1(1-p_1) + p_2(1-p_2)]}{\\delta^2} \\\\ &= \\frac{(1.960 + 0.842)^2\\,[${f(p1 * (1 - p1))} + ${f(p2 * (1 - p2))}]}{${MDE}^2} \\\\ &= ${sim.nNeeded} \\text{ per arm} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "day") {
    const r = sim.rows[s.day - 1], t = r.test;
    const pool = (r.cA + r.cB) / (r.nA + r.nB);
    return <TexBlock>{`\\begin{aligned} \\hat p_A &= \\tfrac{${r.cA}}{${r.nA}} = ${f(t.pA)} \\\\ \\hat p_B &= \\tfrac{${r.cB}}{${r.nB}} = ${f(t.pB)} \\\\ \\bar p &= ${f(pool)} \\\\ z &= \\frac{\\hat p_B - \\hat p_A}{\\sqrt{\\bar p(1-\\bar p)(\\frac{1}{n_A} + \\frac{1}{n_B})}} = ${f(t.z, 3)} \\\\ p &= 2\\,(1 - \\Phi(|z|)) = ${f(t.p)} \\\\ \\text{CI} &= \\Delta \\pm 1.96\\,SE = ${f(t.diff)} \\pm ${f(1.96 * t.se)} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "guardrail") {
    const g = sim.guard;
    return <TexBlock>{`\\begin{aligned} d_t &= \\text{p95}_B(t) - \\text{p95}_A(t) \\\\ U &= \\bar d + 1.96\\,\\tfrac{s_d}{\\sqrt{n}} = ${g.mean.toFixed(1)} + 1.96 \\cdot \\tfrac{${g.sd.toFixed(1)}}{\\sqrt{${sim.rows.length}}} \\\\ &= ${g.upper.toFixed(1)} \\text{ ms} \\; ${g.pass ? "\\le" : ">"} \\; ${LATENCY_MARGIN} \\text{ ms} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "decision") {
    const t = sim.final.test;
    return <TexBlock>{`\\begin{aligned} \\text{ship} &= [p < 0.05] \\wedge [\\Delta > 0] \\wedge [U \\le ${LATENCY_MARGIN}] \\\\ &= [${f(t.p, 3)} < 0.05] \\wedge [${f(t.diff)} > 0] \\\\ &\\quad \\wedge [${sim.guard.upper.toFixed(0)} \\le ${LATENCY_MARGIN}] \\\\ &= \\text{${sim.decision === "ship" ? "true" : "false"}} \\end{aligned}`}</TexBlock>;
  }
  const n = sim.nulls;
  return <TexBlock>{`\\begin{aligned} \\text{FPR}_{\\text{fixed}} &= \\tfrac{${n.finalSig.filter(Boolean).length}}{${NULL_EXPERIMENTS}} = ${f(n.fixedFpr, 3)} \\\\ \\text{FPR}_{\\text{peek}} &= \\tfrac{${n.firstSig.filter((x) => x !== null).length}}{${NULL_EXPERIMENTS}} = ${f(n.peekFpr, 3)} \\end{aligned}`}</TexBlock>;
}
