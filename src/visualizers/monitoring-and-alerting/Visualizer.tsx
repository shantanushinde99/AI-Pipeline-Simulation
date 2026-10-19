import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { BellRing, CheckCheck, CheckCircle2, Clock, RotateCcw, TriangleAlert } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, useReveal } from "./shared";
import {
  ALERT_STATES, CHUNK, MINUTES, PRESETS, RULE_NAMES, SCENARIOS, evaluate, generateMetrics, isPage, ruleText, simulate, summarize,
  type AlertState, type RuleConfig, type RuleId, type ScenarioId, type WindowEval,
} from "./simulation";

/** One schedule: window i of a chunk is revealed at START + i·EVERY seconds (× 1/speed). */
const START = 0.35, EVERY = 0.55;
const stepMs = (i: number, total: number) => (i === 0 ? 3200 : i === total - 1 ? 3600 : (START + CHUNK * EVERY + 0.9) * 1000);

const STATE_UI: Record<AlertState, { label: string; Icon: typeof CheckCircle2; bg: string; fg: string }> = {
  ok: { label: "OK", Icon: CheckCircle2, bg: "var(--ok-soft)", fg: "var(--ok)" },
  pending: { label: "Pending", Icon: Clock, bg: "var(--warn-soft)", fg: "var(--warn)" },
  firing: { label: "Firing", Icon: BellRing, bg: "var(--err-soft)", fg: "var(--err)" },
  resolved: { label: "Resolved", Icon: CheckCheck, bg: "var(--accent-soft)", fg: "var(--accent)" },
};

export default function MonitoringVisualizer({ tab }: { tab: VisualizerTab }) {
  const [scenario, setScenario] = useState<ScenarioId>("quality-regression");
  const [spikes, setSpikes] = useState(true);
  const [config, setConfig] = useState<RuleConfig>(PRESETS.noisy);
  const [replay, setReplay] = useState(0);
  const seed = 7;

  const steps = useMemo(() => simulate({ scenario, seed, spikes, config }), [scenario, spikes, config]);
  const pb = usePlayback(steps.length, (i) => stepMs(i, steps.length));
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${pb.speed}`;
  const n = useReveal(s.phase === "chunk" ? s.upTo - s.from : 0, animKey, EVERY * k, START * k);
  const visible = s.phase === "chunk" ? s.from + n : s.upTo;
  const evals = s.evals.slice(0, visible);
  const cur = evals.at(-1);
  const set = (patch: Partial<RuleConfig>) => setConfig((c) => ({ ...c, ...patch }));
  const rules = Object.keys(s.evals[0]?.rules ?? evaluate(generateMetrics(scenario, seed, spikes).slice(0, 1), config)[0].rules) as RuleId[];

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Live metrics & alert rules" actions={<SimulatedLabel>Simulated metrics</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} id={step.id} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[560px]">
                <Charts evals={evals} from={s.phase === "chunk" ? s.from : visible} config={config} scenario={scenario} animKey={animKey} k={k} />
              </div>
            </div>
            <div className="mt-3 space-y-2">
              {rules.map((id) => <RuleRow key={id} id={id} config={config} ev={cur?.rules[id]} pages={evals.filter((e) => isPage(e.rules[id])).length} animKey={`${animKey}-${visible}`} />)}
              <Budget ev={cur} total={s.evals[0]?.budgetTotal ?? (1 - config.slo / 100) * 1000 * MINUTES} />
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                {s.phase === "setup" ? <StateMachineIntro config={config} rules={rules} k={k} />
                  : s.phase === "summary" ? <Summary scenario={scenario} spikes={spikes} config={config} k={k} />
                  : <WindowTable rows={evals.slice(s.from)} rules={rules} config={config} />}
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          {pb.index === 0 && !pb.playing && <p className="-mt-1 text-xs text-muted">Press <strong>Play</strong> to stream the hour of metrics, then switch the rule preset and replay to compare.</p>}
          <Panel title="Scenario & alert rules" actions={<Button size="sm" variant="ghost" onClick={() => { setScenario("quality-regression"); setSpikes(true); setConfig(PRESETS.noisy); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="What happens in production" value={scenario} options={SCENARIOS.map((x) => ({ value: x.id, label: x.name }))} onChange={setScenario} />
              <div className="space-y-1">
                <span className="block text-xs font-medium">Rule preset</span>
                <div className="flex flex-wrap gap-1.5">
                  <Button size="sm" variant={JSON.stringify(config) === JSON.stringify(PRESETS.noisy) ? "primary" : "secondary"} onClick={() => setConfig(PRESETS.noisy)}>Noisy alerts</Button>
                  <Button size="sm" variant={JSON.stringify(config) === JSON.stringify(PRESETS.useful) ? "primary" : "secondary"} onClick={() => setConfig(PRESETS.useful)}>Useful alerts</Button>
                </div>
              </div>
            </div>
            <p className="mt-1 text-[11px] text-muted">{SCENARIOS.find((x) => x.id === scenario)!.blurb}</p>
            <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Slider label="Latency threshold" value={config.latencyMs} min={500} max={1200} step={50} format={(v) => `${v} ms`} onChange={(v) => set({ latencyMs: v })} />
              <Slider label="Latency: must hold for" value={config.latencyFor} min={1} max={8} format={(v) => (v === 1 ? "1 min (static)" : `${v} min`)} onChange={(v) => set({ latencyFor: v })} />
              <Select label="Error rule type" value={config.errorMode} options={[{ value: "static", label: "Static error-rate threshold" }, { value: "burn", label: "SLO burn rate" }]} onChange={(v) => set({ errorMode: v })} />
              {config.errorMode === "static"
                ? <Slider label="Error-rate threshold" value={config.errorPct} min={0.2} max={3} step={0.1} format={(v) => `${v.toFixed(1)}%`} onChange={(v) => set({ errorPct: v })} hint="Break it: set it just above the normal 0.2–0.4% noise." />
                : <Slider label="Burn-rate factor" value={config.burnFactor} min={1} max={14} step={0.5} format={(v) => `${v}×`} onChange={(v) => set({ burnFactor: v })} hint="Fires when the 5-min error rate spends budget this many times too fast." />}
              <div className="space-y-2">
                <Toggle label="Quality-score rule" checked={config.qualityOn} onChange={(v) => set({ qualityOn: v })} hint="Sampled LLM-judge score (simulated)." />
                <Toggle label="One-minute latency spikes (noise)" checked={spikes} onChange={setSpikes} />
              </div>
              {config.qualityOn && <Slider label="Quality below" value={config.qualityMin} min={0.6} max={0.85} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set({ qualityMin: v })} />}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Pages wake people up. A rule should fire only for problems users feel, fast enough to act, and stay quiet otherwise." />
          <Panel title="Rule math (current window)">
            <RuleMath ev={cur} config={config} />
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["current minute", cur ? cur.w.t : "—"],
              ["pages sent so far", evals.reduce((t, e) => t + rules.filter((id) => isPage(e.rules[id])).length, 0)],
              ["error budget left", cur ? `${Math.max(0, Math.round(cur.budgetLeft * 100))}%` : "100%"],
              ["rules active", rules.length],
            ]} />
          </Panel>
          <Panel title="Events">
            <EventLog events={step.events} />
          </Panel>
        </>
      }
    />
  );
}

const W = 720, L = 84, R = 78, ROW = 92, GAP = 14;
const xOf = (t: number) => L + (t * (W - L - R)) / (MINUTES - 1);
type Row = { id: RuleId; label: string; lo: number; hi: number; val: (e: WindowEval) => number; fmt: (v: number) => string; thr?: number; thrLabel?: string; above: boolean };

function rowsFor(c: RuleConfig): Row[] {
  const burnPct = c.burnFactor * (100 - c.slo);
  return [
    { id: "latency", label: "p95 latency (ms)", lo: 0, hi: 1500, val: (e) => e.w.latency, fmt: (v) => `${Math.round(v)} ms`, thr: c.latencyMs, thrLabel: `${c.latencyMs} ms`, above: true },
    { id: "errors", label: "error rate (%)", lo: 0, hi: 7, val: (e) => e.w.errorRate * 100, fmt: (v) => `${v.toFixed(1)}%`, thr: c.errorMode === "static" ? c.errorPct : burnPct,
      thrLabel: c.errorMode === "static" ? `${c.errorPct}%` : `burn ${c.burnFactor}× ≈ ${burnPct.toFixed(1)}%`, above: true },
    { id: "quality", label: "quality (judge score)", lo: 0.4, hi: 1, val: (e) => e.w.quality, fmt: (v) => v.toFixed(2), thr: c.qualityOn ? c.qualityMin : undefined, thrLabel: c.qualityOn ? `${c.qualityMin}` : "no rule", above: false },
  ];
}

function Charts({ evals, from, config, scenario, animKey, k }: { evals: WindowEval[]; from: number; config: RuleConfig; scenario: ScenarioId; animKey: string; k: number }) {
  const rows = rowsFor(config);
  const H = 3 * ROW + 2 * GAP + 24;
  const cur = evals.length - 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Three metric time series with alert thresholds">
      {rows.map((r, ri) => {
        const top = ri * (ROW + GAP);
        const y = (v: number) => top + ROW - ((Math.min(Math.max(v, r.lo), r.hi) - r.lo) / (r.hi - r.lo)) * ROW;
        const past = evals.slice(0, from).map((e) => `${xOf(e.w.t)},${y(r.val(e))}`).join(" ");
        const breach = (e: WindowEval) => (r.id === "errors" && config.errorMode === "burn" ? !!e.rules.errors?.breach : r.thr !== undefined && (r.above ? r.val(e) > r.thr : r.val(e) < r.thr));
        const curV = cur >= 0 ? r.val(evals[cur]) : undefined;
        return (
          <g key={r.id}>
            <rect x={L} y={top} width={W - L - R} height={ROW} fill="var(--surface-2)" rx={4} />
            <text x={L - 8} y={top + 14} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--fg)">{r.label.split(" (")[0]}</text>
            <text x={L - 8} y={top + 28} textAnchor="end" fontSize={10} fill="var(--muted)">{r.label.includes("(") ? `(${r.label.split("(")[1]}` : ""}</text>
            {r.thr !== undefined && (
              <g>
                <line x1={L} x2={W - R} y1={y(r.thr)} y2={y(r.thr)} stroke="var(--err)" strokeDasharray="5 4" strokeWidth={1.3} />
                <text x={L + 4} y={y(r.thr) + (r.above ? -4 : 12)} fontSize={10} fill="var(--err)">threshold {r.thrLabel}</text>
              </g>
            )}
            {r.thr === undefined && <text x={L + 4} y={top + 12} fontSize={10} fill="var(--muted)">no alert rule on this metric</text>}
            <polyline points={past} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
            {evals.slice(0, from).filter(breach).map((e) => <circle key={e.w.t} cx={xOf(e.w.t)} cy={y(r.val(e))} r={3} fill="var(--err)" />)}
            {evals.slice(from).map((e) => {
              const t = e.w.t;
              const prev = evals[t - 1];
              const b = breach(e);
              return (
                <g key={`${animKey}-${t}`}>
                  {prev && (
                    <motion.line x1={xOf(t - 1)} y1={y(r.val(prev))} x2={xOf(t)} y2={y(r.val(e))} stroke="var(--accent)" strokeWidth={2}
                      initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.4 * k, ease: "linear" }} />
                  )}
                  <motion.circle cx={xOf(t)} cy={y(r.val(e))} r={b ? 4.5 : 2.5} fill={b ? "var(--err)" : "var(--accent)"}
                    initial={{ scale: 0 }} animate={{ scale: b ? [0, 1.8, 1] : 1 }} transition={{ delay: 0.3 * k, duration: 0.4 * k }} />
                </g>
              );
            })}
            {curV !== undefined && (
              <g>
                <rect x={W - R + 6} y={top + ROW / 2 - 12} width={R - 8} height={24} rx={5} fill={breach(evals[cur]) ? "var(--err-soft)" : "var(--surface)"} stroke={breach(evals[cur]) ? "var(--err)" : "var(--line)"} />
                <text x={W - R / 2 + 2} y={top + ROW / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{breach(evals[cur]) ? "▲ " : ""}{r.fmt(curV)}</text>
              </g>
            )}
          </g>
        );
      })}
      {scenario === "quality-regression" && evals.length > 24 && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 * k }}>
          <line x1={xOf(24)} x2={xOf(24)} y1={0} y2={3 * ROW + 2 * GAP} stroke="var(--violet)" strokeDasharray="2 3" />
          <text x={xOf(24) + 4} y={2 * (ROW + GAP) + 12} fontSize={10} fill="var(--violet)">deploy: prompt v2</text>
        </motion.g>
      )}
      {cur >= 0 && (
        <motion.line y1={0} y2={3 * ROW + 2 * GAP} stroke="var(--fg)" strokeOpacity={0.35} initial={false} animate={{ x1: xOf(cur), x2: xOf(cur) }} transition={{ duration: 0.35 * k, ease }} />
      )}
      {[0, 10, 20, 30, 40, 50, 59].map((t) => <text key={t} x={xOf(t)} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{t}</text>)}
      <text x={L - 8} y={H - 6} textAnchor="end" fontSize={10} fill="var(--muted)">minute</text>
    </svg>
  );
}

function RuleRow({ id, config, ev, pages, animKey }: { id: RuleId; config: RuleConfig; ev?: WindowEval["rules"][RuleId]; pages: number; animKey: string }) {
  const state: AlertState = ev?.state ?? "ok";
  const idx = ALERT_STATES.indexOf(state);
  const paged = !!ev && ev.state === "firing" && ev.prev !== "firing";
  return (
    <div className="grid items-center gap-x-3 gap-y-1 sm:grid-cols-[210px_minmax(0,1fr)_120px]">
      <div className="min-w-0 text-xs">
        <div className="font-semibold">{RULE_NAMES[id]}</div>
        <div className="truncate font-mono text-[10px] text-muted" title={ruleText(id, config)}>{ruleText(id, config)}</div>
      </div>
      <div className="relative grid grid-cols-4 rounded-md border border-line bg-surface-2 text-[11px]" role="img" aria-label={`${RULE_NAMES[id]} rule state: ${state}`}>
        <motion.div className="absolute inset-y-0 w-1/4 rounded-md border" initial={false}
          animate={{ left: `${idx * 25}%`, backgroundColor: STATE_UI[state].bg, borderColor: STATE_UI[state].fg }} transition={{ type: "spring", stiffness: 260, damping: 26 }} />
        {ALERT_STATES.map((st, i) => {
          const { Icon, label } = STATE_UI[st];
          return (
            <div key={st} className={cx("relative flex items-center justify-center gap-1 py-1.5", i === idx ? "font-semibold" : "text-muted")} style={i === idx ? { color: STATE_UI[st].fg } : undefined}>
              <Icon size={12} aria-hidden /> <span className="hidden min-[420px]:inline">{label}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-1.5 text-[11px]">
        {ev && ev.forN > 1 && <span className="font-mono text-muted" title="consecutive breaching windows / windows required">{Math.min(ev.streak, ev.forN)}/{ev.forN}</span>}
        <span className="text-muted">pages {pages}</span>
        <AnimatePresence>
          {paged && (
            <motion.span key={animKey} initial={{ scale: 0, opacity: 0 }} animate={{ scale: [0, 1.3, 1], opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>
              <Badge tone="err"><BellRing size={10} aria-hidden />PAGE</Badge>
            </motion.span>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function Budget({ ev, total }: { ev?: WindowEval; total: number }) {
  const left = ev ? ev.budgetLeft : 1;
  const tone = left > 0.5 ? "var(--ok)" : left > 0.2 ? "var(--warn)" : "var(--err)";
  return (
    <div className="grid items-center gap-x-3 gap-y-1 pt-1 sm:grid-cols-[210px_minmax(0,1fr)_120px]">
      <div className="text-xs">
        <div className="font-semibold">Error budget</div>
        <div className="text-[10px] text-muted">{Math.round(total)} failed requests allowed this hour</div>
      </div>
      <div className="relative h-5 overflow-hidden rounded-md bg-surface-2" role="img" aria-label={`Error budget left ${Math.max(0, Math.round(left * 100))}%`}>
        <motion.div className="absolute inset-y-0 left-0 rounded-md" initial={false} animate={{ width: `${Math.max(0, left) * 100}%`, backgroundColor: tone }} transition={{ duration: 0.45, ease }} />
        <span className="absolute inset-0 flex items-center px-2 font-mono text-[11px] text-fg">used {ev?.budgetUsed ?? 0} / {Math.round(total)}</span>
      </div>
      <div className="font-mono text-xs">{left <= 0 ? <span className="text-err">exhausted</span> : <><AnimatedNumber value={left * 100} digits={0} />% left</>}</div>
    </div>
  );
}

function StateMachineIntro({ config, rules, k }: { config: RuleConfig; rules: RuleId[]; k: number }) {
  const arrows = ["condition true", `held for N windows`, "condition clears", "next quiet window"];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        {ALERT_STATES.map((st, i) => {
          const { Icon, label, bg, fg } = STATE_UI[st];
          return (
            <motion.div key={st} className="flex items-center gap-1.5" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.5) * k, duration: 0.4 * k, ease }}>
              <span className="flex items-center gap-1 rounded-md border px-2 py-1 font-semibold" style={{ background: bg, color: fg, borderColor: fg }}><Icon size={12} aria-hidden />{label}</span>
              <span className="text-[10px] text-muted">— {arrows[i]} →</span>
            </motion.div>
          );
        })}
      </div>
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">Rule</th><th>Fires when</th><th>Kind</th></tr></thead>
        <tbody>
          {rules.map((id, i) => (
            <motion.tr key={id} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (2.2 + i * 0.25) * k }}>
              <td className="py-1 font-medium">{RULE_NAMES[id]}</td>
              <td className="font-mono text-[11px]">{ruleText(id, config)}</td>
              <td className="text-muted">{id === "errors" && config.errorMode === "burn" ? "SLO burn rate" : (id === "latency" ? config.latencyFor : id === "quality" ? config.qualityFor : 1) > 1 ? "threshold-for-duration" : "static threshold"}</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      {!config.qualityOn && <p className="flex items-center gap-1 text-xs text-warn"><TriangleAlert size={12} aria-hidden /> Nothing watches answer quality. A bad prompt or model change will not page anyone.</p>}
    </div>
  );
}

function WindowTable({ rows, rules, config }: { rows: WindowEval[]; rules: RuleId[]; config: RuleConfig }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[480px] text-xs">
        <thead className="text-left text-muted">
          <tr><th className="py-1">min</th><th>p95 ms</th><th>err %</th><th>burn</th><th>quality</th>{rules.map((id) => <th key={id}>{RULE_NAMES[id]}</th>)}</tr>
        </thead>
        <tbody className="font-mono">
          <AnimatePresence initial={false}>
            {rows.map((e) => (
              <motion.tr key={e.w.t} className="border-t border-line" initial={{ opacity: 0, x: -14, backgroundColor: "var(--accent-soft)" }} animate={{ opacity: 1, x: 0, backgroundColor: "rgba(0,0,0,0)" }} transition={{ duration: 0.5 }}>
                <td className="py-1">{e.w.t}</td>
                <td className={cx(e.rules.latency?.breach && "text-err")}>{e.w.latency}</td>
                <td className={cx(e.rules.errors?.breach && config.errorMode === "static" && "text-err")}>{(e.w.errorRate * 100).toFixed(1)}</td>
                <td className={cx(e.rules.errors?.breach && config.errorMode === "burn" && "text-err")}>{e.burn.rate.toFixed(1)}×</td>
                <td className={cx(e.rules.quality?.breach && "text-err", !e.rules.quality && e.w.quality < 0.75 && "text-warn")}>{e.w.quality.toFixed(2)}</td>
                {rules.map((id) => {
                  const r = e.rules[id]!;
                  const { Icon, label, fg } = STATE_UI[r.state];
                  return <td key={id}><span className="inline-flex items-center gap-1 font-sans" style={{ color: fg }}><Icon size={11} aria-hidden />{label}{r.state === "firing" && r.prev !== "firing" ? " · page" : ""}</span></td>;
                })}
              </motion.tr>
            ))}
          </AnimatePresence>
        </tbody>
      </table>
      {rows.length === 0 && <p className="py-2 text-xs text-muted">Windows arrive one per tick…</p>}
      <p className="mt-2 text-[11px] text-muted">Red values breach their rule. A breach only pages once the rule has held for its required number of consecutive windows.</p>
    </div>
  );
}

function Summary({ scenario, spikes, config, k }: { scenario: ScenarioId; spikes: boolean; config: RuleConfig; k: number }) {
  const ws = generateMetrics(scenario, 7, spikes);
  const rows = [
    { name: "Your rules", s: summarize(evaluate(ws, config), scenario) },
    { name: "Noisy preset", s: summarize(evaluate(ws, PRESETS.noisy), scenario) },
    { name: "Useful preset", s: summarize(evaluate(ws, PRESETS.useful), scenario) },
  ];
  const max = Math.max(1, ...rows.map((r) => r.s.pages));
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold">Pages in this hour: real incident vs noise</h4>
      {rows.map((r, i) => (
        <div key={r.name} className="grid grid-cols-[96px_minmax(0,1fr)_180px] items-center gap-2 text-xs">
          <span className="font-medium">{r.name}</span>
          <div className="flex h-5 overflow-hidden rounded bg-surface-2">
            <motion.div className="h-full bg-ok/70" initial={{ width: 0 }} animate={{ width: `${(r.s.useful / max) * 100}%` }} transition={{ delay: (0.2 + i * 0.4) * k, duration: 0.6 * k, ease }} />
            <motion.div className="h-full bg-err/60" initial={{ width: 0 }} animate={{ width: `${(r.s.noise / max) * 100}%` }} transition={{ delay: (0.5 + i * 0.4) * k, duration: 0.6 * k, ease }} />
          </div>
          <span className="font-mono text-[11px]">{r.s.useful} real · {r.s.noise} noise · {r.s.incident ? (r.s.minutesToDetect !== undefined ? `TTD ${r.s.minutesToDetect}m` : "missed") : "no incident"}</span>
        </div>
      ))}
      <p className="text-[11px] text-muted">Green = pages during the incident, red = pages with nothing actionable. TTD = minutes from incident start to first page.</p>
    </div>
  );
}

function RuleMath({ ev, config }: { ev?: WindowEval; config: RuleConfig }) {
  const allowed = (100 - config.slo) / 100;
  if (!ev) return <TexBlock>{`\\text{budget} = (1 - \\text{SLO}) \\times N = ${allowed.toFixed(3)} \\times ${MINUTES * 1000} = ${Math.round(allowed * MINUTES * 1000)}`}</TexBlock>;
  return (
    <div className="space-y-2">
      <TexBlock>{`\\text{burn}_{${config.burnWindow}m} = \\frac{${ev.burn.errors} / ${ev.burn.requests}}{1 - ${config.slo / 100}} = \\frac{${(ev.burn.errors / Math.max(1, ev.burn.requests)).toFixed(4)}}{${allowed.toFixed(3)}} = ${ev.burn.rate.toFixed(2)}`}</TexBlock>
      <TexBlock>{`\\text{budget left} = 1 - \\frac{${ev.budgetUsed}}{${Math.round(ev.budgetTotal)}} = ${(ev.budgetLeft * 100).toFixed(1)}\\%`}</TexBlock>
      <p className="text-[11px] text-muted">Burn rate 1 spends the budget exactly over the period; {config.burnFactor}× would exhaust it in 1/{config.burnFactor} of the time.</p>
    </div>
  );
}
