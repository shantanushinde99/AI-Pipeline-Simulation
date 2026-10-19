import { useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { TraceInspector } from "../../components/visualizer/TraceInspector";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Badge, Button, Callout, Formula, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle, cx } from "../../components/ui";
import type { VisualizerTab } from "../types";
import { DEFAULT_AGENT, TASKS, TOOL_SCHEMAS, runAgent, type AgentConfig, type AgentFaults, type AgentState, type AgentTask, type ToolName } from "./simulation";

export const AGENT_FAULTS = {
  none: { label: "No fault", faults: {}, task: undefined, note: "" },
  wrongTool: { label: "Wrong tool selected first", faults: { wrongTool: true }, task: "quote", note: "The planner calls the calculator before it knows the price. Watch the error feed the next decision." },
  staleToolSpec: { label: "Invalid tool arguments (stale tool spec)", faults: { staleToolSpec: true }, task: "quote", note: "The agent sends {expr} but the calculator requires {expression}." },
  retryLoop: { label: "Retries create a loop", faults: { staleDocId: true, retryAllErrors: true }, task: "quote", note: "Search returns a deleted doc id and the retry policy retries 404s." },
  staleDocHandled: { label: "Stale doc id, correct retry policy", faults: { staleDocId: true }, task: "quote", note: "Same stale id, but 404s are surfaced to the planner instead of retried." },
  timeout: { label: "Tool timeout", faults: { slowTool: true }, task: "quote", note: "get_document is slow on a cold cache. Try changing retries or the timeout." },
  injection: { label: "Prompt injection → unauthorized action", faults: { trustToolOutput: true }, task: "summarize", note: "The document contains an instruction to delete data. Toggle guardrails to see what they prevent." },
} satisfies Record<string, { label: string; faults: AgentFaults; task?: AgentTask; note: string }>;
type FaultId = keyof typeof AGENT_FAULTS;

const PHASES = ["Decide", "Select tool", "Build args", "Validate", "Execute", "Observe"] as const;
const activePhases = (s: AgentState): string[] =>
  s.phase === "decide" ? ["Decide", "Select tool", "Build args"] : s.phase === "call" ? (s.call?.validation ? ["Validate"] : ["Validate", "Execute"]) : s.phase === "observe" ? ["Observe"] : ["Final"];

const finalTone = { success: "ok", failed: "error", budget_exceeded: "error", blocked: "error" } as const;

export default function AgentVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<Omit<AgentConfig, "faults">>({ task: DEFAULT_AGENT.task, maxIterations: 8, maxRetries: 2, timeoutMs: 2000, guardrails: true });
  const [fault, setFault] = useState<FaultId>("none");
  const full: AgentConfig = { ...cfg, faults: AGENT_FAULTS[fault].faults };
  const run = useMemo(() => runAgent({ ...cfg, faults: AGENT_FAULTS[fault].faults }), [cfg, fault]);
  const pb = usePlayback(run.steps.length, 1200);
  const step = run.steps[pb.index];
  const s = step.state;
  const set = (p: Partial<typeof cfg>) => setCfg((c) => ({ ...c, ...p }));
  const chooseFault = (f: FaultId) => {
    setFault(f);
    const task = AGENT_FAULTS[f].task;
    if (task) set({ task });
  };

  if (tab === "metrics") {
    const errors = run.toolCalls.filter((c) => !c.ok).length;
    return (
      <div className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Panel title="Final status"><StatusBadge status={finalTone[run.final.status]} label={run.final.status} /></Panel>
          <Panel title="Task correct"><StatusBadge status={run.correct ? "ok" : "error"} label={run.correct ? "yes" : "no"} /></Panel>
          <Panel title="Iterations"><div className="font-mono text-xl">{run.iterations} / {full.maxIterations}</div></Panel>
          <Panel title="Tool calls"><div className="font-mono text-xl">{run.toolCalls.length}</div></Panel>
          <Panel title="Failed calls"><div className="font-mono text-xl">{errors}</div></Panel>
          <Panel title="Wall time (sim)"><div className="font-mono text-xl">{(run.spans[0].durationMs / 1000).toFixed(1)} s</div></Panel>
        </div>
        <Panel title="Execution trace"><TraceInspector spans={run.spans} initial="agent.run" /></Panel>
      </div>
    );
  }

  return (
    <Workspace
      main={
        <>
          <Panel title="Agent loop" actions={<SimulatedLabel>Rule-based planner, not an LLM</SimulatedLabel>}>
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <LoopDiagram active={activePhases(s)} iteration={s.iteration} max={full.maxIterations} final={s.final?.status} />
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={step.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
                  <CallCard s={s} />
                </motion.div>
              </AnimatePresence>
            </div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Execution timeline">
            <ol className="space-y-1">
              {run.steps.map((x, i) => {
                const failed = x.state.phase === "call" && x.state.call && !x.state.call.ok;
                return (
                  <li key={x.id}>
                    <button onClick={() => pb.seek(i)} className={cx("flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-surface-2", i === pb.index && "bg-accent-soft", i > pb.index && "opacity-50")}>
                      <span className="w-6 shrink-0 font-mono text-muted">{i + 1}</span>
                      <span className="flex-1 truncate">{x.title}</span>
                      {failed && <StatusBadge status="error" label={x.state.call!.validation ? "invalid args" : x.state.call!.attempts.at(-1)?.outcome ?? "error"} />}
                      {x.state.final && <StatusBadge status={finalTone[x.state.final.status]} label={x.state.final.status} />}
                    </button>
                  </li>
                );
              })}
            </ol>
          </Panel>
          <Panel title="Task, limits & faults" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg({ task: "quote", maxIterations: 8, maxRetries: 2, timeoutMs: 2000, guardrails: true }); setFault("none"); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="Task" value={cfg.task} onChange={(v) => set({ task: v })} options={(Object.keys(TASKS) as AgentTask[]).map((t) => ({ value: t, label: TASKS[t] }))} />
              <Select label="Inject fault" value={fault} onChange={chooseFault} options={(Object.keys(AGENT_FAULTS) as FaultId[]).map((f) => ({ value: f, label: AGENT_FAULTS[f].label }))} />
              <Slider label="Max iterations" value={cfg.maxIterations} min={2} max={20} onChange={(v) => set({ maxIterations: v })} />
              <Slider label="Max retries per call" value={cfg.maxRetries} min={0} max={4} onChange={(v) => set({ maxRetries: v })} />
              <Slider label="Tool timeout" value={cfg.timeoutMs} min={500} max={5000} step={100} format={(v) => `${v} ms`} onChange={(v) => set({ timeoutMs: v })} />
              <div className="flex items-end"><Toggle label="Guardrails: block privileged tools" checked={cfg.guardrails} onChange={(v) => set({ guardrails: v })} hint="delete_records requires human approval" /></div>
            </div>
            {fault !== "none" && <p className="mt-3 text-xs text-muted">{AGENT_FAULTS[fault].note}</p>}
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.phase === "observe" ? "The observation is the only way a tool result can influence the next decision." : s.phase === "call" ? "Validation and retry policy decide whether a bad call fails fast, recovers, or loops." : undefined} />
          {s.decision && (
            <Panel title="Decision">
              <p className="text-xs leading-relaxed">{s.decision.reason}</p>
              <div className="mt-2"><Formula>{`${s.decision.tool}(${JSON.stringify(s.decision.args, null, 1)})`}</Formula></div>
              {TOOL_SCHEMAS[s.decision.tool].privileged && <div className="mt-2"><Badge tone="warn">privileged tool</Badge></div>}
            </Panel>
          )}
          <Panel title="Working memory">
            {Object.keys(s.memory).length === 0 ? <p className="text-xs text-muted">Empty: nothing learned yet.</p> : (
              <KV rows={Object.entries(s.memory).filter(([, v]) => v !== undefined).map(([k, v]) => [k, typeof v === "string" && v.length > 60 ? `${v.slice(0, 60)}…` : JSON.stringify(v)])} />
            )}
          </Panel>
          {s.final && (
            <Panel title="Final response">
              <StatusBadge status={finalTone[s.final.status]} label={s.final.status} />
              <p className="mt-2 text-sm leading-relaxed" data-testid="agent-final">{s.final.text}</p>
              {cfg.task === "quote" && s.final.status === "success" && <p className="mt-1 text-xs text-muted">Expected: $5,997.6 (12 × $49 × 12 × 0.85). {run.correct ? "Matches." : "Does not match."}</p>}
            </Panel>
          )}
          <Panel title="Available tools">
            <ul className="space-y-1 text-xs">{(Object.keys(TOOL_SCHEMAS) as ToolName[]).map((t) => (
              <li key={t}><span className="font-mono">{t}</span>({TOOL_SCHEMAS[t].required.join(", ")}){TOOL_SCHEMAS[t].privileged && <Badge tone="warn" className="ml-1">privileged</Badge>}<span className="block text-muted">{TOOL_SCHEMAS[t].description}</span></li>
            ))}</ul>
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function LoopDiagram({ active, iteration, max, final }: { active: string[]; iteration: number; max: number; final?: string }) {
  const c = 130, r = 92;
  const pos = PHASES.map((_, i) => { const a = -Math.PI / 2 + (i * 2 * Math.PI) / PHASES.length; return [c + r * Math.cos(a), c + r * Math.sin(a)] as const; });
  const finalOn = active.includes("Final");
  return (
    <svg viewBox="0 0 330 270" className="h-auto w-full" role="img" aria-label={`Agent loop, iteration ${iteration} of ${max}, active: ${active.join(", ")}`}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="var(--line)" strokeWidth={2} strokeDasharray="4 4" />
      <text x={c} y={c - 4} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--fg)">iteration {iteration}</text>
      <text x={c} y={c + 12} textAnchor="middle" fontSize={10} fill="var(--muted)">budget {max}</text>
      {PHASES.map((p, i) => {
        const on = active.includes(p);
        return (
          <g key={p}>
            <motion.rect x={pos[i][0] - 40} y={pos[i][1] - 13} width={80} height={26} rx={13} initial={false} animate={{ fill: on ? "var(--accent)" : "var(--surface)", stroke: on ? "var(--accent)" : "var(--line)" }} transition={{ duration: 0.35 }} />
            <text x={pos[i][0]} y={pos[i][1] + 4} textAnchor="middle" fontSize={11} fontWeight={on ? 600 : 400} fill={on ? "var(--surface)" : "var(--fg)"}>{p}</text>
          </g>
        );
      })}
      <line x1={c + 40} y1={pos[0][1]} x2={262} y2={pos[0][1]} stroke="var(--muted)" strokeDasharray="3 3" />
      <motion.rect x={262} y={pos[0][1] - 15} width={62} height={30} rx={6} initial={false} animate={{ fill: finalOn ? (final === "success" ? "var(--ok)" : "var(--err)") : "var(--surface)" }} stroke={finalOn ? "transparent" : "var(--line)"} />
      {/* A token travels around the loop to whichever phase is running. */}
      <motion.circle r={7} fill="var(--warn)" stroke="var(--surface)" strokeWidth={2} initial={false}
        animate={finalOn ? { cx: 262, cy: pos[0][1] } : { cx: pos[PHASES.indexOf(active[0] as (typeof PHASES)[number])][0] - 40, cy: pos[PHASES.indexOf(active[0] as (typeof PHASES)[number])][1] }}
        transition={{ type: "spring", stiffness: 90, damping: 16 }} />
      <text x={293} y={pos[0][1] + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill={finalOn ? "var(--surface)" : "var(--muted)"}>Final</text>
      {finalOn && <text x={293} y={pos[0][1] + 30} textAnchor="middle" fontSize={10} fill="var(--fg)">{final}</text>}
    </svg>
  );
}

function CallCard({ s }: { s: AgentState }) {
  if (!s.call) return <div className="flex items-center rounded-md border border-dashed border-line p-3 text-xs text-muted">{s.decision ? `About to call ${s.decision.tool}. Step forward to validate and execute it.` : "No tool call at this step."}</div>;
  const c = s.call;
  return (
    <div className="space-y-2 rounded-md border border-line p-3 text-xs">
      <div className="flex items-center gap-2"><span className="font-mono font-semibold">{c.tool}</span><StatusBadge status={c.ok ? "ok" : "error"} label={c.ok ? "succeeded" : c.validation ? "rejected" : "failed"} /></div>
      <div><span className="text-muted">args </span><span className="font-mono">{JSON.stringify(c.args)}</span></div>
      <div><span className="text-muted">schema requires </span><span className="font-mono">{TOOL_SCHEMAS[c.tool].required.join(", ")}</span></div>
      {c.validation ? <Callout tone="err" title="Validation failed">{c.validation}</Callout> : (
        <table className="w-full"><thead className="text-left text-muted"><tr><th>attempt</th><th>outcome</th><th className="text-right">ms</th></tr></thead>
          <tbody>{c.attempts.map((a) => (
            <tr key={a.n} className="border-t border-line"><td className="py-0.5 font-mono">{a.n}</td><td><StatusBadge status={a.outcome === "ok" ? "ok" : a.outcome === "timeout" ? "warning" : "error"} label={a.outcome} /> <span className="text-muted">{a.outcome !== "ok" && a.message}</span></td><td className="text-right font-mono">{a.latencyMs}</td></tr>
          ))}</tbody></table>
      )}
      {c.ok && <div><span className="text-muted">result </span><span className="font-mono break-all">{JSON.stringify(c.result)}</span></div>}
      {s.observation && <div className="border-t border-line pt-2"><span className="text-muted">observation → memory: </span>{s.observation}</div>}
    </div>
  );
}
