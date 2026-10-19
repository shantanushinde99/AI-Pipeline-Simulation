import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Ban, CheckCircle2, CircleSlash, Repeat, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, Meter, StepCaption } from "./kit";
import {
  DEFAULT_LIMITS, OUTPUT_TOKENS, PRESETS, PRICE_PER_1K, STOP_TEXT, SYSTEM_TOKENS, runAgent, sanitize, simulate,
  type Iteration, type Limits, type PresetId, type StopReason, type TermState,
} from "./simulation";

/** Schedule (seconds at 1×): the action chip lands, then meters fill, then the checks tick one by one. */
const T = { chip: 0.15, meters: 0.7, meterDur: 0.9, checks: 1.5, checkGap: 0.22 };
const stepMs = (i: number, total: number) => (i === 0 ? 2200 : i === total - 1 ? 2800 : 3400);

const CONDITIONS: { id: StopReason; label: string }[] = [
  { id: "final_answer", label: "Final answer produced?" },
  { id: "cancelled", label: "User pressed cancel?" },
  { id: "no_progress", label: "Same action repeated too often?" },
  { id: "token_budget", label: "Token budget used up?" },
  { id: "wall_time", label: "Wall time over the limit?" },
  { id: "max_iterations", label: "Iteration limit reached?" },
];

export default function AgentTerminationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [preset, setPreset] = useState<PresetId>("healthy");
  const [limits, setLimits] = useState<Limits>(DEFAULT_LIMITS);
  const [replay, setReplay] = useState(0);
  const l = sanitize(limits);
  const { run, steps } = useMemo(() => simulate(preset, limits), [preset, limits]);
  const pb = usePlayback(steps.length, (i) => stepMs(i, steps.length));
  const step = steps[pb.index];
  const s = step.state;
  const prev = pb.index > 0 ? steps[pb.index - 1].state : s;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<Limits>) => setLimits((x) => ({ ...x, ...p }));
  const comparison = useMemo(() => (Object.keys(PRESETS) as PresetId[]).map((p) => ({ p, r: runAgent(p, limits) })), [limits]);
  const current = s.iterations.at(-1);
  const isIter = step.id.startsWith("it");
  const canCancel = !s.stop && s.usage.iterations > 0;

  const cancelNow = () => {
    const at = s.usage.iterations;
    set({ cancelAfter: at });
    // The stop step of the shortened run comes right after this iteration.
    pb.seek(at + 1);
  };

  if (tab === "metrics") return null;
  const story = narrate(step.id, s, run.stop, current);

  return (
    <Workspace
      main={
        <>
          <Panel title="Agent run & stop conditions" actions={<SimulatedLabel>Scripted agent, not an LLM</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <ActionHistory iterations={s.iterations} isNew={isIter} k={k} animKey={animKey} window={l.noProgressWindow} />
              <div className="space-y-3">
                <Meter label="Iterations" valueText={String(s.usage.iterations)} limitText={String(l.maxIterations)} from={prev.usage.iterations / l.maxIterations} to={s.usage.iterations / l.maxIterations}
                  tripped={s.stop === "max_iterations"} delay={T.meters * k} duration={T.meterDur * k} key={`i-${animKey}`} />
                <Meter label={`Tokens ($${((s.usage.tokens / 1000) * PRICE_PER_1K).toFixed(3)})`} valueText={<AnimatedNumber key={animKey} value={s.usage.tokens} from={prev.usage.tokens} digits={0} delay={T.meters * k} duration={T.meterDur * k} />} limitText={l.maxTokens.toLocaleString("en-US")}
                  from={prev.usage.tokens / l.maxTokens} to={s.usage.tokens / l.maxTokens} tripped={s.stop === "token_budget"} delay={T.meters * k} duration={T.meterDur * k} key={`t-${animKey}`} />
                <Meter label="Wall time" valueText={<><AnimatedNumber key={animKey} value={s.usage.ms / 1000} from={prev.usage.ms / 1000} digits={1} delay={T.meters * k} duration={T.meterDur * k} /> s</>} limitText={`${l.maxWallMs / 1000} s`} from={prev.usage.ms / l.maxWallMs} to={s.usage.ms / l.maxWallMs}
                  tripped={s.stop === "wall_time"} delay={T.meters * k} duration={T.meterDur * k} key={`w-${animKey}`} />
                {l.noProgressWindow > 1 ? (
                  <Meter label="No-progress detector (repeats of this action)" valueText={String(s.usage.repeats)} limitText={String(l.noProgressWindow)}
                    from={prev.usage.repeats / l.noProgressWindow} to={s.usage.repeats / l.noProgressWindow} tripped={s.stop === "no_progress"} delay={T.meters * k} duration={T.meterDur * k} key={`r-${animKey}`} />
                ) : <p className="rounded bg-surface-2 px-2 py-1.5 text-xs text-muted"><CircleSlash size={12} className="mr-1 inline" />No-progress detector off: repeats are not counted.</p>}
                <div className="flex flex-wrap gap-2 text-xs">
                  <Light on={s.stop === "final_answer"} label="final answer" good />
                  <Light on={s.stop === "cancelled"} label="user cancel" />
                  {l.cancelAfter !== null && !s.stop && <Badge tone="warn">cancel queued after iteration {l.cancelAfter}</Badge>}
                </div>
              </div>
            </div>
            <AnimatePresence>
              {s.stop && (
                <motion.div key={`stop-${animKey}`} initial={{ opacity: 0, y: 14, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0 }} transition={{ delay: 0.3 * k, type: "spring", stiffness: 220, damping: 20 }}
                  className={cx("mt-4 flex items-start gap-2 rounded-lg border p-3", STOP_TEXT[s.stop].ok ? "border-ok/40 bg-ok-soft" : "border-err/40 bg-err-soft")} data-testid="final-status">
                  {STOP_TEXT[s.stop].ok ? <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-ok" /> : <XCircle size={18} className="mt-0.5 shrink-0 text-err" />}
                  <div className="text-sm">
                    <div className="font-semibold">Run status: {STOP_TEXT[s.stop].status}</div>
                    <div className="text-muted">{STOP_TEXT[s.stop].message} {s.usage.iterations} iterations · {s.usage.tokens.toLocaleString("en-US")} tokens (${((s.usage.tokens / 1000) * PRICE_PER_1K).toFixed(3)}) · {(s.usage.ms / 1000).toFixed(1)} s.</div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              {step.id === "start" ? <LimitsCard l={l} k={k} /> : current && <IterationCard it={current} stop={isIter ? (step.id === `it${run.iterations.length}` ? run.stop : null) : run.stop} l={l} k={k} final={!isIter} />}
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="primary" className="bg-err" onClick={cancelNow} disabled={!canCancel} title="Simulate the user pressing Stop after the current iteration"><Ban size={14} /> Cancel run now</Button>
            {l.cancelAfter !== null && <Button size="sm" onClick={() => set({ cancelAfter: null })}>Clear cancel</Button>}
            <span className="text-xs text-muted">{canCancel ? `Stops the run after iteration ${s.usage.iterations}; the run is recomputed.` : "Step to an iteration to cancel there."}</span>
          </div>
          <Panel title="Preset & limits" actions={<Button size="sm" variant="ghost" onClick={() => { setLimits(DEFAULT_LIMITS); setPreset("healthy"); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="Preset" value={preset} onChange={(v: PresetId) => { setPreset(v); set({ cancelAfter: null }); }} options={(Object.keys(PRESETS) as PresetId[]).map((p) => ({ value: p, label: PRESETS[p].label }))} />
              <Slider label="Max iterations" value={limits.maxIterations} min={2} max={15} onChange={(v) => set({ maxIterations: v })} />
              <Slider label="Token budget" value={limits.maxTokens} min={5000} max={60000} step={1000} format={(v) => `${v.toLocaleString("en-US")} ($${((v / 1000) * PRICE_PER_1K).toFixed(2)})`} onChange={(v) => set({ maxTokens: v })} />
              <Slider label="Wall-time limit" value={limits.maxWallMs} min={5000} max={90000} step={1000} format={(v) => `${v / 1000} s`} onChange={(v) => set({ maxWallMs: v })} />
              <Slider label="No-progress window" value={Math.max(2, limits.noProgressWindow)} min={2} max={5} onChange={(v) => set({ noProgressWindow: v })} format={(v) => (limits.noProgressWindow > 1 ? `${v} repeats` : "off")} />
              <div className="flex items-end"><Toggle label="Break it: disable the no-progress detector" checked={limits.noProgressWindow <= 1} onChange={(v) => set({ noProgressWindow: v ? 0 : 3 })} /></div>
            </div>
            <p className="mt-3 text-xs text-muted">{PRESETS[preset].note}</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={isIter ? "Every LLM call re-sends the whole history, so token cost grows faster than the iteration count." : s.stop ? "A run that ends must say why: callers handle 'completed', 'cancelled' and 'budget exceeded' differently." : undefined} />
          <Panel title="Calculation">
            {current ? (
              <TexBlock>{`\\begin{aligned} \\text{tokens}_{${current.n}} &= \\text{system} + \\textstyle\\sum \\text{obs} + \\text{output} \\\\ &= ${SYSTEM_TOKENS} + ${current.tokens - SYSTEM_TOKENS - OUTPUT_TOKENS} + ${OUTPUT_TOKENS} = ${current.tokens} \\\\ \\text{cost} &= \\tfrac{${current.cumTokens}}{1000} \\times \\$${PRICE_PER_1K} = \\$${((current.cumTokens / 1000) * PRICE_PER_1K).toFixed(4)} \\end{aligned}`}</TexBlock>
            ) : <TexBlock>{`\\begin{aligned} \\text{stop} = {} & \\text{final} \\lor \\text{cancel} \\lor \\text{repeats} \\ge ${l.noProgressWindow > 1 ? l.noProgressWindow : "\\infty"} \\\\ & \\lor\\ \\text{tokens} \\ge ${l.maxTokens} \\lor t \\ge ${l.maxWallMs / 1000}\\,\\text{s} \\lor n \\ge ${l.maxIterations} \\end{aligned}`}</TexBlock>}
          </Panel>
          <Panel title="Which condition fires first? (current limits)">
            <table className="w-full text-xs">
              <thead className="text-left text-muted"><tr><th className="py-1">preset</th><th>fires</th><th className="text-right">after</th></tr></thead>
              <tbody>
                {comparison.map(({ p, r }) => (
                  <tr key={p} className={cx("border-t border-line", p === preset && "bg-accent-soft")}>
                    <td className="py-1">{PRESETS[p].label}</td>
                    <td><StatusBadge status={STOP_TEXT[r.stop].ok ? "ok" : r.stop === "cancelled" ? "warning" : "error"} label={STOP_TEXT[r.stop].status.replace("stopped: ", "")} /></td>
                    <td className="text-right font-mono">it {r.iterations.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
          <Panel title="Usage so far">
            <KV rows={[
              ["iterations", `${s.usage.iterations} / ${l.maxIterations}`],
              ["tokens", `${s.usage.tokens.toLocaleString("en-US")} / ${l.maxTokens.toLocaleString("en-US")}`],
              ["cost", `$${((s.usage.tokens / 1000) * PRICE_PER_1K).toFixed(4)}`],
              ["wall time", `${(s.usage.ms / 1000).toFixed(1)} s / ${l.maxWallMs / 1000} s`],
              ["status", s.stop ? STOP_TEXT[s.stop].status : "running"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(id: string, s: TermState, stop: StopReason, it?: Iteration) {
  if (id === "start") return { headline: "Every budget starts empty", sub: "Four meters and two signals. The first condition to trip ends the run." };
  if (s.stop) return { headline: `The run ends: ${STOP_TEXT[s.stop].status}`, sub: STOP_TEXT[s.stop].message };
  const last = id === `it${s.iterations.length}` && stop !== "final_answer" && it?.n === s.usage.iterations;
  return {
    headline: `Iteration ${it!.n}: ${it!.action.tool}("${it!.action.args}")`,
    sub: `${it!.repeats > 1 ? `Identical to an earlier action (repeat #${it!.repeats}). ` : ""}+${it!.tokens.toLocaleString("en-US")} tokens, +${(it!.ms / 1000).toFixed(1)} s.${last ? " Watch the checks: one of them is about to trip." : ""}`,
  };
}

function Light({ on, label, good }: { on: boolean; label: string; good?: boolean }) {
  return (
    <motion.span initial={false} animate={{ scale: on ? [1, 1.15, 1] : 1 }} transition={{ duration: 0.4 }}
      className={cx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5", on ? (good ? "border-ok bg-ok-soft text-ok" : "border-err bg-err-soft text-err") : "border-line text-muted")}>
      {on ? <CheckCircle2 size={12} /> : <span className="inline-block h-2 w-2 rounded-full bg-line" />}{label}{on ? " (fired)" : ""}
    </motion.span>
  );
}

function ActionHistory({ iterations, isNew, k, animKey, window }: { iterations: Iteration[]; isNew: boolean; k: number; animKey: string; window: number }) {
  const shown = iterations.slice(-7);
  const hidden = iterations.length - shown.length;
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium">Action history</div>
      {iterations.length === 0 && <div className="rounded-md border border-dashed border-line p-3 text-xs text-muted">No actions yet. The agent is about to make its first decision.</div>}
      {hidden > 0 && <div className="mb-1 text-[11px] text-muted">… {hidden} earlier iteration{hidden > 1 ? "s" : ""}</div>}
      <ol className="space-y-1">
        {shown.map((it, i) => {
          const fresh = isNew && i === shown.length - 1;
          const warn = window > 1 && it.repeats > 1;
          return (
            <motion.li key={fresh ? `${it.n}-${animKey}` : it.n} initial={fresh ? { opacity: 0, x: -60 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: T.chip * k, type: "spring", stiffness: 200, damping: 20 }}
              className={cx("flex items-center gap-2 rounded-md border px-2 py-1 text-xs", fresh ? "border-accent bg-accent-soft" : "border-line", it.action.final && "border-ok bg-ok-soft")}>
              <span className="w-5 shrink-0 font-mono text-muted">{it.n}</span>
              <span className="min-w-0 flex-1 truncate font-mono">{it.action.tool}(<span className="text-muted">{it.action.args}</span>)</span>
              {it.repeats > 1 && <Badge tone={warn ? "warn" : "neutral"}><Repeat size={10} />×{it.repeats}</Badge>}
              <span className="shrink-0 font-mono text-[10px] text-muted">{(it.tokens / 1000).toFixed(1)}k · {(it.ms / 1000).toFixed(1)}s</span>
            </motion.li>
          );
        })}
      </ol>
    </div>
  );
}

function LimitsCard({ l, k }: { l: Limits; k: number }) {
  const items = [
    ["Iterations", `${l.maxIterations}`], ["Tokens", `${l.maxTokens.toLocaleString("en-US")} ($${((l.maxTokens / 1000) * PRICE_PER_1K).toFixed(2)})`],
    ["Wall time", `${l.maxWallMs / 1000} s`], ["No progress", l.noProgressWindow > 1 ? `${l.noProgressWindow} identical actions` : "off"],
    ["Final answer", "stop with success"], ["User cancel", l.cancelAfter !== null ? `after iteration ${l.cancelAfter}` : "button below"],
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {items.map(([a, b], i) => (
        <motion.div key={a} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.15 + i * 0.18) * k, duration: 0.35 * k, ease: EASE }} className="rounded-md border border-line bg-surface-2 p-2">
          <div className="text-[11px] text-muted">{a}</div>
          <div className="font-mono text-xs">{b}</div>
        </motion.div>
      ))}
    </div>
  );
}

function IterationCard({ it, stop, l, k, final }: { it: Iteration; stop: StopReason | null; l: Limits; k: number; final: boolean }) {
  const history = it.tokens - SYSTEM_TOKENS - OUTPUT_TOKENS;
  const parts = [
    { label: "system prompt", v: SYSTEM_TOKENS, cls: "bg-violet/60" },
    { label: "history (all earlier observations)", v: history, cls: "bg-warn/70" },
    { label: "model output", v: OUTPUT_TOKENS, cls: "bg-accent/70" },
  ];
  const firedIdx = stop ? CONDITIONS.findIndex((c) => c.id === stop) : -1;
  const value: Record<StopReason, string> = {
    final_answer: it.action.final ? "yes" : "no",
    cancelled: l.cancelAfter !== null && it.n >= l.cancelAfter ? "yes" : "no",
    no_progress: l.noProgressWindow > 1 ? `${it.repeats} / ${l.noProgressWindow}` : "detector off",
    token_budget: `${it.cumTokens.toLocaleString("en-US")} / ${l.maxTokens.toLocaleString("en-US")}`,
    wall_time: `${(it.cumMs / 1000).toFixed(1)} / ${l.maxWallMs / 1000} s`,
    max_iterations: `${it.n} / ${l.maxIterations}`,
  };
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-2">
        <h4 className="text-sm font-semibold">Where iteration {it.n}'s {it.tokens.toLocaleString("en-US")} tokens go</h4>
        <div className="flex h-6 overflow-hidden rounded bg-surface-2">
          {parts.map((p, i) => (
            <motion.div key={p.label} className={p.cls} initial={{ width: 0 }} animate={{ width: `${(p.v / it.tokens) * 100}%` }} transition={{ delay: (0.2 + i * 0.35) * k, duration: 0.45 * k, ease: EASE }} />
          ))}
        </div>
        <ul className="space-y-0.5 text-xs">
          {parts.map((p) => <li key={p.label} className="flex items-center gap-1.5"><span className={cx("inline-block h-2 w-3 rounded-sm", p.cls)} />{p.label}<span className="ml-auto font-mono">{p.v.toLocaleString("en-US")}</span></li>)}
        </ul>
        <p className="text-xs text-muted">Time: 1.2 s thinking (LLM) + {(it.action.toolMs / 1000).toFixed(1)} s in {it.action.tool}.</p>
      </div>
      <div>
        <h4 className="mb-1.5 text-sm font-semibold">{final ? "The check that fired" : "Checks after this iteration (in order)"}</h4>
        <ol className="space-y-1">
          {CONDITIONS.map((c, i) => {
            const skipped = firedIdx >= 0 && i > firedIdx;
            const fired = i === firedIdx;
            return (
              <motion.li key={c.id} initial={{ opacity: 0, x: 10 }} animate={{ opacity: skipped ? 0.4 : 1, x: 0 }} transition={{ delay: (final ? 0.1 : T.checks + i * T.checkGap) * k, duration: 0.25 * k }}
                className={cx("flex items-center gap-2 rounded px-2 py-1 text-xs", fired ? (stop === "final_answer" ? "bg-ok-soft" : "bg-err-soft") : "bg-surface-2")}>
                {fired ? (stop === "final_answer" ? <CheckCircle2 size={13} className="text-ok" /> : <XCircle size={13} className="text-err" />) : <span className="inline-block w-[13px] text-center text-muted">{skipped ? "–" : "✓"}</span>}
                <span className="flex-1">{c.label}</span>
                <span className="font-mono text-[11px]">{skipped ? "not checked" : value[c.id]}</span>
                <span className={cx("w-12 text-right font-semibold", fired ? "text-err" : "text-muted")}>{fired ? "STOP" : skipped ? "" : "go on"}</span>
              </motion.li>
            );
          })}
        </ol>
        {!stop && <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.checks + CONDITIONS.length * T.checkGap) * k }} className="mt-1.5 text-xs text-muted">Nothing tripped: the loop continues.</motion.p>}
        {stop && stop !== "final_answer" && final && <Callout tone="warn" title="Return a clear status">Callers (and users) need to know the answer is missing because a limit fired, plus any partial results. Silent failure looks like success.</Callout>}
      </div>
    </div>
  );
}
