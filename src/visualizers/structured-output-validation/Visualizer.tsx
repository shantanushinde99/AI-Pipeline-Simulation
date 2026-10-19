import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, NumberField, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, pct } from "../hallucination-analysis/kit";
import {
  DEFAULT_CONFIG, ERROR_KINDS, KINDS, KIND_LABEL, LADDER, SYNTAX_KINDS, parse, repairSchema, retryPassProb, runAttempt, simulate,
  type Attempt, type Config, type Kind, type SOState, type Visit,
} from "./simulation";

/** One schedule (seconds at 1×) for the packet, the code view, the tree walk and the card. */
const T = { line: 0.1, travel: 0.6, sweep: 0.5, perLine: 0.18, visit0: 0.7, perVisit: 0.35, bar: (i: number) => 0.3 + i * 0.35, barDur: 0.8 };
const visitAt = (i: number) => T.visit0 + i * T.perVisit;

const STAGES = ["Model", "Parse", "Repair", "Validate", "Retry", "App"] as const;
const STAGE_OF: Record<SOState["phase"], number> = { generate: 0, parse: 1, "repair-syntax": 2, validate: 3, "repair-schema": 2, retry: 4, final: 5, batch: 5 };

function duration(s: SOState, run: ReturnType<typeof simulate>) {
  const a = run.attempts[s.attempt] ?? run.attempts[0];
  const lines = a.raw.split("\n").length;
  switch (s.phase) {
    case "generate": return 1200 + lines * T.line * 1000 + 800;
    case "parse": return (T.sweep + lines * T.perLine + 1.4) * 1000;
    case "validate": return (visitAt(a.visits.length) + 1.2) * 1000;
    case "batch": return (T.bar(LADDER.length) + 1.6) * 1000;
    default: return 3200;
  }
}

function narrate(s: SOState, run: ReturnType<typeof simulate>, cfg: Config) {
  const a0 = run.attempts[0];
  switch (s.phase) {
    case "generate": return { headline: "The model returns text that should be JSON", sub: cfg.constrained && SYNTAX_KINDS.includes(cfg.kind) ? "Constrained decoding blocked the syntax error at generation time." : "Nothing guarantees it is well-formed, or that it matches your schema." };
    case "parse": return a0.parsed.ok ? { headline: "Parse: the text is well-formed JSON", sub: "That only proves the brackets and quotes line up. Types and fields are unchecked." } : { headline: `Parse fails on line ${a0.parsed.line + 1}`, sub: a0.parsed.message };
    case "repair-syntax": return { headline: a0.reparsed?.ok ? "Syntax repair makes it parse" : "Syntax repair cannot save it", sub: a0.kind === "truncated" ? "Cut back to the last complete field and close the object. Data after the cut is gone." : "Remove the trailing comma, then parse again." };
    case "validate": return a0.issues.length ? { headline: `The validator walks the tree and flags ${a0.issues.length} node${a0.issues.length > 1 ? "s" : ""}`, sub: a0.issues.map((i) => `${i.path}: ${i.message}`).join("; ") } : { headline: "Every node matches the schema", sub: "Required fields present, types right, enum values allowed, nothing extra." };
    case "repair-schema": return a0.finalIssues.length ? { headline: "Repair can't invent the missing or invalid value", sub: a0.schemaChanges.length ? `It tried: ${a0.schemaChanges.join("; ")}.` : "Coercing types and dropping keys doesn't help here. Time to ask the model again." } : { headline: "Safe repairs fix it", sub: a0.schemaChanges.join("; ") };
    case "retry": { const a = run.attempts[s.attempt]; return { headline: `Retry ${s.attempt}: the model tries again with the error message`, sub: `It returns "${KIND_LABEL[a.kind]}", which ${a.accepted ? "passes" : "still fails"}.` }; }
    case "final": return run.accepted ? { headline: `Accepted after ${run.attempts.length} model call${run.attempts.length > 1 ? "s" : ""}`, sub: "The application receives a typed object it can trust." } : { headline: "Rejected: fail loudly instead of passing bad data on", sub: "Raise an error or fall back. Never hand an invalid object downstream." };
    default: return { headline: "Across many outputs: what each layer buys you", sub: "Parse-only lets wrong data through silently. Each later layer turns rejections into valid objects." };
  }
}

export default function StructuredOutputVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(run.steps.length, (i) => duration(run.steps[i].state, run));
  const step = run.steps[pb.index];
  const s = step.state;
  const prev = run.steps[Math.max(0, pb.index - 1)].state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(cfg)}`;
  const story = narrate(s, run, cfg);
  const set = (p: Partial<Config>) => setCfg((c) => ({ ...c, ...p }));
  const { q, fixable } = retryPassProb(cfg);
  const failedFirst = !run.attempts[0].accepted;
  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Parse → validate → repair → retry" actions={<SimulatedLabel>Simulated model outputs</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[560px]"><Strip s={s} prev={prev} accepted={run.accepted} k={k} animKey={animKey} /></div></div>
            <div className="mt-3">
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={animKey} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                  <Stage s={s} run={run} cfg={cfg} k={k} />
                </motion.div>
              </AnimatePresence>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard s={s} run={run} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Output & pipeline settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_CONFIG); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select<Kind> label="Model output variant" value={cfg.kind} options={KINDS.map((x) => ({ value: x, label: KIND_LABEL[x] }))} onChange={(v) => { set({ kind: v }); pb.restart(); }} />
              <Slider label="Max retries" value={cfg.maxRetries} min={0} max={3} onChange={(v) => set({ maxRetries: v })} />
              <Slider label="Retry returns valid JSON" value={cfg.retryFix} min={0} max={1} step={0.05} format={(v) => pct(v)} onChange={(v) => set({ retryFix: v })} hint="Break it: at 0% retries only burn calls." />
              <Toggle label="Repair (strip trailing commas, coerce types, drop unknown keys)" checked={cfg.repair} onChange={(v) => set({ repair: v })} />
              <Toggle label="Constrained decoding (JSON grammar)" checked={cfg.constrained} onChange={(v) => set({ constrained: v })} hint="Simulated: output is always parseable. Schema errors can still happen." />
              <div />
              <Slider label="Batch: model error rate" value={cfg.errorRate} min={0} max={0.8} step={0.05} format={(v) => pct(v)} onChange={(v) => set({ errorRate: v })} />
              <Slider label="Batch: outputs" value={cfg.n} min={50} max={500} step={50} onChange={(v) => set({ n: v })} />
              <NumberField label="Batch: seed" value={cfg.seed} step={1} min={1} max={9999} onChange={(v) => set({ seed: Math.round(v) })} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Downstream code trusts types. One silently wrong field (a string where a number belongs) breaks later, far from the cause." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`\\begin{aligned} q &= p_{\\text{fix}} + (1 - p_{\\text{fix}})\\cdot f \\\\ &= ${cfg.retryFix.toFixed(2)} + ${(1 - cfg.retryFix).toFixed(2)} \\cdot ${fixable.toFixed(2)} = ${q.toFixed(3)} \\end{aligned}`}</TexBlock>
              <TexBlock>{`\\begin{aligned} P_{\\le r} &= 1 - (1 - q)^{r} \\\\ &= 1 - ${(1 - q).toFixed(3)}^{${cfg.maxRetries}} = ${(1 - (1 - q) ** cfg.maxRetries).toFixed(3)} \\end{aligned}`}</TexBlock>
              <p className="text-[11px] text-muted">P = chance a failed output passes within r retries; q = chance one retry passes; f = share of error kinds that repair fixes. Applies to an output whose first attempt failed{failedFirst ? " (like this one)" : ""}.</p>
            </div>
          </Panel>
          <Panel title="This request">
            <KV rows={[
              ["first output", KIND_LABEL[run.attempts[0].kind]],
              ["model calls", run.attempts.length],
              ["parse", run.attempts[0].parsed.ok ? "ok" : "failed"],
              ["schema issues (first output)", run.attempts[0].issues.length],
              ["result", run.accepted ? "accepted" : "rejected"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Strip({ s, prev, accepted, k, animKey }: { s: SOState; prev: SOState; accepted: boolean; k: number; animKey: string }) {
  const W = 560, x = (i: number) => 40 + i * ((W - 80) / (STAGES.length - 1));
  const now = STAGE_OF[s.phase], from = STAGE_OF[prev.phase];
  const done = s.phase === "final" || s.phase === "batch";
  const blocked = s.phase === "final" && !accepted;
  const tx = blocked ? x(4) + 40 : x(now);
  return (
    <svg viewBox={`0 0 ${W} 82`} className="h-auto w-full" role="img" aria-label="Pipeline stages: model, parse, repair, validate, retry, app">
      <line x1={x(0)} x2={x(STAGES.length - 1)} y1={40} y2={40} stroke="var(--line)" strokeWidth={3} />
      {STAGES.map((name, i) => {
        const active = i === now && s.phase !== "batch";
        return (
          <g key={name}>
            <motion.circle cx={x(i)} cy={40} r={15} initial={false} animate={{ fill: active ? "var(--accent-soft)" : "var(--surface)", stroke: active ? "var(--accent)" : "var(--line)" }} strokeWidth={2} />
            <text x={x(i)} y={44} textAnchor="middle" fontSize={11} fontWeight={600} fill={active ? "var(--accent)" : "var(--muted)"}>{i + 1}</text>
            <text x={x(i)} y={74} textAnchor="middle" fontSize={11} fill={active ? "var(--fg)" : "var(--muted)"} fontWeight={active ? 600 : 400}>{name}</text>
          </g>
        );
      })}
      {s.phase !== "batch" && (
        <motion.g key={`pk-${animKey}`} initial={{ x: x(from), y: 40 }} animate={{ x: tx, y: 40 }} transition={{ duration: T.travel * k, ease }}>
          <rect x={-14} y={-38} width={28} height={16} rx={3} fill={blocked ? "var(--err)" : done ? "var(--ok)" : "var(--violet)"} />
          <text y={-26} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--surface)" fontWeight={700}>{blocked ? "✗" : done ? "✓" : "{ }"}</text>
        </motion.g>
      )}
    </svg>
  );
}

function Code({ raw, k, mode, errLine, highlight }: { raw: string; k: number; mode: "type" | "sweep" | "static"; errLine?: number; highlight?: (line: string) => boolean }) {
  const lines = raw.split("\n");
  const stop = errLine ?? lines.length - 1;
  return (
    <div className="relative overflow-x-auto rounded-md border border-line bg-surface-2 p-2 font-mono text-xs leading-5">
      {mode === "sweep" && (
        <motion.div className="pointer-events-none absolute inset-x-0 h-5 bg-accent/20" initial={{ top: 8 }} animate={{ top: 8 + stop * 20 }} transition={{ delay: T.sweep * k, duration: (stop + 1) * T.perLine * k, ease: "linear" }} />
      )}
      {lines.map((ln, i) => (
        <motion.div key={i} className={cx("relative whitespace-pre", highlight?.(ln) && "bg-warn-soft")} initial={mode === "type" ? { opacity: 0, x: -6 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * T.line * k + 0.2 * k }}>
          <span className="mr-3 inline-block w-4 text-right text-muted select-none">{i + 1}</span>{ln}
          {mode === "sweep" && errLine === i && (
            <motion.span className="ml-2 rounded bg-err px-1 text-[10px] font-semibold text-white dark:text-black" initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (T.sweep + (stop + 1) * T.perLine) * k }}>✗ parse error here</motion.span>
          )}
        </motion.div>
      ))}
      {mode === "sweep" && errLine === undefined && (
        <motion.div className="mt-1 text-[11px] font-semibold text-ok" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.sweep + lines.length * T.perLine) * k }}>✓ parsed: well-formed JSON</motion.div>
      )}
    </div>
  );
}

function Tree({ visits, k, animate }: { visits: Visit[]; k: number; animate: boolean }) {
  return (
    <ul className="space-y-1">
      {visits.map((v, i) => (
        <motion.li key={v.path + i} className="relative flex items-center gap-2 rounded-md border px-2 py-1 font-mono text-xs" style={{ marginLeft: v.depth * 18 }}
          initial={animate ? { opacity: 0.25, borderColor: "var(--line)" } : false}
          animate={{ opacity: 1, borderColor: v.ok ? "var(--ok)" : "var(--err)", backgroundColor: v.ok ? "var(--surface)" : "var(--err-soft)" }}
          transition={{ delay: animate ? visitAt(i) * k : 0, duration: 0.25 * k }}>
          {animate && <motion.span className="absolute -left-4 text-accent" initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 0] }} transition={{ delay: (visitAt(i) - 0.2) * k, duration: 0.5 * k }}>▶</motion.span>}
          <span className={cx("shrink-0 font-semibold", v.ok ? "text-ok" : "text-err")}>{v.ok ? "✓" : "✗"}</span>
          <span className="shrink-0 text-accent">{v.label}</span>
          <span className="min-w-0 truncate text-muted">{v.message}</span>
        </motion.li>
      ))}
    </ul>
  );
}

function Stage({ s, run, cfg, k }: { s: SOState; run: ReturnType<typeof simulate>; cfg: Config; k: number }) {
  const a0 = run.attempts[0];
  if (s.phase === "generate") return <Code raw={a0.raw} k={k} mode="type" />;
  if (s.phase === "parse") return <Code raw={a0.raw} k={k} mode="sweep" errLine={a0.parsed.ok ? undefined : a0.parsed.line} />;
  if (s.phase === "repair-syntax") {
    return (
      <div className="grid gap-3 md:grid-cols-2">
        <div><p className="mb-1 text-[11px] font-semibold uppercase text-muted">Before</p><Code raw={a0.raw} k={k} mode="static" highlight={(l) => (a0.kind === "trailing-comma" ? /"tags".*,\s*$/.test(l) : l.includes("estimate_ho"))} /></div>
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.6 * k, duration: 0.5 * k, ease }}>
          <p className="mb-1 text-[11px] font-semibold uppercase text-muted">After repair · re-parse {a0.reparsed?.ok ? "✓" : "✗"}</p>
          <Code raw={a0.syntaxRepaired ?? ""} k={k} mode="static" />
        </motion.div>
      </div>
    );
  }
  if (s.phase === "validate") return <div className="pl-4"><Tree visits={a0.visits} k={k} animate /></div>;
  if (s.phase === "repair-schema") {
    return (
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-1.5">
          <p className="text-[11px] font-semibold uppercase text-muted">Repairs attempted</p>
          {a0.schemaChanges.length ? a0.schemaChanges.map((c, i) => (
            <motion.div key={c} className="rounded-md border border-ok/40 bg-ok-soft px-2 py-1 font-mono text-xs" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.4) * k }}>🔧 {c}</motion.div>
          )) : <motion.p className="rounded-md border border-line bg-surface-2 px-2 py-1 text-xs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 * k }}>No safe repair: a missing value or an unknown enum value would have to be guessed.</motion.p>}
        </div>
        <motion.div className="pl-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 * k }}>
          <p className="mb-1 text-[11px] font-semibold uppercase text-muted">Re-validate</p>
          <Tree visits={a0.finalVisits} k={k} animate={false} />
        </motion.div>
      </div>
    );
  }
  if (s.phase === "retry") {
    const a = run.attempts[s.attempt];
    return (
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
        <Code raw={a.raw} k={k} mode="type" />
        <MiniPipeline a={a} k={k} />
      </div>
    );
  }
  if (s.phase === "final") {
    const last = run.attempts.at(-1)!;
    return (
      <motion.div className={cx("rounded-lg border-2 p-3", run.accepted ? "border-ok bg-ok-soft" : "border-err bg-err-soft")} initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: T.travel * k, type: "spring", stiffness: 220, damping: 16 }}>
        {run.accepted ? (
          <>
            <p className="mb-1 text-sm font-semibold">✓ Typed object delivered to the application</p>
            <pre className="overflow-x-auto font-mono text-xs">{`ticket: Ticket = ${JSON.stringify(validValue(last), null, 1).replace(/\n\s*/g, " ")}`}</pre>
          </>
        ) : (
          <>
            <p className="mb-1 text-sm font-semibold">✗ ValidationError raised after {run.attempts.length} call(s)</p>
            <p className="font-mono text-xs">{last.finalIssues.map((i) => `${i.path}: ${i.message}`).join("; ") || (last.parsed.ok ? "" : "unparseable output")}</p>
          </>
        )}
      </motion.div>
    );
  }
  return <BatchBars run={run} k={k} n={cfg.n} />;
}

function validValue(a: Attempt): unknown {
  const p = a.reparsed?.ok ? a.reparsed : a.parsed;
  return p.ok ? repairSchema(p.value).value : null;
}

function MiniPipeline({ a, k }: { a: Attempt; k: number }) {
  const parsed = a.parsed.ok || a.reparsed?.ok;
  const rows: [string, boolean | null][] = [
    ["parse", a.parsed.ok ? true : a.reparsed?.ok ? null : false],
    ["validate", parsed ? a.issues.length === 0 : false],
    ["repair", a.issues.length || !a.parsed.ok ? a.accepted : null],
    ["accepted", a.accepted],
  ];
  return (
    <div className="space-y-1.5">
      {rows.map(([name, ok], i) => (
        <motion.div key={name} className="flex items-center justify-between rounded-md border border-line px-2 py-1 text-xs" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.8 + i * 0.35) * k }}>
          <span>{name}</span>
          <span className={cx("font-semibold", ok === null ? "text-muted" : ok ? "text-ok" : "text-err")}>{ok === null ? (name === "parse" ? "repaired" : "n/a") : ok ? "✓" : "✗"}</span>
        </motion.div>
      ))}
    </div>
  );
}

function BatchBars({ run, k, n }: { run: ReturnType<typeof simulate>; k: number; n: number }) {
  return (
    <div className="space-y-2">
      {run.batch.rows.map((r, i) => {
        const segs: [string, number, string][] = [["valid", r.validAccepted, "bg-ok/80"], ["bad", r.badAccepted, "bg-err"], ["rejected", r.rejected, "bg-line"]];
        let left = 0;
        return (
          <div key={r.id} className="grid grid-cols-[150px_minmax(0,1fr)_64px] items-center gap-2 text-xs">
            <span className="font-medium">{r.label}</span>
            <div className="relative h-6 overflow-hidden rounded bg-surface-2">
              {segs.map(([name, v, cls]) => {
                const l = left; left += v / n;
                return v ? (
                  <motion.div key={name} className={cx("absolute inset-y-0 flex items-center justify-center text-[10px] font-semibold", cls, name === "bad" && "text-white dark:text-black")} style={{ left: `${l * 100}%` }}
                    initial={{ width: 0 }} animate={{ width: `${(v / n) * 100}%` }} transition={{ delay: (T.bar(i) + l * T.barDur) * k, duration: (v / n) * T.barDur * k, ease: "linear" }}>
                    {v / n > 0.07 ? (name === "bad" ? `⚠ ${v}` : v) : ""}
                  </motion.div>
                ) : null;
              })}
            </div>
            <span className="text-right font-mono"><AnimatedNumber value={(r.validAccepted / n) * 100} from={0} digits={1} delay={T.bar(i) * k} duration={T.barDur * k} />%</span>
          </div>
        );
      })}
      <div className="flex flex-wrap gap-3 pt-1 text-[11px] text-muted">
        <span className="flex items-center gap-1"><span className="h-2.5 w-3 rounded-sm bg-ok/80" />valid object accepted</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-3 rounded-sm bg-err" />⚠ wrong data accepted silently</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-3 rounded-sm bg-line" />rejected (error raised)</span>
        <span>· right column = valid success rate · calls: {run.batch.rows.map((r) => r.calls).join(" / ")}</span>
      </div>
    </div>
  );
}

function StepCard({ s, run, k }: { s: SOState; run: ReturnType<typeof simulate>; k: number }) {
  if (s.phase === "batch") {
    return (
      <div className="overflow-x-auto">
        <p className="mb-1 text-xs font-semibold">What each layer catches, by error type (computed by running each variant through the pipeline)</p>
        <table className="w-full min-w-[520px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">error type</th><th>in batch</th><th>caught by</th><th>repair</th><th>constrained decoding</th></tr></thead>
          <tbody>
            {ERROR_KINDS.map((kind, i) => {
              const plain = runAttempt(kind, false);
              const fixed = runAttempt(kind, true).accepted;
              return (
                <motion.tr key={kind} className="border-t border-line" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + i * 0.2) * k }}>
                  <td className="py-1">{KIND_LABEL[kind]}</td>
                  <td className="font-mono">{run.batch.kinds[kind]}</td>
                  <td><Badge tone={plain.parsed.ok ? "warn" : "err"}>{plain.parsed.ok ? "schema validator" : "parser"}</Badge></td>
                  <td>{fixed ? <span className="text-ok">✓ fixes it</span> : <span className="text-muted">✗ needs a retry</span>}</td>
                  <td>{SYNTAX_KINDS.includes(kind) ? <span className="text-ok">✓ prevents it</span> : <span className="text-muted">— still possible</span>}</td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }
  const layers: [string, string, "ok" | "err" | "neutral" | "warn"][] = [];
  const a0 = run.attempts[0];
  const idx = run.steps.findIndex((x) => x.state === s);
  const seen = new Set(run.steps.slice(0, idx + 1).map((x) => x.state.phase));
  layers.push(["generate", `${KIND_LABEL[a0.kind]} · ${a0.raw.length} chars`, "neutral"]);
  if (seen.has("parse")) layers.push(["parse", a0.parsed.ok ? "well-formed" : `failed on line ${a0.parsed.line + 1}`, a0.parsed.ok ? "ok" : "err"]);
  if (seen.has("repair-syntax")) layers.push(["repair syntax", a0.reparsed?.ok ? "now parses" : "still broken", a0.reparsed?.ok ? "ok" : "err"]);
  if (seen.has("validate")) layers.push(["validate", a0.issues.length ? `${a0.issues.length} issue(s)` : "schema OK", a0.issues.length ? "err" : "ok"]);
  if (seen.has("repair-schema")) layers.push(["repair schema", a0.finalIssues.length ? `${a0.finalIssues.length} issue(s) left` : "fixed", a0.finalIssues.length ? "warn" : "ok"]);
  run.attempts.slice(1, s.phase === "retry" ? s.attempt + 1 : s.phase === "final" ? undefined : 1).forEach((a, i) => layers.push([`retry ${i + 1}`, `${KIND_LABEL[a.kind]} → ${a.accepted ? "accepted" : "rejected"}`, a.accepted ? "ok" : "err"]));
  if (s.phase === "final") layers.push(["result", run.accepted ? "accepted" : "rejected", run.accepted ? "ok" : "err"]);
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-semibold">Layer by layer so far</p>
      {layers.map(([name, text, tone], i) => (
        <motion.div key={name} className="grid grid-cols-[110px_minmax(0,1fr)] items-center gap-2 text-xs" initial={i === layers.length - 1 ? { opacity: 0, x: -12 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 * k, duration: 0.3 * k }}>
          <span className="text-muted">{name}</span>
          <span><Badge tone={tone}>{tone === "ok" ? "✓" : tone === "err" ? "✗" : tone === "warn" ? "!" : "·"} {text}</Badge></span>
        </motion.div>
      ))}
      {s.phase === "parse" && a0.parsed.ok && a0.kind !== "valid" && <p className="pt-1 text-xs text-warn">Parsed fine, yet this output is wrong ({KIND_LABEL[a0.kind]}). Parse-only pipelines would accept it.</p>}
      {s.phase === "generate" && <p className="pt-1 text-xs text-muted">Schema: <span className="font-mono">{"{ title: string, priority: \"low\"|\"medium\"|\"high\", estimate_hours: number 0–100, tags: string[] }"}</span>, no extra keys. Raw: {parse(a0.raw).ok ? "parseable" : "not parseable"}.</p>}
    </div>
  );
}
