import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, Typewriter, ease } from "./shared";
import {
  DEFAULT_CALLING, LANES, MODEL_TURN_MS, QUESTIONS, simulateCalling,
  type Arrow, type CallingConfig, type CallingState, type ChatMessage, type Detail, type Lane, type Mode, type OnParseError, type Question,
} from "./simulation";

/** One timing schedule (seconds at 1×): arrows draw and packets travel, then the card and transcript react on landing. */
const T = { start: 0.15, travel: 1.0, land: 1.15, item: 0.3 };
const DUR: Record<Detail["kind"], number> = { user: 2600, tool_call: 3600, parse: 3000, validate: 3600, execute: 3200, result: 3000, tool_msg: 3200, final: 3600, crash: 2800 };
const stepMsFor = (steps: { state: CallingState }[]) => (i: number) => DUR[steps[i]?.state.detail.kind ?? "user"];

const LANE_X: Record<Lane, number> = { user: 58, model: 188, runtime: 318, tool: 448 };
const LANE_NAME: Record<Lane, string> = { user: "User", model: "Model", runtime: "Runtime", tool: "Tool" };
const ROW0 = 70, ROW_H = 30;

const WHY: Partial<Record<Detail["kind"], string>> = {
  tool_call: "The model only proposes a call. Whatever executes is the runtime's job, and the runtime is your security boundary.",
  parse: "Arguments are generated token by token as text. They can be truncated or invalid JSON, so parsing can fail.",
  validate: "Schema checks stop wrong or missing fields before they reach a real system.",
  execute: "Independent calls in one turn can run concurrently: the turn costs the slowest call, not the sum.",
  tool_msg: "The tool_call_id is how the model matches each result to the call that asked for it.",
  crash: "A parse error is recoverable if you hand it back to the model. Crashing turns a small model mistake into an outage.",
};

export default function ToolCallingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<CallingConfig>(DEFAULT_CALLING);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulateCalling(cfg), [cfg]);
  const stepMs = useMemo(() => stepMsFor(run.steps), [run]);
  const pb = usePlayback(run.steps.length, stepMs);
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<CallingConfig>) => setCfg((c) => ({ ...c, ...p }));

  if (tab === "metrics") return null;

  const seq = cfg.mode !== "parallel";
  const lat = run.toolLatencies;
  const toolTex = lat.length === 0 ? "0" : seq ? `\\textstyle\\sum_i t_i = ${lat.join(" + ")} = ${run.wallToolMs}` : `\\max(${lat.join(", ")}) = ${run.wallToolMs}`;

  return (
    <Workspace
      main={
        <>
          <Panel title="Sequence diagram" actions={<SimulatedLabel>Simulated model &amp; tool</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} stepKey={step.id} headline={step.title} sub={step.description} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_250px]">
              <div className="overflow-x-auto">
                <div className="min-w-[440px]">
                  <SequenceSvg s={s} k={k} animKey={animKey} />
                </div>
              </div>
              <Transcript s={s} k={k} animKey={animKey} />
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>message in flight</span>
              <span className="flex items-center gap-1"><svg width="22" height="10"><rect x="1" y="1" width="20" height="8" rx="3" fill="var(--surface-2)" stroke="var(--line)" /></svg>work inside the runtime</span>
              <span>time flows downward; each row is one message</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <DetailCard d={s.detail} k={k} s={s} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Scenario & faults" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_CALLING); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="User asks" value={cfg.question} onChange={(v: Question) => set({ question: v })} options={(Object.keys(QUESTIONS) as Question[]).map((q) => ({ value: q, label: QUESTIONS[q].text }))} />
              <Select label="Mode" value={cfg.mode} onChange={(v: Mode) => set({ mode: v })} options={[
                { value: "sequential", label: "Sequential tool calls" },
                { value: "parallel", label: "Parallel tool calls" },
                { value: "malformed", label: "Break it: malformed JSON arguments" },
              ]} />
              <Select label="Runtime on parse error" value={cfg.onParseError} onChange={(v: OnParseError) => set({ onParseError: v })} options={[
                { value: "return", label: "Return the error to the model" },
                { value: "crash", label: "Throw (no error handling)" },
              ]} />
            </div>
            <p className="mt-2 text-xs text-muted">
              {cfg.mode === "malformed" ? "The model's first call is missing its closing brace. Switch the parse-error policy to see recovery vs a crash."
                : cfg.mode === "parallel" ? "With one city there is nothing to parallelise; pick the two-city question to see concurrent calls."
                : "The runtime-side policy only matters when parsing fails: try the malformed JSON mode."}
            </p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[s.detail.kind]} />
          <Panel title="Latency of the whole run (simulated)">
            <TexBlock>{`\\begin{aligned} T_{\\text{tools}} &= ${toolTex}\\,\\text{ms} \\\\ T_{\\text{total}} &= n_{\\text{turns}} \\cdot ${MODEL_TURN_MS} + T_{\\text{tools}} \\\\ &= ${run.final.modelTurns} \\cdot ${MODEL_TURN_MS} + ${run.wallToolMs} = ${run.final.clockMs}\\,\\text{ms} \\end{aligned}`}</TexBlock>
            <p className="mt-2 text-[11px] text-muted">{seq ? "Sequential: each call waits for the previous result and costs an extra model turn." : "Parallel: one turn asks for every city, and the runtime waits only for the slowest call."}</p>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["messages in transcript", s.transcript.length],
              ["model turns so far", s.modelTurns],
              ["tool executions so far", s.toolExecutions],
              ["simulated clock", `${s.clockMs} ms`],
              ["status", s.status],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function SequenceSvg({ s, k, animKey }: { s: CallingState; k: number; animKey: string }) {
  const rows = s.arrows.length;
  const H = ROW0 + Math.max(rows, 6) * ROW_H + 10;
  const fresh = new Set(s.newArrowIds);
  const activeLanes = new Set(s.arrows.filter((a) => fresh.has(a.id)).map((a) => a.to));
  return (
    <svg viewBox={`0 0 506 ${H}`} className="h-auto w-full" role="img" aria-label={`Sequence diagram with ${rows} messages between user, model, runtime and tool`}>
      {LANES.map((l) => (
        <g key={l}>
          <line x1={LANE_X[l]} x2={LANE_X[l]} y1={38} y2={H - 4} stroke="var(--line)" strokeWidth={1.5} strokeDasharray="4 4" />
          {activeLanes.has(l) && (
            <motion.rect key={`pulse-${animKey}-${l}`} x={LANE_X[l] - 50} y={6} width={100} height={28} rx={8} fill="none" stroke="var(--accent)" strokeWidth={2}
              initial={{ opacity: 0.7, scale: 1 }} animate={{ opacity: 0, scale: 1.25 }} style={{ transformBox: "fill-box", transformOrigin: "center" }}
              transition={{ duration: 1.4, repeat: Infinity, delay: T.land * k }} />
          )}
          <motion.rect x={LANE_X[l] - 46} y={8} width={92} height={24} rx={7} initial={false}
            animate={{ fill: activeLanes.has(l) ? "var(--accent)" : "var(--surface-2)" }} transition={{ duration: 0.3, delay: activeLanes.has(l) ? T.land * k : 0 }} stroke="var(--line)" />
          <motion.text x={LANE_X[l]} y={24} textAnchor="middle" fontSize={12} fontWeight={600} initial={false}
            animate={{ fill: activeLanes.has(l) ? "var(--surface)" : "var(--fg)" }} transition={{ duration: 0.3, delay: activeLanes.has(l) ? T.land * k : 0 }}>{LANE_NAME[l]}</motion.text>
        </g>
      ))}
      {s.arrows.map((a) => <ArrowRow key={fresh.has(a.id) ? `${a.id}-${animKey}` : a.id} a={a} fresh={fresh.has(a.id)} k={k} />)}
    </svg>
  );
}

function ArrowRow({ a, fresh, k }: { a: Arrow; fresh: boolean; k: number }) {
  const y = ROW0 + a.row * ROW_H;
  const color = a.ok ? "var(--accent)" : "var(--err)";
  const travel = T.travel * Math.max(0.35, a.travel) * k;
  if (a.from === a.to) {
    const x = LANE_X[a.from];
    return (
      <motion.g initial={fresh ? { opacity: 0, scale: 0.6 } : false} animate={{ opacity: fresh ? 1 : 0.55, scale: 1 }} style={{ transformBox: "fill-box", transformOrigin: "center" }}
        transition={{ delay: T.start * k, duration: 0.5 * k, ease }}>
        <rect x={x - 60} y={y - 11} width={120} height={22} rx={6} fill={a.ok ? "var(--surface-2)" : "var(--err-soft)"} stroke={fresh ? color : "var(--line)"} strokeWidth={fresh ? 1.5 : 1} />
        <text x={x} y={y + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill={a.ok ? "var(--fg)" : "var(--err)"}>{a.label}</text>
      </motion.g>
    );
  }
  const dir = LANE_X[a.to] > LANE_X[a.from] ? 1 : -1;
  const x1 = LANE_X[a.from] + dir * 6, x2 = LANE_X[a.to] - dir * 8;
  const mid = (x1 + x2) / 2;
  return (
    <g opacity={fresh ? 1 : 0.55}>
      <motion.line x1={x1} y1={y} x2={x2} y2={y} stroke={fresh ? color : "var(--muted)"} strokeWidth={fresh ? 2 : 1.5} strokeDasharray={a.kind === "result" || a.kind === "tool_msg" ? "5 3" : undefined}
        initial={fresh ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={{ delay: T.start * k, duration: travel, ease: "easeInOut" }} />
      <motion.path d={`M${x2} ${y} l${-dir * 7} -4 l0 8 z`} fill={fresh ? color : "var(--muted)"} initial={fresh ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: (T.start * k) + travel }} />
      <motion.text x={mid} y={y - 6} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill={a.ok ? "var(--fg)" : "var(--err)"}
        initial={fresh ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: (T.start + 0.25) * k, duration: 0.4 * k }}>{a.label}</motion.text>
      {fresh && (
        <motion.circle r={6} cy={y} fill={color} stroke="var(--surface)" strokeWidth={2} initial={{ cx: x1, opacity: 0 }} animate={{ cx: [x1, x2], opacity: [0, 1, 1, 0] }}
          transition={{ delay: T.start * k, duration: travel, ease: "easeInOut", opacity: { delay: T.start * k, duration: travel, times: [0, 0.1, 0.85, 1] } }} />
      )}
    </g>
  );
}

const ROLE_TONE = { user: "neutral", assistant: "accent", tool: "ok" } as const;
function msgText(m: ChatMessage) {
  if (m.role === "assistant" && m.tool_calls) return m.tool_calls.map((c) => `${c.id}: ${c.name}(${c.arguments})`).join("\n");
  return m.content ?? "";
}

function Transcript({ s, k, animKey }: { s: CallingState; k: number; animKey: string }) {
  const firstNew = s.transcript.length - s.newMessages;
  return (
    <div className="min-w-0 rounded-md border border-line bg-surface-2/50 p-2">
      <div className="mb-1.5 flex items-baseline justify-between text-[11px] text-muted"><span className="font-semibold uppercase tracking-wide">messages[]</span><span>sent to the model each turn</span></div>
      <ol className="space-y-1.5">
        {s.transcript.map((m, i) => {
          const isNew = i >= firstNew;
          const err = m.role === "tool" && m.isError;
          return (
            <motion.li key={isNew ? `${i}-${animKey}` : i} initial={isNew ? { opacity: 0, x: 24 } : false} animate={{ opacity: 1, x: 0 }}
              transition={{ delay: (T.land + (i - firstNew) * 0.2) * k, duration: 0.45 * k, ease }}
              className={cx("rounded border bg-surface px-2 py-1 text-[11px]", isNew ? "border-accent" : "border-line", err && "border-err")}>
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-muted">{i}</span>
                <Badge tone={err ? "err" : ROLE_TONE[m.role]}>{m.role}</Badge>
                {m.role === "assistant" && m.tool_calls && <span className="text-muted">tool_calls</span>}
                {m.role === "tool" && <span className="font-mono text-muted">{m.tool_call_id}</span>}
              </div>
              <div className="mt-0.5 font-mono leading-snug break-words whitespace-pre-wrap">{msgText(m)}</div>
            </motion.li>
          );
        })}
      </ol>
    </div>
  );
}

function DetailCard({ d, k, s }: { d: Detail; k: number; s: CallingState }) {
  const at = (sec: number) => ({ delay: sec * k, duration: 0.4 * k, ease });
  switch (d.kind) {
    case "user":
      return (
        <div className="space-y-2">
          <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={at(0.2)} className="w-fit max-w-full rounded-2xl rounded-bl-sm bg-surface-2 px-3 py-2 text-sm">
            <Typewriter text={d.text} delay={0.3 * k} duration={0.9 * k} />
          </motion.div>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(T.land)} className="text-xs text-muted">
            Appended as <span className="font-mono">{"{ role: \"user\" }"}</span>. The model also receives the tool list: <span className="font-mono">get_weather(city: "Paris" | "Tokyo")</span>.
          </motion.p>
        </div>
      );
    case "tool_call":
      return (
        <div className="space-y-2">
          <pre className="overflow-x-auto rounded-md bg-surface-2 p-2.5 font-mono text-xs leading-relaxed">
            {"{ \"role\": \"assistant\", \"content\": null,\n  \"tool_calls\": ["}
            {d.calls.map((c, i) => (
              <span key={c.id}>{`\n    { "id": "${c.id}", "type": "function",\n      "function": { "name": "${c.name}",\n                    "arguments": `}<span className="rounded bg-warn-soft px-0.5 text-fg">"<Typewriter text={c.arguments.replace(/"/g, '\\"')} delay={(0.3 + i * 0.9) * k} duration={0.8 * k} />"</span>{" } }"}</span>
            ))}
            {"\n  ] }"}
          </pre>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.4 + d.calls.length * 0.9)} className="text-xs">
            <strong>arguments is a string</strong>, generated token by token. Nothing has run yet: the runtime must parse it, check it and decide whether to execute it.
          </motion.p>
        </div>
      );
    case "parse":
      return (
        <div className="space-y-2">
          {d.items.map((p, i) => (
            <div key={p.callId} className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono text-muted">{p.callId}</span>
              <motion.span initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={at(0.2 + i * 0.3)} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono">'{p.raw}'</motion.span>
              <motion.span initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={at(0.7 + i * 0.3)} className="flex items-center gap-1 text-muted">JSON.parse <ArrowRight size={12} /></motion.span>
              {p.ok ? (
                <motion.span initial={{ opacity: 0, scale: 0.7 }} animate={{ opacity: 1, scale: 1 }} transition={at(1.1 + i * 0.3)} className="flex items-center gap-1 rounded border border-ok/40 bg-ok-soft px-1.5 py-0.5 font-mono">
                  <CheckCircle2 size={12} className="text-ok" /> {Object.entries(p.parsed!).map(([key, v]) => `${key}: ${JSON.stringify(v)} (${typeof v})`).join(", ")}
                </motion.span>
              ) : (
                <motion.span initial={{ opacity: 0, scale: 0.7 }} animate={{ opacity: 1, scale: [0.7, 1.08, 1] }} transition={at(1.1)} className="flex items-center gap-1 rounded border border-err/40 bg-err-soft px-1.5 py-0.5 font-mono text-err">
                  <XCircle size={12} /> SyntaxError: {p.error}
                </motion.span>
              )}
            </div>
          ))}
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(1.6)} className="text-xs text-muted">
            {d.items.every((p) => p.ok) ? "The string became a typed object. Only now can the runtime check field names and types." : "The string ends before the closing brace, so there is no object to validate or execute."}
          </motion.p>
        </div>
      );
    case "validate":
      return (
        <div className="grid gap-3 sm:grid-cols-2">
          {d.items.map((v, j) => (
            <div key={v.callId}>
              <div className="mb-1 font-mono text-[11px] text-muted">{v.callId}</div>
              <ul className="space-y-1">
                {v.checks.map((c, i) => (
                  <motion.li key={c.label} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={at(0.2 + (i + j * 0.5) * T.item)} className="flex items-center gap-1.5 text-xs">
                    {c.ok ? <CheckCircle2 size={13} className="text-ok" aria-label="pass" /> : <XCircle size={13} className="text-err" aria-label="fail" />}<span>{c.label}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      );
    case "execute": {
      return (
        <div className="space-y-2">
          {d.items.map((e) => (
            <div key={e.callId} className="grid grid-cols-[150px_minmax(0,1fr)_60px] items-center gap-2 text-xs">
              <span className="truncate font-mono">get_weather("{e.city}")</span>
              <div className="h-4 rounded bg-surface-2">
                <motion.div className="h-full rounded bg-accent/70" initial={{ width: 0 }} animate={{ width: `${(e.latencyMs / 200) * 100}%` }} transition={{ delay: T.land * k, duration: (e.latencyMs / 100) * k, ease: "linear" }} />
              </div>
              <span className="text-right font-mono"><AnimatedNumber value={e.latencyMs} from={0} digits={0} delay={T.land * k} duration={(e.latencyMs / 100) * k} /> ms</span>
            </div>
          ))}
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(T.land + d.wallMs / 100)} className="text-xs">
            {d.items.length > 1 ? <>Both started together, so this turn waits <strong>max({d.items.map((e) => e.latencyMs).join(", ")}) = {d.wallMs} ms</strong>, not {d.items.reduce((a, e) => a + e.latencyMs, 0)} ms.</> : <>This call takes <strong>{d.wallMs} ms</strong> (simulated latency).</>}
          </motion.p>
        </div>
      );
    }
    case "result":
      return (
        <div className="flex flex-wrap gap-2">
          {d.items.map((r) => (
            <motion.div key={r.callId} initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={at(T.start + T.travel * (r.latencyMs / Math.max(...d.items.map((x) => x.latencyMs))))}
              className="rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-xs">
              <div className="text-muted">{r.callId} · {r.city} · {r.latencyMs} ms</div>
              <div>{"{ temp_c: "}{r.result.temp_c}{", condition: \""}{r.result.condition}{"\" }"}</div>
            </motion.div>
          ))}
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(T.land + 0.3)} className="w-full text-xs text-muted">A native object in the runtime. The model cannot see it until it is written into the transcript.</motion.p>
        </div>
      );
    case "tool_msg":
      return (
        <div className="space-y-2">
          {d.items.map((t, i) => (
            <div key={t.callId} className="flex flex-wrap items-center gap-2 text-xs">
              <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.2 + i * 0.3)} className="text-muted">JSON.stringify <ArrowRight size={12} className="inline" /></motion.span>
              <motion.span initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={at(0.5 + i * 0.3)} className={cx("max-w-full rounded px-1.5 py-0.5 font-mono break-all", t.isError ? "bg-err-soft text-err" : "bg-surface-2")}>
                {"{ role: \"tool\", tool_call_id: \""}{t.callId}{"\", content: '"}{t.serialized}{"' }"}
              </motion.span>
            </div>
          ))}
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(T.land + 0.3)} className="text-xs">
            Transcript grows to <strong>{s.transcript.length} messages</strong>. The model is called again and reads all of them{d.items.some((t) => t.isError) ? ", including the error, which tells it how to fix the call." : "."}
          </motion.p>
        </div>
      );
    case "final":
      return (
        <div className="space-y-2">
          <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={at(T.land - 0.4)} className="ml-auto w-fit max-w-full rounded-2xl rounded-br-sm bg-accent-soft px-3 py-2 text-sm">
            <Typewriter text={d.text} delay={T.land * k} duration={1 * k} />
          </motion.div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge>model turns: {s.modelTurns}</Badge><Badge>tool executions: {s.toolExecutions}</Badge><Badge>messages: {s.transcript.length}</Badge><Badge>simulated time: {s.clockMs} ms</Badge>
          </div>
        </div>
      );
    case "crash":
      return <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={at(T.land)}><Callout tone="err" title="Request failed">{d.error}. The user sees an error, and the model never got the chance to correct a one-character mistake.</Callout></motion.div>;
  }
}
