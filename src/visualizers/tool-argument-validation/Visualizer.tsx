import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, CheckCircle2, CircleHelp, Eraser, RotateCcw, TriangleAlert, Wrench, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Badge, Button, Callout, Formula, KV, Panel, Select, SimulatedLabel, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, Typewriter, ease } from "../tool-calling/shared";
import {
  CANDIDATES, DEFAULT_ARGS, TOOL_NAME, constraintText, fieldName, jsonLines, schemaNodes,
  simulateValidation, type ArgConfig, type ArgState, type CandidateId, type Check, type RepairAction, type Visit,
} from "./simulation";

/** Timing (seconds at 1×): cursor moves, connector draws, then each check resolves `check` apart. */
const T = { move: 0.45, link: 0.45, firstCheck: 0.9, check: 0.4 };
const checkAt = (i: number) => T.firstCheck + i * T.check;
const stepMsFor = (steps: { state: ArgState }[]) => (i: number) => {
  const s = steps[i]?.state;
  if (!s) return 2500;
  if (s.phase === "node") return (checkAt(s.visits.at(-1)!.checks.length) + 0.9) * 1000;
  return { propose: 2600, format: 4400, repair: 3200, revalidate: 3200, execute: 2800, node: 0 }[s.phase];
};

const NODES = schemaNodes();
const ROW = 44, TOP = 22, NODE_W = 250, IND = 22, JX = 330, LINE = 25;

const WHY: Record<ArgState["phase"], string> = {
  propose: "Model output is untrusted input. Validate it like you would validate a form submitted from the internet.",
  node: "Checking node by node gives a precise path for every error, which is what makes the error message actionable.",
  format: "The model reads the error. 'must be at most 14 (you sent 30)' gets fixed; a stack trace or 'invalid input' gets guessed at.",
  repair: "A repair can fix form (types, casing, unknown fields) but not missing facts. Clamping a number passes validation and silently changes intent.",
  revalidate: "Always validate the repaired call again: repairs introduce their own mistakes, and stop-at-first validation hides later errors.",
  execute: "Only arguments that pass the schema reach the real system.",
};

export default function ToolArgumentValidationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<ArgConfig>(DEFAULT_ARGS);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulateValidation(cfg), [cfg]);
  const stepMs = useMemo(() => stepMsFor(sim.steps), [sim]);
  const pb = usePlayback(sim.steps.length, stepMs);
  const step = sim.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${cfg.candidate}-${cfg.coerce}-${cfg.allErrors}`;
  const set = (p: Partial<ArgConfig>) => setCfg((c) => ({ ...c, ...p }));

  if (tab === "metrics") return null;

  const visit = s.phase === "node" ? s.visits.at(-1)! : null;
  const checksRun = s.visits.reduce((a, v) => a + v.checks.length, 0);
  const failed = s.visits.reduce((a, v) => a + v.checks.filter((c) => !c.ok).length, 0);

  return (
    <Workspace
      main={
        <>
          <Panel title="Schema tree ↔ arguments" actions={<SimulatedLabel>Scripted model</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={step.title} sub={s.phase === "node" ? "Each check on this node resolves in turn; the badge shows the result." : step.description} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[600px]">
                <TreeSvg s={s} k={k} animKey={animKey} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><CheckCircle2 size={12} className="text-ok" /> all checks pass</span>
              <span className="flex items-center gap-1"><XCircle size={12} className="text-err" /> a check failed</span>
              <span className="flex items-center gap-1"><span className="font-mono">–</span> value not present</span>
              <span>left: JSON Schema for <span className="font-mono">{TOOL_NAME}</span> · right: the arguments the model sent</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <PhaseCard s={s} k={k} visit={visit} allErrors={cfg.allErrors} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          <Panel title="Candidate arguments & validator mode" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_ARGS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <Select label="Arguments the model sends" value={cfg.candidate} onChange={(v: CandidateId) => set({ candidate: v })} options={(Object.keys(CANDIDATES) as CandidateId[]).map((c) => ({ value: c, label: CANDIDATES[c].label }))} />
              <div className="space-y-2 pt-1">
                <Toggle label="Coerce types (lenient mode)" checked={cfg.coerce} onChange={(v) => set({ coerce: v })} hint={'Like ajv coerceTypes: "3" → 3, 42 → "42". Off = strict.'} />
                <Toggle label="Report all errors" checked={cfg.allErrors} onChange={(v) => set({ allErrors: v })} hint="Off = stop at the first error (ajv default)." />
              </div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[s.phase]} />
          <Panel title="Check in symbols">
            <TexBlock>{visit ? visitTex(visit) : `\\text{checks run} = ${checksRun},\\quad \\text{failed} = ${failed}`}</TexBlock>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["mode", cfg.coerce ? "coerce types" : "strict"],
              ["error reporting", cfg.allErrors ? "all errors" : "stop at first"],
              ["nodes checked", `${s.visits.length} / ${NODES.length}`],
              ["checks run / failed", `${checksRun} / ${failed}`],
              ["coercions applied", s.first.coercions],
              ["first-pass errors", s.first.errors.length],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const esc = (t: string) => t.replace(/[\\{}_#$%&]/g, (c) => `\\${c}`);
/** Plain check text → KaTeX: comparison symbols in math mode, everything else in \text{}. */
const texText = (t: string) => t.split(/(≥|≤|<|>|→)/).map((part) => ({ "≥": "\\ge", "≤": "\\le", "<": "<", ">": ">", "→": "\\to" } as Record<string, string>)[part] ?? (part.trim() ? `\\text{${esc(part)}}` : "")).join(" ");
function visitTex(v: Visit): string {
  const rows = v.checks.map((c: Check) => `${texText(c.text)} & ${c.ok ? "\\checkmark" : "\\times"}`);
  return `\\begin{array}{ll} \\textbf{${esc(fieldName(v.path))}} & \\\\ ${rows.join(" \\\\ ")} \\end{array}`;
}

function TreeSvg({ s, k, animKey }: { s: ArgState; k: number; animKey: string }) {
  const lines = jsonLines(s.args);
  const H = Math.max(TOP + NODES.length * ROW, 30 + lines.length * LINE) + 8;
  const visitOf = (p: string) => s.visits.find((v) => v.path === p);
  const curIdx = s.current === null ? -1 : NODES.findIndex((n) => n.path === s.current);
  const curLine = s.current === null ? -1 : lines.findIndex((l) => l.path === s.current);
  const nodeY = (i: number) => TOP + i * ROW + 17;
  const lineY = (j: number) => 30 + j * LINE;
  const errPaths = new Set((s.phase === "revalidate" ? s.second?.errors ?? [] : s.first.errors).map((e) => e.instancePath));
  const repairOf = (p: string | null): RepairAction | undefined => (s.phase === "revalidate" && p ? s.repaired?.actions.find((a) => a.path === p) : undefined);
  const coerced = (p: string | null) => (p ? visitOf(p)?.checks.find((c) => c.coerced)?.coerced : undefined);
  const nChecks = s.phase === "node" && s.visits.length ? s.visits.at(-1)!.checks.length : 0;
  return (
    <svg viewBox={`0 0 680 ${H}`} className="h-auto w-full" role="img" aria-label="JSON Schema tree on the left, candidate arguments on the right">
      {/* tree connectors */}
      {NODES.map((n, i) => {
        if (n.depth === 0) return null;
        const parent = NODES.findLastIndex((m, j) => j < i && m.depth === n.depth - 1);
        const x = 16 + (n.depth - 1) * IND + 10;
        return <path key={`c${n.path}`} d={`M${x} ${nodeY(parent) + 17} V${nodeY(i)} H${16 + n.depth * IND}`} fill="none" stroke="var(--line)" strokeWidth={1.5} />;
      })}
      {/* cursor */}
      {curIdx >= 0 && (
        <motion.rect x={16 + NODES[curIdx].depth * IND - 4} width={NODE_W - NODES[curIdx].depth * IND + 8} height={42} rx={9} fill="none" stroke="var(--accent)" strokeWidth={2.5}
          initial={false} animate={{ y: TOP + curIdx * ROW - 4, x: 16 + NODES[curIdx].depth * IND - 4, width: NODE_W - NODES[curIdx].depth * IND + 8 }} transition={{ duration: T.move * k, ease }} />
      )}
      {NODES.map((n, i) => {
        const v = visitOf(n.path);
        const isCur = i === curIdx;
        const status = v?.status;
        const x = 16 + n.depth * IND, w = NODE_W - n.depth * IND, y = TOP + i * ROW;
        const badgeDelay = isCur && s.phase === "node" ? checkAt(nChecks) * k : 0;
        return (
          <g key={n.path}>
            <rect x={x} y={y} width={w} height={34} rx={7} fill={isCur ? "var(--accent-soft)" : "var(--surface)"} stroke="var(--line)" />
            <text x={x + 9} y={y + 14} fontSize={12} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">{n.key}{(SCHEMA_REQ[n.path] ? "*" : "")}</text>
            <text x={x + 9} y={y + 28} fontSize={10} fill="var(--muted)">{constraintText(n.schema).slice(0, Math.floor((w - 34) / 5))}</text>
            {status && (
              <motion.g key={`${animKey}-${n.path}-${status}`} initial={isCur ? { scale: 0, opacity: 0 } : false} animate={{ scale: 1, opacity: 1 }} transition={{ delay: badgeDelay, type: "spring", stiffness: 300, damping: 16 }}
                style={{ transformBox: "fill-box", transformOrigin: "center" }}>
                <circle cx={x + w - 15} cy={y + 17} r={9} fill={status === "ok" ? "var(--ok)" : status === "error" ? "var(--err)" : "var(--surface-2)"} stroke={status === "absent" ? "var(--line)" : "none"} />
                <text x={x + w - 15} y={y + 21} textAnchor="middle" fontSize={12} fontWeight={700} fill={status === "absent" ? "var(--muted)" : "var(--surface)"}>{status === "ok" ? "✓" : status === "error" ? "✗" : "–"}</text>
              </motion.g>
            )}
          </g>
        );
      })}
      {/* JSON panel */}
      <rect x={JX - 10} y={6} width={680 - JX + 4} height={H - 10} rx={8} fill="var(--surface-2)" opacity={0.6} />
      {lines.map((l, j) => {
        const v = l.path ? visitOf(l.path) : undefined;
        const bad = l.path !== null && errPaths.has(l.path) && (s.phase !== "node" || v?.status === "error" || (l.path.split("/").length === 2 && !!visitOf("")));
        const rep = repairOf(l.path);
        const co = coerced(l.path);
        const text = co && v ? l.text.replace(JSON.stringify(co.from), JSON.stringify(co.to)) : l.text;
        const isCur = j === curLine;
        const tint = rep ? (rep.kind === "clamped" ? "var(--warn-soft)" : rep.kind === "ask_user" ? "var(--err-soft)" : "var(--ok-soft)") : bad ? "var(--err-soft)" : isCur ? "var(--accent-soft)" : "transparent";
        return (
          <g key={`${j}-${l.text}`}>
            <motion.rect x={JX - 4} y={lineY(j) - 16} width={680 - JX - 8} height={22} rx={4} initial={false} animate={{ fill: tint }} transition={{ duration: 0.3 * k, delay: isCur ? (T.move + T.link) * k : 0 }} />
            <AnimatePresence mode="wait" initial={false}>
              <motion.text key={text} x={JX + l.depth * 16} y={lineY(j)} fontSize={12} fontFamily="var(--font-mono)" fill={bad ? "var(--err)" : "var(--fg)"}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 * k, delay: co ? checkAt(0) * k : 0 }}>
                {text}
              </motion.text>
            </AnimatePresence>
            {co && v && <text x={668} y={lineY(j)} textAnchor="end" fontSize={10} fill="var(--warn)">coerced</text>}
            {rep && <text x={668} y={lineY(j)} textAnchor="end" fontSize={10} fill={rep.kind === "clamped" ? "var(--warn)" : rep.kind === "ask_user" ? "var(--err)" : "var(--ok)"}>{rep.kind.replace("_", " ")}</text>}
          </g>
        );
      })}
      {/* connector from the current schema node to its value */}
      {curIdx >= 0 && (
        <motion.path key={`link-${animKey}`} d={`M${16 + NODE_W} ${nodeY(curIdx)} C${JX - 30} ${nodeY(curIdx)} ${JX - 40} ${curLine >= 0 ? lineY(curLine) - 5 : nodeY(curIdx)} ${JX - 6} ${curLine >= 0 ? lineY(curLine) - 5 : nodeY(curIdx)}`}
          fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray={curLine >= 0 ? undefined : "4 4"}
          initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: T.move * k, duration: T.link * k, ease }} />
      )}
    </svg>
  );
}
const SCHEMA_REQ: Record<string, boolean> = Object.fromEntries(NODES.filter((n) => n.depth > 0).map((n) => {
  const parent = NODES.find((p) => p.path === n.path.split("/").slice(0, -1).join("/"));
  return [n.path, !!parent?.schema.required?.includes(n.key)];
}));

const ACTION_ICON = { fixed: Wrench, removed: Eraser, clamped: TriangleAlert, ask_user: CircleHelp } as const;
const ACTION_TONE = { fixed: "ok", removed: "ok", clamped: "warn", ask_user: "err" } as const;

function PhaseCard({ s, k, visit, allErrors }: { s: ArgState; k: number; visit: Visit | null; allErrors: boolean }) {
  const at = (sec: number) => ({ delay: sec * k, duration: 0.35 * k, ease });
  if (s.phase === "propose")
    return (
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={at(0.2)}><Formula>{`${TOOL_NAME}(${JSON.stringify(s.args, null, 2)})`}</Formula></motion.div>
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.8)} className="text-sm leading-relaxed">
          The validator will visit <strong>{NODES.length} schema nodes</strong> top to bottom. At each node it runs that node's checks (type, required, enum, range, pattern, unknown fields) against the matching value.
        </motion.p>
      </div>
    );
  if (s.phase === "node" && visit) {
    const node = NODES.find((n) => n.path === visit.path)!;
    const range = visit.checks.find((c) => c.keyword === "minimum" || c.keyword === "maximum");
    const value = visit.path.split("/").slice(1).reduce<unknown>((o, key) => (o && typeof o === "object" ? (o as Record<string, unknown>)[key] : undefined), s.first.data);
    return (
      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <ul className="space-y-1.5">
          {visit.checks.map((c, i) => (
            <motion.li key={i} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={at(checkAt(i))} className="flex items-start gap-2 text-sm">
              {c.ok ? <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-ok" aria-label="pass" /> : <XCircle size={15} className="mt-0.5 shrink-0 text-err" aria-label="fail" />}
              <span><Badge className="mr-1 font-mono">{c.keyword}</Badge>{c.text}</span>
            </motion.li>
          ))}
        </ul>
        <div className="text-xs">
          {range && typeof value === "number" && node.schema.minimum !== undefined && node.schema.maximum !== undefined
            ? <NumberLine min={node.schema.minimum} max={node.schema.maximum} value={value} k={k} />
            : node.schema.enum && typeof value === "string"
              ? <EnumMatch allowed={node.schema.enum} value={value} k={k} />
              : <p className="text-muted">Schema for <span className="font-mono">{fieldName(visit.path)}</span>: {constraintText(node.schema)}. Value: <span className="font-mono">{value === undefined ? "absent" : JSON.stringify(value)}</span></p>}
        </div>
      </div>
    );
  }
  if (s.phase === "format")
    return (
      <div className="grid items-start gap-3 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1.3fr)]">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.1)}>
          <div className="mb-1 text-[11px] text-muted">raw validator errors</div>
          <Formula>{JSON.stringify(s.first.errors.map(({ instancePath, keyword, params }) => ({ instancePath, keyword, params })), null, 1)}</Formula>
        </motion.div>
        <motion.div initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={at(0.8)} className="hidden self-center md:block"><ArrowRight className="text-muted" /></motion.div>
        <div>
          <div className="mb-1 text-[11px] text-muted">tool result sent back to the model</div>
          <pre className="overflow-x-auto rounded-md border border-err/40 bg-err-soft/50 p-2 font-mono text-xs leading-relaxed whitespace-pre-wrap"><Typewriter text={s.friendly ?? ""} delay={1 * k} duration={2.4 * k} /></pre>
        </div>
      </div>
    );
  if (s.phase === "repair" && s.repaired)
    return (
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <ul className="space-y-1.5">
          {s.repaired.actions.map((a, i) => {
            const Icon = ACTION_ICON[a.kind];
            return (
              <motion.li key={i} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={at(0.3 + i * 0.45)} className="flex items-start gap-2 text-sm">
                <Icon size={15} className={cx("mt-0.5 shrink-0", a.kind === "clamped" ? "text-warn" : a.kind === "ask_user" ? "text-err" : "text-ok")} aria-hidden />
                <span><Badge tone={ACTION_TONE[a.kind]} className="mr-1">{a.kind.replace("_", " ")}</Badge>{a.text}</span>
              </motion.li>
            );
          })}
        </ul>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.5 + s.repaired.actions.length * 0.45)}>
          <div className="mb-1 text-[11px] text-muted">next attempt</div>
          <Formula>{`${TOOL_NAME}(${JSON.stringify(s.repaired.args, null, 2)})`}</Formula>
        </motion.div>
      </div>
    );
  const v = s.verdict!;
  const second = s.second;
  return (
    <div className="space-y-2">
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={at(0.3)}>
        <Callout tone={v.tone} title={v.tone === "ok" ? "Call goes through" : v.tone === "warn" ? "Valid, but not what the user asked for" : "Not executable yet"}>{v.text}</Callout>
      </motion.div>
      {second && (
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.9)} className="text-xs text-muted">
          Second pass: {second.visits.filter((x) => x.status === "ok").length} nodes pass, {second.visits.filter((x) => x.status === "error").length} fail, {second.errors.length} error(s).
          {!allErrors && !second.valid && " Stop-at-first validation surfaces one error per round trip, so more attempts are needed."}
        </motion.p>
      )}
    </div>
  );
}

function NumberLine({ min, max, value, k }: { min: number; max: number; value: number; k: number }) {
  const lo = Math.min(0, value) - 1, hi = Math.max(max, value) + 2;
  const X = (v: number) => 10 + ((v - lo) / (hi - lo)) * 280;
  const ok = value >= min && value <= max;
  return (
    <svg viewBox="0 0 300 64" className="h-auto w-full max-w-[320px]" role="img" aria-label={`allowed range ${min} to ${max}, value ${value}`}>
      <line x1={10} x2={290} y1={34} y2={34} stroke="var(--line)" strokeWidth={2} />
      <motion.rect x={X(min)} y={28} height={12} rx={3} fill="var(--ok)" opacity={0.35} initial={{ width: 0 }} animate={{ width: X(max) - X(min) }} transition={{ delay: 0.3 * k, duration: 0.6 * k, ease }} />
      <text x={X(min)} y={56} fontSize={10} textAnchor="middle" fill="var(--muted)">{min}</text>
      <text x={X(max)} y={56} fontSize={10} textAnchor="middle" fill="var(--muted)">{max}</text>
      <text x={(X(min) + X(max)) / 2} y={22} fontSize={10} textAnchor="middle" fill="var(--ok)">allowed</text>
      <motion.g initial={{ x: X(lo), opacity: 0 }} animate={{ x: X(value), opacity: 1 }} transition={{ delay: 0.8 * k, duration: 0.8 * k, ease }}>
        <circle cy={34} r={7} fill={ok ? "var(--ok)" : "var(--err)"} stroke="var(--surface)" strokeWidth={2} />
        <text y={56} fontSize={11} fontWeight={700} textAnchor="middle" fill={ok ? "var(--ok)" : "var(--err)"}>{value}</text>
      </motion.g>
    </svg>
  );
}

function EnumMatch({ allowed, value, k }: { allowed: string[]; value: string; k: number }) {
  return (
    <div className="space-y-2">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 * k, duration: 0.4 * k }} className="font-mono">value: "{value}"</motion.div>
      <div className="flex flex-wrap gap-1.5">
        {allowed.map((a, i) => {
          const hit = a === value;
          const close = !hit && a.toLowerCase() === value.toLowerCase();
          return (
            <motion.span key={a} initial={{ opacity: 0.4 }} animate={{ opacity: 1, scale: hit || close ? [1, 1.15, 1] : 1 }} transition={{ delay: (0.9 + i * 0.25) * k, duration: 0.4 * k }}
              className={cx("rounded border px-2 py-0.5 font-mono", hit ? "border-ok bg-ok-soft" : close ? "border-warn bg-warn-soft" : "border-line")}>
              "{a}"{hit ? " ✓" : close ? " (differs only in case)" : ""}
            </motion.span>
          );
        })}
      </div>
    </div>
  );
}
