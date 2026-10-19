import { useMemo, useState, type FormEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, CheckCircle2, RotateCcw, Wrench, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, Formula, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../tool-calling/shared";
import {
  CATALOGUES, DEFAULT_BUDGET, PHASES, REQUESTS, WEIGHTS, accuracyOnSamples, isStopword, simulateSelection, stem,
  type CatalogueId, type Phase, type SelectionResult, type ToolScore,
} from "./simulation";

/** Timing schedule (seconds at 1×). Rows are processed top to bottom, `rowGap` apart; each bar grows for `grow`. */
const T = { lead: 0.25, rowGap: 0.07, grow: 0.6 };
const BASE: Record<Phase, number> = { request: 2800, catalogue: 2800, overlap: 1800, args: 1800, penalty: 1800, rank: 2600, pick: 3400 };
const phaseMs = (p: Phase, n: number) => BASE[p] + (["overlap", "args", "penalty", "catalogue"].includes(p) ? n * T.rowGap * 1000 : 0);

const SCALE = 2.3; // score axis spans −1 … +1.3
const Z = (1 / SCALE) * 100; // zero line in %
const pct = (v: number) => (v / SCALE) * 100;

const WHY: Record<Phase, string> = {
  request: "Models match requests to tools largely through the words in names and descriptions; vague words carry no signal.",
  catalogue: "Every tool definition costs prompt space. Past a budget, something gets cut or ignored, and it is usually the disambiguating detail at the end.",
  overlap: "If two descriptions say the same thing, overlap cannot tell them apart.",
  args: "A tool that cannot be called with what the user said is a poor match even if its description sounds right.",
  penalty: "Saying what a tool is NOT for is one of the cheapest, most effective description fixes.",
  rank: "A small margin between first and second place means the choice is fragile: one reworded request can flip it.",
  pick: "Evaluate tool selection on a labelled set of requests, not on one demo query.",
};

export default function ToolSelectionVisualizer({ tab }: { tab: VisualizerTab }) {
  const [request, setRequest] = useState<string>(REQUESTS[0].text);
  const [draft, setDraft] = useState("");
  const [catalogue, setCatalogue] = useState<CatalogueId>("clear");
  const [budget, setBudget] = useState(DEFAULT_BUDGET);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulateSelection({ request, catalogue, budget }), [request, catalogue, budget]);
  const nTools = sim.result.scores.length;
  const stepMs = useMemo(() => (i: number) => phaseMs(PHASES[i], nTools), [nTools]);
  const pb = usePlayback(sim.steps.length, stepMs);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${request}-${catalogue}-${budget}`;
  const acc = useMemo(() => accuracyOnSamples(catalogue, budget), [catalogue, budget]);
  const r = sim.result;

  if (tab === "metrics") return null;

  const submit = (e: FormEvent) => { e.preventDefault(); if (draft.trim()) setRequest(draft.trim()); };
  const byName = (n: string) => r.scores.find((s) => s.name === n)!;
  const w = byName(r.winner);
  const second = r.runnerUp ? byName(r.runnerUp) : null;
  const sTex = (s: ToolScore) => `s = ${WEIGHTS.overlap}\\cdot ${s.overlap.toFixed(2)} + ${WEIGHTS.args}\\cdot ${s.argFit.toFixed(2)} - ${WEIGHTS.penalty}\\cdot ${s.penalty} = ${s.total.toFixed(2)}`;

  return (
    <Workspace
      main={
        <>
          <Panel title="Request vs. tool catalogue" actions={<SimulatedLabel>Rule-based scorer, not an LLM</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={step.title} sub={step.description} onReplay={() => setReplay((x) => x + 1)} />
            <RequestStrip request={r.request} phase={phase} k={k} animKey={animKey} />
            <div className="overflow-x-auto">
              <div className="min-w-[680px]">
                <ScoreBoard r={r} phase={phase} k={k} animKey={animKey} expected={sim.expected} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-accent/80" />keyword overlap ×{WEIGHTS.overlap}</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-violet/80" />argument fit ×{WEIGHTS.args}</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-err/70" />exclusion penalty ×{WEIGHTS.penalty} (left of 0)</span>
              <span className="flex items-center gap-1"><span className="font-mono line-through opacity-60">word</span> cut by the budget</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <PhaseCard phase={phase} r={r} k={k} expected={sim.expected} acc={acc} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Request, catalogue & budget" actions={<Button size="sm" variant="ghost" onClick={() => { setRequest(REQUESTS[0].text); setCatalogue("clear"); setBudget(DEFAULT_BUDGET); setDraft(""); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="flex flex-wrap gap-1.5">
              {REQUESTS.map((q) => <Button key={q.id} size="sm" variant={request === q.text ? "primary" : "secondary"} onClick={() => setRequest(q.text)}>{q.text}</Button>)}
            </div>
            <form onSubmit={submit} className="mt-2 flex gap-2">
              <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Or type your own request, e.g. 'convert 40 EUR to JPY'" aria-label="Custom request" className="h-8 min-w-0 flex-1 rounded-md border border-line bg-surface px-2.5 text-sm" />
              <Button type="submit">Score it</Button>
            </form>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Select label="Tool catalogue" value={catalogue} onChange={setCatalogue} options={(Object.keys(CATALOGUES) as CatalogueId[]).map((c) => ({ value: c, label: CATALOGUES[c].label }))} />
              <Slider label="Prompt budget for tool descriptions" value={budget} min={40} max={400} step={10} format={(v) => `${v} words`} onChange={setBudget} hint="Toy stand-in for tokens: 1 word = 1 unit." />
            </div>
            {catalogue === "overlapping" && (
              <div className="mt-3 space-y-2">
                <Callout tone="warn" title="Two tools share one copy-pasted description">web_search and search_handbook both say "Search for information and answer questions." Nothing in the text tells them apart.</Callout>
                <Button size="sm" variant="primary" onClick={() => { setCatalogue("clear"); setReplay((x) => x + 1); }}><Wrench size={12} /> Fix: clarify both descriptions (add scope and "Not for")</Button>
              </div>
            )}
            {r.wordsPerTool !== null && (
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <span className="text-xs text-muted">Descriptions are cut to {r.wordsPerTool} words each.</span>
                <Button size="sm" onClick={() => setBudget(Math.min(400, Math.ceil(r.totalWords / 10) * 10))}><Wrench size={12} /> Raise the budget to fit all {r.totalWords} words</Button>
              </div>
            )}
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[phase]} />
          <Panel title="Score calculation">
            <TexBlock>{`s = ${WEIGHTS.overlap}\\cdot\\text{overlap} + ${WEIGHTS.args}\\cdot\\text{argFit} - ${WEIGHTS.penalty}\\cdot\\text{penalty}`}</TexBlock>
            {[w, ...(second ? [second] : [])].map((x, i) => (
              <div key={x.name} className="mt-2">
                <div className="mb-0.5 font-mono text-[11px]">{i === 0 ? "winner" : "runner-up"}: {x.name}</div>
                <TexBlock>{sTex(x)}</TexBlock>
              </div>
            ))}
            <p className="mt-2 text-[11px] text-muted">overlap = matched request words ÷ request words; argFit = fillable required args ÷ required args; penalty = 1 if a visible "Not for" word appears in the request.</p>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["tools in catalogue", nTools],
              ["description words / budget", `${r.totalWords} / ${r.budget}`],
              ["words kept per tool", r.wordsPerTool ?? "all"],
              ["winner", r.winner],
              ["margin over 2nd", `${r.margin.toFixed(2)}${r.tie ? " (tie)" : ""}`],
              ["accuracy on 4 sample requests", `${acc.correct}/${acc.total}`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function RequestStrip({ request, phase, k, animKey }: { request: string; phase: Phase; k: number; animKey: string }) {
  const words = request.split(/\s+/);
  const first = phase === "request";
  return (
    <div className="mb-3 flex flex-wrap items-center gap-1 rounded-md border border-line bg-surface-2/60 px-2 py-1.5 text-xs">
      <span className="mr-1 font-semibold text-muted">Request</span>
      {words.map((wd, i) => {
        const clean = wd.toLowerCase().replace(/[^a-z0-9#]+/g, " ").trim().split(" ")[0] ?? "";
        const stop = !clean || isStopword(clean);
        const stemmed = stop ? null : stem(clean);
        return (
          <motion.span key={`${animKey}-${i}`} initial={first ? { opacity: 0, y: -8 } : false} animate={{ opacity: stop ? 0.35 : 1, y: 0 }}
            transition={{ delay: first ? (0.1 + i * 0.12) * k : 0, duration: 0.35 * k, opacity: { delay: first && stop ? (1.1 + i * 0.05) * k : first ? (0.1 + i * 0.12) * k : 0 } }}
            className={cx("rounded px-1.5 py-0.5 font-mono", stop ? "line-through" : "bg-accent-soft text-accent")}>
            {wd}{stemmed && stemmed !== clean && <span className="text-muted"> → {stemmed}</span>}
          </motion.span>
        );
      })}
    </div>
  );
}

function ScoreBoard({ r, phase, k, animKey, expected }: { r: SelectionResult; phase: Phase; k: number; animKey: string; expected: string | null }) {
  const p = PHASES.indexOf(phase);
  const ranked = p >= PHASES.indexOf("rank");
  const order = ranked ? r.ranked.map((n) => r.scores.find((s) => s.name === n)!) : r.scores;
  const compact = r.scores.length > 8;
  return (
    <div className="space-y-1">
      <div className="grid grid-cols-[150px_minmax(0,1fr)_34%_48px] gap-2 px-1.5 text-[10px] uppercase tracking-wide text-muted">
        <span>tool</span><span>description the model sees</span><span className="flex justify-between"><span>−1</span><span>0</span><span>+1.3</span></span><span className="text-right">score</span>
      </div>
      {order.map((s) => {
        const pos = r.scores.indexOf(s);
        const d = (T.lead + pos * T.rowGap) * k;
        const win = phase === "pick" && s.name === r.winner;
        const exp = phase === "pick" && expected === s.name && s.name !== r.winner;
        return (
          <motion.div key={s.name} layout="position" transition={{ type: "spring", stiffness: 170, damping: 24, delay: ranked && phase === "rank" ? 0.2 * k : 0 }}
            className={cx("grid grid-cols-[150px_minmax(0,1fr)_34%_48px] items-center gap-2 rounded-md border px-1.5 py-1", win ? "border-accent bg-accent-soft" : exp ? "border-warn bg-warn-soft" : "border-transparent")}>
            <span className="flex items-center gap-1 truncate font-mono text-xs font-semibold">
              {win && <CheckCircle2 size={12} className="shrink-0 text-accent" aria-label="selected" />}{s.name}
            </span>
            <span className={cx("text-[11px] leading-snug text-muted", compact && "truncate")}>
              {s.words.map((wd, i) => (
                <motion.span key={`${animKey}-${i}`} className={cx("rounded-sm", !wd.kept && "line-through opacity-40")}
                  initial={false}
                  animate={{
                    backgroundColor: wd.matched && p >= 2 ? "var(--accent-soft)" : wd.excluded && p >= 4 ? "var(--err-soft)" : "rgba(0,0,0,0)",
                    color: wd.matched && p >= 2 ? "var(--accent)" : wd.excluded && p >= 4 ? "var(--err)" : "var(--muted)",
                  }}
                  transition={{ delay: (wd.matched && phase === "overlap") || (wd.excluded && phase === "penalty") ? d : 0, duration: 0.3 * k }}>
                  {wd.w}{" "}
                </motion.span>
              ))}
            </span>
            <ScoreBar s={s} p={p} phase={phase} d={d} k={k} />
            <span className="text-right font-mono text-xs">
              {p >= PHASES.indexOf("rank") ? <AnimatedNumber key={phase === "rank" ? animKey : "static"} value={s.total} from={phase === "rank" ? 0 : undefined} delay={0.1 * k} duration={0.6 * k} /> : <span className="text-muted">…</span>}
            </span>
          </motion.div>
        );
      })}
    </div>
  );
}

function ScoreBar({ s, p, phase, d, k }: { s: ToolScore; p: number; phase: Phase; d: number; k: number }) {
  const ov = pct(WEIGHTS.overlap * s.overlap), ar = pct(WEIGHTS.args * s.argFit), pe = pct(WEIGHTS.penalty * s.penalty);
  const grow = (on: boolean, now: Phase) => ({ delay: on && phase === now ? d : 0, duration: (phase === now ? T.grow : 0.3) * k, ease });
  return (
    <div className="relative h-4 rounded bg-surface-2">
      <div className="absolute inset-y-0 w-px bg-fg/40" style={{ left: `${Z}%` }} />
      <motion.div className="absolute inset-y-0.5 rounded-sm bg-accent/80" style={{ left: `${Z}%` }} initial={false} animate={{ width: p >= 2 ? `${ov}%` : "0%" }} transition={grow(true, "overlap")} />
      <motion.div className="absolute inset-y-0.5 rounded-sm bg-violet/80" initial={false} animate={{ left: `${Z + (p >= 2 ? ov : 0)}%`, width: p >= 3 ? `${ar}%` : "0%" }} transition={grow(true, "args")} />
      <motion.div className="absolute inset-y-0.5 rounded-sm bg-err/70" style={{ right: `${100 - Z}%` }} initial={false} animate={{ width: p >= 4 ? `${pe}%` : "0%" }} transition={grow(true, "penalty")} />
      {p >= 5 && (
        <motion.div className="absolute -top-0.5 -bottom-0.5 w-1 rounded bg-fg" initial={{ left: `${Z}%`, opacity: 0 }} animate={{ left: `calc(${Z + pct(s.total)}% - 2px)`, opacity: 1 }}
          transition={{ duration: 0.6 * k, ease, delay: phase === "rank" ? 0.1 * k : 0 }} aria-label={`total ${s.total.toFixed(2)}`} />
      )}
    </div>
  );
}

function PhaseCard({ phase, r, k, expected, acc }: { phase: Phase; r: SelectionResult; k: number; expected: string | null; acc: ReturnType<typeof accuracyOnSamples> }) {
  const at = (s: number) => ({ delay: s * k, duration: 0.4 * k, ease });
  const byName = (n: string) => r.scores.find((s) => s.name === n)!;
  const focus = [...new Set([r.winner, ...(expected ? [expected] : []), ...(r.runnerUp ? [r.runnerUp] : [])])].map(byName);
  switch (phase) {
    case "request":
      return (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-muted">Content words to match:</span>
          {r.queryTokens.length === 0 && <span className="text-xs text-warn">none: every word is a stopword, so no tool can match on keywords.</span>}
          {r.queryTokens.map((t, i) => (
            <motion.span key={t} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={at(1 + i * 0.2)} className="rounded-md border border-accent/40 bg-accent-soft px-2 py-0.5 font-mono">{t}</motion.span>
          ))}
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(1.6)} className="w-full text-xs text-muted">Filler words ("how", "do", "the") are dropped, and plurals are reduced (days → day) so they match the descriptions.</motion.p>
        </div>
      );
    case "catalogue": {
      const used = Math.min(r.totalWords, 400);
      const exp = expected ? byName(expected) : byName(r.winner);
      return (
        <div className="space-y-2 text-xs">
          <div className="flex items-baseline justify-between"><span className="font-medium">Description words in the prompt</span><span className="font-mono"><AnimatedNumber value={r.totalWords} from={0} digits={0} delay={0.2 * k} duration={1 * k} /> / {r.budget}</span></div>
          <div className="relative h-5 overflow-hidden rounded bg-surface-2">
            <motion.div className={cx("absolute inset-y-0 left-0", r.wordsPerTool ? "bg-err/60" : "bg-ok/60")} initial={{ width: 0 }} animate={{ width: `${(used / Math.max(used, r.budget)) * 100}%` }} transition={{ delay: 0.2 * k, duration: 1 * k, ease }} />
            <div className="absolute inset-y-0 w-0.5 bg-fg" style={{ left: `${(r.budget / Math.max(used, r.budget)) * 100}%` }} />
          </div>
          {r.wordsPerTool ? (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={at(1.3)} className="space-y-1">
              <p><StatusBadge status="warning" label="over budget" /> Each of the {r.scores.length} descriptions keeps its first <strong>{r.wordsPerTool}</strong> words. What {exp.name} loses:</p>
              <p className="font-mono">{exp.words.map((wd, i) => <span key={i} className={cx(!wd.kept && "text-err line-through")}>{wd.w} </span>)}</p>
            </motion.div>
          ) : <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(1.3)}><StatusBadge status="ok" label="fits" /> All {r.scores.length} descriptions are sent in full.</motion.p>}
        </div>
      );
    }
    case "overlap":
    case "penalty":
      return (
        <div className="space-y-1.5 text-xs">
          {focus.map((s, i) => (
            <motion.div key={s.name} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={at(0.3 + i * 0.35)} className="flex flex-wrap items-center gap-1.5">
              <span className="w-36 truncate font-mono font-semibold">{s.name}</span>
              {phase === "overlap" ? (
                <>
                  {r.queryTokens.map((t) => <Badge key={t} tone={s.matched.includes(t) ? "accent" : "neutral"}>{s.matched.includes(t) ? "✓" : "✗"} {t}</Badge>)}
                  <ArrowRight size={12} className="text-muted" /><span className="font-mono">{s.matched.length}/{r.queryTokens.length} = {s.overlap.toFixed(2)}</span>
                </>
              ) : s.penalty ? <><Badge tone="err">excludes {s.exclusionHits.join(", ")}</Badge><span className="font-mono">−{WEIGHTS.penalty}</span></>
                : <span className="text-muted">no exclusion hit{s.words.some((wd) => !wd.kept && wd.w === "Not") ? " (its 'Not for' clause was cut by the budget)" : ""}</span>}
            </motion.div>
          ))}
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(1.5)} className="text-muted">
            {phase === "overlap" ? `Showing the tools that end up competing; bars above show all ${r.scores.length}.` : "A visible 'Not for' clause pushes a tool below zero for requests it should not handle."}
          </motion.p>
        </div>
      );
    case "args":
      return (
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">tool</th><th>required argument</th><th>found in request</th><th className="text-right">fit</th></tr></thead>
          <tbody>
            {focus.flatMap((s, i) => (s.filled.length ? s.filled : [{ name: "(none)", value: "—" }]).map((f, j) => (
              <motion.tr key={`${s.name}-${f.name}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.3 + (i * 2 + j) * 0.25)} className="border-t border-line">
                <td className="py-1 font-mono">{j === 0 ? s.name : ""}</td><td className="font-mono">{f.name}</td>
                <td className="font-mono">{f.value === null ? <span className="text-err">✗ missing</span> : <span className="text-ok">✓ {f.value.length > 28 ? `${f.value.slice(0, 28)}…` : f.value}</span>}</td>
                <td className="text-right font-mono">{j === 0 ? s.argFit.toFixed(2) : ""}</td>
              </motion.tr>
            )))}
          </tbody>
        </table>
      );
    case "rank":
      return (
        <div className="space-y-1.5 text-xs">
          {r.ranked.slice(0, 3).map((n, i) => {
            const s = byName(n);
            return (
              <motion.div key={n} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={at(0.4 + i * 0.25)} className="flex items-center gap-2">
                <span className="w-5 font-mono text-muted">#{i + 1}</span><span className="w-40 truncate font-mono font-semibold">{n}</span>
                <span className="font-mono"><AnimatedNumber value={s.total} from={0} delay={(0.4 + i * 0.25) * k} duration={0.6 * k} /></span>
              </motion.div>
            );
          })}
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(1.3)}>
            {r.tie ? <Callout tone="warn" title="Tie">The top two score exactly the same. The one listed first wins, which is an accident of ordering, not understanding.</Callout>
              : <p>Margin over second place: <strong className="font-mono">{r.margin.toFixed(2)}</strong>{r.margin < 0.2 ? " (fragile)" : ""}.</p>}
          </motion.div>
        </div>
      );
    case "pick": {
      const ok = expected ? r.winner === expected : null;
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={at(0.3)}>
              <Formula>{JSON.stringify({ name: r.call.name, arguments: r.call.arguments }, null, 2)}</Formula>
            </motion.div>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(0.9)}>
              {ok === null ? <Badge>custom request: no expected tool to compare</Badge> : ok ? <StatusBadge status="ok" label={`correct: ${expected}`} /> : <StatusBadge status="error" label={`wrong tool: expected ${expected}`} />}
            </motion.div>
          </div>
          <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={at(1.2)} className="text-xs">
            <div className="mb-1 font-medium">Same catalogue on all 4 sample requests: {acc.correct}/{acc.total}</div>
            <ul className="space-y-0.5">
              {acc.rows.map((row) => (
                <li key={row.id} className="flex items-center gap-1.5">
                  {row.picked === row.expected ? <CheckCircle2 size={12} className="text-ok" aria-label="correct" /> : <XCircle size={12} className="text-err" aria-label="wrong" />}
                  <span className="font-mono">{row.id}</span><span className="text-muted">→ {row.picked}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
      );
    }
  }
}
