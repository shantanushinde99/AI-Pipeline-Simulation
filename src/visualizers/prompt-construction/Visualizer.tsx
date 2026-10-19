import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { QUESTIONS } from "../rag/corpus";
import { StepCaption } from "../similarity-search/StepCaption";
import type { VisualizerTab } from "../types";
import { DEFAULT_PC, VARIANTS, simulate, type Gen, type PCConfig, type PCState, type Slot, type SlotKind, type Variant } from "./simulation";

const ease = [0.22, 1, 0.36, 1] as const;
/** One schedule (seconds at 1×): the slot slides in, then the token counter and budget bar grow together. Answer step: before → after → highlight. */
const T = { enter: 0.7, count: 0.5, countDur: 0.6, before: 0.3, after: 1.1, mark: 1.8 };
const DUR = { slot: 2000, answer: 3800 };
const COLOR: Record<SlotKind, string> = { system: "var(--muted)", grounding: "var(--ok)", citation: "var(--violet)", passage: "var(--accent)", question: "var(--warn)", output: "var(--fg)" };
const WHY: Record<SlotKind, string> = {
  system: "Sets who the model is. Weak on its own: it says nothing about where answers must come from.",
  grounding: "Tells the model to use only the passages and to admit when they lack the answer. Without it, memorised facts leak in.",
  citation: "Defines how to point back at evidence. No format, no citations, so nobody can check the answer.",
  passage: "Retrieved evidence, numbered so it can be cited. Passages are data from documents, which may be untrusted.",
  question: "The user's question. Placing it after the evidence lets the model read the passages with the question fresh.",
  output: "Shape of the reply (one sentence + citation), so downstream code can parse it.",
};
const colorOf = (s: Slot) => (s.injected ? "var(--err)" : COLOR[s.kind]);

export default function PromptConstructionVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<PCConfig>(DEFAULT_PC);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length, (i) => DUR[sim.steps[i]?.state.phase ?? "slot"]);
  if (tab === "metrics") return null;
  const step = sim.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(cfg)}`;
  const prevUsed = pb.index > 0 ? sim.steps[pb.index - 1].state.used : 0;
  const set = (p: Partial<PCConfig>) => setCfg((c) => ({ ...c, ...p }));
  const story = narrate(s, sim);
  const B = sim.cfg.budget;

  return (
    <Workspace
      main={
        <>
          <Panel title="Building the prompt" actions={<SimulatedLabel>Rule-based model, toy tokens</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="mb-1 flex justify-between text-xs">
              <span className="font-medium">Token budget</span>
              <span className="font-mono"><AnimatedNumber key={animKey} value={s.used} from={s.phase === "slot" ? prevUsed : s.used} digits={0} delay={T.count * k} duration={T.countDur * k} /> / {B}</span>
            </div>
            <div className="flex h-6 overflow-hidden rounded-md border border-line bg-surface-2" aria-label="Budget meter">
              {sim.slots.slice(0, s.shown).filter((x) => x.kept).map((x) => {
                const isNew = s.phase === "slot" && x.id === s.slot?.id;
                return (
                  <motion.div key={isNew ? `${x.id}-${animKey}` : x.id} className="h-full shrink-0 border-r border-surface" style={{ background: colorOf(x) }} title={`${x.label}: ${x.tokens}`}
                    initial={isNew ? { width: 0 } : false} animate={{ width: `${(x.tokens / B) * 100}%` }} transition={{ delay: isNew ? T.count * k : 0, duration: T.countDur * k, ease }} />
                );
              })}
            </div>
            <div className="mt-3 space-y-1.5">
              {sim.slots.slice(0, s.shown).map((x, i) => <SlotCard key={x.id} slot={x} i={i} isNew={s.phase === "slot" && x.id === s.slot?.id} mark={s.phase === "answer" ? markOf(x, sim.after) : null} k={k} animKey={animKey} />)}
            </div>
            {s.phase === "answer" && (
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                <AnswerCard title="Before: complete prompt" g={sim.before} delay={T.before * k} animKey={animKey} />
                <AnswerCard title={`After: ${VARIANTS[sim.cfg.variant]}`} g={sim.after} delay={T.after * k} animKey={animKey} />
              </div>
            )}
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                <StepCard s={s} sim={sim} prevUsed={prevUsed} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_PC); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Question" value={cfg.questionId} onChange={(v) => { set({ questionId: v }); pb.restart(); }} options={QUESTIONS.map((q) => ({ value: q.id, label: q.text }))} />
              <Select label="Prompt variant (break it)" value={cfg.variant} onChange={(v: Variant) => { set({ variant: v }); pb.restart(); }} options={(Object.keys(VARIANTS) as Variant[]).map((v) => ({ value: v, label: VARIANTS[v] }))} />
              <Slider label="Token budget" value={cfg.budget} min={80} max={420} step={10} onChange={(v) => set({ budget: v })} hint="Fixed slots are reserved first; passages that no longer fit are dropped." />
              <Toggle label="Add “passages are data, not instructions” to the grounding rule" checked={cfg.guard} onChange={(v) => set({ guard: v })} hint="The fix for the injected-passage variant." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why="The model only sees this one string. Every rule, every passage and their order decide what it can and will answer." />
          <Panel title="Calculation">
            <TexBlock>{`\\begin{aligned}
t_{\\text{slot}} &\\approx \\operatorname{round}(1.3 \\times \\text{words}) \\\\
T &= ${sim.slots.slice(0, s.shown).filter((x) => x.kept).map((x) => x.tokens).join(" + ") || "0"} \\\\
&= ${s.used} \\le B = ${B}
\\end{aligned}`}</TexBlock>
          </Panel>
          <Panel title="Key values">
            <KV rows={[["question", sim.q.id], ["variant", VARIANTS[sim.cfg.variant]], ["prompt tokens", `${sim.total} / ${B}`], ["passages kept", `${sim.slots.filter((x) => x.kind === "passage" && x.kept).length} / ${sim.slots.filter((x) => x.kind === "passage").length}`], ["expected fact", sim.q.expect], ["answer correct", sim.after.correct ? "yes" : "no"]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

type Sim = ReturnType<typeof simulate>;
const markOf = (x: Slot, g: Gen): "used" | "attack" | null => (g.from === "injection" && x.injected ? "attack" : g.from === "passage" && x.text.includes(g.sentence) && x.kind === "passage" ? "used" : null);

function narrate(s: PCState, sim: Sim) {
  if (s.phase === "answer") {
    const a = sim.after;
    if (sim.cfg.variant === "complete") return { headline: a.correct ? "Complete prompt: correct, cited answer" : "Even the complete prompt fails here", sub: a.rule };
    const same = a.text === sim.before.text;
    return { headline: same ? `${VARIANTS[sim.cfg.variant]}: the answer did not change this time` : `${VARIANTS[sim.cfg.variant]}: the answer changed`, sub: a.rule };
  }
  const x = s.slot!;
  if (!x.kept) return { headline: `${x.label} does not fit: dropped`, sub: `${x.tokens} tokens would push the prompt past ${sim.cfg.budget}. Its evidence never reaches the model.` };
  if (x.injected) return { headline: `${x.label} carries a hidden command`, sub: "Retrieved text from a forum post. To the model it looks just like an instruction." };
  return { headline: `${x.label} slides in: +${x.tokens} tokens`, sub: WHY[x.kind] };
}

function SlotCard({ slot, i, isNew, mark, k, animKey }: { slot: Slot; i: number; isNew: boolean; mark: "used" | "attack" | null; k: number; animKey: string }) {
  return (
    <motion.div key={isNew ? `${slot.id}-${animKey}` : slot.id} initial={isNew ? { x: -60, opacity: 0 } : false} animate={{ x: slot.kept ? 0 : [0, 0, 24], opacity: slot.kept ? 1 : [1, 1, 0.45] }}
      transition={{ duration: (slot.kept ? T.enter : T.enter + 0.8) * k, ease }}
      className={cx("relative flex items-start gap-2 rounded-md border bg-surface-2 py-1 pr-2 pl-3 text-xs", mark === "used" ? "border-ok ring-2 ring-ok" : mark === "attack" ? "border-err ring-2 ring-err" : "border-line", !slot.kept && "line-through")}>
      <span className="absolute inset-y-0 left-0 w-1.5 rounded-l-md" style={{ background: colorOf(slot) }} />
      <span className="w-4 shrink-0 font-mono text-muted">{i + 1}</span>
      <div className="min-w-0 flex-1">
        <p className="font-semibold">{slot.label}{slot.injected && <Badge tone="err" className="ml-1">untrusted</Badge>}{!slot.kept && <Badge tone="warn" className="ml-1">dropped: over budget</Badge>}</p>
        <p className="line-clamp-2 break-words text-muted">{slot.text}</p>
      </div>
      <span className="shrink-0 font-mono">{slot.tokens}t</span>
      {mark && (
        <motion.span key={`m-${animKey}`} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: T.mark * k, type: "spring", stiffness: 300, damping: 18 }} className={cx("absolute -top-2 right-1 rounded px-1 text-[10px] font-semibold text-white dark:text-black", mark === "used" ? "bg-ok" : "bg-err")}>
          {mark === "used" ? "answer came from here" : "model obeyed this"}
        </motion.span>
      )}
    </motion.div>
  );
}

function AnswerCard({ title, g, delay, animKey }: { title: string; g: Gen; delay: number; animKey: string }) {
  return (
    <motion.div key={`${title}-${animKey}`} initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay, duration: 0.5, ease }} className="rounded-md border border-line p-2 text-xs">
      <p className="mb-1 font-semibold">{title}</p>
      <p className="font-mono">{g.text}</p>
      <div className="mt-1.5 flex flex-wrap gap-1">
        <Check ok={g.correct} label="correct fact" />
        <Check ok={g.citationOk} label="valid citation" />
      </div>
    </motion.div>
  );
}

const Check = ({ ok, label }: { ok: boolean; label: string }) => (
  <Badge tone={ok ? "ok" : "err"}>{ok ? <CheckCircle2 size={11} /> : <XCircle size={11} />} {label}{ok ? "" : ": no"}</Badge>
);

function StepCard({ s, sim, prevUsed, k }: { s: PCState; sim: Sim; prevUsed: number; k: number }) {
  const B = sim.cfg.budget;
  if (s.phase === "answer")
    return (
      <div className="space-y-2 text-xs">
        <table className="w-full table-fixed border-collapse">
          <thead><tr className="text-left text-muted"><th className="w-24 py-1">Check</th><th>Complete prompt</th><th>{VARIANTS[sim.cfg.variant]}</th></tr></thead>
          <tbody>
            {([["answer from", sim.before.from, sim.after.from], ["correct fact", sim.before.correct ? "✓ yes" : "✗ no", sim.after.correct ? "✓ yes" : "✗ no"], ["citation", sim.before.cited ? `[${sim.before.cited}] ${sim.before.citationOk ? "✓" : "✗"}` : "✗ none", sim.after.cited ? `[${sim.after.cited}] ${sim.after.citationOk ? "✓" : "✗"}` : "✗ none"]] as const).map(([h, a, b], i) => (
              <motion.tr key={h} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (T.after + i * 0.2) * k }} className="border-t border-line">
                <td className="py-1 font-medium">{h}</td><td className="break-words">{a}</td><td className="break-words">{b}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
        <p className="text-muted">Why: {sim.after.rule}</p>
      </div>
    );
  const x = s.slot!;
  return (
    <div className="space-y-2 text-xs">
      <p><span className="rounded px-1.5 py-0.5 font-semibold text-white dark:text-black" style={{ background: colorOf(x) }}>{x.label}</span> <span className="text-muted">{WHY[x.kind]}</span></p>
      <p className="rounded border border-line bg-surface-2 p-2 font-mono break-words">{x.text}</p>
      <p className="font-mono">{x.kept ? `${prevUsed} + ${x.tokens} = ${s.used} ≤ ${B}` : `${prevUsed} + ${x.tokens} would leave no room for the question and format (budget ${B}): dropped`}</p>
      <div className="relative h-3 overflow-hidden rounded bg-surface-2">
        <div className="absolute inset-y-0 left-0 bg-line" style={{ width: `${Math.min(100, (prevUsed / B) * 100)}%` }} />
        <motion.div className="absolute inset-y-0" style={{ left: `${Math.min(100, (prevUsed / B) * 100)}%`, background: x.kept ? colorOf(x) : "var(--err)" }} initial={{ width: 0 }} animate={{ width: `${Math.min(100 - (prevUsed / B) * 100, (x.tokens / B) * 100)}%` }} transition={{ delay: T.count * k, duration: T.countDur * k, ease }} />
      </div>
    </div>
  );
}
