import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, Repeat, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../neurons-and-activation-functions/kit";
import {
  PROMPT_PRESETS, SMOOTHING_LABEL, occurrences, simulate, train,
  type GenState, type Model, type Order, type Prediction, type Smoothing,
} from "./simulation";

const MODEL = train();

/* one schedule (seconds at 1×) for the sequence, corpus highlights, bars and card */
const T = { ctx: 0.1, match: 0.5, matchStep: 0.1, count: 1.3, countStep: 0.06, prob: 2.3, pick: 3.0, fly: 3.3 };
const DUR = { count: 3200, predict: 4500, stuck: 3000, done: 3200 };
const TOPN = 8;

const tx = (s: string) => `\\text{${s.replace(/[^a-z.,!? ]/g, "")}}`;

function narrate(s: GenState, order: Order) {
  if (s.phase === "count") return { headline: `Training = counting which word follows which ${order === 2 ? "word" : "pair of words"}`, sub: "No neural network: the whole model is a table of counts from the corpus below." };
  if (s.phase === "done") return s.loopAt !== null
    ? { headline: "The text is stuck in a loop", sub: "Same context in → same greedy pick out. Once a context repeats, the cycle never ends." }
    : { headline: "Generation finished", sub: "Every word came from counts → probabilities → pick the top one → append → repeat." };
  const p = s.pred!;
  const ctx = p.wanted.join(" ");
  if (s.phase === "stuck") return { headline: `"${ctx}" never appears in the corpus, so there is nothing to predict`, sub: "With no smoothing or backoff every count is 0. Switch the unseen-context setting to fix it." };
  if (p.unseen && p.smoothed) return { headline: `"${ctx}" was never seen: add-one smoothing makes every word almost equally likely`, sub: "All counts are 0, so every word gets 1/(0+V). The tie is broken by corpus order: junk, but not a crash." };
  if (p.unseen) return { headline: `"${ctx}" was never seen, so the model backs off to ${p.usedOrder === 1 ? "plain word frequency" : `"${p.context.join(" ")}"`}`, sub: "A shorter context has counts. Its distribution is used instead." };
  return { headline: `What follows "${ctx}"? Count, divide, pick the top word`, sub: `Found ${p.total} times in the corpus. The most likely next word slides into the sequence.` };
}

export default function NextTokenVisualizer({ tab }: { tab: VisualizerTab }) {
  const [prompt, setPrompt] = useState(PROMPT_PRESETS[0].prompt);
  const [order, setOrder] = useState<Order>(2);
  const [smoothing, setSmoothing] = useState<Smoothing>("backoff");
  const [n, setN] = useState(8);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate({ prompt, order, smoothing, n }, MODEL), [prompt, order, smoothing, n]);
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "predict"]);
  if (tab === "metrics") return null;

  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${prompt}-${order}-${smoothing}`;
  const story = narrate(s, order);
  const reset = () => { setPrompt(PROMPT_PRESETS[0].prompt); setOrder(2); setSmoothing("backoff"); setN(8); pb.restart(); };

  return (
    <Workspace
      main={
        <>
          <Panel title="N-gram language model" actions={<SimulatedLabel>counting model, not a neural LLM</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((v) => v + 1)} />
            <Sequence s={s} order={order} k={k} animKey={animKey} />
            <CorpusStrip model={MODEL} s={s} order={order} k={k} animKey={animKey} />
            {s.pred && s.phase === "predict" && <Bars pred={s.pred} k={k} animKey={animKey} />}
            {s.phase === "stuck" && <Callout tone="err" title="No prediction">The context "{s.pred!.wanted.join(" ")}" has no counts. Choose “Back off” or “Add-one” below, or change the prompt.</Callout>}
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard s={s} order={order} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Prompt & model" actions={<Button size="sm" variant="ghost" onClick={reset}><RotateCcw size={12} /> Reset</Button>}>
            <div className="space-y-1">
              <label htmlFor="ntp-prompt" className="block text-xs font-medium">Prompt (lower-cased; only letters and . , ! ? are kept)</label>
              <input id="ntp-prompt" value={prompt} maxLength={60} onChange={(e) => setPrompt(e.target.value)} className="h-8 w-full rounded-md border border-line bg-surface px-2 font-mono text-sm" />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Try:</span>
              {PROMPT_PRESETS.map((p) => <Button key={p.label} size="sm" variant={prompt === p.prompt ? "primary" : "secondary"} onClick={() => setPrompt(p.prompt)}>{p.label}</Button>)}
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <Select label="Model order" value={String(order) as "2" | "3"} options={[{ value: "2", label: "Bigram (1 word of context)" }, { value: "3", label: "Trigram (2 words of context)" }]} onChange={(v) => setOrder(Number(v) as Order)} />
              <Select label="Unseen context" value={smoothing} options={(Object.keys(SMOOTHING_LABEL) as Smoothing[]).map((v) => ({ value: v, label: SMOOTHING_LABEL[v] }))} onChange={setSmoothing} />
              <Slider label="Tokens to generate" value={n} min={1} max={16} onChange={setN} />
            </div>
            <p className="mt-2 text-[11px] text-muted">Decoding is greedy: always the most likely word; ties go to the word that appears first in the corpus.</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation s={s} order={order} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["corpus tokens / vocabulary", `${MODEL.tokens.length} / ${MODEL.vocab.length}`],
              ["context asked for", s.pred ? `"${s.pred.wanted.join(" ")}"` : "—"],
              ["context used", s.pred ? (s.pred.context.length ? `"${s.pred.context.join(" ")}"` : "none (unigram)") : "—"],
              ["times context seen", s.pred ? s.pred.total : "—"],
              ["P(picked word)", s.pred?.candidates[0] ? s.pred.candidates[0].p.toFixed(3) : "—"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- sequence strip ---------- */
function Sequence({ s, order, k, animKey }: { s: GenState; order: Order; k: number; animKey: string }) {
  const ctxLen = s.pred ? s.pred.wanted.length : 0;
  const usedLen = s.pred ? s.pred.context.length : 0;
  const tokens = s.seq;
  return (
    <div className="mb-3">
      <div className="mb-1 flex justify-between text-[11px] text-muted"><span>Sequence (prompt in grey, generated in blue)</span><span>{order === 2 ? "bigram" : "trigram"} · greedy</span></div>
      <div className="flex min-h-10 flex-wrap items-center gap-1 rounded-md border border-line p-2">
        {tokens.map((t, i) => {
          const fromPrompt = i < s.prompt.length;
          const inCtx = s.phase !== "done" && s.phase !== "count" && i >= tokens.length - ctxLen;
          const inUsed = inCtx && i >= tokens.length - usedLen;
          const looping = s.phase === "done" && s.loopAt !== null && i >= s.loopAt;
          return (
            <motion.span key={`${i}-${t}`} className={cx("relative rounded border px-1.5 py-0.5 font-mono text-sm", fromPrompt ? "border-line bg-surface-2" : "border-accent/40 bg-accent-soft", looping && "border-warn bg-warn-soft")}
              initial={false} animate={{ opacity: 1 }}>
              {t}
              {inCtx && <motion.span key={animKey} className={cx("pointer-events-none absolute -inset-1 rounded-md border-2", inUsed ? "border-accent" : "border-dashed border-warn")} initial={{ opacity: 0, scale: 1.3 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: T.ctx * k, duration: 0.35 * k }} />}
            </motion.span>
          );
        })}
        {s.phase === "predict" && s.pick && (
          <motion.span key={`${animKey}-new`} className="rounded border border-ok bg-ok-soft px-1.5 py-0.5 font-mono text-sm font-semibold"
            initial={{ opacity: 0, y: 70, scale: 0.7 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ delay: T.fly * k, duration: 0.6 * k, ease: EASE }}>
            {s.pick}
          </motion.span>
        )}
        {s.phase === "predict" && <motion.span key={`${animKey}-q`} className="font-mono text-sm text-muted" initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ delay: T.fly * k, duration: 0.2 * k }}>▢?</motion.span>}
        {s.phase === "stuck" && <Badge tone="err"><XCircle size={11} aria-hidden /> no prediction</Badge>}
        {s.phase === "done" && s.loopAt !== null && <Badge tone="warn"><Repeat size={11} aria-hidden /> loop</Badge>}
      </div>
    </div>
  );
}

/* ---------- corpus strip: where the counts come from ---------- */
function CorpusStrip({ model, s, order, k, animKey }: { model: Model; s: GenState; order: Order; k: number; animKey: string }) {
  const ctx = s.pred?.context ?? [];
  const occ = s.pred && s.phase === "predict" ? occurrences(model, ctx) : [];
  const role = new Map<number, { kind: "ctx" | "next"; j: number }>();
  if (ctx.length) occ.forEach((p, j) => { ctx.forEach((_, c) => role.set(p + c, { kind: "ctx", j })); role.set(p + ctx.length, { kind: "next", j }); });
  const counting = s.phase === "count";
  return (
    <div className="mb-3">
      <div className="mb-1 flex flex-wrap justify-between gap-2 text-[11px] text-muted">
        <span>Training corpus ({model.tokens.length} tokens)</span>
        {s.phase === "predict" && (ctx.length ? <span><span className="rounded bg-accent-soft px-1">context</span> → <span className="rounded bg-ok-soft px-1">what came next</span> · {occ.length} matches</span> : <span>no context: every token counts (unigram)</span>)}
      </div>
      <div className="flex flex-wrap gap-0.5 rounded-md bg-surface-2 p-1.5">
        {model.tokens.map((t, i) => {
          const r = role.get(i);
          const at = r ? (T.match + Math.min(r.j, 10) * T.matchStep) * k : 0;
          return (
            <span key={i} className="relative rounded px-1 font-mono text-[11px] leading-5 text-muted">
              {r && <motion.span key={animKey} className={cx("absolute inset-0 rounded", r.kind === "ctx" ? "bg-accent/25" : "bg-ok/30")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: at, duration: 0.25 * k }} />}
              {counting && order === 2 && i > 0 && <motion.span key={`${animKey}-c`} className="absolute inset-0 rounded bg-accent/20" initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 0] }} transition={{ delay: (0.3 + i * 0.04) * k, duration: 0.5 * k }} />}
              {counting && order === 3 && i > 1 && <motion.span key={`${animKey}-c`} className="absolute inset-0 rounded bg-accent/20" initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 0] }} transition={{ delay: (0.3 + i * 0.04) * k, duration: 0.5 * k }} />}
              <span className={cx("relative", r && "text-fg")}>{t}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

/* ---------- next-token bars: counts first, then rescaled to probabilities ---------- */
function Bars({ pred, k, animKey }: { pred: Prediction; k: number; animKey: string }) {
  const shown = pred.candidates.slice(0, TOPN);
  const others = pred.candidates.slice(TOPN);
  const plus = pred.smoothed ? 1 : 0;
  const maxC = Math.max(1, ...shown.map((c) => c.count + plus));
  const D = T.prob + 0.5 - T.count;
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px] text-muted"><span>Next-word candidates: {pred.smoothed ? "count + 1" : "count"}, then probability</span><span>P(next | context)</span></div>
      <div className="space-y-1">
        {shown.map((c, i) => {
          const a = ((c.count + plus) / maxC) * 100;
          const b = c.p * 100;
          const at = (T.count + i * T.countStep) * k;
          return (
            <div key={`${animKey}-${c.token}`} className="relative grid grid-cols-[64px_minmax(0,1fr)_56px] items-center gap-2">
              <span className={cx("truncate font-mono text-xs", i === 0 && "font-semibold")}>{c.token}</span>
              <div className="relative h-5 rounded bg-surface-2">
                <motion.div className={cx("absolute inset-y-0 left-0 rounded", i === 0 ? "bg-accent" : "bg-muted/45")} initial={{ width: "0%" }}
                  animate={{ width: ["0%", `${a}%`, `${a}%`, `${b}%`] }} transition={{ delay: at, duration: D * k, times: [0, 0.3, (T.prob - T.count) / D, 1], ease: "easeInOut" }} />
                {i === 0 && <motion.div className="absolute -inset-0.5 rounded border-2 border-ok" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.pick * k, duration: 0.3 * k }} />}
              </div>
              <span className="relative h-4 text-right font-mono text-xs">
                <motion.span className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 1, 0] }} transition={{ delay: at, duration: (T.prob - T.count + 0.2) * k, times: [0, 0.2, 0.85, 1] }}>
                  ×{c.count + plus}
                </motion.span>
                <motion.span className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.prob + 0.2) * k, duration: 0.3 * k }}>{c.p.toFixed(3)}</motion.span>
              </span>
            </div>
          );
        })}
        {others.length > 0 && <p className="text-[11px] text-muted">+ {others.length} more words sharing {(others.reduce((s, c) => s + c.p, 0) * 100).toFixed(1)}% probability</p>}
      </div>
    </div>
  );
}

/* ---------- card ---------- */
function StepCard({ s, order, k }: { s: GenState; order: Order; k: number }) {
  if (s.phase === "count") {
    const contexts = [...MODEL.counts.entries()].filter(([c]) => c && c.split(" ").length === order - 1).slice(0, 8);
    return (
      <div>
        <p className="mb-2 text-xs">The trained model is this table: for each context, how often each next word followed it.</p>
        <div className="space-y-1">
          {contexts.map(([c, m], i) => (
            <motion.div key={c} className="flex flex-wrap items-center gap-1 text-xs" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.22) * k, duration: 0.3 * k }}>
              <span className="w-24 shrink-0 truncate rounded bg-accent-soft px-1 font-mono">{c}</span>→
              {[...m].sort((a, b) => b[1] - a[1]).map(([t, v]) => <span key={t} className="rounded border border-line px-1 font-mono">{t} <span className="text-muted">×{v}</span></span>)}
            </motion.div>
          ))}
          <p className="text-[11px] text-muted">… {[...MODEL.counts.keys()].filter((c) => c && c.split(" ").length === order - 1).length} contexts in total.</p>
        </div>
      </div>
    );
  }
  if (s.phase === "stuck") return <Callout tone="err" title="Unseen context, no fallback">A real model must always produce a distribution. Count-based models need smoothing (give unseen events a little probability) or backoff (use a shorter context). Neural LMs avoid this because similar contexts share learned representations.</Callout>;
  if (s.phase === "done") {
    const gen = s.seq.slice(s.prompt.length);
    return (
      <div className="space-y-2 text-xs">
        <div className="flex flex-wrap gap-3">
          <Stat label="tokens generated" v={gen.length} k={k} />
          <Stat label="distinct generated words" v={new Set(gen).size} k={k} />
        </div>
        {s.loopAt !== null ? (
          <Callout tone="warn" title="Repetition loop"><Repeat size={12} className="mr-1 inline" aria-hidden />From word {s.loopAt + 1} the same contexts recur, so greedy decoding repeats itself forever. Remedies: sampling with temperature (next topic), repetition penalties, or a model with longer context.</Callout>
        ) : <p className="text-muted">No context repeated within {gen.length} tokens. Generate more tokens to see whether it eventually loops.</p>}
      </div>
    );
  }
  const p = s.pred!;
  const shown = p.candidates.slice(0, 6);
  const V = MODEL.vocab.length;
  return (
    <div className="grid gap-4 sm:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">next word</th><th className="text-right">count</th><th className="text-right">÷ total</th><th className="text-right">= P</th></tr></thead>
        <tbody className="font-mono">
          {shown.map((c, i) => (
            <motion.tr key={c.token} className={cx("border-t border-line", i === 0 && "font-semibold")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.count + i * T.countStep) * k }}>
              <td className="py-0.5">{c.token}</td>
              <td className="text-right">{p.smoothed ? `${c.count}+1` : c.count}</td>
              <td className="text-right text-muted">÷ {p.smoothed ? `${p.total}+${V}` : p.total}</td>
              <td className="text-right"><motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.prob + 0.2) * k }}>{c.p.toFixed(3)}</motion.span></td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      <div className="space-y-2 text-xs">
        {p.unseen && <p className="flex gap-1 rounded-md bg-warn-soft px-2 py-1"><AlertTriangle size={13} className="mt-0.5 shrink-0 text-warn" aria-hidden />{p.smoothed ? `"${p.wanted.join(" ")}" never occurs: with add-one, every one of the ${V} words gets (0+1)/(0+${V}).` : `"${p.wanted.join(" ")}" never occurs, so the model used ${p.usedOrder === 1 ? "word frequencies with no context" : `the shorter context "${p.context.join(" ")}"`}.`}</p>}
        <p>Probabilities sum to 1 across all next words. Greedy decoding takes the top one: <strong className="font-mono">{p.candidates[0]?.token}</strong>.</p>
        <motion.p className="text-muted" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.fly * k }}>A neural LLM does the same last step (a distribution over the vocabulary, then pick), but its probabilities come from a network, not a lookup table.</motion.p>
      </div>
    </div>
  );
}

function Stat({ label, v, k }: { label: string; v: number; k: number }) {
  return <div className="rounded-md border border-line px-3 py-1.5"><div className="font-mono text-lg"><AnimatedNumber value={v} from={0} digits={0} delay={0.2 * k} /></div><div className="text-[11px] text-muted">{label}</div></div>;
}

function Calculation({ s, order }: { s: GenState; order: Order }) {
  const V = MODEL.vocab.length;
  const general = order === 2 ? "P(w_t \\mid w_{t-1}) = \\frac{C(w_{t-1}\\, w_t)}{C(w_{t-1})}" : "P(w_t \\mid w_{t-2} w_{t-1}) = \\frac{C(w_{t-2}\\, w_{t-1}\\, w_t)}{C(w_{t-2}\\, w_{t-1})}";
  if (!s.pred || !s.pred.candidates[0]) return <TexBlock>{s.phase === "stuck" ? `${general} = \\frac{0}{0}\\ \\text{(undefined)}` : general}</TexBlock>;
  const p = s.pred;
  const top = p.candidates[0];
  const ctx = p.context.length ? tx(p.context.join(" ")) : "\\varnothing";
  const rhs = p.smoothed ? `\\frac{${top.count} + 1}{${p.total} + ${V}}` : `\\frac{${top.count}}{${p.total}}`;
  return (
    <div className="space-y-2">
      <TexBlock>{p.smoothed ? `P_{+1}(w \\mid c) = \\frac{C(c\\, w) + 1}{C(c) + V}` : general}</TexBlock>
      <TexBlock>{`P(${tx(top.token)} \\mid ${ctx}) = ${rhs} = ${top.p.toFixed(3)}`}</TexBlock>
      {p.unseen && !p.smoothed && <p className="text-[11px] text-muted">Backed off from a {order}-gram to a {p.usedOrder}-gram.</p>}
    </div>
  );
}
