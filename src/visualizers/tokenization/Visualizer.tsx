import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption, useFrames } from "../neurons-and-activation-functions/kit";
import {
  FIRST_MERGE_ID, MAX_TEXT, TEXT_PRESETS, UNK_ID, countWords, simulate, trainBPE,
  type Merge, type TokState, type Token,
} from "./simulation";

/* one schedule (seconds at 1×) for canvas + card */
const M = { bars: 0, stagger: 0.08, win: 1.0, glow: 1.2, join: 1.6, vocab: 2.1 };
const FRAME_MS = 450;
const ENC_DELAY = 500;
const DEFAULT_MERGES = 16;
const MAX_MERGES = trainBPE(200).merges.length;

const show = (s: string) => s.replace(/ /g, "␣");
const hue = (id: number) => (id * 67) % 360;
const chipStyle = (id: number) => (id === UNK_ID ? { background: "var(--err-soft)", borderColor: "var(--err)" } : { background: `hsl(${hue(id)} 75% 55% / 0.2)`, borderColor: `hsl(${hue(id)} 60% 45% / 0.55)` });

function duration(s: TokState) {
  if (s.phase === "corpus") return 2800;
  if (s.phase === "merge") return 3100;
  if (s.phase === "split") return 2400;
  if (s.phase === "encode") return ENC_DELAY + (s.encoded.frames.length - 1) * FRAME_MS + 1200;
  return 2800;
}

function narrate(s: TokState) {
  const m = s.trained.merges[s.mergeIndex];
  switch (s.phase) {
    case "corpus": return { headline: "Training starts from single characters", sub: "Every word in the corpus is split into characters. A leading space ␣ stays attached to its word." };
    case "merge": return { headline: `"${show(m.pair[0])}" + "${show(m.pair[1])}" is the most common pair (${m.count}×), so it becomes one token`, sub: `Count every adjacent pair, merge the winner everywhere, add "${show(m.token)}" to the vocabulary as id ${m.id}.` };
    case "split": return { headline: "Your text is cut into pre-tokens, then characters", sub: "Same rules as training: words keep their leading space, digits and punctuation split off." };
    case "encode": return { headline: `Replaying the learned merges: ${s.encoded.frames[0].tokens.length} characters → ${s.encoded.tokens.length} tokens`, sub: "Merges apply in the order they were learned. Words seen in training collapse fast; new words stay in pieces." };
    case "compare": return { headline: `${s.text.length} characters, ${countWords(s.text)} words, ${s.encoded.tokens.length} tokens`, sub: "Models see token ids, so token count is what drives context length and cost." };
  }
}

export default function TokenizationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [merges, setMerges] = useState(DEFAULT_MERGES);
  const [text, setText] = useState(TEXT_PRESETS[0].text);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(merges, text), [merges, text]);
  const pb = usePlayback(steps.length, (i) => duration(steps[i]?.state ?? steps[0].state));
  if (tab === "metrics") return null;

  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${merges}-${text}`;
  const story = narrate(s);
  const learned = s.mergeIndex + 1;

  return (
    <Workspace
      main={
        <>
          <Panel title="Byte-pair encoding (BPE)" actions={<SimulatedLabel>toy tokenizer</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((v) => v + 1)} />
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={s.phase === "merge" ? "merge" : s.phase} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                {s.phase === "corpus" || s.phase === "merge" ? <TrainingCanvas s={s} k={k} animKey={animKey} /> : <EncodeCanvas s={s} k={k} animKey={animKey} />}
              </motion.div>
            </AnimatePresence>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard s={s} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Tokenizer settings" actions={<Button size="sm" variant="ghost" onClick={() => { setMerges(DEFAULT_MERGES); setText(TEXT_PRESETS[0].text); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <Slider label="Number of merges to learn" value={merges} min={0} max={MAX_MERGES} onChange={setMerges}
              hint={`0 = character-level tokenizer. After ${MAX_MERGES} merges no pair occurs twice in this corpus, so training stops.`} />
            <div className="mt-3 space-y-1">
              <label htmlFor="tok-text" className="block text-xs font-medium">Text to tokenize (max {MAX_TEXT} characters)</label>
              <input id="tok-text" value={text} maxLength={MAX_TEXT} onChange={(e) => setText(e.target.value)} className="h-8 w-full rounded-md border border-line bg-surface px-2 font-mono text-sm" />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Try:</span>
              {TEXT_PRESETS.map((p) => <Button key={p.label} size="sm" variant={text === p.text ? "primary" : "secondary"} onClick={() => setText(p.text)}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation s={s} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["merges learned", `${Math.max(0, learned)} / ${s.trained.merges.length}`],
              ["vocabulary size", 96 + Math.max(0, learned)],
              ["your text: characters", s.text.length],
              ["your text: BPE tokens", s.encoded.tokens.length],
              ["unseen words", s.encoded.unseen.length],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- training canvas: corpus words + pair counts + vocabulary ---------- */
function TrainingCanvas({ s, k, animKey }: { s: TokState; k: number; animKey: string }) {
  const { trained } = s;
  const i = s.mergeIndex;
  const snap = i >= 0 ? trained.snapshots[i] : null;
  const m = i >= 0 ? trained.merges[i] : null;
  const seg = snap ? snap.after : trained.initial;
  return (
    <div className="grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <div>
        <div className="mb-1 flex items-baseline justify-between text-[11px] text-muted"><span>Training corpus (each distinct word once, ×frequency)</span></div>
        <div className="flex flex-wrap gap-x-2 gap-y-1.5 rounded-md bg-surface-2 p-2">
          {trained.words.map((w, wi) => (
            <motion.span key={`${animKey}-${wi}`} className="inline-flex items-center gap-0.5" initial={s.phase === "corpus" ? { opacity: 0, y: 6 } : false} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: (0.1 + wi * 0.04) * k, duration: 0.3 * k }}>
              {seg[wi].map((t, ti) => snap?.fresh[wi][ti] ? <MergingChip key={ti} left={m!.pair[0]} right={m!.pair[1]} k={k} /> : (
                <span key={ti} className="rounded border border-line bg-surface px-1 font-mono text-[11px] leading-5">{show(t)}</span>
              ))}
              <span className="ml-0.5 font-mono text-[10px] text-muted">×{w.count}</span>
            </motion.span>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        {m ? <PairBars m={m} k={k} /> : <p className="rounded-md border border-dashed border-line p-2 text-xs text-muted">Next: count how often each pair of neighbouring symbols occurs, weighted by word frequency.</p>}
        <VocabList merges={trained.merges.slice(0, i + 1)} k={k} animKey={animKey} />
      </div>
    </div>
  );
}

function MergingChip({ left, right, k }: { left: string; right: string; k: number }) {
  return (
    <motion.span className="inline-flex rounded border border-warn bg-warn-soft font-mono text-[11px] leading-5"
      initial={{ scale: 1 }} animate={{ scale: [1, 1, 1.15, 1] }} transition={{ delay: M.glow * k, duration: 0.9 * k, times: [0, 0.4, 0.7, 1] }}>
      <span className="pl-1">{show(left)}</span>
      <motion.span className="w-px self-stretch bg-warn" initial={{ opacity: 1, marginLeft: 5, marginRight: 1 }} animate={{ opacity: 0, marginLeft: 0, marginRight: 0 }} transition={{ delay: M.join * k, duration: 0.45 * k, ease: EASE }} />
      <span className="pr-1">{show(right)}</span>
    </motion.span>
  );
}

function PairBars({ m, k }: { m: Merge; k: number }) {
  const max = m.top[0].count;
  return (
    <div>
      <div className="mb-1 text-[11px] text-muted">Most frequent adjacent pairs right now</div>
      <div className="space-y-1">
        {m.top.map((p, i) => (
          <div key={i} className="grid grid-cols-[64px_minmax(0,1fr)_28px] items-center gap-2">
            <span className={cx("truncate font-mono text-[11px]", i === 0 && "font-semibold")}>{show(p.pair[0])} + {show(p.pair[1])}</span>
            <div className="relative h-4 rounded bg-surface-2">
              <motion.div className={cx("absolute inset-y-0 left-0 rounded", i === 0 ? "bg-accent" : "bg-muted/40")} initial={{ width: 0 }} animate={{ width: `${(p.count / max) * 100}%` }}
                transition={{ delay: (M.bars + i * M.stagger) * k, duration: 0.45 * k, ease: EASE }} />
              {i === 0 && <motion.span className="absolute -right-1 top-1/2 -translate-y-1/2 rounded bg-fg px-1 text-[9px] font-semibold text-surface" initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: M.win * k, duration: 0.25 * k }}>MERGE</motion.span>}
            </div>
            <span className="text-right font-mono text-[11px]"><AnimatedNumber value={p.count} from={0} digits={0} delay={(M.bars + i * M.stagger) * k} duration={0.45 * k} /></span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VocabList({ merges, k, animKey }: { merges: Merge[]; k: number; animKey: string }) {
  const recent = merges.slice(-6).reverse();
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px] text-muted"><span>Vocabulary: 95 chars + &lt;unk&gt; + merges</span><span className="font-mono">size {96 + merges.length}</span></div>
      <ul className="space-y-1">
        <AnimatePresence initial={false}>
          {recent.map((m, i) => (
            <motion.li key={m.id} layout className={cx("flex items-center gap-2 rounded border px-2 py-0.5 text-[11px]", i === 0 ? "border-accent bg-accent-soft" : "border-line")}
              initial={i === 0 ? { opacity: 0, x: -24 } : false} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ delay: i === 0 ? M.vocab * k : 0, duration: 0.4 * k, ease: EASE }}>
              <span className="font-mono text-muted">#{m.id}</span>
              <span className="rounded border px-1 font-mono" style={chipStyle(m.id)}>{show(m.token)}</span>
              <span className="ml-auto font-mono text-muted">{show(m.pair[0])}+{show(m.pair[1])}</span>
            </motion.li>
          ))}
        </AnimatePresence>
        {merges.length > 6 && <li key={`${animKey}-more`} className="text-[11px] text-muted">… {merges.length - 6} earlier merges</li>}
        {merges.length === 0 && <li className="text-[11px] text-muted">No merges yet: every token is a single character.</li>}
      </ul>
    </div>
  );
}

/* ---------- encoding canvas: the learner's text as token chips ---------- */
function EncodeCanvas({ s, k, animKey }: { s: TokState; k: number; animKey: string }) {
  return s.phase === "encode" ? <EncodeFrames key={animKey} s={s} k={k} /> : <Chips tokens={s.phase === "split" ? s.encoded.frames[0].tokens : s.encoded.tokens} s={s} k={k} stagger={s.phase === "split"} withIds={s.phase === "compare"} />;
}

function EncodeFrames({ s, k }: { s: TokState; k: number }) {
  const frames = s.encoded.frames;
  const f = useFrames(frames.length - 1, FRAME_MS * k, ENC_DELAY * k);
  const cur = frames[f];
  return (
    <div>
      <div className="mb-2 h-6 text-xs">
        <motion.div key={f} initial={{ opacity: 0.2, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
            {cur.merge ? <>Applying merge <span className="font-mono">#{cur.merge.rank + 1}</span>: <span className="font-mono">{show(cur.merge.pair[0])} + {show(cur.merge.pair[1])} → </span><span className="rounded border px-1 font-mono" style={chipStyle(cur.merge.id)}>{show(cur.merge.token)}</span> <span className="text-muted">· {cur.tokens.length} tokens</span></>
              : frames.length > 1 ? <span className="text-muted">Characters, before any merge ({cur.tokens.length} tokens)</span> : <span className="text-muted">No learned merge matches this text: it stays at {cur.tokens.length} character tokens.</span>}
        </motion.div>
      </div>
      <Chips tokens={cur.tokens} s={s} k={k} layout />
      <div className="mt-2 h-1.5 overflow-hidden rounded bg-surface-2"><motion.div className="h-full bg-accent" animate={{ width: `${frames.length > 1 ? (f / (frames.length - 1)) * 100 : 100}%` }} transition={{ duration: 0.3 }} /></div>
    </div>
  );
}

function Chips({ tokens, s, k, stagger, withIds, layout }: { tokens: Token[]; s: TokState; k: number; stagger?: boolean; withIds?: boolean; layout?: boolean }) {
  const unseen = new Set(s.encoded.unseen);
  const groups = s.encoded.chunks.map((_, c) => tokens.filter((t) => t.chunk === c));
  return (
    <div className="flex flex-wrap gap-x-2 gap-y-2 rounded-md bg-surface-2 p-2">
      {groups.map((g, c) => (
        <motion.span key={c} className={cx("inline-flex flex-wrap items-start gap-0.5 rounded px-0.5", unseen.has(c) && withIds && "ring-1 ring-warn")}
          initial={stagger ? { opacity: 0, y: 6 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.15 + c * 0.12) * k, duration: 0.3 * k }}>
          {g.map((t) => (
            <motion.span key={`${t.chunk}-${t.start}`} layout={layout} transition={{ duration: 0.3 * k, ease: EASE }} className="inline-flex flex-col items-center">
              <span className="rounded border px-1 font-mono text-xs leading-5" style={chipStyle(t.id)}>{show(t.text)}</span>
              {withIds && <motion.span className="font-mono text-[9px] text-muted" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 * k }}>{t.id}</motion.span>}
            </motion.span>
          ))}
        </motion.span>
      ))}
      {tokens.length === 0 && <span className="text-xs text-muted">Type some text below.</span>}
    </div>
  );
}

/* ---------- card ---------- */
function StepCard({ s, k }: { s: TokState; k: number }) {
  const { trained, encoded } = s;
  if (s.phase === "corpus") {
    const total = trained.words.reduce((a, w) => a + w.count, 0);
    const chars = trained.initial.reduce((a, w, i) => a + w.length * trained.words[i].count, 0);
    const stats = [["words in corpus", total], ["distinct words", trained.words.length], ["character tokens", chars], ["starting vocabulary", 96]] as const;
    return (
      <div className="grid gap-2 sm:grid-cols-4">
        {stats.map(([label, v], i) => (
          <motion.div key={label} className="rounded-md border border-line p-2" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.2) * k }}>
            <div className="font-mono text-xl"><AnimatedNumber value={v} from={0} digits={0} delay={(0.2 + i * 0.2) * k} /></div>
            <div className="text-[11px] text-muted">{label}</div>
          </motion.div>
        ))}
        <p className="text-xs text-muted sm:col-span-4">Training repeats one rule: find the most frequent pair of neighbouring symbols and fuse it into a new symbol. The vocabulary grows by one each time.</p>
      </div>
    );
  }
  if (s.phase === "merge") {
    const m = trained.merges[s.mergeIndex];
    const rows = m.contributions.slice(0, 6);
    const rest = m.contributions.slice(6).reduce((a, c) => a + c.freq * c.occ, 0);
    return (
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <p className="mb-1 text-xs font-semibold">Where the count {m.count} comes from</p>
          <table className="w-full text-xs">
            <tbody className="font-mono">
              {rows.map((c, i) => (
                <motion.tr key={c.word} className="border-t border-line" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.1 + i * 0.12) * k }}>
                  <td className="py-0.5">{show(c.word)}</td><td className="text-muted">{c.freq} × {c.occ}</td><td className="text-right">{c.freq * c.occ}</td>
                </motion.tr>
              ))}
              {rest > 0 && <tr className="border-t border-line text-muted"><td className="py-0.5">{m.contributions.length - 6} more words</td><td /><td className="text-right">{rest}</td></tr>}
              <tr className="border-t-2 border-line font-semibold"><td className="py-0.5">total</td><td /><td className="text-right"><AnimatedNumber value={m.count} from={0} digits={0} delay={0.8 * k} /></td></tr>
            </tbody>
          </table>
        </div>
        <div className="space-y-2 text-xs">
          <p>word frequency × times the pair appears in that word, summed over the corpus.</p>
          <motion.div className="flex items-center gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: M.join * k }}>
            <span className="rounded border border-warn bg-warn-soft px-1.5 font-mono">{show(m.pair[0])}</span>+
            <span className="rounded border border-warn bg-warn-soft px-1.5 font-mono">{show(m.pair[1])}</span>→
            <motion.span className="rounded border px-1.5 font-mono" style={chipStyle(m.id)} initial={{ scale: 0.5 }} animate={{ scale: 1 }} transition={{ delay: M.vocab * k, type: "spring", stiffness: 300, damping: 15 }}>{show(m.token)}</motion.span>
            <span className="font-mono text-muted">id {m.id}</span>
          </motion.div>
          <p className="text-muted">New tokens can themselves be merged later, which is how whole words like "{show(trained.merges.reduce((a, x) => (x.token.length > a.length ? x.token : a), "") || "␣the")}" end up as a single token.</p>
        </div>
      </div>
    );
  }
  if (s.phase === "split") {
    return (
      <div className="space-y-1 text-xs">
        {encoded.chunks.map((c, i) => (
          <motion.div key={i} className="flex flex-wrap items-center gap-1" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.15 + i * 0.12) * k }}>
            <span className="w-24 truncate font-mono">"{show(c)}"</span><span className="text-muted">→</span>
            {[...c].map((ch, j) => <span key={j} className="rounded border border-line bg-surface-2 px-1 font-mono">{show(ch)}</span>)}
          </motion.div>
        ))}
        {encoded.chunks.length === 0 && <p className="text-muted">No text to tokenize.</p>}
      </div>
    );
  }
  if (s.phase === "encode") return <AppliedList s={s} k={k} />;
  const bars = [
    { label: "characters", v: s.text.length },
    { label: "words (split on spaces)", v: countWords(s.text) },
    { label: "BPE tokens", v: encoded.tokens.length },
  ];
  const max = Math.max(1, ...bars.map((b) => b.v));
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <div className="space-y-1.5">
        {bars.map((b, i) => (
          <div key={b.label} className="grid grid-cols-[150px_minmax(0,1fr)_32px] items-center gap-2 text-xs">
            <span>{b.label}</span>
            <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-full rounded", i === 2 ? "bg-accent" : "bg-muted/40")} initial={{ width: 0 }} animate={{ width: `${(b.v / max) * 100}%` }} transition={{ delay: (0.2 + i * 0.3) * k, duration: 0.5 * k, ease: EASE }} /></div>
            <span className="text-right font-mono"><AnimatedNumber value={b.v} from={0} digits={0} delay={(0.2 + i * 0.3) * k} /></span>
          </div>
        ))}
        <p className="text-[11px] text-muted">{encoded.tokens.length ? `${(s.text.length / encoded.tokens.length).toFixed(2)} characters per token.` : ""} More merges → fewer, longer tokens (try the slider).</p>
      </div>
      <div className="space-y-1 text-xs">
        <p className="font-semibold">Words never seen in training</p>
        {encoded.unseen.length === 0 && <p className="text-muted">None: every pre-token appeared in the corpus.</p>}
        {encoded.unseen.slice(0, 5).map((c, i) => (
          <motion.div key={c} className="flex flex-wrap items-center gap-1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (1.0 + i * 0.2) * k }}>
            <AlertTriangle size={11} className="text-warn" aria-hidden /><span className="font-mono">"{show(encoded.chunks[c])}"</span>→
            {encoded.tokens.filter((t) => t.chunk === c).map((t, j) => <span key={j} className="rounded border px-1 font-mono" style={chipStyle(t.id)}>{show(t.text)}</span>)}
          </motion.div>
        ))}
        <p className="text-[11px] text-muted">Unseen words still encode, as known sub-word pieces. Upper-case letters and digits barely appear in this corpus, so they stay as single characters. {encoded.tokens.some((t) => t.id === UNK_ID) && <Badge tone="err">non-ASCII → &lt;unk&gt;</Badge>}</p>
      </div>
    </div>
  );
}

function AppliedList({ s, k }: { s: TokState; k: number }) {
  const frames = s.encoded.frames;
  const f = useFrames(frames.length - 1, FRAME_MS * k, ENC_DELAY * k);
  const applied = frames.slice(1, f + 1);
  return (
    <div className="text-xs">
      <p className="mb-1"><span className="font-semibold">{frames.length - 1}</span> of {s.trained.merges.length} learned merges change this text. Token count: <span className="font-mono">{frames[0].tokens.length} → <strong>{frames[f].tokens.length}</strong></span></p>
      <div className="flex flex-wrap gap-1">
        {applied.map((fr) => (
          <motion.span key={fr.merge!.id} className="rounded border px-1 font-mono" style={chipStyle(fr.merge!.id)} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.25 }}>
            #{fr.merge!.rank + 1} {show(fr.merge!.token)}
          </motion.span>
        ))}
      </div>
      <p className="mt-2 text-muted">Merges that never fire here exist for other words. Ids ≥ {FIRST_MERGE_ID} are learned tokens; ids below are single characters.</p>
    </div>
  );
}

function Calculation({ s }: { s: TokState }) {
  const tex = (() => {
    if (s.phase === "corpus") return "|V_0| = 95\\ \\text{chars} + 1\\ (\\langle\\text{unk}\\rangle) = 96";
    if (s.phase === "merge") {
      const m = s.trained.merges[s.mergeIndex];
      const terms = m.contributions.slice(0, 5).map((c) => `${c.freq}\\cdot${c.occ}`);
      const rest = m.contributions.slice(5).reduce((a, c) => a + c.freq * c.occ, 0);
      return `\\begin{aligned} \\text{count}(a,b) &= \\textstyle\\sum_{w} f(w)\\, n_w(a,b) \\\\ &= ${terms.join(" + ")}${rest ? ` + ${rest}` : ""} = ${m.count} \\\\ |V| &= 96 + ${s.mergeIndex + 1} = ${97 + s.mergeIndex} \\end{aligned}`;
    }
    const n = s.encoded.tokens.length;
    if (s.phase === "split") return `\\text{${s.encoded.chunks.length} pre-tokens} \\to ${s.encoded.frames[0].tokens.length}\\ \\text{characters}`;
    if (s.phase === "encode") return `${s.encoded.frames[0].tokens.length}\\ \\text{chars} \\xrightarrow{${s.encoded.frames.length - 1}\\ \\text{merges}} ${n}\\ \\text{tokens}`;
    return n ? `\\frac{\\text{characters}}{\\text{tokens}} = \\frac{${s.text.length}}{${n}} = ${(s.text.length / n).toFixed(2)}` : "\\text{no tokens}";
  })();
  return <TexBlock>{tex}</TexBlock>;
}
