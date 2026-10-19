import { useEffect, useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Archive, ArrowRight, Brain, CheckCircle2, Database, FileText, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { dot, norm } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../agent-termination/kit";
import {
  DEFAULT_MEMORY, DIMS, QUESTION, SLOTS, sanitize, simulate, vectorize,
  type Hit, type MemState, type MemoryConfig, type SlotAnswer, type StoredFact, type Turn,
} from "./simulation";

/** One schedule (seconds at 1×) for canvas, card and counters. */
const T = { enter: 0.15, enterGap: 0.35, evict: 1.0, evictDur: 0.7, summaryIn: 1.55, write: 1.25, writeDur: 0.8, gauge: 0.9, scores: 0.5, scoreGap: 0.15, sort: 1.9, slots: 0.3, slotGap: 0.45 };
const DUR: Record<MemState["phase"], number> = { intro: 2400, exchange: 3300, question: 3000, retrieve: 3800, answer: 3600 };

const SOURCE_TONE = { "short-term": "accent", summary: "neutral", "long-term": "ok", missing: "err" } as const;

export default function MemoryVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<MemoryConfig>(DEFAULT_MEMORY);
  const [replay, setReplay] = useState(0);
  const c = sanitize(cfg);
  const steps = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "exchange"]);
  const step = steps[pb.index];
  const s = step.state;
  const prev = pb.index > 0 ? steps[pb.index - 1].state : s;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<MemoryConfig>) => setCfg((x) => ({ ...x, ...p }));

  if (tab === "metrics") return null;
  const story = narrate(s, c);
  const showScores = s.phase === "retrieve" || s.phase === "answer";

  return (
    <Workspace
      main={
        <>
          <Panel title="Agent memory" actions={<SimulatedLabel>Rule-based facts &amp; keyword vectors</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-4 md:grid-cols-2">
              <section aria-label="Short-term memory" className="min-w-0 space-y-2">
                <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted"><Brain size={13} /> Short-term: context window</h4>
                <Gauge used={s.used} from={prev.used} window={c.windowTokens} k={k} animKey={animKey} />
                {c.summarize && <SummaryCard lines={s.summary} fresh={s.evicted.filter((e) => e.summarized).map((e) => e.turn.id)} k={k} animKey={animKey} />}
                <TurnList state={s} k={k} animKey={animKey} />
              </section>
              <section aria-label="Long-term memory" className="min-w-0 space-y-2">
                <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted"><Database size={13} /> Long-term: vector store</h4>
                {!c.longTerm ? (
                  <div className="rounded-md border border-dashed border-line p-4 text-center text-xs text-muted">Long-term memory is off. Once a fact leaves the window (and the summary), it is gone.</div>
                ) : (
                  <>
                    {showScores && <QueryCard k={k} animKey={animKey} />}
                    {s.store.length === 0 && <div className="rounded-md border border-dashed border-line p-3 text-xs text-muted">Empty. Facts the user states will be written here.</div>}
                    <StoreList state={s} showScores={showScores} k={k} animKey={animKey} topK={c.topK} />
                  </>
                )}
              </section>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <StepCard s={s} prev={prev} cfg={c} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Memory settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_MEMORY); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Slider label="Context window" value={cfg.windowTokens} min={36} max={120} step={4} format={(v) => `${v} tokens`} onChange={(v) => set({ windowTokens: v })} hint="Toy tokenizer: 1 word = 1 token." />
              <Slider label="Retrieve top-k" value={cfg.topK} min={1} max={4} onChange={(v) => set({ topK: v })} />
              <Slider label="Similarity threshold" value={cfg.threshold} min={0} max={0.8} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ threshold: v })} />
              <div className="space-y-2">
                <Toggle label="Summarise evicted turns" checked={cfg.summarize} onChange={(v) => set({ summarize: v })} hint="Compress instead of dropping (lossy: numbers are dropped)." />
                <Toggle label="Long-term memory (vector store)" checked={cfg.longTerm} onChange={(v) => set({ longTerm: v })} />
                <Toggle label="Break it: user later corrects a fact" checked={cfg.conflict} onChange={(v) => set({ conflict: v })} hint="Adds 'my flight moved to 19 May'. The store appends; it never overwrites." />
              </div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.phase === "exchange" && s.evicted.length ? "The model only sees what is in the window. Anything evicted is invisible unless a summary or a long-term store keeps it." : s.phase === "retrieve" ? "Retrieval ranks by similarity, not by truth or recency. A stale fact can outrank its correction." : undefined} />
          <Panel title="Calculation"><Calc s={s} cfg={c} /></Panel>
          <Panel title="Memory now">
            <KV rows={[
              ["window used", `${s.used} / ${c.windowTokens} tokens`],
              ["turns in window", s.buffer.length],
              ["summary lines", c.summarize ? s.summary.length : "off"],
              ["long-term facts", c.longTerm ? s.store.length : "off"],
              ...(s.answer ? [["facts correct", `${s.answer.correct} / 3`] as [string, string]] : []),
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(s: MemState, c: MemoryConfig) {
  if (s.phase === "intro") return { headline: "Three places an agent can remember things", sub: `A ${c.windowTokens}-token window, ${c.summarize ? "a running summary" : "no summary"}, and ${c.longTerm ? "a long-term vector store" : "no long-term store"}.` };
  if (s.phase === "exchange") {
    const n = s.entering.length ? s.buffer.find((t) => t.id === s.entering[0]) : undefined;
    const ev = s.evicted.length;
    return {
      headline: ev ? `The window is full: ${ev} old turn${ev > 1 ? "s" : ""} ${s.evicted.some((e) => e.summarized) ? "get compressed into the summary" : "fall out"}` : "A new exchange fills the window",
      sub: `"${n?.text ?? ""}"${s.written.length ? " · a salient fact flies into long-term memory" : ""}`,
    };
  }
  if (s.phase === "question") return { headline: "Now the user asks something that needs old facts", sub: "Diet, city and arrival date were all mentioned several turns ago." };
  if (s.phase === "retrieve") return c.longTerm ? { headline: "The question is embedded and compared with every stored fact", sub: `Cosine similarity ranks them; the top ${c.topK} above ${c.threshold.toFixed(2)} join the context.` } : { headline: "No long-term memory to search", sub: "Only the window and the summary are available." };
  const a = s.answer!;
  return { headline: a.correct === 3 ? "The answer uses every fact correctly" : `The answer gets ${3 - a.correct} fact${3 - a.correct > 1 ? "s" : ""} wrong or missing`, sub: a.text };
}

function Gauge({ used, from, window, k, animKey }: { used: number; from: number; window: number; k: number; animKey: string }) {
  const pct = (v: number) => `${Math.min(100, (v / window) * 100)}%`;
  return (
    <div>
      <div className="flex justify-between text-[11px]"><span className="text-muted">tokens in window</span><span className="font-mono"><AnimatedNumber key={animKey} value={used} from={from} digits={0} delay={T.gauge * k} duration={0.6 * k} /> / {window}</span></div>
      <div className="relative h-2.5 overflow-hidden rounded bg-surface-2">
        <motion.div key={animKey} className={cx("absolute inset-y-0 left-0 rounded", used / window > 0.9 ? "bg-warn/80" : "bg-accent/70")} initial={{ width: pct(from) }} animate={{ width: pct(used) }} transition={{ delay: T.gauge * k, duration: 0.6 * k, ease: EASE }} />
      </div>
    </div>
  );
}

function SummaryCard({ lines, fresh, k, animKey }: { lines: MemState["summary"]; fresh: string[]; k: number; animKey: string }) {
  return (
    <div className="rounded-md border border-violet/40 bg-violet/5 p-2">
      <div className="mb-1 flex items-center gap-1 text-[11px] font-semibold text-violet"><Archive size={12} /> Running summary ({lines.reduce((a, l) => a + l.tokens, 0)} tokens, counts toward the window)</div>
      {lines.length === 0 ? <div className="text-[11px] text-muted">Nothing summarised yet.</div> : (
        <p className="text-xs leading-relaxed">
          {lines.map((l) => {
            const isFresh = fresh.includes(l.id);
            return (
              <motion.span key={isFresh ? `${l.id}-${animKey}` : l.id} initial={isFresh ? { opacity: 0, backgroundColor: "rgba(124,58,237,0.35)" } : false} animate={{ opacity: 1, backgroundColor: "rgba(124,58,237,0)" }}
                transition={{ delay: T.summaryIn * k, duration: 0.9 * k }} className="mr-1 rounded px-0.5">{l.text}</motion.span>
            );
          })}
        </p>
      )}
    </div>
  );
}

function Bubble({ t, className }: { t: Turn; className?: string }) {
  return (
    <div className={cx("flex", t.role === "user" ? "justify-end" : "justify-start")}>
      <div className={cx("max-w-[88%] rounded-lg px-2 py-1 text-xs leading-snug", t.role === "user" ? "bg-accent-soft" : "bg-surface-2", t.id === "q" && "border border-accent", className)}>
        <span className={cx(t.fact && t.fact.text && "underline decoration-ok decoration-2 underline-offset-2")}>{t.text}</span>
        <span className="ml-1.5 font-mono text-[10px] text-muted">{t.tokens}t</span>
      </div>
    </div>
  );
}

function TurnList({ state, k, animKey }: { state: MemState; k: number; animKey: string }) {
  return (
    <div className="space-y-1 rounded-md border border-line p-2" aria-label="Turns in the context window">
      {state.buffer.length === 0 && state.evicted.length === 0 && <div className="text-xs text-muted">The conversation has not started.</div>}
      {state.evicted.map(({ turn, summarized }) => (
        <motion.div key={`${turn.id}-${animKey}`} initial={{ opacity: 1, x: 0, y: 0, scale: 1, height: "auto" }} style={{ transformOrigin: "top center", overflow: "hidden" }}
          animate={summarized ? { opacity: 0, y: -40, scale: 0.4, height: 0 } : { opacity: 0, x: -90, height: 0 }}
          transition={{ default: { delay: T.evict * k, duration: T.evictDur * k, ease: EASE }, height: { delay: (T.evict + T.evictDur) * k, duration: 0.3 * k } }}>
          <Bubble t={turn} className="line-through opacity-80" />
        </motion.div>
      ))}
      {state.buffer.map((t) => {
        const i = state.entering.indexOf(t.id);
        return (
          <motion.div key={i >= 0 ? `${t.id}-${animKey}` : t.id} initial={i >= 0 ? { opacity: 0, y: 18 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: (T.enter + Math.max(0, i) * T.enterGap) * k, duration: 0.4 * k, ease: EASE }}>
            <Bubble t={t} />
          </motion.div>
        );
      })}
      <div className="flex gap-3 pt-1 text-[10px] text-muted"><span><span className="underline decoration-ok decoration-2">underlined</span> = contains a salient fact</span>{state.evicted.length > 0 && <span>struck-through = leaving the window</span>}</div>
    </div>
  );
}

function VecBars({ v, max = 4 }: { v: number[]; max?: number }) {
  return (
    <span className="inline-flex items-end gap-px align-middle" title={DIMS.map((d, i) => `${d}: ${v[i]}`).join(", ")} aria-label={`vector ${DIMS.map((d, i) => `${d} ${v[i]}`).join(", ")}`}>
      {v.map((x, i) => <span key={i} className="w-1.5 rounded-sm bg-violet/70" style={{ height: 2 + (Math.min(x, max) / max) * 14 }} />)}
    </span>
  );
}

function QueryCard({ k, animKey }: { k: number; animKey: string }) {
  return (
    <motion.div key={animKey} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 * k, duration: 0.4 * k }} className="rounded-md border border-accent bg-accent-soft p-2 text-xs">
      <div className="mb-1 flex items-center justify-between gap-2"><span className="font-semibold">query vector</span><VecBars v={vectorize(QUESTION)} /></div>
      <div className="font-mono text-[10px] text-muted">[{vectorize(QUESTION).join(", ")}] over ({DIMS.join(", ")})</div>
    </motion.div>
  );
}

function StoreList({ state, showScores, k, animKey, topK }: { state: MemState; showScores: boolean; k: number; animKey: string; topK: number }) {
  const hitOf = (f: StoredFact): Hit | undefined => state.hits.find((h) => h.fact.id === f.id);
  return (
    <SortAfter key={animKey} delayMs={showScores && state.phase === "retrieve" ? T.sort * 1000 * k : 0} sortable={showScores}>
      {(sorted) => {
        const order = sorted ? state.hits.map((h) => h.fact) : state.store;
        return (
          <ul className="space-y-1.5">
            {order.map((f, i) => {
              const h = hitOf(f);
              const fresh = state.written.includes(f.id);
              const storeIdx = state.store.indexOf(f);
              return (
                <motion.li layout key={fresh ? `${f.id}-${animKey}` : f.id} initial={fresh ? { opacity: 0, x: -160, scale: 0.6 } : false} animate={{ opacity: showScores && h && !h.retrieved ? 0.5 : 1, x: 0, scale: 1 }}
                  transition={{ delay: fresh ? T.write * k : 0, duration: (fresh ? T.writeDur : 0.5) * k, ease: EASE, layout: { duration: 0.6 * k, ease: EASE } }}
                  className={cx("rounded-md border p-1.5 text-xs", showScores && h?.retrieved ? "border-ok bg-ok-soft" : fresh ? "border-accent bg-accent-soft" : "border-line bg-surface")}>
                  <div className="flex items-center gap-2">
                    <FileText size={12} className="shrink-0 text-muted" />
                    <span className="min-w-0 flex-1">{f.text}</span>
                    <VecBars v={f.vector} />
                  </div>
                  {showScores && h && (
                    <div className="mt-1 flex items-center gap-2">
                      <div className="relative h-2 flex-1 overflow-hidden rounded bg-surface-2">
                        <motion.div className={cx("absolute inset-y-0 left-0 rounded", h.retrieved ? "bg-ok" : "bg-muted/50")} initial={{ width: state.phase === "retrieve" ? 0 : `${h.score * 100}%` }} animate={{ width: `${h.score * 100}%` }}
                          transition={{ delay: (T.scores + storeIdx * T.scoreGap) * k, duration: 0.5 * k, ease: EASE }} />
                      </div>
                      <span className="w-10 text-right font-mono text-[11px]">{h.score.toFixed(2)}</span>
                      {sorted && <span className="w-[72px] text-right">{h.retrieved ? <Badge tone="ok"><CheckCircle2 size={10} />rank {i + 1}</Badge> : <span className="text-[10px] text-muted">{i < topK ? "below threshold" : "not in top-k"}</span>}</span>}
                    </div>
                  )}
                </motion.li>
              );
            })}
          </ul>
        );
      }}
    </SortAfter>
  );
}

/** Shows the list in insertion order first, then re-sorts it (animated via `layout`) after a delay. Remount via `key` to replay. */
function SortAfter({ delayMs, sortable, children }: { delayMs: number; sortable: boolean; children: (sorted: boolean) => ReactNode }) {
  const [sorted, setSorted] = useState(delayMs === 0);
  useEffect(() => {
    if (delayMs === 0) return;
    const t = window.setTimeout(() => setSorted(true), delayMs);
    return () => window.clearTimeout(t);
  }, [delayMs]);
  return <>{children(sortable && sorted)}</>;
}

function StepCard({ s, prev, cfg, k }: { s: MemState; prev: MemState; cfg: MemoryConfig; k: number }) {
  if (s.phase === "intro") {
    const items = [
      { Icon: Brain, title: "Short-term window", body: `The last turns, verbatim, up to ${cfg.windowTokens} tokens. Cheap and exact, but small.` },
      { Icon: Archive, title: "Running summary", body: cfg.summarize ? "Old turns compressed into a few words. Saves space, loses detail." : "Off: old turns are simply dropped." },
      { Icon: Database, title: "Long-term store", body: cfg.longTerm ? "Salient facts saved as vectors, searched by similarity when needed." : "Off: nothing survives outside the window." },
    ];
    return (
      <div className="grid gap-2 sm:grid-cols-3">
        {items.map(({ Icon, title, body }, i) => (
          <motion.div key={title} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.35) * k, duration: 0.4 * k, ease: EASE }} className="rounded-md border border-line bg-surface-2 p-2 text-xs">
            <div className="mb-1 flex items-center gap-1.5 font-semibold"><Icon size={14} />{title}</div>
            <p className="text-muted">{body}</p>
          </motion.div>
        ))}
      </div>
    );
  }
  if (s.phase === "exchange" || s.phase === "question") {
    const added = s.buffer.filter((t) => s.entering.includes(t.id)).reduce((a, t) => a + t.tokens, 0);
    const removed = s.evicted.reduce((a, e) => a + e.turn.tokens, 0);
    const sumAdded = s.summary.length > prev.summary.length ? s.summary.slice(prev.summary.length).reduce((a, l) => a + l.tokens, 0) : 0;
    const fact = s.written.length ? s.store.find((f) => f.id === s.written[0]) : undefined;
    const rows = [
      { label: "before", v: prev.used, sign: "" },
      { label: "new turns", v: added, sign: "+" },
      ...(removed ? [{ label: "evicted turns", v: removed, sign: "−" }] : []),
      ...(sumAdded ? [{ label: "summary grows", v: sumAdded, sign: "+" }] : []),
      { label: "now", v: s.used, sign: "=" },
    ];
    return (
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <h4 className="mb-1.5 text-sm font-semibold">Token budget ({cfg.windowTokens} max)</h4>
          <ul className="space-y-1">
            {rows.map((r, i) => (
              <motion.li key={r.label} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.15 + i * 0.3) * k }} className={cx("flex justify-between rounded px-2 py-0.5 text-xs", r.sign === "=" ? "bg-accent-soft font-semibold" : "bg-surface-2")}>
                <span>{r.label}</span><span className="font-mono">{r.sign} {r.v}</span>
              </motion.li>
            ))}
          </ul>
          {s.evicted.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs">
              {s.evicted.map((e, i) => (
                <motion.li key={e.turn.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.evict + i * 0.2) * k }} className="flex flex-wrap items-center gap-1">
                  <span className="line-through text-muted">"{e.turn.text}"</span><ArrowRight size={11} />
                  {e.summarized ? <Badge>summary: "{s.summary.find((l) => l.id === e.turn.id)?.text}"</Badge> : <Badge tone="err">dropped{e.turn.role === "assistant" && cfg.summarize ? " (assistant reply)" : ""}</Badge>}
                </motion.li>
              ))}
            </ul>
          )}
        </div>
        <div>
          {s.phase === "question" ? <WhereAreFacts s={s} cfg={cfg} k={k} /> : fact ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 * k }} className="space-y-1.5 text-xs">
              <h4 className="text-sm font-semibold">Salient fact → long-term memory</h4>
              <p className="text-muted">A (rule-based) extractor turns the user's sentence into a short fact and a vector:</p>
              <div className="rounded border border-line p-1.5 font-mono">"{fact.text}"</div>
              <div className="flex items-center gap-2"><VecBars v={fact.vector} /><span className="font-mono">[{fact.vector.join(", ")}]</span></div>
              <p className="text-[10px] text-muted">dims: {DIMS.join(", ")}</p>
            </motion.div>
          ) : <p className="text-xs text-muted">{cfg.longTerm ? "No salient fact in this exchange: nothing is written to long-term memory." : "Long-term memory is off, so nothing is saved outside the window."}</p>}
        </div>
      </div>
    );
  }
  if (s.phase === "retrieve") {
    if (!cfg.longTerm) return <p className="text-sm text-muted">Nothing to retrieve. Whatever is not in the window or the summary is unavailable to the model.</p>;
    return (
      <div className="space-y-2 text-xs">
        <p>Each stored fact's vector is compared with the query vector. Scores (cosine similarity, 0 to 1):</p>
        <div className="grid gap-1">
          {s.hits.map((h, i) => (
            <motion.div key={h.fact.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (T.scores + i * T.scoreGap) * k }} className="flex items-center gap-2">
              <span className="w-48 truncate">{h.fact.text}</span>
              <span className="font-mono">{h.score.toFixed(3)}</span>
              {h.retrieved ? <Badge tone="ok">retrieved</Badge> : <Badge>skipped</Badge>}
            </motion.div>
          ))}
        </div>
        {s.hits.filter((h) => h.retrieved && h.fact.values.date).length > 1 && <Callout tone="warn" title="Two dates retrieved">Both the original and the corrected date came back. Similarity cannot tell which is current.</Callout>}
      </div>
    );
  }
  return <AnswerCard a={s.answer!} k={k} />;
}

function WhereAreFacts({ s, cfg, k }: { s: MemState; cfg: MemoryConfig; k: number }) {
  return (
    <div className="text-xs">
      <h4 className="mb-1.5 text-sm font-semibold">Where is each needed fact right now?</h4>
      <table className="w-full">
        <thead className="text-left text-muted"><tr><th>fact</th><th>window</th><th>summary</th><th>long-term</th></tr></thead>
        <tbody>
          {SLOTS.map((slot, i) => {
            const inBuf = s.buffer.some((t) => t.fact?.values[slot]);
            const inSum = cfg.summarize && s.summary.some((l) => l.values[slot]);
            const inStore = cfg.longTerm && s.store.some((f) => f.values[slot]);
            const mark = (b: boolean) => (b ? <CheckCircle2 size={13} className="text-ok" aria-label="yes" /> : <XCircle size={13} className="text-muted" aria-label="no" />);
            return (
              <motion.tr key={slot} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.8 + i * 0.3) * k }} className="border-t border-line">
                <td className="py-1 font-medium">{slot}</td><td>{mark(inBuf)}</td><td>{mark(inSum)}</td><td>{mark(inStore)}</td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AnswerCard({ a, k }: { a: { slots: SlotAnswer[]; text: string; correct: number }; k: number }) {
  return (
    <div className="space-y-2">
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">needed fact</th><th>value used</th><th>came from</th><th className="text-right">correct?</th></tr></thead>
        <tbody>
          {a.slots.map((s, i) => (
            <motion.tr key={s.slot} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (T.slots + i * T.slotGap) * k, duration: 0.3 * k }} className="border-t border-line">
              <td className="py-1 font-medium">{s.slot}</td>
              <td className="font-mono">{s.value ?? "?"}</td>
              <td><Badge tone={SOURCE_TONE[s.source]}>{s.source}</Badge></td>
              <td className="text-right">{s.correct ? <span className="text-ok">✓ yes</span> : <span className="text-err">✕ no</span>}</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (T.slots + 3 * T.slotGap + 0.2) * k }} className={cx("rounded-lg border p-2 text-sm", a.correct === 3 ? "border-ok/40 bg-ok-soft" : "border-err/40 bg-err-soft")}>
        <span className="mr-1 text-xs text-muted">Assistant (template answer):</span>{a.text}
      </motion.div>
      <AnimatePresence>
        {a.slots.some((s) => s.conflict) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.slots + 3 * T.slotGap + 0.6) * k }}>
            <Callout tone="warn" title="Stale fact won">{a.slots.find((s) => s.conflict)!.conflict}. Fix: upsert facts by key (trip.date), keep timestamps and prefer the newest, or let the correction delete the old fact.</Callout>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Calc({ s, cfg }: { s: MemState; cfg: MemoryConfig }) {
  if ((s.phase === "retrieve" || s.phase === "answer") && s.hits.length) {
    const q = vectorize(QUESTION), f = s.hits[0].fact.vector;
    return <TexBlock>{`\\begin{aligned} \\cos(q, f) &= \\frac{q \\cdot f}{\\lVert q \\rVert\\,\\lVert f \\rVert} \\\\ &= \\frac{${dot(q, f)}}{${norm(q).toFixed(3)} \\times ${norm(f).toFixed(3)}} = ${s.hits[0].score.toFixed(3)} \\end{aligned}`}</TexBlock>;
  }
  return <TexBlock>{`\\text{used} = \\textstyle\\sum \\text{turns} + \\sum \\text{summary} = ${s.buffer.reduce((a, t) => a + t.tokens, 0)} + ${s.summary.reduce((a, l) => a + l.tokens, 0)} = ${s.used} \\le ${cfg.windowTokens}`}</TexBlock>;
}
