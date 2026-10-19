import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { splitSentences } from "../rag/simulation";
import { SUPPORT_TAU, pct, wrap } from "./claims";
import { EASE, StepCaption } from "./StepCaption";
import { DEFAULT_GG, SCENARIOS, simulate, type AnswerSentence, type GGParams, type GGResult, type GGState } from "./simulation";

// ---------- one timing schedule (seconds at 1×) shared by canvas, card and counters ----------
const SCAN0 = 0.15, SCAN_GAP = 0.07;
const T = (nSpans: number, lines: number, prior: boolean) => {
  const pick = prior ? 0.1 : SCAN0 + nSpans * SCAN_GAP + 0.35;
  const write = pick + 0.3;
  const link = write + lines * 0.16 + 0.25;
  const verdict = link + 0.7;
  return { pick, write, link, verdict, end: verdict + 1.1 };
};
const LINE = 14, LEFT_W = 420, RIGHT_X = 500, RIGHT_W = 272, W = 780, LEFT_CH = 54, RIGHT_CH = 40;

type Layout = {
  spans: Record<string, { y: number; h: number; lines: string[]; passage: number }>;
  passages: { label: number; y: number; h: number; head: string; rows: ({ kind: "span"; id: string } | { kind: "plain"; y: number; lines: string[] } | { kind: "removed"; y: number })[] }[];
  answers: { y: number; h: number; lines: string[] }[];
  height: number;
};

function layout(r: GGResult): Layout {
  const spans: Layout["spans"] = {};
  const passages: Layout["passages"] = [];
  let y = 44;
  for (const p of r.passages) {
    const top = y;
    y += 24;
    const rows: Layout["passages"][number]["rows"] = [];
    for (const s of splitSentences(p.original)) {
      if (p.removed.includes(s)) {
        rows.push({ kind: "removed", y });
        y += LINE + 10;
        continue;
      }
      const span = r.spans.find((x) => x.passage === p.label && x.text === s);
      const lines = wrap(s, LEFT_CH);
      if (span) {
        spans[span.id] = { y, h: lines.length * LINE + 4, lines, passage: p.label };
        rows.push({ kind: "span", id: span.id });
      } else rows.push({ kind: "plain", y, lines });
      y += lines.length * LINE + 6;
    }
    passages.push({ label: p.label, y: top, h: y - top + 4, head: `[${p.label}] ${p.chunkId} · ${p.file}`, rows });
    y += 14;
  }
  let ay = 68;
  const answers = r.sentences.map((s) => {
    const lines = wrap(s.text, RIGHT_CH);
    const box = { y: ay, h: lines.length * LINE + 30, lines };
    ay += box.h + 12;
    return box;
  });
  return { spans, passages, answers, height: Math.max(y, ay + 70) };
}

const tone = (s: AnswerSentence) => (s.kind === "abstain" ? "var(--muted)" : s.check!.supported ? "var(--ok)" : "var(--err)");
const originLabel: Record<AnswerSentence["origin"], string> = { context: "from context", guess: "guess (no evidence)", prior: "from model memory", none: "abstained" };

export default function GroundedGenerationVisualizer() {
  const [params, setParams] = useState<GGParams>(DEFAULT_GG);
  const [replay, setReplay] = useState(0);
  const { result, steps } = useMemo(() => simulate(params), [params]);
  const lay = useMemo(() => layout(result), [result]);
  const durations = useMemo(() => steps.map((s) => {
    if (s.state.phase === "context") return 2600;
    if (s.state.phase === "plan") return 2400 + result.scenario.intents.length * 300;
    if (s.state.phase === "score") return 3200;
    const a = result.sentences[s.state.focus!];
    return T(result.spans.length, lay.answers[s.state.focus!].lines.length, a.intent === null).end * 1000;
  }), [steps, result, lay]);
  const pb = usePlayback(steps.length, (i) => durations[i] ?? 2500);
  const step = steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(params)}`;
  const set = (p: Partial<GGParams>) => setParams((v) => ({ ...v, ...p }));
  const story = narrate(step.state, result);

  return (
    <Workspace
      main={
        <>
          <Panel title="Grounded generation" actions={<SimulatedLabel>Rule-based generator</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={animKey} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[700px]">
                <Canvas r={result} lay={lay} state={step.state} k={k} animKey={animKey} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="22" height="6" aria-hidden><line x1="0" y1="3" x2="22" y2="3" stroke="var(--ok)" strokeWidth="2" /></svg>claim → supporting span</span>
              <span className="flex items-center gap-1"><svg width="22" height="6" aria-hidden><line x1="0" y1="3" x2="22" y2="3" stroke="var(--err)" strokeWidth="2" strokeDasharray="4 3" /></svg>closest span, but it does not support the claim</span>
              <span>% pill = share of the point's search terms found in that sentence</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard r={result} lay={lay} state={step.state} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Question & generator settings" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_GG); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="Question" value={params.scenarioId} onChange={(v) => set({ scenarioId: v })} options={SCENARIOS.map((s) => ({ value: s.id, label: s.question.text }))} />
              <Slider label="Evidence threshold τ" value={params.tau} min={0.25} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ tau: v })}
                hint="A point is only written if one sentence covers at least τ of its search terms. Too low and weak matches pass." />
              <Toggle label="Allow prior knowledge" checked={params.allowPrior} onChange={(v) => set({ allowPrior: v })} hint="Adds a plausible detail from model memory that no passage contains." />
              <Toggle label="Context missing the fact (break it)" checked={params.missingFact} onChange={(v) => set({ missingFact: v })} hint="Removes the sentence holding the answer, as if retrieval missed it." />
              <Toggle label="Abstain when evidence is weak" checked={params.abstain} onChange={(v) => set({ abstain: v })} hint="Off = the generator always answers, guessing when it has no evidence." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[step.state.phase]} />
          <Panel title="Calculation"><Calculation r={result} state={step.state} /></Panel>
          <Panel title="Answer so far">
            <KV rows={[
              ["evidence threshold τ", result.params.tau.toFixed(2)],
              ["support threshold (verifier)", SUPPORT_TAU.toFixed(2)],
              ["claims written", result.sentences.slice(0, step.state.shown).filter((s) => s.kind === "claim").length],
              ["abstentions", result.sentences.slice(0, step.state.shown).filter((s) => s.kind === "abstain").length],
              ["groundedness (final)", step.state.phase === "score" ? (result.groundedness === null ? "n/a" : pct(result.groundedness)) : "—"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const WHY: Record<GGState["phase"], string> = {
  context: "A grounded answer can only be as good as its context: anything missing here must be abstained on, not invented.",
  plan: "Writing point by point makes every sentence individually checkable against a source span.",
  write: "Each sentence either links to a span that supports it or it is a hallucination risk. The link is what a citation will later point to.",
  score: "Groundedness is a precision metric over claims: one invented sentence lowers it even if the rest is perfect.",
};

function narrate(s: GGState, r: GGResult) {
  if (s.phase === "context") return { headline: "The question arrives with three retrieved passages", sub: r.passages.some((p) => p.removed.length) ? "Fault on: the sentence that holds the answer is missing." : "The generator may only say what these sentences support." };
  if (s.phase === "plan") return { headline: `The generator outlines ${r.scenario.intents.length} points to cover`, sub: "Each point becomes one sentence, written only if evidence is found." };
  if (s.phase === "score") {
    const g = r.groundedness;
    return { headline: g === null ? "No claims made: the answer fully abstained" : `Groundedness: ${r.supported} of ${r.claims} claims supported (${pct(g)})`, sub: g === 1 ? "Every sentence traces back to a span in the context." : "Red sentences are what a user could not verify in the sources." };
  }
  const a = r.sentences[s.focus!];
  if (a.kind === "abstain") return { headline: `No evidence for ${a.label}: the generator abstains`, sub: "Saying \"the documents don't say\" is better than a confident guess." };
  if (a.origin === "context") return { headline: `Writing ${a.label}, copied from span ${a.check!.spanId}`, sub: "The scan scores every sentence; the best one becomes the claim and is linked back." };
  if (a.origin === "guess") return { headline: `No evidence for ${a.label}, but it answers anyway`, sub: "A confident guess: fluent, specific, and unsupported (it glows red)." };
  return { headline: "An extra detail slips in from model memory", sub: "Plausible, but no passage contains it, so the verifier cannot link it." };
}

function Canvas({ r, lay, state, k, animKey }: { r: GGResult; lay: Layout; state: GGState; k: number; animKey: string }) {
  const focus = state.phase === "write" ? state.focus : null;
  const fs = focus !== null ? r.sentences[focus] : null;
  const tl = fs ? T(r.spans.length, lay.answers[focus!].lines.length, fs.intent === null) : null;
  const spanOrder = r.spans.map((s) => s.id);
  const midOf = (id: string) => lay.spans[id].y + lay.spans[id].h / 2 - 6;
  const linkPath = (spanId: string, i: number) => {
    const y1 = midOf(spanId), y2 = lay.answers[i].y + 14;
    return `M ${LEFT_W + 8} ${y1} C ${LEFT_W + 50} ${y1}, ${RIGHT_X - 40} ${y2}, ${RIGHT_X} ${y2}`;
  };
  const supportedSpans = new Set(r.sentences.slice(0, state.shown).filter((s) => s.check?.supported).map((s) => s.check!.spanId!));
  return (
    <svg viewBox={`0 0 ${W} ${lay.height}`} className="h-auto w-full" role="img" aria-label="Context passages on the left, the answer being written on the right, with lines linking each claim to its supporting sentence">
      <text x={8} y={20} fontSize={12.5} fontWeight={600} fill="var(--fg)">Q: {r.scenario.question.text}</text>
      <text x={8} y={38} fontSize={10.5} fill="var(--muted)" letterSpacing={0.5}>CONTEXT PASSAGES</text>
      <text x={RIGHT_X} y={38} fontSize={10.5} fill="var(--muted)" letterSpacing={0.5}>ANSWER (WRITTEN SENTENCE BY SENTENCE)</text>

      {lay.passages.map((p, pi) => (
        <motion.g key={`${p.label}-${state.phase === "context" ? animKey : "static"}`} initial={state.phase === "context" ? { opacity: 0, x: -30 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: pi * 0.35 * k, duration: 0.5 * k, ease: EASE }}>
          <rect x={8} y={p.y} width={LEFT_W} height={p.h} rx={8} fill="var(--surface-2)" stroke="var(--line)" />
          <text x={18} y={p.y + 16} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--accent)">{p.head}</text>
          {p.rows.map((row, ri) => {
            if (row.kind === "plain") return row.lines.map((l, li) => <text key={`${ri}-${li}`} x={20} y={row.y + 11 + li * LINE} fontSize={11} fill="var(--muted)">{l}</text>);
            if (row.kind === "removed") return (
              <g key={ri}>
                <rect x={18} y={row.y} width={LEFT_W - 24} height={LINE + 4} rx={4} fill="var(--err-soft)" stroke="var(--err)" strokeDasharray="4 3" />
                <text x={26} y={row.y + 13} fontSize={10.5} fill="var(--err)">✂ sentence with “{r.scenario.question.expect}” missing from the context</text>
              </g>
            );
            const sp = lay.spans[row.id];
            const isBest = fs?.evidence?.best?.spanId === row.id && fs.evidence.best.coverage >= r.params.tau;
            const linked = supportedSpans.has(row.id);
            return (
              <g key={row.id}>
                <motion.rect x={14} y={sp.y - 2} width={LEFT_W - 52} height={sp.h} rx={4} initial={false}
                  animate={{ fill: isBest || linked ? "var(--ok-soft)" : "var(--surface-2)", stroke: isBest ? "var(--ok)" : "rgba(0,0,0,0)" }}
                  transition={{ delay: isBest && tl ? tl.pick * k : 0, duration: 0.3 * k }} />
                {sp.lines.map((l, li) => <text key={li} x={20} y={sp.y + 10 + li * LINE} fontSize={11} fill="var(--fg)">{l}</text>)}
                <text x={LEFT_W - 40} y={sp.y + 10} fontSize={9} textAnchor="end" fontFamily="var(--font-mono)" fill="var(--muted)">{row.id}</text>
              </g>
            );
          })}
        </motion.g>
      ))}

      {/* Evidence scan: a coverage pill pops up next to every sentence in order; the best one is picked. */}
      {fs?.evidence && fs.evidence.scores.map((sc) => {
        const sp = lay.spans[sc.spanId];
        const j = spanOrder.indexOf(sc.spanId);
        const pass = sc.coverage >= r.params.tau;
        return (
          <motion.g key={`${animKey}-pill-${sc.spanId}`} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (SCAN0 + j * SCAN_GAP) * k, duration: 0.25 * k }}
            style={{ transformOrigin: `${LEFT_W - 16}px ${sp.y + 7}px` }}>
            <rect x={LEFT_W - 34} y={sp.y} width={36} height={14} rx={7} fill={pass ? "var(--ok)" : sc.coverage > 0 ? "var(--warn-soft)" : "var(--surface)"} stroke="var(--line)" />
            <text x={LEFT_W - 16} y={sp.y + 10.5} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill={pass ? "var(--surface)" : "var(--fg)"}>{pct(sc.coverage)}</text>
          </motion.g>
        );
      })}
      {fs?.evidence && tl && (
        <motion.rect key={`${animKey}-sweep`} x={10} width={LEFT_W - 4} height={3} rx={1.5} fill="var(--accent)" initial={{ y: 44, opacity: 0 }}
          animate={{ y: [44, lay.height - 80], opacity: [0, 0.8, 0.8, 0] }} transition={{ duration: (tl.pick - 0.1) * k, ease: "linear", times: [0, 0.1, 0.9, 1] }} />
      )}

      {/* Outline placeholders, then the written sentences. */}
      {r.sentences.map((s, i) => {
        const box = lay.answers[i];
        const written = i < state.shown;
        const isFocus = i === focus;
        const planAnim = state.phase === "plan";
        if (!written) {
          return (
            <motion.g key={`ghost-${i}-${planAnim ? animKey : "s"}`} initial={planAnim ? { opacity: 0, y: 10 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + i * 0.3) * k, duration: 0.4 * k }}>
              {state.phase !== "context" && (
                <>
                  <rect x={RIGHT_X} y={box.y} width={RIGHT_W} height={box.h} rx={8} fill="none" stroke="var(--line)" strokeDasharray="5 4" />
                  <text x={RIGHT_X + 10} y={box.y + 18} fontSize={10.5} fill="var(--muted)">{s.intent === null ? "(not in outline)" : `${i + 1}. ${s.label}`}</text>
                </>
              )}
            </motion.g>
          );
        }
        const c = tone(s);
        const d = isFocus && tl ? tl : null;
        return (
          <g key={`ans-${i}-${isFocus ? animKey : "s"}`}>
            {s.check?.spanId && (
              <motion.path d={linkPath(s.check.spanId, i)} fill="none" stroke={c} strokeWidth={s.check.supported ? 2.2 : 1.6} strokeDasharray={s.check.supported ? undefined : "5 4"}
                initial={d ? { pathLength: 0, opacity: 0 } : false} animate={{ pathLength: 1, opacity: 0.9 }} transition={{ delay: (d?.link ?? 0) * k, duration: 0.6 * k, ease: EASE }} />
            )}
            <motion.g initial={d ? { opacity: 0, x: 16 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: (d?.write ?? 0) * k, duration: 0.35 * k }}>
              {isFocus && s.check && !s.check.supported && (
                <motion.rect x={RIGHT_X - 3} y={box.y - 3} width={RIGHT_W + 6} height={box.h + 6} rx={10} fill="none" stroke="var(--err)" strokeWidth={3}
                  initial={{ opacity: 0 }} animate={{ opacity: [0, 0.7, 0.2, 0.7] }} transition={{ delay: (d?.verdict ?? 0) * k, duration: 1.6, repeat: Infinity, repeatType: "reverse" }} />
              )}
              <rect x={RIGHT_X} y={box.y} width={RIGHT_W} height={box.h} rx={8} fill={s.kind === "abstain" ? "var(--surface-2)" : s.check!.supported ? "var(--ok-soft)" : "var(--err-soft)"} stroke={c} strokeWidth={1.4} />
              {box.lines.map((l, li) => (
                <motion.text key={li} x={RIGHT_X + 10} y={box.y + 16 + li * LINE} fontSize={11} fill="var(--fg)" fontStyle={s.kind === "abstain" ? "italic" : undefined}
                  initial={d ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: ((d?.write ?? 0) + li * 0.16) * k, duration: 0.2 * k }}>{l}</motion.text>
              ))}
              <text x={RIGHT_X + 10} y={box.y + box.h - 8} fontSize={9.5} fill="var(--muted)">{originLabel[s.origin]}</text>
            </motion.g>
            <motion.g initial={d ? { opacity: 0, scale: 0.5 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (d?.verdict ?? 0) * k, type: "spring", stiffness: 300, damping: 18 }}
              style={{ transformOrigin: `${RIGHT_X + RIGHT_W - 40}px ${box.y + box.h - 11}px` }}>
              <rect x={RIGHT_X + RIGHT_W - 92} y={box.y + box.h - 20} width={86} height={15} rx={7.5} fill={c} />
              <text x={RIGHT_X + RIGHT_W - 49} y={box.y + box.h - 9.5} textAnchor="middle" fontSize={9.5} fontWeight={600} fill="var(--surface)">
                {s.kind === "abstain" ? "— abstained" : s.check!.supported ? `✓ ${s.check!.spanId}` : "✗ unsupported"}
              </text>
            </motion.g>
          </g>
        );
      })}

      {state.phase === "score" && (
        <motion.g key={`${animKey}-score`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 * k, duration: 0.4 * k }}>
          <rect x={RIGHT_X} y={lay.height - 58} width={RIGHT_W} height={48} rx={8} fill="var(--surface)" stroke="var(--accent)" />
          <text x={RIGHT_X + 10} y={lay.height - 40} fontSize={11} fill="var(--muted)">groundedness = supported / claims</text>
          <text x={RIGHT_X + 10} y={lay.height - 20} fontSize={16} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">
            {r.groundedness === null ? "n/a (no claims)" : <>{r.supported}/{r.claims} = <AnimatedNumber svg value={r.groundedness * 100} from={0} digits={0} delay={0.5 * k} duration={1 * k} />%</>}
          </text>
        </motion.g>
      )}
    </svg>
  );
}

function StepCard({ r, lay, state, k }: { r: GGResult; lay: Layout; state: GGState; k: number }) {
  if (state.phase === "context") {
    return (
      <div className="space-y-2 text-xs">
        <p>Retrieval (the same simulated pipeline as the RAG visualizer) handed over these passages. The generator splits them into sentences it can quote:</p>
        <div className="grid gap-2 sm:grid-cols-3">
          {r.passages.map((p, i) => (
            <motion.div key={p.label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.35 * k }} className="rounded-md border border-line p-2">
              <div className="font-mono text-accent">[{p.label}] {p.chunkId}</div>
              <div className="text-muted">{p.file}</div>
              <div className="mt-1">{r.spans.filter((s) => s.passage === p.label).length} quotable sentences</div>
              {p.removed.length > 0 && <Badge tone="err">fact removed</Badge>}
            </motion.div>
          ))}
        </div>
      </div>
    );
  }
  if (state.phase === "plan") {
    return (
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">#</th><th>point to cover</th><th>search terms (normalized)</th></tr></thead>
        <tbody>
          {r.scenario.intents.map((it, i) => (
            <motion.tr key={i} className="border-t border-line" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.3) * k }}>
              <td className="py-1 font-mono">{i + 1}</td><td>{it.label}</td>
              <td className="space-x-1">{r.sentences[i].evidence!.terms.map((t) => <Badge key={t} tone="accent">{t}</Badge>)}</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (state.phase === "score") {
    const claims = r.sentences.filter((s) => s.kind === "claim");
    return (
      <div className="space-y-2 text-xs">
        <ul className="space-y-1">
          {r.sentences.map((s, i) => (
            <motion.li key={s.id} className="flex items-start gap-2" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.25) * k }}>
              <Badge tone={s.kind === "abstain" ? "neutral" : s.check!.supported ? "ok" : "err"}>{s.kind === "abstain" ? "not a claim" : s.check!.supported ? "✓ supported" : "✗ unsupported"}</Badge>
              <span className={cx(s.kind === "abstain" && "italic text-muted")}>{s.text}</span>
            </motion.li>
          ))}
        </ul>
        {claims.length > 0 && <Meter value={r.groundedness!} label={`groundedness = ${r.supported}/${r.claims}`} delay={(0.3 + r.sentences.length * 0.25) * k} k={k} />}
      </div>
    );
  }
  const s = r.sentences[state.focus!];
  const tl = T(r.spans.length, lay.answers[state.focus!].lines.length, s.intent === null);
  const top = s.evidence ? [...s.evidence.scores].sort((a, b) => b.coverage - a.coverage).slice(0, 4) : [];
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="space-y-1.5 text-xs">
        <div className="font-semibold">1 · Evidence search {s.evidence && <>for {"{"}{s.evidence.terms.join(", ")}{"}"}</>}</div>
        {s.evidence ? top.map((sc, i) => (
          <div key={sc.spanId} className="grid grid-cols-[38px_1fr_38px] items-center gap-2">
            <span className="font-mono text-muted">{sc.spanId}</span>
            <div className="relative h-3 rounded-sm bg-surface-2">
              <motion.div className={cx("absolute inset-y-0 left-0 rounded-sm", sc.coverage >= r.params.tau ? "bg-ok" : "bg-warn")} initial={{ width: 0 }} animate={{ width: `${sc.coverage * 100}%` }} transition={{ delay: (SCAN0 + i * 0.15) * k, duration: 0.5 * k, ease: EASE }} />
              <div className="absolute inset-y-[-3px] w-0.5 bg-fg" style={{ left: `${r.params.tau * 100}%` }} title="τ" />
            </div>
            <span className="text-right font-mono">{pct(sc.coverage)}</span>
          </div>
        )) : <p className="text-muted">No search: this sentence comes from model memory, not from the context.</p>}
        {s.evidence && <p className="text-muted">Black tick = τ {r.params.tau.toFixed(2)}. {s.kind === "abstain" ? "Nothing reaches it → abstain." : s.origin === "guess" ? "Nothing reaches it, but abstention is off → guess." : `${s.evidence.best!.spanId} passes → it becomes the sentence.`}</p>}
      </div>
      <div className="space-y-1.5 text-xs">
        <div className="font-semibold">2 · Verifier: is the written claim supported?</div>
        {s.check ? (
          <>
            <div className="flex flex-wrap gap-1">
              {s.check.claimTokens.map((t, i) => {
                const hit = s.check!.matched.includes(t);
                return (
                  <motion.span key={t} initial={{ opacity: 0.25 }} animate={{ opacity: 1 }} transition={{ delay: (tl.link + i * 0.05) * k }}>
                    <Badge tone={hit ? "ok" : /\d/.test(t) ? "err" : "neutral"}>{hit ? "✓" : "✗"} {t}</Badge>
                  </motion.span>
                );
              })}
            </div>
            <p>
              closest span <span className="font-mono">{s.check.spanId ?? "none"}</span> contains <strong><AnimatedNumber key={`${state.focus}`} value={s.check.matched.length} from={0} digits={0} delay={tl.link * k} duration={0.6 * k} /></strong> of {s.check.claimTokens.length} claim words
              ({pct(s.check.coverage)}; needs ≥ {pct(SUPPORT_TAU)}){s.check.numbers.length > 0 && <>, numbers {s.check.missingNumbers.length ? <strong className="text-err">{s.check.missingNumbers.join(", ")} missing</strong> : "all present"}</>}.
            </p>
            <Badge tone={s.check.supported ? "ok" : "err"}>{s.check.supported ? "✓ supported" : "✗ unsupported: hallucination risk"}</Badge>
          </>
        ) : <p className="text-muted">An abstention asserts nothing, so there is nothing to verify.</p>}
      </div>
    </div>
  );
}

function Meter({ value, label, delay, k }: { value: number; label: string; delay: number; k: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between"><span>{label}</span><span className="font-mono"><AnimatedNumber value={value * 100} from={0} digits={0} delay={delay} duration={0.8 * k} />%</span></div>
      <div className="h-3 overflow-hidden rounded-sm bg-surface-2">
        <motion.div className={cx("h-full", value === 1 ? "bg-ok" : value >= 0.5 ? "bg-warn" : "bg-err")} initial={{ width: 0 }} animate={{ width: `${value * 100}%` }} transition={{ delay, duration: 0.8 * k, ease: EASE }} />
      </div>
    </div>
  );
}

function Calculation({ r, state }: { r: GGResult; state: GGState }) {
  if (state.phase === "write") {
    const s = r.sentences[state.focus!];
    const b = s.evidence?.best;
    return (
      <div className="space-y-2">
        {s.evidence && <TexBlock>{`\\text{evidence} = \\frac{|T \\cap S|}{|T|} = \\frac{${b ? Math.round(b.coverage * s.evidence.terms.length) : 0}}{${s.evidence.terms.length}} = ${(b?.coverage ?? 0).toFixed(2)} ${(b?.coverage ?? 0) >= r.params.tau ? "\\ge" : "<"} \\tau`}</TexBlock>}
        <p className="text-[11px] text-muted">T = the point's search terms, S = a context sentence, C = the written claim's words.</p>
        {s.check && <TexBlock>{`\\text{support} = \\frac{|C \\cap S|}{|C|} = \\frac{${s.check.matched.length}}{${s.check.claimTokens.length}} = ${s.check.coverage.toFixed(2)}${s.check.missingNumbers.length ? `,\\ ${s.check.missingNumbers.map((n) => n.replace("$", "\\$")).join(", ")} \\notin S` : ""}`}</TexBlock>}
      </div>
    );
  }
  const g = r.groundedness;
  return (
    <TexBlock>{state.phase === "score" && g !== null
      ? `\\text{groundedness} = \\frac{\\#\\text{supported}}{\\#\\text{claims}} = \\frac{${r.supported}}{${r.claims}} = ${g.toFixed(2)}`
      : "\\text{groundedness} = \\frac{\\#\\text{supported claims}}{\\#\\text{claims}}"}</TexBlock>
  );
}
