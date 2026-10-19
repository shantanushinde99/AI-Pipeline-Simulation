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
import type { VisualizerTab } from "../types";
import { BarStrip, HBarRaw, StepHeader, ease, heat } from "./shared";
import { tn } from "./attention";
import { D, DEFAULT_PARAMS, DIMS, TOKENS, compute, simulate, type Phase, type SAParams, type SAState } from "./simulation";

/** One schedule (seconds at 1×) shared by canvas, card and counters. */
const T = {
  cell: (j: number) => 0.3 + j * 0.3,
  bar: (j: number) => 0.2 + j * 0.08,
  arc: (j: number) => 0.2 + j * 0.1,
  packet: (j: number) => 0.3 + j * 0.32,
  TRAVEL: 0.6,
  row: (i: number) => 0.15 + i * 0.14,
};
const DURATIONS: Record<Phase, number> = { tokens: 2600, query: 2600, scores: 4400, scale: 2600, softmax: 3000, lines: 2800, mix: 4800, heatmap: 3600 };

const NARRATION: Record<Phase, (tok: string, top: string) => { headline: string; sub: string }> = {
  tokens: () => ({ headline: "Each word arrives as a small vector", sub: "Four teaching-scale features per token: animate, place, pronoun, verb-ish." }),
  query: (t) => ({ headline: `"${t}" builds a query: what am I looking for?`, sub: "Every token also shows its key: what it can offer to others." }),
  scores: (t) => ({ headline: `"${t}"'s query visits every key and scores the match`, sub: "Each score is a dot product q·k. Watch the cells fill in, one token at a time." }),
  scale: () => ({ headline: "Divide every score by √d", sub: "With d = 4, √d = 2. This stops scores growing with dimension and saturating softmax." }),
  softmax: (_, top) => ({ headline: "Softmax turns scores into attention weights", sub: `Weights are positive and add up to 1. "${top}" gets the tallest bar.` }),
  lines: (t, top) => ({ headline: `"${t}" attends most strongly to "${top}"`, sub: "Line thickness is proportional to the attention weight." }),
  mix: (t) => ({ headline: `Values flow in and stack into the new "${t}"`, sub: "Each token contributes weight × value; the pieces add up dimension by dimension." }),
  heatmap: () => ({ headline: "Every token runs the same computation in parallel", sub: "Rows are queries, columns are keys. Click any row (or token) to re-run for that token." }),
};

const W = 800, LX = 110, GAP = 68;
const tx = (j: number) => LX + 34 + j * GAP;
const TOK_Y = 70, STRIP_Y = 104, CELL_Y = 168, BAR_TOP = 210, BAR_H = 80, OUT_Y = 330;

export default function SelfAttentionVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<SAParams>(DEFAULT_PARAMS);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(params), [params]);
  const c = useMemo(() => compute(params), [params]);
  const pb = usePlayback(steps.length, (i) => DURATIONS[steps[i]?.state.phase ?? "tokens"]);
  const step = steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${params.query}`;
  const tok = TOKENS[c.i];
  const story = NARRATION[step.state.phase](tok, TOKENS[c.top]);
  const set = (p: Partial<SAParams>) => setParams((o) => ({ ...o, ...p }));
  const pick = (j: number) => {
    set({ query: j });
    if (step.state.phase === "heatmap" || step.state.phase === "tokens") pb.seek(2);
    setReplay((r) => r + 1);
  };
  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Self-attention on one sentence" actions={<SimulatedLabel>hand-designed d = 4 vectors</SimulatedLabel>}>
            <StepHeader index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[640px]">
                {step.state.phase === "heatmap"
                  ? <HeatmapSvg state={step.state} k={k} animKey={animKey} onPick={pick} />
                  : <RowSvg state={step.state} c={c} k={k} animKey={animKey} onPick={pick} />}
              </div>
            </div>
            <p className="mt-1 text-[11px] text-muted">Click any token to make it the query. Bars: <span className="text-accent">positive</span> feature values; all embeddings here are hand-placed, not learned.</p>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <Card state={step.state} c={c} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_PARAMS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Query token" value={String(params.query)} options={TOKENS.map((t, j) => ({ value: String(j), label: `${j}: ${t}` }))} onChange={(v) => pick(Number(v))} />
              <Select<"designed" | "random"> label="Projection weights" value={params.weights} onChange={(v) => set({ weights: v })}
                options={[{ value: "designed", label: "Hand-designed (pronoun → animate)" }, { value: "random", label: "Random (untrained) — break it" }]} />
              {params.weights === "random" && <Slider label="Random seed" value={params.seed} min={1} max={50} onChange={(v) => set({ seed: v })} />}
              <Toggle label="Scale by 1/√d" checked={params.scaled} onChange={(v) => set({ scaled: v })} hint="Turn off to see a sharper, more peaked softmax." />
              <Toggle label="Causal mask (decoder style)" checked={params.causal} onChange={(v) => set({ causal: v })} hint="Each token may only look at itself and earlier tokens." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation phase={step.state.phase} c={c} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["query token", `"${tok}" (#${c.i})`],
              ["most attended", `"${TOKENS[c.top]}" · ${(c.row.weights[c.top] * 100).toFixed(1)}%`],
              ["entropy", `${c.entropy.toFixed(3)} nats`],
              ["scaling", params.scaled ? "÷ √4 = 2" : "off"],
              ["mask", params.causal ? "causal" : "none (bidirectional)"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

type C = ReturnType<typeof compute>;
const PHASES: Phase[] = ["tokens", "query", "scores", "scale", "softmax", "lines", "mix", "heatmap"];
const reached = (s: SAState, p: Phase) => PHASES.indexOf(s.phase) >= PHASES.indexOf(p);

function RowSvg({ state, c, k, animKey, onPick }: { state: SAState; c: C; k: number; animKey: string; onPick: (j: number) => void }) {
  const i = state.query;
  const ph = state.phase;
  const maxW = Math.max(...state.weights, 1e-9);
  const t = (d: number, dur = 0.5) => ({ delay: d * k, duration: dur * k, ease });
  const outMax = 1.1;
  const outScale = 520 / outMax;
  return (
    <svg viewBox={`0 0 ${W} 440`} className="h-auto w-full" role="img" aria-label={`Self-attention for the token ${TOKENS[i]}`}>
      {/* row labels */}
      {[["tokens", TOK_Y + 4], [ph === "tokens" ? "embedding x" : "key k", STRIP_Y + 24], ["score q·k", CELL_Y + 4], ["weight", BAR_TOP + 46]].map(([l, y]) => (
        <text key={l} x={8} y={Number(y)} fontSize={11} fill="var(--muted)">{l}</text>
      ))}
      {reached(state, "mix") && <text x={8} y={OUT_Y - 10} fontSize={11} fill="var(--muted)">new vector for "{TOKENS[i]}"</text>}

      {/* attention arcs */}
      {reached(state, "lines") && TOKENS.map((_, j) => {
        if (j === i || state.weights[j] === 0) return null;
        const x1 = tx(i), x2 = tx(j), top = TOK_Y - 14;
        const h = 8 + Math.abs(x2 - x1) * 0.075;
        return (
          <motion.path key={`${animKey}-arc-${j}`} d={`M ${x1} ${top} Q ${(x1 + x2) / 2} ${top - 2 * h} ${x2} ${top}`} fill="none" stroke="var(--accent)" strokeLinecap="round"
            strokeWidth={1 + 16 * state.weights[j]} strokeOpacity={0.25 + 0.65 * (state.weights[j] / maxW)}
            initial={ph === "lines" ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={t(T.arc(j), 0.6)} />
        );
      })}

      {/* query vector floating above the selected token, then visiting each key */}
      {(ph === "query" || ph === "scores") && (
        <motion.g key={`${animKey}-q`} initial={{ x: tx(i) - 30, opacity: 0, y: 10 }}
          animate={ph === "query" ? { x: tx(i) - 30, opacity: 1, y: 0 } : { x: TOKENS.map((_, j) => tx(j) - 30), opacity: 1, y: 0 }}
          transition={ph === "query" ? t(0.3, 0.6) : { x: { delay: T.cell(0) * k, duration: T.cell(TOKENS.length - 1) * k - T.cell(0) * k, ease: "linear" }, opacity: { duration: 0.2 } }}>
          <BarStrip x={0} y={4} values={c.q} max={Math.max(3, ...c.q.map(Math.abs))} w={11} gap={3} h={36} k={k} delay={0.5} color="var(--violet)" />
          <text x={58} y={18} fontSize={11} fontWeight={700} fill="var(--violet)">q</text>
        </motion.g>
      )}

      {/* tokens */}
      {TOKENS.map((w, j) => {
        const sel = j === i;
        return (
          <g key={`tok-${j}`} role="button" tabIndex={0} aria-label={`Use "${w}" as the query`} className="cursor-pointer" onClick={() => onPick(j)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onPick(j)}>
            <motion.rect x={tx(j) - 31} y={TOK_Y - 14} width={62} height={26} rx={6} initial={ph === "tokens" ? { opacity: 0, y: -20 } : false}
              animate={{ opacity: 1, y: 0 }} transition={t(j * 0.08, 0.5)}
              fill={sel ? "var(--violet)" : "var(--surface)"} stroke={sel ? "var(--violet)" : "var(--line)"} strokeWidth={sel ? 2 : 1} />
            <motion.text x={tx(j)} y={TOK_Y + 4} textAnchor="middle" fontSize={12} fontWeight={sel ? 700 : 500} fill={sel ? "var(--surface)" : "var(--fg)"}
              initial={ph === "tokens" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={t(j * 0.08 + 0.2)}>{w}</motion.text>
          </g>
        );
      })}

      {/* per-token strips: embedding x in step 1, key k afterwards */}
      {TOKENS.map((_, j) => (
        <BarStrip key={`${ph === "tokens" ? "x" : "k"}-${j}-${ph === "tokens" || ph === "query" ? animKey : "k"}`} x={tx(j) - 27} y={STRIP_Y} values={ph === "tokens" ? c.X[j] : c.K[j]}
          from={ph === "query" ? c.X[j] : undefined} max={ph === "tokens" ? 1 : 2} w={11} gap={3} h={40} k={k} delay={ph === "tokens" ? 0.6 + j * 0.08 : 0.3} />
      ))}

      {/* score cells */}
      {reached(state, "scores") && TOKENS.map((_, j) => {
        const raw = state.scores[j], sc = state.scaled[j];
        const masked = !Number.isFinite(sc);
        const showScaled = reached(state, "scale");
        return (
          <motion.g key={`${ph === "scores" ? animKey : "cells"}-cell-${j}`} initial={ph === "scores" ? { opacity: 0, scale: 0.6 } : false} animate={{ opacity: 1, scale: 1 }}
            transition={t(T.cell(j), 0.35)} style={{ transformOrigin: `${tx(j)}px ${CELL_Y}px` }}>
            <rect x={tx(j) - 29} y={CELL_Y - 13} width={58} height={24} rx={4} {...(masked && showScaled ? { fill: "var(--warn-soft)", fillOpacity: 1 } : heat(showScaled ? sc : raw, showScaled ? 2.2 : 4.4))}
              stroke={j === state.top ? "var(--fg)" : "var(--line)"} />
            <text x={tx(j)} y={CELL_Y + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">
              {masked && showScaled ? "−∞ mask" : ph === "scale" && Number.isFinite(sc)
                ? <AnimatedNumber key={animKey} svg value={sc} from={raw} delay={0.5 * k} duration={0.8 * k} />
                : ph === "scores" ? <AnimatedNumber key={animKey} svg value={raw} from={0} delay={T.cell(j) * k} duration={0.3 * k} />
                : (showScaled ? sc : raw).toFixed(2)}
            </text>
          </motion.g>
        );
      })}
      {ph === "scale" && (
        <motion.g key={`${animKey}-div`} initial={{ opacity: 0, y: -16 }} animate={{ opacity: [0, 1, 1, 0], y: [-16, 0, 0, 6] }} transition={{ duration: 1.3 * k, times: [0, 0.3, 0.7, 1] }}>
          {TOKENS.map((_, j) => (
            <text key={j} x={tx(j)} y={CELL_Y - 18} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--violet)">{c.p.scaled ? "÷ 2" : "× 1"}</text>
          ))}
        </motion.g>
      )}

      {/* softmax bars */}
      {reached(state, "softmax") && TOKENS.map((_, j) => {
        const wv = state.weights[j];
        const hgt = (wv / maxW) * BAR_H;
        const base = BAR_TOP + BAR_H;
        return (
          <g key={`bar-${j}`}>
            <rect x={tx(j) - 16} y={BAR_TOP} width={32} height={BAR_H} rx={3} fill="var(--surface-2)" />
            <motion.rect key={`${ph === "softmax" ? animKey : "bars"}-b${j}`} x={tx(j) - 16} width={32} rx={3} fill={j === state.top ? "var(--violet)" : "var(--accent)"} fillOpacity={0.85}
              initial={ph === "softmax" ? { y: base, height: 0 } : false} animate={{ y: base - hgt, height: hgt }} transition={t(T.bar(j), 0.7)} />
            <text x={tx(j)} y={base + 14} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill={j === state.top ? "var(--violet)" : "var(--fg)"} fontWeight={j === state.top ? 700 : 400}>
              {ph === "softmax" ? <AnimatedNumber key={animKey} svg value={wv * 100} from={0} digits={1} delay={T.bar(j) * k} duration={0.7 * k} /> : (wv * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}

      {/* output vector: stacked contributions per dimension */}
      {reached(state, "mix") && DIMS.map((dim, d) => {
        const y = OUT_Y + d * 26;
        let acc = 0;
        return (
          <g key={`out-${d}`}>
            <text x={LX + 60} y={y + 13} textAnchor="end" fontSize={11} fill="var(--muted)">{dim}</text>
            <rect x={LX + 68} y={y} width={outMax * outScale} height={18} rx={3} fill="var(--surface-2)" />
            {state.weights.map((wv, j) => {
              const v = wv * c.V[j][d];
              const x0 = LX + 68 + acc * outScale;
              acc += v;
              if (v < 1e-4) return null;
              return (
                <motion.rect key={`${ph === "mix" ? animKey : "mixed"}-${d}-${j}`} x={x0} y={y + 1} height={16} rx={2} fill={j === state.top ? "var(--violet)" : "var(--accent)"}
                  fillOpacity={j % 2 ? 0.55 : 0.85} initial={ph === "mix" ? { width: 0 } : false} animate={{ width: v * outScale }}
                  transition={t(T.packet(j) + T.TRAVEL, 0.3)} />
              );
            })}
            <text x={LX + 76 + outMax * outScale} y={y + 13} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">
              {ph === "mix" ? <AnimatedNumber key={animKey} svg value={state.out[d]} from={0} digits={3} delay={T.packet(0) * k} duration={(T.packet(TOKENS.length - 1) + T.TRAVEL) * k} /> : state.out[d].toFixed(3)}
            </text>
          </g>
        );
      })}
      {ph === "mix" && TOKENS.map((_, j) => state.weights[j] > 0.005 && (
        <motion.circle key={`${animKey}-pk-${j}`} r={4 + 10 * state.weights[j]} fill={j === state.top ? "var(--violet)" : "var(--accent)"} stroke="var(--surface)" strokeWidth={1.5}
          initial={{ cx: tx(j), cy: BAR_TOP + BAR_H - 4, opacity: 0 }}
          animate={{ cx: [tx(j), LX + 140 + j * 40], cy: [BAR_TOP + BAR_H - 4, OUT_Y + 40], opacity: [0, 1, 1, 0] }}
          transition={{ delay: T.packet(j) * k, duration: T.TRAVEL * k, ease: "easeIn", opacity: { delay: T.packet(j) * k, duration: T.TRAVEL * k, times: [0, 0.1, 0.85, 1] } }} />
      ))}
    </svg>
  );
}

function HeatmapSvg({ state, k, animKey, onPick }: { state: SAState; k: number; animKey: string; onPick: (j: number) => void }) {
  const n = TOKENS.length, cell = 46, L = 110, Tp = 86;
  return (
    <svg viewBox={`0 0 ${L + n * cell + 20} ${Tp + n * cell + 10}`} className="mx-auto h-auto w-full max-w-[640px]" role="img" aria-label="Full attention matrix heatmap">
      <text x={L + (n * cell) / 2} y={14} textAnchor="middle" fontSize={11} fill="var(--muted)">keys (who is looked at) →</text>
      {TOKENS.map((t, j) => (
        <text key={`c${j}`} x={L + j * cell + cell / 2 - 4} y={Tp - 8} fontSize={11} textAnchor="start" fill="var(--muted)" transform={`rotate(-60 ${L + j * cell + cell / 2} ${Tp - 8})`}>{t}</text>
      ))}
      {state.matrix.map((row, i) => (
        <g key={`r${i}`} role="button" tabIndex={0} aria-label={`Re-run with "${TOKENS[i]}" as the query`} className="cursor-pointer" onClick={() => onPick(i)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onPick(i)}>
          <text x={L - 8} y={Tp + i * cell + cell / 2 + 4} textAnchor="end" fontSize={12} fontWeight={i === state.query ? 700 : 400} fill={i === state.query ? "var(--violet)" : "var(--fg)"}>{TOKENS[i]}</text>
          {row.map((w, j) => (
            <motion.g key={`${animKey}-${i}-${j}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.row(i) + j * 0.02) * k, duration: 0.3 * k }}>
              <rect x={L + j * cell + 1} y={Tp + i * cell + 1} width={cell - 2} height={cell - 2} rx={3} fill="var(--accent)" fillOpacity={0.06 + Math.min(1, w / 0.5) * 0.85} />
              <text x={L + j * cell + cell / 2} y={Tp + i * cell + cell / 2 + 4} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={w > 0.3 ? "var(--surface)" : "var(--muted)"}>{w === 0 ? "0" : w.toFixed(2)}</text>
            </motion.g>
          ))}
          {i === state.query && <rect x={L} y={Tp + i * cell} width={n * cell} height={cell} fill="none" stroke="var(--violet)" strokeWidth={2.5} rx={4} />}
        </g>
      ))}
    </svg>
  );
}

function Card({ state, c, k }: { state: SAState; c: C; k: number }) {
  const i = state.query, tok = TOKENS[i], top = state.top;
  const tr = (d: number, dur = 0.5) => ({ delay: d * k, duration: dur * k, ease });
  switch (state.phase) {
    case "tokens":
      return (
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <p className="text-sm font-semibold">Embedding of "{tok}"</p>
            {DIMS.map((d, j) => (
              <div key={d} className="grid grid-cols-[70px_1fr_40px] items-center gap-2 text-xs"><span className="text-muted">{d}</span><HBarRaw k={k} v={c.X[i][j]} max={1} at={0.3 + j * 0.15} /><span className="text-right font-mono">{c.X[i][j].toFixed(1)}</span></div>
            ))}
          </div>
          <p className="text-sm leading-relaxed text-muted">Real models learn hundreds of unnamed dimensions. Here four hand-placed, named features keep every number readable. "it" is all <strong>pronoun</strong>, "animal" is all <strong>animate</strong>, "street" is all <strong>place</strong>.</p>
        </div>
      );
    case "query": {
      const prods = c.q.map((q, d) => q * c.K[top][d]);
      return (
        <div className="space-y-2">
          <p className="text-sm">How the query of <strong>"{tok}"</strong> matches the key of <strong>"{TOKENS[top]}"</strong>: multiply feature by feature and add.</p>
          {DIMS.map((d, j) => (
            <div key={d} className="grid grid-cols-[70px_1fr_140px] items-center gap-2 text-xs">
              <span className="text-muted">{d}</span><HBarRaw k={k} v={prods[j]} max={Math.max(...prods.map(Math.abs), 1e-9)} at={0.3 + j * 0.25} tone="bg-violet/70" />
              <span className="text-right font-mono">{c.q[j].toFixed(2)} × {c.K[top][j].toFixed(2)} = {prods[j].toFixed(2)}</span>
            </div>
          ))}
          <motion.p className="text-right font-mono text-sm font-semibold" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={tr(1.4)}>q·k = <AnimatedNumber value={state.scores[top]} from={0} delay={1.4 * k} /></motion.p>
        </div>
      );
    }
    case "scores":
    case "scale": {
      const max = Math.max(...state.scores.map(Math.abs), 1e-9);
      return (
        <div className="space-y-1">
          <p className="mb-1 text-xs text-muted">{state.phase === "scores" ? "Raw scores fill in the same order as the cells above." : c.p.scaled ? "Every score halves (÷ √4). The ranking does not change; only the spread shrinks." : "No scaling: the raw scores are used directly."}</p>
          {TOKENS.map((t, j) => {
            const v = state.phase === "scale" ? state.scaled[j] : state.scores[j];
            const masked = !Number.isFinite(v);
            return (
              <div key={j} className="grid grid-cols-[64px_1fr_56px] items-center gap-2 text-xs">
                <span className={cx(j === top && "font-semibold")}>{t}</span>
                {masked ? <Badge tone="warn">masked (future)</Badge> : <HBarRaw k={k} v={v} max={max} at={state.phase === "scores" ? T.cell(j) : 0.5} tone={j === top ? "bg-violet/70" : "bg-accent"} />}
                <span className="text-right font-mono">{masked ? "−∞" : <AnimatedNumber value={v} from={state.phase === "scale" ? state.scores[j] : 0} delay={(state.phase === "scores" ? T.cell(j) : 0.5) * k} />}</span>
              </div>
            );
          })}
        </div>
      );
    }
    case "softmax": {
      const ex = state.scaled.map((s) => (Number.isFinite(s) ? Math.exp(s) : 0));
      const sum = ex.reduce((a, b) => a + b, 0);
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">token</th><th>score</th><th>e^score</th><th>÷ {sum.toFixed(2)}</th></tr></thead>
            <tbody className="font-mono">
              {TOKENS.map((t, j) => (
                <motion.tr key={j} className={cx("border-t border-line", j === top && "font-semibold text-violet")} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={tr(T.bar(j), 0.3)}>
                  <td className="py-0.5 font-sans">{t}</td><td>{Number.isFinite(state.scaled[j]) ? state.scaled[j].toFixed(2) : "−∞"}</td><td>{ex[j].toFixed(3)}</td><td>{state.weights[j].toFixed(3)}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    case "lines":
    case "heatmap":
      return (
        <div className="space-y-2">
          <p className="text-xs text-muted">The sentence shaded by how much "{tok}" attends to each word:</p>
          <p className="flex flex-wrap gap-1 text-base">
            {TOKENS.map((t, j) => (
              <motion.span key={j} className={cx("rounded px-1.5 py-0.5", j === i && "outline-2 outline-violet")} initial={{ backgroundColor: "rgba(0,0,0,0)" }}
                animate={{ backgroundColor: `color-mix(in srgb, var(--accent) ${Math.round((state.weights[j] / Math.max(...state.weights)) * 70)}%, transparent)` }} transition={tr(T.arc(j), 0.6)}>
                {t}
              </motion.span>
            ))}
          </p>
          <p className="text-xs leading-relaxed text-muted">
            {state.phase === "lines" ? `Top three: ${[...state.weights.keys()].sort((a, b) => state.weights[b] - state.weights[a]).slice(0, 3).map((j) => `"${TOKENS[j]}" ${(state.weights[j] * 100).toFixed(1)}%`).join(", ")}.` : "Every row of the heatmap is one of these shadings. The diagonal is each token attending to itself."}
            {c.p.weights === "random" && " With random projection weights the pattern is arbitrary: nothing links \"it\" to \"animal\" until the weights are trained."}
          </p>
        </div>
      );
    case "mix":
      return (
        <div className="space-y-1.5">
          <p className="text-xs text-muted">Before vs after attention for "{tok}" (value = embedding here, so features keep their names):</p>
          {DIMS.map((d, j) => (
            <div key={d} className="grid grid-cols-[70px_1fr_1fr_88px] items-center gap-2 text-xs">
              <span className="text-muted">{d}</span>
              <HBarRaw k={k} v={c.X[i][j]} max={1} at={0.1} tone="bg-line" />
              <HBarRaw k={k} v={state.out[j]} max={1} at={T.packet(0) + T.TRAVEL} tone="bg-violet/70" />
              <span className="text-right font-mono">{c.X[i][j].toFixed(2)} → {state.out[j].toFixed(2)}</span>
            </div>
          ))}
          <p className="text-xs text-muted">Grey = before, violet = after. {state.out[0] > c.X[i][0] + 0.05 ? `The "animate" feature rose from ${c.X[i][0].toFixed(2)} to ${state.out[0].toFixed(2)}: "${tok}" now carries meaning borrowed from "${TOKENS[top]}".` : "The output is a blend of the attended tokens' values."}</p>
        </div>
      );
  }
}

function Calculation({ phase, c }: { phase: Phase; c: C }) {
  const { i, row, top } = c;
  const tok = TOKENS[i], kt = TOKENS[top];
  const q = c.q, kk = c.K[top];
  const tex: Record<Phase, string> = {
    tokens: `x_{\\text{${tok}}} = [${c.X[i].map((v) => v.toFixed(1)).join(",\\ ")}]`,
    query: `\\begin{aligned} q &= W_Q\\, x_{\\text{${tok}}} = [${q.map((v) => v.toFixed(2)).join(",\\ ")}] \\\\ k_{\\text{${kt}}} &= W_K\\, x_{\\text{${kt}}} = [${kk.map((v) => v.toFixed(2)).join(",\\ ")}] \\end{aligned}`,
    scores: `q \\cdot k_{\\text{${kt}}} = ${q.map((v, d) => `${tn(v)} \\cdot ${tn(kk[d])}`).join(" + ")} = ${row.scores[top].toFixed(2)}`,
    scale: c.p.scaled ? `\\frac{q \\cdot k}{\\sqrt{d}} = \\frac{${row.scores[top].toFixed(2)}}{\\sqrt{${D}}} = ${row.scaled[top].toFixed(2)}` : `\\text{no scaling: } ${row.scaled[top].toFixed(2)}`,
    softmax: `w_{\\text{${kt}}} = \\frac{e^{${row.scaled[top].toFixed(2)}}}{\\sum_j e^{s_j}} = ${row.weights[top].toFixed(3)}`,
    lines: `\\operatorname{Attention}(Q,K,V) = \\operatorname{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right) V`,
    mix: `\\text{out}_{\\text{${tok}}} = \\sum_j w_j\\, v_j = [${row.out.map((v) => v.toFixed(3)).join(",\\ ")}]`,
    heatmap: `A = \\operatorname{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right) \\in \\mathbb{R}^{${TOKENS.length} \\times ${TOKENS.length}}`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}
