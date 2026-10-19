import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { BarStrip, HBarRaw, StepHeader, ease, heat } from "../self-attention/shared";
import { texMat, tn } from "../self-attention/attention";
import { DEFAULT_PARAMS, DIMS, EMBEDDINGS, QK_DIMS, SWAP, TOKENS, compute, setEntry, simulate, type MatName, type Phase, type QKVParams, type QKVState } from "./simulation";

/** One timing schedule (seconds at 1×). */
const T = {
  row: (r: number) => 0.4 + r * 0.9, // project-q: row r of W·x
  col: (j: number) => 0.3 + j * 0.8, // project-k/v: token column j
  cell: (j: number, r: number) => 0.3 + j * 0.8 + 0.15 * r,
  swap: 3.7,
  dot: (j: number) => 0.3 + j * 0.7,
  bar: (j: number) => 0.2 + j * 0.12,
  packet: (j: number) => 0.3 + j * 0.6,
  TRAVEL: 0.55,
};
const DURATIONS: Record<Phase, number> = { embed: 2400, "project-q": 4600, "project-k": 5000, "project-v": 3700, dot: 3600, scale: 2400, softmax: 2600, mix: 3600 };
const MAT_LABEL: Record<MatName, string> = { WQ: "W_Q (query)", WK: "W_K (key)", WV: "W_V (value)" };

const narrate = (p: Phase, tok: string, top: string, swapped: boolean) => ({
  embed: { headline: "Four tokens, each a 4-number embedding", sub: `"${tok}" will ask the question. Every token can answer.` },
  "project-q": { headline: `Turn "${tok}" into a query: q = W_Q · x`, sub: "Each row of W_Q is dotted with x to produce one entry of q. Watch it go row by row." },
  "project-k": { headline: "Every token gets a key: k = W_K · x", sub: swapped ? `The same matrix for every token. Then the keys of "${TOKENS[SWAP[0]]}" and "${TOKENS[SWAP[1]]}" get swapped.` : "The same matrix is applied to every token, column by column." },
  "project-v": { headline: "Every token gets a value: v = W_V · x", sub: "Keys are for matching; values are the content that gets handed back." },
  dot: { headline: `"${tok}"'s query is compared with every key`, sub: "Dot product q·k: large when they point the same way." },
  scale: { headline: "Scale every score by 1/√d = 1/2", sub: "Same ranking, gentler spread for the softmax." },
  softmax: { headline: `Softmax: "${top}" wins most of the attention`, sub: "Weights are positive and sum to 1." },
  mix: { headline: "Values flow back, weighted, and stack into the output", sub: "Where to look came from keys; what comes back comes from values." },
})[p];

export default function QueryKeyValueVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<QKVParams>(DEFAULT_PARAMS);
  const [edit, setEdit] = useState<{ m: MatName; r: number; c: number }>({ m: "WQ", r: 3, c: 1 });
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(params), [params]);
  const c = useMemo(() => compute(params), [params]);
  const pb = usePlayback(steps.length, (i) => DURATIONS[steps[i]?.state.phase ?? "embed"]);
  const step = steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(step.state.phase, TOKENS[c.i], TOKENS[c.top], params.swapKeys);
  const set = (p: Partial<QKVParams>) => setParams((o) => ({ ...o, ...p }));
  if (tab === "metrics") return null;
  const projecting = ["embed", "project-q", "project-k", "project-v"].includes(step.state.phase);

  return (
    <Workspace
      main={
        <>
          <Panel title="Query, key and value" actions={<SimulatedLabel>hand-designed d = 4 matrices</SimulatedLabel>}>
            <StepHeader index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[640px]">
                {projecting ? <ProjectionSvg state={step.state} params={params} k={k} animKey={animKey} /> : <AttentionSvg state={step.state} c={c} k={k} animKey={animKey} swap={params.swapKeys} />}
              </div>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <Card state={step.state} c={c} params={params} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_PARAMS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Query token" value={String(params.query)} options={TOKENS.map((t, j) => ({ value: String(j), label: t }))} onChange={(v) => set({ query: Number(v) })} />
              <Toggle label={`Swap the keys of "${TOKENS[SWAP[0]]}" and "${TOKENS[SWAP[1]]}"`} checked={params.swapKeys} onChange={(v) => set({ swapKeys: v })}
                hint="Break it: values stay put, so attention moves to the wrong content." />
            </div>
            <div className="mt-3 grid gap-3 rounded-md border border-line p-2.5 sm:grid-cols-4">
              <Select<MatName> label="Edit matrix" value={edit.m} options={(["WQ", "WK", "WV"] as MatName[]).map((m) => ({ value: m, label: MAT_LABEL[m] }))} onChange={(m) => setEdit({ ...edit, m })} />
              <Select label="Row (output)" value={String(edit.r)} options={[0, 1, 2, 3].map((r) => ({ value: String(r), label: `${r}: ${edit.m === "WV" ? DIMS[r] : QK_DIMS[r]}` }))} onChange={(r) => setEdit({ ...edit, r: Number(r) })} />
              <Select label="Column (input)" value={String(edit.c)} options={[0, 1, 2, 3].map((x) => ({ value: String(x), label: `${x}: ${DIMS[x]}` }))} onChange={(x) => setEdit({ ...edit, c: Number(x) })} />
              <Slider key={`${edit.m}${edit.r}${edit.c}`} label={`${edit.m}[${edit.r}][${edit.c}]`} value={params.weights[edit.m][edit.r][edit.c]} min={-3} max={3} step={0.1}
                format={(v) => v.toFixed(1)} onChange={(v) => set({ weights: setEntry(params.weights, edit.m, edit.r, edit.c, Math.round(v * 10) / 10) })} />
            </div>
            <p className="mt-2 text-[11px] text-muted">Try W_Q[3][1] (how strongly an action seeks something drinkable): set it to 0 and "drank" stops looking at "tea".</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation phase={step.state.phase} c={c} params={params} /></Panel>
          <Panel title="Library analogy">
            <div className="space-y-2 text-xs leading-relaxed">
              <KV rows={[["query q", "the request you type at the desk"], ["key k", "the catalogue card on each shelf"], ["value v", "the book's actual contents"]]} />
              <p className="rounded-md bg-surface-2 p-2">
                "{TOKENS[c.i]}" asks for <span className="font-mono">[{c.q.map((v) => v.toFixed(1)).join(", ")}]</span>. The cards match best for{" "}
                {[...c.weights.keys()].sort((a, b) => c.weights[b] - c.weights[a]).slice(0, 2).map((j) => `"${TOKENS[j]}" (${(c.weights[j] * 100).toFixed(0)}%)`).join(" and ")}.
                Unlike a real library you don't get one book: you leave with a blend of every book, mixed by those percentages.
              </p>
              {params.swapKeys && <p className="rounded-md bg-warn-soft p-2">Two catalogue cards were swapped: the request now finds the shelf of "{TOKENS[c.top]}" and gets its contents, even though the card described "{TOKENS[c.top === SWAP[0] ? SWAP[1] : SWAP[0]]}".</p>}
            </div>
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

type C = ReturnType<typeof compute>;
const CELL = 46;

function Grid({ x, y, M, label, cols, rows, hl, k, fadeIn, digits = 1 }: {
  x: number; y: number; M: number[][]; label: string; cols?: string[]; rows?: string[]; hl?: { r?: number; c?: number }; k: number; fadeIn?: (r: number, c: number) => number; digits?: number;
}) {
  const max = Math.max(1, ...M.flat().map(Math.abs));
  return (
    <g>
      <text x={x + (M[0].length * CELL) / 2} y={y - (cols ? 22 : 8)} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--fg)">{label}</text>
      {cols?.map((c, j) => <text key={c + j} x={x + j * CELL + CELL / 2} y={y - 6} textAnchor="middle" fontSize={10} fill="var(--muted)">{c}</text>)}
      {rows?.map((r, i) => <text key={r + i} x={x - 6} y={y + i * CELL + CELL / 2 + 3} textAnchor="end" fontSize={10} fill="var(--muted)">{r}</text>)}
      {M.map((row, i) => row.map((v, j) => (
        <motion.g key={`${i}-${j}`} initial={fadeIn ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: (fadeIn?.(i, j) ?? 0) * k, duration: 0.25 * k }}>
          <rect x={x + j * CELL + 1} y={y + i * CELL + 1} width={CELL - 2} height={CELL - 2} rx={4} {...heat(v, max)}
            stroke={hl && (hl.r === i || hl.c === j) ? "var(--violet)" : "var(--line)"} strokeWidth={hl && (hl.r === i || hl.c === j) ? 2 : 1} />
          <text x={x + j * CELL + CELL / 2} y={y + i * CELL + CELL / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{v.toFixed(digits)}</text>
        </motion.g>
      )))}
    </g>
  );
}

function ProjectionSvg({ state, params, k, animKey }: { state: QKVState; params: QKVParams; k: number; animKey: string }) {
  const ph = state.phase;
  const X = EMBEDDINGS;
  const colsX = X[0].map((_, r) => X.map((x) => x[r])); // 4 dims × 4 tokens
  if (ph === "embed") {
    return (
      <svg viewBox="0 0 800 300" className="h-auto w-full" role="img" aria-label="Token embeddings as columns">
        <Grid x={260} y={70} M={colsX} label="X (one column per token)" cols={TOKENS} rows={DIMS} k={k} fadeIn={(r, j) => 0.2 + j * 0.3 + r * 0.06} hl={{ c: state.query }} />
        <text x={400} y={290} textAnchor="middle" fontSize={11} fill="var(--muted)">Highlighted column = the query token "{TOKENS[state.query]}"</text>
      </svg>
    );
  }
  if (ph === "project-q") {
    const W = params.weights.WQ, x = X[state.query];
    const xc = x.map((v) => [v]);
    const W0 = 100, Y0 = 70, XC = 370, RC = 480;
    return (
      <svg viewBox="0 0 800 330" className="h-auto w-full" role="img" aria-label="Matrix-vector multiply W_Q times x">
        <Grid x={W0} y={Y0} M={W} label="W_Q" rows={QK_DIMS.map((d) => `seek ${d}`)} k={k} />
        <text x={W0 + 4 * CELL + 30} y={Y0 + 2 * CELL + 6} textAnchor="middle" fontSize={20} fill="var(--muted)">×</text>
        <Grid x={XC} y={Y0} M={xc} label={`x (${TOKENS[state.query]})`} k={k} />
        <text x={XC + CELL + 32} y={Y0 + 2 * CELL + 6} textAnchor="middle" fontSize={20} fill="var(--muted)">=</text>
        <text x={RC + CELL / 2} y={Y0 - 8} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--violet)">q</text>
        {/* moving row highlight across W and the result */}
        <motion.rect key={`${animKey}-band`} x={W0 - 4} width={4 * CELL + 8} height={CELL} rx={6} fill="none" stroke="var(--violet)" strokeWidth={3}
          initial={{ y: Y0, opacity: 0 }} animate={{ y: [0, 1, 2, 3].map((r) => Y0 + r * CELL), opacity: 1 }}
          transition={{ y: { delay: T.row(0) * k, duration: T.row(3) * k - T.row(0) * k, times: [0, 1 / 3, 2 / 3, 1], ease: "easeInOut" }, opacity: { delay: T.row(0) * k, duration: 0.2 } }} />
        <rect x={XC - 3} y={Y0 - 3} width={CELL + 6} height={4 * CELL + 6} rx={6} fill="none" stroke="var(--violet)" strokeOpacity={0.5} strokeDasharray="4 3" />
        {[0, 1, 2, 3].map((r) => (
          <g key={`${animKey}-r${r}`}>
            {/* the row's sum travels to its slot in q */}
            <motion.circle r={7} fill="var(--violet)" initial={{ cx: W0 + 4 * CELL, cy: Y0 + r * CELL + CELL / 2, opacity: 0 }}
              animate={{ cx: [W0 + 4 * CELL, RC + CELL / 2], opacity: [0, 1, 1, 0] }} transition={{ delay: (T.row(r) + 0.35) * k, duration: 0.5 * k, opacity: { delay: (T.row(r) + 0.35) * k, duration: 0.5 * k, times: [0, 0.1, 0.8, 1] } }} />
            <motion.g initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (T.row(r) + 0.8) * k, duration: 0.3 * k }} style={{ transformOrigin: `${RC + CELL / 2}px ${Y0 + r * CELL + CELL / 2}px` }}>
              <rect x={RC + 1} y={Y0 + r * CELL + 1} width={CELL - 2} height={CELL - 2} rx={4} fill="var(--violet)" fillOpacity={0.15 + Math.min(1, Math.abs(state.q[r]) / 3) * 0.6} stroke="var(--violet)" />
              <text x={RC + CELL / 2} y={Y0 + r * CELL + CELL / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{state.q[r].toFixed(2)}</text>
            </motion.g>
            {/* the row's products, shown while it is active */}
            <motion.text x={W0} y={Y0 + 4 * CELL + 36} fontSize={12} fontFamily="var(--font-mono)" fill="var(--fg)" initial={{ opacity: 0 }}
              animate={{ opacity: r === 3 ? [0, 1] : [0, 1, 1, 0] }} transition={{ delay: T.row(r) * k, duration: (r === 3 ? 0.3 : 0.9) * k, times: r === 3 ? undefined : [0, 0.15, 0.85, 1] }}>
              {`q[${r}] = ${W[r].map((w, j) => `${w.toFixed(1)}×${x[j].toFixed(1)}`).join(" + ")} = ${state.q[r].toFixed(2)}`}
            </motion.text>
          </g>
        ))}
        <text x={RC + CELL + 14} y={Y0 + 2 * CELL + 4} fontSize={11} fill="var(--muted)">what "{TOKENS[state.query]}" is looking for</text>
      </svg>
    );
  }
  // project-k / project-v: W · X, column by column
  const isK = ph === "project-k";
  const W = isK ? params.weights.WK : params.weights.WV;
  const R = isK ? state.K : state.V;
  const Rcols = R[0].map((_, r) => R.map((v) => v[r]));
  const W0 = 100, Y0 = 80, XC = 350, RC = 580;
  const name = isK ? "k" : "v";
  return (
    <svg viewBox="0 0 800 330" className="h-auto w-full" role="img" aria-label={`Matrix multiply ${isK ? "W_K" : "W_V"} times every embedding`}>
      <Grid x={W0} y={Y0} M={W} label={isK ? "W_K" : "W_V"} rows={(isK ? QK_DIMS : DIMS).map((d) => (isK ? `is ${d}` : d))} k={k} />
      <text x={W0 + 4 * CELL + 30} y={Y0 + 2 * CELL + 6} textAnchor="middle" fontSize={20} fill="var(--muted)">×</text>
      <Grid x={XC} y={Y0} M={colsX} label="X" cols={TOKENS} k={k} />
      <text x={XC + 4 * CELL + 32} y={Y0 + 2 * CELL + 6} textAnchor="middle" fontSize={20} fill="var(--muted)">=</text>
      <motion.rect key={`${animKey}-xband`} y={Y0 - 3} width={CELL + 6} height={4 * CELL + 6} rx={6} fill="none" stroke="var(--violet)" strokeWidth={2.5}
        initial={{ x: XC - 3, opacity: 0 }} animate={{ x: [0, 1, 2, 3].map((j) => XC - 3 + j * CELL), opacity: 1 }}
        transition={{ x: { delay: T.col(0) * k, duration: (T.col(3) - T.col(0)) * k, ease: "easeInOut" }, opacity: { delay: T.col(0) * k, duration: 0.2 } }} />
      <text x={RC + 2 * CELL} y={Y0 - 30} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--fg)">{isK ? "K (keys)" : "V (values)"}</text>
      {TOKENS.map((t, j) => {
        // when keys are swapped, the two columns physically trade places after being computed
        const swapped = isK && params.swapKeys && (j === SWAP[0] || j === SWAP[1]);
        const src = swapped ? (j === SWAP[0] ? SWAP[1] : SWAP[0]) : j;
        return (
          <motion.g key={`${animKey}-col${j}`} initial={{ x: swapped ? (src - j) * CELL : 0 }} animate={{ x: 0 }} transition={{ delay: T.swap * k, duration: 0.7 * k, ease }}>
            <text x={RC + j * CELL + CELL / 2} y={Y0 - 8} textAnchor="middle" fontSize={10} fill={swapped ? "var(--warn)" : "var(--muted)"}>{name}_{t}</text>
            {[0, 1, 2, 3].map((r) => (
              <motion.g key={r} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.cell(src, r) * k + 0.4 * k, duration: 0.2 * k }}>
                <rect x={RC + j * CELL + 1} y={Y0 + r * CELL + 1} width={CELL - 2} height={CELL - 2} rx={4} {...heat(Rcols[r][j], 2)} stroke={swapped ? "var(--warn)" : "var(--line)"} />
                <text x={RC + j * CELL + CELL / 2} y={Y0 + r * CELL + CELL / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{Rcols[r][j].toFixed(2)}</text>
              </motion.g>
            ))}
          </motion.g>
        );
      })}
      {isK && params.swapKeys && (
        <motion.text x={RC + 3 * CELL} y={Y0 + 4 * CELL + 22} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--warn)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.swap * k }}>
          ⚠ keys swapped
        </motion.text>
      )}
      <text x={400} y={Y0 + 4 * CELL + 44} textAnchor="middle" fontSize={11} fill="var(--muted)">One matrix, every token: column j of the result is {isK ? "W_K" : "W_V"} · x_j</text>
    </svg>
  );
}

const ACX = (j: number) => 330 + j * 130;
function AttentionSvg({ state, c, k, animKey, swap }: { state: QKVState; c: C; k: number; animKey: string; swap: boolean }) {
  const ph = state.phase;
  const order: Phase[] = ["dot", "scale", "softmax", "mix"];
  const at = (p: Phase) => order.indexOf(ph) >= order.indexOf(p);
  const kMax = Math.max(2, ...state.K.flat().map(Math.abs));
  const vMax = Math.max(1, ...state.V.flat().map(Math.abs));
  const maxW = Math.max(...state.weights);
  const SC_Y = 175, BAR_T = 205, BAR_H = 70, V_Y = 315;
  const outMax = Math.max(vMax, ...c.partial.flat().map(Math.abs));
  return (
    <svg viewBox="0 0 800 400" className="h-auto w-full" role="img" aria-label="Query compared with keys, softmax weights and value mixing">
      {/* query card */}
      <rect x={20} y={50} width={170} height={110} rx={10} fill="var(--violet)" fillOpacity={0.08} stroke="var(--violet)" />
      <text x={105} y={72} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--violet)">q of "{TOKENS[state.query]}"</text>
      <BarStrip x={52} y={84} values={state.q} max={Math.max(2, ...state.q.map(Math.abs))} w={22} gap={6} h={56} k={k} color="var(--violet)" showValues />
      <text x={205} y={SC_Y + 4} fontSize={11} fill="var(--muted)">{ph === "dot" ? "score q·k" : "score / √d"}</text>
      <text x={205} y={BAR_T + 40} fontSize={11} fill="var(--muted)">weight</text>
      <text x={205} y={V_Y + 30} fontSize={11} fill="var(--muted)">values v</text>
      {TOKENS.map((t, j) => {
        const cx0 = ACX(j);
        return (
          <g key={t}>
            <text x={cx0} y={40} textAnchor="middle" fontSize={13} fontWeight={j === state.top ? 700 : 500} fill={j === state.top ? "var(--violet)" : "var(--fg)"}>{t}</text>
            <BarStrip x={cx0 - 42} y={58} values={state.K[j]} max={kMax} w={18} gap={4} h={56} k={k} labels={j === 0 ? QK_DIMS.map((d) => d.slice(0, 3)) : undefined} />
            <text x={cx0 + 52} y={90} fontSize={10} fill={swap && SWAP.includes(j) ? "var(--warn)" : "var(--muted)"}>{swap && SWAP.includes(j) ? "k ⚠" : "k"}</text>
            {/* score cell */}
            <motion.g key={`${ph === "dot" ? animKey : "s"}-sc${j}`} initial={ph === "dot" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: (T.dot(j) + 0.45) * k, duration: 0.25 * k }}>
              <rect x={cx0 - 36} y={SC_Y - 15} width={72} height={26} rx={5} {...heat(ph === "dot" ? state.scores[j] : state.scaled[j], ph === "dot" ? 4 : 2)} stroke={j === state.top ? "var(--fg)" : "var(--line)"} />
              <text x={cx0} y={SC_Y + 3} textAnchor="middle" fontSize={12} fontFamily="var(--font-mono)" fill="var(--fg)">
                {ph === "dot" ? <AnimatedNumber key={animKey} svg value={state.scores[j]} from={0} delay={(T.dot(j) + 0.45) * k} duration={0.3 * k} />
                  : ph === "scale" ? <AnimatedNumber key={animKey} svg value={state.scaled[j]} from={state.scores[j]} delay={0.5 * k} duration={0.8 * k} /> : state.scaled[j].toFixed(2)}
              </text>
            </motion.g>
            {/* softmax bar */}
            {at("softmax") && (
              <g>
                <rect x={cx0 - 22} y={BAR_T} width={44} height={BAR_H} rx={4} fill="var(--surface-2)" />
                <motion.rect key={`${ph === "softmax" ? animKey : "b"}-w${j}`} x={cx0 - 22} width={44} rx={4} fill={j === state.top ? "var(--violet)" : "var(--accent)"} fillOpacity={0.85}
                  initial={ph === "softmax" ? { y: BAR_T + BAR_H, height: 0 } : false} animate={{ y: BAR_T + BAR_H - (state.weights[j] / maxW) * BAR_H, height: (state.weights[j] / maxW) * BAR_H }}
                  transition={{ delay: T.bar(j) * k, duration: 0.7 * k, ease }} />
                <text x={cx0 + 30} y={BAR_T + BAR_H - 4} fontSize={12} fontFamily="var(--font-mono)" fill="var(--fg)">{(state.weights[j] * 100).toFixed(1)}%</text>
              </g>
            )}
            {/* values */}
            <BarStrip x={cx0 - 42} y={V_Y} values={state.V[j]} max={vMax} w={18} gap={4} h={56} k={k} color="var(--ok)" />
            {ph === "mix" && (
              <motion.g key={`${animKey}-pk${j}`} initial={{ x: cx0, y: V_Y + 28, opacity: 0 }} animate={{ x: [cx0, 105], y: [V_Y + 28, 250], opacity: [0, 1, 1, 0] }}
                transition={{ delay: T.packet(j) * k, duration: T.TRAVEL * k, ease: "easeIn", opacity: { delay: T.packet(j) * k, duration: T.TRAVEL * k, times: [0, 0.1, 0.85, 1] } }}>
                <circle r={5 + 12 * state.weights[j]} fill="var(--ok)" stroke="var(--surface)" strokeWidth={2} />
                <text y={-14} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">{state.weights[j].toFixed(2)}×v</text>
              </motion.g>
            )}
          </g>
        );
      })}
      {/* q visits each key during the dot step */}
      {ph === "dot" && (
        <motion.circle key={`${animKey}-qv`} r={9} fill="var(--violet)" stroke="var(--surface)" strokeWidth={2} initial={{ cx: 190, cy: 110, opacity: 0 }}
          animate={{ cx: [190, ...TOKENS.map((_, j) => ACX(j))], cy: [110, ...TOKENS.map(() => SC_Y - 22)], opacity: 1 }}
          transition={{ duration: (T.dot(3) + 0.45) * k, delay: 0.1 * k, times: [0, ...TOKENS.map((_, j) => (T.dot(j) + 0.35) / (T.dot(3) + 0.45))], ease: "easeInOut" }} />
      )}
      {/* output strip: each bar steps through the running total as packets land */}
      {ph === "mix" && (
        <g>
          <rect x={20} y={215} width={170} height={110} rx={10} fill="var(--ok)" fillOpacity={0.08} stroke="var(--ok)" />
          <text x={105} y={234} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ok)">output for "{TOKENS[state.query]}"</text>
          {[0, 1, 2, 3].map((d) => {
            const mid = 280, half = 34, s = half / Math.max(outMax, 1e-9);
            const vals = [0, ...c.partial.map((p) => p[d])];
            const ys = vals.map((v) => (v >= 0 ? mid - v * s : mid));
            const hs = vals.map((v) => Math.max(Math.abs(v) * s, 0.5));
            const times = [0, ...TOKENS.map((_, j) => (T.packet(j) + T.TRAVEL) / (T.packet(3) + T.TRAVEL))];
            return (
              <g key={d}>
                <motion.rect key={`${animKey}-o${d}`} x={50 + d * 28} width={22} rx={2} fill={state.out[d] < 0 ? "var(--err)" : "var(--ok)"} fillOpacity={0.85}
                  initial={{ y: ys[0], height: hs[0] }} animate={{ y: ys, height: hs }} transition={{ duration: (T.packet(3) + T.TRAVEL) * k, times, ease: "easeOut" }} />
                <text x={61 + d * 28} y={318} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--fg)">{state.out[d].toFixed(2)}</text>
              </g>
            );
          })}
          <line x1={44} x2={166} y1={280} y2={280} stroke="var(--line)" />
        </g>
      )}
    </svg>
  );
}

function Card({ state, c, params, k }: { state: QKVState; c: C; params: QKVParams; k: number }) {
  const ph = state.phase, tok = TOKENS[state.query];
  const row = (cells: ReactNode[], key: string | number, delay: number, cls?: string) => (
    <motion.tr key={key} className={cx("border-t border-line", cls)} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: delay * k, duration: 0.3 * k }}>
      {cells.map((x, i) => <td key={i} className={i ? "font-mono" : "py-0.5"}>{x}</td>)}
    </motion.tr>
  );
  const table = (head: string[], body: ReactNode) => (
    <div className="overflow-x-auto"><table className="w-full text-xs"><thead className="text-left text-muted"><tr>{head.map((h) => <th key={h} className="py-1 font-medium">{h}</th>)}</tr></thead><tbody>{body}</tbody></table></div>
  );
  switch (ph) {
    case "embed":
      return table(["token", ...DIMS], TOKENS.map((t, j) => row([<span className={cx(j === state.query && "font-semibold text-violet")}>{t}{j === state.query && " (query)"}</span>, ...c.X[j].map((v) => v.toFixed(1))], j, 0.2 + j * 0.3)));
    case "project-q":
      return (
        <div className="space-y-1.5">
          <p className="text-xs text-muted">Each row of W_Q is a question about x. Row r's dot product with x becomes q[r]:</p>
          {QK_DIMS.map((d, r) => (
            <motion.div key={d} className="grid grid-cols-[110px_1fr_50px] items-center gap-2 text-xs" initial={{ opacity: 0.25 }} animate={{ opacity: 1 }} transition={{ delay: (T.row(r) + 0.8) * k }}>
              <span>q[{r}] seek <strong>{d}</strong></span><HBarRaw v={state.q[r]} max={Math.max(...state.q.map(Math.abs), 1e-9)} at={T.row(r) + 0.8} tone="bg-violet/70" k={k} /><span className="text-right font-mono">{state.q[r].toFixed(2)}</span>
            </motion.div>
          ))}
          <p className="text-xs text-muted">"{tok}" mostly seeks <strong>{QK_DIMS[state.q.reduce((b, v, r) => (v > state.q[b] ? r : b), 0)]}</strong>.</p>
        </div>
      );
    case "project-k":
    case "project-v": {
      const M = ph === "project-k" ? state.K : state.V;
      return (
        <div className="space-y-1">
          <p className="text-xs text-muted">{ph === "project-k" ? "Each key describes what the token is (agent, action, modifier, drinkable):" : "Each value is the content a token hands over (in embedding feature space):"}</p>
          {table(["token", ...(ph === "project-k" ? QK_DIMS : DIMS)], TOKENS.map((t, j) => row([<span className={cx(params.swapKeys && ph === "project-k" && SWAP.includes(j) && "text-warn")}>{t}{params.swapKeys && ph === "project-k" && SWAP.includes(j) ? " ⚠ swapped" : ""}</span>, ...M[j].map((v) => v.toFixed(2))], j, T.col(j) + 0.5)))}
        </div>
      );
    }
    case "dot":
      return (
        <div className="space-y-1.5">
          {TOKENS.map((t, j) => (
            <div key={t} className="grid grid-cols-[60px_1fr_220px] items-center gap-2 text-xs">
              <span className={cx(j === state.top && "font-semibold text-violet")}>{t}</span>
              <HBarRaw v={state.scores[j]} max={Math.max(...state.scores.map(Math.abs), 1e-9)} at={T.dot(j) + 0.45} k={k} tone={j === state.top ? "bg-violet/70" : "bg-accent"} />
              <span className="truncate text-right font-mono">{state.q.map((v, d) => `${v.toFixed(1)}·${state.K[j][d].toFixed(1)}`).join("+")} = {state.scores[j].toFixed(2)}</span>
            </div>
          ))}
        </div>
      );
    case "scale":
    case "softmax": {
      const ex = state.scaled.map(Math.exp), sum = ex.reduce((a, b) => a + b, 0);
      return table(["token", "q·k", "÷ √4", "e^s", `weight (÷ ${sum.toFixed(2)})`], TOKENS.map((t, j) => row([t, state.scores[j].toFixed(2), state.scaled[j].toFixed(2), ph === "softmax" ? ex[j].toFixed(3) : "…", ph === "softmax" ? `${(state.weights[j] * 100).toFixed(1)}%` : "…"], j, T.bar(j), j === state.top ? "font-semibold text-violet" : undefined)));
    }
    case "mix":
      return (
        <div className="space-y-1">
          {table(["token", "weight", "× value", "= contribution"], [
            ...TOKENS.map((t, j) => row([t, state.weights[j].toFixed(3), `[${state.V[j].map((v) => v.toFixed(1)).join(", ")}]`, `[${state.V[j].map((v) => (v * state.weights[j]).toFixed(2)).join(", ")}]`], j, T.packet(j) + T.TRAVEL)),
            row([<strong>sum = out</strong>, "1.000", "", <strong>[{state.out.map((v) => v.toFixed(2)).join(", ")}]</strong>], "sum", T.packet(3) + T.TRAVEL + 0.2),
          ])}
          <p className="text-xs text-muted">{params.swapKeys ? `Because the keys were swapped, the biggest share now comes from "${TOKENS[state.top]}"'s value, not the token the query was really looking for.` : mixNote(state, tok)}</p>
        </div>
      );
  }
}

function mixNote(s: QKVState, tok: string) {
  const d = s.out.reduce((b, v, i) => (Math.abs(v) > Math.abs(s.out[b]) ? i : b), 0);
  const j = s.weights.reduce((b, w, i) => (Math.abs(w * s.V[i][d]) > Math.abs(s.weights[b] * s.V[b][d]) ? i : b), 0);
  return `The largest output dimension of "${tok}" is "${DIMS[d]}" (${s.out[d].toFixed(2)}), and most of it comes from "${TOKENS[j]}"'s value.`;
}

function Calculation({ phase, c, params }: { phase: Phase; c: C; params: QKVParams }) {
  const tok = TOKENS[c.i], top = TOKENS[c.top];
  const x = c.X[c.i];
  const tex: Record<Phase, string> = {
    embed: `x_{\\text{${tok}}} = [${x.map((v) => v.toFixed(1)).join(",\\ ")}]`,
    "project-q": `q = ${texMat(params.weights.WQ)} \\begin{bmatrix} ${x.map((v) => v.toFixed(1)).join(" \\\\ ")} \\end{bmatrix} = \\begin{bmatrix} ${c.q.map((v) => v.toFixed(2)).join(" \\\\ ")} \\end{bmatrix}`,
    "project-k": `k_{\\text{${top}}} = W_K\\, x_{\\text{${top}}} = [${c.K[c.top].map((v) => v.toFixed(2)).join(",\\ ")}]`,
    "project-v": `v_{\\text{${top}}} = W_V\\, x_{\\text{${top}}} = [${c.V[c.top].map((v) => v.toFixed(2)).join(",\\ ")}]`,
    dot: `q \\cdot k_{\\text{${top}}} = ${c.q.map((v, d) => `${tn(v)} \\cdot ${tn(c.K[c.top][d])}`).join(" + ")} = ${c.scores[c.top].toFixed(2)}`,
    scale: `\\frac{q \\cdot k}{\\sqrt{d}} = \\frac{${c.scores[c.top].toFixed(2)}}{2} = ${c.scaled[c.top].toFixed(2)}`,
    softmax: `w_{\\text{${top}}} = \\frac{e^{${c.scaled[c.top].toFixed(2)}}}{${c.scaled.map((s) => `e^{${s.toFixed(2)}}`).join(" + ")}} = ${c.weights[c.top].toFixed(3)}`,
    mix: `\\text{out} = ${c.weights.map((w, j) => `${w.toFixed(2)}\\, v_{\\text{${TOKENS[j]}}}`).join(" + ")} = [${c.out.map((v) => v.toFixed(2)).join(",\\ ")}]`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}
