import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { BarStrip, StepHeader, ease, heat } from "../self-attention/shared";
import type { Vec } from "../self-attention/attention";
import { DEFAULT_PARAMS, PRESETS, ROPE_K, ROPE_Q, orderCheck, peMatrix, ropeDot, simulate, sinusoidalPE, type Phase, type PosParams, type PresetId } from "./simulation";

const DURATIONS: Record<Phase, number> = { words: 2600, "attn-a": 2600, "attn-b": 4200, "pe-matrix": 4000, "pe-add": 3600, "attn-pe": 4200, "rope-rotate": 3400, "rope-relative": 3400 };
const NARR: Record<Phase, { headline: string; sub: string }> = {
  words: { headline: "Two sentences, the same three words", sub: "Order is the only thing that tells you who bit whom." },
  "attn-a": { headline: "Self-attention on order A, with no position information", sub: "Each word's output is a weighted mix of all three embeddings." },
  "attn-b": { headline: "Shuffle the words and recompute: nothing changes", sub: "Each word gets exactly the same output. Attention alone cannot see order." },
  "pe-matrix": { headline: "Give every position its own fingerprint", sub: "Sinusoidal encodings: sines and cosines at different frequencies." },
  "pe-add": { headline: "Add the position vector to each word's embedding", sub: "The result strip morphs from x to x + PE." },
  "attn-pe": { headline: "Recompute both orders: now the outputs differ", sub: "The same word at a different position now produces a different vector." },
  "rope-rotate": { headline: "RoPE: rotate the query and key by their positions", sub: "Instead of adding a vector, each 2-D pair is turned by position × θ." },
  "rope-relative": { headline: "Shift both positions: q·k does not change", sub: "Both arrows turn by the same angle, so the angle between them is fixed." },
};
const SX = (s: number) => 250 + s * 200;

export default function PositionalVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<PosParams>(DEFAULT_PARAMS);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(params), [params]);
  const none = useMemo(() => orderCheck(params.preset, 0), [params.preset]);
  const withPe = useMemo(() => orderCheck(params.preset, params.pe), [params.preset, params.pe]);
  const pb = usePlayback(steps.length, (i) => DURATIONS[steps[i]?.state.phase ?? "words"]);
  const step = steps[pb.index];
  const ph = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<PosParams>) => setParams((o) => ({ ...o, ...p }));
  if (tab === "metrics") return null;
  const r0 = ropeDot(params.m, params.n, params.theta);
  const r1 = ropeDot(params.m + params.shift, params.n + params.shift, params.theta);

  return (
    <Workspace
      main={
        <>
          <Panel title="Positional information" actions={<SimulatedLabel>hand-placed d = 4 vectors</SimulatedLabel>}>
            <StepHeader index={pb.index} total={steps.length} stepId={step.id} headline={NARR[ph].headline} sub={NARR[ph].sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[620px]">
                {ph === "pe-matrix" ? <PeSvg k={k} animKey={animKey} />
                  : ph === "pe-add" ? <AddSvg params={params} check={withPe} k={k} animKey={animKey} />
                  : ph === "rope-rotate" || ph === "rope-relative" ? <RopeSvg params={params} relative={ph === "rope-relative"} k={k} animKey={animKey} />
                  : <OrderSvg phase={ph} check={ph === "attn-pe" ? withPe : none} pe={ph === "attn-pe" ? params.pe : 0} k={k} animKey={animKey} />}
              </div>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <Card phase={ph} params={params} none={none} withPe={withPe} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_PARAMS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select<PresetId> label="Sentence pair" value={params.preset} options={PRESETS.map((p) => ({ value: p.id, label: `${p.a.join(" ")} ↔ ${p.b.join(" ")}` }))} onChange={(v) => set({ preset: v })} />
              <Slider label="PE strength" value={params.pe} min={0} max={2} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ pe: v })} hint="0 = no positions (break it): order-blind again." />
              <Slider label="RoPE θ (radians per position)" value={params.theta} min={0.1} max={1.5} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ theta: v })} />
              <Slider label="Query position m" value={params.m} min={0} max={8} onChange={(v) => set({ m: v })} />
              <Slider label="Key position n" value={params.n} min={0} max={8} onChange={(v) => set({ n: v })} />
              <Slider label="Shift both by s" value={params.shift} min={0} max={6} onChange={(v) => set({ shift: v })} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation phase={ph} params={params} none={none} withPe={withPe} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["max |Δ| no PE", none.maxDiff.toFixed(6)],
              [`max |Δ| PE × ${params.pe.toFixed(1)}`, withPe.maxDiff.toFixed(4)],
              [`RoPE q·k (m=${params.m}, n=${params.n})`, r0.dot.toFixed(4)],
              [`RoPE q·k (m=${params.m + params.shift}, n=${params.n + params.shift})`, r1.dot.toFixed(4)],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

type Check = ReturnType<typeof orderCheck>;

function OrderSvg({ phase, check, pe, k, animKey }: { phase: Phase; check: Check; pe: number; k: number; animKey: string }) {
  const { preset, A, B, rows } = check;
  const max = Math.max(1, ...A.out.flat().map(Math.abs), ...B.out.flat().map(Math.abs));
  const showA = phase !== "words", showB = phase === "attn-b" || phase === "attn-pe";
  const RA = 40, RB = 245;
  const slide = phase === "attn-b" || phase === "words";
  return (
    <svg viewBox="0 0 800 420" className="h-auto w-full" role="img" aria-label="Self-attention outputs for two word orders">
      {[["Order A", RA], ["Order B", RB]].map(([l, y]) => (
        <text key={l} x={20} y={Number(y) + 20} fontSize={13} fontWeight={700} fill="var(--fg)">{l}</text>
      ))}
      {pe > 0 && <text x={20} y={RA + 38} fontSize={10} fill="var(--violet)">+ PE × {pe.toFixed(1)}</text>}
      {pe > 0 && <text x={20} y={RB + 38} fontSize={10} fill="var(--violet)">+ PE × {pe.toFixed(1)}</text>}
      {preset.a.map((w, s) => (
        <g key={`a${s}`}>
          <rect x={SX(s) - 55} y={RA} width={110} height={30} rx={7} fill="var(--surface)" stroke="var(--line)" />
          <text x={SX(s)} y={RA + 20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--fg)">{w}<tspan fontSize={10} fill="var(--muted)"> @{s}</tspan></text>
          {showA && <BarStrip key={`${phase === "attn-a" || phase === "attn-pe" ? animKey : "a"}-oa${s}`} x={SX(s) - 62} y={RA + 50} values={A.out[s]} max={max} w={24} gap={8} h={56} k={k} delay={0.3 + s * 0.3} showValues />}
        </g>
      ))}
      {preset.b.map((w, s) => {
        const from = preset.a.indexOf(w);
        return (
          <motion.g key={`${slide ? animKey : "b"}-b${s}`} initial={slide ? { x: SX(from) - SX(s), y: RA - RB, opacity: 0.4 } : false} animate={{ x: 0, y: 0, opacity: 1 }} transition={{ delay: 0.3 * k, duration: 1.1 * k, ease }}>
            <rect x={SX(s) - 55} y={RB} width={110} height={30} rx={7} fill="var(--accent-soft)" stroke="var(--accent)" />
            <text x={SX(s)} y={RB + 20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--fg)">{w}<tspan fontSize={10} fill="var(--muted)"> @{s}</tspan></text>
          </motion.g>
        );
      })}
      {showB && preset.b.map((_, s) => (
        <BarStrip key={`${animKey}-ob${s}`} x={SX(s) - 62} y={RB + 50} values={B.out[s]} max={max} w={24} gap={8} h={56} k={k} delay={1.5 + s * 0.3} showValues />
      ))}
      {/* link each word's two outputs and report the difference */}
      {showB && rows.map((r, i) => {
        const x1 = SX(r.posA), y1 = RA + 122, x2 = SX(r.posB), y2 = RB - 6;
        const same = r.diff < 1e-9;
        const lx = x1 + (x2 - x1) * 0.82, ly = y1 + (y2 - y1) * 0.82;
        return (
          <g key={`${animKey}-link${i}`}>
            <motion.line x1={x1} y1={y1} x2={x2} y2={y2} stroke={same ? "var(--ok)" : "var(--warn)"} strokeWidth={2} strokeDasharray="5 4"
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: (2.6 + i * 0.35) * k, duration: 0.5 * k }} />
            <motion.g initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (3.0 + i * 0.35) * k, duration: 0.3 * k }} style={{ transformOrigin: `${lx}px ${ly}px` }}>
              <rect x={lx - 44} y={ly - 11} width={88} height={22} rx={11} fill={same ? "var(--ok-soft)" : "var(--warn-soft)"} stroke={same ? "var(--ok)" : "var(--warn)"} />
              <text x={lx} y={ly + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{same ? "✓ identical" : `≠ Δ ${r.diff.toFixed(3)}`}</text>
            </motion.g>
          </g>
        );
      })}
      {showB && (
        <motion.text x={400} y={410} textAnchor="middle" fontSize={12} fontWeight={600} fill={check.maxDiff < 1e-9 ? "var(--ok)" : "var(--warn)"} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.8 * k }}>
          {check.maxDiff < 1e-9 ? "Every word's output is the same in both orders: max |Δ| = 0.000000" : `Order now matters: max |Δ| = ${check.maxDiff.toFixed(4)}`}
        </motion.text>
      )}
    </svg>
  );
}

function PeSvg({ k, animKey }: { k: number; animKey: string }) {
  const N = 8, P = peMatrix(N), X0 = 60, Y0 = 60, CW = 64, CH = 32;
  const heads = ["sin(p)", "cos(p)", "sin(p/100)", "cos(p/100)"];
  const PX0 = 400, PW = 370, PY = (v: number) => 200 - v * 110, px = (p: number) => PX0 + (p / (N - 1)) * PW;
  const curve = (f: (p: number) => number) => Array.from({ length: 71 }, (_, i) => (i / 70) * (N - 1)).map((p, i) => `${i ? "L" : "M"}${px(p).toFixed(1)} ${PY(f(p)).toFixed(1)}`).join(" ");
  const fns = [(p: number) => Math.sin(p), (p: number) => Math.cos(p), (p: number) => Math.sin(p / 100), (p: number) => Math.cos(p / 100)];
  const colors = ["var(--accent)", "var(--violet)", "var(--ok)", "var(--warn)"];
  return (
    <svg viewBox="0 0 800 340" className="h-auto w-full" role="img" aria-label="Sinusoidal positional encoding matrix and its waves">
      {heads.map((h, j) => <text key={h} x={X0 + j * CW + CW / 2} y={Y0 - 8} textAnchor="middle" fontSize={10} fill={colors[j]}>{h}</text>)}
      {P.map((row, p) => (
        <g key={p}>
          <text x={X0 - 8} y={Y0 + p * CH + CH / 2 + 4} textAnchor="end" fontSize={11} fill="var(--muted)">pos {p}</text>
          {row.map((v, j) => (
            <motion.g key={`${animKey}-${p}-${j}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + p * 0.25 + j * 0.05) * k, duration: 0.25 * k }}>
              <rect x={X0 + j * CW + 1} y={Y0 + p * CH + 1} width={CW - 2} height={CH - 2} rx={3} {...heat(v, 1)} />
              <text x={X0 + j * CW + CW / 2} y={Y0 + p * CH + CH / 2 + 4} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">{v.toFixed(2)}</text>
            </motion.g>
          ))}
        </g>
      ))}
      <line x1={PX0} x2={PX0 + PW} y1={PY(0)} y2={PY(0)} stroke="var(--line)" />
      <text x={PX0 + PW} y={PY(0) + 14} textAnchor="end" fontSize={10} fill="var(--muted)">position →</text>
      <text x={PX0} y={PY(1) - 8} fontSize={10} fill="var(--muted)">value (−1 … 1)</text>
      {fns.map((f, j) => (
        <g key={`${animKey}-c${j}`}>
          <motion.path d={curve(f)} fill="none" stroke={colors[j]} strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: (0.3 + j * 0.4) * k, duration: 1.6 * k, ease: "linear" }} />
          {P.map((row, p) => (
            <motion.circle key={p} cx={px(p)} cy={PY(row[j])} r={3} fill={colors[j]} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: (0.3 + j * 0.4 + (p / (N - 1)) * 1.6) * k, duration: 0.2 * k }} />
          ))}
        </g>
      ))}
      <text x={PX0 + PW / 2} y={330} textAnchor="middle" fontSize={11} fill="var(--muted)">Low dimensions oscillate fast; high dimensions (÷100) barely move over 8 positions.</text>
    </svg>
  );
}

function AddSvg({ params, check, k, animKey }: { params: PosParams; check: Check; k: number; animKey: string }) {
  const { A, preset } = check;
  const max = Math.max(1.5, ...A.Xp.flat().map(Math.abs));
  return (
    <svg viewBox="0 0 800 370" className="h-auto w-full" role="img" aria-label="Embedding plus positional encoding">
      {["embedding x", `PE(pos) × ${params.pe.toFixed(1)}`, "x + PE"].map((h, i) => <text key={h} x={264 + i * 200} y={20} textAnchor="middle" fontSize={11} fontWeight={600} fill={i === 1 ? "var(--violet)" : "var(--fg)"}>{h}</text>)}
      {preset.a.map((w, s) => {
        const y = 40 + s * 110;
        return (
          <g key={`${animKey}-${s}`}>
            <text x={20} y={y + 35} fontSize={13} fontWeight={600} fill="var(--fg)">{w}<tspan fontSize={10} fill="var(--muted)"> @ pos {s}</tspan></text>
            <BarStrip x={200} y={y} values={A.X[s]} max={max} w={24} gap={8} h={70} k={k} delay={0.1 + s * 0.2} showValues />
            <text x={364} y={y + 42} textAnchor="middle" fontSize={20} fill="var(--muted)">+</text>
            <BarStrip x={400} y={y} values={A.P[s]} max={max} w={24} gap={8} h={70} k={k} delay={0.6 + s * 0.3} color="var(--violet)" showValues />
            <text x={564} y={y + 42} textAnchor="middle" fontSize={20} fill="var(--muted)">=</text>
            {/* the result starts as x and morphs into x + PE */}
            <BarStrip x={600} y={y} values={A.Xp[s]} from={A.X[s]} max={max} w={24} gap={8} h={70} k={k} delay={1.6 + s * 0.4} color="var(--ok)" showValues />
            <motion.circle r={6} fill="var(--violet)" initial={{ cx: 464, cy: y + 35, opacity: 0 }} animate={{ cx: [464, 664], opacity: [0, 1, 1, 0] }} transition={{ delay: (1.1 + s * 0.4) * k, duration: 0.6 * k }} />
          </g>
        );
      })}
    </svg>
  );
}

const arcPts = (v: Vec, a0: number, a1: number, cx: number, cy: number, R: number, n = 16) =>
  Array.from({ length: n + 1 }, (_, i) => {
    const a = a0 + ((a1 - a0) * i) / n;
    return [cx + R * (v[0] * Math.cos(a) - v[1] * Math.sin(a)), cy - R * (v[0] * Math.sin(a) + v[1] * Math.cos(a))];
  });

function RopeSvg({ params, relative, k, animKey }: { params: PosParams; relative: boolean; k: number; animKey: string }) {
  const { m, n, theta, shift } = params;
  const CX = 250, CY = 190, R = 130;
  const [qa0, qa1] = relative ? [m * theta, (m + shift) * theta] : [0, m * theta];
  const [ka0, ka1] = relative ? [n * theta, (n + shift) * theta] : [0, n * theta];
  const qp = arcPts(ROPE_Q, qa0, qa1, CX, CY, R), kp = arcPts(ROPE_K, ka0, ka1, CX, CY, R);
  const tr = { delay: 0.4 * k, duration: 1.6 * k, ease: "easeInOut" as const };
  const before = ropeDot(relative ? m : 0, relative ? n : 0, theta).dot;
  const after = ropeDot(relative ? m + shift : m, relative ? n + shift : n, theta).dot;
  const vec = (pts: number[][], color: string, label: string) => (
    <g>
      <motion.line x1={CX} y1={CY} stroke={color} strokeWidth={3.5} strokeLinecap="round" initial={{ x2: pts[0][0], y2: pts[0][1] }} animate={{ x2: pts.map((p) => p[0]), y2: pts.map((p) => p[1]) }} transition={tr} />
      <motion.circle r={7} fill={color} initial={{ cx: pts[0][0], cy: pts[0][1] }} animate={{ cx: pts.map((p) => p[0]), cy: pts.map((p) => p[1]) }} transition={tr} />
      <motion.text fontSize={13} fontWeight={700} fill={color} initial={{ x: pts[0][0] + 10, y: pts[0][1] - 10 }} animate={{ x: pts.map((p) => p[0] + 10), y: pts.map((p) => p[1] - 10) }} transition={tr}>{label}</motion.text>
    </g>
  );
  return (
    <svg viewBox="0 0 800 380" className="h-auto w-full" role="img" aria-label="RoPE rotating query and key vectors">
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--line)" strokeDasharray="3 4" />
      <line x1={CX - R - 20} x2={CX + R + 20} y1={CY} y2={CY} stroke="var(--line)" />
      <line x1={CX} x2={CX} y1={CY - R - 20} y2={CY + R + 20} stroke="var(--line)" />
      {/* ghost of the starting vectors */}
      <line x1={CX} y1={CY} x2={qp[0][0]} y2={qp[0][1]} stroke="var(--violet)" strokeOpacity={0.25} strokeWidth={2} strokeDasharray="4 3" />
      <line x1={CX} y1={CY} x2={kp[0][0]} y2={kp[0][1]} stroke="var(--accent)" strokeOpacity={0.25} strokeWidth={2} strokeDasharray="4 3" />
      <g key={animKey}>
        {vec(qp, "var(--violet)", relative ? `q @ ${m + shift}` : `q @ ${m}`)}
        {vec(kp, "var(--accent)", relative ? `k @ ${n + shift}` : `k @ ${n}`)}
      </g>
      <g fontSize={12} fill="var(--fg)">
        <text x={460} y={60} fontWeight={700}>{relative ? `Shift both by s = ${shift}` : "Rotate by position"}</text>
        <text x={460} y={88}>q turns by {relative ? `${shift}·θ` : `m·θ = ${m}·${theta.toFixed(2)}`} = {((qa1 - qa0) * 180 / Math.PI).toFixed(0)}°</text>
        <text x={460} y={110}>k turns by {relative ? `${shift}·θ` : `n·θ = ${n}·${theta.toFixed(2)}`} = {((ka1 - ka0) * 180 / Math.PI).toFixed(0)}°</text>
        <text x={460} y={140}>gap between them: (n − m)·θ = {((n - m) * theta * 180 / Math.PI).toFixed(0)}°{relative ? " (unchanged)" : ""}</text>
        <text x={460} y={180} fontWeight={700}>q · k</text>
        <text x={520} y={180} fontFamily="var(--font-mono)" fontSize={16} fontWeight={700} fill={relative ? "var(--ok)" : "var(--fg)"}>
          <AnimatedNumber key={animKey} svg value={after} from={before} delay={0.4 * k} duration={1.6 * k} digits={4} />
        </text>
        <text x={460} y={206} fill="var(--muted)" fontSize={11}>{relative ? "Same number: RoPE scores depend only on n − m." : `Before rotation: ${before.toFixed(4)}`}</text>
      </g>
    </svg>
  );
}

function Card({ phase, params, none, withPe, k }: { phase: Phase; params: PosParams; none: Check; withPe: Check; k: number }) {
  const row = (cells: ReactNode[], key: string | number, delay: number) => (
    <motion.tr key={key} className="border-t border-line" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: delay * k, duration: 0.3 * k }}>
      {cells.map((x, i) => <td key={i} className={cx("py-0.5", i > 0 && "font-mono")}>{x}</td>)}
    </motion.tr>
  );
  const table = (head: string[], body: ReactNode) => (
    <div className="overflow-x-auto"><table className="w-full text-xs"><thead className="text-left text-muted"><tr>{head.map((h) => <th key={h} className="py-1 font-medium">{h}</th>)}</tr></thead><tbody>{body}</tbody></table></div>
  );
  const v = (x: Vec, d = 3) => `[${x.map((y) => y.toFixed(d)).join(", ")}]`;
  const diffTable = (c: Check, delay: number) => table(["word", "pos A → B", "out (order A)", "out (order B)", "max |Δ|"], c.rows.map((r, i) => row([
    r.word, `${r.posA} → ${r.posB}`, v(r.outA, 2), v(r.outB, 2),
    r.diff < 1e-9 ? <Badge tone="ok">0.000 same</Badge> : <Badge tone="warn">{r.diff.toFixed(3)} differs</Badge>,
  ], i, delay + i * 0.35)));
  switch (phase) {
    case "words":
      return <p className="text-sm leading-relaxed">"{none.preset.a.join(" ")}" vs "{none.preset.b.join(" ")}". A model that cannot tell these apart cannot tell who did what to whom. Watch what self-attention does with only the word embeddings.</p>;
    case "attn-a":
      return table(["word @ pos", "embedding x", "attention output"], none.rows.map((r, i) => row([`${r.word} @ ${r.posA}`, v(none.A.X[r.posA], 1), v(r.outA)], i, 0.3 + i * 0.3)));
    case "attn-b":
      return <div className="space-y-2">{diffTable(none, 2.6)}<p className="text-xs text-muted">Permuting the input just permutes the outputs. Scores q·k and values depend only on the word vectors, and softmax plus the weighted sum treat the set of tokens the same regardless of order.</p></div>;
    case "pe-matrix":
      return table(["pos", "sin(p)", "cos(p)", "sin(p/100)", "cos(p/100)"], [0, 1, 2].map((p) => row([`${p} (${none.preset.a[p]})`, ...sinusoidalPE(p).map((x) => x.toFixed(3))], p, 0.2 + p * 0.25)));
    case "pe-add":
      return table(["word @ pos", "x", `${params.pe.toFixed(1)}·PE`, "x + PE"], withPe.preset.a.map((w, s) => row([`${w} @ ${s}`, v(withPe.A.X[s], 2), v(withPe.A.P[s], 2), v(withPe.A.Xp[s], 2)], s, 1.6 + s * 0.4)));
    case "attn-pe":
      return <div className="space-y-2">{diffTable(withPe, 2.6)}<p className="text-xs text-muted">{withPe.maxDiff < 1e-9 ? "PE strength is 0: no position signal, so the model is order-blind again." : "Each word's input now depends on its position, so its scores and output change when the order changes. The middle word keeps its position, but its neighbours moved, so it changes too."}</p></div>;
    case "rope-rotate":
    case "rope-relative": {
      const pairs = [[params.m, params.n], [params.m + 1, params.n + 1], [params.m + params.shift, params.n + params.shift], [params.m, params.n + 1]];
      const base = ropeDot(params.m, params.n, params.theta).dot;
      return table(["m (query pos)", "n (key pos)", "n − m", "q·k"], pairs.map(([m, n], i) => {
        const d = ropeDot(m, n, params.theta).dot;
        return row([m, n, n - m, <span className={Math.abs(d - base) < 1e-9 ? "text-ok" : "text-warn"}>{d.toFixed(4)} {Math.abs(d - base) < 1e-9 ? "✓ same" : "differs"}</span>], i, 0.4 + i * 0.3);
      }));
    }
  }
}

function Calculation({ phase, params, none, withPe }: { phase: Phase; params: PosParams; none: Check; withPe: Check }) {
  const r = ropeDot(params.m, params.n, params.theta);
  const tex: Record<Phase, string> = {
    words: `\\text{order A: } (${none.preset.a.join(",\\ ")}) \\quad \\text{order B: } (${none.preset.b.join(",\\ ")})`,
    "attn-a": `\\text{out}_i = \\sum_j \\operatorname{softmax}_j\\!\\left(\\tfrac{q_i \\cdot k_j}{\\sqrt d}\\right) v_j`,
    "attn-b": `\\operatorname{Attn}(PX) = P\\,\\operatorname{Attn}(X) \\quad \\max|\\Delta| = ${none.maxDiff.toFixed(6)}`,
    "pe-matrix": `\\begin{aligned} PE_{(p,\\,2i)} &= \\sin\\!\\left(p / 10000^{2i/d}\\right) \\\\ PE_{(p,\\,2i+1)} &= \\cos\\!\\left(p / 10000^{2i/d}\\right) \\end{aligned}`,
    "pe-add": `x'_{\\text{${withPe.preset.a[0]}}} = x + ${params.pe.toFixed(1)}\\,PE(0) = [${withPe.A.Xp[0].map((v) => v.toFixed(2)).join(",\\ ")}]`,
    "attn-pe": `\\max|\\text{out}_A - \\text{out}_B| = ${withPe.maxDiff.toFixed(4)}`,
    "rope-rotate": `R(m\\theta)\\,q \\cdot R(n\\theta)\\,k = q^\\top R\\big((n-m)\\theta\\big)\\,k = ${r.dot.toFixed(4)}`,
    "rope-relative": `R(\\phi) = \\begin{bmatrix} \\cos\\phi & -\\sin\\phi \\\\ \\sin\\phi & \\cos\\phi \\end{bmatrix},\\quad n - m = ${params.n - params.m}`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}
