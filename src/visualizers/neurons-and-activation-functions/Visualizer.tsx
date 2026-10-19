import { useId, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { clamp } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "./kit";
import {
  ACTS, ACT_NAMES, LEAKY_ALPHA, LIMITS, PHASES, PRESETS, SATURATION, chainedGradient, compareAll, flatRegions, simulate,
  type ActName, type NeuronInput, type NeuronResult, type Phase, type Zone,
} from "./simulation";

/* ---------- one timing schedule (seconds at 1×) shared by the diagram, plots and card ---------- */
const TRAVEL = 0.8;
const pulse = (i: number) => 0.2 + i * 0.85;
const arrive = (i: number) => pulse(i) + TRAVEL;
const BIAS = arrive(2) + 0.2;
const SUM = BIAS + 0.6;
const ACT = { curve: 0, zMark: 0.5, rise: 0.9, across: 1.7 };
const DER = { curve: 0, shade: 0.4, tangent: 0.8, dot: 1.2, chain: 1.5 };
const CMP = { stagger: 0.22, dots: 1.4 };
const DURATIONS: Record<Phase, number> = { inputs: 1900, sum: 4900, activate: 3300, output: 2400, derivative: 4200, compare: 3600 };

const sub = (i: number) => "₁₂₃₄₅"[i];
const n2 = (v: number) => (v < 0 ? `(${v.toFixed(2)})` : v.toFixed(2));

const ACT_TEXT: Record<ActName, string> = {
  relu: "ReLU keeps a positive z as it is and turns a negative z into exactly 0.",
  leaky: `Leaky ReLU is ReLU with a small slope (${LEAKY_ALPHA}) for negative z, so it never goes completely silent.`,
  sigmoid: "Sigmoid squashes any z into 0…1. Far from 0 the curve goes flat.",
  tanh: "Tanh squashes any z into −1…1 and is centred on 0. Far from 0 it also goes flat.",
  gelu: "GELU is a smooth ReLU: it lets small negative z through a little, then fades to 0. Used in Transformers.",
};

function narrate(phase: Phase, act: ActName, r: NeuronResult) {
  const L = ACTS[act].label;
  switch (phase) {
    case "inputs": return { headline: "Three numbers arrive at the neuron", sub: "Each one travels on its own connection, and each connection has a weight." };
    case "sum": return { headline: "Each input is multiplied by its weight, then everything is added", sub: "Watch the products travel in and the bias drop in from above; the counter is z." };
    case "activate": return { headline: `z = ${r.z.toFixed(2)} goes through ${L}`, sub: "Find z on the horizontal axis, go up to the curve, read the output on the vertical axis." };
    case "output": return { headline: `The neuron outputs a = ${r.a.toFixed(3)}`, sub: "This is the number the next layer receives." };
    case "derivative": return {
      headline: r.zone === "dead" ? "The slope here is 0: this neuron cannot learn from this input" : r.zone === "vanishing" ? "The curve is almost flat here: the gradient vanishes" : `The slope at z is ${r.d.toFixed(3)}: learning signal gets through`,
      sub: "Training multiplies the error signal by this slope. A flat curve means almost no update.",
    };
    case "compare": return { headline: `Same z = ${r.z.toFixed(2)}, five different activations`, sub: "Each curve gives a different output and a different slope for the same input." };
  }
}

export default function NeuronsVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<NeuronInput>(PRESETS[0].input);
  const [overlay, setOverlay] = useState(false);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(input), [input]);
  const pb = usePlayback(steps.length, (i) => DURATIONS[PHASES[i]] ?? 2500);
  if (tab === "metrics") return null;

  const step = steps[pb.index];
  const { phase, r } = step.state;
  const pi = PHASES.indexOf(phase);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(phase, input.act, r);
  const set = (patch: Partial<NeuronInput>) => setInput((s) => ({ ...s, ...patch }));
  const setArr = (key: "x" | "w", i: number, v: number) => setInput((s) => ({ ...s, [key]: s[key].map((o, j) => (j === i ? v : o)) }));
  const compare = phase === "compare";
  const plotActs: ActName[] = compare || overlay ? ACT_NAMES : [input.act];

  return (
    <Workspace
      main={
        <>
          <Panel title="A single neuron" actions={<span className="text-[11px] text-muted">real arithmetic on the numbers shown</span>}>
            <StepCaption index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((v) => v + 1)} />
            <div className="grid items-start gap-3 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
              <NeuronSvg input={input} r={r} pi={pi} phase={phase} k={k} animKey={animKey} />
              <div className="space-y-1">
                <Plot kind="f" acts={plotActs} main={input.act} z={r.z} phase={phase} pi={pi} k={k} animKey={animKey} d={r.d} />
                <Plot kind="d" acts={plotActs} main={input.act} z={r.z} phase={phase} pi={pi} k={k} animKey={animKey} d={r.d} />
              </div>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard phase={phase} input={input} r={r} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Inputs, weights, activation" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(PRESETS[0].input); setOverlay(false); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Activation function" value={input.act} options={ACT_NAMES.map((a) => ({ value: a, label: ACTS[a].label }))} onChange={(act) => set({ act })} />
              <Slider label="bias b" value={input.b} min={-LIMITS.b} max={LIMITS.b} step={0.1} format={(v) => v.toFixed(1)} onChange={(b) => set({ b })} hint="A large negative bias can kill a ReLU." />
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              {input.x.map((xv, i) => (
                <div key={i} className="space-y-2 rounded-md border border-line p-2">
                  <Slider label={`input x${sub(i)}`} value={xv} min={-LIMITS.x} max={LIMITS.x} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => setArr("x", i, v)} />
                  <Slider label={`weight w${sub(i)}`} value={input.w[i]} min={-LIMITS.w} max={LIMITS.w} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => setArr("w", i, v)} />
                </div>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Presets:</span>
              {PRESETS.map((p) => <Button key={p.id} size="sm" onClick={() => setInput(p.input)}>{p.label}</Button>)}
            </div>
            <div className="mt-3"><Toggle label="Compare all activations on the plots" checked={overlay} onChange={setOverlay} hint="Overlay every curve in every step (the last step always does)." /></div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation phase={phase} input={input} r={r} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["weighted sum z", r.z.toFixed(4)],
              [`output a = ${ACTS[input.act].label}(z)`, r.a.toFixed(4)],
              ["slope f'(z)", r.d.toFixed(4)],
              ["gradient zone", <ZoneBadge key="z" zone={r.zone} />],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function ZoneBadge({ zone }: { zone: Zone }) {
  if (zone === "dead") return <Badge tone="err"><XCircle size={11} aria-hidden /> dead (slope 0)</Badge>;
  if (zone === "vanishing") return <Badge tone="warn"><AlertTriangle size={11} aria-hidden /> vanishing</Badge>;
  return <Badge tone="ok"><CheckCircle2 size={11} aria-hidden /> healthy</Badge>;
}

/* ---------- neuron diagram ---------- */
const IN_POS: [number, number][] = [[48, 52], [48, 128], [48, 204]];
const NEU: [number, number] = [232, 128];
const OUT: [number, number] = [362, 128];

function NeuronSvg({ input, r, pi, phase, k, animKey }: { input: NeuronInput; r: NeuronResult; pi: number; phase: Phase; k: number; animKey: string }) {
  const summing = phase === "sum";
  const activating = phase === "activate";
  return (
    <svg viewBox="0 0 400 250" className="h-auto w-full" role="img" aria-label={`Neuron: inputs ${input.x.join(", ")}, weights ${input.w.join(", ")}, bias ${input.b}, z ${r.z.toFixed(2)}, output ${r.a.toFixed(2)}`}>
      {IN_POS.map(([x, y], i) => {
        const w = input.w[i];
        const lx = x + (NEU[0] - x) * 0.45, ly = y + (NEU[1] - y) * 0.45;
        return (
          <g key={i}>
            <line x1={x} y1={y} x2={NEU[0]} y2={NEU[1]} stroke="var(--muted)" strokeOpacity={w === 0 ? 0.2 : 0.55} strokeWidth={Math.min(1 + Math.abs(w) * 1.2, 5)} strokeDasharray={w < 0 ? "5 4" : undefined} />
            {summing && <motion.line key={animKey} x1={x} y1={y} x2={NEU[0]} y2={NEU[1]} stroke="var(--accent)" strokeWidth={3} strokeLinecap="round"
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 0.8 }} transition={{ delay: pulse(i) * k, duration: TRAVEL * k, ease: EASE }} />}
            <rect x={lx - 22} y={ly - 9} width={44} height={18} rx={4} fill="var(--surface)" stroke="var(--line)" />
            <text x={lx} y={ly + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">w={w.toFixed(1)}</text>
            <motion.g key={`in-${animKey}`} initial={phase === "inputs" ? { opacity: 0, x: -30 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.35) * k, duration: 0.5 * k, ease: EASE }}>
              <circle cx={x} cy={y} r={22} fill="var(--accent-soft)" stroke="var(--accent)" strokeWidth={1.5} />
              <text x={x} y={y + 4} textAnchor="middle" fontSize={12} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">{input.x[i].toFixed(1)}</text>
              <text x={x} y={y - 27} textAnchor="middle" fontSize={10} fill="var(--muted)">x{sub(i)}</text>
            </motion.g>
          </g>
        );
      })}
      {/* bias enters from above */}
      <line x1={NEU[0]} y1={30} x2={NEU[0]} y2={NEU[1] - 46} stroke="var(--violet)" strokeOpacity={0.5} strokeDasharray="3 3" />
      <rect x={NEU[0] - 30} y={10} width={60} height={18} rx={9} fill="var(--violet)" />
      <text x={NEU[0]} y={23} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--surface)">b={input.b.toFixed(1)}</text>
      <line x1={NEU[0] + 46} y1={NEU[1]} x2={OUT[0] - 22} y2={OUT[1]} stroke="var(--muted)" strokeOpacity={0.55} strokeWidth={1.5} />

      {/* neuron body: Σ half and f half */}
      {(summing || activating) && <motion.circle cx={NEU[0]} cy={NEU[1]} fill="none" stroke="var(--accent)" strokeWidth={2} initial={{ r: 46, opacity: 0.6 }} animate={{ r: 60, opacity: 0 }} transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }} />}
      <circle cx={NEU[0]} cy={NEU[1]} r={46} fill="var(--surface)" stroke={summing || activating ? "var(--accent)" : "var(--line)"} strokeWidth={summing || activating ? 3 : 1.5} />
      <motion.rect x={NEU[0] - 44} y={NEU[1] - 44} width={44} height={88} fill="var(--accent)" initial={false} animate={{ fillOpacity: summing ? 0.14 : 0 }} transition={{ duration: 0.3 }} />
      <motion.rect x={NEU[0]} y={NEU[1] - 44} width={44} height={88} fill="var(--violet)" initial={false} animate={{ fillOpacity: activating ? 0.14 : 0 }} transition={{ duration: 0.3 }} />
      <circle cx={NEU[0]} cy={NEU[1]} r={46} fill="none" stroke={summing || activating ? "var(--accent)" : "var(--line)"} strokeWidth={summing || activating ? 3 : 1.5} />
      <line x1={NEU[0]} y1={NEU[1] - 46} x2={NEU[0]} y2={NEU[1] + 46} stroke="var(--line)" />
      <text x={NEU[0] - 22} y={NEU[1] - 14} textAnchor="middle" fontSize={15} fill="var(--muted)">Σ</text>
      <text x={NEU[0] + 22} y={NEU[1] - 14} textAnchor="middle" fontSize={14} fontStyle="italic" fill="var(--muted)">f</text>
      <text x={NEU[0] - 22} y={NEU[1] + 8} textAnchor="middle" fontSize={11} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">
        {pi < 1 ? "·" : summing ? <AnimatedNumber key={animKey} svg value={r.z} from={0} delay={SUM * k} duration={0.6 * k} /> : r.z.toFixed(2)}
      </text>
      <text x={NEU[0] + 22} y={NEU[1] + 8} textAnchor="middle" fontSize={11} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">
        {pi < 2 ? "·" : activating ? <AnimatedNumber key={animKey} svg value={r.a} from={r.z} delay={ACT.across * k} duration={0.5 * k} /> : r.a.toFixed(2)}
      </text>
      <text x={NEU[0] - 22} y={NEU[1] + 24} textAnchor="middle" fontSize={9} fill="var(--muted)">z</text>
      <text x={NEU[0] + 22} y={NEU[1] + 24} textAnchor="middle" fontSize={9} fill="var(--muted)">a</text>
      <text x={NEU[0]} y={NEU[1] + 64} textAnchor="middle" fontSize={10} fill="var(--muted)">{ACTS[input.act].label}</text>

      {/* output node */}
      <circle cx={OUT[0]} cy={OUT[1]} r={22} fill={pi >= 3 ? "var(--ok-soft)" : "var(--surface)"} stroke={pi >= 3 ? "var(--ok)" : "var(--line)"} strokeWidth={1.5} />
      <text x={OUT[0]} y={OUT[1] + 4} textAnchor="middle" fontSize={11} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">{pi >= 3 ? r.a.toFixed(2) : "·"}</text>
      <text x={OUT[0]} y={OUT[1] - 27} textAnchor="middle" fontSize={10} fill="var(--muted)">output</text>

      {/* travelling products */}
      {summing && IN_POS.map(([x, y], i) => {
        const label = `${input.x[i].toFixed(1)}×${input.w[i].toFixed(1)} = ${r.products[i].toFixed(2)}`;
        const lw = label.length * 6.4 + 10;
        return (
          <motion.g key={`${animKey}-p${i}`} initial={{ x, y, opacity: 0 }} animate={{ x: [x, NEU[0] - 30], y: [y, NEU[1]], opacity: [0, 1, 1, 0] }}
            transition={{ delay: pulse(i) * k, duration: TRAVEL * k, ease: "easeInOut", opacity: { delay: pulse(i) * k, duration: TRAVEL * k, times: [0, 0.12, 0.85, 1] } }}>
            <circle r={7} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
            <rect x={-12} y={-28} width={lw} height={17} rx={8} fill="var(--fg)" />
            <text x={lw / 2 - 12} y={-16} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--surface)">{label}</text>
          </motion.g>
        );
      })}
      {summing && (
        <motion.g key={`${animKey}-b`} initial={{ opacity: 0, x: NEU[0], y: 20 }} animate={{ opacity: [0, 1, 1, 0], y: [20, 40, 50, NEU[1] - 20] }} transition={{ delay: BIAS * k, duration: 0.6 * k, times: [0, 0.2, 0.5, 1] }}>
          <circle r={7} fill="var(--violet)" stroke="var(--surface)" strokeWidth={2} />
        </motion.g>
      )}
      {phase === "output" && (
        <motion.g key={`${animKey}-o`} initial={{ x: NEU[0] + 46, y: NEU[1], opacity: 0 }} animate={{ x: [NEU[0] + 46, OUT[0]], opacity: [0, 1, 1, 0] }} transition={{ delay: 0.3 * k, duration: 1 * k, ease: "easeInOut" }}>
          <circle r={7} fill="var(--ok)" stroke="var(--surface)" strokeWidth={2} />
          <rect x={-30} y={-28} width={60} height={17} rx={8} fill="var(--fg)" />
          <text y={-16} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--surface)">a={r.a.toFixed(2)}</text>
        </motion.g>
      )}
    </svg>
  );
}

/* ---------- activation (kind f) and derivative (kind d) plots ---------- */
function Plot({ kind, acts, main, z, d, phase, pi, k, animKey }: {
  kind: "f" | "d"; acts: ActName[]; main: ActName; z: number; d: number; phase: Phase; pi: number; k: number; animKey: string;
}) {
  const clipId = useId();
  const W = 320, H = kind === "f" ? 170 : 120, L = 30, R = 10, T = 14, B = 20;
  const zr = Math.min(10, Math.max(5, Math.ceil(Math.abs(z)) + 1));
  const zc = clamp(z, -zr, zr);
  const fn = (a: ActName) => (kind === "f" ? ACTS[a].f : ACTS[a].d);
  const xs = Array.from({ length: 121 }, (_, i) => -zr + (2 * zr * i) / 120);
  const allY = acts.flatMap((a) => xs.map(fn(a)));
  let y0 = Math.min(0, ...allY), y1 = Math.max(kind === "d" ? 1 : 0.5, ...allY);
  const pad = (y1 - y0) * 0.08;
  y0 -= pad; y1 += pad;
  const sx = (v: number) => L + ((v + zr) / (2 * zr)) * (W - L - R);
  const sy = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const path = (a: ActName) => xs.map((v, i) => `${i ? "L" : "M"}${sx(v).toFixed(1)} ${sy(fn(a)(v)).toFixed(1)}`).join(" ");
  const compare = phase === "compare";
  const showDot = kind === "f" ? pi >= 2 : pi >= 4;
  const focus = kind === "f" ? phase === "activate" : phase === "derivative";
  const t = (s: number, dur = 0.5) => ({ delay: s * k, duration: dur * k, ease: EASE });
  const regions = kind === "d" && !compare ? flatRegions(main, -zr, zr) : [];
  const yv = fn(main)(zc);
  const label = kind === "f" ? "output a = f(z)" : "slope f'(z)";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${label} for ${acts.map((a) => ACTS[a].label).join(", ")} at z ${z.toFixed(2)}`}>
      <defs><clipPath id={clipId}><rect x={L} y={T} width={W - L - R} height={H - T - B} /></clipPath></defs>
      {regions.map(([a, b], i) => (
        <motion.rect key={`${animKey}-r${i}`} x={sx(a)} y={T} width={Math.max(1, sx(b) - sx(a))} height={H - T - B} fill="var(--err)" initial={{ opacity: phase === "derivative" ? 0 : 0.08 }} animate={{ opacity: phase === "derivative" ? 0.14 : 0.08 }} transition={t(DER.shade)} />
      ))}
      {regions.length > 0 && <text x={sx(regions[0][0]) + 3} y={H - B - 4} fontSize={9} fill="var(--err)">flat: |f'| &lt; {SATURATION}</text>}
      <line x1={L} x2={W - R} y1={sy(0)} y2={sy(0)} stroke="var(--line)" />
      <line x1={sx(0)} x2={sx(0)} y1={T} y2={H - B} stroke="var(--line)" />
      {[-zr, 0, zr].map((v) => <text key={v} x={sx(v)} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>)}
      {[y0 + pad, y1 - pad].map((v) => <text key={v} x={L - 4} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{v.toFixed(v % 1 ? 1 : 0)}</text>)}
      <text x={L + 4} y={T - 3} fontSize={10} fill="var(--muted)">{label}</text>
      <g clipPath={`url(#${clipId})`}>
        {acts.map((a, i) => {
          const strong = a === main || compare;
          const draw = compare || focus;
          return (
            <motion.path key={`${animKey}-${a}-${acts.length}`} d={path(a)} fill="none" stroke={ACTS[a].color} strokeWidth={strong ? 2.4 : 1.2} strokeOpacity={strong ? 1 : 0.35}
              initial={draw ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={t(compare ? i * CMP.stagger : kind === "f" ? ACT.curve : DER.curve, 0.6)} />
          );
        })}
        {kind === "f" && phase === "derivative" && (
          <motion.line key={`${animKey}-tan`} x1={sx(zc - 2)} x2={sx(zc + 2)} y1={sy(yv - 2 * d)} y2={sy(yv + 2 * d)} stroke="var(--fg)" strokeWidth={1.5} strokeDasharray="5 3"
            initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 0.8 }} transition={t(DER.tangent, 0.5)} />
        )}
      </g>
      {showDot && !compare && (
        <g key={`${animKey}-dot`}>
          <motion.line x1={sx(zc)} x2={sx(zc)} y1={sy(0)} y2={sy(yv)} stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="3 3"
            initial={focus ? { pathLength: 0, opacity: 0 } : false} animate={{ pathLength: 1, opacity: 1 }} transition={t(kind === "f" ? ACT.rise : DER.dot, 0.6)} />
          <motion.line x1={sx(zc)} x2={L} y1={sy(yv)} y2={sy(yv)} stroke="var(--ok)" strokeWidth={1.5} strokeDasharray="3 3"
            initial={focus ? { pathLength: 0, opacity: 0 } : false} animate={{ pathLength: 1, opacity: 1 }} transition={t(kind === "f" ? ACT.across : DER.dot + 0.4, 0.5)} />
          <motion.circle r={5.5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} cx={sx(zc)}
            initial={focus ? { cy: sy(0), opacity: 0 } : false} animate={{ cy: sy(yv), opacity: 1 }} transition={{ delay: (kind === "f" ? ACT.zMark : DER.dot) * k, duration: 0.8 * k, ease: EASE }} />
          <motion.g initial={focus ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={t(kind === "f" ? ACT.across + 0.3 : DER.dot + 0.6, 0.3)}>
            <rect x={L + 3} y={sy(yv) - 17} width={86} height={14} rx={3} fill="var(--ok)" />
            <text x={L + 46} y={sy(yv) - 7} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--surface)">{kind === "f" ? "a" : "f'"} = {(kind === "f" ? ACTS[main].f(z) : d).toFixed(3)}</text>
          </motion.g>
          <rect x={clamp(sx(zc) - 30, L, W - R - 60)} y={H - B + 1} width={60} height={13} rx={3} fill="var(--accent)" />
          <text x={clamp(sx(zc) - 30, L, W - R - 60) + 30} y={H - B + 11} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--surface)">z={z.toFixed(2)}{z !== zc ? "→" : ""}</text>
        </g>
      )}
      {compare && acts.map((a, i) => (
        <motion.circle key={`${animKey}-c${a}`} cx={sx(zc)} r={4.5} fill={ACTS[a].color} stroke="var(--surface)" strokeWidth={1.2}
          initial={{ cy: sy(0), opacity: 0 }} animate={{ cy: sy(fn(a)(zc)), opacity: 1 }} transition={t(CMP.dots + i * 0.12, 0.6)} />
      ))}
      {compare && kind === "f" && (
        <g>
          {acts.map((a, i) => (
            <g key={a} transform={`translate(${L + 8}, ${T + 6 + i * 12})`}>
              <line x1={0} x2={12} y1={0} y2={0} stroke={ACTS[a].color} strokeWidth={2.5} />
              <text x={16} y={3} fontSize={9} fill="var(--fg)">{ACTS[a].label.split(" (")[0]}</text>
            </g>
          ))}
        </g>
      )}
    </svg>
  );
}

/* ---------- explanatory card ---------- */
function StepCard({ phase, input, r, k }: { phase: Phase; input: NeuronInput; r: NeuronResult; k: number }) {
  const L = ACTS[input.act].label;
  if (phase === "inputs") {
    return (
      <div className="flex flex-wrap gap-2">
        {input.x.map((x, i) => (
          <motion.div key={i} className="rounded-lg border border-accent/40 bg-accent-soft px-3 py-1.5 font-mono text-sm" initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.35) * k, duration: 0.5 * k, ease: EASE }}>
            x{sub(i)} = {x.toFixed(1)} <span className="text-muted">· weight w{sub(i)} = {input.w[i].toFixed(1)}</span>
          </motion.div>
        ))}
        <p className="w-full text-xs text-muted">A weight says how much an input matters and in which direction (negative weights push the sum down). The bias b = {input.b.toFixed(1)} is a fixed nudge added at the end.</p>
      </div>
    );
  }
  if (phase === "sum") {
    const rows = [
      ...r.products.map((v, i) => ({ name: `x${sub(i)}·w${sub(i)}`, v, at: arrive(i), kind: "term" })),
      { name: "+ bias", v: input.b, at: BIAS + 0.6, kind: "bias" },
      { name: "z (sum)", v: r.z, at: SUM, kind: "total" },
    ];
    const scale = Math.max(1e-9, ...rows.map((x) => Math.abs(x.v)));
    return (
      <div className="space-y-1.5">
        {rows.map((row, i) => (
          <div key={i} className={cx("grid grid-cols-[72px_minmax(0,1fr)_56px] items-center gap-2", row.kind === "total" && "border-t border-line pt-1.5")}>
            <span className={cx("font-mono text-xs", row.kind === "total" && "font-semibold")}>{row.name}</span>
            <div className="relative h-5 rounded bg-surface-2">
              <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
              <motion.div className={cx("absolute inset-y-0.5 rounded", row.kind === "total" ? "bg-accent" : row.kind === "bias" ? "bg-violet/70" : row.v < 0 ? "bg-err/70" : "bg-ok/70")}
                style={row.v < 0 ? { right: "50%" } : { left: "50%" }} initial={{ width: 0 }} animate={{ width: `${(Math.abs(row.v) / scale) * 50}%` }} transition={{ delay: row.at * k, duration: 0.5 * k, ease: EASE }} />
            </div>
            <span className={cx("text-right font-mono text-xs", row.kind === "total" && "font-semibold")}><AnimatedNumber value={row.v} from={0} delay={row.at * k} duration={0.5 * k} /></span>
          </div>
        ))}
        <p className="pt-1 text-xs text-muted">Green pushes z up, red pushes it down. Nothing non-linear has happened yet: z is just a weighted sum.</p>
      </div>
    );
  }
  if (phase === "activate" || phase === "output") {
    const scale = Math.max(1e-9, Math.abs(r.z), Math.abs(r.a));
    const rows = [{ name: "before: z", v: r.z, c: "bg-accent" }, { name: "after: a", v: r.a, c: "bg-ok" }];
    return (
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <p className="font-mono text-lg">{L}(<span>{r.z.toFixed(2)}</span>) = <strong><AnimatedNumber value={r.a} from={phase === "activate" ? r.z : r.a} delay={ACT.across * k} duration={0.5 * k} digits={3} /></strong></p>
          <p className="text-xs text-muted">{ACT_TEXT[input.act]}</p>
          {r.zone === "dead" && <p className="rounded-md bg-err-soft px-2 py-1 text-xs"><XCircle size={12} className="mr-1 inline" aria-hidden />z is not positive, so ReLU outputs 0. The neuron is silent for this input.</p>}
        </div>
        <div className="space-y-1.5">
          {rows.map((row, i) => (
            <div key={row.name} className="grid grid-cols-[72px_minmax(0,1fr)_52px] items-center gap-2 whitespace-nowrap">
              <span className="font-mono text-xs">{row.name}</span>
              <div className="relative h-5 rounded bg-surface-2">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <motion.div className={cx("absolute inset-y-0.5 rounded", row.c)} style={row.v < 0 ? { right: "50%" } : { left: "50%" }} initial={{ width: 0 }} animate={{ width: `${(Math.abs(row.v) / scale) * 50}%` }} transition={{ delay: (i ? ACT.across : 0.2) * k, duration: 0.5 * k, ease: EASE }} />
              </div>
              <span className="text-right font-mono text-xs">{row.v.toFixed(3)}</span>
            </div>
          ))}
          <p className="text-[11px] text-muted">{phase === "output" ? "Only a travels on. If a is 0, every weight after this neuron is multiplied by 0." : "The activation reshapes z. This bend is what lets networks learn curved rules."}</p>
        </div>
      </div>
    );
  }
  if (phase === "derivative") {
    const chain = chainedGradient(r.d, 5);
    return (
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-1.5 text-sm">
          <p className="font-mono">{L}′({r.z.toFixed(2)}) = <strong><AnimatedNumber value={r.d} from={0} delay={DER.dot * k} digits={4} /></strong></p>
          <ZoneBadge zone={r.zone} />
          <p className="text-xs text-muted">
            {r.zone === "dead" ? "Slope exactly 0: no gradient reaches w or b, so gradient descent never changes them for this input. Try leaky ReLU or a smaller negative bias."
              : r.zone === "vanishing" ? "The curve is almost flat at this z, so the weights barely move. Large |z| saturates sigmoid and tanh."
              : "A healthy slope: the error signal passes back with useful size."}
          </p>
        </div>
        <div>
          <p className="mb-1 text-xs font-semibold">If 5 stacked layers all sat at this z, the gradient shrinks like f′(z)ⁿ</p>
          <div className="space-y-1">
            {chain.map((v, i) => (
              <div key={i} className="grid grid-cols-[60px_minmax(0,1fr)_72px] items-center gap-2 whitespace-nowrap">
                <span className="font-mono text-[11px] text-muted">layer −{i + 1}</span>
                <div className="relative h-3.5 rounded bg-surface-2">
                  <motion.div className={cx("absolute inset-y-0 left-0 rounded", Math.abs(v) < SATURATION ? "bg-err/70" : "bg-ok/70")} initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, Math.abs(v) * 100)}%` }} transition={{ delay: (DER.chain + i * 0.25) * k, duration: 0.4 * k, ease: EASE }} />
                </div>
                <span className="text-right font-mono text-[11px]">{Math.abs(v) < 1e-4 && v !== 0 ? v.toExponential(1) : v.toFixed(4)}</span>
              </div>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-muted">Bar = fraction of the signal left (weights ignored). Sigmoid's slope is at most 0.25, so 0.25⁵ ≈ 0.001.</p>
        </div>
      </div>
    );
  }
  const all = compareAll(r.z);
  const fScale = Math.max(1e-9, ...all.map((c) => Math.abs(c.a)));
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[340px] text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">activation</th><th>output f(z)</th><th>slope f′(z)</th><th>gradient</th></tr></thead>
        <tbody>
          {all.map((c, i) => (
            <tr key={c.act} className={cx("border-t border-line", c.act === input.act && "bg-accent-soft/60")}>
              <td className="py-1.5 pr-2"><span className="mr-1 inline-block h-2 w-3 rounded-sm" style={{ background: ACTS[c.act].color }} />{ACTS[c.act].label.split(" (")[0]}</td>
              <td className="pr-2">
                <div className="flex items-center gap-1.5">
                  <div className="relative h-3 w-16 rounded bg-surface-2"><div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                    <motion.div className="absolute inset-y-0 rounded bg-accent/70" style={c.a < 0 ? { right: "50%" } : { left: "50%" }} initial={{ width: 0 }} animate={{ width: `${(Math.abs(c.a) / fScale) * 50}%` }} transition={{ delay: (CMP.dots + i * 0.12) * k, duration: 0.4 * k }} />
                  </div>
                  <span className="font-mono">{c.a.toFixed(3)}</span>
                </div>
              </td>
              <td className="pr-2">
                <div className="flex items-center gap-1.5">
                  <div className="relative h-3 w-12 rounded bg-surface-2">
                    <motion.div className="absolute inset-y-0 left-0 rounded bg-ok/70" initial={{ width: 0 }} animate={{ width: `${Math.min(100, Math.abs(c.d) * 100)}%` }} transition={{ delay: (CMP.dots + i * 0.12) * k, duration: 0.4 * k }} />
                  </div>
                  <span className="font-mono">{c.d.toFixed(3)}</span>
                </div>
              </td>
              <td><ZoneBadge zone={c.zone} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Calculation({ phase, input, r }: { phase: Phase; input: NeuronInput; r: NeuronResult }) {
  const A = ACTS[input.act];
  const { x, w, b } = input;
  if (phase === "inputs") return <TexBlock>{`\\mathbf{x} = [${x.map((v) => v.toFixed(1)).join(",\\ ")}] \\quad \\mathbf{w} = [${w.map((v) => v.toFixed(1)).join(",\\ ")}] \\quad b = ${b.toFixed(1)}`}</TexBlock>;
  if (phase === "sum")
    return <TexBlock>{`\\begin{aligned} z &= \\textstyle\\sum_i w_i x_i + b \\\\ &= ${x.map((v, i) => `${n2(w[i])} \\cdot ${n2(v)}`).join(" + ")} + ${n2(b)} \\\\ &= ${r.z.toFixed(4)} \\end{aligned}`}</TexBlock>;
  if (phase === "activate" || phase === "output")
    return <TexBlock>{`\\begin{aligned} &${A.tex} \\\\ &a = ${A.name}(${r.z.toFixed(3)}) = ${r.a.toFixed(4)} \\end{aligned}`}</TexBlock>;
  if (phase === "derivative")
    return (
      <div className="space-y-2">
        <TexBlock>{`\\begin{aligned} &${A.dtex} \\\\ &${A.name}'(${r.z.toFixed(3)}) = ${r.d.toFixed(4)} \\end{aligned}`}</TexBlock>
        <TexBlock>{`\\frac{\\partial L}{\\partial w_i} = \\frac{\\partial L}{\\partial a} \\cdot \\underbrace{${r.d.toFixed(3)}}_{f'(z)} \\cdot x_i`}</TexBlock>
      </div>
    );
  return <TexBlock>{`\\begin{aligned} ${compareAll(r.z).map((c) => `${ACTS[c.act].name}(${r.z.toFixed(2)}) &= ${c.a.toFixed(3)} & ${ACTS[c.act].name}' &= ${c.d.toFixed(3)}`).join(" \\\\ ")} \\end{aligned}`}</TexBlock>;
}
