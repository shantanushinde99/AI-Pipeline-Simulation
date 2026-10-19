import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw, RotateCw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Badge, Button, Callout, KV, NumberField, Panel, Select, Slider, cx } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { StepStage } from "./StepStage";
import { ACT, TRAVEL, biasAt, narrate, pulseStart, stepDuration, sumAt } from "./timeline";
import { fmt } from "../../lib/math";
import type { VisualizerTab } from "../types";
import {
  ACTIVATIONS, DATASETS, edgeId, evaluateDataset, forward, neuronIds, randomWeights, simulate,
  type Activation, type NetworkInput, type NNState, type Weights,
} from "./simulation";

type Param = { kind: "w1"; j: number; i: number } | { kind: "b1"; j: number } | { kind: "w2"; j: number } | { kind: "b2" };

const getParam = (w: Weights, p: Param) =>
  p.kind === "w1" ? w.w1[p.j][p.i] : p.kind === "b1" ? w.b1[p.j] : p.kind === "w2" ? w.w2[p.j] : w.b2;

function setParam(w: Weights, p: Param, v: number): Weights {
  const n = structuredClone(w);
  if (p.kind === "w1") n.w1[p.j][p.i] = v;
  else if (p.kind === "b1") n.b1[p.j] = v;
  else if (p.kind === "w2") n.w2[p.j] = v;
  else n.b2 = v;
  return n;
}

const paramLabel = (p: Param) =>
  p.kind === "w1" ? `weight x${p.i + 1} → h${p.j + 1}` : p.kind === "b1" ? `bias of h${p.j + 1}` : p.kind === "w2" ? `weight h${p.j + 1} → y` : "bias of y";
const paramKey = (p: Param) => JSON.stringify(p);
const paramTex = (p: Param) =>
  p.kind === "w1" ? `w_{x_{${p.i + 1}} \\to h_{${p.j + 1}}}` : p.kind === "b1" ? `b_{h_{${p.j + 1}}}` : p.kind === "w2" ? `w_{h_{${p.j + 1}} \\to y}` : "b_{y}";
/** Wrap negatives in parentheses so substituted values read correctly. */
const n = (v: number, d?: number) => { const s = d === undefined ? String(v) : v.toFixed(d); return v < 0 ? `(${s})` : s; };

const POS: Record<string, [number, number]> = { x1: [90, 130], x2: [90, 250], h1: [330, 70], h2: [330, 190], h3: [330, 310], y: [570, 190] };

type EdgeDef = { from: string; to: string; param: Param; t: number };
const EDGES: EdgeDef[] = [
  ...neuronIds.hidden.flatMap((h, j) => neuronIds.input.map((x, i) => ({ from: x, to: h, param: { kind: "w1", j, i } as Param, t: 0.72 }))),
  ...neuronIds.hidden.map((h, j) => ({ from: h, to: "y", param: { kind: "w2", j } as Param, t: 0.5 })),
];

export default function NeuralNetworkVisualizer({ tab }: { tab: VisualizerTab }) {
  const [datasetId, setDatasetId] = useState(DATASETS[0].id);
  const ds = DATASETS.find((d) => d.id === datasetId)!;
  const [x, setX] = useState<[number, number]>(ds.samples[2].x);
  const [weights, setWeights] = useState<Weights>(ds.preset);
  const [activation, setActivation] = useState<Activation>(ds.presetActivation);
  const [selected, setSelected] = useState<Param>({ kind: "w2", j: 0 });
  const [replay, setReplay] = useState(0);

  const input: NetworkInput = { x, weights, activation };
  const steps = useMemo(() => simulate({ x, weights, activation }), [x, weights, activation]);
  const pb = usePlayback(steps.length, stepDuration);
  const step = steps[pb.index];
  const evaluation = useMemo(() => evaluateDataset(ds, weights, activation), [ds, weights, activation]);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;

  const changeDataset = (id: string) => {
    const d = DATASETS.find((v) => v.id === id)!;
    setDatasetId(id);
    setX(d.samples[0].x);
    setWeights(d.preset);
    setActivation(d.presetActivation);
  };
  const reset = () => {
    setWeights(ds.preset);
    setActivation(ds.presetActivation);
    pb.restart();
  };

  if (tab === "metrics") return <DatasetMetrics evaluation={evaluation} features={ds.features} />;

  const out = forward(input);
  const value = getParam(weights, selected);
  const nudged = forward({ ...input, weights: setParam(weights, selected, value + 0.5) }).y;
  const h = 1e-4;
  const grad = (forward({ ...input, weights: setParam(weights, selected, value + h) }).y - forward({ ...input, weights: setParam(weights, selected, value - h) }).y) / (2 * h);
  const story = narrate(step.state, activation, out.y);

  return (
    <Workspace
      main={
        <>
          <Panel title="Network" actions={<span className="text-[11px] text-muted">Click a weight or neuron to edit it</span>}>
            <div className="mb-2 flex items-start gap-3">
              <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{pb.index + 1}/{steps.length}</span>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={step.id} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
                  <p className="text-base font-semibold leading-snug" data-testid="nn-headline">{story.headline}</p>
                  <p className="text-sm text-muted">{story.sub}</p>
                </motion.div>
              </AnimatePresence>
              <Button size="sm" variant="ghost" onClick={() => setReplay((r) => r + 1)} title="Play this step's animation again"><RotateCw size={13} /> Replay step</Button>
            </div>
            <NetworkSvg weights={weights} state={step.state} selected={selected} onSelect={setSelected} prediction={pb.atEnd ? out.y : undefined} k={k} animKey={animKey} />
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="currentColor" strokeWidth="2" /></svg>positive weight</span>
              <span className="flex items-center gap-1"><svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="currentColor" strokeWidth="2" strokeDasharray="4 3" /></svg>negative weight</span>
              <span>line width ∝ |weight|</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>a value travelling along a line</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepStage state={step.state} input={input} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          {pb.index === 0 && !pb.playing && <p className="-mt-1 text-xs text-muted">Press <strong>Play</strong> to watch the whole forward pass, or <strong>→</strong> to go one step at a time. Use <strong>Speed</strong> to slow it down.</p>}
          <Panel title="Inputs & parameters" actions={<Button size="sm" variant="ghost" onClick={reset}><RotateCcw size={12} /> Reset parameters</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Select label="Example dataset" value={datasetId} options={DATASETS.map((d) => ({ value: d.id, label: d.name }))} onChange={changeDataset} />
              <NumberField label={`x1 (${ds.features[0]})`} value={x[0]} onChange={(v) => setX([v, x[1]])} min={-10} max={10} />
              <NumberField label={`x2 (${ds.features[1]})`} value={x[1]} onChange={(v) => setX([x[0], v])} min={-10} max={10} />
              <Select
                label="Hidden activation"
                value={activation}
                options={(Object.keys(ACTIVATIONS) as Activation[]).map((a) => ({ value: a, label: a }))}
                onChange={setActivation}
              />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Samples:</span>
              {ds.samples.map((s) => (
                <Button key={s.label} size="sm" variant={s.x[0] === x[0] && s.x[1] === x[1] ? "primary" : "secondary"} onClick={() => setX(s.x)}>
                  {s.label} <span className="opacity-70">→ {s.target}</span>
                </Button>
              ))}
              <span className="ml-2 text-xs text-muted">Weights:</span>
              <Button size="sm" onClick={() => setWeights(ds.preset)}>Hand-tuned</Button>
              <Button size="sm" onClick={() => setWeights(randomWeights(7))}>Random (seed 7)</Button>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation">
            <Calculation input={input} active={step.state.active} phase={step.state.phase} />
          </Panel>
          <Panel title={`Selected: ${paramLabel(selected)}`}>
            <Slider key={paramKey(selected)} label="Value" value={value} min={-8} max={8} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => setWeights(setParam(weights, selected, Math.round(v * 10) / 10))} />
            <div className="mt-3 space-y-2 text-xs leading-relaxed">
              <KV rows={[
                [<>current <Tex>{"\\hat{y}"}</Tex></>, <span data-testid="nn-output">{fmt(out.y, 4)}</span>],
                [<Tex>{`\\hat{y}\\,\\text{ with } ${paramTex(selected)} + 0.5`}</Tex>, fmt(nudged, 4)],
                [<><Tex>{`\\dfrac{\\partial \\hat{y}}{\\partial ${paramTex(selected)}}`}</Tex> <span className="text-[10px]">(finite diff.)</span></>, fmt(grad, 4)],
              ]} />
              <p className="text-muted">
                {Math.abs(grad) < 1e-6
                  ? "This parameter has no effect on the output for this input: a downstream neuron is inactive (e.g. ReLU output 0) or a connecting weight is 0."
                  : `Increasing it ${grad > 0 ? "raises" : "lowers"} the output for this input. Training (backpropagation) uses exactly this kind of sensitivity to decide which way to move each weight.`}
              </p>
            </div>
          </Panel>
          <Panel title="Events">
            <EventLog events={step.events} />
          </Panel>
        </>
      }
    />
  );
}

const LAYER_X = { input: 90, hidden: 330, output: 570 };
const ease = [0.22, 1, 0.36, 1] as const;

function NetworkSvg({ weights, state, selected, onSelect, prediction, k, animKey }: {
  weights: Weights; state: NNState; selected: Param; onSelect: (p: Param) => void; prediction?: number; k: number; animKey: string;
}) {
  const sel = paramKey(selected);
  const layer = state.active === null ? (state.phase === "input" ? "input" : "output") : state.active === "y" ? "output" : "hidden";
  // Incoming values for the neuron currently summing: they travel along its lines in order.
  const sources = state.phase === "sum" && state.active ? (state.active === "y" ? [...neuronIds.hidden] : [...neuronIds.input]) : [];
  const target = state.active ? POS[state.active] : null;
  const edgeOf = (from: string) => EDGES.find((e) => e.from === from && e.to === state.active)!;
  return (
    <svg viewBox="0 0 680 385" className="h-auto w-full" role="img" aria-label="Neural network diagram with 2 inputs, 3 hidden neurons and 1 output">
      <motion.rect y={18} width={140} height={342} rx={16} fill="var(--accent)" fillOpacity={0.07} stroke="var(--accent)" strokeOpacity={0.25}
        initial={false} animate={{ x: LAYER_X[layer] - 70 }} transition={{ type: "spring", stiffness: 120, damping: 20 }} />
      {EDGES.map((e) => {
        const [x1, y1] = POS[e.from];
        const [x2, y2] = POS[e.to];
        const w = getParam(weights, e.param);
        const id = edgeId(e.from, e.to);
        const active = state.activeEdges.includes(id);
        const isSel = paramKey(e.param) === sel;
        const lx = x1 + (x2 - x1) * e.t;
        const ly = y1 + (y2 - y1) * e.t;
        return (
          <g key={id}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={isSel ? "var(--accent)" : "var(--muted)"} strokeOpacity={w === 0 ? 0.2 : isSel ? 1 : active ? 0.35 : 0.5}
              strokeWidth={Math.min(1 + Math.abs(w) * 0.9, 7)} strokeDasharray={w < 0 ? "6 4" : undefined} style={{ transition: "stroke-opacity .4s" }} />
            {active && (
              <motion.line key={animKey} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--accent)" strokeWidth={Math.min(2 + Math.abs(w) * 0.9, 8)} strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 0.85 }} transition={{ duration: 0.5 * k, ease }} />
            )}
            <g role="button" tabIndex={0} aria-label={`Select ${paramLabel(e.param)} (${w})`} className="cursor-pointer" onClick={() => onSelect(e.param)}
              onKeyDown={(ev) => (ev.key === "Enter" || ev.key === " ") && onSelect(e.param)}>
              <rect x={lx - 19} y={ly - 9} width={38} height={18} rx={4} fill={isSel ? "var(--accent)" : active ? "var(--accent-soft)" : "var(--surface)"} stroke={isSel || active ? "var(--accent)" : "var(--line)"} />
              <text x={lx} y={ly + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill={isSel ? "var(--surface)" : "var(--fg)"}>{w.toFixed(1)}</text>
            </g>
          </g>
        );
      })}

      {Object.entries(POS).map(([id, [cx0, cy]]) => {
        const v = state.values[id];
        const isActive = state.active === id;
        const isInput = id.startsWith("x");
        const biasParam: Param | null = id === "y" ? { kind: "b2" } : id.startsWith("h") ? { kind: "b1", j: Number(id[1]) - 1 } : null;
        const bias = biasParam ? getParam(weights, biasParam) : null;
        const isSel = biasParam && paramKey(biasParam) === sel;
        const computed = v?.a !== undefined;
        const summing = isActive && state.phase === "sum";
        const activating = isActive && state.phase === "activate";
        const nIn = id === "y" ? 3 : 2;
        return (
          <g key={id} className={cx(biasParam && "cursor-pointer")} onClick={() => biasParam && onSelect(biasParam)}
            role={biasParam ? "button" : undefined} tabIndex={biasParam ? 0 : undefined} aria-label={biasParam ? `Select ${paramLabel(biasParam)} (${bias})` : `Input ${id} = ${v?.a}`}
            onKeyDown={(ev) => biasParam && (ev.key === "Enter" || ev.key === " ") && onSelect(biasParam)}>
            {isActive && (
              <motion.circle cx={cx0} cy={cy} fill="none" stroke="var(--accent)" strokeWidth={2}
                initial={{ r: 30, opacity: 0.6 }} animate={{ r: 44, opacity: 0 }} transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }} />
            )}
            <circle cx={cx0} cy={cy} r={28} fill="var(--surface)" />
            <motion.circle cx={cx0} cy={cy} r={28} fill="var(--accent)" initial={false}
              animate={{ fillOpacity: computed ? 0.12 + 0.45 * Math.min(1, Math.abs(v.a!)) : 0 }} transition={{ duration: 0.6 * k, delay: activating ? ACT.morph * k : 0 }} />
            <motion.circle cx={cx0} cy={cy} r={28} fill="none" initial={false}
              animate={{ stroke: isActive || isSel ? "var(--accent)" : "var(--line)", strokeWidth: isActive ? 4 : isSel ? 3 : 1.5 }} transition={{ duration: 0.3 }} />
            <text x={cx0} y={cy - 2} textAnchor="middle" fontSize={13} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">
              {summing ? <AnimatedNumber key={animKey} svg value={v!.z!} from={0} delay={sumAt(nIn) * k} duration={0.6 * k} />
                : activating ? <AnimatedNumber key={animKey} svg value={v!.a!} from={v!.z!} delay={ACT.morph * k} duration={0.5 * k} />
                : computed ? <AnimatedNumber svg value={v.a!} /> : "·"}
            </text>
            <text x={cx0} y={cy + 13} textAnchor="middle" fontSize={10} fill="var(--muted)">{summing ? "sum z" : id}</text>
            {!isInput && v?.z !== undefined && !summing && (
              <text x={cx0} y={cy - 36} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">z={v.z.toFixed(2)}</text>
            )}
            {bias !== null && (
              <text x={cx0} y={cy + 44} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={isSel ? "var(--accent)" : "var(--muted)"}>b={bias.toFixed(1)}</text>
            )}
          </g>
        );
      })}
      {/* Travelling values: each source's contribution (value × weight) rides its line into the active neuron. */}
      {target && sources.map((src, i) => {
        const [sx, sy] = POS[src];
        const v = state.values[src]?.a ?? 0;
        const w = getParam(weights, edgeOf(src).param);
        const label = `${v.toFixed(v % 1 ? 2 : 0)} × ${w} = ${(v * w).toFixed(2)}`;
        const lw = label.length * 6.3 + 12;
        return (
          <motion.g key={`${animKey}-${src}`} initial={{ x: sx, y: sy, opacity: 0 }} animate={{ x: [sx, target[0]], y: [sy, target[1]], opacity: [0, 1, 1, 0] }}
            transition={{ delay: pulseStart(i) * k, duration: TRAVEL * k, ease: "easeInOut", opacity: { delay: pulseStart(i) * k, duration: TRAVEL * k, times: [0, 0.12, 0.85, 1] } }}>
            <circle r={8} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
            {/* The label fades before the dot lands so it never covers the neuron's own number. */}
            <motion.g initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 1, 0] }} transition={{ delay: pulseStart(i) * k, duration: TRAVEL * k, times: [0, 0.15, 0.6, 0.8] }}>
              <rect x={-lw / 2} y={-30} width={lw} height={18} rx={9} fill="var(--fg)" />
              <text y={-17} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--surface)">{label}</text>
            </motion.g>
          </motion.g>
        );
      })}
      {target && state.phase === "sum" && (() => {
        const b = state.active === "y" ? weights.b2 : weights.b1[Number(state.active![1]) - 1];
        return (
          <motion.g key={`${animKey}-bias`} initial={{ opacity: 0, x: target[0], y: target[1] - 70 }} animate={{ opacity: [0, 1, 1, 0], y: [target[1] - 70, target[1] - 58, target[1] - 58, target[1]] }}
            transition={{ delay: biasAt(sources.length) * k, duration: 0.8 * k, times: [0, 0.25, 0.6, 1] }}>
            <rect x={-34} y={-10} width={68} height={20} rx={10} fill="var(--violet)" />
            <text y={4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--surface)">+ bias {b}</text>
          </motion.g>
        );
      })()}

      {(["input", "hidden", "output"] as const).map((l) => (
        <text key={l} x={LAYER_X[l]} y={378} textAnchor="middle" fontSize={11} fontWeight={layer === l ? 700 : 400} fill={layer === l ? "var(--accent)" : "var(--muted)"}>
          {l === "input" ? "input layer" : l === "hidden" ? "1 hidden layer · 3 neurons" : "output (sigmoid)"}
        </text>
      ))}
      <AnimatePresence>
        {prediction !== undefined && (
          <motion.g key="pred" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ type: "spring", stiffness: 200, damping: 18, delay: 0.3 * k }}>
            <rect x={518} y={100} width={104} height={40} rx={6} fill={prediction >= 0.5 ? "var(--ok-soft)" : "var(--surface-2)"} stroke="var(--line)" />
            <text x={570} y={116} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">ŷ = {prediction.toFixed(3)}</text>
            <text x={570} y={132} textAnchor="middle" fontSize={11} fill="var(--fg)">class {prediction >= 0.5 ? 1 : 0}</text>
          </motion.g>
        )}
      </AnimatePresence>
    </svg>
  );
}

function Calculation({ input, active, phase }: { input: NetworkInput; active: string | null; phase: string }) {
  const r = forward(input);
  const { x, weights: w, activation } = input;
  if (!active) {
    return phase === "input" ? (
      <KV rows={[["x1", x[0]], ["x2", x[1]]]} />
    ) : (
      <TexBlock>{`\\hat{y} = ${r.y.toFixed(4)} \\qquad \\text{class} = \\begin{cases} 1 & \\hat{y} \\ge 0.5 \\\\ 0 & \\text{otherwise} \\end{cases} = ${r.y >= 0.5 ? 1 : 0}`}</TexBlock>
    );
  }
  if (active === "y") {
    return phase === "sum" ? (
      <TexBlock>{`\\begin{aligned} z_y &= \\textstyle\\sum_{j} w_{h_j \\to y}\\, h_j + b_y \\\\ &= ${w.w2.map((v, j) => `${n(v)} \\cdot ${n(r.h[j], 3)}`).join(" + ")} + ${n(w.b2)} \\\\ &= ${r.z2.toFixed(4)} \\end{aligned}`}</TexBlock>
    ) : (
      <TexBlock>{`\\begin{aligned} ${ACTIVATIONS.sigmoid.tex.replace("=", "&=")} \\\\ \\hat{y} &= \\sigma(${r.z2.toFixed(4)}) = ${r.y.toFixed(4)} \\end{aligned}`}</TexBlock>
    );
  }
  const j = Number(active[1]) - 1;
  return (
    <div className="space-y-2">
      {phase === "sum" ? (
        <TexBlock>{`\\begin{aligned} z_{h_${j + 1}} &= w_{x_1 \\to h_${j + 1}}\\, x_1 + w_{x_2 \\to h_${j + 1}}\\, x_2 + b_{h_${j + 1}} \\\\ &= ${n(w.w1[j][0])} \\cdot ${n(x[0])} + ${n(w.w1[j][1])} \\cdot ${n(x[1])} + ${n(w.b1[j])} \\\\ &= ${r.z1[j].toFixed(4)} \\end{aligned}`}</TexBlock>
      ) : (
        <TexBlock>{`\\begin{aligned} ${ACTIVATIONS[activation].tex.replace("=", "&=")} \\\\ a_{h_${j + 1}} &= ${ACTIVATIONS[activation].name}(${r.z1[j].toFixed(4)}) = ${r.h[j].toFixed(4)} \\end{aligned}`}</TexBlock>
      )}
      <div className="flex gap-1"><Badge>neuron {active}</Badge><Badge>{phase === "sum" ? "linear step" : "non-linear step"}</Badge></div>
    </div>
  );
}

function DatasetMetrics({ evaluation, features }: { evaluation: ReturnType<typeof evaluateDataset>; features: [string, string] }) {
  return (
    <div className="space-y-3">
      <Callout title="Evaluated with the current weights">
        Every row is a real forward pass with the parameters you set on the Visualize tab. Edit a weight there and these numbers change.
      </Callout>
      <div className="grid gap-3 sm:grid-cols-2">
        <Panel title="Accuracy"><div className="font-mono text-2xl" data-testid="nn-accuracy">{(evaluation.accuracy * 100).toFixed(0)}%</div><p className="text-xs text-muted">correct predictions at threshold 0.5</p></Panel>
        <Panel title="Log loss"><div className="font-mono text-2xl">{evaluation.logLoss.toFixed(4)}</div><div className="mt-1 text-xs text-muted"><Tex>{"\\mathcal{L} = -\\frac{1}{n}\\sum_i \\big[y_i \\log \\hat{y}_i + (1 - y_i)\\log(1 - \\hat{y}_i)\\big]"}</Tex><p className="mt-1">Lower is better; punishes confident mistakes.</p></div></Panel>
      </div>
      <Panel title="Per-sample predictions">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">{features[0]}</th><th>{features[1]}</th><th>target</th><th><Tex>{"\\hat{y}"}</Tex></th><th>predicted</th><th>result</th></tr></thead>
            <tbody className="font-mono">
              {evaluation.rows.map((r) => (
                <tr key={r.label} className="border-t border-line">
                  <td className="py-1">{r.x[0]}</td><td>{r.x[1]}</td><td>{r.target}</td><td>{r.y.toFixed(4)}</td><td>{r.pred}</td>
                  <td>{r.pred === r.target ? <Badge tone="ok">correct</Badge> : <Badge tone="err">wrong</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
