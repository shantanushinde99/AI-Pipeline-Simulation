import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw, RotateCw, StepForward } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, KV, Panel, Select, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { ACTIVATIONS, DATASETS, randomWeights, type Activation, type Weights } from "../neural-network/simulation";
import { DERIV, simulate, type BPState, type Grads } from "./simulation";

const xor = DATASETS[0];
const PRESETS: Record<string, { name: string; w: Weights }> = {
  r7: { name: "Random (seed 7)", w: randomWeights(7) },
  r3: { name: "Random (seed 3)", w: randomWeights(3) },
  hand: { name: "Hand-tuned XOR", w: xor.preset },
};
const ease = [0.22, 1, 0.36, 1] as const;
/** One schedule (seconds at 1×) shared by the diagram and the step card. */
const T = { travel: 0.8, a: 0.2, b: 1.1, c: 1.6, land: 2.4, count: 0.6 };
const DUR: Record<BPState["phase"], number> = { forward: 3000, loss: 2600, delta: 2400, "grad-out": 3200, hidden: 3800, update: 2800, after: 2600 };

const POS: Record<string, [number, number]> = { x1: [80, 130], x2: [80, 250], h1: [310, 70], h2: [310, 190], h3: [310, 310], y: [530, 190], L: [645, 190] };
type Edge = { from: string; to: string; key: string; get: (w: Weights) => number; t: number };
const EDGES: Edge[] = [
  ...[0, 1, 2].flatMap((j) => [0, 1].map((i) => ({ from: `x${i + 1}`, to: `h${j + 1}`, key: `w1-${j}-${i}`, get: (w: Weights) => w.w1[j][i], t: 0.72 }))),
  ...[0, 1, 2].map((j) => ({ from: `h${j + 1}`, to: "y", key: `w2-${j}`, get: (w: Weights) => w.w2[j], t: 0.55 })),
];
const gradOf = (g: Grads, key: string) => {
  const [k, a, b] = key.split("-");
  return k === "w1" ? g.w1[+a][+b] : k === "b1" ? g.b1[+a] : k === "w2" ? g.w2[+a] : g.b2;
};
const n = (v: number, d = 3) => (v < 0 ? `(${v.toFixed(d)})` : v.toFixed(d));

export default function BackpropVisualizer({ tab }: { tab: VisualizerTab }) {
  void tab; // no metrics tab
  const [preset, setPreset] = useState("r7");
  const [weights, setWeights] = useState<Weights>(PRESETS.r7.w);
  const [sample, setSample] = useState(2);
  const [activation, setActivation] = useState<Activation>("tanh");
  const [lr, setLr] = useState(0.5);
  const [history, setHistory] = useState<number[]>([]);
  const [replay, setReplay] = useState(0);
  const s = xor.samples[sample];
  const sim = useMemo(() => simulate({ x: s.x, weights, activation, target: s.target, lr }), [s, weights, activation, lr]);
  const { steps, bp, next, after } = sim;
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "forward"]);
  const step = steps[Math.min(pb.index, steps.length - 1)];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const st = step.state;

  const reset = () => { setWeights(PRESETS[preset].w); setHistory([]); pb.restart(); };
  const train = () => { setHistory((h) => [...(h.length ? h : [bp.loss]), after.loss]); setWeights(next); pb.restart(); };
  const story = narrate(st, bp.loss, after.loss);

  return (
    <Workspace
      main={
        <>
          <Panel title="Network: forward, then backward">
            <div className="mb-2 flex items-start gap-3">
              <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{pb.index + 1}/{steps.length}</span>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={step.id} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
                  <p className="text-base font-semibold leading-snug">{story[0]}</p>
                  <p className="text-sm text-muted">{story[1]}</p>
                </motion.div>
              </AnimatePresence>
              <Button size="sm" variant="ghost" onClick={() => setReplay((r) => r + 1)} title="Play this step's animation again"><RotateCw size={13} /> Replay step</Button>
            </div>
            <div className="overflow-x-auto"><div className="min-w-[520px]">
              <NetSvg sim={sim} st={st} weights={weights} x={s.x} k={k} animKey={animKey} />
            </div></div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>forward value</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--violet)" /></svg>gradient travelling backward</span>
              <span>violet glow width ∝ |∂L/∂w| (gradient already computed)</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <Stage sim={sim} st={st} weights={weights} target={s.target} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Example & training settings" actions={<Button size="sm" variant="ghost" onClick={reset}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Starting weights" value={preset} options={Object.entries(PRESETS).map(([v, p]) => ({ value: v, label: p.name }))} onChange={(v) => { setPreset(v); setWeights(PRESETS[v].w); setHistory([]); }} />
              <Select label="Hidden activation" value={activation} options={(Object.keys(ACTIVATIONS) as Activation[]).map((a) => ({ value: a, label: a }))} onChange={setActivation} />
              <Slider label="Learning rate η" value={lr} min={0.05} max={8} step={0.05} format={(v) => v.toFixed(2)} onChange={setLr} hint="Try 6+ to overshoot (break it)." />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">XOR example:</span>
              {xor.samples.map((x, i) => (
                <Button key={x.label} size="sm" variant={i === sample ? "primary" : "secondary"} onClick={() => { setSample(i); setHistory([]); }}>{x.label} → {x.target}</Button>
              ))}
              <Button size="sm" variant="primary" onClick={train} className="ml-auto"><StepForward size={13} /> Apply update & backprop again</Button>
            </div>
            {history.length > 1 && (
              <p className="mt-2 font-mono text-xs text-muted">loss per update: {history.map((v) => v.toFixed(3)).join(" → ")}</p>
            )}
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Chain rule, with numbers"><ChainTex sim={sim} st={st} weights={weights} activation={activation} target={s.target} /></Panel>
          <Panel title="Key values">
            <KV rows={[["target t", s.target], ["prediction ŷ", bp.f.y.toFixed(4)], ["loss before", bp.loss.toFixed(4)], ["η", lr], ["loss after one step", after.loss.toFixed(4)]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(st: BPState, before: number, afterLoss: number): [string, string] {
  switch (st.phase) {
    case "forward": return ["First, a normal forward pass makes a prediction", "Values flow left to right exactly as in the neural-network visualizer."];
    case "loss": return ["The loss scores how wrong the prediction is", "Cross-entropy is small when ŷ is close to the true label and huge when confidently wrong."];
    case "delta": return ["The error signal starts its trip backward", "At the output, the chain rule simplifies to ŷ − t."];
    case "grad-out": return ["Each output weight learns how much it is to blame", "Gradient = error signal × the value that travelled along that weight."];
    case "hidden": return [`The signal flows back through h${st.hidden! + 1}`, "Multiply by the weight it came through, then by the activation's slope, then hand it to the inputs."];
    case "update": return ["Every weight takes a small step downhill", "w ← w − η·∂L/∂w. Watch each weight label change."];
    case "after": return [afterLoss < before ? "Loss went down: the network learned a little" : "Loss went UP: the step overshot", "Press “Apply update & backprop again” to keep training."];
  }
}

function Dot({ from, to, delay, k, color, label, animKey }: { from: [number, number]; to: [number, number]; delay: number; k: number; color: string; label?: string; animKey: string }) {
  const w = (label?.length ?? 0) * 6.3 + 12;
  return (
    <motion.g key={animKey} initial={{ x: from[0], y: from[1], opacity: 0 }} animate={{ x: [from[0], to[0]], y: [from[1], to[1]], opacity: [0, 1, 1, 0] }}
      transition={{ delay: delay * k, duration: T.travel * k, ease: "easeInOut", opacity: { delay: delay * k, duration: T.travel * k, times: [0, 0.12, 0.85, 1] } }}>
      <circle r={7} fill={color} stroke="var(--surface)" strokeWidth={2} />
      {label && <><rect x={-w / 2} y={-28} width={w} height={17} rx={8} fill="var(--fg)" /><text y={-16} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--surface)">{label}</text></>}
    </motion.g>
  );
}

function NetSvg({ sim, st, weights, x, k, animKey }: { sim: ReturnType<typeof simulate>; st: BPState; weights: Weights; x: [number, number]; k: number; animKey: string }) {
  const { bp, next, after } = sim;
  const g = bp.grads;
  const maxG = Math.max(1e-9, ...EDGES.map((e) => Math.abs(gradOf(g, e.key))));
  const live = (e: Edge) => (st.phase === "grad-out" && e.to === "y") || (st.phase === "hidden" && e.to === `h${st.hidden! + 1}`);
  const nodeVal: Record<string, number> = { x1: x[0], x2: x[1],  h1: bp.f.h[0], h2: bp.f.h[1], h3: bp.f.h[2], y: bp.f.y };
  const j = st.hidden ?? 0;
  const hId = `h${j + 1}`;
  return (
    <svg viewBox="0 0 700 380" className="h-auto w-full" role="img" aria-label="2-3-1 network with gradients flowing backward">
      {EDGES.map((e) => {
        const [x1, y1] = POS[e.from], [x2, y2] = POS[e.to];
        const w = e.get(weights), gv = gradOf(g, e.key);
        const shown = st.shown.includes(e.key);
        const lx = x1 + (x2 - x1) * e.t, ly = y1 + (y2 - y1) * e.t;
        const gt = e.to === "y" ? 0.22 : 0.38;
        return (
          <g key={e.key}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--muted)" strokeOpacity={0.45} strokeWidth={Math.min(1 + Math.abs(w) * 0.8, 6)} strokeDasharray={w < 0 ? "6 4" : undefined} />
            {shown && <motion.line key={`${animKey}-${e.key}`} x1={x2} y1={y2} x2={x1} y2={y1} stroke="var(--violet)" strokeOpacity={0.55} strokeLinecap="round" strokeWidth={2 + 8 * Math.abs(gv) / maxG}
              initial={{ pathLength: live(e) ? 0 : 1 }} animate={{ pathLength: 1 }} transition={{ delay: (e.to === "y" ? T.a : T.c) * k, duration: T.travel * k, ease }} />}
            <rect x={lx - 22} y={ly - 9} width={44} height={18} rx={4} fill={st.phase === "update" ? "var(--accent-soft)" : "var(--surface)"} stroke="var(--line)" />
            <text x={lx} y={ly + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">
              {st.phase === "update" ? <AnimatedNumber key={animKey} svg value={e.get(next)} from={w} delay={0.3 * k} duration={1.2 * k} /> : st.phase === "after" ? e.get(next).toFixed(2) : w.toFixed(2)}
            </text>
            {live(e) && (
              <motion.g key={`${animKey}-g-${e.key}`} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (e.to === "y" ? T.a + T.travel : T.land) * k, duration: 0.3 * k }}
                style={{ transformOrigin: `${x1 + (x2 - x1) * gt}px ${y1 + (y2 - y1) * gt + 18}px` }}>
                <rect x={x1 + (x2 - x1) * gt - 40} y={y1 + (y2 - y1) * gt + 9} width={80} height={18} rx={9} fill="var(--violet)" />
                <text x={x1 + (x2 - x1) * gt} y={y1 + (y2 - y1) * gt + 22} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--surface)">∂L/∂w={gv.toFixed(3)}</text>
              </motion.g>
            )}
          </g>
        );
      })}
      {/* loss box */}
      <line x1={558} y1={190} x2={605} y2={190} stroke="var(--line)" strokeWidth={2} />
      <rect x={605} y={160} width={82} height={60} rx={8} fill={st.phase === "loss" || st.phase === "after" ? "var(--err-soft)" : "var(--surface-2)"} stroke="var(--line)" />
      <text x={646} y={182} textAnchor="middle" fontSize={11} fill="var(--muted)">loss L</text>
      <text x={646} y={205} textAnchor="middle" fontSize={14} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">
        {st.phase === "forward" ? "·" : st.phase === "loss" ? <AnimatedNumber key={animKey} svg value={bp.loss} from={0} digits={3} delay={(T.a + T.travel) * k} duration={T.count * k} />
          : st.phase === "after" ? <AnimatedNumber key={animKey} svg value={after.loss} from={bp.loss} digits={3} delay={0.3 * k} duration={1 * k} /> : bp.loss.toFixed(3)}
      </text>
      {Object.entries(POS).filter(([id]) => id !== "L").map(([id, [x, y]]) => {
        const active = (st.phase === "hidden" && id === hId) || (["delta", "grad-out"].includes(st.phase) && id === "y");
        const v = nodeVal[id];
        return (
          <g key={id}>
            {active && <motion.circle cx={x} cy={y} fill="none" stroke="var(--violet)" strokeWidth={2} initial={{ r: 28, opacity: 0.6 }} animate={{ r: 42, opacity: 0 }} transition={{ duration: 1.4, repeat: Infinity }} />}
            <circle cx={x} cy={y} r={27} fill="var(--surface)" stroke={active ? "var(--violet)" : "var(--line)"} strokeWidth={active ? 3.5 : 1.5} />
            <text x={x} y={y - 1} textAnchor="middle" fontSize={12.5} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">
              {st.phase === "forward" && id !== "x1" && id !== "x2" ? <AnimatedNumber key={animKey} svg value={v} from={0} delay={(id === "y" ? T.b + T.travel : T.a + T.travel) * k} duration={0.4 * k} /> : v.toFixed(2)}
            </text>
            <text x={x} y={y + 13} textAnchor="middle" fontSize={10} fill="var(--muted)">{id === "y" ? "ŷ" : id}</text>
            {id.startsWith("h") && st.phase !== "forward" && (
              <text x={x} y={y + 42} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={bp.dact[+id[1] - 1] === 0 ? "var(--err)" : "var(--muted)"}>f′(z)={bp.dact[+id[1] - 1].toFixed(2)}</text>
            )}
          </g>
        );
      })}
      {/* travelling values */}
      {st.phase === "forward" && EDGES.map((e, i) => <Dot key={`${animKey}-f${i}`} animKey={`${animKey}-f${i}`} from={POS[e.from]} to={POS[e.to]} delay={e.to === "y" ? T.b : T.a} k={k} color="var(--accent)" />)}
      {st.phase === "loss" && <Dot animKey={`${animKey}-l`} from={POS.y} to={POS.L} delay={T.a} k={k} color="var(--accent)" label={`ŷ=${bp.f.y.toFixed(3)}`} />}
      {st.phase === "delta" && <Dot animKey={`${animKey}-d`} from={POS.L} to={POS.y} delay={T.a} k={k} color="var(--violet)" label={`δ=${bp.dz2.toFixed(3)}`} />}
      {st.phase === "grad-out" && [0, 1, 2].map((i) => <Dot key={i} animKey={`${animKey}-g${i}`} from={POS.y} to={POS[`h${i + 1}`]} delay={T.a} k={k} color="var(--violet)" />)}
      {st.phase === "hidden" && (
        <>
          <Dot animKey={`${animKey}-h`} from={POS.y} to={POS[hId]} delay={T.a} k={k} color="var(--violet)" label={`δ·w=${bp.dh[j].toFixed(3)}`} />
          <motion.text key={`${animKey}-fp`} x={POS[hId][0]} y={POS[hId][1] - 36} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--violet)"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.b * k }}>× f′ → δ={bp.dz1[j].toFixed(3)}</motion.text>
          {[0, 1].map((i) => <Dot key={i} animKey={`${animKey}-x${i}`} from={POS[hId]} to={POS[`x${i + 1}`]} delay={T.c} k={k} color="var(--violet)" />)}
        </>
      )}
      {(["input", "hidden", "output"] as const).map((l, i) => <text key={l} x={[80, 310, 530][i]} y={372} textAnchor="middle" fontSize={11} fill="var(--muted)">{l}</text>)}
    </svg>
  );
}
function Stage({ sim, st, weights, target, k }: { sim: ReturnType<typeof simulate>; st: BPState; weights: Weights; target: number; k: number }) {
  const { bp, next, after } = sim;
  const bar = (v: number, max: number, at: number, color = "bg-violet/70") => (
    <div className="relative h-4 rounded bg-surface-2">
      <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
      <motion.div className={cx("absolute inset-y-0.5 rounded", color)} style={v < 0 ? { right: "50%" } : { left: "50%" }} initial={{ width: 0 }} animate={{ width: `${(Math.abs(v) / Math.max(max, 1e-9)) * 50}%` }} transition={{ delay: at * k, duration: 0.5 * k, ease }} />
    </div>
  );
  if (st.phase === "forward") return <p className="text-sm">Hidden outputs <span className="font-mono">h = [{bp.f.h.map((v) => v.toFixed(3)).join(", ")}]</span>, prediction <span className="font-mono">ŷ = {bp.f.y.toFixed(4)}</span>. Backprop will reuse every one of these numbers.</p>;
  if (st.phase === "loss" || st.phase === "delta") {
    return (
      <div className="space-y-2">
        <div className="relative h-8 rounded-md bg-surface-2">
          <motion.div className="absolute inset-y-0 left-0 rounded-md bg-accent/50" initial={{ width: 0 }} animate={{ width: `${bp.f.y * 100}%` }} transition={{ delay: T.a * k, duration: 0.8 * k, ease }} />
          <div className="absolute inset-y-0 w-1 bg-ok" style={{ left: `calc(${target * 100}% - ${target * 4}px)` }} />
          <span className="absolute top-1/2 left-2 -translate-y-1/2 font-mono text-xs">ŷ = {bp.f.y.toFixed(3)}</span>
          <span className="absolute top-1/2 right-2 -translate-y-1/2 text-xs text-ok">✓ target t = {target}</span>
        </div>
        <p className="text-sm">{st.phase === "loss" ? <>Loss L = <span className="font-mono"><AnimatedNumber value={bp.loss} from={0} digits={4} delay={(T.a + T.travel) * k} duration={T.count * k} /></span>. The gap between the bar and the green mark is what training will shrink.</>
          : <>δ_y = ŷ − t = <span className="font-mono"><AnimatedNumber value={bp.dz2} from={0} digits={4} delay={T.a * k} duration={T.travel * k} /></span>. Negative means “ŷ is too low, push z_y up”; positive means “too high”.</>}</p>
      </div>
    );
  }
  if (st.phase === "grad-out" || st.phase === "hidden") {
    const j = st.hidden ?? 0;
    const rows = st.phase === "grad-out"
      ? [0, 1, 2].map((i) => ({ label: `w h${i + 1}→y`, expr: `${n(bp.dz2)} × h${i + 1}=${n(bp.f.h[i])}`, v: bp.grads.w2[i], at: T.a + T.travel }))
      : [
        { label: `∂L/∂h${j + 1}`, expr: `δ_y × w=${weights.w2[j].toFixed(2)}`, v: bp.dh[j], at: T.a + T.travel },
        { label: `δ_h${j + 1}`, expr: `× f′(z)=${bp.dact[j].toFixed(3)}`, v: bp.dz1[j], at: T.b },
        ...[0, 1].map((i) => ({ label: `w x${i + 1}→h${j + 1}`, expr: `δ_h × x${i + 1}`, v: bp.grads.w1[j][i], at: T.land })),
      ];
    const max = Math.max(...rows.map((r) => Math.abs(r.v)));
    return (
      <div className="space-y-1.5">
        {rows.map((r) => (
          <motion.div key={r.label} className="grid grid-cols-[110px_minmax(0,1fr)_64px] items-center gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (r.at - 0.2) * k }}>
            <div className="font-mono text-xs leading-tight">{r.label}<div className="text-[10px] text-muted">{r.expr}</div></div>
            {bar(r.v, max, r.at)}
            <div className="text-right font-mono text-xs"><AnimatedNumber value={r.v} from={0} digits={4} delay={r.at * k} duration={0.5 * k} /></div>
          </motion.div>
        ))}
        {st.phase === "hidden" && bp.dact[j] === 0 && <p className="rounded-md bg-warn-soft px-2 py-1 text-xs">⚠ h{j + 1} is a dead ReLU here (z ≤ 0, slope 0): the signal stops and its weights get no gradient.</p>}
      </div>
    );
  }
  if (st.phase === "update") {
    const keys = [...EDGES.map((e) => e.key), "b1-0", "b1-1", "b1-2", "b2"];
    const get = (w: Weights, key: string) => gradOf(w, key);
    return (
      <div className="grid grid-cols-1 gap-x-4 gap-y-0.5 font-mono text-[11px] sm:grid-cols-2">
        {keys.map((key, i) => (
          <motion.div key={key} className="flex justify-between gap-2 border-b border-line py-0.5" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.1 + i * 0.06) * k }}>
            <span className="text-muted">{key}</span>
            <span>{get(weights, key).toFixed(3)} − {sim.lr}×{n(gradOf(bp.grads, key))} = <strong><AnimatedNumber value={get(next, key)} from={get(weights, key)} digits={3} delay={0.3 * k} duration={1.2 * k} /></strong></span>
          </motion.div>
        ))}
      </div>
    );
  }
  const max = Math.max(bp.loss, after.loss);
  return (
    <div className="space-y-2">
      {[["before", bp.loss, bp.loss], ["after", after.loss, bp.loss]].map(([name, v, from]) => (
        <div key={name as string} className="grid grid-cols-[60px_minmax(0,1fr)_64px] items-center gap-2 text-xs">
          <span>{name}</span>
          <div className="h-5 rounded bg-surface-2"><motion.div className={cx("h-full rounded", name === "after" ? (after.loss < bp.loss ? "bg-ok/70" : "bg-err/70") : "bg-muted/50")} initial={{ width: `${((from as number) / max) * 100}%` }} animate={{ width: `${((v as number) / max) * 100}%` }} transition={{ delay: 0.3 * k, duration: 1 * k, ease }} /></div>
          <span className="text-right font-mono">{(v as number).toFixed(4)}</span>
        </div>
      ))}
      <p className="text-sm">{after.loss < bp.loss ? `✓ Loss fell by ${(bp.loss - after.loss).toFixed(4)}.` : `✗ Loss rose by ${(after.loss - bp.loss).toFixed(4)}: η = ${sim.lr} is too big for this slope.`}</p>
    </div>
  );
}

function ChainTex({ sim, st, weights, activation, target }: { sim: ReturnType<typeof simulate>; st: BPState; weights: Weights; activation: Activation; target: number }) {
  const { bp, after } = sim;
  const j = st.hidden ?? 0;
  const tex =
    st.phase === "forward" ? `\\hat{y} = \\sigma(z_y) = \\sigma(${bp.f.z2.toFixed(4)}) = ${bp.f.y.toFixed(4)}`
      : st.phase === "loss" ? `L = -\\big[t\\log\\hat{y} + (1-t)\\log(1-\\hat{y})\\big] = -\\log(${(target ? bp.f.y : 1 - bp.f.y).toFixed(4)}) = ${bp.loss.toFixed(4)}`
      : st.phase === "delta" ? `\\delta_y = \\frac{\\partial L}{\\partial \\hat{y}}\\frac{\\partial \\hat{y}}{\\partial z_y} = \\hat{y} - t = ${bp.f.y.toFixed(4)} - ${target} = ${bp.dz2.toFixed(4)}`
      : st.phase === "grad-out" ? `\\begin{aligned} \\frac{\\partial L}{\\partial w_{h_j \\to y}} &= \\delta_y\\, h_j \\\\ ${[0, 1, 2].map((i) => `\\frac{\\partial L}{\\partial w_{h_${i + 1}\\to y}} &= ${n(bp.dz2, 4)} \\cdot ${n(bp.f.h[i], 4)} = ${bp.grads.w2[i].toFixed(4)}`).join(" \\\\ ")} \\end{aligned}`
      : st.phase === "hidden" ? `\\begin{aligned} \\frac{\\partial L}{\\partial w_{x_i \\to h_${j + 1}}} &= \\underbrace{\\delta_y\\, w_{h_${j + 1}\\to y}}_{\\partial L/\\partial h_${j + 1}}\\; f'(z_{h_${j + 1}})\\; x_i \\\\ \\delta_{h_${j + 1}} &= ${n(bp.dz2, 4)} \\cdot ${n(weights.w2[j], 2)} \\cdot ${bp.dact[j].toFixed(4)} = ${bp.dz1[j].toFixed(4)} \\\\ ${DERIV[activation].tex.replace("=", "&=")} \\end{aligned}`
      : st.phase === "update" ? `w \\leftarrow w - \\eta\\,\\frac{\\partial L}{\\partial w}, \\quad \\eta = ${sim.lr}`
      : `L_{\\text{before}} = ${bp.loss.toFixed(4)} \\;\\to\\; L_{\\text{after}} = ${after.loss.toFixed(4)}`;
  return <TexBlock>{tex}</TexBlock>;
}
