import { useCallback, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { LineChart } from "../../components/visualizer/LineChart";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, Slider } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, GrowBar, StepCaption } from "./StepCaption";
import {
  CLASS_NAME, DATASETS, FEATURE, HI, LO, MAX_DEPTH, N_TRAIN, gini, predict, simulate,
  type Candidate, type DatasetId, type Pt, type TreeConfig, type TreeNode, type TreeSim, type TreeState,
} from "./simulation";

const DEFAULT: TreeConfig = { dataset: "blocks", maxDepth: 3, minSamples: 2, noise: 0.1, seed: 4 };
const PRESETS: { label: string; cfg: Partial<TreeConfig> }[] = [
  { label: "Stump (depth 1)", cfg: { maxDepth: 1, minSamples: 2 } },
  { label: "Balanced (depth 3)", cfg: { maxDepth: 3, minSamples: 2 } },
  { label: "Break it: depth 8 + 20% noise", cfg: { maxDepth: 8, minSamples: 2, noise: 0.2 } },
  { label: "Regularised (min 12 samples)", cfg: { maxDepth: 8, minSamples: 12, noise: 0.2 } },
];

/** One schedule (seconds at 1×) for a split step: x-sweep, y-sweep, lock the winner, grow the children. */
const T = { xs: 0.3, sweep: 1.6, ys: 2.1, lock: 3.9, grow: 4.4 };
const DURATION = { data: 2600, split: 5800, evaluate: 3800 };
const CLASS_COLOR = ["var(--accent)", "var(--warn)"] as const;

export default function DecisionTreeVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT);
  const set = (p: Partial<TreeConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useCallback((i: number) => DURATION[sim.steps[i]?.state.phase ?? "split"], [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const [replay, setReplay] = useState(0);
  const step = sim.steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const st = step.state;
  const node = st.search ? sim.nodes[st.search.nodeId] : null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Feature space and tree" actions={<span className="flex gap-1"><Badge tone="ok">train {(sim.trainAcc * 100).toFixed(0)}%</Badge><Badge tone={sim.trainAcc - sim.testAcc > 0.1 ? "warn" : "neutral"}>test {(sim.testAcc * 100).toFixed(0)}%</Badge></span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={st.headline} sub={st.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="space-y-3">
              <div className="mx-auto max-w-[500px]"><PlaneSvg sim={sim} st={st} k={k} animKey={animKey} /></div>
              <TreeSvg sim={sim} st={st} k={k} animKey={animKey} />
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="4" fill={CLASS_COLOR[0]} /></svg>class A</span>
              <span className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><path d="M5 1 L9 9 L1 9 Z" fill={CLASS_COLOR[1]} /></svg>class B</span>
              <span>hollow = test point</span>
              <span>shaded rectangle = a leaf, tinted by the class it predicts</span>
              <span>curve = weighted Gini for each threshold (lower is better)</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} st={st} k={k} maxDepth={cfg.maxDepth} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Dataset" value={cfg.dataset} options={DATASETS} onChange={(v: DatasetId) => set({ dataset: v })} />
              <Slider label="Max depth" value={cfg.maxDepth} min={1} max={MAX_DEPTH} onChange={(v) => set({ maxDepth: v })} hint="Deeper = more rectangles = more chances to memorise noise." />
              <Slider label="Min samples to split" value={cfg.minSamples} min={2} max={20} onChange={(v) => set({ minSamples: v })} hint="Nodes smaller than this become leaves." />
              <Slider label="Label noise" value={cfg.noise} min={0} max={0.3} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ noise: v })} hint="Share of labels flipped at random." />
              <Slider label="Random seed" value={cfg.seed} min={1} max={50} onChange={(v) => set({ seed: v })} hint="Same seed → same points." />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => set(p.cfg)}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="A tree is a stack of greedy, one-feature yes/no questions. Each split is the locally best one; nothing guarantees the whole tree is the best possible, and nothing stops it carving a rectangle around a single noisy point." />
          <Panel title="Calculation"><Calculation sim={sim} st={st} node={node} /></Panel>
          <Panel title="Outputs">
            <KV rows={[
              ["leaves", sim.leaves],
              ["depth reached", Math.max(...sim.nodes.map((n) => n.depth))],
              ["train accuracy", `${(sim.trainAcc * 100).toFixed(1)}%`],
              ["test accuracy", `${(sim.testAcc * 100).toFixed(1)}%`],
              ["train − test gap", `${((sim.trainAcc - sim.testAcc) * 100).toFixed(1)} pts`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

// ---------- plane ----------
const P = { x0: 40, size: 300, top: 14 };
const sx = (v: number) => P.x0 + ((v - LO) / (HI - LO)) * P.size;
const sy = (v: number) => P.top + P.size - ((v - LO) / (HI - LO)) * P.size;
/** Gini charts: below the plot for x₁ (g = 0.5 at the top), right of the plot for x₂ (g = 0 at the left). */
const GX = { top: P.top + P.size + 22, h: 70 };
const gy = (g: number) => GX.top + GX.h - (g / 0.5) * GX.h;
const GY = { left: P.x0 + P.size + 22, w: 70 };
const gxp = (g: number) => GY.left + (g / 0.5) * GY.w;

function Marker({ p, test, dim, delay = 0, k }: { p: Pt; test?: boolean; dim?: boolean; delay?: number; k: number }) {
  const c = CLASS_COLOR[p.label];
  const shape = p.label === 0
    ? <circle r={4.5} fill={test ? "var(--surface)" : c} stroke={c} strokeWidth={test ? 2 : 1} />
    : <path d="M0 -5.5 L5 4 L-5 4 Z" fill={test ? "var(--surface)" : c} stroke={c} strokeWidth={test ? 2 : 1} />;
  return (
    <motion.g initial={{ opacity: 0, scale: 0, x: sx(p.x[0]), y: sy(p.x[1]) }} animate={{ opacity: dim ? 0.18 : 1, scale: 1, x: sx(p.x[0]), y: sy(p.x[1]) }}
      transition={{ delay: delay * k, duration: 0.35 * k, ease: EASE }}>
      {shape}
    </motion.g>
  );
}

const frontier = (nodes: TreeNode[], vis: number[]) => {
  const s = new Set(vis);
  return vis.filter((id) => { const n = nodes[id]; return !n.split || !s.has(n.split.left); });
};

function RegionRect({ n, delay, k, fade }: { n: TreeNode; delay: number; k: number; fade?: "in" | "out" }) {
  const r = n.region;
  const total = n.counts[0] + n.counts[1];
  const purity = total ? Math.max(...n.counts) / total : 0;
  const op = 0.06 + 0.2 * (purity - 0.5) * 2;
  return (
    <motion.rect x={sx(r.x0)} y={sy(r.y1)} width={sx(r.x1) - sx(r.x0)} height={sy(r.y0) - sy(r.y1)} fill={CLASS_COLOR[n.prediction]} stroke="var(--fg)" strokeOpacity={0.35} strokeWidth={1}
      initial={{ opacity: fade === "in" ? 0 : 1, fillOpacity: op }} animate={{ opacity: fade === "out" ? 0 : 1, fillOpacity: op }} transition={{ delay: delay * k, duration: 0.6 * k }} />
  );
}

function curvePath(c: Candidate[], toXY: (c: Candidate) => [number, number]) {
  return c.map((v, i) => { const [x, y] = toXY(v); return `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`; }).join(" ");
}

function PlaneSvg({ sim, st, k, animKey }: { sim: TreeSim; st: TreeState; k: number; animKey: string }) {
  const { nodes, train, test } = sim;
  const before = st.visible.filter((id) => !st.added.includes(id));
  const node = st.search ? nodes[st.search.nodeId] : null;
  const split = node?.split;
  const inNode = node ? new Set(node.idx) : null;
  const regionsBefore = frontier(nodes, before);
  const r = node?.region;
  const sweeping = st.phase === "split" && node && split && r && st.search;
  const [bx, by] = st.search?.bestPerFeature ?? [null, null];
  return (
    <svg viewBox="0 0 440 430" className="h-auto w-full" role="img" aria-label="Scatter plot of the two features with the tree's rectangular regions">
      <rect x={sx(LO)} y={sy(HI)} width={P.size} height={P.size} fill="none" stroke="var(--line)" />
      {[0, 2, 4, 6, 8, 10].map((v) => (
        <g key={v}>
          <text x={sx(v)} y={sy(LO) + 12} fontSize={9} textAnchor="middle" fill="var(--muted)">{v}</text>
          <text x={sx(LO) - 6} y={sy(v) + 3} fontSize={9} textAnchor="end" fill="var(--muted)">{v}</text>
        </g>
      ))}
      <text x={12} y={sy(5)} fontSize={11} fill="var(--muted)" transform={`rotate(-90 12 ${sy(5)})`} textAnchor="middle">{FEATURE[1]}</text>
      <g key={animKey}>
        {regionsBefore.map((id) => <RegionRect key={id} n={nodes[id]} delay={st.phase === "split" && id === node?.id ? T.grow : 0} k={k} fade={st.phase === "split" && id === node?.id ? "out" : undefined} />)}
        {st.added.map((id) => <RegionRect key={id} n={nodes[id]} delay={T.grow} k={k} fade="in" />)}
        {st.phase === "data" && <RegionRect n={nodes[0]} delay={1.4} k={k} fade="in" />}
      </g>
      {/* current node outline */}
      {r && <motion.rect key={`outline-${animKey}`} x={sx(r.x0)} y={sy(r.y1)} width={sx(r.x1) - sx(r.x0)} height={sy(r.y0) - sy(r.y1)} fill="none" stroke="var(--fg)" strokeWidth={2.5} strokeDasharray="6 4"
        initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 1, 0.4] }} transition={{ duration: T.grow * k, times: [0, 0.1, 0.85, 1] }} />}
      {train.map((p, i) => <Marker key={`tr-${i}-${st.phase === "data" ? animKey : ""}`} p={p} dim={!!inNode && !inNode.has(i)} delay={st.phase === "data" ? 0.1 + i * 0.03 : 0} k={k} />)}
      {st.phase === "evaluate" && test.map((p, i) => {
        const wrong = predict(nodes, p.x) !== p.label;
        return (
          <g key={`te-${i}-${animKey}`}>
            <Marker p={p} test delay={0.3 + i * 0.025} k={k} />
            {wrong && <motion.circle cx={sx(p.x[0])} cy={sy(p.x[1])} fill="none" stroke="var(--err)" strokeWidth={2} initial={{ r: 0, opacity: 0 }} animate={{ r: 9, opacity: 1 }} transition={{ delay: (1.6 + i * 0.02) * k, duration: 0.4 * k }} />}
          </g>
        );
      })}

      {/* Gini charts */}
      <rect x={sx(LO)} y={GX.top} width={P.size} height={GX.h} fill="var(--surface-2)" rx={3} />
      <rect x={GY.left} y={sy(HI)} width={GY.w} height={P.size} fill="var(--surface-2)" rx={3} />
      <text x={sx(LO) - 4} y={GX.top + 8} fontSize={8} textAnchor="end" fill="var(--muted)">0.5</text>
      <text x={sx(LO) - 4} y={GX.top + GX.h} fontSize={8} textAnchor="end" fill="var(--muted)">0</text>
      <text x={sx(5)} y={GX.top + GX.h + 13} fontSize={10} textAnchor="middle" fill="var(--muted)">{FEATURE[0]} threshold → weighted Gini (lower = better)</text>
      <text x={GY.left} y={sy(LO) + 12} fontSize={8} fill="var(--muted)">0</text>
      <text x={GY.left + GY.w} y={sy(LO) + 12} fontSize={8} textAnchor="end" fill="var(--muted)">0.5</text>
      <text x={GY.left + GY.w / 2} y={sy(HI) - 3} fontSize={9} textAnchor="middle" fill="var(--muted)">{FEATURE[1]} → Gini</text>
      {!sweeping && (
        <g fontSize={9} fill="var(--muted)" textAnchor="middle">
          <text x={sx(5)} y={GX.top + GX.h / 2 + 3}>threshold search runs during split steps</text>
          <text x={GY.left + GY.w / 2} y={sy(5)} transform={`rotate(90 ${GY.left + GY.w / 2} ${sy(5)})`}>search runs during splits</text>
        </g>
      )}
      {sweeping && (
        <g key={`sweep-${animKey}`}>
          {/* parent impurity reference */}
          <line x1={sx(r.x0)} x2={sx(r.x1)} y1={gy(node.gini)} y2={gy(node.gini)} stroke="var(--muted)" strokeDasharray="3 3" />
          <line y1={sy(r.y0)} y2={sy(r.y1)} x1={gxp(node.gini)} x2={gxp(node.gini)} stroke="var(--muted)" strokeDasharray="3 3" />
          <clipPath id={`clipx-${animKey}`}><motion.rect x={sx(r.x0)} y={GX.top - 2} height={GX.h + 4} initial={{ width: 0 }} animate={{ width: sx(r.x1) - sx(r.x0) }} transition={{ delay: T.xs * k, duration: T.sweep * k, ease: "linear" }} /></clipPath>
          <clipPath id={`clipy-${animKey}`}><motion.rect x={GY.left - 2} width={GY.w + 4} initial={{ y: sy(r.y0), height: 0 }} animate={{ y: sy(r.y1), height: sy(r.y0) - sy(r.y1) }} transition={{ delay: T.ys * k, duration: T.sweep * k, ease: "linear" }} /></clipPath>
          <g clipPath={`url(#clipx-${animKey})`}>
            <path d={curvePath(st.search!.cands[0], (c) => [sx(c.t), gy(c.g)])} fill="none" stroke="var(--violet)" strokeWidth={2} />
            {st.search!.cands[0].map((c) => <circle key={c.t} cx={sx(c.t)} cy={gy(c.g)} r={2} fill="var(--violet)" />)}
          </g>
          <g clipPath={`url(#clipy-${animKey})`}>
            <path d={curvePath(st.search!.cands[1], (c) => [gxp(c.g), sy(c.t)])} fill="none" stroke="var(--violet)" strokeWidth={2} />
            {st.search!.cands[1].map((c) => <circle key={c.t} cx={gxp(c.g)} cy={sy(c.t)} r={2} fill="var(--violet)" />)}
          </g>
          {/* the sweeping threshold lines */}
          <motion.line y1={sy(r.y0)} y2={GX.top + GX.h} stroke="var(--violet)" strokeWidth={2}
            initial={{ x1: sx(r.x0), x2: sx(r.x0), opacity: 0 }} animate={{ x1: sx(r.x1), x2: sx(r.x1), opacity: [0, 1, 1, 0] }}
            transition={{ delay: T.xs * k, duration: T.sweep * k, ease: "linear", opacity: { delay: T.xs * k, duration: T.sweep * k, times: [0, 0.05, 0.95, 1] } }} />
          <motion.line x1={sx(r.x0)} x2={GY.left + GY.w} stroke="var(--violet)" strokeWidth={2}
            initial={{ y1: sy(r.y0), y2: sy(r.y0), opacity: 0 }} animate={{ y1: sy(r.y1), y2: sy(r.y1), opacity: [0, 1, 1, 0] }}
            transition={{ delay: T.ys * k, duration: T.sweep * k, ease: "linear", opacity: { delay: T.ys * k, duration: T.sweep * k, times: [0, 0.05, 0.95, 1] } }} />
          {/* best per feature, then the winner */}
          {bx && <BestDot cx={sx(bx.t)} cy={gy(bx.g)} at={T.xs + T.sweep} win={split.feature === 0} k={k} label={`best ${bx.g.toFixed(3)}`} />}
          {by && <BestDot cx={gxp(by.g)} cy={sy(by.t)} at={T.ys + T.sweep} win={split.feature === 1} k={k} label={`best ${by.g.toFixed(3)}`} />}
          {split.feature === 0
            ? <motion.line x1={sx(split.threshold)} x2={sx(split.threshold)} y1={sy(r.y0)} y2={sy(r.y1)} stroke="var(--fg)" strokeWidth={3} initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: T.lock * k, duration: 0.5 * k, ease: EASE }} />
            : <motion.line y1={sy(split.threshold)} y2={sy(split.threshold)} x1={sx(r.x0)} x2={sx(r.x1)} stroke="var(--fg)" strokeWidth={3} initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: T.lock * k, duration: 0.5 * k, ease: EASE }} />}
          <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.lock * k, duration: 0.3 * k }}>
            {(() => {
              const lx = split.feature === 0 ? sx(split.threshold) : (sx(r.x0) + sx(r.x1)) / 2;
              const ly = split.feature === 0 ? (sy(r.y0) + sy(r.y1)) / 2 : sy(split.threshold);
              const label = `${FEATURE[split.feature]} ≤ ${split.threshold.toFixed(2)}`;
              const w = label.length * 6.4 + 12;
              return (<><rect x={lx - w / 2} y={ly - 9} width={w} height={18} rx={9} fill="var(--fg)" /><text x={lx} y={ly + 4} fontSize={11} textAnchor="middle" fill="var(--surface)" fontFamily="var(--font-mono)">{label}</text></>);
            })()}
          </motion.g>
        </g>
      )}
    </svg>
  );
}

function BestDot({ cx, cy, at, win, k, label }: { cx: number; cy: number; at: number; win: boolean; k: number; label: string }) {
  return (
    <g>
      <motion.circle cx={cx} cy={cy} r={5} fill={win ? "var(--ok)" : "var(--muted)"} stroke="var(--surface)" strokeWidth={1.5} initial={{ scale: 0 }} animate={{ scale: [0, 1.6, 1] }} transition={{ delay: at * k, duration: 0.4 * k }} />
      <motion.text x={cx + 7} y={cy - 6} fontSize={9} fontFamily="var(--font-mono)" fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: at * k }}>{label}</motion.text>
      {win && <motion.circle cx={cx} cy={cy} r={10} fill="none" stroke="var(--ok)" strokeWidth={2} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.lock * k, duration: 0.3 * k }} />}
    </g>
  );
}

// ---------- tree diagram ----------
const NW = 76, NH = 46, SLOT = 82, LEVEL = 70;

function layout(nodes: TreeNode[]) {
  const pos = new Map<number, [number, number]>();
  let i = 0;
  const walk = (id: number) => {
    const n = nodes[id];
    if (n.split) walk(n.split.left);
    pos.set(id, [i++ * SLOT + SLOT / 2, n.depth * LEVEL + NH / 2 + 6]);
    if (n.split) walk(n.split.right);
  };
  walk(0);
  return { pos, width: i * SLOT, height: (Math.max(...nodes.map((n) => n.depth)) + 1) * LEVEL };
}

function TreeSvg({ sim, st, k, animKey }: { sim: TreeSim; st: TreeState; k: number; animKey: string }) {
  const { pos, width, height } = useMemo(() => layout(sim.nodes), [sim.nodes]);
  const vis = new Set(st.visible);
  const active = st.search?.nodeId;
  return (
    <div className="overflow-x-auto rounded-md border border-line bg-surface-2/40">
      <svg viewBox={`0 0 ${Math.max(width, 240)} ${height}`} style={{ width: Math.max(width, 240) }} className="mx-auto block h-auto max-w-none" role="img" aria-label="Decision tree diagram">
        {st.visible.map((id) => {
          const n = sim.nodes[id];
          if (n.parent === undefined) return null;
          const [x1, y1] = pos.get(n.parent)!, [x2, y2] = pos.get(id)!;
          const isNew = st.added.includes(id);
          const isLeft = sim.nodes[n.parent].split!.left === id;
          return (
            <g key={`e-${id}-${isNew ? animKey : ""}`}>
              <motion.line x1={x1} y1={y1 + NH / 2} x2={x2} y2={y2 - NH / 2} stroke="var(--muted)" strokeWidth={1.5} initial={isNew ? { pathLength: 0, opacity: 0 } : false} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: T.grow * k, duration: 0.4 * k }} />
              <motion.text x={(x1 + x2) / 2 + (isLeft ? -6 : 6)} y={(y1 + y2) / 2 + 3} fontSize={9} textAnchor={isLeft ? "end" : "start"} fill="var(--muted)" initial={isNew ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: T.grow * k }}>{isLeft ? "yes" : "no"}</motion.text>
            </g>
          );
        })}
        {st.visible.map((id) => {
          const n = sim.nodes[id];
          const [x, y] = pos.get(id)!;
          const isNew = st.added.includes(id);
          const isActive = id === active;
          const showsSplit = n.split && vis.has(n.split.left);
          const ruleKnownAt = isActive ? T.lock : 0;
          const finalLeaf = !n.split && (st.phase === "evaluate" || n.leafReason);
          return (
            <motion.g key={`n-${id}-${isNew ? animKey : ""}`} initial={isNew ? { opacity: 0, scale: 0.6 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (isNew ? T.grow + 0.2 : 0) * k, duration: 0.4 * k, ease: EASE }} style={{ transformOrigin: `${x}px ${y}px` }}>
              {isActive && <motion.rect x={x - NW / 2 - 4} y={y - NH / 2 - 4} width={NW + 8} height={NH + 8} rx={8} fill="none" stroke="var(--accent)" strokeWidth={2} initial={{ opacity: 0.8 }} animate={{ opacity: [0.8, 0.15, 0.8] }} transition={{ duration: 1.4, repeat: Infinity }} />}
              <rect x={x - NW / 2} y={y - NH / 2} width={NW} height={NH} rx={6} fill="var(--surface)" stroke={isActive ? "var(--accent)" : "var(--line)"} strokeWidth={isActive ? 2 : 1} />
              <rect x={x - NW / 2} y={y - NH / 2} width={NW} height={NH} rx={6} fill={CLASS_COLOR[n.prediction]} fillOpacity={0.12} />
              {showsSplit ? (
                <motion.text key={`rule-${animKey}`} x={x} y={y - 9} fontSize={10} fontWeight={600} textAnchor="middle" fontFamily="var(--font-mono)" fill="var(--fg)" initial={{ opacity: isActive ? 0 : 1 }} animate={{ opacity: 1 }} transition={{ delay: ruleKnownAt * k }}>
                  {FEATURE[n.split!.feature]} ≤ {n.split!.threshold.toFixed(2)}
                </motion.text>
              ) : (
                <text x={x} y={y - 9} fontSize={10} fontWeight={600} textAnchor="middle" fill="var(--fg)">{finalLeaf ? `leaf → ${CLASS_NAME[n.prediction]}` : "not split yet"}</text>
              )}
              <text x={x} y={y + 4} fontSize={9.5} textAnchor="middle" fontFamily="var(--font-mono)" fill="var(--fg)">A {n.counts[0]} · B {n.counts[1]}</text>
              <text x={x} y={y + 16} fontSize={9} textAnchor="middle" fontFamily="var(--font-mono)" fill="var(--muted)">#{id} gini {n.gini.toFixed(2)}</text>
            </motion.g>
          );
        })}
      </svg>
    </div>
  );
}

// ---------- step card ----------
function StepCard({ sim, st, k, maxDepth }: { sim: TreeSim; st: TreeState; k: number; maxDepth: number }) {
  if (st.phase === "data") {
    const [a, b] = sim.nodes[0].counts;
    const pa = a / N_TRAIN, pbb = b / N_TRAIN;
    return (
      <div className="space-y-2">
        <h4 className="text-sm font-semibold">How mixed is the root node?</h4>
        <GrowBar label={`class A: ${a} points`} value={pa} max={1} color={CLASS_COLOR[0]} delay={0.3 * k} duration={0.6 * k} digits={2} />
        <GrowBar label={`class B: ${b} points`} value={pbb} max={1} color={CLASS_COLOR[1]} delay={0.5 * k} duration={0.6 * k} digits={2} />
        <GrowBar label="(share A)²" value={pa * pa} max={1} color="var(--muted)" delay={1 * k} duration={0.5 * k} />
        <GrowBar label="(share B)²" value={pbb * pbb} max={1} color="var(--muted)" delay={1.2 * k} duration={0.5 * k} />
        <GrowBar label="Gini = 1 − sum" value={sim.nodes[0].gini} max={0.5} color="var(--violet)" delay={1.6 * k} duration={0.6 * k} />
        <p className="text-xs text-muted">Gini is the chance that two points drawn at random from the node have different classes. 0 means pure; 0.5 is the worst for two classes. Every split tries to push it down.</p>
      </div>
    );
  }
  if (st.phase === "evaluate") {
    const best = sim.depthSweep.reduce((a, b) => (b.test > a.test ? b : a));
    return (
      <div className="grid gap-4 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="space-y-2">
          <h4 className="text-sm font-semibold">Same tree, two scores</h4>
          <GrowBar label="train accuracy" value={sim.trainAcc * 100} max={100} color="var(--ok)" delay={0.4 * k} duration={0.8 * k} digits={0} suffix="%" />
          <GrowBar label="test accuracy" value={sim.testAcc * 100} max={100} color="var(--warn)" delay={1.6 * k} duration={0.8 * k} digits={0} suffix="%" />
          <motion.p className="text-xs leading-relaxed" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.4 * k }}>
            {sim.trainAcc - sim.testAcc > 0.1
              ? `A ${(100 * (sim.trainAcc - sim.testAcc)).toFixed(0)}-point gap: the deep splits fenced off individual noisy points. Depth ${best.depth} gives the best test accuracy here.`
              : `Train and test are close, so this tree's rules carry over to new data. Try the "Break it" preset to see the gap open.`}
          </motion.p>
        </div>
        <div>
          <h4 className="mb-1 text-sm font-semibold">Accuracy vs. max depth (every depth regrown)</h4>
          <LineChart height={190} xLabel="max depth" yLabel="accuracy" markerX={maxDepth} series={[
            { name: "train", points: sim.depthSweep.map((d) => [d.depth, d.train]), color: "var(--ok)" },
            { name: "test", points: sim.depthSweep.map((d) => [d.depth, d.test]), color: "var(--warn)" },
          ]} />
        </div>
      </div>
    );
  }
  const n = sim.nodes[st.search!.nodeId], s = n.split!;
  const [bx, by] = st.search!.bestPerFeature;
  const row = (f: 0 | 1, c: Candidate | null, at: number) => (
    <motion.div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-md border border-line px-2 py-1.5 text-xs" initial={{ opacity: 0.25 }}
      animate={{ opacity: 1, borderColor: s.feature === f ? "var(--ok)" : "var(--line)" }} transition={{ opacity: { delay: at * k }, borderColor: { delay: T.lock * k } }}>
      <span>Best on <strong>{FEATURE[f]}</strong>: {c ? <span className="font-mono">{FEATURE[f]} ≤ {c.t.toFixed(2)} ({c.nL} | {c.nR})</span> : "no candidate (all values equal)"}</span>
      <span className="font-mono">{c ? <AnimatedNumber value={c.g} from={n.gini} delay={at * k} duration={0.4 * k} digits={3} /> : "—"}{s.feature === f && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.lock * k }}> ✓ winner</motion.span>}</span>
    </motion.div>
  );
  const child = (id: number, side: string) => {
    const c = sim.nodes[id];
    return (
      <motion.div className="rounded-md border border-line p-2 text-xs" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (T.grow + 0.2) * k, duration: 0.4 * k }}>
        <div className="font-semibold">{side}</div>
        <div className="font-mono">A {c.counts[0]} · B {c.counts[1]}</div>
        <div className="font-mono text-muted">gini {gini(c.counts).toFixed(3)}{c.depth >= maxDepth ? " · at max depth" : ""}</div>
      </motion.div>
    );
  };
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-sm font-semibold">Node #{n.id}: {n.idx.length} points (A {n.counts[0]} · B {n.counts[1]}), Gini {n.gini.toFixed(3)}</h4>
        <span className="text-[11px] text-muted">{st.search!.cands[0].length + st.search!.cands[1].length} thresholds tried</span>
      </div>
      {row(0, bx, T.xs + T.sweep)}
      {row(1, by, T.ys + T.sweep)}
      <div className="grid grid-cols-2 gap-2">
        {child(s.left, `yes: ${FEATURE[s.feature]} ≤ ${s.threshold.toFixed(2)}`)}
        {child(s.right, `no: ${FEATURE[s.feature]} > ${s.threshold.toFixed(2)}`)}
      </div>
      <GrowBar label="impurity removed" value={n.gini - s.g} max={0.5} color="var(--ok)" delay={(T.grow + 0.3) * k} duration={0.6 * k} />
    </div>
  );
}

function Calculation({ sim, st, node }: { sim: TreeSim; st: TreeState; node: TreeNode | null }) {
  if (st.phase === "data") {
    const [a, b] = sim.nodes[0].counts;
    return <TexBlock>{`\\begin{aligned} G &= 1 - \\sum_{c} p_c^2 \\\\ &= 1 - \\left(\\tfrac{${a}}{${N_TRAIN}}\\right)^2 - \\left(\\tfrac{${b}}{${N_TRAIN}}\\right)^2 \\\\ &= ${sim.nodes[0].gini.toFixed(3)} \\end{aligned}`}</TexBlock>;
  }
  if (st.phase === "evaluate") {
    return <TexBlock>{`\\begin{aligned} \\text{acc}_{\\text{train}} &= ${(sim.trainAcc * 100).toFixed(1)}\\% \\\\ \\text{acc}_{\\text{test}} &= ${(sim.testAcc * 100).toFixed(1)}\\% \\end{aligned}`}</TexBlock>;
  }
  const n = node!, s = n.split!, N = n.idx.length;
  return (
    <div className="space-y-2">
      <TexBlock>{`\\begin{aligned} G_{\\text{split}} &= \\tfrac{n_L}{n} G_L + \\tfrac{n_R}{n} G_R \\\\ &= \\tfrac{${s.nL}}{${N}} \\cdot ${s.giniL.toFixed(3)} + \\tfrac{${s.nR}}{${N}} \\cdot ${s.giniR.toFixed(3)} \\\\ &= ${s.g.toFixed(3)} \\\\ \\Delta G &= ${n.gini.toFixed(3)} - ${s.g.toFixed(3)} = ${(n.gini - s.g).toFixed(3)} \\end{aligned}`}</TexBlock>
      <p className="text-[11px] text-muted">Candidates are midpoints between neighbouring values, so a threshold never sits on a training point. The greedy rule picks the lowest <span className="font-mono">G_split</span> over both features.</p>
    </div>
  );
}
