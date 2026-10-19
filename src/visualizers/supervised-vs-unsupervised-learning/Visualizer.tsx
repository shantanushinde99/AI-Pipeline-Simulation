import { useCallback, useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, GrowBar, StepCaption } from "../decision-trees/StepCaption";
import { DATASETS, MAX_K, N, type DatasetId, type SuConfig, type SuPhase, type SuSim, type SuState, type Vec } from "./simulation";
import { simulate } from "./simulation";

const DEFAULT: SuConfig = { dataset: "aligned", k: 3, dataSeed: 7, initSeed: 2 };
const PRESETS: { label: string; cfg: Partial<SuConfig> }[] = [
  { label: "Clusters match classes", cfg: { dataset: "aligned", k: 3 } },
  { label: "Clusters ≠ classes", cfg: { dataset: "mismatch", k: 2 } },
  { label: "Break it: wrong k (5)", cfg: { dataset: "aligned", k: 5 } },
  { label: "Unlucky start (seed 6)", cfg: { dataset: "aligned", k: 3, initSeed: 6 } },
];
const CLASS_COLOR = ["var(--accent)", "var(--warn)", "var(--violet)"];
const CLUSTER_COLOR = ["#0d9488", "#c026d3", "#ca8a04", "#4f46e5", "#e11d48", "#65a30d"];
const GREY = "#9ca3af";
const ORDER: SuPhase[] = ["data", "sup-means", "sup-boundary", "km-init", "km-iter", "compare"];
/** Seconds at 1× for each phase's animation; auto-play waits this long plus a beat. */
const DURATION: Record<SuPhase, number> = { data: 2400, "sup-means": 3000, "sup-boundary": 3000, "km-init": 2600, "km-iter": 3300, compare: 3600 };
const T = { assign: 0.1, recolor: 0.45, move: 1.6, moveDur: 0.9 };

export default function SupervisedUnsupervisedVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT);
  const set = (p: Partial<SuConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useCallback((i: number) => DURATION[sim.steps[i]?.state.phase ?? "data"], [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const [replay, setReplay] = useState(0);
  const step = sim.steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const st = step.state;
  const at = (p: SuPhase) => ORDER.indexOf(st.phase) >= ORDER.indexOf(p);

  return (
    <Workspace
      main={
        <>
          <Panel title="Same data, two kinds of learning" actions={<span className="flex gap-1"><Badge tone="accent">classifier {Math.round(sim.sup.acc * 100)}%</Badge><Badge tone={sim.ct.purity >= 0.9 ? "ok" : "warn"}>purity {Math.round(sim.ct.purity * 100)}%</Badge></span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={st.headline} sub={st.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 sm:grid-cols-2">
              <Side title="Supervised: labels visible" active={!at("km-init")} badge={at("sup-boundary") ? "nearest class mean" : "training"}>
                <SupervisedSvg sim={sim} st={st} k={k} animKey={animKey} />
              </Side>
              <Side title="Unsupervised: labels hidden" active={at("km-init")} badge={`k-means, k = ${sim.cfg.k}`}>
                <KMeansSvg sim={sim} st={st} k={k} animKey={animKey} />
              </Side>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>shape + colour on the left = true class (● 1, ▲ 2, ■ 3)</span>
              <span>colour on the right = k-means cluster (C1, C2, …)</span>
              <span>★ = class mean · ✚ = cluster centroid</span>
              <span>red ring = point in the "wrong" region / cluster</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} st={st} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="Dataset" value={cfg.dataset} options={DATASETS.map((d) => ({ value: d.value, label: d.label }))} onChange={(v: DatasetId) => set({ dataset: v })} />
              <Slider label="k (number of clusters)" value={cfg.k} min={1} max={MAX_K} onChange={(v) => set({ k: v })} hint="k-means must be told k. The labels are never consulted." />
              <Slider label="Data seed" value={cfg.dataSeed} min={1} max={50} onChange={(v) => set({ dataSeed: v })} hint="Draws a new sample of points." />
              <Slider label="k-means init seed" value={cfg.initSeed} min={1} max={50} onChange={(v) => set({ initSeed: v })} hint="Which points the centroids start on. Bad starts can get stuck." />
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
            why="Supervised learning optimises for the labels you supply. Unsupervised learning finds whatever structure distance reveals, which may or may not be the structure you care about." />
          <Panel title="Calculation"><Calculation sim={sim} st={st} /></Panel>
          <Panel title="Outputs">
            <KV rows={[
              ["classifier training accuracy", `${(sim.sup.acc * 100).toFixed(1)}%`],
              ["k-means iterations", `${sim.km.iters.length}${sim.km.converged ? " (converged)" : " (cap)"}`],
              ["final inertia J", sim.km.iters.at(-1)!.inertia.toFixed(2)],
              ["cluster purity", `${(sim.ct.purity * 100).toFixed(1)}%`],
              ["classes / clusters", `${sim.classes} / ${sim.cfg.k}`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Side({ title, active, badge, children }: { title: string; active: boolean; badge: string; children: ReactNode }) {
  return (
    <div className={cx("rounded-md border p-2 transition-colors duration-500", active ? "border-accent/60 bg-accent-soft/30" : "border-line")}>
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className={cx("text-xs font-semibold", !active && "text-muted")}>{title}</span>
        <Badge tone={active ? "accent" : "neutral"}>{badge}</Badge>
      </div>
      {children}
    </div>
  );
}

const W = 300, PAD = 12;
const s = (v: number) => PAD + (v / 10) * (W - 2 * PAD);
const sy = (v: number) => W - PAD - (v / 10) * (W - 2 * PAD);

function Shape({ label, color, r = 4.5, hollow }: { label: number; color: string; r?: number; hollow?: boolean }) {
  const fill = hollow ? "var(--surface)" : color;
  if (label === 1) return <path d={`M0 ${-r - 1} L${r} ${r - 0.5} L${-r} ${r - 0.5} Z`} fill={fill} stroke={color} strokeWidth={1.5} />;
  if (label === 2) return <rect x={-r + 0.5} y={-r + 0.5} width={2 * r - 1} height={2 * r - 1} fill={fill} stroke={color} strokeWidth={1.5} />;
  return <circle r={r} fill={fill} stroke={color} strokeWidth={1.5} />;
}

function Star({ at, color, delay, k, label }: { at: Vec; color: string; delay: number; k: number; label: string }) {
  return (
    <motion.g initial={{ opacity: 0, scale: 0, x: s(at[0]), y: sy(at[1]) }} animate={{ opacity: 1, scale: 1, x: s(at[0]), y: sy(at[1]) }} transition={{ delay: delay * k, duration: 0.45 * k, ease: EASE }}>
      <path d="M0 -10 L2.9 -3.1 L9.5 -3.1 L4.2 1.2 L6.2 8 L0 4 L-6.2 8 L-4.2 1.2 L-9.5 -3.1 L-2.9 -3.1 Z" fill={color} stroke="var(--surface)" strokeWidth={1.5} />
      <text y={20} fontSize={9} textAnchor="middle" fill="var(--fg)" fontWeight={600} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">{label}</text>
    </motion.g>
  );
}

function SupervisedSvg({ sim, st, k, animKey }: { sim: SuSim; st: SuState; k: number; animKey: string }) {
  const showBoundary = ORDER.indexOf(st.phase) >= ORDER.indexOf("sup-boundary");
  const showMeans = st.phase !== "data";
  const G = 30;
  const cells = useMemo(() => Array.from({ length: G * G }, (_, i) => {
    const cx0 = (i % G) * (10 / G), cy0 = Math.floor(i / G) * (10 / G);
    return { cx0, cy0, c: sim.sup.predict([cx0 + 5 / G, cy0 + 5 / G]) };
  }), [sim]);
  return (
    <svg viewBox={`0 0 ${W} ${W}`} className="h-auto w-full" role="img" aria-label="Labelled points with a nearest-class-mean decision boundary">
      <rect x={PAD} y={PAD} width={W - 2 * PAD} height={W - 2 * PAD} fill="none" stroke="var(--line)" />
      {showBoundary && (
        <g>
          <clipPath id={`sup-clip-${animKey}`}>
            <motion.rect x={PAD} y={PAD} height={W - 2 * PAD} initial={{ width: st.phase === "sup-boundary" ? 0 : W }} animate={{ width: W }} transition={{ delay: 0.2 * k, duration: 1.4 * k, ease: "linear" }} />
          </clipPath>
          <g clipPath={`url(#sup-clip-${animKey})`} opacity={0.16}>
            {cells.map((c, i) => <rect key={i} x={s(c.cx0)} y={sy(c.cy0 + 10 / G)} width={(W - 2 * PAD) / G + 0.3} height={(W - 2 * PAD) / G + 0.3} fill={CLASS_COLOR[c.c]} />)}
          </g>
        </g>
      )}
      {st.phase === "sup-means" && sim.pts.map((p, i) => {
        const m = sim.sup.centroids[p.label];
        return <motion.line key={`l-${i}-${animKey}`} x1={s(p.x[0])} y1={sy(p.x[1])} x2={s(m[0])} y2={sy(m[1])} stroke={CLASS_COLOR[p.label]} strokeWidth={1}
          initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 0.7, 0.7, 0] }} transition={{ delay: (0.2 + i * 0.012) * k, duration: 1.6 * k, opacity: { delay: (0.2 + i * 0.012) * k, duration: 1.8 * k, times: [0, 0.2, 0.7, 1] } }} />;
      })}
      {sim.pts.map((p, i) => {
        const wrong = showBoundary && sim.sup.predict(p.x) !== p.label;
        return (
          <motion.g key={`p-${i}-${st.phase === "data" ? animKey : ""}`} initial={{ opacity: 0, scale: 0, x: s(p.x[0]), y: sy(p.x[1]) }} animate={{ opacity: 1, scale: 1, x: s(p.x[0]), y: sy(p.x[1]) }} transition={{ delay: (st.phase === "data" ? 0.1 + i * 0.018 : 0) * k, duration: 0.3 * k }}>
            <Shape label={p.label} color={CLASS_COLOR[p.label]} />
            {wrong && <motion.circle r={8} fill="none" stroke="var(--err)" strokeWidth={1.8} initial={{ opacity: st.phase === "sup-boundary" ? 0 : 1 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 * k }} />}
          </motion.g>
        );
      })}
      {showMeans && sim.sup.centroids.map((c, j) => <Star key={`m-${j}-${st.phase === "sup-means" ? animKey : ""}`} at={c} color={CLASS_COLOR[j]} delay={st.phase === "sup-means" ? 1.3 + j * 0.15 : 0} k={k} label={`mean ${j + 1}`} />)}
    </svg>
  );
}

function Cross({ x, color, label }: { x: Vec; color: string; label: string }) {
  return (
    <>
      <circle r={9} fill="var(--surface)" stroke={color} strokeWidth={2.5} />
      <path d="M-5 0 H5 M0 -5 V5" stroke={color} strokeWidth={2.5} />
      <text y={-13} fontSize={9.5} fontWeight={700} textAnchor="middle" fill="var(--fg)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">{label}</text>
      <title>{`${label} at (${x[0].toFixed(2)}, ${x[1].toFixed(2)})`}</title>
    </>
  );
}

function KMeansSvg({ sim, st, k, animKey }: { sim: SuSim; st: SuState; k: number; animKey: string }) {
  const { km, pts, ct } = sim;
  const it = st.phase === "km-iter" ? km.iters[st.iter!] : null;
  const last = km.iters.at(-1)!;
  const prevAssign = st.phase === "km-iter" && st.iter! > 0 ? km.iters[st.iter! - 1].assign : null;
  const assign = it ? it.assign : st.phase === "compare" ? last.assign : null;
  const centersFrom: Vec[] | null = it ? it.before : st.phase === "compare" ? km.final : st.phase === "km-init" ? km.initIdx.map((i) => pts[i].x) : null;
  const centersTo: Vec[] | null = it ? it.after : centersFrom;
  return (
    <svg viewBox={`0 0 ${W} ${W}`} className="h-auto w-full" role="img" aria-label="Unlabelled points being clustered by k-means">
      <rect x={PAD} y={PAD} width={W - 2 * PAD} height={W - 2 * PAD} fill="none" stroke="var(--line)" />
      {it && pts.map((p, i) => {
        const c = it.before[it.assign[i]];
        return <motion.line key={`al-${i}-${animKey}`} x1={s(p.x[0])} y1={sy(p.x[1])} x2={s(c[0])} y2={sy(c[1])} stroke={CLUSTER_COLOR[it.assign[i]]} strokeWidth={1}
          initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 0.65, 0.65, 0] }} transition={{ delay: (T.assign + i * 0.008) * k, duration: 1.2 * k, opacity: { delay: (T.assign + i * 0.008) * k, duration: 1.4 * k, times: [0, 0.2, 0.75, 1] } }} />;
      })}
      {pts.map((p, i) => {
        const from = prevAssign ? CLUSTER_COLOR[prevAssign[i]] : GREY;
        const to = assign ? CLUSTER_COLOR[assign[i]] : GREY;
        const switched = it && (!prevAssign || prevAssign[i] !== it.assign[i]);
        const reveal = st.phase === "compare";
        const wrong = reveal && ct.majority[last.assign[i]] !== p.label;
        return (
          <motion.g key={`kp-${i}-${it || reveal ? animKey : ""}`} initial={{ x: s(p.x[0]), y: sy(p.x[1]), scale: 1 }} animate={{ x: s(p.x[0]), y: sy(p.x[1]), scale: switched ? [1, 1.7, 1] : 1 }}
            transition={{ delay: (T.recolor + i * 0.012) * k, duration: 0.45 * k }}>
            {reveal ? (
              <motion.g initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: (0.3 + i * 0.01) * k, duration: 0.35 * k }}>
                <Shape label={p.label} color={to} />
                {wrong && <motion.circle r={8.5} fill="none" stroke="var(--err)" strokeWidth={1.8} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 * k }} />}
              </motion.g>
            ) : (
              <motion.circle r={4.5} stroke="var(--surface)" strokeWidth={1} initial={{ fill: from }} animate={{ fill: to }} transition={{ delay: (T.recolor + i * 0.012) * k, duration: 0.25 * k }} />
            )}
          </motion.g>
        );
      })}
      {centersFrom && centersTo && centersFrom.map((c, j) => {
        const to = centersTo[j];
        const dropping = st.phase === "km-init";
        return (
          <g key={`c-${j}-${animKey}`}>
            {it && <motion.line x1={s(c[0])} y1={sy(c[1])} x2={s(to[0])} y2={sy(to[1])} stroke={CLUSTER_COLOR[j]} strokeWidth={2} strokeDasharray="3 3"
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: T.move * k, duration: T.moveDur * k, ease: EASE }} />}
            <motion.g initial={dropping ? { x: s(c[0]), y: sy(c[1]) - 40, opacity: 0, scale: 0.4 } : { x: s(c[0]), y: sy(c[1]) }}
              animate={{ x: s(to[0]), y: sy(to[1]), opacity: 1, scale: 1 }}
              transition={{ delay: (dropping ? 0.3 + j * 0.25 : T.move) * k, duration: (dropping ? 0.5 : T.moveDur) * k, ease: EASE }}>
              <Cross x={to} color={CLUSTER_COLOR[j]} label={`C${j + 1}`} />
            </motion.g>
          </g>
        );
      })}
      {(st.phase === "data" || st.phase === "sup-means" || st.phase === "sup-boundary") && (
        <text x={W / 2} y={W - 18} fontSize={10} textAnchor="middle" fill="var(--muted)">no labels: every point looks the same</text>
      )}
    </svg>
  );
}

function StepCard({ sim, st, k }: { sim: SuSim; st: SuState; k: number }) {
  if (st.phase === "data") {
    const rows = sim.pts.slice(0, 5);
    const table = (withLabel: boolean, delay: number) => (
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-0.5">x₁</th><th>x₂</th><th>label</th></tr></thead>
        <tbody className="font-mono">
          {rows.map((p, i) => (
            <motion.tr key={i} className="border-t border-line" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (delay + i * 0.12) * k, duration: 0.3 * k }}>
              <td className="py-0.5">{p.x[0].toFixed(2)}</td><td>{p.x[1].toFixed(2)}</td>
              <td>{withLabel ? <span style={{ color: CLASS_COLOR[p.label] }}>class {p.label + 1}</span> : <span className="text-muted line-through">hidden</span>}</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    );
    return (
      <div className="grid gap-4 sm:grid-cols-2">
        <div><h4 className="mb-1 text-sm font-semibold">Supervised input: features + label</h4>{table(true, 0.2)}</div>
        <div><h4 className="mb-1 text-sm font-semibold">Unsupervised input: features only</h4>{table(false, 0.8)}</div>
        <p className="text-xs text-muted sm:col-span-2">Showing 5 of {N} rows. The coordinates are identical; only the label column differs.</p>
      </div>
    );
  }
  if (st.phase === "sup-means") {
    return (
      <div className="space-y-1.5">
        <h4 className="text-sm font-semibold">Training = averaging each class</h4>
        {sim.sup.centroids.map((c, j) => (
          <motion.div key={j} className="grid grid-cols-[6rem_1fr_1fr] items-center gap-2 rounded-md border border-line px-2 py-1 font-mono text-xs" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + j * 0.3) * k }}>
            <span style={{ color: CLASS_COLOR[j] }}>class {j + 1} ({sim.sup.counts[j]})</span>
            <span>x̄₁ = <AnimatedNumber value={c[0]} from={0} delay={(0.4 + j * 0.3) * k} duration={0.9 * k} /></span>
            <span>x̄₂ = <AnimatedNumber value={c[1]} from={0} delay={(0.4 + j * 0.3) * k} duration={0.9 * k} /></span>
          </motion.div>
        ))}
        <p className="text-xs text-muted">The labels did the hard work: we already know which points belong together, so the model only has to summarise them.</p>
      </div>
    );
  }
  if (st.phase === "sup-boundary") {
    return (
      <div className="space-y-2">
        <h4 className="text-sm font-semibold">Predict = closest class mean</h4>
        <GrowBar label="training accuracy" value={sim.sup.acc * 100} max={100} color="var(--accent)" delay={0.4 * k} duration={1.2 * k} digits={0} suffix="%" />
        <p className="text-xs text-muted">The boundary between two shaded regions is the perpendicular bisector of the two means. Red rings mark labelled points that fall on the wrong side.</p>
      </div>
    );
  }
  if (st.phase === "km-init") {
    return (
      <div className="space-y-1.5">
        <h4 className="text-sm font-semibold">Random start: {sim.cfg.k} points become the first centroids</h4>
        <div className="flex flex-wrap gap-2">
          {sim.km.initIdx.map((i, j) => (
            <motion.span key={j} className="rounded-md border px-2 py-1 font-mono text-xs" style={{ borderColor: CLUSTER_COLOR[j] }} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + j * 0.25) * k }}>
              C{j + 1} = point #{i} ({sim.pts[i].x[0].toFixed(1)}, {sim.pts[i].x[1].toFixed(1)})
            </motion.span>
          ))}
        </div>
        <p className="text-xs text-muted">k-means then repeats two moves until nothing changes: assign each point to its nearest centroid, move each centroid to the mean of its points.</p>
      </div>
    );
  }
  if (st.phase === "km-iter") {
    const i = st.iter!, it = sim.km.iters[i];
    const maxJ = Math.max(...sim.km.iters.map((v) => v.inertia));
    return (
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <h4 className="text-sm font-semibold">Inertia J after each iteration (lower = tighter clusters)</h4>
          {sim.km.iters.slice(0, i + 1).map((v, j) => (
            <GrowBar key={j} label={`iteration ${j + 1}`} value={v.inertia} max={maxJ} color={j === i ? "var(--accent)" : "var(--muted)"} delay={j === i ? (T.move + 0.4) * k : 0} duration={j === i ? 0.7 * k : 0.01} digits={1} />
          ))}
        </div>
        <div className="space-y-1">
          <h4 className="text-sm font-semibold">This iteration</h4>
          <p className="text-xs">Points {i ? "switching cluster" : "assigned"}: <span className="font-mono text-base font-semibold"><AnimatedNumber value={it.changed} from={0} digits={0} delay={T.recolor * k} duration={0.8 * k} /></span></p>
          {it.moved.map((m, j) => (
            <motion.p key={j} className="font-mono text-xs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.move * k }}>
              <span style={{ color: CLUSTER_COLOR[j] }}>C{j + 1}</span> moved <AnimatedNumber value={m} from={0} delay={T.move * k} duration={T.moveDur * k} /> {m < 1e-9 && "(stayed)"}
            </motion.p>
          ))}
          {i === sim.km.iters.length - 1 && <motion.p className="text-xs text-ok" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.6 * k }}>✓ {sim.km.converged ? "Next assignment would change nothing: converged." : "Iteration cap reached."}</motion.p>}
        </div>
      </div>
    );
  }
  const { table, purity, majority } = sim.ct;
  return (
    <div className="grid gap-4 sm:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div className="overflow-x-auto">
        <h4 className="mb-1 text-sm font-semibold">Which classes ended up in which cluster?</h4>
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-0.5">cluster</th>{Array.from({ length: sim.classes }, (_, c) => <th key={c} style={{ color: CLASS_COLOR[c] }}>class {c + 1}</th>)}<th>majority</th></tr></thead>
          <tbody className="font-mono">
            {table.map((row, j) => (
              <tr key={j} className="border-t border-line">
                <td className="py-0.5" style={{ color: CLUSTER_COLOR[j] }}>C{j + 1}</td>
                {row.map((v, c) => <td key={c} className={cx(c === majority[j] && v > 0 && "font-semibold")}><AnimatedNumber value={v} from={0} digits={0} delay={(0.4 + j * 0.25) * k} duration={0.6 * k} /></td>)}
                <td>{row.some((v) => v > 0) ? `class ${majority[j] + 1}` : "empty"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-2">
        <GrowBar label="purity" value={purity * 100} max={100} color={purity >= 0.9 ? "var(--ok)" : "var(--warn)"} delay={1.4 * k} duration={0.9 * k} digits={0} suffix="%" />
        <GrowBar label="classifier accuracy" value={sim.sup.acc * 100} max={100} color="var(--accent)" delay={1.6 * k} duration={0.9 * k} digits={0} suffix="%" />
        <motion.p className="text-xs leading-relaxed" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.4 * k }}>
          {sim.cfg.k !== sim.classes
            ? `k = ${sim.cfg.k} but there are ${sim.classes} classes, so clusters must split or merge classes. Choosing k is a modelling decision, not something the data hands you.`
            : purity >= 0.9 ? "The natural groups happen to be the classes, so clustering recovers them without seeing a single label." : "Distance found a different grouping than the labels encode. Without labels, k-means has no way to know which grouping you wanted."}
        </motion.p>
      </div>
    </div>
  );
}

function Calculation({ sim, st }: { sim: SuSim; st: SuState }) {
  const f = (v: number) => v.toFixed(2);
  if (st.phase === "data" || st.phase === "sup-means" || st.phase === "sup-boundary") {
    const c = sim.sup.centroids;
    return <TexBlock>{`\\begin{aligned} \\mu_c &= \\tfrac{1}{n_c} \\textstyle\\sum_{i:\\,y_i = c} x_i \\\\ \\mu_1 &= (${f(c[0][0])},\\ ${f(c[0][1])}) \\\\ \\hat{y}(x) &= \\arg\\min_c \\lVert x - \\mu_c \\rVert \\\\ \\text{acc} &= ${(sim.sup.acc * 100).toFixed(1)}\\% \\end{aligned}`}</TexBlock>;
  }
  if (st.phase === "km-iter") {
    const it = sim.km.iters[st.iter!];
    const n1 = it.assign.filter((a) => a === 0).length;
    return <TexBlock>{`\\begin{aligned} c_i &= \\arg\\min_j \\lVert x_i - \\mu_j \\rVert^2 \\\\ \\mu_j &\\leftarrow \\tfrac{1}{|S_j|} \\textstyle\\sum_{i \\in S_j} x_i \\\\ \\mu_1 &= \\tfrac{1}{${n1}}\\textstyle\\sum = (${f(it.after[0][0])},\\ ${f(it.after[0][1])}) \\\\ J &= \\textstyle\\sum_i \\lVert x_i - \\mu_{c_i} \\rVert^2 = ${it.inertia.toFixed(2)} \\end{aligned}`}</TexBlock>;
  }
  if (st.phase === "compare") {
    return <TexBlock>{`\\text{purity} = \\frac{1}{n} \\sum_{j} \\max_c\\, |S_j \\cap C_c| = \\frac{${Math.round(sim.ct.purity * N)}}{${N}} = ${(sim.ct.purity * 100).toFixed(1)}\\%`}</TexBlock>;
  }
  return <TexBlock>{`\\mu_j^{(0)} = x_{\\text{random}_j}, \\quad j = 1, \\dots, ${sim.cfg.k}`}</TexBlock>;
}
