import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";
import { animate, motion, useMotionValue, useTransform } from "motion/react";
import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../document-ingestion/kit";
import { DEFAULT_VDB, INDEX_INFO, QUERIES, dist, simulate, tradeoff, voronoiCell, type IndexType, type Pt, type VdbParams, type VdbRun, type VdbState } from "./simulation";

// ---------- one schedule (seconds at 1×) ----------
const SCAN = 2.4;
const TRAIN = 3.2;
const centroidAt = (i: number) => 0.3 + i * 0.15;
const hopWindow = (hops: number) => Math.min(4, Math.max(1, hops * 0.5));
const hopAt = (i: number, hops: number) => 0.4 + (i * hopWindow(hops)) / Math.max(1, hops);

const S = 460, px = (p: Pt) => 10 + p[0] * 4.4, py = (p: Pt) => 450 - p[1] * 4.4;
const cellColor = (j: number, n: number) => `hsl(${Math.round((j * 360) / n)} 62% 52%)`;

/** A count-up that runs linearly (so it stays in step with a sweep), unlike the eased AnimatedNumber. */
function Counter({ from, to, delay, duration }: { from: number; to: number; delay: number; duration: number }) {
  const mv = useMotionValue(from);
  const text = useTransform(mv, (v) => Math.round(v).toString());
  useEffect(() => {
    mv.set(from);
    const c = animate(mv, to, { delay, duration, ease: "linear" });
    return () => c.stop();
  }, [mv, from, to, delay, duration]);
  return <motion.span>{text}</motion.span>;
}

/** Distance computations already spent before this phase starts, and how many this phase adds. */
function costWindow(run: VdbRun, s: VdbState): { from: number; to: number } {
  const { p, iv, hs, pts } = run;
  if (p.index === "flat") return s.phase === "scan" ? { from: 0, to: pts.length } : s.phase === "data" ? { from: 0, to: 0 } : { from: pts.length, to: pts.length };
  if (p.index === "ivf") {
    if (s.phase === "probe") return { from: 0, to: p.nlist };
    if (s.phase === "scan-cells") return { from: p.nlist, to: iv.computations };
    return ["topk", "compare", "insert"].includes(s.phase) ? { from: iv.computations, to: iv.computations } : { from: 0, to: 0 };
  }
  if (s.phase === "layer") {
    const before = 1 + hs.layers.filter((l) => l.layer > s.layer!).reduce((a, l) => a + l.evaluated.length, 0);
    return { from: before, to: before + hs.layers.find((l) => l.layer === s.layer)!.evaluated.length };
  }
  return ["topk", "compare", "insert"].includes(s.phase) ? { from: hs.computations, to: hs.computations } : { from: 0, to: 0 };
}

function phaseSeconds(run: VdbRun, s: VdbState): number {
  switch (s.phase) {
    case "data": return 2.2;
    case "scan": return SCAN + 1.2;
    case "train": return TRAIN + 1.2;
    case "assign": return 2.8;
    case "probe": return centroidAt(run.p.nlist) + 1.4;
    case "scan-cells": return SCAN + 1;
    case "build": return 3;
    case "layer": { const l = run.hs.layers.find((x) => x.layer === s.layer)!; return 0.4 + hopWindow(l.hops.length) + 1.2; }
    case "compare": return 3.8;
    case "insert": return 3.6;
    default: return 2.8;
  }
}

function narrate(run: VdbRun, s: VdbState) {
  const { p, iv, hs, truth, pts, insert } = run;
  const res = p.index === "flat" ? truth : p.index === "ivf" ? iv : hs;
  switch (s.phase) {
    case "data": return { headline: `${pts.length} stored vectors, one query, find the ${p.k} nearest`, sub: INDEX_INFO[p.index].short };
    case "scan": return { headline: `Brute force: ${pts.length} distances, one after another`, sub: "Exact by construction, but the cost grows linearly with the number of vectors." };
    case "train": return { headline: `k-means slides ${p.nlist} centroids into the clusters`, sub: "This happens once, offline, when the index is built." };
    case "assign": return { headline: `Every vector is filed under its nearest of ${p.nlist} centroids`, sub: "Each coloured region is a cell: the part of space closest to one centroid." };
    case "probe": return { headline: `Query vs. ${p.nlist} centroids → probe the ${p.nprobe} closest cell${p.nprobe > 1 ? "s" : ""}`, sub: "Only centroids are compared here: a handful of distances instead of hundreds." };
    case "scan-cells": return { headline: `Scan ${iv.scanned.length} vectors in the probed cells, skip ${pts.length - iv.scanned.length}`, sub: "A true neighbour sitting just across a cell border is never even looked at." };
    case "build": return { headline: `A ${hs.layers.length}-layer graph: sparse highways on top, every vector at the bottom`, sub: `Each node links to its M = ${p.m} nearest earlier nodes on every layer it belongs to.` };
    case "layer": {
      const l = hs.layers.find((x) => x.layer === s.layer)!;
      return s.layer! > 0
        ? { headline: l.hops.length ? `Layer ${s.layer}: ${l.hops.length} greedy hop${l.hops.length > 1 ? "s" : ""} toward the query` : `Layer ${s.layer}: no neighbour is closer, drop down a layer`, sub: "Long links on sparse layers cover big distances in a few hops." }
        : { headline: `Layer 0: best-first search keeping the ef = ${p.ef} closest found`, sub: "Expand the most promising candidate, score its neighbours, repeat until nothing can improve the list." };
    }
    case "topk": return { headline: `Recall@${p.k} = ${res.recall.toFixed(2)} after ${res.computations} distance computations`, sub: res.recall < 1 ? `${Math.round((1 - res.recall) * p.k)} true neighbour(s) missed (red dashed ring). Raise ${p.index === "ivf" ? "nprobe" : "ef"} to trade cost for recall.` : p.index === "flat" ? "Flat search is the ground truth the others are measured against." : "All true neighbours found, at a fraction of brute-force cost." };
    case "compare": return { headline: "Recall vs. distance computations for this query", sub: "Each dot is one setting. Up and to the left is better." };
    case "insert": return { headline: p.index === "flat" ? "Insert = append one row" : p.index === "ivf" ? `Insert: nearest centroid is cell ${insert.ivfCell}, append to its list` : `Insert: join layers 0–${insert.level} and link to M = ${p.m} neighbours on each`, sub: p.index === "ivf" ? "Centroids are not retrained; after many inserts the cells drift out of shape." : p.index === "hnsw" ? "Links go both ways, so existing nodes can reach the new one immediately." : "No structure to maintain, but every query now scans one more vector." };
  }
}

export default function VectorDatabasesVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<VdbParams>(DEFAULT_VDB);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(params), [params]);
  const { run } = sim;
  const duration = useCallback((i: number) => phaseSeconds(run, sim.steps[i]?.state ?? { phase: "data" }) * 1000, [run, sim.steps]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${JSON.stringify(run.p)}-${step.id}-${replay}`;
  const story = narrate(run, step.state);
  const set = (p: Partial<VdbParams>) => setParams((v) => ({ ...v, ...p }));
  const res = run.p.index === "flat" ? run.truth : run.p.index === "ivf" ? run.iv : run.hs;
  const cost = costWindow(run, step.state);
  const curves = useMemo(() => tradeoff(run), [run]);

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title={`Index: ${INDEX_INFO[run.p.index].name}`} actions={<span className="text-[11px] text-muted">Click the plane to move the query</span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={`${run.p.index}-${step.id}`} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_170px]">
              <Plane run={run} state={step.state} k={k} animKey={animKey} onPlace={(q) => set({ query: q })} />
              <div className="flex flex-row gap-3 md:flex-col">
                <div className="flex-1 rounded-md border border-line bg-surface-2/60 p-2.5">
                  <div className="text-[11px] uppercase text-muted">distance computations</div>
                  <div className="font-mono text-3xl font-semibold" data-testid="vdb-cost"><Counter key={animKey} from={cost.from} to={cost.to} delay={(step.state.phase === "probe" ? 0.3 : 0.4) * k} duration={(step.state.phase === "probe" ? centroidAt(run.p.nlist) - 0.3 : step.state.phase === "layer" ? hopWindow(run.hs.layers.find((l) => l.layer === step.state.layer)!.hops.length) : SCAN) * k} /></div>
                  <div className="mt-1 h-2 rounded bg-surface"><motion.div className="h-full rounded bg-accent" initial={{ width: `${(cost.from / run.pts.length) * 100}%` }} animate={{ width: `${Math.min(100, (cost.to / run.pts.length) * 100)}%` }} transition={{ delay: 0.4 * k, duration: SCAN * k, ease: "linear" }} /></div>
                  <div className="mt-1 text-[10px] text-muted">bar = share of a full scan ({run.pts.length})</div>
                </div>
                <div className="flex-1 rounded-md border border-line bg-surface-2/60 p-2.5">
                  <div className="text-[11px] uppercase text-muted">recall@{run.p.k}</div>
                  <div className={cx("font-mono text-3xl font-semibold", ["topk", "compare", "insert"].includes(step.state.phase) ? (res.recall < 1 ? "text-warn" : "text-ok") : "text-muted")}>
                    {["topk", "compare", "insert"].includes(step.state.phase) ? <AnimatedNumber key={animKey} value={res.recall} from={0} delay={0.6 * k} duration={0.8 * k} /> : "?"}
                  </div>
                  <div className="text-[10px] text-muted">vs. exact flat search</div>
                </div>
              </div>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
              <StageDetail run={run} state={step.state} k={k} curves={curves} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Index settings" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_VDB); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="mb-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Query:</span>
              {QUERIES.map((q) => <Button key={q.id} size="sm" variant={q.q[0] === params.query[0] && q.q[1] === params.query[1] ? "primary" : "secondary"} onClick={() => set({ query: q.q })}>{q.label}</Button>)}
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Index type" value={params.index} onChange={(v: IndexType) => set({ index: v })} options={(Object.keys(INDEX_INFO) as IndexType[]).map((i) => ({ value: i, label: INDEX_INFO[i].name }))} />
              <Slider label="k (neighbours wanted)" value={params.k} min={1} max={10} onChange={(v) => set({ k: v, ef: Math.max(v, params.ef) })} />
              {params.index === "ivf" && <>
                <Slider label="nlist (cells)" value={params.nlist} min={2} max={16} onChange={(v) => set({ nlist: v, nprobe: Math.min(v, params.nprobe) })} />
                <Slider label="nprobe (cells scanned)" value={run.p.nprobe} min={1} max={run.p.nlist} onChange={(v) => set({ nprobe: v })} hint="1 is fastest and most likely to miss neighbours across a border." />
              </>}
              {params.index === "hnsw" && <>
                <Slider label="M (links per node per layer)" value={params.m} min={2} max={10} onChange={(v) => set({ m: v })} />
                <Slider label="ef (search beam width)" value={run.p.ef} min={run.p.k} max={64} onChange={(v) => set({ ef: v })} hint="ef = k is the cheapest and can get stuck in a local neighbourhood." />
              </>}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why="Every millisecond of retrieval latency is mostly distance computations. Approximate indexes buy speed with a little recall, and you choose the exchange rate." />
          <Panel title="Calculation" actions={<SimulatedLabel>2-D teaching vectors</SimulatedLabel>}><Calculation run={run} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["vectors (N)", run.pts.length], ["query", `(${run.p.query[0]}, ${run.p.query[1]})`], ["returned ids", res.ids.join(", ")], ["exact ids", run.truth.ids.join(", ")],
              ["distance computations", res.computations], ["share of full scan", `${((res.computations / run.pts.length) * 100).toFixed(0)}%`], [`recall@${run.p.k}`, res.recall.toFixed(2)],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Plane({ run, state, k, animKey, onPlace }: { run: VdbRun; state: VdbState; k: number; animKey: string; onPlace: (q: Pt) => void }) {
  const { p, pts, ivf, iv, h, hs, truth, insert } = run;
  const ph = state.phase;
  const res = p.index === "flat" ? truth : p.index === "ivf" ? iv : hs;
  const showCells = p.index === "ivf" && ph !== "data" && ph !== "train";
  const cells = useMemo(() => ivf.centroids.map((_, j) => voronoiCell(ivf.centroids, j)), [ivf]);
  const after = ["topk", "compare", "insert"].includes(ph);
  const probing = ph === "probe" || ph === "scan-cells" || (p.index === "ivf" && after);
  const scannedSet = new Set(p.index === "ivf" ? iv.scanned : []);
  const layer = ph === "layer" ? state.layer! : ph === "build" ? -1 : null;
  const trace = layer !== null && layer >= 0 ? hs.layers.find((l) => l.layer === layer)! : null;
  const evaluatedBefore = new Set(p.index === "hnsw" && (ph === "layer" || after) ? hs.layers.filter((l) => layer === null || l.layer > layer).flatMap((l) => l.evaluated) : []);
  const click = (e: MouseEvent<SVGSVGElement>) => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(e.currentTarget.getScreenCTM()!.inverse());
    const q: Pt = [Math.round((pt.x - 10) / 4.4), Math.round((450 - pt.y) / 4.4)];
    if (q[0] >= 0 && q[0] <= 100 && q[1] >= 0 && q[1] <= 100) onPlace(q);
  };
  const kth = dist(p.query, pts[truth.ids[truth.ids.length - 1]]);
  const edgeLayers = p.index === "hnsw" && (ph === "build" || ph === "layer" || ph === "insert") ? (ph === "layer" ? [layer!] : [0, 1, 2].filter((l) => l <= h.maxLevel)) : [];
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto h-auto w-full max-w-[520px] cursor-crosshair touch-manipulation rounded-md border border-line bg-surface" onClick={click} role="img" aria-label={`Vector plane with ${pts.length} points and a query`}>
      {/* IVF cells */}
      {showCells && cells.map((poly, j) => {
        const probed = iv.probed.includes(j);
        return (
          <motion.polygon key={`${animKey}-cell${j}`} points={poly.map((q) => `${px(q)},${py(q)}`).join(" ")} fill={cellColor(j, p.nlist)} stroke="var(--surface)" strokeWidth={2}
            initial={{ fillOpacity: ph === "assign" ? 0 : probing ? (probed ? 0.22 : 0.05) : 0.12 }}
            animate={{ fillOpacity: probing ? (probed ? 0.22 : 0.04) : 0.12 }} transition={{ delay: ph === "probe" ? centroidAt(p.nlist) * k : 0.2 * k, duration: 0.5 * k }} />
        );
      })}
      {/* HNSW edges */}
      {edgeLayers.map((l) => (
        <motion.g key={`${animKey}-edges${l}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: ph === "build" ? (0.3 + (h.maxLevel - l) * 0.7) * k : 0, duration: 0.6 * k }}>
          {[...h.adj[l]].flatMap(([i, nbs]) => nbs.filter((j) => j > i).map((j) => (
            <line key={`${i}-${j}`} x1={px(pts[i])} y1={py(pts[i])} x2={px(pts[j])} y2={py(pts[j])} stroke={l === 2 ? "var(--violet)" : l === 1 ? "var(--accent)" : "var(--line)"} strokeWidth={l === 0 ? 0.8 : l === 1 ? 1.2 : 2} strokeOpacity={l === 0 ? 0.9 : 0.6} />
          )))}
        </motion.g>
      ))}
      {/* Points */}
      {pts.map((pt, i) => {
        const inLayer = layer === null || layer < 0 || h.levels[i] >= layer;
        const r = p.index === "hnsw" && (ph === "build" || ph === "layer") ? 2 + h.levels[i] * 1.6 : 2.6;
        const fill = p.index === "ivf" && ph !== "data" && ph !== "train" ? cellColor(ivf.assign[i], p.nlist) : "var(--muted)";
        const dim = (p.index === "ivf" && (ph === "scan-cells" || after) && !scannedSet.has(i)) || !inLayer;
        const flatScan = ph === "scan";
        const cellScan = ph === "scan-cells" && scannedSet.has(i);
        const order = cellScan ? iv.scanned.indexOf(i) / iv.scanned.length : i / pts.length;
        return (
          <motion.circle key={`${animKey}-p${i}`} cx={px(pt)} cy={py(pt)} r={r} fill={fill}
            initial={ph === "data" ? { opacity: 0, scale: 0 } : flatScan || cellScan ? { opacity: 0.35 } : false}
            animate={{ opacity: dim ? 0.18 : 0.9, scale: 1 }}
            style={{ transformBox: "fill-box", transformOrigin: "center" }}
            transition={{ delay: (ph === "data" ? i * 0.006 : flatScan || cellScan ? 0.4 + order * SCAN : 0) * k, duration: 0.25 * k }} />
        );
      })}
      {/* Flat / IVF sweep: a line from the query to the point being scored */}
      {(ph === "scan" || ph === "scan-cells") && (() => {
        const ids = ph === "scan" ? pts.map((_, i) => i) : iv.scanned;
        if (!ids.length) return null;
        return (
          <motion.line key={`${animKey}-sweep`} x1={px(p.query)} y1={py(p.query)} stroke="var(--accent)" strokeWidth={1.2}
            initial={{ x2: px(pts[ids[0]]), y2: py(pts[ids[0]]), opacity: 0 }}
            animate={{ x2: ids.map((i) => px(pts[i])), y2: ids.map((i) => py(pts[i])), opacity: [0, 1, 1, 0] }}
            transition={{ delay: 0.4 * k, duration: SCAN * k, ease: "linear", opacity: { delay: 0.4 * k, duration: SCAN * k, times: [0, 0.05, 0.95, 1] } }} />
        );
      })()}
      {/* IVF centroids */}
      {p.index === "ivf" && ph !== "data" && ivf.centroids.map((c, j) => {
        const hist = ivf.history.map((hc) => hc[j]);
        return (
          <motion.g key={`${animKey}-c${j}`} initial={ph === "train" ? { x: px(hist[0]), y: py(hist[0]) } : { x: px(c), y: py(c) }}
            animate={ph === "train" ? { x: hist.map(px), y: hist.map(py) } : { x: px(c), y: py(c) }} transition={{ delay: 0.3 * k, duration: TRAIN * k, ease: "easeInOut" }}>
            <rect x={-6} y={-6} width={12} height={12} rx={2} fill={cellColor(j, p.nlist)} stroke="var(--fg)" strokeWidth={1.2} transform="rotate(45)" />
            <text y={-10} textAnchor="middle" fontSize={9} fontWeight={700} fill="var(--fg)">{j}</text>
          </motion.g>
        );
      })}
      {ph === "probe" && iv.centroidOrder.map((j, rank) => (
        <motion.line key={`${animKey}-pl${j}`} x1={px(p.query)} y1={py(p.query)} x2={px(ivf.centroids[j])} y2={py(ivf.centroids[j])}
          stroke={rank < p.nprobe ? "var(--accent)" : "var(--muted)"} strokeWidth={rank < p.nprobe ? 2 : 1} strokeDasharray={rank < p.nprobe ? undefined : "3 3"}
          initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: centroidAt(rank) * k, duration: 0.3 * k }} />
      ))}
      {/* HNSW search trace */}
      {trace && (
        <g key={`${animKey}-trace`}>
          {trace.evaluated.map((i, j) => (
            <motion.circle key={i} cx={px(pts[i])} cy={py(pts[i])} r={5} fill="none" stroke="var(--warn)" strokeWidth={1.2}
              initial={{ opacity: 0 }} animate={{ opacity: 0.9 }} transition={{ delay: (0.4 + (j / Math.max(1, trace.evaluated.length)) * hopWindow(trace.hops.length)) * k, duration: 0.2 * k }} />
          ))}
          {trace.hops.map((hp, i) => (
            <motion.line key={`h${i}`} x1={px(pts[hp.from])} y1={py(pts[hp.from])} x2={px(pts[hp.to])} y2={py(pts[hp.to])} stroke="var(--accent)" strokeWidth={2.5} strokeLinecap="round"
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: hopAt(i, trace.hops.length) * k, duration: Math.min(0.45, hopWindow(trace.hops.length) / Math.max(1, trace.hops.length)) * k, ease }} />
          ))}
          <motion.circle cx={px(pts[trace.start])} cy={py(pts[trace.start])} r={7} fill="none" stroke="var(--violet)" strokeWidth={2} initial={{ scale: 0 }} animate={{ scale: 1 }} style={{ transformBox: "fill-box", transformOrigin: "center" }} />
          <text x={px(pts[trace.start]) + 9} y={py(pts[trace.start]) - 7} fontSize={10} fontWeight={600} fill="var(--violet)">entry</text>
        </g>
      )}
      {p.index === "hnsw" && ph === "layer" && [...evaluatedBefore].map((i) => <circle key={`eb${i}`} cx={px(pts[i])} cy={py(pts[i])} r={4} fill="none" stroke="var(--warn)" strokeOpacity={0.35} />)}
      {/* Results */}
      {after && (
        <g key={`${animKey}-res`}>
          <motion.circle cx={px(p.query)} cy={py(p.query)} r={kth * 4.4} fill="none" stroke="var(--fg)" strokeOpacity={0.35} strokeDasharray="4 4" initial={{ scale: 0 }} animate={{ scale: 1 }} style={{ transformBox: "fill-box", transformOrigin: "center" }} transition={{ duration: 0.6 * k, ease }} />
          {res.ids.map((i, r) => (
            <motion.circle key={`r${i}`} cx={px(pts[i])} cy={py(pts[i])} r={6.5} fill="none" stroke="var(--ok)" strokeWidth={2.2} initial={{ scale: 0 }} animate={{ scale: 1 }} style={{ transformBox: "fill-box", transformOrigin: "center" }} transition={{ delay: (0.3 + r * 0.12) * k, type: "spring", stiffness: 300, damping: 16 }} />
          ))}
          {truth.ids.filter((i) => !res.ids.includes(i)).map((i) => (
            <motion.g key={`m${i}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 * k }}>
              <circle cx={px(pts[i])} cy={py(pts[i])} r={7.5} fill="none" stroke="var(--err)" strokeWidth={2} strokeDasharray="3 2" />
              <text x={px(pts[i]) + 9} y={py(pts[i]) + 4} fontSize={9} fill="var(--err)">missed</text>
            </motion.g>
          ))}
        </g>
      )}
      {/* Insert */}
      {ph === "insert" && (
        <g key={`${animKey}-ins`}>
          {p.index === "ivf" && ivf.centroids.map((c, j) => (
            <motion.line key={j} x1={px(insert.p)} y1={py(insert.p)} x2={px(c)} y2={py(c)} stroke={j === insert.ivfCell ? "var(--accent)" : "var(--muted)"} strokeWidth={j === insert.ivfCell ? 2.2 : 0.8} strokeDasharray={j === insert.ivfCell ? undefined : "3 3"}
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: (0.8 + j * 0.12) * k, duration: 0.3 * k }} />
          ))}
          {p.index === "hnsw" && insert.links.flatMap((l, li) => l.to.map((j, ji) => (
            <motion.line key={`${l.layer}-${j}`} x1={px(insert.p)} y1={py(insert.p)} x2={px(pts[j])} y2={py(pts[j])} stroke={l.layer === 0 ? "var(--accent)" : "var(--violet)"} strokeWidth={l.layer === 0 ? 1.6 : 2.4}
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ delay: (0.8 + li * 1 + ji * 0.15) * k, duration: 0.35 * k }} />
          )))}
          <motion.g initial={{ y: -40, opacity: 0, scale: 0.4 }} animate={{ y: 0, opacity: 1, scale: 1 }} transition={{ delay: 0.2 * k, type: "spring", stiffness: 200, damping: 14 }}>
            <circle cx={px(insert.p)} cy={py(insert.p)} r={6} fill={p.index === "ivf" ? cellColor(insert.ivfCell, p.nlist) : "var(--violet)"} stroke="var(--fg)" strokeWidth={1.5} />
            <text x={px(insert.p) + 9} y={py(insert.p) - 8} fontSize={10} fontWeight={600} fill="var(--fg)">new{p.index === "flat" ? ` → row ${pts.length}` : ""}</text>
          </motion.g>
        </g>
      )}
      {/* Query */}
      <motion.g pointerEvents="none" initial={false} animate={{ x: px(p.query), y: py(p.query) }} transition={{ type: "spring", stiffness: 170, damping: 22 }}>
        <path d="M-7 -7 l14 14 M7 -7 l-14 14" stroke="var(--err)" strokeWidth={3} />
        <text x={9} y={15} fontSize={11} fontWeight={600} fill="var(--err)">query</text>
      </motion.g>
      {p.index === "hnsw" && ph === "layer" && <text x={S - 12} y={20} textAnchor="end" fontSize={12} fontWeight={700} fill="var(--accent)">layer {layer} ({h.levels.filter((l) => l >= layer!).length} nodes)</text>}
    </svg>
  );
}

function StageDetail({ run, state, k, curves }: { run: VdbRun; state: VdbState; k: number; curves: ReturnType<typeof tradeoff> }) {
  const { p, pts, truth, ivf, iv, h, hs, insert } = run;
  const res = p.index === "flat" ? truth : p.index === "ivf" ? iv : hs;
  const rowIn = (i: number, d = 0.2) => ({ initial: { opacity: 0, x: -8 }, animate: { opacity: 1, x: 0 }, transition: { delay: (d + i * 0.15) * k } });
  switch (state.phase) {
    case "data":
      return (
        <div className="grid gap-3 sm:grid-cols-2">
          <KV rows={[["stored vectors", pts.length], ["dimensions here", "2 (real: 384–3072)"], ["k", p.k], ["metric", "Euclidean (L2)"]]} />
          <p className="text-xs text-muted">The exact answer (what a flat scan returns) is ids <span className="font-mono text-fg">{truth.ids.join(", ")}</span>. Every index is graded against it.</p>
        </div>
      );
    case "scan":
      return <ScanBar n={pts.length} total={pts.length} k={k} label="vectors compared" />;
    case "scan-cells":
      return <ScanBar n={iv.scanned.length} total={pts.length} k={k} label={`vectors in cells ${iv.probed.join(", ")} (plus ${p.nlist} centroid distances)`} />;
    case "train": {
      const moves = ivf.history.slice(1).map((hc, it) => hc.reduce((s, c, j) => s + dist(c, ivf.history[it][j]), 0));
      const max = Math.max(...moves, 1e-9);
      return (
        <div className="space-y-1">
          <p className="text-xs text-muted">Total centroid movement per iteration: big at first, then it settles (converges).</p>
          {moves.map((m, it) => (
            <div key={it} className="grid grid-cols-[70px_minmax(0,1fr)_48px] items-center gap-2 text-xs">
              <span>iteration {it + 1}</span>
              <div className="h-3 rounded bg-surface-2"><motion.div className="h-full rounded bg-violet/70" initial={{ width: 0 }} animate={{ width: `${(m / max) * 100}%` }} transition={{ delay: (0.3 + (it * TRAIN) / moves.length) * k, duration: 0.3 * k }} /></div>
              <span className="text-right font-mono">{m.toFixed(1)}</span>
            </div>
          ))}
        </div>
      );
    }
    case "assign": {
      const sizes = ivf.centroids.map((_, j) => ivf.assign.filter((a) => a === j).length);
      const max = Math.max(...sizes);
      return (
        <div className="flex h-28 items-end gap-1.5">
          {sizes.map((s, j) => (
            <div key={j} className="flex flex-1 flex-col items-center gap-0.5">
              <span className="font-mono text-[10px]">{s}</span>
              <motion.div className="w-full rounded-t" style={{ background: cellColor(j, p.nlist) }} initial={{ height: 0 }} animate={{ height: (s / max) * 80 }} transition={{ delay: (0.2 + j * 0.1) * k, duration: 0.4 * k, ease }} />
              <span className="text-[10px] text-muted">cell {j}</span>
            </div>
          ))}
        </div>
      );
    }
    case "probe":
      return (
        <ol className="grid gap-1 sm:grid-cols-2">
          {iv.centroidOrder.map((j, r) => (
            <motion.li key={j} {...rowIn(r, 0.3)} className={cx("flex items-center gap-2 rounded px-2 py-0.5 text-xs", r < p.nprobe ? "bg-accent-soft" : "")}>
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: cellColor(j, p.nlist) }} />
              <span>cell {j}</span><span className="font-mono text-muted">d = {dist(p.query, ivf.centroids[j]).toFixed(1)}</span>
              {r < p.nprobe ? <Badge tone="accent">probe</Badge> : <span className="ml-auto text-[10px] text-muted">skip ({ivf.assign.filter((a) => a === j).length} vectors)</span>}
            </motion.li>
          ))}
        </ol>
      );
    case "build":
      return (
        <div className="space-y-1">
          {[...Array(h.maxLevel + 1).keys()].reverse().map((l, r) => {
            const n = h.levels.filter((x) => x >= l).length;
            const edges = [...h.adj[l].values()].reduce((s, v) => s + v.length, 0) / 2;
            return (
              <div key={l} className="grid grid-cols-[60px_minmax(0,1fr)_120px] items-center gap-2 text-xs">
                <span>layer {l}</span>
                <div className="h-3.5 rounded bg-surface-2"><motion.div className={cx("h-full rounded", l === 2 ? "bg-violet/70" : l === 1 ? "bg-accent/70" : "bg-muted/50")} initial={{ width: 0 }} animate={{ width: `${(n / pts.length) * 100}%` }} transition={{ delay: (0.3 + r * 0.7) * k, duration: 0.5 * k, ease }} /></div>
                <span className="font-mono">{n} nodes · {edges} links</span>
              </div>
            );
          })}
        </div>
      );
    case "layer": {
      const l = hs.layers.find((x) => x.layer === state.layer)!;
      return (
        <div className="space-y-1.5">
          {l.hops.length === 0 && <p className="text-xs">Entry node {l.start} is already closer to the query than all of its layer-{l.layer} neighbours ({l.evaluated.length} checked), so the search drops straight to layer {l.layer - 1}.</p>}
          <ol className="grid gap-x-3 gap-y-0.5 sm:grid-cols-2">
            {l.hops.slice(0, 16).map((hp, i) => (
              <motion.li key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: hopAt(i, l.hops.length) * k }} className="font-mono text-[11px]">
                {l.layer > 0 ? "hop" : "expand"} {hp.from} → {hp.to} <span className="text-muted">d = {dist(p.query, pts[hp.to]).toFixed(1)}</span>
              </motion.li>
            ))}
          </ol>
          {l.hops.length > 16 && <p className="text-[11px] text-muted">… and {l.hops.length - 16} more expansions.</p>}
          <p className="text-[11px] text-muted">Orange rings: nodes whose distance was computed on this layer ({l.evaluated.length}). Faint rings: computed on higher layers.</p>
        </div>
      );
    }
    case "topk":
      return (
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">rank</th><th>returned id</th><th className="text-right">distance</th><th className="text-right">in exact top-{p.k}?</th></tr></thead>
          <tbody>
            {res.ids.map((i, r) => (
              <motion.tr key={i} {...rowIn(r)} className="border-t border-line">
                <td className="py-0.5 font-mono">{r + 1}</td><td className="font-mono">{i}</td><td className="text-right font-mono">{dist(p.query, pts[i]).toFixed(2)}</td>
                <td className="text-right">{truth.ids.includes(i) ? <Badge tone="ok"><CheckCircle2 size={10} /> yes</Badge> : <Badge tone="err"><XCircle size={10} /> no</Badge>}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      );
    case "compare":
      return <Tradeoff run={run} curves={curves} k={k} />;
    case "insert":
      return p.index === "flat"
        ? <KV rows={[["work at insert", "append 1 row"], ["distance computations", 0], ["rows after insert", pts.length + 1], ["every future query", `${pts.length + 1} distances`]]} />
        : p.index === "ivf"
          ? <KV rows={[["centroid distances", insert.ivfComputations], ["assigned cell", insert.ivfCell], ["cell size after", ivf.assign.filter((a) => a === insert.ivfCell).length + 1], ["centroids retrained?", "no (needs periodic rebuild)"]]} />
          : <KV rows={[["level drawn", insert.level], ...insert.links.map((l) => [`layer ${l.layer} links`, l.to.join(", ")] as [string, string]), ["distances (exhaustive here)", insert.hnswComputations]]} />;
  }
}

function ScanBar({ n, total, k, label }: { n: number; total: number; k: number; label: string }) {
  return (
    <div className="space-y-1.5">
      <div className="relative h-6 overflow-hidden rounded bg-surface-2">
        <motion.div className="absolute inset-y-0 left-0 bg-accent/70" initial={{ width: 0 }} animate={{ width: `${(n / total) * 100}%` }} transition={{ delay: 0.4 * k, duration: SCAN * k, ease: "linear" }} />
        <span className="absolute inset-0 flex items-center justify-center font-mono text-xs"><Counter from={0} to={n} delay={0.4 * k} duration={SCAN * k} />&nbsp;/ {total}</span>
      </div>
      <p className="text-xs text-muted">{label}. At a million vectors the same strategy would cost {n === total ? "a million" : `about ${Math.round((n / total) * 1e6).toLocaleString()}`} distance computations per query.</p>
    </div>
  );
}

function Tradeoff({ run, curves, k }: { run: VdbRun; curves: ReturnType<typeof tradeoff>; k: number }) {
  const W = 560, H = 220, L = 44, R = 14, T = 14, B = 34;
  const maxX = Math.max(run.pts.length + run.p.nlist, ...curves.hnswCurve.map((c) => c.x)) * 1.05;
  const sx = (x: number) => L + (x / maxX) * (W - L - R), sy = (y: number) => T + (1 - y) * (H - T - B);
  const line = (c: { x: number; y: number }[]) => c.map((q, i) => `${i ? "L" : "M"}${sx(q.x).toFixed(1)} ${sy(q.y).toFixed(1)}`).join(" ");
  const cur = { ivf: { x: run.iv.computations, y: run.iv.recall }, hnsw: { x: run.hs.computations, y: run.hs.recall }, flat: curves.flat };
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Recall versus distance computations for IVF, HNSW and flat search">
        {[0, 0.5, 1].map((y) => <g key={y}><line x1={L} x2={W - R} y1={sy(y)} y2={sy(y)} stroke="var(--line)" /><text x={L - 6} y={sy(y) + 3} textAnchor="end" fontSize={10} fill="var(--muted)">{y.toFixed(1)}</text></g>)}
        {[0, 50, 100, 150, 200].map((x) => <text key={x} x={sx(x)} y={H - B + 14} textAnchor="middle" fontSize={10} fill="var(--muted)">{x}</text>)}
        <text x={(L + W) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--muted)">distance computations per query</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={11} fill="var(--muted)" transform={`rotate(-90 12 ${H / 2})`}>recall@{run.p.k}</text>
        <motion.path d={line(curves.ivfCurve)} fill="none" stroke="var(--warn)" strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3 * k, duration: 1.2 * k }} />
        <motion.path d={line(curves.hnswCurve)} fill="none" stroke="var(--violet)" strokeWidth={2} strokeDasharray="5 3" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1.2 * k, duration: 1.2 * k }} />
        {curves.ivfCurve.map((c, i) => <motion.circle key={`i${i}`} cx={sx(c.x)} cy={sy(c.y)} r={3} fill="var(--warn)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.3 + (i / curves.ivfCurve.length) * 1.2) * k }}><title>{c.label}: {c.x} computations, recall {c.y.toFixed(2)}</title></motion.circle>)}
        {curves.hnswCurve.map((c, i) => <motion.circle key={`h${i}`} cx={sx(c.x)} cy={sy(c.y)} r={3} fill="var(--violet)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (1.2 + (i / curves.hnswCurve.length) * 1.2) * k }}><title>{c.label}: {c.x} computations, recall {c.y.toFixed(2)}</title></motion.circle>)}
        <motion.g initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 2.5 * k }} style={{ transformBox: "fill-box", transformOrigin: "center" }}>
          <rect x={sx(cur.flat.x) - 5} y={sy(1) - 5} width={10} height={10} fill="var(--fg)" />
        </motion.g>
        <text x={sx(cur.flat.x)} y={sy(1) + 18} textAnchor="middle" fontSize={10} fill="var(--fg)">flat</text>
        {(["ivf", "hnsw"] as const).map((key) => (
          <motion.circle key={key} cx={sx(cur[key].x)} cy={sy(cur[key].y)} r={8} fill="none" stroke={key === "ivf" ? "var(--warn)" : "var(--violet)"} strokeWidth={2.5}
            initial={{ scale: 0 }} animate={{ scale: [0, 1.4, 1] }} style={{ transformBox: "fill-box", transformOrigin: "center" }} transition={{ delay: 2.8 * k, duration: 0.5 * k }} />
        ))}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1"><svg width="20" height="6" aria-hidden><line x1="0" y1="3" x2="20" y2="3" stroke="var(--warn)" strokeWidth="2" /></svg>IVF, nprobe 1 → {run.p.nlist}</span>
        <span className="flex items-center gap-1"><svg width="20" height="6" aria-hidden><line x1="0" y1="3" x2="20" y2="3" stroke="var(--violet)" strokeWidth="2" strokeDasharray="4 3" /></svg>HNSW, ef {run.p.k} → 64</span>
        <span>■ flat</span><span>◯ your current settings</span>
      </figcaption>
    </figure>
  );
}

function Calculation({ run }: { run: VdbRun }) {
  const { p, pts, truth, iv, hs } = run;
  const nn = pts[truth.ids[0]];
  const n = (v: number) => (v < 0 ? `(${v})` : `${v}`);
  return (
    <div className="space-y-2">
      <TexBlock>{`\\begin{aligned} d(\\mathbf{q}, \\mathbf{x}) &= \\sqrt{(q_1 - x_1)^2 + (q_2 - x_2)^2} \\\\ d(\\mathbf{q}, \\mathbf{x}_{${truth.ids[0]}}) &= \\sqrt{(${p.query[0]} - ${n(nn[0])})^2 + (${p.query[1]} - ${n(nn[1])})^2} = ${dist(p.query, nn).toFixed(2)} \\end{aligned}`}</TexBlock>
      <TexBlock>{p.index === "flat" ? `C_{\\text{flat}} = N = ${pts.length}` : p.index === "ivf" ? `C_{\\text{IVF}} = n_{\\text{list}} + \\textstyle\\sum_{c \\in \\text{probed}} |c| = ${p.nlist} + ${iv.scanned.length} = ${iv.computations}` : `C_{\\text{HNSW}} = |\\text{nodes whose distance was computed}| = ${hs.computations}`}</TexBlock>
      <TexBlock>{`\\text{recall@}k = \\frac{|\\text{returned} \\cap \\text{exact}|}{k} = \\frac{${Math.round((p.index === "flat" ? 1 : p.index === "ivf" ? iv.recall : hs.recall) * p.k)}}{${p.k}}`}</TexBlock>
    </div>
  );
}
