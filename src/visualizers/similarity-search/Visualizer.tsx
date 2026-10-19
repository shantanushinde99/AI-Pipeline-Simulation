import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { motion } from "motion/react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { cosine, dot, euclidean, norm } from "../../lib/math";
import { StepCaption } from "./StepCaption";
import type { VisualizerTab } from "../types";
import { DATASETS, METRIC_NAME, simulate, unit, type DatasetId, type Metric, type Ranked, type SSPhase, type SSState, type Vec2 } from "./simulation";

const ease = [0.22, 1, 0.36, 1] as const;
/** One schedule (seconds at 1×) shared by the plane, the card and playback. */
const T = { scanGap: 0.12, line: 0.35, ring: 0.9, slide: 1.2, sortAt: 0.7 };
const S = 420, C = S / 2, SC = (C - 16) / 4.5, SCN = 150;

function narrate(s: SSState, sim: ReturnType<typeof simulate>, metric: Metric, normalize: boolean) {
  const r = sim.byL2[s.captured - 1];
  const map: Record<SSPhase, { headline: string; sub: string }> = {
    scene: { headline: "One query, a handful of stored vectors", sub: "Exact nearest-neighbour search has no index tricks: it will compare the query with every vector." },
    scan: { headline: `Measure the distance from q to all ${sim.items.length} vectors`, sub: "One line per stored vector, in storage order. Cost grows linearly with the collection." },
    capture: { headline: `The ring grows until it touches neighbour #${s.captured}: ${r?.item.id}`, sub: r?.tiedWithNext ? `Careful: ${sim.byL2[s.captured]?.item.id} sits at exactly the same distance.` : `Radius = ‖q − ${r?.item.id}‖ = ${r ? (-r.score).toFixed(3) : ""}. Everything inside the ring is nearer.` },
    metrics: { headline: "Now rank the same vectors three ways", sub: "Cosine looks only at direction, dot product rewards length, L2 measures straight-line distance. Watch the lists re-sort." },
    normalize: { headline: "Normalize: every vector slides onto the unit circle", sub: "Divide each vector by its length. Once all lengths are 1, the three rankings become identical." },
    cutoff: { headline: `Keep the top ${sim.k} by ${METRIC_NAME[metric]}${normalize ? " (normalized)" : " (raw)"}`, sub: sim.tieAtCutoff ? "A tie straddles the cutoff: the tie-break rule, not similarity, decides." : sim.sources < sim.final.length ? `Only ${sim.sources} distinct source${sim.sources === 1 ? "" : "s"} survived: near-duplicates crowd the results.` : "Everything below the line is never seen by the next stage." },
  };
  return map[s.phase];
}
const DUR = (phase: SSPhase, n: number) => ({ scene: 2000, scan: (n * T.scanGap + T.line + 1) * 1000, capture: 2200, metrics: 3200, normalize: 3400, cutoff: 3000 })[phase];

export default function SimilaritySearchVisualizer({ tab }: { tab: VisualizerTab }) {
  const [dataset, setDataset] = useState<DatasetId>("default");
  const [query, setQuery] = useState<Vec2>(DATASETS.default.query);
  const [k, setK] = useState(3);
  const [metric, setMetric] = useState<Metric>("cosine");
  const [normalize, setNormalize] = useState(false);
  const [onePerSource, setOnePerSource] = useState(false);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate({ dataset, query, k, metric, normalize, onePerSource }), [dataset, query, k, metric, normalize, onePerSource]);
  const pb = usePlayback(sim.steps.length, (i) => DUR(sim.steps[i]?.state.phase ?? "scene", sim.items.length));
  if (tab === "metrics") return null;

  const step = sim.steps[pb.index];
  const s = step.state;
  const kk = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${dataset}-${query.join()}-${metric}-${normalize}-${onePerSource}-${k}`;
  const story = narrate(s, sim, metric, normalize);
  const pick = (id: DatasetId) => { setDataset(id); setQuery(DATASETS[id].query); setK(DATASETS[id].k); setOnePerSource(false); };
  const reset = () => { pick("default"); setMetric("cosine"); setNormalize(false); pb.restart(); };
  const top = sim.final[0];
  const tv = top.item.v;

  return (
    <Workspace
      main={
        <>
          <Panel title="Exact nearest-neighbour search" actions={<SimulatedLabel>Hand-placed 2-D vectors</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <Plane sim={sim} s={s} normalize={normalize} k={kk} animKey={animKey} onPlace={setQuery} />
            <p className="mt-1 text-[11px] text-muted">Click the plane to move the query. ✕ = query, ● = stored vector, #n = neighbour rank.</p>
          </Panel>
          <Panel title="What's happening in this step">
            <Card sim={sim} s={s} metric={metric} normalize={normalize} k={kk} animKey={animKey} />
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={reset}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Data preset" value={dataset} onChange={pick} options={(Object.keys(DATASETS) as DatasetId[]).map((d) => ({ value: d, label: DATASETS[d].name }))} />
              <Select label="Metric for the final top-k" value={metric} onChange={setMetric} options={(Object.keys(METRIC_NAME) as Metric[]).map((m) => ({ value: m, label: METRIC_NAME[m] }))} />
              <Slider label="k" value={k} min={1} max={6} onChange={setK} />
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <Toggle label="Index stores normalized vectors" checked={normalize} onChange={setNormalize} hint="Final ranking uses unit-length vectors." />
              <Toggle label="Collapse near-duplicates (one per source)" checked={onePerSource} onChange={setOnePerSource} hint="Fix for the crowding preset." />
            </div>
            <p className="mt-2 text-xs text-muted">{DATASETS[dataset].note}</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why="Every RAG answer is capped by this list: whatever falls below the k cutoff is invisible to the model." />
          <Panel title={`Calculation for top hit ${top.item.id}`}>
            <TexBlock>{`\\begin{aligned}
q &= [${sim.q.join(",\\ ")}],\\ ${top.item.id} = [${tv.join(",\\ ")}] \\\\
q \\cdot ${top.item.id} &= ${dot(sim.q, tv).toFixed(3)} \\\\
\\cos &= \\frac{${dot(sim.q, tv).toFixed(3)}}{${norm(sim.q).toFixed(3)} \\times ${norm(tv).toFixed(3)}} = ${cosine(sim.q, tv).toFixed(4)} \\\\
\\lVert q - ${top.item.id} \\rVert &= ${euclidean(sim.q, tv).toFixed(3)} \\\\
\\hat a &= a / \\lVert a \\rVert \\Rightarrow \\lVert \\hat q - \\hat d \\rVert^2 = 2 - 2\\cos
\\end{aligned}`}</TexBlock>
          </Panel>
          <Panel title="Key values">
            <KV rows={[["‖q‖", norm(sim.q).toFixed(3)], ["unit q", `[${unit(sim.q).map((v) => v.toFixed(3)).join(", ")}]`], ["final top-k", sim.final.map((r) => r.item.id).join(", ")], ["distinct sources", `${sim.sources} / ${sim.final.length}`], ["tie at cutoff", sim.tieAtCutoff ? `yes (${sim.tieAtCutoff.item.id} dropped)` : "no"]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Plane({ sim, s, normalize, k, animKey, onPlace }: { sim: ReturnType<typeof simulate>; s: SSState; normalize: boolean; k: number; animKey: string; onPlace: (v: Vec2) => void }) {
  const normView = s.phase === "normalize" || (s.phase === "cutoff" && normalize);
  const px = (v: Vec2): Vec2 => (normView ? [C + unit(v)[0] * SCN, C - unit(v)[1] * SCN] : [C + v[0] * SC, C - v[1] * SC]);
  const [qx, qy] = px(sim.q);
  const rankL2 = (id: string) => sim.byL2.findIndex((r) => r.item.id === id) + 1;
  const prevR = s.captured > 1 ? -sim.byL2[s.captured - 2].score * SC : 0;
  const curR = s.captured > 0 ? -sim.byL2[s.captured - 1].score * SC : 0;
  const finalIds = sim.final.map((r) => r.item.id);
  const click = (e: MouseEvent<SVGSVGElement>) => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(e.currentTarget.getScreenCTM()!.inverse());
    const v: Vec2 = [Math.round(((pt.x - C) / SC) * 10) / 10, Math.round(((C - pt.y) / SC) * 10) / 10];
    if (Math.abs(v[0]) <= 4.5 && Math.abs(v[1]) <= 4.5 && norm(v) > 0.2) onPlace(v);
  };
  return (
    <svg viewBox={`0 0 ${S} ${S}`} onClick={click} className="mx-auto h-auto w-full max-w-[440px] cursor-crosshair touch-manipulation" role="img" aria-label="Query and stored vectors on a 2-D plane">
      <line x1={8} x2={S - 8} y1={C} y2={C} stroke="var(--line)" /><line y1={8} y2={S - 8} x1={C} x2={C} stroke="var(--line)" />
      <motion.circle cx={C} cy={C} r={SCN} fill="none" stroke="var(--violet)" strokeDasharray="4 4" initial={false} animate={{ opacity: normView ? 1 : 0 }} transition={{ duration: 0.6 * k }} />
      {normView && <text x={C + 6} y={C - SCN - 6} fontSize={10} fill="var(--violet)">unit circle (length 1)</text>}
      {s.phase === "scan" && sim.items.map((it, i) => {
        const [x, y] = px(it.v);
        return (
          <g key={`${it.id}-${animKey}`}>
            <motion.line x1={qx} y1={qy} x2={x} y2={y} stroke="var(--accent)" strokeOpacity={0.6} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: T.line * k, delay: i * T.scanGap * k }} />
            <motion.text x={(qx + x) / 2} y={(qy + y) / 2 - 3} fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)" textAnchor="middle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (i * T.scanGap + T.line) * k }}>
              {euclidean(sim.q, it.v).toFixed(2)}
            </motion.text>
          </g>
        );
      })}
      {s.phase === "capture" && (
        <motion.circle key={`ring-${animKey}`} cx={qx} cy={qy} fill="var(--accent)" fillOpacity={0.06} stroke="var(--accent)" strokeWidth={2} initial={{ r: prevR }} animate={{ r: curR }} transition={{ duration: T.ring * k, ease }} />
      )}
      {sim.items.map((it) => {
        const [x, y] = px(it.v);
        const r = rankL2(it.id);
        const captured = (s.phase === "capture" && r <= s.captured) || (s.phase !== "capture" && s.phase !== "cutoff" && s.captured > 0 && r <= s.captured && !normView);
        const isNew = s.phase === "capture" && r === s.captured;
        const tied = s.phase === "capture" && r === s.captured + 1 && sim.byL2[s.captured - 1]?.tiedWithNext;
        const inFinal = s.phase === "cutoff" && finalIds.includes(it.id);
        return (
          <motion.g key={it.id} initial={false} animate={{ x, y }} transition={{ duration: T.slide * k, ease }}>
            {(inFinal || tied) && <motion.circle key={`hl-${animKey}`} r={11} fill="none" stroke={tied ? "var(--warn)" : "var(--accent)"} strokeWidth={2} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: (tied ? T.ring : 0.3) * k, type: "spring", stiffness: 300, damping: 18 }} />}
            <motion.circle r={5.5} initial={false} animate={{ scale: isNew ? [1, 1.8, 1] : 1, fill: captured || inFinal ? "var(--accent)" : "var(--fg)" }} transition={{ duration: 0.5 * k, delay: isNew ? T.ring * k : 0 }} />
            <text x={8} y={-7} fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">{it.id}{captured && s.phase === "capture" ? ` #${r}` : ""}{inFinal ? ` #${finalIds.indexOf(it.id) + 1}` : ""}</text>
          </motion.g>
        );
      })}
      <motion.g initial={false} animate={{ x: qx, y: qy }} transition={{ duration: T.slide * k, ease }} pointerEvents="none">
        <path d="M-7 -7 l14 14 M7 -7 l-14 14" stroke="var(--warn)" strokeWidth={3} />
        <text x={9} y={14} fontSize={11} fontWeight={600} fill="var(--warn)">q</text>
      </motion.g>
    </svg>
  );
}

function Card({ sim, s, metric, normalize, k, animKey }: { sim: ReturnType<typeof simulate>; s: SSState; metric: Metric; normalize: boolean; k: number; animKey: string }) {
  if (s.phase === "scene" || s.phase === "scan")
    return (
      <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4" key={animKey}>
        {sim.items.map((it, i) => (
          <motion.div key={it.id} className="rounded border border-line bg-surface-2 px-2 py-1 font-mono text-[11px]" initial={{ opacity: s.phase === "scan" ? 0.3 : 1 }} animate={{ opacity: 1 }} transition={{ delay: (i * T.scanGap + T.line) * k }}>
            <b>{it.id}</b> [{it.v.join(", ")}]<br />
            <span className="text-muted">{s.phase === "scan" ? `dist ${euclidean(sim.q, it.v).toFixed(3)}` : `len ${norm(it.v).toFixed(2)}`}</span>
          </motion.div>
        ))}
      </div>
    );
  if (s.phase === "capture")
    return (
      <ol className="space-y-1 text-xs">
        {sim.byL2.slice(0, s.captured).map((r) => (
          <motion.li layout key={r.item.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: r.rank === s.captured ? T.ring * k : 0, duration: 0.3 * k }} className="flex items-center gap-2 rounded border border-line px-2 py-1">
            <span className="font-mono">#{r.rank}</span><b>{r.item.id}</b>
            <span className="ml-auto font-mono">‖q − {r.item.id}‖ = {(-r.score).toFixed(3)}</span>
            {r.tiedWithNext && <Badge tone="warn"><AlertTriangle size={11} /> tie with {sim.byL2[r.rank].item.id}</Badge>}
          </motion.li>
        ))}
      </ol>
    );
  const space = s.phase === "metrics" ? sim.raw : s.phase === "normalize" ? sim.unitRanks : normalize ? sim.unitRanks : sim.raw;
  const fromSpace = s.phase === "normalize" ? sim.raw : null;
  return (
    <div>
      <div className="grid grid-cols-3 gap-2">
        {(["cosine", "dot", "l2"] as Metric[]).map((m) => (
          <RankColumn key={`${m}-${animKey}`} title={METRIC_NAME[m]} from={fromSpace ? fromSpace[m] : sim.byL2} to={space[m]} delay={T.sortAt * k} cut={s.phase === "cutoff" ? sim.k : 0}
            picked={s.phase === "cutoff" && m === metric ? sim.final.map((r) => r.item.id) : null} active={s.phase === "cutoff" && m === metric} />
        ))}
      </div>
      <p className="mt-2 text-xs text-muted">
        {s.phase === "metrics" ? "Lists start in L2 order and re-sort by each metric's own score (raw vectors)." : s.phase === "normalize" ? "Lists start in their raw order and re-sort on unit vectors: they end up identical." : `Final list: ${METRIC_NAME[metric]} on ${normalize ? "normalized" : "raw"} vectors. Rows below the line are discarded${sim.final.length && sim.finalRanked.slice(0, sim.k).some((r) => !sim.final.includes(r)) ? "; struck-through rows were skipped as duplicates of a source already kept" : ""}.`}
      </p>
    </div>
  );
}

function RankColumn({ title, from, to, delay, cut, picked, active }: { title: string; from: Ranked[]; to: Ranked[]; delay: number; cut: number; picked: string[] | null; active: boolean }) {
  const [sorted, setSorted] = useState(false);
  useEffect(() => { const t = window.setTimeout(() => setSorted(true), delay * 1000); return () => window.clearTimeout(t); }, [delay]);
  const list = sorted ? to : from.map((f) => to.find((r) => r.item.id === f.item.id)!);
  return (
    <div className={cx("min-w-0 rounded border p-1.5", active ? "border-accent" : "border-line")}>
      <p className="mb-1 truncate text-[11px] font-semibold">{title}{active && " ✓"}</p>
      <ul className="space-y-0.5">
        {list.map((r, i) => {
          const skipped = picked && i < cut + 3 && !picked.includes(r.item.id) && to.indexOf(r) < to.indexOf(to.find((x) => x.item.id === picked[picked.length - 1])!);
          return (
            <motion.li layout transition={{ type: "spring", stiffness: 220, damping: 28 }} key={r.item.id}
              className={cx("flex justify-between gap-1 rounded px-1 font-mono text-[10px] sm:text-[11px]", cut > 0 && i === cut - 1 && "mb-1 border-b-2 border-dashed border-err", picked?.includes(r.item.id) && "bg-accent-soft", skipped && "line-through opacity-60", cut > 0 && i >= cut && !picked?.includes(r.item.id) && "opacity-50")}>
              <span>{sorted ? `${i + 1}.` : "·"} {r.item.id}</span><span className="text-muted">{r.score.toFixed(2)}</span>
            </motion.li>
          );
        })}
      </ul>
    </div>
  );
}
