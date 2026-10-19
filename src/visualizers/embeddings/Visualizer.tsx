import { useMemo, useState, type FormEvent, type MouseEvent } from "react";
import { AlertTriangle } from "lucide-react";
import { motion } from "motion/react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Badge, Button, Callout, KV, Panel, Select, Slider, cx } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import { dot, euclidean, norm } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { DOCS, METRIC_INFO, PRESET_QUERIES, embedText, precisionAtK, simulate, type Metric, type Vec2 } from "./simulation";

type Query =
  | { kind: "preset"; id: string }
  | { kind: "text"; text: string; v: Vec2 | null; matched: string[] }
  | { kind: "point"; v: Vec2 };

const R = 5.2; // plot range ±R

export default function EmbeddingsVisualizer({ tab }: { tab: VisualizerTab }) {
  const [query, setQuery] = useState<Query>({ kind: "preset", id: "cancel" });
  const [metric, setMetric] = useState<Metric>("cosine");
  const [k, setK] = useState(2);
  const [text, setText] = useState("");
  const [inspect, setInspect] = useState<string | null>(null);

  const preset = query.kind === "preset" ? PRESET_QUERIES.find((p) => p.id === query.id)! : null;
  const qv: Vec2 | null = query.kind === "preset" ? preset!.v : query.v;
  const qLabel = query.kind === "preset" ? preset!.text : query.kind === "text" ? query.text : `point (${query.v[0].toFixed(1)}, ${query.v[1].toFixed(1)})`;
  const sim = useMemo(() => (qv ? simulate(qv, metric, k, preset?.relevant) : null), [qv, metric, k, preset]);
  const pb = usePlayback(sim?.steps.length ?? 1, 900);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    const { v, matched } = embedText(text);
    setQuery({ kind: "text", text: text.trim(), v, matched });
  };

  if (tab === "metrics") return null;

  const step = sim?.steps[pb.index];
  const state = step?.state;
  const showRank = state && ["rank", "topk", "judge"].includes(state.phase);
  const inspected = DOCS.find((d) => d.id === (inspect ?? state?.current)) ?? null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Query">
            <div className="flex flex-wrap gap-1.5">
              {PRESET_QUERIES.map((p) => (
                <Button key={p.id} size="sm" variant={preset?.id === p.id ? "primary" : "secondary"} onClick={() => setQuery({ kind: "preset", id: p.id })}>{p.text}</Button>
              ))}
            </div>
            <form onSubmit={submit} className="mt-2 flex gap-2">
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Or type a query, e.g. 'reset password on tv'" aria-label="Custom query"
                className="h-8 min-w-0 flex-1 rounded-md border border-line bg-surface px-2.5 text-sm" />
              <Button type="submit" size="md">Embed</Button>
            </form>
            {query.kind === "text" && (
              <p className="mt-1.5 text-xs text-muted">
                {query.v ? <>Matched lexicon words: <span className="font-mono">{query.matched.join(", ")}</span></> : "None of these words are in the teaching lexicon, so this query has no vector. Try words like refund, password, video, kids."}
              </p>
            )}
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Select label="Similarity" value={metric} onChange={setMetric} options={(Object.keys(METRIC_INFO) as Metric[]).map((m) => ({ value: m, label: METRIC_INFO[m].name }))} />
              <Slider label="top-k" value={k} min={1} max={6} onChange={setK} />
            </div>
          </Panel>

          <Panel title="Embedding space" actions={<span className="text-[11px] text-muted">Click the plane to place a query vector</span>}>
            <Plane qv={qv} metric={metric} k={k} state={state} ranked={sim?.ranked ?? []} relevant={preset?.relevant} inspect={inspected?.id ?? null}
              onPlace={(v) => setQuery({ kind: "point", v })} onInspect={setInspect} />
            <p className="mt-1 text-[11px] text-muted">2-D teaching vectors, hand-placed. Production embeddings have hundreds to thousands of dimensions; this keeps the geometry visible, not faithful.</p>
          </Panel>

          {sim ? <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} /> : <Callout tone="warn" title="No query vector">Enter a query containing known words, pick a preset, or click the plane.</Callout>}

          {sim && state && (
            <Panel title="Ranking">
              <table className="w-full text-xs">
                <thead className="text-left text-muted"><tr><th className="py-1">rank</th><th>document</th><th className="text-right">{metric}</th>{preset && <th className="text-right">relevant?</th>}</tr></thead>
                <tbody>
                  {(showRank ? sim.ranked : DOCS.map((doc) => sim.ranked.find((r) => r.doc.id === doc.id)!)).map((r) => {
                    const scored = state.scoredIds.includes(r.doc.id);
                    const inTop = showRank && r.rank <= k && state.phase !== "rank";
                    return (
                      <motion.tr layout="position" transition={{ type: "spring", stiffness: 260, damping: 30 }} key={r.doc.id} onClick={() => setInspect(r.doc.id)} className={cx("cursor-pointer border-t border-line transition-colors duration-300", inTop && "bg-accent-soft", state.current === r.doc.id && "outline outline-accent")}>
                        <td className="py-1 font-mono">{showRank ? r.rank : "—"}</td>
                        <td><span className="font-mono text-muted">{r.doc.id}</span> {r.doc.title} {inTop && <Badge tone="accent">retrieved</Badge>}</td>
                        <td className="text-right font-mono">{scored ? r.score.toFixed(3) : "…"}</td>
                        {preset && <td className="text-right">{state.phase === "judge" ? (preset.relevant.includes(r.doc.id) ? <Badge tone="ok">relevant</Badge> : <span className="text-muted">no</span>) : "?"}</td>}
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </Panel>
          )}
        </>
      }
      inspector={
        <>
          {step && <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Retrieval quality is capped by this ranking: anything outside top-k never reaches the model." />}
          <Panel title="Query vector">
            <KV rows={[["text", qLabel], [<Tex>{"q"}</Tex>, qv ? `[${qv.map((v) => v.toFixed(2)).join(", ")}]` : "none"], [<Tex>{"\\lVert q \\rVert"}</Tex>, qv ? norm(qv).toFixed(3) : "—"]]} />
          </Panel>
          {inspected && qv && (
            <Panel title={`Document ${inspected.id}`}>
              <p className="text-sm font-medium">{inspected.title}</p>
              <p className="mt-1 text-xs text-muted">{inspected.text}</p>
              <div className="mt-2"><TexBlock>{`\\begin{aligned}
d &= [${inspected.v.join(",\\ ")}], \\quad \\lVert d \\rVert = ${norm(inspected.v).toFixed(3)} \\\\
q \\cdot d &= ${qv.map((v, i) => `(${v.toFixed(2)})(${inspected.v[i]})`).join(" + ")} = ${dot(qv, inspected.v).toFixed(3)} \\\\
\\cos(q, d) &= \\frac{${dot(qv, inspected.v).toFixed(3)}}{${norm(qv).toFixed(3)} \\times ${norm(inspected.v).toFixed(3)}} = ${(dot(qv, inspected.v) / (norm(qv) * norm(inspected.v))).toFixed(4)} \\\\
\\lVert q - d \\rVert &= ${euclidean(qv, inspected.v).toFixed(3)}
\\end{aligned}`}</TexBlock></div>
            </Panel>
          )}
          {preset && sim && (
            <Panel title="Similarity vs. relevance">
              <KV rows={[[`precision@${k}`, precisionAtK(sim.ranked, k, preset.relevant).toFixed(2)], ["relevant docs", preset.relevant.join(", ")]]} />
              <p className="mt-2 text-xs leading-relaxed">{preset.note}</p>
            </Panel>
          )}
          <Panel title="Metric"><TexBlock>{METRIC_INFO[metric].tex}</TexBlock></Panel>
          {step && <Panel title="Events"><EventLog events={step.events} /></Panel>}
        </>
      }
    />
  );
}

function Plane({ qv, metric, k, state, ranked, relevant, inspect, onPlace, onInspect }: {
  qv: Vec2 | null; metric: Metric; k: number; state?: ReturnType<typeof simulate>["steps"][number]["state"]; ranked: ReturnType<typeof simulate>["ranked"];
  relevant?: string[]; inspect: string | null; onPlace: (v: Vec2) => void; onInspect: (id: string) => void;
}) {
  const S = 440, c = S / 2, sc = (S / 2 - 16) / R;
  const px = (v: Vec2): [number, number] => [c + v[0] * sc, c - v[1] * sc];
  const click = (e: MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM()!.inverse());
    const v: Vec2 = [Math.round(((pt.x - c) / sc) * 10) / 10, Math.round(((c - pt.y) / sc) * 10) / 10];
    if (Math.abs(v[0]) <= R && Math.abs(v[1]) <= R && (v[0] !== 0 || v[1] !== 0)) onPlace(v);
  };
  const showTop = state && (state.phase === "topk" || state.phase === "judge");
  const rankOf = (id: string) => ranked.find((r) => r.doc.id === id)?.rank ?? 99;
  const cur = state?.current ? ranked.find((r) => r.doc.id === state.current)?.doc : undefined;
  const angle = (v: Vec2) => Math.atan2(v[1], v[0]);
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto h-auto w-full max-w-[460px] cursor-crosshair touch-manipulation" onClick={click} role="img" aria-label="Two-dimensional embedding plane with documents and query">
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="var(--accent)" /></marker></defs>
      {[-4, -2, 2, 4].map((g) => <g key={g}><line x1={px([g, 0])[0]} x2={px([g, 0])[0]} y1={8} y2={S - 8} stroke="var(--line)" strokeDasharray="2 4" /><line y1={px([0, g])[1]} y2={px([0, g])[1]} x1={8} x2={S - 8} stroke="var(--line)" strokeDasharray="2 4" /></g>)}
      <line x1={8} x2={S - 8} y1={c} y2={c} stroke="var(--line)" /><line y1={8} y2={S - 8} x1={c} x2={c} stroke="var(--line)" />
      <text x={S - 10} y={c - 6} textAnchor="end" fontSize={10} fill="var(--muted)">billing →</text>
      <text x={c + 6} y={18} fontSize={10} fill="var(--muted)">↑ account</text>
      <text x={10} y={c - 6} fontSize={10} fill="var(--muted)">← playback</text>
      <text x={c + 6} y={S - 10} fontSize={10} fill="var(--muted)">↓ family</text>
      {qv && cur && (
        <g pointerEvents="none">
          {/* The document arrow sweeps from one document to the next as each is scored. */}
          <motion.line x1={c} y1={c} initial={false} animate={{ x2: px(cur.v)[0], y2: px(cur.v)[1] }} transition={{ type: "spring", stiffness: 170, damping: 22 }} stroke="var(--accent)" strokeWidth={1.5} markerEnd="url(#arrow)" />
          <motion.line x1={c} y1={c} initial={false} animate={{ x2: px(qv)[0], y2: px(qv)[1] }} transition={{ type: "spring", stiffness: 170, damping: 22 }} stroke="var(--warn)" strokeWidth={1.5} />
          {metric === "euclidean" && <motion.line initial={false} animate={{ x1: px(qv)[0], y1: px(qv)[1], x2: px(cur.v)[0], y2: px(cur.v)[1] }} transition={{ type: "spring", stiffness: 170, damping: 22 }} stroke="var(--violet)" strokeWidth={2} strokeDasharray="4 3" />}
          {metric === "cosine" && (() => {
            const a1 = angle(qv), a2 = angle(cur.v), r0 = 34;
            let d = a2 - a1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
            const [x1, y1] = [c + r0 * Math.cos(a1), c - r0 * Math.sin(a1)], [x2, y2] = [c + r0 * Math.cos(a1 + d), c - r0 * Math.sin(a1 + d)];
            // Arc flags must stay 0/1, so the angle arc fades in per document instead of morphing.
            return <motion.path key={cur.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25, duration: 0.3 }} d={`M${x1} ${y1} A${r0} ${r0} 0 0 ${d > 0 ? 0 : 1} ${x2} ${y2}`} fill="none" stroke="var(--violet)" strokeWidth={2} />;
          })()}
        </g>
      )}
      {DOCS.map((d) => {
        const [x, y] = px(d.v);
        const r = rankOf(d.id);
        const top = showTop && r <= k;
        const rel = state?.phase === "judge" && relevant?.includes(d.id);
        const scored = state?.scoredIds.includes(d.id);
        return (
          <g key={d.id} onClick={(e) => { e.stopPropagation(); onInspect(d.id); }} className="cursor-pointer" role="button" aria-label={`Inspect ${d.title}`}>
            <title>{d.title}</title>
            {top && <motion.circle cx={x} cy={y} r={11} fill="none" stroke="var(--accent)" strokeWidth={2} style={{ transformBox: "fill-box", transformOrigin: "center" }} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 300, damping: 18 }} />}
            {rel && <motion.circle cx={x} cy={y} r={15} fill="none" stroke="var(--ok)" strokeWidth={2} strokeDasharray="3 2" style={{ transformBox: "fill-box", transformOrigin: "center" }} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 300, damping: 18 }} />}
            <circle cx={x} cy={y} r={5.5} fill={state?.current === d.id || inspect === d.id ? "var(--accent)" : scored ? "var(--fg)" : "var(--muted)"} fillOpacity={scored || !state ? 1 : 0.5} />
            <text x={x + 8} y={y - 6} fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">{d.id}{top ? ` #${r}` : ""}</text>
          </g>
        );
      })}
      {qv && (
        <motion.g pointerEvents="none" initial={false} animate={{ x: px(qv)[0], y: px(qv)[1] }} transition={{ type: "spring", stiffness: 170, damping: 22 }}>
          <path d="M-7 -7 l14 14 M7 -7 l-14 14" stroke="var(--warn)" strokeWidth={3} />
          <text x={9} y={14} fontSize={11} fontWeight={600} fill="var(--warn)">query</text>
        </motion.g>
      )}
      <g transform={`translate(10 ${S - 52})`} fontSize={10} fill="var(--muted)">
        <text y={0}>✕ query   ● document</text>
        <text y={14}>◯ retrieved (top-k)</text>
        {relevant && <text y={28}><tspan fill="var(--ok)">◌</tspan> labelled relevant</text>}
      </g>
      {state?.phase === "judge" && relevant && ranked[0] && !relevant.includes(ranked[0].doc.id) && (
        <foreignObject x={S - 170} y={S - 50} width={160} height={44}><div className="flex items-center gap-1 rounded bg-warn-soft px-2 py-1 text-[10px] text-fg"><AlertTriangle size={12} className="text-warn" />Top hit is similar, not relevant</div></foreignObject>
      )}
    </svg>
  );
}
