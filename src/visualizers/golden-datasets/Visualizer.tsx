import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, NumberField, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, pct } from "../offline-evaluation/kit";
import {
  CATS, DEFAULT_GOLDEN, GATE, HISTORY_SIZE, LABELS, LOG_SIZE, MIN_SUPPORT, NEAR_DUP, jaccard, simulate,
  type Cat, type GoldenParams, type GoldenSim, type Item, type Phase, type Strategy,
} from "./simulation";

const PHASES: Phase[] = ["logs", "sample", "dedupe", "label", "adjudicate", "coverage", "freeze", "evalV1", "v2", "evalV2"];
const ord = (p: Phase) => PHASES.indexOf(p);
const DUR: Record<Phase, number> = { logs: 2800, sample: 3000, dedupe: 3400, label: 3600, adjudicate: 3400, coverage: 3200, freeze: 2600, evalV1: 3600, v2: 3600, evalV2: 3600 };
const catOf = (c: Cat) => CATS.find((x) => x.id === c)!;

const PRESETS: { label: string; p: Partial<GoldenParams> }[] = [
  { label: "Coverage gap (default)", p: DEFAULT_GOLDEN },
  { label: "Stratified sampling", p: { ...DEFAULT_GOLDEN, strategy: "stratified" } },
  { label: "Sloppy reviewers (60%)", p: { ...DEFAULT_GOLDEN, reviewerAcc: 0.6 } },
];

function narrate(sim: GoldenSim, ph: Phase) {
  const reg = sim.params.regression === "none" ? null : catOf(sim.params.regression).name;
  switch (ph) {
    case "logs": return { headline: "Start from real production traffic", sub: `${LOG_SIZE} logged queries. Colour = category. Rare categories are barely visible.` };
    case "sample": return { headline: `Sample ${sim.sampled.length} candidates (${sim.params.strategy})`, sub: sim.params.strategy === "random" ? "Random sampling copies production's skew: lots of billing, almost no non-English." : "Stratified: a quota per category, so rare ones are represented." };
    case "dedupe": return { headline: `Duplicates drop out: ${sim.dups.length} of ${sim.sampled.length}`, sub: `Popular questions repeat in logs. Normalized text with token overlap ≥ ${NEAR_DUP} counts as the same example.` };
    case "label": return { headline: `Two reviewers label each example: κ = ${sim.kappa.kappa.toFixed(2)}`, sub: "Green ring = both gave the same expected behaviour; amber ≠ = they disagree." };
    case "adjudicate": return { headline: `${sim.disagreements} disagreements resolved by a senior reviewer`, sub: sim.agreedWrong ? `${sim.agreedWrong} label(s) are wrong even though both reviewers agreed.` : "Agreement is not proof of correctness, but here no agreed label is wrong." };
    case "coverage": return { headline: sim.gaps.length ? `Coverage gaps: ${sim.gaps.map((g) => catOf(g.cat).name).join(", ")}` : "Every category meets its coverage target", sub: `Dashed slots = target of ${sim.params.target} per category. Empty slots are blind spots.` };
    case "freeze": return { headline: `golden-v1 frozen: ${sim.v1.length} examples`, sub: `Fingerprint ${sim.hashV1}. The same set is used for every release, so scores are comparable.` };
    case "evalV1": return reg
      ? { headline: sim.g1.pass ? `A ${reg} regression sails through v1` : `v1 already catches the ${reg} regression`, sub: sim.g1.pass ? `Overall ${pct(sim.g1.overall)} ≥ 90%, and ${reg} has fewer than ${MIN_SUPPORT} examples, so no category rule can fire.` : "Enough examples in that category for the per-category rule to fire." }
      : { headline: `Candidate passes ${pct(sim.g1.overall)} of v1`, sub: "No injected regression in this run." };
    case "v2": return { headline: `v2: +${sim.added.length} targeted examples, ~${sim.relabelled} relabelled`, sub: `Searched a ${HISTORY_SIZE.toLocaleString()}-query 90-day log for the gap categories, deduped against v1.` };
    case "evalV2": return { headline: sim.g2.pass ? "The candidate passes v2" : `v2 catches it: ${reg ?? "a category"} fails the per-category gate`, sub: sim.g2.pass ? "With full coverage and no regression, the gate passes." : "Same model, same rule, better golden set: the regression is now visible." };
  }
}

export default function GoldenDatasetsVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<GoldenParams>(DEFAULT_GOLDEN);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(params), [params]);
  const pb = usePlayback(sim.steps.length, (i) => DUR[PHASES[i]] ?? 3000);
  const step = sim.steps[pb.index];
  const ph = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<GoldenParams>) => setParams((c) => ({ ...c, ...p }));
  if (tab === "metrics") return null;
  const story = narrate(sim, ph);

  return (
    <Workspace
      main={
        <>
          <Panel title="Golden set pipeline" actions={<SimulatedLabel>Simulated logs & reviewers</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[580px]"><Canvas sim={sim} ph={ph} k={k} animKey={animKey} /></div></div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
              {CATS.map((c) => <span key={c.id} className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="4.5" fill={c.color} /></svg>{c.name}</span>)}
              <span>· ✓ / ✗ = candidate passes / fails that example</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} ph={ph} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_GOLDEN); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="mb-3 flex flex-wrap gap-1.5">
              <span className="self-center text-xs text-muted">Presets:</span>
              {PRESETS.map((x) => <Button key={x.label} size="sm" onClick={() => { set(x.p); pb.restart(); }}>{x.label}</Button>)}
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Sampling strategy" value={params.strategy} options={[{ value: "random" as Strategy, label: "Random from logs" }, { value: "stratified" as Strategy, label: "Stratified by category" }]} onChange={(v) => set({ strategy: v })} />
              <Slider label="Candidates sampled" value={params.sampleSize} min={20} max={80} step={4} onChange={(v) => set({ sampleSize: v })} />
              <Slider label="Reviewer accuracy" value={params.reviewerAcc} min={0.5} max={0.99} step={0.01} format={(v) => pct(v, 0)} onChange={(v) => set({ reviewerAcc: v })} hint="Chance each reviewer gives the right label" />
              <Slider label="Coverage target per category" value={params.target} min={3} max={8} onChange={(v) => set({ target: v })} />
              <Select label="Break it: candidate regresses on" value={params.regression} options={[{ value: "none" as const, label: "nothing (healthy release)" }, ...CATS.map((c) => ({ value: c.id, label: c.name }))]} onChange={(v) => set({ regression: v })} />
              <NumberField label="Seed (logs, sampling, reviewers)" value={params.seed} step={1} min={1} max={9999} onChange={(v) => set({ seed: Math.round(v) })} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={ord(ph) >= ord("evalV1") ? "A golden set can only catch regressions in what it contains. Coverage is a property you must measure and maintain." : "The golden set is the yardstick for every release; bias, duplicates and label noise in it become bias in every decision."} />
          <Panel title="Calculation"><Calculation sim={sim} ph={ph} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["sampled → unique", ord(ph) >= ord("dedupe") ? `${sim.sampled.length} → ${sim.v1.length}` : sim.sampled.length],
              ["Cohen's κ", ord(ph) >= ord("label") ? sim.kappa.kappa.toFixed(3) : "—"],
              ["v1 / v2 size", ord(ph) >= ord("v2") ? `${sim.v1.length} / ${sim.v2.length}` : ord(ph) >= ord("freeze") ? sim.v1.length : "—"],
              ["gate on v1", ord(ph) >= ord("evalV1") ? (sim.g1.pass ? "PASS" : "BLOCK") : "—"],
              ["gate on v2", ord(ph) >= ord("evalV2") ? (sim.g2.pass ? "PASS" : "BLOCK") : "—"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- canvas ---------- */
const W = 680, H = 420;
const LOG = { x: 10, y: 44, cols: 12, pitch: 7 };
const TRAY = { x: 118, y: 44, cols: 8, pitch: 13 };
const REJ = { x: 118, y: 300, cols: 8, pitch: 13 };
const LAB = { x: 248, y: 44, cols: 8, pitch: 18 };
const BUCK = { x: 432, gap: 40, base: 388, cols: 3, pitch: 11 };
const HIST = { x: 10, y: 300 };
const logXY = (id: number): [number, number] => [LOG.x + (id % LOG.cols) * LOG.pitch + 2.5, LOG.y + Math.floor(id / LOG.cols) * LOG.pitch + 2.5];
const grid = (g: { x: number; y: number; cols: number; pitch: number }, i: number): [number, number] => [g.x + (i % g.cols) * g.pitch + g.pitch / 2, g.y + Math.floor(i / g.cols) * g.pitch + g.pitch / 2];
const bucketXY = (cat: Cat, i: number): [number, number] => {
  const c = CATS.findIndex((x) => x.id === cat);
  return [BUCK.x + c * BUCK.gap + (i % BUCK.cols) * BUCK.pitch + 5, BUCK.base - Math.floor(i / BUCK.cols) * BUCK.pitch - 5];
};

type Pos = { xy: [number, number]; opacity: number } | null;

function itemPos(sim: GoldenSim, it: Item & { dupOf: number | null }, ph: Phase): Pos {
  const o = ord(ph);
  const isDup = it.dupOf !== null;
  if (it.source === "v2") {
    if (o < ord("v2")) return null;
    const idx = sim.v2.filter((x) => x.cat === it.cat).findIndex((x) => x.id === it.id);
    return { xy: bucketXY(it.cat, idx), opacity: 1 };
  }
  if (o < ord("sample")) return null;
  const j = sim.sampled.findIndex((x) => x.id === it.id);
  if (o === ord("sample")) return { xy: grid(TRAY, j), opacity: 1 };
  if (isDup) return { xy: grid(REJ, sim.dups.findIndex((d) => d.id === it.id)), opacity: o === ord("dedupe") ? 1 : 0.35 };
  if (o === ord("dedupe")) return { xy: grid(TRAY, j), opacity: 1 };
  const u = sim.v1.findIndex((x) => x.id === it.id);
  if (o <= ord("adjudicate")) return { xy: grid(LAB, u), opacity: 1 };
  const list = o >= ord("v2") ? sim.v2 : sim.v1;
  return { xy: bucketXY(it.cat, list.filter((x) => x.cat === it.cat).findIndex((x) => x.id === it.id)), opacity: 1 };
}

function delayFor(sim: GoldenSim, it: Item & { dupOf: number | null }, ph: Phase) {
  switch (ph) {
    case "sample": return 0.2 + sim.sampled.findIndex((x) => x.id === it.id) * 0.045;
    case "dedupe": return it.dupOf !== null ? 0.5 + sim.dups.findIndex((d) => d.id === it.id) * 0.12 : 0;
    case "label": return 0.1 + Math.max(0, sim.v1.findIndex((x) => x.id === it.id)) * 0.025;
    case "coverage": return 0.2 + Math.max(0, sim.v1.findIndex((x) => x.id === it.id)) * 0.04;
    case "v2": return it.source === "v2" ? 0.5 + sim.added.findIndex((x) => x.id === it.id) * 0.18 : 0;
    default: return 0;
  }
}
const ringAt = (u: number) => 1.2 + u * 0.035;
const adjAt = (i: number) => 0.4 + i * 0.3;
const evalAt = (i: number) => 0.3 + i * 0.05;

function Canvas({ sim, ph, k, animKey }: { sim: GoldenSim; ph: Phase; k: number; animKey: string }) {
  const o = ord(ph);
  const prev = PHASES[Math.max(0, o - 1)];
  const items: (Item & { dupOf: number | null })[] = [
    ...sim.v1, ...sim.dups.map((d) => ({ ...d, source: "v1" as const, basePass: true, candPass: true })), ...sim.added,
  ];
  const sampledIds = new Set(sim.sampled.map((e) => e.id));
  const disagreeIdx = sim.v1.filter((i) => i.r1 !== i.r2).map((i) => i.id);
  const list = o >= ord("v2") ? sim.v2 : sim.v1;
  const evalOn = ph === "evalV1" || ph === "evalV2";
  const g = ph === "evalV2" ? sim.g2 : sim.g1;
  const cov = o >= ord("v2") ? sim.coverage2 : sim.coverage;
  const head = (x: number, t: string, active: boolean) => <text x={x} y={22} fontSize={10.5} fontWeight={active ? 700 : 500} fill={active ? "var(--accent)" : "var(--muted)"}>{t}</text>;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Golden dataset pipeline, step ${ph}`}>
      {head(LOG.x, `Logs (${LOG_SIZE})`, ph === "logs")}
      {head(TRAY.x, "Candidates", ph === "sample" || ph === "dedupe")}
      {head(LAB.x, "Labeling (R1 vs R2)", ph === "label" || ph === "adjudicate")}
      {head(BUCK.x, `Coverage by category${o >= ord("freeze") ? (o >= ord("v2") ? " · golden-v2" : " · golden-v1") : ""}`, o >= ord("coverage"))}
      {/* production log: a static grid that streams in on step 1 */}
      <g key={ph === "logs" ? `log-${animKey}` : "log"}>
        {sim.log.map((e) => {
          const [x, y] = logXY(e.id);
          const picked = o >= ord("sample") && sampledIds.has(e.id);
          return ph === "logs"
            ? <motion.rect key={e.id} x={x - 2.5} y={y - 2.5} width={5} height={5} fill={catOf(e.cat).color} initial={{ opacity: 0 }} animate={{ opacity: 0.85 }} transition={{ delay: (0.1 + e.id * 0.008) * k, duration: 0.2 * k }} />
            : <rect key={e.id} x={x - 2.5} y={y - 2.5} width={5} height={5} fill={catOf(e.cat).color} opacity={picked ? 0.15 : 0.55} />;
        })}
      </g>
      <line x1={108} x2={108} y1={34} y2={240} stroke="var(--line)" />
      <line x1={238} x2={238} y1={34} y2={240} stroke="var(--line)" />
      <line x1={420} x2={420} y1={34} y2={H - 10} stroke="var(--line)" />
      {o >= ord("dedupe") && <text x={REJ.x} y={REJ.y - 10} fontSize={10} fill="var(--err)">✗ duplicates removed ({sim.dups.length})</text>}
      {ph === "v2" && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 * k }}>
          <rect x={HIST.x} y={HIST.y - 24} width={96} height={56} rx={8} fill="var(--violet)" fillOpacity={0.08} stroke="var(--violet)" strokeDasharray="4 3" />
          <text x={HIST.x + 48} y={HIST.y - 6} textAnchor="middle" fontSize={9.5} fill="var(--violet)">90-day log search</text>
          <text x={HIST.x + 48} y={HIST.y + 10} textAnchor="middle" fontSize={9} fill="var(--muted)">{HISTORY_SIZE.toLocaleString()} queries</text>
        </motion.g>
      )}
      {/* coverage buckets: dashed target slots, labels, per-category eval results */}
      {o >= ord("coverage") && CATS.map((c, ci) => {
        const n = cov.find((x) => x.cat === c.id)!.n;
        const x0 = BUCK.x + ci * BUCK.gap;
        const ge = g.per.find((p) => p.cat === c.id)!;
        return (
          <g key={c.id}>
            {Array.from({ length: sim.params.target }, (_, i) => { const [x, y] = bucketXY(c.id, i); return <circle key={i} cx={x} cy={y} r={4.5} fill="none" stroke="var(--muted)" strokeDasharray="2 2" />; })}
            <line x1={x0 - 1} x2={x0 + 34} y1={BUCK.base + 2} y2={BUCK.base + 2} stroke="var(--line)" />
            <text x={x0 + 16} y={BUCK.base + 14} textAnchor="middle" fontSize={8.5} fill="var(--fg)">{c.name}</text>
            <text x={x0 + 16} y={BUCK.base + 26} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fontWeight={n < sim.params.target ? 700 : 400} fill={n < sim.params.target ? "var(--warn)" : "var(--muted)"}>{n}/{sim.params.target}{n < sim.params.target ? "⚠" : ""}</text>
            {evalOn && (
              <motion.text x={x0 + 16} y={BUCK.base - 96} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fontWeight={700} fill={!ge.gated ? "var(--muted)" : ge.ok ? "var(--ok)" : "var(--err)"}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (evalAt(list.length) + 0.2) * k }}>
                {ge.n ? `${Math.round(ge.rate * 100)}%` : "—"}{!ge.gated ? "*" : ge.ok ? "✓" : "✗"}
              </motion.text>
            )}
          </g>
        );
      })}
      {items.map((it) => {
        const to = itemPos(sim, it, ph);
        if (!to) return null;
        const from = it.source === "v2" && ph === "v2" ? { xy: [HIST.x + 48, HIST.y + 4] as [number, number], opacity: 0 } : (itemPos(sim, it, prev) ?? (ph === "sample" ? { xy: logXY(it.id), opacity: 0 } : to));
        const d = delayFor(sim, it, ph) * k;
        const u = sim.v1.findIndex((x) => x.id === it.id);
        const showRing = (ph === "label" || ph === "adjudicate") && it.dupOf === null && it.source === "v1";
        const disagree = it.r1 !== it.r2;
        const adjI = disagreeIdx.indexOf(it.id);
        const evIdx = list.findIndex((x) => x.id === it.id);
        const relabel = ph === "v2" && sim.v2.find((x) => x.id === it.id)?.relabel;
        return (
          <motion.g key={`${it.id}-${animKey}`} initial={{ x: from.xy[0], y: from.xy[1], opacity: from.opacity }} animate={{ x: to.xy[0], y: to.xy[1], opacity: to.opacity }} transition={{ delay: d, duration: 0.7 * k, ease }}>
            <circle r={ph === "label" || ph === "adjudicate" ? 6 : 4.5} fill={catOf(it.cat).color} />
            {it.dupOf !== null && o >= ord("dedupe") && <path d="M-3 -3 L3 3 M3 -3 L-3 3" stroke="var(--surface)" strokeWidth={1.5} />}
            {showRing && (
              <motion.circle r={8.5} fill="none" strokeWidth={2} strokeDasharray={disagree && (ph === "label") ? "2.5 2" : undefined}
                stroke={disagree && ph === "label" ? "var(--warn)" : "var(--ok)"} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (ph === "label" ? ringAt(u) : 0) * k, duration: 0.25 * k }} />
            )}
            {showRing && disagree && ph === "label" && <motion.text x={7} y={-6} fontSize={9} fontWeight={700} fill="var(--warn)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: ringAt(u) * k }}>≠</motion.text>}
            {showRing && disagree && ph === "adjudicate" && (
              <motion.circle r={8.5} fill="none" stroke="var(--warn)" strokeWidth={2} strokeDasharray="2.5 2" initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ delay: adjAt(adjI) * k, duration: 0.3 * k }} />
            )}
            {showRing && disagree && ph === "adjudicate" && (
              <motion.circle r={12} fill="none" stroke="var(--accent)" strokeWidth={1.5} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: [0, 1, 0], scale: [0.6, 1.2, 1.4] }} transition={{ delay: adjAt(adjI) * k, duration: 0.6 * k }} />
            )}
            {evalOn && evIdx >= 0 && (
              <motion.g initial={{ opacity: 0, scale: 0.3 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: evalAt(evIdx) * k, duration: 0.25 * k }}>
                {it.candPass
                  ? <path d="M-2.6 0 L-0.8 2 L2.8 -2.2" stroke="var(--surface)" strokeWidth={1.6} fill="none" />
                  : <><circle r={5.5} fill="var(--surface)" stroke="var(--err)" strokeWidth={1.8} /><path d="M-2.2 -2.2 L2.2 2.2 M2.2 -2.2 L-2.2 2.2" stroke="var(--err)" strokeWidth={1.6} /></>}
              </motion.g>
            )}
            {it.source === "v2" && o >= ord("v2") && <circle r={7} fill="none" stroke="var(--violet)" strokeWidth={1.2} />}
            {relabel && <motion.text x={6} y={-5} fontSize={9} fontWeight={700} fill="var(--violet)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.6 * k }}>~</motion.text>}
          </motion.g>
        );
      })}
      {ph === "freeze" && (
        <motion.g initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 * k, duration: 0.5 * k, ease }}>
          <rect x={BUCK.x - 6} y={250} width={244} height={162} rx={10} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="6 4" />
          <rect x={BUCK.x + 30} y={240} width={170} height={20} rx={10} fill="var(--accent)" />
          <text x={BUCK.x + 115} y={254} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--surface)">🔒 golden-v1 · {sim.hashV1}</text>
        </motion.g>
      )}
      {evalOn && (
        <motion.g initial={{ opacity: 0, scale: 1.3 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (evalAt(list.length) + 0.5) * k, type: "spring", stiffness: 220, damping: 16 }} style={{ transformOrigin: "550px 72px" }}>
          <rect x={440} y={50} width={222} height={46} rx={8} fill={g.pass ? "var(--ok-soft)" : "var(--err-soft)"} stroke={g.pass ? "var(--ok)" : "var(--err)"} strokeWidth={2} />
          <text x={551} y={70} textAnchor="middle" fontSize={13} fontWeight={800} fill={g.pass ? "var(--ok)" : "var(--err)"}>{g.pass ? "✓ GATE PASSES" : "✗ GATE BLOCKS"}</text>
          <text x={551} y={87} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--fg)">overall {pct(g.overall)} · * = under {MIN_SUPPORT} examples, not gated</text>
        </motion.g>
      )}
      {ph === "label" && <text x={LAB.x} y={250} fontSize={9.5} fill="var(--muted)">ring: green = R1 and R2 agree · amber ≠ = disagree</text>}
    </svg>
  );
}

/* ---------- card ---------- */
function StepCard({ sim, ph, k }: { sim: GoldenSim; ph: Phase; k: number }) {
  if (ph === "logs" || ph === "sample") {
    return (
      <div className="space-y-1.5">
        {CATS.map((c, i) => {
          const logN = sim.log.filter((e) => e.cat === c.id).length;
          const sN = sim.sampled.filter((e) => e.cat === c.id).length;
          return (
            <div key={c.id} className="grid grid-cols-[64px_minmax(0,1fr)_92px] items-center gap-2 text-xs">
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c.color }} />{c.name}</span>
              <div className="h-4 rounded bg-surface-2"><motion.div className="h-4 rounded" style={{ background: c.color, opacity: 0.75 }} initial={{ width: 0 }}
                animate={{ width: `${(ph === "logs" ? logN / sim.log.length : sN / sim.sampled.length) * 100 / 0.4}%` }} transition={{ delay: (0.3 + i * 0.2) * k, duration: 0.5 * k, ease }} /></div>
              <span className="text-right font-mono">{ph === "logs" ? `${logN} (${pct(logN / sim.log.length, 0)})` : `${sN} sampled`}</span>
            </div>
          );
        })}
        <p className="pt-1 text-xs text-muted">{ph === "logs" ? "A golden set should reflect real traffic, but also the rare, high-stakes cases (safety, other languages) that traffic under-represents." : sim.params.strategy === "random" ? "Random sampling: rare categories may get 0–2 examples. Try the stratified preset." : "Stratified sampling: roughly equal quota per category."}</p>
      </div>
    );
  }
  if (ph === "dedupe") {
    const pairs = sim.dups.slice(0, 4).map((d) => ({ dup: d, orig: sim.v1.find((x) => x.id === d.dupOf) }));
    return (
      <div className="space-y-1.5 text-xs">
        {pairs.map(({ dup, orig }, i) => (
          <motion.div key={dup.id} className="grid gap-1 rounded-md bg-surface-2 p-1.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_70px]" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.5 + i * 0.12) * k }}>
            <span className="truncate">kept: “{orig?.text}”</span>
            <span className="truncate text-err">✗ dup: “{dup.text}”</span>
            <span className="font-mono sm:text-right">J = {orig ? jaccard(orig.text, dup.text).toFixed(2) : "—"}</span>
          </motion.div>
        ))}
        {sim.dups.length > 4 && <p className="text-muted">…and {sim.dups.length - 4} more. Duplicates over-weight popular questions and inflate scores.</p>}
        {!sim.dups.length && <p className="text-muted">No duplicates in this sample.</p>}
      </div>
    );
  }
  if (ph === "label") {
    const max = Math.max(1, ...sim.matrix.flat());
    return (
      <div className="grid gap-3 sm:grid-cols-[auto_minmax(0,1fr)]">
        <table className="text-[11px]">
          <thead><tr><th className="pr-1 text-right text-muted">R1 ↓ R2 →</th>{LABELS.map((l) => <th key={l} className="px-1 font-medium text-muted">{l}</th>)}</tr></thead>
          <tbody className="font-mono">
            {LABELS.map((a, i) => (
              <tr key={a}><th className="pr-1 text-right font-sans font-medium text-muted">{a}</th>
                {LABELS.map((b, j) => (
                  <td key={b} className="p-0.5">
                    <motion.div className={cx("flex h-7 w-12 items-center justify-center rounded", i === j ? "text-ok" : sim.matrix[i][j] ? "text-warn" : "text-muted")}
                      style={{ background: i === j ? `color-mix(in srgb, var(--ok) ${(sim.matrix[i][j] / max) * 40}%, transparent)` : sim.matrix[i][j] ? "var(--warn-soft)" : "var(--surface-2)" }}
                      initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (1.2 + (i * 4 + j) * 0.08) * k }}>{sim.matrix[i][j]}</motion.div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="space-y-1 text-xs">
          <KV rows={[["observed agreement pₒ", sim.kappa.po.toFixed(3)], ["chance agreement pₑ", sim.kappa.pe.toFixed(3)], ["Cohen's κ", <AnimatedNumber key="k" value={sim.kappa.kappa} from={0} digits={3} delay={2.6 * k} duration={0.6 * k} />]]} />
          <p className="text-muted">Diagonal = agreement. κ corrects for agreement expected by chance: ≥ 0.8 strong, 0.6–0.8 substantial, &lt; 0.6 the labeling guide needs work.</p>
        </div>
      </div>
    );
  }
  if (ph === "adjudicate") {
    const dis = sim.v1.filter((i) => i.r1 !== i.r2).slice(0, 6);
    return (
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">query</th><th>R1</th><th>R2</th><th>final</th></tr></thead>
          <tbody>
            {dis.map((d, i) => (
              <motion.tr key={d.id} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: adjAt(i) * k }}>
                <td className="max-w-[220px] truncate py-1">{d.text}</td><td className="font-mono">{d.r1}</td><td className="font-mono">{d.r2}</td><td className="font-mono text-ok">✓ {d.label}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
        {sim.agreedWrong > 0 && <Callout tone="warn">{sim.agreedWrong} item(s) have a wrong label that both reviewers agreed on. Only an audit finds these (v2 fixes them).</Callout>}
      </div>
    );
  }
  if (ph === "coverage" || ph === "freeze") {
    if (ph === "freeze") {
      return (
        <pre className="overflow-x-auto rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">{`{
  "name": "support-golden",
  "version": "v1",
  "fingerprint": "fnv1a:${sim.hashV1}",
  "items": ${sim.v1.length},
  "label_schema": ["answer", "lookup", "escalate", "refuse"],
  "coverage": { ${sim.coverage.map((c) => `"${c.cat}": ${c.n}`).join(", ")} },
  "kappa": ${sim.kappa.kappa.toFixed(3)}
}`}</pre>
      );
    }
    return (
      <div className="overflow-x-auto">
        <table className="w-full min-w-[300px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">category</th><th>examples</th><th>target</th><th>status</th></tr></thead>
          <tbody className="font-mono">
            {sim.coverage.map((c, i) => (
              <motion.tr key={c.cat} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.4 + i * 0.25) * k }}>
                <td className="py-1 font-sans">{catOf(c.cat).name}</td><td>{c.n}</td><td>{sim.params.target}</td>
                <td>{c.n >= sim.params.target ? <Badge tone="ok"><CheckCircle2 size={11} /> covered</Badge> : <Badge tone="warn">gap: needs {sim.params.target - c.n}</Badge>}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (ph === "v2") {
    return (
      <div className="grid gap-2 text-xs sm:grid-cols-2">
        <div>
          <div className="mb-1 font-semibold text-ok">+ {sim.added.length} added</div>
          <ul className="space-y-0.5">{sim.added.slice(0, 7).map((a, i) => (
            <motion.li key={a.id} className="truncate font-mono text-[11px]" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.5 + i * 0.18) * k }}>+ [{catOf(a.cat).name}] {a.text}</motion.li>
          ))}</ul>
        </div>
        <div>
          <div className="mb-1 font-semibold text-violet">~ {sim.relabelled} relabelled</div>
          {sim.v2.filter((i) => i.relabel).map((i) => {
            const old = sim.v1.find((x) => x.id === i.id)!;
            return <div key={i.id} className="truncate font-mono text-[11px]">~ {i.text}: {old.label} → {i.label}</div>;
          })}
          {!sim.relabelled && <p className="text-muted">No label errors found in the audit.</p>}
          <p className="mt-2 text-muted">v1 {sim.hashV1} → v2 {sim.hashV2}. Keep v1 around: scores are only comparable within a version.</p>
        </div>
      </div>
    );
  }
  const g = ph === "evalV2" ? sim.g2 : sim.g1;
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[380px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">category</th><th>n</th><th>pass</th><th>rate</th><th>gated?</th><th>result</th></tr></thead>
          <tbody className="font-mono">
            {g.per.map((p, i) => (
              <motion.tr key={p.cat} className={cx("border-t border-line", p.gated && !p.ok && "bg-err-soft")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.4 + i * 0.2) * k }}>
                <td className="py-1 font-sans">{catOf(p.cat).name}</td><td>{p.n}</td><td>{p.pass}</td><td>{p.n ? pct(p.rate, 0) : "—"}</td>
                <td>{p.gated ? "yes" : `no (<${MIN_SUPPORT})`}</td>
                <td>{!p.gated ? <Badge>not checked</Badge> : p.ok ? <Badge tone="ok"><CheckCircle2 size={11} /> ok</Badge> : <Badge tone="err"><XCircle size={11} /> fail</Badge>}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="font-mono text-[11px]">gate = overall ≥ {GATE.overall * 100}% AND every category with n ≥ {MIN_SUPPORT} has pass rate ≥ {GATE.perCat * 100}% → <strong className={g.pass ? "text-ok" : "text-err"}>{g.pass ? "PASS" : "BLOCK"}</strong></p>
    </div>
  );
}

function Calculation({ sim, ph }: { sim: GoldenSim; ph: Phase }) {
  const q = sim.params.reviewerAcc;
  switch (ph) {
    case "logs": return <TexBlock>{`\\text{share}_c = \\frac{n_c}{${LOG_SIZE}}:\\; \\text{non-EN} = \\tfrac{${sim.log.filter((e) => e.cat === "nonen").length}}{${LOG_SIZE}}`}</TexBlock>;
    case "sample": return <TexBlock>{`\\mathbb{E}[n_{\\text{non-EN}}] = N \\cdot \\text{share} = ${sim.sampled.length} \\cdot ${(sim.log.filter((e) => e.cat === "nonen").length / LOG_SIZE).toFixed(3)} = ${(sim.sampled.length * sim.log.filter((e) => e.cat === "nonen").length / LOG_SIZE).toFixed(1)}`}</TexBlock>;
    case "dedupe": {
      const d = sim.dups[0];
      const o = d ? sim.v1.find((x) => x.id === d.dupOf) : undefined;
      return <TexBlock>{`J(a,b) = \\frac{|A \\cap B|}{|A \\cup B|}${o && d ? ` = ${jaccard(o.text, d.text).toFixed(2)}` : ""} \\ge ${NEAR_DUP} \\Rightarrow \\text{duplicate}`}</TexBlock>;
    }
    case "label": return <TexBlock>{`\\begin{aligned} p_e &= \\textstyle\\sum_{\\ell} p_{1\\ell}\\, p_{2\\ell} = ${sim.kappa.pe.toFixed(3)} \\\\ \\kappa &= \\frac{p_o - p_e}{1 - p_e} = \\frac{${sim.kappa.po.toFixed(3)} - ${sim.kappa.pe.toFixed(3)}}{1 - ${sim.kappa.pe.toFixed(3)}} = ${sim.kappa.kappa.toFixed(3)} \\end{aligned}`}</TexBlock>;
    case "adjudicate": return <TexBlock>{`P(\\text{agree, both wrong}) \\approx \\frac{(1-q)^2}{3} = \\frac{(1-${q.toFixed(2)})^2}{3} = ${(((1 - q) ** 2) / 3).toFixed(4)}`}</TexBlock>;
    case "coverage": return <TexBlock>{`\\text{gap}_c = \\max(0,\\, ${sim.params.target} - n_c):\\; ${sim.coverage.map((c) => Math.max(0, sim.params.target - c.n)).join(",\\,")}`}</TexBlock>;
    case "freeze": return <TexBlock>{`\\text{fingerprint} = \\text{FNV-1a}(\\text{id}{:}\\text{label}\\,\\|\\,\\dots) = \\texttt{${sim.hashV1}}`}</TexBlock>;
    case "v2": return <TexBlock>{`|v2| = |v1| + |\\text{added}| = ${sim.v1.length} + ${sim.added.length} = ${sim.v2.length}`}</TexBlock>;
    default: {
      const g = ph === "evalV2" ? sim.g2 : sim.g1;
      const n = ph === "evalV2" ? sim.v2.length : sim.v1.length;
      return <TexBlock>{`\\begin{aligned} \\text{overall} &= \\tfrac{${Math.round(g.overall * n)}}{${n}} = ${g.overall.toFixed(3)} ${g.overall >= GATE.overall ? "\\ge" : "<"} ${GATE.overall} \\\\ \\text{gate} &= [\\text{overall} \\ge ${GATE.overall}] \\wedge \\forall c{:}\\, n_c \\ge ${MIN_SUPPORT} \\Rightarrow r_c \\ge ${GATE.perCat} \\\\ &= \\text{${g.pass ? "PASS" : "BLOCK"}} \\end{aligned}`}</TexBlock>;
    }
  }
}
