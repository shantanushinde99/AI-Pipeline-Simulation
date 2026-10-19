import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, StepCard, ease } from "../model-versioning/kit";
import { DEFAULT_THRESHOLDS, EXPECTED_COLUMNS, PRESETS, STAGES, fmtCheck, runPipeline, type CIState, type Check, type Preset, type StageId, type Status, type Thresholds } from "./simulation";

/** One timing schedule (seconds at 1×): the token arrives, then each gauge fills, then the verdict shows. */
const T = { travel: 0.6, gauge: (i: number) => 0.7 + i * 0.5, gaugeDur: 0.7, verdict: (n: number) => 0.9 + n * 0.5 };
const DURATION: Record<StageId | "result", number> = { commit: 1900, unit: 2600, data: 3200, train: 2900, eval: 3800, canary: 4800, promote: 2900, result: 3200 };

const NX = (i: number) => 58 + i * 107;
const NY = 66, NWD = 96, NHT = 56;
const idx = (id: StageId) => STAGES.findIndex((s) => s.id === id);
const STATUS_GLYPH: Record<Status, { g: string; text: string; color: string }> = {
  pending: { g: "○", text: "pending", color: "var(--muted)" }, active: { g: "●", text: "running", color: "var(--accent)" }, ok: { g: "✓", text: "passed", color: "var(--ok)" },
  warning: { g: "⚠", text: "warning", color: "var(--warn)" }, error: { g: "✗", text: "failed", color: "var(--err)" }, skipped: { g: "–", text: "skipped", color: "var(--muted)" },
};

function narrate(s: CIState, preset: Preset, th: Thresholds): { headline: string; sub: string } {
  const failed = s.checks.find((c) => !c.pass);
  switch (s.active) {
    case "commit": return { headline: `A commit enters the pipeline: "${PRESETS[preset].commit}"`, sub: "Every change, to code or data, goes through the same automated stages before reaching users." };
    case "unit": return { headline: "Unit tests pass: the code does what its author intended", sub: "They check code paths, not whether the data or the model is still good." };
    case "data": return failed
      ? { headline: failed.id === "schema" ? "Data validation catches a schema change" : "Data validation catches drifted inputs", sub: failed.id === "schema" ? `Expected column '${s.data.missing[0]}' is gone. Training would silently use a column of nulls.` : `PSI ${fmtCheck(failed, failed.measured)} is above ${th.maxPsi}. The world changed; someone should look before retraining.` }
      : { headline: "Data validation: schema matches, drift is small", sub: "Columns and types are as expected and each feature's distribution is close to the reference." };
    case "train": return { headline: "Training produces a candidate model", sub: "Training succeeding says nothing about whether the model is better. That is the next stage's job." };
    case "eval": return failed
      ? { headline: `Gate blocks the candidate: ${failed.label}`, sub: failed.id === "slice" ? "Overall accuracy went up, which is exactly why an overall-only gate would have shipped this." : `Measured ${fmtCheck(failed, failed.measured)}, allowed ${failed.op === "<=" ? "at most" : "at least"} ${fmtCheck(failed, failed.threshold)}.` }
      : { headline: "Candidate vs baseline: every gate passes", sub: s.checks.some((c) => c.disabled) ? "But the slice gate is disabled, so slice regressions are not checked." : "Accuracy, latency and the worst slice are all within thresholds." };
    case "canary": return s.canary.rollbackAt
      ? { headline: `Canary at ${th.canaryPct}%: errors spike, automatic rollback`, sub: "Offline tests could not see this: the failure only shows up on the real serving path." }
      : { headline: `Canary at ${th.canaryPct}% of traffic stays healthy`, sub: "Live error rate is compared with the threshold before anyone else sees the model." };
    case "promote": return s.outcome === "promoted-with-regression"
      ? { headline: "Promoted to 100%, carrying a slice regression", sub: "Every enabled gate passed. The one that would have caught this was switched off." }
      : { headline: "Promoted: the candidate now serves 100% of traffic", sub: "It also becomes the baseline the next commit is compared against." };
    case "result": return s.outcome === "rolled-back"
      ? { headline: "Rolled back: the baseline serves everyone again", sub: "The blast radius was limited to the canary slice for a few minutes." }
      : { headline: `Pipeline stopped at "${STAGES[idx(s.failedAt!)].label}"`, sub: "Later stages are skipped; production keeps serving the current model." };
  }
}

export default function CiCdVisualizer({ tab }: { tab: VisualizerTab }) {
  const [preset, setPreset] = useState<Preset>("healthy");
  const [th, setTh] = useState<Thresholds>(DEFAULT_THRESHOLDS);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => runPipeline(preset, th), [preset, th]);
  const pb = usePlayback(steps.length, (i) => DURATION[steps[i]?.state.active ?? "commit"]);
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(s, preset, th);
  const set = (p: Partial<Thresholds>) => setTh((v) => ({ ...v, ...p }));
  if (tab === "metrics") return null;
  const prevIdx = pb.index > 0 ? steps[pb.index - 1].state.active : null;

  return (
    <Workspace
      main={
        <>
          <Panel title="ML pipeline" actions={<SimulatedLabel>Simulated pipeline & traffic</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[620px]"><PipelineSvg s={s} prev={prevIdx} k={k} animKey={animKey} canaryPct={th.canaryPct} /></div></div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>the change travelling through the pipeline</span>
              <span>│ black tick = threshold · ▼ = measured value</span>
              <span>✓ passed · ✗ failed · – skipped · ⚠ gate disabled</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <StepCard animKey={animKey}><StepDetail s={s} k={k} th={th} /></StepCard>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Scenario & gate thresholds" actions={<Button size="sm" variant="ghost" onClick={() => { setTh(DEFAULT_THRESHOLDS); setPreset("healthy"); }}><RotateCcw size={12} /> Reset</Button>}>
            <Select<Preset> label="Break it: failure preset" value={preset} options={(Object.keys(PRESETS) as Preset[]).map((p) => ({ value: p, label: PRESETS[p].label }))} onChange={setPreset} />
            <p className="mt-1 text-xs text-muted">{PRESETS[preset].note}</p>
            <div className="mt-3 grid gap-4 sm:grid-cols-2">
              <Slider label="Max PSI (drift)" value={th.maxPsi} min={0.05} max={0.6} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ maxPsi: v })} />
              <Slider label="Min accuracy change vs baseline" value={th.minAccDelta} min={-0.03} max={0.03} step={0.005} format={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)} pts`} onChange={(v) => set({ minAccDelta: v })} />
              <Slider label="Max p95 latency" value={th.maxP95} min={60} max={200} step={5} format={(v) => `${v} ms`} onChange={(v) => set({ maxP95: v })} />
              <Slider label="Max drop on any slice" value={th.maxSliceDrop} min={0} max={0.12} step={0.01} format={(v) => `${(v * 100).toFixed(0)} pts`} onChange={(v) => set({ maxSliceDrop: v })} />
              <Slider label="Max canary error rate" value={th.maxCanaryErr} min={0.005} max={0.06} step={0.005} format={(v) => `${(v * 100).toFixed(1)}%`} onChange={(v) => set({ maxCanaryErr: v })} />
              <Slider label="Canary traffic share" value={th.canaryPct} min={1} max={50} format={(v) => `${v}%`} onChange={(v) => set({ canaryPct: v })} />
              <Toggle label="Fairness slice gate enabled" checked={th.sliceGate} onChange={(v) => set({ sliceGate: v })} hint="Turn off with the 'Regression on one slice' preset to see it ship" />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.active === "data" ? "Unit tests use fixtures; only checks on the real incoming data catch upstream changes." : s.active === "eval" ? "A gate is only as good as what it measures: averages hide slices." : s.active === "canary" ? "Some failures only exist on the production path: a canary limits who sees them." : undefined} />
          <Panel title="Calculation"><Calc s={s} /></Panel>
          <Panel title="Gates at this stage">
            {s.checks.length === 0 ? <p className="text-xs text-muted">No gate at this stage.</p> : (
              <KV rows={s.checks.map((c) => [c.label, `${fmtCheck(c, c.measured)} ${c.op === "<=" ? "≤" : "≥"} ${fmtCheck(c, c.threshold)} ${c.disabled ? "(off)" : c.pass ? "✓" : "✗"}`])} />
            )}
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Calc({ s }: { s: CIState }) {
  if (s.active === "data") {
    if (s.data.missing.length) return <TexBlock>{`\\{${EXPECTED_COLUMNS.join(",")}\\} \\setminus \\text{columns} = \\{\\text{${s.data.missing.join(",")}}\\} \\neq \\varnothing`}</TexBlock>;
    const w = s.data.psi.reduce((a, b) => ((b.value ?? 0) > (a.value ?? 0) ? b : a));
    const terms = w.base.map((e, i) => `(${w.cur![i]} - ${e})\\ln\\tfrac{${w.cur![i]}}{${e}}`);
    return <TexBlock>{`\\begin{aligned} \\mathrm{PSI}_{\\text{${w.feature}}} &= \\textstyle\\sum_b (a_b - e_b)\\ln\\tfrac{a_b}{e_b} \\\\ &= ${terms.slice(0, 2).join(" + ")} \\\\ &\\quad + \\dots = ${w.value!.toFixed(3)} \\end{aligned}`}</TexBlock>;
  }
  if (s.active === "eval") {
    const base = s.evalD.slices.reduce((a, x) => a + x.base, 0), cand = s.evalD.slices.reduce((a, x) => a + x.cand, 0), n = s.evalD.slices.reduce((a, x) => a + x.n, 0);
    const worst = s.evalD.slices.reduce((a, x) => (x.base - x.cand) / x.n > (a.base - a.cand) / a.n ? x : a);
    return <TexBlock>{`\\begin{aligned} \\Delta\\text{acc} &= \\tfrac{${cand}}{${n}} - \\tfrac{${base}}{${n}} = ${((cand - base) / n).toFixed(4)} \\\\ \\text{drop}_{\\text{${worst.name.replace("–", "-")}}} &= \\tfrac{${worst.base}}{${worst.n}} - \\tfrac{${worst.cand}}{${worst.n}} = ${((worst.base - worst.cand) / worst.n).toFixed(4)} \\\\ p_{95} &= ${s.evalD.p95}\\text{ ms (190th of 200 sorted)} \\end{aligned}`}</TexBlock>;
  }
  if (s.active === "canary" || (s.active === "result" && s.outcome === "rolled-back")) {
    const m = s.canary.minutes;
    const req = m.reduce((a, x) => a + x.requests, 0), err = m.reduce((a, x) => a + x.errors, 0);
    return <TexBlock>{`\\text{error rate} = \\frac{\\sum \\text{errors}}{\\sum \\text{requests}} = \\frac{${err}}{${req}} = ${(err / req * 100).toFixed(2)}\\%`}</TexBlock>;
  }
  if (s.active === "train") return <TexBlock>{`\\frac{L_{20}}{L_1} = \\frac{${s.loss[19]}}{${s.loss[0]}} = ${(s.loss[19] / s.loss[0]).toFixed(3)} \\le 0.6`}</TexBlock>;
  if (s.active === "unit") return <TexBlock>{"\\text{passed} = 48 \\ge 48 \\;(\\text{all})"}</TexBlock>;
  return <TexBlock>{"\\text{ship} \\iff \\bigwedge_{g \\in \\text{gates}} \\big(\\text{measured}_g \\text{ within threshold}_g\\big)"}</TexBlock>;
}

/* ───────────────────────────── canvas ───────────────────────────── */

function Gauge({ c, i, y, k, animKey }: { c: Check; i: number; y: number; k: number; animKey: string }) {
  const X0 = 210, X1 = 600;
  const [lo, hi] = c.scale;
  const sx = (v: number) => X0 + (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (X1 - X0);
  const m = c.measured;
  const color = c.disabled ? "var(--warn)" : c.pass ? "var(--ok)" : "var(--err)";
  const okFrom = c.op === "<=" ? lo : c.threshold, okTo = c.op === "<=" ? c.threshold : hi;
  return (
    <g key={`${c.id}-${animKey}`}>
      <text x={20} y={y + 4} fontSize={11} fontWeight={600} fill="var(--fg)">{c.label}</text>
      {c.detail && <text x={20} y={y + 18} fontSize={9.5} fill="var(--muted)">{c.detail.length > 34 ? `${c.detail.slice(0, 33)}…` : c.detail}</text>}
      <rect x={X0} y={y - 6} width={X1 - X0} height={12} rx={6} fill="var(--surface-2)" stroke="var(--line)" />
      <rect x={sx(okFrom)} y={y - 6} width={Math.max(0, sx(okTo) - sx(okFrom))} height={12} rx={6} fill="var(--ok)" fillOpacity={0.15} />
      <line x1={sx(c.threshold)} x2={sx(c.threshold)} y1={y - 12} y2={y + 12} stroke="var(--fg)" strokeWidth={2} />
      <text x={sx(c.threshold)} y={y + 24} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">limit {fmtCheck(c, c.threshold)}</text>
      {m !== null ? (
        <g transform={`translate(0 ${y})`}>
          <motion.g initial={{ x: sx(lo) }} animate={{ x: sx(m) }} transition={{ delay: T.gauge(i) * k, duration: T.gaugeDur * k, ease }}>
            <path d="M-6 -16 L6 -16 L0 -7 Z" fill={color} />
            <rect x={-1.5} y={-7} width={3} height={14} fill={color} />
          </motion.g>
        </g>
      ) : <text x={(X0 + X1) / 2} y={y + 4} textAnchor="middle" fontSize={10} fill="var(--err)">not computable</text>}
      <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.gauge(i) + T.gaugeDur) * k }}>
        <text x={X1 + 14} y={y + 4} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{fmtCheck(c, m)}</text>
        <text x={X1 + 14} y={y + 18} fontSize={10} fontWeight={600} fill={color}>{c.disabled ? "⚠ gate off" : c.pass ? "✓ pass" : "✗ fail"}</text>
      </motion.g>
    </g>
  );
}

function PipelineSvg({ s, prev, k, animKey, canaryPct }: { s: CIState; prev: StageId | "result" | null; k: number; animKey: string; canaryPct: number }) {
  const at = s.active === "result" ? idx(s.failedAt!) : idx(s.active);
  const from = prev && prev !== "result" ? idx(prev) : Math.max(0, at - 1);
  const failedNow = s.checks.some((c) => !c.pass);
  const tokenColor = s.active === "result" || failedNow ? "var(--err)" : s.statuses[s.active as StageId] === "warning" ? "var(--warn)" : "var(--accent)";
  const nGauges = s.checks.length;
  return (
    <svg viewBox="0 0 760 330" className="h-auto w-full" role="img" aria-label={`ML pipeline, active stage ${s.active}`}>
      {STAGES.slice(1).map((st, i) => (
        <line key={st.id} x1={NX(i) + NWD / 2} x2={NX(i + 1) - NWD / 2} y1={NY} y2={NY} stroke={s.statuses[st.id] === "skipped" ? "var(--line)" : "var(--muted)"} strokeWidth={1.5} strokeDasharray={s.statuses[st.id] === "skipped" ? "3 3" : undefined} />
      ))}
      {STAGES.map((st, i) => {
        const status: Status = i === at && s.active !== "result" && s.statuses[st.id] === "pending" ? "active" : s.statuses[st.id];
        const g = STATUS_GLYPH[status];
        const isAt = i === at;
        return (
          <g key={st.id}>
            {isAt && <motion.rect key={`pulse-${animKey}`} x={NX(i) - NWD / 2 - 4} y={NY - NHT / 2 - 4} width={NWD + 8} height={NHT + 8} rx={12} fill="none" stroke={g.color} strokeWidth={2}
              initial={{ opacity: 0 }} animate={{ opacity: [0, 0.8, 0.2, 0.8] }} transition={{ delay: T.travel * k, duration: 1.6 * k }} />}
            <motion.rect x={NX(i) - NWD / 2} y={NY - NHT / 2} width={NWD} height={NHT} rx={9} fill="var(--surface)" initial={false}
              animate={{ stroke: status === "pending" || status === "skipped" ? "var(--line)" : g.color }} strokeWidth={isAt ? 2.5 : 1.5} strokeDasharray={status === "skipped" ? "4 3" : undefined}
              transition={{ delay: isAt ? T.verdict(nGauges) * k * 0.6 : 0, duration: 0.3 }} />
            <text x={NX(i)} y={NY - 6} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={status === "skipped" ? "var(--muted)" : "var(--fg)"}>{st.label}</text>
            <text x={NX(i)} y={NY + 12} textAnchor="middle" fontSize={10} fill={g.color}>{g.g} {g.text}</text>
          </g>
        );
      })}
      {/* travelling token */}
      <motion.circle key={`tok-${animKey}`} r={8} cy={NY - NHT / 2 - 12} fill={tokenColor} stroke="var(--surface)" strokeWidth={2}
        initial={{ cx: NX(from) }} animate={{ cx: NX(at) }} transition={{ duration: T.travel * k, ease: "easeInOut" }} />

      {/* stage-specific lower half */}
      {nGauges > 0 && (
        <g>
          <text x={20} y={140} fontSize={11} fontWeight={600} fill="var(--muted)">{STAGES[at].label.toUpperCase()}: MEASURED vs THRESHOLD</text>
          {s.checks.map((c, i) => <Gauge key={`${c.id}-${animKey}`} c={c} i={i} y={178 + i * 52} k={k} animKey={animKey} />)}
        </g>
      )}
      {s.active === "commit" && (
        <motion.g key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 * k }}>
          <rect x={20} y={130} width={330} height={70} rx={8} fill="var(--surface-2)" stroke="var(--line)" />
          <text x={34} y={154} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">git push origin main</text>
          <text x={34} y={176} fontSize={10.5} fill="var(--muted)">→ webhook starts pipeline run #{1042}</text>
          <text x={34} y={192} fontSize={10.5} fill="var(--muted)">7 stages, each must pass to continue</text>
        </motion.g>
      )}
      {s.active === "canary" && (
        <g key={`split-${animKey}`}>
          <text x={20} y={252} fontSize={11} fontWeight={600} fill="var(--muted)">LIVE TRAFFIC SPLIT</text>
          <rect x={20} y={262} width={720} height={26} rx={6} fill="var(--surface-2)" stroke="var(--line)" />
          <motion.rect x={20} y={262} height={26} rx={6} fill={s.canary.rollbackAt ? "var(--err)" : "var(--accent)"} fillOpacity={0.7}
            initial={{ width: 0 }} animate={{ width: (720 * canaryPct) / 100 }} transition={{ delay: T.travel * k, duration: 0.8 * k, ease }} />
          <text x={730} y={279} textAnchor="end" fontSize={11} fill="var(--fg)">baseline {100 - canaryPct}%</text>
          <motion.text x={26 + (720 * canaryPct) / 100} y={306} fontSize={11} fontWeight={600} fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.travel + 0.8) * k }}>
            ↑ candidate {canaryPct}% ({Math.round(10 * canaryPct)} req/min)
          </motion.text>
        </g>
      )}
      {(s.active === "promote" || s.active === "result") && (
        <TrafficBar s={s} k={k} animKey={animKey} canaryPct={canaryPct} />
      )}
    </svg>
  );
}

function TrafficBar({ s, k, animKey, canaryPct }: { s: CIState; k: number; animKey: string; canaryPct: number }) {
  const X0 = 20, X1 = 740, Y = 190;
  const promoted = s.active === "promote";
  const rolled = s.outcome === "rolled-back";
  const startPct = promoted ? canaryPct : rolled ? canaryPct : 0;
  const endPct = promoted ? 100 : 0;
  const w = (p: number) => ((X1 - X0) * p) / 100;
  return (
    <g key={animKey}>
      <text x={X0} y={150} fontSize={11} fontWeight={600} fill="var(--muted)">LIVE TRAFFIC SPLIT</text>
      <rect x={X0} y={Y - 16} width={X1 - X0} height={32} rx={6} fill="var(--surface-2)" stroke="var(--line)" />
      <motion.rect x={X0} y={Y - 16} height={32} rx={6} fill={s.outcome === "promoted-with-regression" ? "var(--warn)" : "var(--accent)"} fillOpacity={0.75}
        initial={{ width: w(startPct) }} animate={{ width: w(endPct) }} transition={{ delay: 0.6 * k, duration: 1.4 * k, ease }} />
      <text x={X0 + 10} y={Y + 4} fontSize={11} fontWeight={600} fill="var(--fg)">{promoted ? "candidate" : "baseline (current production)"}</text>
      <text x={X1 - 10} y={Y + 4} textAnchor="end" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">
        candidate <AnimatedNumber svg value={endPct} from={startPct} digits={0} delay={0.6 * k} duration={1.4 * k} />%
      </text>
      <motion.text x={X0} y={Y + 46} fontSize={11} fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.1 * k }}>
        {promoted ? (s.outcome === "promoted-with-regression" ? "⚠ Shipped to everyone with an unchecked slice regression." : "✓ Candidate is the new production model.")
          : rolled ? `↩ Rolled back: baseline serves 100%. Users exposed: ${canaryPct}% for ${s.canary.rollbackAt} min.` : "✗ Nothing deployed: users never saw the candidate."}
      </motion.text>
    </g>
  );
}

/* ───────────────────────────── step card ───────────────────────────── */

function Bins({ base, cur }: { base: number[]; cur: number[] | null }) {
  return (
    <div className="flex h-8 items-end gap-0.5">
      {base.map((b, i) => (
        <div key={i} className="flex items-end gap-px">
          <div className="w-1.5 rounded-t bg-muted/50" style={{ height: `${b * 70}px` }} />
          {cur && <motion.div className="w-1.5 rounded-t bg-accent" initial={{ height: 0 }} animate={{ height: `${cur[i] * 70}px` }} transition={{ delay: 0.3 + i * 0.08, duration: 0.4 }} />}
        </div>
      ))}
    </div>
  );
}

function StepDetail({ s, k, th }: { s: CIState; k: number; th: Thresholds }) {
  switch (s.active) {
    case "commit":
      return (
        <div className="flex flex-wrap gap-1.5">
          {STAGES.map((st, i) => <motion.span key={st.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.15) * k }}><Badge>{i + 1}. {st.label}</Badge></motion.span>)}
        </div>
      );
    case "unit":
      return (
        <div className="space-y-2">
          <div className="grid w-fit grid-cols-[repeat(16,minmax(0,1fr))] gap-1">
            {Array.from({ length: 48 }, (_, i) => <motion.span key={i} className="block h-3 w-3 rounded-sm bg-ok" initial={{ opacity: 0.15, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (0.3 + i * 0.03) * k, duration: 0.2 * k }} />)}
          </div>
          <p className="text-xs text-muted">48 / 48 passed. These tests use fixed fixtures, so an upstream data change cannot fail them.</p>
        </div>
      );
    case "data":
      return (
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <h4 className="mb-1.5 text-sm font-semibold">Schema: expected vs received</h4>
            <div className="flex flex-wrap gap-1">
              {EXPECTED_COLUMNS.map((c, i) => (
                <motion.span key={c} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.12) * k }}>
                  <Badge tone={s.data.missing.includes(c) ? "err" : "ok"}>{s.data.missing.includes(c) ? "✗" : "✓"} {c}</Badge>
                </motion.span>
              ))}
              {s.data.extra.map((c) => <Badge key={c} tone="warn">+ {c} (unexpected)</Badge>)}
            </div>
          </div>
          <div>
            <h4 className="mb-1.5 text-sm font-semibold">Drift per feature (grey = reference, blue = new)</h4>
            <div className="space-y-1.5">
              {s.data.psi.map((p) => (
                <div key={p.feature} className="grid grid-cols-[60px_100px_minmax(0,1fr)] items-end gap-2 text-xs">
                  <span className="font-mono">{p.feature}</span><Bins base={p.base} cur={p.cur} />
                  <span className={cx("font-mono", p.value !== null && p.value > th.maxPsi && "text-err")}>{p.value === null ? "n/a (missing)" : `PSI ${p.value.toFixed(3)}`}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    case "train": {
      const L = s.loss, W = 300, H = 110;
      const d = L.map((v, i) => `${i ? "L" : "M"}${(10 + (i / 19) * (W - 20)).toFixed(1)} ${(H - 10 - ((v - 0.2) / 0.55) * (H - 20)).toFixed(1)}`).join(" ");
      return (
        <div className="flex flex-wrap items-center gap-4">
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full max-w-[300px]" role="img" aria-label="Training loss curve">
            <line x1={10} x2={W - 10} y1={H - 10} y2={H - 10} stroke="var(--line)" />
            <motion.path d={d} fill="none" stroke="var(--violet)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.2 * k, duration: 1.6 * k, ease: "linear" }} />
            <text x={W - 10} y={H - 14} textAnchor="end" fontSize={10} fill="var(--muted)">epoch 20</text>
          </svg>
          <p className="max-w-xs text-sm">Loss <span className="font-mono"><AnimatedNumber value={L[19]} from={L[0]} digits={3} delay={0.2 * k} duration={1.6 * k} /></span> after 20 epochs. A finished training job is a candidate, not a release.</p>
        </div>
      );
    }
    case "eval":
      return (
        <div className="space-y-1.5">
          <h4 className="text-sm font-semibold">Accuracy per slice: baseline (grey) vs candidate (colour)</h4>
          {s.evalD.slices.map((x, i) => {
            const b = x.base / x.n, c = x.cand / x.n;
            const drop = b - c;
            const bad = drop > th.maxSliceDrop;
            const pct = (v: number) => `${((v - 0.7) / 0.25) * 100}%`;
            return (
              <div key={x.name} className="grid grid-cols-[90px_minmax(0,1fr)_110px] items-center gap-2 text-xs">
                <span>{x.name} <span className="text-muted">n={x.n}</span></span>
                <div className="space-y-0.5">
                  <motion.div className="h-2 rounded bg-muted/50" initial={{ width: 0 }} animate={{ width: pct(b) }} transition={{ delay: (0.2 + i * 0.25) * k, duration: 0.5 * k, ease }} />
                  <motion.div className={cx("h-2 rounded", bad ? "bg-err" : "bg-ok")} initial={{ width: 0 }} animate={{ width: pct(c) }} transition={{ delay: (0.35 + i * 0.25) * k, duration: 0.5 * k, ease }} />
                </div>
                <span className={cx("font-mono", bad && "text-err")}>{b.toFixed(3)} → {c.toFixed(3)}{bad && " ✗"}</span>
              </div>
            );
          })}
          <p className="text-xs text-muted">Overall: {s.evalD.baseAcc.toFixed(3)} → {s.evalD.candAcc.toFixed(3)}. p95 latency on 200 replayed requests: {s.evalD.p95} ms.</p>
        </div>
      );
    case "canary":
    case "result":
    case "promote": {
      if (s.active !== "canary" && s.outcome !== "rolled-back") {
        return s.active === "promote"
          ? <p className="text-sm">{s.outcome === "promoted-with-regression" ? "⚠ The slice gate was off, so customers aged 65+ now get 9 points worse predictions. Monitoring by slice in production is the last line of defence." : "✓ The registry's production alias moves to the candidate; the previous model stays registered for instant rollback."}</p>
          : <div className="flex flex-wrap gap-1.5">{STAGES.map((st) => <StatusBadge key={st.id} status={s.statuses[st.id] === "skipped" ? "pending" : s.statuses[st.id] === "error" ? "error" : s.statuses[st.id] === "warning" ? "warning" : "ok"} label={`${st.label}: ${STATUS_GLYPH[s.statuses[st.id]].text}`} />)}</div>;
      }
      const m = s.canary.minutes;
      const max = Math.max(0.06, ...m.map((x) => x.cumRate));
      return (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold">Candidate's cumulative error rate, minute by minute ({th.canaryPct}% of 1,000 req/min)</h4>
          <div className="relative flex h-28 items-end gap-1 border-b border-line">
            <div className="absolute inset-x-0 border-t-2 border-dashed border-err" style={{ bottom: `${(th.maxCanaryErr / max) * 100}%` }}><span className="absolute -top-4 right-0 text-[10px] text-err">limit {(th.maxCanaryErr * 100).toFixed(1)}%</span></div>
            {Array.from({ length: 10 }, (_, i) => {
              const x = m[i];
              return (
                <div key={i} className="flex h-full flex-1 flex-col items-center justify-end">
                  {x ? <motion.div className={cx("w-full rounded-t", x.cumRate > th.maxCanaryErr ? "bg-err/80" : "bg-ok/70")} initial={{ height: 0 }} animate={{ height: `${(x.cumRate / max) * 100}%` }} transition={{ delay: (0.3 + i * 0.3) * k, duration: 0.3 * k }} />
                    : <div className="w-full rounded-t border border-dashed border-line" style={{ height: "4px" }} />}
                </div>
              );
            })}
          </div>
          <div className="flex gap-1 font-mono text-[10px] text-muted">{Array.from({ length: 10 }, (_, i) => <span key={i} className="flex-1 text-center">{i + 1}</span>)}</div>
          <p className="text-xs">{s.canary.rollbackAt ? `✗ Minute ${s.canary.rollbackAt}: ${(m.at(-1)!.cumRate * 100).toFixed(2)}% > ${(th.maxCanaryErr * 100).toFixed(1)}% → rollback (judged after ≥ ${s.canary.minRequests} requests). Baseline error rate: ${(m.at(-1)!.baseRate * 100).toFixed(2)}%.` : `✓ 10 minutes at ${(m.at(-1)!.cumRate * 100).toFixed(2)}%, baseline ${(m.at(-1)!.baseRate * 100).toFixed(2)}%.`}</p>
        </div>
      );
    }
  }
}
