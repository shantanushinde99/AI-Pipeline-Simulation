import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { StepCaption, ease } from "../offline-evaluation/kit";
import type { VisualizerTab } from "../types";
import { CASES, simulate, type CaseId, type FailureCase, type RFPhase, type Ranking } from "./simulation";

const ROW_H = 28, SHOW = 10;
/** One schedule (seconds at 1×): rows settle, then the needed-chunk marker slides, then the score bars count. */
const T = { marker: 0.5, slide: 1.4, bars: 0.4, facts: 0.35 };
const DUR: Record<RFPhase, number> = { healthy: 3200, broken: 3800, diagnose: 4200, fixed: 3800, summary: 4200 };

const slotOf = (r: Ranking) => (r.relStatus !== "ranked" ? SHOW + 1 : Math.min(r.relRank!, SHOW + 1) - 1);
const markerLabel = (r: Ranking) => (r.relStatus === "ranked" ? `#${r.relRank}` : r.relStatus === "excluded" ? "filtered" : "missing");

function narrate(phase: RFPhase, c: FailureCase, k: number, inK: (r: Ranking) => boolean) {
  if (phase === "healthy") return { headline: `Healthy: the chunk with “${c.fact}” ranks ${markerLabel(c.healthy)}`, sub: `It is inside the top-${k}, so the generator gets the evidence it needs.` };
  if (phase === "broken") return { headline: `Break it: ${c.fault}`, sub: inK(c.broken) ? "The needed chunk slips but survives the cut." : `Watch the marker slide out of the top-${k}: the answer will be wrong or “not found”.` };
  if (phase === "diagnose") return { headline: "Diagnose: why did it fall?", sub: c.diagnosticText };
  if (phase === "fixed") return { headline: `Fix: ${c.fixLabel}`, sub: inK(c.fixed) ? `The needed chunk climbs back to ${markerLabel(c.fixed)}.` : `Better, but still outside the top-${k}.` };
  return { headline: `${c.title}: symptom → diagnostic → fix`, sub: "The pattern to remember when this shows up in production." };
}

export default function RetrievalFailuresVisualizer({ tab }: { tab: VisualizerTab }) {
  void tab;
  const [caseId, setCaseId] = useState<CaseId>("vocab");
  const [topK, setTopK] = useState(3);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(caseId, topK), [caseId, topK]);
  const { c, k: K, steps, inK, where } = sim;
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "healthy"]);
  const step = steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${caseId}-${step.id}-${replay}`;
  const phase = step.state.phase;
  const now = c[step.state.ranking];
  const prev = phase === "broken" ? c.healthy : phase === "fixed" ? c.broken : now;
  const story = narrate(phase, c, K, inK);
  const cut = now.rows[K - 1]?.score ?? 0;

  return (
    <Workspace
      main={
        <>
          <Panel title="Ranked chunks for the query" actions={<SimulatedLabel>toy hashed embeddings</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={animKey} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <p className="mb-2 rounded-md bg-surface-2 px-2 py-1 font-mono text-xs">q = “{now.query}” <span className="text-muted">· {now.label} · {now.model}</span></p>
            <RankList r={now} prev={prev} K={K} k={k} animKey={animKey} />
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                {phase === "diagnose" ? <Diagnose c={c} k={k} /> : phase === "summary" ? <Summary c={c} k={k} where={where} /> : <ScoreRace r={now} cut={cut} K={K} k={k} ok={inK(now)} />}
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Failure case & parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setCaseId("vocab"); setTopK(3); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Failure case" value={caseId} options={CASES.map((x) => ({ value: x.id, label: x.title }))} onChange={(v) => { setCaseId(v); pb.restart(); }} />
              <Slider label="top-k sent to the generator" value={topK} min={1} max={5} onChange={setTopK} hint="Small k makes every rank slip fatal; large k hides it until context overflows." />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {CASES.map((x) => (
                <Button key={x.id} size="sm" variant={x.id === caseId ? "primary" : "secondary"} onClick={() => { setCaseId(x.id); pb.restart(); }}>{x.title}</Button>
              ))}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation">
            <TexBlock>{`s(q, c) = \\frac{q \\cdot c}{\\lVert q \\rVert\\,\\lVert c \\rVert} = \\hat q \\cdot \\hat c`}</TexBlock>
            <div className="mt-2">
              <TexBlock>{now.relScore === null
                ? `c^{*} \\notin \\text{ranked pool} \\Rightarrow \\text{never retrieved}`
                : `s(q, c^{*}) = ${now.relScore.toFixed(3)} \\;${now.relScore >= cut && inK(now) ? "\\ge" : "<"}\\; s_{(${K})} = ${cut.toFixed(3)}`}</TexBlock>
            </div>
            <p className="mt-1 text-[11px] text-muted"><span className="font-mono">c*</span> is the chunk containing “{c.fact}”; <span className="font-mono">s₍ₖ₎</span> is the score of the last chunk that makes the top-{K}.</p>
          </Panel>
          <Panel title="Key values">
            <KV rows={[["case", c.title], ["healthy rank", where(c.healthy)], ["broken rank", where(c.broken)], ["fixed rank", where(c.fixed)], [`in top-${K}`, `${inK(c.healthy) ? "yes" : "no"} → ${inK(c.broken) ? "yes" : "no"} → ${inK(c.fixed) ? "yes" : "no"}`]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function RankList({ r, prev, K, k, animKey }: { r: Ranking; prev: Ranking; K: number; k: number; animKey: string }) {
  const rows = r.rows.slice(0, SHOW);
  const more = r.rows.length - rows.length;
  return (
    <div className="relative pl-16" style={{ height: (SHOW + 2) * ROW_H }}>
      <AnimatePresence initial={false}>
        {rows.map((row, i) => (
          <motion.div key={row.key} layout initial={{ opacity: 0 }} animate={{ opacity: i < K ? 1 : 0.55 }} exit={{ opacity: 0 }} transition={{ duration: 0.5 * k, ease }}
            className={cx("absolute right-0 left-16 flex items-center gap-2 rounded px-2 text-xs", row.relevant ? "bg-accent-soft ring-1 ring-accent" : i < K ? "bg-surface-2" : "")}
            style={{ top: i * ROW_H, height: ROW_H - 3 }}>
            <span className="w-6 shrink-0 font-mono text-muted">#{i + 1}</span>
            <span className="w-28 shrink-0 truncate font-mono sm:w-40">{row.id}</span>
            <span className="min-w-0 flex-1 truncate text-muted">{row.text}</span>
            <span className="shrink-0 font-mono">{row.score.toFixed(3)}</span>
          </motion.div>
        ))}
      </AnimatePresence>
      <div className="absolute right-0 left-16 px-2 text-xs text-muted" style={{ top: SHOW * ROW_H + 5 }}>{more > 0 ? `… ${more} more ranked chunk${more > 1 ? "s" : ""}` : ""}</div>
      <div className="absolute right-0 left-16 px-2 text-xs text-muted" style={{ top: (SHOW + 1) * ROW_H + 5 }}>
        {r.excluded.length ? `${r.excluded.length} chunk(s) removed before scoring` : "not ranked / not indexed"}
      </div>
      <motion.div className="absolute right-0 left-14 border-t-2 border-dashed border-warn" initial={false} animate={{ top: K * ROW_H - 2 }} transition={{ duration: 0.4 }}>
        <span className="absolute -top-2 right-1 rounded bg-warn px-1 text-[10px] font-semibold text-white dark:text-black">top-{K} cut</span>
      </motion.div>
      <motion.div key={animKey} className="absolute left-0 flex items-center" style={{ height: ROW_H - 3 }}
        initial={{ y: slotOf(prev) * ROW_H }} animate={{ y: slotOf(r) * ROW_H }} transition={{ delay: T.marker * k, duration: T.slide * k, ease }}>
        <span className={cx("flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[11px] font-semibold text-white dark:text-black", r.relStatus === "ranked" && (r.relRank ?? 99) <= K ? "bg-ok" : "bg-err")}>
          {r.relStatus === "ranked" && (r.relRank ?? 99) <= K ? <CheckCircle2 size={11} /> : <XCircle size={11} />}{markerLabel(r)}
        </span>
      </motion.div>
    </div>
  );
}

function ScoreRace({ r, cut, K, k, ok }: { r: Ranking; cut: number; K: number; k: number; ok: boolean }) {
  if (r.relScore === null)
    return <p className="text-sm">The needed chunk is <strong>{r.relStatus === "excluded" ? "removed by a filter before scoring" : "not in the index at all"}</strong>. No similarity score can rescue it: <XCircle size={13} className="inline text-err" /> the generator will answer from other chunks or say it can't find the answer.</p>;
  const max = Math.max(r.rows[0]?.score ?? 1, 1e-6);
  const bars = [{ label: "needed chunk c*", v: r.relScore, cls: "bg-accent" }, { label: `#${K} (the cut)`, v: cut, cls: "bg-warn/70" }, { label: "#1 (best)", v: r.rows[0].score, cls: "bg-muted/50" }];
  return (
    <div className="space-y-2">
      {bars.map((b, i) => (
        <div key={b.label} className="grid grid-cols-[110px_minmax(0,1fr)_56px] items-center gap-2 text-xs">
          <span>{b.label}</span>
          <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-full rounded", b.cls)} initial={{ width: 0 }} animate={{ width: `${(Math.max(0, b.v) / max) * 100}%` }} transition={{ delay: (T.bars + i * 0.2) * k, duration: 0.7 * k, ease }} /></div>
          <span className="text-right font-mono"><AnimatedNumber value={b.v} from={0} digits={3} delay={(T.bars + i * 0.2) * k} duration={0.7 * k} /></span>
        </div>
      ))}
      <motion.p className="text-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.marker + T.slide) * k }}>
        {ok ? <><CheckCircle2 size={13} className="inline text-ok" /> c* scores at least the cut, so it is in the top-{K} at rank #{r.relRank}.</> : <><XCircle size={13} className="inline text-err" /> c* scores below the #{K} chunk, so it lands at #{r.relRank} and is cut.</>}
      </motion.p>
    </div>
  );
}

function Diagnose({ c, k }: { c: FailureCase; k: number }) {
  const t = c.diag.tokens;
  return (
    <div className="space-y-3">
      <div className="space-y-1">
        {c.diag.facts.map(([key, v, bad], i) => (
          <motion.div key={key} className="grid grid-cols-[minmax(0,140px)_minmax(0,1fr)] gap-2 border-b border-line py-1 text-xs" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * T.facts) * k, duration: 0.3 * k }}>
            <span className="text-muted">{key}</span>
            <span className={cx("break-words font-mono", bad && "text-warn")}>{bad && "⚠ "}{v}</span>
          </motion.div>
        ))}
      </div>
      {t && (
        <div className="text-xs">
          <p className="mb-1 font-semibold">Term overlap between the query and {t.chunkId} (shared terms glow)</p>
          {[["query", t.query], ["chunk", t.chunk]].map(([label, toks], row) => (
            <div key={String(label)} className="mb-1 flex flex-wrap items-center gap-1">
              <span className="w-12 text-muted">{label}</span>
              {(toks as string[]).map((w, i) => {
                const shared = t.shared.includes(w);
                return <motion.span key={w + i} className={cx("rounded border px-1 font-mono", shared ? "border-ok bg-ok-soft" : "border-line")} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (1 + row * 0.4 + i * 0.03) * k }}>{w}</motion.span>;
              })}
            </div>
          ))}
          <p className="text-muted">{t.shared.length} shared term{t.shared.length === 1 ? "" : "s"}: {t.shared.join(", ") || "none"}.</p>
        </div>
      )}
    </div>
  );
}

function Summary({ c, k, where }: { c: FailureCase; k: number; where: (r: Ranking) => string }) {
  const cards = [["Symptom", c.symptom, "warn"], ["Diagnostic", c.diagnosticText, "accent"], ["Fix", c.fixText, "ok"]] as const;
  return (
    <div className="space-y-3">
      <div className="grid gap-2 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-stretch">
        {cards.map(([h, body, tone], i) => (
          <motion.div key={h} className="contents" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.8 * k }}>
            {i > 0 && <ArrowRight size={16} className="hidden self-center text-muted md:block" />}
            <motion.div className="rounded-md border border-line p-2 text-xs" initial={{ y: 12 }} animate={{ y: 0 }} transition={{ delay: i * 0.8 * k, duration: 0.4 * k, ease }}>
              <Badge tone={tone}>{h}</Badge>
              <p className="mt-1 leading-relaxed">{body}</p>
            </motion.div>
          </motion.div>
        ))}
      </div>
      <p className="font-mono text-xs">rank of c*: {where(c.healthy)} → <span className="text-err">{where(c.broken)}</span> → <span className="text-ok">{where(c.fixed)}</span></p>
    </div>
  );
}
