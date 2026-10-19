import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, StepCard, ease } from "./kit";
import { DEFAULT_MV, GAPS, LRS, SNAPSHOTS, simulateRegistry, type Gap, type MVConfig, type MVPhase, type MVState, type ModelVersion } from "./simulation";

/** One timing schedule (seconds at 1×) shared by the canvas, the step card and the counters. */
const T = {
  snapshot: { node: 0.1, hashChar: (i: number) => 0.5 + i * 0.09 },
  train: { toRun: 0.2, travel: 0.7, row: (i: number) => 0.9 + i * 0.28, toModel: 2.4, model: 3.1 },
  alias: { slide: 0.2, edge: 1.0 },
  promote: { bars: 0.2, verdict: 1.5, slide: 1.9, edge: 2.7 },
  repro: { data: 0.2, toRun: 0.5, toModel: 1.5, char: (i: number) => 2.4 + i * 0.13, verdict: 3.5 },
  audit: { hop: (i: number) => 0.2 + i * 0.5 },
};
const DURATION: Record<MVPhase, number> = { snapshot: 2200, train: 4400, alias: 2400, promote: 4000, repro: 4700, audit: 3000 };

const COL = { data: 90, run: 270, model: 450, serve: 670 };
const ROW = [100, 220] as const;
const REPRO_Y = 330;
const NW = 120, NH = 60;

function narrate(s: MVState, gap: Gap): { headline: string; sub: string } {
  switch (s.phase) {
    case "snapshot":
      return s.focusRow === 0
        ? { headline: "Freeze the training data and give it a fingerprint", sub: "The hash is computed from the rows themselves, so the same rows always give the same id." }
        : gap === "unpinned-data"
          ? { headline: "Run 2's config points at 'latest', a pointer that will move", sub: "Today it resolves to the October rows, but nothing records which rows those were." }
          : { headline: "A newer snapshot is frozen with its own hash", sub: "Old snapshots are never overwritten, so v1 can still be traced to its exact rows." };
    case "train":
      return { headline: `A training run turns data + code + settings into model v${s.focusRow + 1}`, sub: "The registry stores the weights together with the lineage that produced them." };
    case "alias":
      return { headline: `Alias '${s.moved?.alias}' now points at v${s.moved?.to}`, sub: "Aliases are movable labels. Deploying = moving a pointer, not copying a model." };
    case "promote":
      return s.gate?.pass
        ? { headline: "v2 clears the gate, so 'production' slides from v1 to v2", sub: "The serving endpoint follows the alias and reloads. Rolling back would be the same move in reverse." }
        : { headline: "v2 is not better enough, so 'production' stays on v1", sub: "Lower the required gain or change v2's hyperparameters to see it promoted." };
    case "repro":
      return s.repro?.match
        ? { headline: "Retrain v2 from its recorded lineage: the hash matches", sub: "Every input was recorded exactly, so the result is bit-for-bit the same model." }
        : { headline: "Retrain v2 from its recorded lineage: a different model comes out", sub: `What was recorded is not what ran (${s.repro?.diffs.join(", ")}). v2 can no longer be rebuilt.` };
    case "audit":
      return { headline: "Walk the lineage backwards from the endpoint", sub: "Endpoint → alias → version → run → data and code: every link is a recorded id." };
  }
}

export default function ModelVersioningVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<MVConfig>(DEFAULT_MV);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulateRegistry(cfg), [cfg]);
  const pb = usePlayback(steps.length, (i) => DURATION[steps[i]?.state.phase ?? "alias"]);
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(s, cfg.gap);
  const set = (p: Partial<MVConfig>) => setCfg((c) => ({ ...c, ...p }));
  if (tab === "metrics") return null;
  const m2 = s.models[1];

  return (
    <Workspace
      main={
        <>
          <Panel title="Model registry & lineage" actions={<SimulatedLabel>Simulated training & hashes</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[560px]"><RegistrySvg s={s} k={k} animKey={animKey} pinned2={cfg.gap !== "unpinned-data"} /></div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>lineage flowing into a run / model</span>
              <span className="flex items-center gap-1"><Badge tone="warn">staging</Badge><Badge tone="ok">production</Badge> movable aliases</span>
              <span>dashed = reproduction attempt</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <StepCard animKey={animKey}><StepDetail s={s} k={k} pinned2={cfg.gap !== "unpinned-data"} /></StepCard>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Run 2 settings & lineage faults" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT_MV)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="v2 learning rate" value={String(cfg.v2.lr)} options={LRS.map((v) => ({ value: String(v), label: String(v) }))} onChange={(v) => set({ v2: { ...cfg.v2, lr: Number(v) } })} />
              <Slider label="v2 tree depth" value={cfg.v2.depth} min={2} max={10} onChange={(v) => set({ v2: { ...cfg.v2, depth: v } })} />
              <Slider label="Promotion gate: v2 must beat v1 by" value={cfg.minGain} min={0} max={0.08} step={0.005} format={(v) => `+${(v * 100).toFixed(1)} pts`} onChange={(v) => set({ minGain: v })} />
              <Select label="Break it: lineage gap in run 2" value={cfg.gap} options={(Object.keys(GAPS) as Gap[]).map((g) => ({ value: g, label: GAPS[g].label }))} onChange={(g) => set({ gap: g })} />
            </div>
            <p className="mt-2 text-xs text-muted">{GAPS[cfg.gap].note} Jump to the reproducibility step to see the consequence.</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.phase === "repro" ? "Audits, debugging and rollbacks all assume you can rebuild or at least explain the exact model that served a prediction." : s.phase === "alias" || s.phase === "promote" ? "Because serving resolves an alias, promotion and rollback are instant, atomic and logged." : undefined} />
          <Panel title="Calculation"><Calc s={s} /></Panel>
          <Panel title="Registry state">
            <KV rows={[
              ["versions", s.models.map((m) => `v${m.version}`).join(", ") || "none"],
              ["staging →", s.aliases.staging ? `v${s.aliases.staging}` : "unset"],
              ["production →", s.aliases.production ? `v${s.aliases.production}` : "unset"],
              ["v2 accuracy", m2 ? m2.accuracy.toFixed(4) : "not trained"],
              ["lineage gap", cfg.gap === "none" ? "none" : GAPS[cfg.gap].label],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Calc({ s }: { s: MVState }) {
  const f = s.models[s.focusRow];
  if (s.phase === "snapshot") {
    const sn = s.snapshots[s.focusRow];
    return <TexBlock>{`d = H(\\text{rows}_{\\le ${sn.date}}) = \\texttt{${sn.hash}}`}</TexBlock>;
  }
  if (s.phase === "train" && f) {
    const hp = f.lineage.hparams;
    const n = s.runs[s.focusRow].rows;
    return (
      <div className="space-y-2">
        <TexBlock>{`h_{v${f.version}} = H(d \\Vert c \\Vert \\theta \\Vert s) = \\texttt{${f.hash}}`}</TexBlock>
        <TexBlock>{`\\begin{aligned} \\text{acc} &= 0.84 + 0.03\\log_2\\tfrac{n}{12000} - 0.08(\\log_{10}\\eta + 2)^2 \\\\ &\\quad - 0.004(\\text{depth} - 6)^2 + \\epsilon_s \\\\ &= 0.84 + 0.03\\log_2\\tfrac{${n}}{12000} - 0.08(\\log_{10}${hp.lr} + 2)^2 \\\\ &\\quad - 0.004(${hp.depth} - 6)^2 + \\epsilon_{${s.runs[s.focusRow].actualSeed}} = ${f.accuracy.toFixed(4)} \\end{aligned}`}</TexBlock>
        <p className="text-[11px] text-muted">A made-up response surface standing in for real training; ε is a seeded jitter of at most ±0.005.</p>
      </div>
    );
  }
  if (s.phase === "promote" && s.gate) {
    const g = s.gate;
    return <TexBlock>{`\\text{acc}(v_2) \\ge \\text{acc}(v_1) + \\Delta_{\\min} \\\\ ${g.v2.toFixed(4)} \\ge ${g.v1.toFixed(4)} + ${g.minGain} = ${(g.v1 + g.minGain).toFixed(4)} \\Rightarrow \\text{${g.pass ? "promote" : "block"}}`}</TexBlock>;
  }
  if (s.phase === "repro" && s.repro) {
    const r = s.repro;
    return <TexBlock>{`\\begin{aligned} H(\\text{recorded lineage}) &= \\texttt{${r.hash}} \\\\ h_{v2} &= \\texttt{${s.models[1].hash}} \\\\ \\Rightarrow\\; &\\text{${r.match ? "equal: reproducible" : "different: irreproducible"}} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "audit") {
    const v = s.aliases.production ?? 1;
    const m = s.models[v - 1];
    return <TexBlock>{`\\begin{aligned} &\\text{endpoint} \\to \\text{production} \\to v_${v} \\\\ &\\to \\text{${m.runId}} \\to (\\text{data } \\texttt{${m.lineage.dataHash}}, \\text{ code } \\texttt{${m.lineage.commit}}) \\end{aligned}`}</TexBlock>;
  }
  return <TexBlock>{`\\text{${s.moved?.alias ?? "alias"}} \\mapsto v_{${s.moved?.to ?? "?"}}`}</TexBlock>;
}

/* ───────────────────────────── canvas ───────────────────────────── */

function NodeBox({ x, y, title, lines, tone = "neutral", dashed, delay = 0, k, glow }: {
  x: number; y: number; title: string; lines: string[]; tone?: "neutral" | "accent" | "ok" | "err" | "warn"; dashed?: boolean; delay?: number; k: number; glow?: boolean;
}) {
  const stroke = tone === "neutral" ? "var(--line)" : `var(--${tone})`;
  return (
    <motion.g initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: delay * k, duration: 0.45 * k, ease }} style={{ transformOrigin: `${x}px ${y}px` }}>
      <rect x={x - NW / 2} y={y - NH / 2} width={NW} height={NH} rx={8} fill="var(--surface)" stroke={stroke} strokeWidth={glow ? 2.5 : 1.5} strokeDasharray={dashed ? "5 4" : undefined} />
      <text x={x} y={y - 12} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--fg)">{title}</text>
      {lines.map((l, i) => <text key={i} x={x} y={y + 4 + i * 14} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={i === 0 ? "var(--fg)" : "var(--muted)"}>{l}</text>)}
    </motion.g>
  );
}

function Token({ from, to, y, delay, k, color = "var(--accent)" }: { from: number; to: number; y: number; delay: number; k: number; color?: string }) {
  return (
    <motion.circle r={7} cy={y} fill={color} stroke="var(--surface)" strokeWidth={2} initial={{ cx: from, opacity: 0 }}
      animate={{ cx: [from, to], opacity: [0, 1, 1, 0] }} transition={{ delay: delay * k, duration: T.train.travel * k, ease: "easeInOut", opacity: { delay: delay * k, duration: T.train.travel * k, times: [0, 0.1, 0.85, 1] } }} />
  );
}

function RegistrySvg({ s, k, animKey, pinned2 }: { s: MVState; k: number; animKey: string; pinned2: boolean }) {
  const fresh = (row: number) => s.focusRow === row; // node created at this step animates in
  const prod = s.aliases.production;
  const auditHop = (i: number) => (s.phase === "audit" ? T.audit.hop(i) : 0);
  const prodRow = prod ? ROW[prod - 1] : ROW[0];
  const aliasDelay = (a: "staging" | "production") => (s.moved?.alias === a ? (s.phase === "promote" ? T.promote.slide : T.alias.slide) : 0);
  const edgeDelay = s.phase === "promote" ? T.promote.edge : T.alias.edge;
  const isAudit = s.phase === "audit";
  return (
    <svg viewBox="0 0 760 380" className="h-auto w-full" role="img" aria-label={`Lineage graph: ${s.snapshots.length} snapshots, ${s.models.length} model versions, production alias on ${prod ? `v${prod}` : "nothing"}`}>
      {[["Data snapshot", COL.data], ["Training run", COL.run], ["Model registry", COL.model], ["Serving", COL.serve]].map(([t, x]) => (
        <text key={t} x={x} y={22} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--muted)">{t}</text>
      ))}
      {/* row edges */}
      {s.runs.map((r, i) => (
        <g key={r.id}>
          <line x1={COL.data + NW / 2} x2={COL.run - NW / 2} y1={ROW[i]} y2={ROW[i]} stroke={isAudit && s.focusRow === i ? "var(--accent)" : "var(--muted)"} strokeWidth={isAudit && s.focusRow === i ? 2.5 : 1.25} markerEnd="url(#mv-arrow)" />
          <line x1={COL.run + NW / 2} x2={COL.model - NW / 2} y1={ROW[i]} y2={ROW[i]} stroke={isAudit && s.focusRow === i ? "var(--accent)" : "var(--muted)"} strokeWidth={isAudit && s.focusRow === i ? 2.5 : 1.25} markerEnd="url(#mv-arrow)" />
        </g>
      ))}
      <defs>
        <marker id="mv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted)" /></marker>
      </defs>
      {s.snapshots.map((sn, i) => {
        const unpinned = i === 1 && !pinned2;
        return (
          <NodeBox key={`${sn.id}-${fresh(i) && s.phase === "snapshot" ? animKey : "static"}`} x={COL.data} y={ROW[i]} title={unpinned ? "latest ⚠ unpinned" : `snap ${sn.date}`} lines={[sn.hash, `${sn.rows.toLocaleString("en-US")} rows`]}
            tone={isAudit && s.focusRow === i ? "accent" : unpinned ? "warn" : "neutral"} glow={isAudit && s.focusRow === i} delay={isAudit ? auditHop(3) : T.snapshot.node} k={k} />
        );
      })}
      {s.runs.map((r, i) => (
        <NodeBox key={`${r.id}-${fresh(i) && s.phase === "train" ? animKey : "static"}`} x={COL.run} y={ROW[i]} title={r.id} lines={[`commit ${r.lineage.commit}`, `lr ${r.lineage.hparams.lr} · depth ${r.lineage.hparams.depth}`]}
          tone={isAudit && s.focusRow === i ? "accent" : r.lineage.seed === null || r.lineage.codeHash !== r.lineage.commit ? "warn" : "neutral"} glow={isAudit && s.focusRow === i}
          delay={s.phase === "train" && fresh(i) ? T.train.toRun + T.train.travel : isAudit ? auditHop(2) : 0} k={k} />
      ))}
      {s.models.map((m, i) => (
        <NodeBox key={`${m.version}-${fresh(i) && s.phase === "train" ? animKey : "static"}`} x={COL.model} y={ROW[i]} title={`model v${m.version}`} lines={[m.hash, `acc ${m.accuracy.toFixed(4)}`]}
          tone={isAudit && s.focusRow === i ? "accent" : "neutral"} glow={isAudit && s.focusRow === i}
          delay={s.phase === "train" && fresh(i) ? T.train.model : isAudit ? auditHop(1) : 0} k={k} />
      ))}
      {s.phase === "train" && (
        <g key={animKey}>
          <Token from={COL.data + NW / 2} to={COL.run - NW / 2} y={ROW[s.focusRow]} delay={T.train.toRun} k={k} />
          <Token from={COL.run + NW / 2} to={COL.model - NW / 2} y={ROW[s.focusRow]} delay={T.train.toModel} k={k} />
        </g>
      )}

      {/* serving endpoint + its edge to whatever 'production' points at */}
      <NodeBox x={COL.serve} y={160} title="/predict endpoint" lines={[prod ? `serves v${prod}` : "no model", prod ? s.models[prod - 1].hash : "—"]}
        tone={prod ? (isAudit ? "accent" : "ok") : "neutral"} glow={isAudit} k={k} />
      {prod && (
        <motion.path key={`edge-${prod}-${s.moved?.alias === "production" ? animKey : "static"}`} d={`M${COL.serve - NW / 2} 160 C ${COL.serve - 90} 160, ${COL.model + 100} ${prodRow}, ${COL.model + NW / 2} ${prodRow}`}
          fill="none" stroke={isAudit ? "var(--accent)" : "var(--ok)"} strokeWidth={2.5} strokeDasharray={isAudit ? undefined : "6 4"}
          initial={{ pathLength: s.moved?.alias === "production" ? 0 : 1 }} animate={{ pathLength: 1 }} transition={{ delay: edgeDelay * k, duration: 0.6 * k, ease }} />
      )}

      {/* alias pills slide between versions */}
      {(["staging", "production"] as const).map((a, j) => {
        const v = s.aliases[a];
        const from = s.moved?.alias === a ? s.moved.from : v;
        const y = (ver: number | null) => (ver ? ROW[ver - 1] - 44 : ROW[0] - 44);
        const w = a === "production" ? 86 : 62;
        const x = COL.model - 42 + j * 80;
        const fill = a === "production" ? "var(--ok)" : "var(--warn)";
        return v ? (
          <motion.g key={`${a}-${animKey}`} initial={{ y: y(from), opacity: from ? 1 : 0, scale: from ? 1 : 0.6 }} animate={{ y: y(v), opacity: 1, scale: 1 }}
            transition={{ delay: aliasDelay(a) * k, type: "spring", stiffness: 70, damping: 14, opacity: { delay: aliasDelay(a) * k, duration: 0.3 * k } }}>
            <rect x={x - w / 2} y={-10} width={w} height={20} rx={10} fill={fill} />
            <text x={x} y={4} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--surface)">{a === "production" ? "★ production" : "staging"}</text>
          </motion.g>
        ) : null;
      })}

      {/* reproduction row */}
      {s.phase === "repro" && s.repro && (() => {
        const r = s.repro;
        const pinned = r.recorded.pinned;
        return (
          <g key={animKey}>
            <text x={20} y={REPRO_Y - 40} fontSize={11} fontWeight={600} fill="var(--fg)">Reproduce v2 using only the recorded lineage</text>
            <NodeBox x={COL.data} y={REPRO_Y} title={pinned ? `snap ${SNAPSHOTS.s2.date}` : `latest → ${SNAPSHOTS.s3.date}`} lines={[r.replay.dataHash, `${r.replay.rows.toLocaleString("en-US")} rows`]}
              tone={r.diffs.includes("dataHash") ? "err" : "ok"} dashed delay={T.repro.data} k={k} />
            <NodeBox x={COL.run} y={REPRO_Y} title="retrain" lines={[`commit ${r.recorded.commit}`, `seed ${r.replay.seedUsed}${r.recorded.seed === null ? " (new)" : ""}`]}
              tone={r.diffs.includes("seed") || r.diffs.includes("codeHash") ? "err" : "ok"} dashed delay={T.repro.toRun + T.train.travel} k={k} />
            <NodeBox x={COL.model} y={REPRO_Y} title={r.match ? "✓ same model" : "✗ different model"} lines={[r.hash, `acc ${r.accuracy.toFixed(4)}`]} tone={r.match ? "ok" : "err"} dashed delay={T.repro.toModel + T.train.travel} k={k} />
            <line x1={COL.data + NW / 2} x2={COL.run - NW / 2} y1={REPRO_Y} y2={REPRO_Y} stroke="var(--muted)" strokeDasharray="4 3" />
            <line x1={COL.run + NW / 2} x2={COL.model - NW / 2} y1={REPRO_Y} y2={REPRO_Y} stroke="var(--muted)" strokeDasharray="4 3" />
            <Token from={COL.data + NW / 2} to={COL.run - NW / 2} y={REPRO_Y} delay={T.repro.toRun} k={k} color="var(--violet)" />
            <Token from={COL.run + NW / 2} to={COL.model - NW / 2} y={REPRO_Y} delay={T.repro.toModel} k={k} color="var(--violet)" />
            <motion.g initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: T.repro.verdict * k, duration: 0.4 * k }}>
              <rect x={COL.model + 70} y={REPRO_Y - 16} width={150} height={32} rx={6} fill={r.match ? "var(--ok-soft)" : "var(--err-soft)"} stroke={r.match ? "var(--ok)" : "var(--err)"} />
              <text x={COL.model + 145} y={REPRO_Y - 1} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--fg)">{r.match ? "✓ hash matches v2" : "✗ ≠ v2 " + s.models[1].hash}</text>
              <text x={COL.model + 145} y={REPRO_Y + 11} textAnchor="middle" fontSize={9.5} fill="var(--muted)">{r.match ? "reproducible" : `differs in ${r.diffs.join(", ")}`}</text>
            </motion.g>
          </g>
        );
      })()}
      {s.phase !== "repro" && s.phase !== "audit" && (
        <g opacity={0.6}>
          <rect x={20} y={REPRO_Y - 30} width={COL.model + NW / 2 - 20} height={60} rx={8} fill="none" stroke="var(--line)" strokeDasharray="5 4" />
          <text x={(20 + COL.model + NW / 2) / 2} y={REPRO_Y + 4} textAnchor="middle" fontSize={11} fill="var(--muted)">reproducibility check runs here (step 9)</text>
        </g>
      )}
      {s.phase === "audit" && (
        <motion.text key={animKey} x={20} y={REPRO_Y} fontSize={11} fill="var(--accent)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.audit.hop(4) * k }}>
          ← traced back: endpoint → alias → version → run → data + code
        </motion.text>
      )}
    </svg>
  );
}

/* ───────────────────────────── step card ───────────────────────────── */

function HashReveal({ hash, start, k, compare }: { hash: string; start: (i: number) => number; k: number; compare?: string }) {
  return (
    <span className="inline-flex gap-px font-mono text-base">
      {hash.split("").map((c, i) => {
        const bad = compare !== undefined && compare[i] !== c;
        return (
          <motion.span key={i} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: start(i) * k, duration: 0.2 * k }}
            className={cx("rounded px-0.5", compare !== undefined && (bad ? "bg-err-soft text-err" : "bg-ok-soft text-ok"))}>{c}</motion.span>
        );
      })}
    </span>
  );
}

function LineageRows({ m, k, start }: { m: ModelVersion; k: number; start: (i: number) => number }) {
  const l = m.lineage;
  const rows: [string, string, boolean][] = [
    ["data", l.pinned ? `${l.dataRef} @ ${l.dataHash}` : `${l.dataRef} (no hash recorded)`, !l.pinned],
    ["code", l.codeHash === l.commit ? `commit ${l.commit}` : `commit ${l.commit} + uncommitted edit`, l.codeHash !== l.commit],
    ["hyperparams", `lr=${l.hparams.lr}, depth=${l.hparams.depth}`, false],
    ["seed", l.seed === null ? "not recorded" : String(l.seed), l.seed === null],
    ["weights hash", m.hash, false],
  ];
  return (
    <div className="space-y-1">
      {rows.map(([key, v, warn], i) => (
        <motion.div key={key} initial={{ opacity: 0, x: -24 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: start(i) * k, duration: 0.35 * k, ease }}
          className={cx("grid grid-cols-[100px_minmax(0,1fr)] items-center gap-2 rounded border px-2 py-1 text-xs", warn ? "border-warn/50 bg-warn-soft" : "border-line bg-surface-2")}>
          <span className="text-muted">{key}</span><span className="truncate font-mono">{warn && "⚠ "}{v}</span>
        </motion.div>
      ))}
    </div>
  );
}

function StepDetail({ s, k, pinned2 }: { s: MVState; k: number; pinned2: boolean }) {
  if (s.phase === "snapshot") {
    const sn = s.snapshots[s.focusRow];
    const unpinned = s.focusRow === 1 && !pinned2;
    return (
      <div className="flex flex-wrap items-center gap-4">
        <div className="rounded-lg border border-line bg-surface-2 px-3 py-2 font-mono text-xs leading-relaxed">
          <div>rows ≤ {sn.date}</div><div className="text-muted">{sn.rows.toLocaleString("en-US")} rows</div>
        </div>
        <span className="text-muted">→ H(·) →</span>
        <HashReveal hash={sn.hash} start={T.snapshot.hashChar} k={k} />
        <p className={cx("max-w-sm text-xs", unpinned ? "rounded-md bg-warn-soft px-2 py-1" : "text-muted")}>{unpinned ? "⚠ The hash exists, but run 2's config only stores the name 'latest'. Once new rows land, nobody can tell which rows v2 was trained on." : "Change one row and the hash changes. A run that records this hash records exactly which data it saw."}</p>
      </div>
    );
  }
  if (s.phase === "train") {
    const m = s.models[s.focusRow];
    return (
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_170px]">
        <div><h4 className="mb-1.5 text-sm font-semibold">Lineage recorded for v{m.version}</h4><LineageRows m={m} k={k} start={T.train.row} /></div>
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: T.train.model * k, duration: 0.4 * k }} className="self-center rounded-lg border border-accent/40 bg-accent-soft p-3 text-center">
          <div className="text-xs text-muted">validation accuracy</div>
          <div className="font-mono text-2xl font-semibold"><AnimatedNumber value={m.accuracy} from={0.5} digits={4} delay={T.train.model * k} duration={0.7 * k} /></div>
          {s.models[0] && m.version === 2 && <div className="mt-1 text-[11px] text-muted">v1: {s.models[0].accuracy.toFixed(4)}</div>}
        </motion.div>
      </div>
    );
  }
  if (s.phase === "alias" || s.phase === "promote") {
    const g = s.gate;
    const lo = 0.75, hi = 0.9;
    const pct = (v: number) => `${Math.max(2, Math.min(100, ((v - lo) / (hi - lo)) * 100))}%`;
    return (
      <div className="space-y-3">
        {g && (
          <div className="space-y-1.5">
            <h4 className="text-sm font-semibold">Promotion gate: v2 must beat v1 by ≥ {(g.minGain * 100).toFixed(1)} pts</h4>
            {([["v1 (production)", g.v1], ["v2 (candidate)", g.v2]] as const).map(([name, v], i) => (
              <div key={name} className="grid grid-cols-[110px_minmax(0,1fr)_60px] items-center gap-2 text-xs">
                <span>{name}</span>
                <div className="relative h-5 rounded bg-surface-2">
                  <motion.div className={cx("absolute inset-y-0.5 left-0 rounded", i === 0 ? "bg-muted/50" : g.pass ? "bg-ok/70" : "bg-err/70")} initial={{ width: 0 }} animate={{ width: pct(v) }} transition={{ delay: (T.promote.bars + i * 0.3) * k, duration: 0.6 * k, ease }} />
                  <motion.div className="absolute inset-y-0 w-0.5 bg-fg" style={{ left: pct(g.v1 + g.minGain) }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.promote.verdict * k * 0.7 }} />
                </div>
                <span className="text-right font-mono"><AnimatedNumber value={v} from={lo} digits={4} delay={(T.promote.bars + i * 0.3) * k} duration={0.6 * k} /></span>
              </div>
            ))}
            <p className="text-[11px] text-muted">black tick = v1 + required gain = {(g.v1 + g.minGain).toFixed(4)}</p>
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: T.promote.verdict * k }}>
              <StatusBadge status={g.pass ? "ok" : "error"} label={g.pass ? "gate passed: promote" : "gate failed: stay on v1"} />
            </motion.div>
          </div>
        )}
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">alias</th><th>points to</th><th>meaning</th></tr></thead>
          <tbody>
            {(["staging", "production"] as const).map((a) => {
              const changed = s.moved?.alias === a;
              return (
                <motion.tr key={a} className="border-t border-line" initial={{ backgroundColor: "rgba(0,0,0,0)" }}
                  animate={changed ? { backgroundColor: ["rgba(0,0,0,0)", "var(--accent-soft)", "var(--accent-soft)"] } : {}} transition={{ delay: (s.phase === "promote" ? T.promote.slide : T.alias.slide) * k, duration: 0.8 * k }}>
                  <td className="py-1 font-mono">{a}</td>
                  <td className="font-mono">{changed && s.moved?.from ? <><span className="text-muted line-through">v{s.moved.from}</span> → </> : null}{s.aliases[a] ? `v${s.aliases[a]}` : "unset"}</td>
                  <td className="text-muted">{a === "production" ? "what /predict serves" : "what QA and shadow tests use"}</td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }
  if (s.phase === "repro" && s.repro) {
    const r = s.repro;
    const rows: [string, string, string, boolean][] = [
      ["data", r.recorded.pinned ? r.recorded.dataHash : `${r.recorded.dataRef} (resolved to ${r.recorded.dataHash})`, r.replay.dataHash, r.diffs.includes("dataHash")],
      ["code", r.recorded.codeHash, r.replay.codeHash, r.diffs.includes("codeHash")],
      ["seed", String(s.runs[1].actualSeed) + (r.recorded.seed === null ? " (unlogged)" : ""), String(r.replay.seedUsed), r.diffs.includes("seed")],
      ["hyperparams", `lr=${r.recorded.hparams.lr}, d=${r.recorded.hparams.depth}`, `lr=${r.replay.hparams.lr}, d=${r.replay.hparams.depth}`, false],
    ];
    return (
      <div className="space-y-3">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">input</th><th>what originally ran</th><th>what the retrain used</th><th /></tr></thead>
            <tbody className="font-mono">
              {rows.map(([n, a, b, bad], i) => (
                <motion.tr key={n} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + i * 0.3) * k }}>
                  <td className="py-1 font-sans">{n}</td><td>{a}</td><td>{b}</td><td>{bad ? <Badge tone="err">✗ differs</Badge> : <Badge tone="ok">✓ same</Badge>}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="text-muted">registered v2</span><HashReveal hash={s.models[1].hash} start={() => 0} k={k} />
          <span className="text-muted">retrained</span><HashReveal hash={r.hash} start={T.repro.char} k={k} compare={s.models[1].hash} />
        </div>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.repro.verdict * k }}>
          <StatusBadge status={r.match ? "ok" : "error"} label={r.match ? "reproducible" : "irreproducible: the recorded lineage is incomplete"} />
        </motion.div>
      </div>
    );
  }
  // audit
  const v = s.aliases.production ?? 1;
  const m = s.models[v - 1];
  const chain = ["/predict endpoint", `alias production → v${v}`, `model v${v} · ${m.hash}`, `${m.runId} · lr ${m.lineage.hparams.lr}, depth ${m.lineage.hparams.depth}, seed ${m.lineage.seed ?? "?"}`, `data ${m.lineage.dataRef} · code ${m.lineage.commit}`];
  return (
    <ol className="space-y-1">
      {chain.map((c, i) => (
        <motion.li key={c} initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: T.audit.hop(i) * k, duration: 0.35 * k, ease }}
          className="flex items-center gap-2 text-xs"><span className="w-5 font-mono text-muted">{i + 1}</span><span className="rounded border border-line bg-surface-2 px-2 py-1 font-mono">{c}</span></motion.li>
      ))}
    </ol>
  );
}
