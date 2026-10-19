import { motion } from "motion/react";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, cx } from "../../components/ui";
import { ease } from "../linear-regression/kit";
import { T } from "./canvas";
import { METHOD_LABEL, PROBES, TARGET_TRAIN, type FTState, type Sim, type Snapshot } from "./simulation";

const Pct = ({ v, from, delay, k }: { v: number; from: number; delay: number; k: number }) => (
  <><AnimatedNumber value={v * 100} from={from * 100} digits={0} delay={delay * k} duration={T.updDur * k} />%</>
);

/** P(positive) for the four probe reviews; bars slide from the previous value to the new one. */
function Probes({ prev, cur, k, at }: { prev: Snapshot; cur: Snapshot; k: number; at: number }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[11px] text-muted"><span>probe review</span><span>P(positive) · bar crosses 50% = predicts positive</span></div>
      {PROBES.map((q, i) => {
        const p = cur.probes[i], ok = (p >= 0.5 ? 1 : 0) === q.ex.y;
        return (
          <div key={q.label} className="grid grid-cols-[minmax(0,190px)_minmax(0,1fr)_74px] items-center gap-2 text-xs">
            <span className="flex items-center gap-1.5 font-mono">
              <Badge tone={q.domain === "phone" ? "accent" : "neutral"}>{q.domain}</Badge>{q.label}
            </span>
            <div className="relative h-4 rounded bg-surface-2">
              <motion.div className={cx("absolute inset-y-0 left-0 rounded", q.domain === "phone" ? "bg-accent/70" : "bg-violet/60")}
                initial={{ width: `${prev.probes[i] * 100}%` }} animate={{ width: `${p * 100}%` }} transition={{ delay: at * k, duration: T.updDur * k, ease }} />
              <div className="absolute inset-y-0 left-1/2 w-px bg-fg/50" />
            </div>
            <span className="flex items-center justify-end gap-1 font-mono">
              <Pct v={p} from={prev.probes[i]} delay={at} k={k} />
              <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (at + T.updDur) * k }}>
                {ok ? <Badge tone="ok">✓</Badge> : <Badge tone="err">✗</Badge>}
              </motion.span>
            </span>
          </div>
        );
      })}
      <p className="text-[11px] text-muted">Correct answers: phone · long = positive, phone · unpredictable = negative, movie · long = negative, movie · unpredictable = positive.</p>
    </div>
  );
}

function Kpi({ label, v, from, at, k, tone, digits = 0, unit = "%" }: { label: string; v: number; from: number; at: number; k: number; tone?: string; digits?: number; unit?: string }) {
  const scaleBy = unit === "%" ? 100 : 1;
  return (
    <div className="rounded-md border border-line px-2.5 py-1.5">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={cx("font-mono text-lg font-semibold", tone)}>
        <AnimatedNumber value={v * scaleBy} from={from * scaleBy} digits={digits} delay={at * k} duration={T.updDur * k} />{unit}
      </div>
    </div>
  );
}

export function StepStage({ sim, state, k }: { sim: Sim; state: FTState; k: number }) {
  const hs = sim.history, h0 = hs[0];
  if (state.kind === "base") {
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <Kpi label="movie reviews (what it was trained on)" v={h0.m.origAcc} from={0} at={0.2} k={k} tone="text-ok" />
          <Kpi label="phone reviews (the new task)" v={h0.m.valAcc} from={0} at={0.5} k={k} tone="text-err" />
        </div>
        <Probes prev={{ ...h0, probes: h0.probes.map(() => 0.5) }} cur={h0} k={k} at={0.9} />
      </div>
    );
  }
  if (state.kind === "data") {
    return (
      <div className="space-y-2">
        <div className="grid gap-1.5 sm:grid-cols-2">
          {TARGET_TRAIN.map((e, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -24 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.15 + i * 0.18) * k, duration: 0.4 * k, ease }}
              className={cx("flex items-center justify-between gap-2 rounded-md border px-2 py-1 text-xs", e.noisy ? "border-warn/50 bg-warn-soft" : "border-line")}>
              <span className="font-mono">{e.words.join(" · ")}</span>
              <span className="flex items-center gap-1">
                {e.noisy && <Badge tone="warn">mislabeled</Badge>}
                <Badge tone={e.y ? "ok" : "err"}>{e.y ? "positive" : "negative"}</Badge>
              </span>
            </motion.div>
          ))}
        </div>
        <motion.p className="text-xs text-muted" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 * k }}>
          {sim.nReplay ? `Plus ${sim.nReplay} movie reviews replayed from the original training data, so the old task keeps getting practice. ` : ""}
          {METHOD_LABEL[sim.cfg.method]} will update <strong>{sim.params.trainable}</strong> of the model's {sim.params.total} numbers.
        </motion.p>
      </div>
    );
  }
  if (state.kind === "epoch") {
    const a = hs[state.prevEpoch], b = hs[state.epoch];
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Kpi label="gradient updates so far" v={b.updates} from={a.updates} at={T.grad} k={k} unit="" />
          <Kpi label="phone accuracy (new)" v={b.m.valAcc} from={a.m.valAcc} at={T.update} k={k} tone={b.m.valAcc > a.m.valAcc ? "text-ok" : undefined} />
          <Kpi label="movie accuracy (original)" v={b.m.origAcc} from={a.m.origAcc} at={T.update} k={k} tone={b.m.origAcc < a.m.origAcc ? "text-err" : undefined} />
          <Kpi label="distance from base weights" v={b.drift} from={a.drift} at={T.update} k={k} digits={2} unit="" />
        </div>
        <Probes prev={a} cur={b} k={k} at={T.update} />
      </div>
    );
  }
  const last = hs.at(-1)!;
  const bars = [
    { label: "phone reviews (new task)", before: h0.m.valAcc, after: last.m.valAcc },
    { label: "movie reviews (original task)", before: h0.m.origAcc, after: last.m.origAcc },
  ];
  return (
    <div className="space-y-3">
      {bars.map((row, i) => (
        <div key={row.label} className="space-y-1">
          <div className="text-xs font-medium">{row.label}</div>
          {(["before", "after"] as const).map((w, j) => {
            const v = row[w];
            return (
              <div key={w} className="grid grid-cols-[52px_minmax(0,1fr)_44px] items-center gap-2 text-xs">
                <span className="text-muted">{w}</span>
                <div className="h-4 rounded bg-surface-2">
                  <motion.div className={cx("h-full rounded", w === "before" ? "bg-muted/50" : row.after >= row.before ? "bg-ok/70" : "bg-err/70")}
                    initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: (0.2 + i * 0.9 + j * 0.4) * k, duration: 0.6 * k, ease }} />
                </div>
                <span className="text-right font-mono"><AnimatedNumber value={v * 100} from={0} digits={0} delay={(0.2 + i * 0.9 + j * 0.4) * k} duration={0.6 * k} />%</span>
              </div>
            );
          })}
        </div>
      ))}
      <motion.div className="flex flex-wrap gap-1.5 text-xs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.2 * k }}>
        <Badge tone={sim.regime === "good" ? "ok" : sim.regime === "diverged" ? "err" : "warn"}>outcome: {sim.regime}</Badge>
        <Badge>trained {sim.params.trainable} / {sim.params.total} numbers</Badge>
        <Badge>best validation epoch: {sim.bestEpoch}</Badge>
        {sim.detached && <Badge tone="accent">adapter detached → movies back to {Math.round(sim.detached.acc * 100)}% (W untouched)</Badge>}
      </motion.div>
    </div>
  );
}
