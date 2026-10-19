import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { RotateCw } from "lucide-react";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, cx } from "../../components/ui";

/** Shared by the five ML-metrics visualizers (classification-metrics, confusion-matrices, regression-metrics, data-leakage, model-drift). */
export const ease = [0.22, 1, 0.36, 1] as const;

export const pct = (v: number, d = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : "undefined");

/** Step badge + crossfading headline/sub-caption + Replay button, placed right above the canvas. */
export function StepCaption({ index, total, stepId, headline, sub, onReplay }: {
  index: number; total: number; stepId: string; headline: ReactNode; sub: ReactNode; onReplay: () => void;
}) {
  return (
    <div className="mb-2 flex items-start gap-3">
      <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{index + 1}/{total}</span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={stepId} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
          <p className="text-base font-semibold leading-snug" data-testid="step-headline">{headline}</p>
          <p className="text-sm text-muted">{sub}</p>
        </motion.div>
      </AnimatePresence>
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again"><RotateCw size={13} /> <span className="hidden sm:inline">Replay step</span></Button>
    </div>
  );
}

/** A 0..1 metric as a growing bar with a counting value. `value` may be NaN (shown as "undefined"). */
export function MetricBar({ label, value, delay = 0, k = 1, tone = "accent", note }: {
  label: ReactNode; value: number; delay?: number; k?: number; tone?: "accent" | "ok" | "warn" | "err" | "violet"; note?: ReactNode;
}) {
  const ok = Number.isFinite(value);
  const fill = { accent: "bg-accent", ok: "bg-ok", warn: "bg-warn", err: "bg-err", violet: "bg-violet" }[tone];
  return (
    <div className="grid grid-cols-[96px_minmax(0,1fr)_56px] items-center gap-2 text-xs">
      <span className="font-medium">{label}</span>
      <div className="relative h-4 overflow-hidden rounded bg-surface-2">
        <motion.div className={cx("absolute inset-y-0 left-0 rounded", fill)} initial={{ width: 0 }} animate={{ width: ok ? `${Math.max(0, Math.min(1, value)) * 100}%` : 0 }}
          transition={{ delay: delay * k, duration: 0.6 * k, ease }} />
        {note && <span className="absolute top-1/2 right-1 -translate-y-1/2 rounded bg-surface/85 px-1 text-[10px] leading-tight text-fg">{note}</span>}
      </div>
      <span className="text-right font-mono">{ok ? <AnimatedNumber value={value} from={0} digits={3} delay={delay * k} duration={0.6 * k} /> : "undef."}</span>
    </div>
  );
}

/**
 * Counts how many of `times` (seconds from now) have passed; restarts whenever `resetKey` or the times change.
 * Used for counters that tick once per landing item rather than gliding.
 */
export function useTicks(times: number[], resetKey: string) {
  const reduce = useReducedMotion();
  const sig = `${resetKey}|${times.join(",")}`;
  const [st, setSt] = useState({ sig, n: 0 });
  useEffect(() => {
    const list = sig.slice(sig.lastIndexOf("|") + 1);
    const ts = list ? list.split(",").map(Number).sort((a, b) => a - b) : [];
    // Absolute counts (not increments), so a re-armed effect can never overshoot.
    const ids = ts.map((t, i) => window.setTimeout(() => setSt((s) => ({ sig, n: s.sig === sig ? Math.max(s.n, i + 1) : i + 1 })), reduce ? 0 : t * 1000));
    return () => ids.forEach(clearTimeout);
  }, [sig, reduce]);
  return st.sig === sig ? st.n : 0;
}
