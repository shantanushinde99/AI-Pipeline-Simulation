import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";

/** Shared by the ML-fundamentals visualizers (decision-trees, supervised-vs-unsupervised-learning, …). */
export const EASE = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + Replay button, shown right above a canvas. */
export function StepCaption({ index, total, stepId, headline, sub, onReplay }: {
  index: number; total: number; stepId: string; headline: string; sub: string; onReplay: () => void;
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
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again" className="shrink-0"><RotateCw size={13} /> <span className="hidden sm:inline">Replay step</span></Button>
    </div>
  );
}

/** A labelled horizontal bar that grows from 0 to `value / max` and counts up its number. */
export function GrowBar({ label, value, max, color, delay, duration, digits = 3, suffix }: {
  label: string; value: number; max: number; color: string; delay: number; duration: number; digits?: number; suffix?: string;
}) {
  const pct = max > 0 ? Math.max(0, Math.min(1, value / max)) * 100 : 0;
  return (
    <div className="grid grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)_4.5rem] items-center gap-2 text-xs">
      <span className="truncate">{label}</span>
      <div className="relative h-4 rounded bg-surface-2">
        <motion.div className="absolute inset-y-0.5 left-0 rounded" style={{ background: color }} initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ delay, duration, ease: EASE }} />
      </div>
      <span className="text-right font-mono">
        <AnimatedNumber value={value} from={0} delay={delay} duration={duration} digits={digits} />{suffix}
      </span>
    </div>
  );
}
