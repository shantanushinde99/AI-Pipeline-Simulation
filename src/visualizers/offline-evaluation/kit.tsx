import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the five evaluation & reliability visualizers (offline-evaluation, online-evaluation, golden-datasets, regression-testing, silent-quality-degradation). */
export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + Replay button, shown right above each canvas. */
export function StepCaption({ index, total, stepKey, headline, sub, onReplay }: {
  index: number; total: number; stepKey: string; headline: ReactNode; sub: ReactNode; onReplay: () => void;
}) {
  return (
    <div className="mb-2 flex flex-wrap items-start gap-x-3 gap-y-1">
      <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{index + 1}/{total}</span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={stepKey} className="min-w-0 flex-1 basis-56" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
          <p className="text-base font-semibold leading-snug">{headline}</p>
          <p className="text-sm text-muted">{sub}</p>
        </motion.div>
      </AnimatePresence>
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again"><RotateCw size={13} /> Replay step</Button>
    </div>
  );
}

/** Signed percentage-point formatting: +11.3 pts. */
export const pts = (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)} pts`;
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
