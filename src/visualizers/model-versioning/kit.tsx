import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the production-engineering visualizers (model-versioning, experiment-tracking, ci-cd-for-ml, rollbacks, fallback-strategies). */
export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + replay button, shown right above each canvas. */
export function StepCaption({ index, total, stepKey, headline, sub, onReplay }: {
  index: number; total: number; stepKey: string; headline: ReactNode; sub: ReactNode; onReplay: () => void;
}) {
  return (
    <div className="mb-2 flex items-start gap-3">
      <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{index + 1}/{total}</span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={stepKey} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
          <p className="text-base font-semibold leading-snug" data-testid="step-headline">{headline}</p>
          <p className="text-sm text-muted">{sub}</p>
        </motion.div>
      </AnimatePresence>
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again" className="shrink-0"><RotateCw size={13} /> <span className="hidden sm:inline">Replay step</span></Button>
    </div>
  );
}

/** Crossfading body for the "What's happening in this step" card. */
export function StepCard({ animKey, children }: { animKey: string; children: ReactNode }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
