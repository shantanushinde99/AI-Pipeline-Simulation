import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the five AI-production visualizers (monitoring, drift, tracing, skew, cost). */
export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + headline + sub-caption above the canvas, crossfaded per step, with a Replay button. */
export function StepCaption({ index, total, id, headline, sub, onReplay }: {
  index: number; total: number; id: string; headline: string; sub: string; onReplay: () => void;
}) {
  return (
    <div className="mb-2 flex items-start gap-3">
      <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{index + 1}/{total}</span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={id} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
          <p className="text-base font-semibold leading-snug" data-testid="step-headline">{headline}</p>
          <p className="text-sm text-muted">{sub}</p>
        </motion.div>
      </AnimatePresence>
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again" aria-label="Replay step">
        <RotateCw size={13} /> <span className="hidden sm:inline">Replay step</span>
      </Button>
    </div>
  );
}

/**
 * Reveals items 1..total one at a time (first after `startSec`, then every `everySec`), restarting whenever `key` changes.
 * Canvas, card and counters all read the same count, so they stay in sync. Pass seconds already scaled by 1 / speed.
 */
export function useReveal(total: number, key: string, everySec: number, startSec = 0.3) {
  const reduce = useReducedMotion();
  const [st, setSt] = useState({ key, n: 0 });
  useEffect(() => {
    if (reduce) return;
    const ids = Array.from({ length: total }, (_, i) => window.setTimeout(() => setSt({ key, n: i + 1 }), (startSec + i * everySec) * 1000));
    return () => ids.forEach((id) => window.clearTimeout(id));
  }, [key, total, everySec, startSec, reduce]);
  if (reduce) return total;
  return st.key === key ? st.n : 0;
}
