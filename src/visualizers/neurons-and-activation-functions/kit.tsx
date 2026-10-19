import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by this group's visualizers (neurons, tokenization, next-token, sampling, context windows). */
export const EASE = [0.22, 1, 0.36, 1] as const;

/** Step badge + headline + sub-caption above a canvas, crossfading per step, with a Replay button. */
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

/**
 * Counts 0 → `count`, one frame every `ms` after `delay` ms. Remount (via `key`) to restart.
 * With reduced motion it returns the final frame immediately.
 */
export function useFrames(count: number, ms: number, delay = 0) {
  const reduce = useReducedMotion();
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    if (reduce || count <= 0) return;
    let iv: number | undefined;
    let ticks = 0;
    const t = window.setTimeout(() => {
      iv = window.setInterval(() => {
        ticks++;
        setFrame((f) => Math.min(f + 1, count));
        if (ticks >= count) window.clearInterval(iv);
      }, Math.max(16, ms));
    }, delay);
    return () => {
      window.clearTimeout(t);
      window.clearInterval(iv);
    };
  }, [count, ms, delay, reduce]);
  return reduce ? count : Math.min(frame, count);
}
