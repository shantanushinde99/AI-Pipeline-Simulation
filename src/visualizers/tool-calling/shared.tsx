import { useEffect } from "react";
import { AnimatePresence, animate, motion, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the five agent-tooling visualizers (tool-calling, tool-selection, tool-argument-validation, planning-and-execution, multi-step-workflows). */
export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + Replay button, shown right above each canvas. */
export function StepCaption({ index, total, stepKey, headline, sub, onReplay }: {
  index: number; total: number; stepKey: string; headline: string; sub: string; onReplay: () => void;
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

/** Types `text` out character by character (remount via `key` to replay). */
export function Typewriter({ text, delay = 0, duration = 1 }: { text: string; delay?: number; duration?: number }) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(reduce ? text.length : 0);
  const shown = useTransform(mv, (v) => text.slice(0, Math.round(v)));
  useEffect(() => {
    const c = animate(mv, text.length, reduce ? { duration: 0 } : { delay, duration, ease: "linear" });
    return () => c.stop();
  }, [mv, text, delay, duration, reduce]);
  return <motion.span>{shown}</motion.span>;
}
