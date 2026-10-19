import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the five AI Evaluation & Reliability visualizers (hallucination, injection, structured output, safety, RCA). */
export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + replay button, shown right above each canvas. */
export function StepCaption({ index, total, stepKey, headline, sub, onReplay }: {
  index: number; total: number; stepKey: string; headline: ReactNode; sub: ReactNode; onReplay: () => void;
}) {
  return (
    <div className="mb-3 flex items-start gap-3">
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

/** Points along a cubic Bézier from (x1,y1) to (x2,y2) with horizontal tangents, for a dot that rides a connector. */
export function bezier(x1: number, y1: number, x2: number, y2: number, n = 10) {
  const mx = (x1 + x2) / 2;
  const pts: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    pts.push([u ** 3 * x1 + 3 * u * u * t * mx + 3 * u * t * t * mx + t ** 3 * x2, u ** 3 * y1 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3 * y2]);
  }
  return { d: `M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`, xs: pts.map((p) => p[0]), ys: pts.map((p) => p[1]) };
}

export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
