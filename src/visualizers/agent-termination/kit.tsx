import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "../../components/ui";

/** Shared by the five agent-reliability visualizers (agent-termination, retries-and-timeouts, memory, multi-agent, guardrails). */
export const EASE = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + Replay button, shown right above each canvas. */
export function StepCaption({ index, total, stepKey, headline, sub, onReplay }: {
  index: number; total: number; stepKey: string; headline: string; sub: string; onReplay: () => void;
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

/** A labelled progress meter that grows from `from` to `to` (fractions of the limit). Turns red with an icon-free text tag when it trips. */
export function Meter({ label, valueText, from, to, tripped, delay, duration, limitText }: {
  label: string; valueText: ReactNode; from: number; to: number; tripped: boolean; delay: number; duration: number; limitText: string;
}) {
  const pct = (f: number) => `${Math.min(1, Math.max(0, f)) * 100}%`;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-muted">{valueText} / {limitText}{tripped && <span className="ml-1 font-semibold text-err">LIMIT HIT</span>}</span>
      </div>
      <div className="relative h-4 overflow-hidden rounded bg-surface-2">
        <motion.div className={tripped ? "absolute inset-y-0 left-0 rounded bg-err/80" : to >= 0.8 ? "absolute inset-y-0 left-0 rounded bg-warn/80" : "absolute inset-y-0 left-0 rounded bg-accent/70"}
          initial={{ width: pct(from) }} animate={{ width: pct(to) }} transition={{ delay, duration, ease: EASE }} />
        <div className="absolute inset-y-0 right-0 w-0.5 bg-fg/50" aria-hidden />
      </div>
    </div>
  );
}
