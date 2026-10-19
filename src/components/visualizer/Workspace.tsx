import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";

/** Center visualization + right inspector. The inspector stacks under the canvas below the xl breakpoint. */
export function Workspace({ main, inspector }: { main: ReactNode; inspector: ReactNode }) {
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="min-w-0 space-y-3">{main}</div>
      <aside className="min-w-0 space-y-3" aria-label="Inspector">{inspector}</aside>
    </div>
  );
}

export function StepExplainer({ index, title, what, why }: { index: number; title: string; what: ReactNode; why?: ReactNode }) {
  return (
    <section className="overflow-hidden rounded-lg border border-accent/30 bg-accent-soft/50 p-3" aria-live="polite">
      <AnimatePresence mode="wait" initial={false}>
      <motion.div key={`${index}-${title}`} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: 0.2 }}>
      <div className="font-mono text-[11px] text-accent">STEP {index + 1}</div>
      <h3 className="mt-0.5 text-sm font-semibold">{title}</h3>
      <div className="mt-1.5 text-xs leading-relaxed">{what}</div>
      {why && (
        <div className="mt-2 border-t border-accent/20 pt-2 text-xs leading-relaxed">
          <span className="font-semibold">Why it matters: </span>
          {why}
        </div>
      )}
      </motion.div>
      </AnimatePresence>
    </section>
  );
}
