import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the RAG-stage visualizers (document-ingestion, parsing, chunking, vector DBs, similarity search). */
export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + replay button, shown right above each canvas. */
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
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again"><RotateCw size={13} /> <span className="hidden sm:inline">Replay step</span></Button>
    </div>
  );
}

/** FNV-1a 32-bit, hex. A fast, deterministic stand-in for the SHA-256 a production pipeline would use. */
export function fnv1aHex(s: string): string {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}
