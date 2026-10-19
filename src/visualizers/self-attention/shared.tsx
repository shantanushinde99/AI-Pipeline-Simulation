/** UI pieces shared by the transformer visualizers (imported read-only from sibling folders). */
import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button, cx } from "../../components/ui";

export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + Replay button, placed right above a canvas. */
export function StepHeader({ index, total, stepId, headline, sub, onReplay }: {
  index: number; total: number; stepId: string; headline: string; sub: string; onReplay: () => void;
}) {
  return (
    <div className="mb-2 flex items-start gap-3">
      <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{index + 1}/{total}</span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={stepId} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
          <p className="text-base font-semibold leading-snug">{headline}</p>
          <p className="text-sm text-muted">{sub}</p>
        </motion.div>
      </AnimatePresence>
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again"><RotateCw size={13} /> Replay step</Button>
    </div>
  );
}

/**
 * A vector drawn as a strip of vertical bars around a midline (SVG <g>). Bars grow from `from` (or 0) to `values`,
 * so the same strip visibly transforms between stages. `max` sets the scale: |v| = max fills half the height.
 */
export function BarStrip({ x, y, values, from, max, w = 14, gap = 3, h = 60, k = 1, delay = 0, color = "var(--accent)", labels, showValues = false, stagger = 0.06 }: {
  x: number; y: number; values: number[]; from?: number[]; max: number; w?: number; gap?: number; h?: number; k?: number; delay?: number;
  color?: string; labels?: string[]; showValues?: boolean; stagger?: number;
}) {
  const mid = y + h / 2;
  const s = (h / 2) / Math.max(max, 1e-9);
  const geo = (v: number) => {
    const c = Math.max(-h / 2, Math.min(h / 2, v * s));
    return { y: c >= 0 ? mid - c : mid, height: Math.max(Math.abs(c), 0.5) };
  };
  return (
    <g>
      <rect x={x - 3} y={y} width={values.length * (w + gap) + 3} height={h} rx={4} fill="var(--surface-2)" />
      <line x1={x - 3} x2={x + values.length * (w + gap)} y1={mid} y2={mid} stroke="var(--line)" />
      {values.map((v, i) => {
        const g0 = geo(from?.[i] ?? 0);
        const g1 = geo(v);
        return (
          <g key={i}>
            <motion.rect x={x + i * (w + gap)} width={w} rx={2} fill={v < 0 ? "var(--err)" : color} fillOpacity={0.85}
              initial={g0} animate={g1} transition={{ delay: (delay + i * stagger) * k, duration: 0.6 * k, ease }} />
            {showValues && (
              <motion.text x={x + i * (w + gap) + w / 2} y={y + h + 11} textAnchor="middle" fontSize={8.5} fontFamily="var(--font-mono)" fill="var(--muted)"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (delay + i * stagger + 0.3) * k }}>{v.toFixed(2)}</motion.text>
            )}
            {labels && <text x={x + i * (w + gap) + w / 2} y={y - 3} textAnchor="middle" fontSize={8} fill="var(--muted)">{labels[i]}</text>}
          </g>
        );
      })}
    </g>
  );
}

/** Diverging colour for heatmaps: blue-ish accent for positive, red for negative, opacity ∝ |v|/max. */
export const heat = (v: number, max: number) => ({ fill: v < 0 ? "var(--err)" : "var(--accent)", fillOpacity: Math.min(1, Math.abs(v) / Math.max(max, 1e-9)) * 0.85 + 0.05 });

/** Horizontal bar that grows from 0 at time `at` (seconds at 1×). */
export function HBarRaw({ v, max, at, tone = "bg-accent", k }: { v: number; max: number; at: number; tone?: string; k: number }) {
  return (
    <div className="relative h-4 rounded bg-surface-2">
      <motion.div className={cx("absolute inset-y-0.5 left-0 rounded", v < 0 ? "bg-err/70" : tone)} initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, (Math.abs(v) / Math.max(max, 1e-9)) * 100)}%` }} transition={{ delay: at * k, duration: 0.5 * k, ease }} />
    </div>
  );
}
