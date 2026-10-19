import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button } from "../../components/ui";

/** Shared by the ML-fundamentals visualizers (linear/logistic regression, gradient descent, loss functions, backprop). */
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
      <Button size="sm" variant="ghost" onClick={onReplay} title="Play this step's animation again" className="shrink-0">
        <RotateCw size={13} /> <span className="hidden sm:inline">Replay step</span>
      </Button>
    </div>
  );
}

/** Linear map from a data domain to pixel range. */
export const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

/** "Nice" tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/** Gridlines, tick labels and axis titles for a plot area [L, R] × [T, B] (pixels). */
export function Axes({ sx, sy, xDomain, yDomain, L, R, T, B, xLabel, yLabel, xCount = 6, yCount = 5 }: {
  sx: (v: number) => number; sy: (v: number) => number; xDomain: [number, number]; yDomain: [number, number];
  L: number; R: number; T: number; B: number; xLabel: string; yLabel: string; xCount?: number; yCount?: number;
}) {
  const fmt = (v: number) => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(Math.abs(v) < 1 ? 2 : 1));
  return (
    <g fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">
      {ticks(yDomain[0], yDomain[1], yCount).map((t) => (
        <g key={`y${t}`}>
          <line x1={L} x2={R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" strokeOpacity={t === 0 ? 1 : 0.6} />
          <text x={L - 5} y={sy(t) + 3} textAnchor="end">{fmt(t)}</text>
        </g>
      ))}
      {ticks(xDomain[0], xDomain[1], xCount).map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={T} y2={B} stroke="var(--line)" strokeOpacity={t === 0 ? 1 : 0.4} />
          <text x={sx(t)} y={B + 13} textAnchor="middle">{fmt(t)}</text>
        </g>
      ))}
      <text x={(L + R) / 2} y={B + 27} textAnchor="middle" fontFamily="var(--font-sans)" fontSize={11}>{xLabel}</text>
      <text x={11} y={(T + B) / 2} textAnchor="middle" fontFamily="var(--font-sans)" fontSize={11} transform={`rotate(-90 11 ${(T + B) / 2})`}>{yLabel}</text>
    </g>
  );
}
