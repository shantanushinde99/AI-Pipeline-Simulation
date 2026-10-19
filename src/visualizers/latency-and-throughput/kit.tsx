import { useLayoutEffect, useState, type ReactNode } from "react";
import { AnimatePresence, animate, motion, useMotionValue, useMotionValueEvent, useReducedMotion, type MotionValue } from "motion/react";
import { RotateCw } from "lucide-react";
import { Button, cx } from "../../components/ui";

/**
 * Shared pieces for the queue-style visualizers (latency-and-throughput, model-serving, caching, rate-limits-and-retries,
 * training-vs-inference). Each step of those visualizers is a window of simulated time; this clock replays the window
 * on screen so dots, counters and charts all read the same simulated instant.
 */
export function useSimClock(from: number, to: number, seconds: number, key: string): MotionValue<number> {
  const reduce = useReducedMotion();
  const mv = useMotionValue(from);
  // Layout effect: the jump back to `from` lands before the first paint of a new step.
  useLayoutEffect(() => {
    if (reduce || seconds <= 0) {
      mv.jump(to);
      return;
    }
    mv.jump(from);
    const c = animate(mv, to, { duration: seconds, ease: "linear" });
    return () => c.stop();
  }, [mv, from, to, seconds, key, reduce]);
  return mv;
}

/** Subscribe a component to the clock. Only components that call this re-render every frame. */
export function useTime(clock: MotionValue<number>): number {
  const [t, setT] = useState(clock.get());
  useMotionValueEvent(clock, "change", setT);
  return clock.get() === t ? t : clock.get();
}

export const ease = [0.22, 1, 0.36, 1] as const;

/** Step badge + crossfading headline/sub-caption + Replay button, placed right above each canvas. */
export function StepCaption({ index, total, id, headline, sub, onReplay }: {
  index: number; total: number; id: string; headline: ReactNode; sub: ReactNode; onReplay: () => void;
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

export type StatTone = "neutral" | "ok" | "warn" | "err" | "accent";
const toneText: Record<StatTone, string> = { neutral: "text-fg", ok: "text-ok", warn: "text-warn", err: "text-err", accent: "text-accent" };

/** Row of live counters. Values are pre-formatted strings derived from the simulation clock, so they tick as it runs. */
export function Stats({ items }: { items: { label: string; value: string; tone?: StatTone; hint?: string }[] }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {items.map((s) => (
        <div key={s.label} className="rounded-md border border-line bg-surface-2/60 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-muted">{s.label}</div>
          <div className={cx("font-mono text-base font-semibold tabular-nums", toneText[s.tone ?? "neutral"])}>{s.value}</div>
          {s.hint && <div className="text-[10px] text-muted">{s.hint}</div>}
        </div>
      ))}
    </div>
  );
}

/** Bars over time. `values` per bucket are already clipped to the current clock by the caller; `yMax` stays fixed so bars grow instead of rescaling. */
export function TimeBars({ buckets, series, yMax, t, label, shade, height = 120 }: {
  buckets: { t0: number; t1: number; values: number[] }[];
  series: { name: string; color: string }[];
  yMax: number; t: number; label: string; shade?: [number, number][]; height?: number;
}) {
  const W = 520, H = height, L = 30, B = 18, T = 6;
  const x0 = buckets[0]?.t0 ?? 0, x1 = buckets.at(-1)?.t1 ?? 1;
  const sx = (v: number) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - 4);
  const sy = (v: number) => T + (1 - v / (yMax || 1)) * (H - T - B);
  const ticks = [0, Math.round(yMax / 2), Math.round(yMax)].filter((v, i, a) => a.indexOf(v) === i);
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={label}>
        {shade?.map(([a, b]) => <rect key={a} x={sx(a)} y={T} width={sx(b) - sx(a)} height={H - T - B} fill="var(--warn)" fillOpacity={0.12} />)}
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - 4} y1={sy(v)} y2={sy(v)} stroke="var(--line)" />
            <text x={L - 4} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>
          </g>
        ))}
        {buckets.map((b) => {
          let acc = 0;
          const w = Math.max(1, sx(b.t1) - sx(b.t0) - 1);
          return (
            <g key={b.t0}>
              {b.values.map((v, i) => {
                const y = sy(acc + v), h = sy(acc) - y;
                acc += v;
                return h > 0 ? <rect key={i} x={sx(b.t0)} y={y} width={w} height={h} fill={series[i].color} /> : null;
              })}
            </g>
          );
        })}
        <line x1={sx(Math.min(t, x1))} x2={sx(Math.min(t, x1))} y1={T} y2={H - B} stroke="var(--fg)" strokeOpacity={0.5} strokeDasharray="2 2" />
        <text x={L} y={H - 4} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{(x0 / 1000).toFixed(1)} s</text>
        <text x={W - 4} y={H - 4} textAnchor="end" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{(x1 / 1000).toFixed(1)} s (simulated)</text>
      </svg>
      <figcaption className="flex flex-wrap gap-3 text-[11px] text-muted">
        <span className="font-medium text-fg">{label}</span>
        {series.map((s) => <span key={s.name} className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} />{s.name}</span>)}
      </figcaption>
    </figure>
  );
}

/** Latency histogram with percentile markers. */
export function Histogram({ counts, lo, hi, markers, label, unit = "ms" }: {
  counts: number[]; lo: number; hi: number; markers: { v: number; label: string; color: string }[]; label: string; unit?: string;
}) {
  const W = 520, H = 130, L = 8, B = 18, T = 16;
  const max = Math.max(1, ...counts);
  const bw = (W - 2 * L) / counts.length;
  const sx = (v: number) => L + ((v - lo) / (hi - lo || 1)) * (W - 2 * L);
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={label}>
        <line x1={L} x2={W - L} y1={H - B} y2={H - B} stroke="var(--line)" />
        {counts.map((c, i) => {
          const h = (c / max) * (H - T - B);
          return <rect key={i} x={L + i * bw + 0.5} y={H - B - h} width={bw - 1} height={h} rx={1.5} fill="var(--accent)" fillOpacity={0.55} style={{ transition: "y .25s, height .25s" }} />;
        })}
        {markers.filter((m) => Number.isFinite(m.v)).map((m, i) => (
          <g key={m.label}>
            <line x1={sx(m.v)} x2={sx(m.v)} y1={T - 4} y2={H - B} stroke={m.color} strokeWidth={2} style={{ transition: "x1 .25s, x2 .25s" }} />
            <text x={Math.min(W - 60, Math.max(L, sx(m.v) + 3))} y={T - 6 + i * 11 + 4} fontSize={9} fill={m.color} fontFamily="var(--font-mono)">{m.label}</text>
          </g>
        ))}
        <text x={L} y={H - 4} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{Math.round(lo)} {unit}</text>
        <text x={W - L} y={H - 4} textAnchor="end" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{Math.round(hi)} {unit}</text>
      </svg>
      <figcaption className="text-[11px] font-medium">{label}</figcaption>
    </figure>
  );
}

/** Formats a number or an em dash when undefined. */
export const f0 = (v: number, unit = "") => (Number.isFinite(v) ? `${Math.round(v)}${unit}` : "—");
export const f1 = (v: number, unit = "") => (Number.isFinite(v) ? `${v.toFixed(1)}${unit}` : "—");
export const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(0)}%` : "—");
