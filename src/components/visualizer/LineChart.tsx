export type Series = { name: string; points: [number, number][]; color: string; dashed?: boolean };

/** Minimal SVG line chart. Hand-rolled: two charts in the app don't justify a charting dependency. */
export function LineChart({ series, xLabel, yLabel, logY, markerX, height = 220 }: {
  series: Series[]; xLabel: string; yLabel: string; logY?: boolean; markerX?: number; height?: number;
}) {
  const W = 560, H = height, L = 52, R = 12, T = 12, B = 34;
  const all = series.flatMap((s) => s.points).filter(([, y]) => Number.isFinite(y) && (!logY || y > 0));
  const xs = all.map(([x]) => x), ys = all.map(([, y]) => (logY ? Math.log10(y) : y));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const sx = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y: number) => T + (1 - ((logY ? Math.log10(y) : y) - y0) / (y1 - y0)) * (H - T - B);
  const yTicks = Array.from({ length: 4 }, (_, i) => y0 + ((y1 - y0) * i) / 3);
  const xTicks = Array.from(new Set(xs)).filter((_, i, a) => a.length <= 16 || i % 2 === 0);
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${yLabel} by ${xLabel}: ${series.map((s) => s.name).join(", ")}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={sy(logY ? 10 ** t : t)} y2={sy(logY ? 10 ** t : t)} stroke="var(--line)" />
            <text x={L - 6} y={sy(logY ? 10 ** t : t) + 3} textAnchor="end" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{logY ? (10 ** t).toPrecision(2) : t.toFixed(2)}</text>
          </g>
        ))}
        {xTicks.map((x) => <text key={x} x={sx(x)} y={H - B + 14} textAnchor="middle" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{x}</text>)}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--muted)">{xLabel}</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={11} fill="var(--muted)" transform={`rotate(-90 12 ${H / 2})`}>{yLabel}{logY ? " (log)" : ""}</text>
        {markerX !== undefined && <line x1={sx(markerX)} x2={sx(markerX)} y1={T} y2={H - B} stroke="var(--accent)" strokeDasharray="3 3" />}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined}
              points={s.points.filter(([, y]) => Number.isFinite(y) && (!logY || y > 0)).map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
            {s.points.filter(([, y]) => Number.isFinite(y) && (!logY || y > 0)).map(([x, y]) => <circle key={x} cx={sx(x)} cy={sy(y)} r={2.5} fill={s.color} />)}
          </g>
        ))}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1">
            <svg width="20" height="6" aria-hidden><line x1="0" y1="3" x2="20" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined} /></svg>{s.name}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
