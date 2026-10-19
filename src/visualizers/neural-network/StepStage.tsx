import { motion } from "motion/react";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { cx } from "../../components/ui";
import { ACTIVATIONS, forward, type NetworkInput, type NNState } from "./simulation";
import { ACT, COUNT, arrive, biasAt, sumAt } from "./timeline";

const ease = [0.22, 1, 0.36, 1] as const;
const sub = (s: string) => s.replace(/(\d)/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]);

const ACT_TEXT: Record<string, string> = {
  relu: "ReLU keeps a positive sum as it is and turns a negative sum into 0.",
  sigmoid: "Sigmoid squashes any number into the range 0 to 1.",
  tanh: "Tanh squashes any number into the range −1 to 1.",
  linear: "Linear passes the sum through unchanged. With no bend, the network can only learn straight-line rules.",
};

/** The big explanatory card under the diagram. `k` = 1 / playback speed; every delay is multiplied by it. */
export function StepStage({ state, input, k }: { state: NNState; input: NetworkInput; k: number }) {
  const r = forward(input);
  if (state.phase === "input") return <Inputs x={input.x} k={k} />;
  if (state.phase === "predict") return <Gauge y={r.y} k={k} />;
  const isOut = state.active === "y";
  const j = isOut ? -1 : Number(state.active![1]) - 1;
  if (state.phase === "sum") {
    const w = input.weights;
    const terms = isOut
      ? w.w2.map((wj, i) => ({ name: `h${sub(String(i + 1))} × w`, expr: `${r.h[i].toFixed(2)} × ${wj}`, v: wj * r.h[i] }))
      : [0, 1].map((i) => ({ name: `x${sub(String(i + 1))} × w`, expr: `${input.x[i]} × ${w.w1[j][i]}`, v: w.w1[j][i] * input.x[i] }));
    const bias = isOut ? w.b2 : w.b1[j];
    const z = isOut ? r.z2 : r.z1[j];
    return <SumBars who={state.active!} terms={terms} bias={bias} z={z} k={k} />;
  }
  const fn = isOut ? "sigmoid" : input.activation;
  const z = isOut ? r.z2 : r.z1[j];
  const a = isOut ? r.y : r.h[j];
  return <ActivationPlot who={state.active!} fn={fn} z={z} a={a} k={k} />;
}

function Inputs({ x, k }: { x: [number, number]; k: number }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      {x.map((v, i) => (
        <motion.div key={i} initial={{ opacity: 0, x: -40 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.35) * k, duration: 0.6 * k, ease }}
          className="rounded-lg border border-accent/40 bg-accent-soft px-4 py-2 font-mono text-lg">
          x{sub(String(i + 1))} = {v}
        </motion.div>
      ))}
      <p className="max-w-sm text-sm text-muted">One example, described by two numbers. Next, each hidden neuron will combine them.</p>
    </div>
  );
}

function SumBars({ who, terms, bias, z, k }: { who: string; terms: { name: string; expr: string; v: number }[]; bias: number; z: number; k: number }) {
  const n = terms.length;
  const rows = [
    ...terms.map((t, i) => ({ ...t, at: arrive(i), kind: "term" as const })),
    { name: "bias", expr: "+ b", v: bias, at: biasAt(n), kind: "bias" as const },
    { name: `z (sum)`, expr: "total", v: z, at: sumAt(n), kind: "total" as const },
  ];
  const scale = Math.max(1e-9, ...rows.map((r) => Math.abs(r.v)));
  const biggest = terms.reduce((a, b) => (Math.abs(b.v) > Math.abs(a.v) ? b : a));
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h4 className="text-sm font-semibold">What pushes {who} up or down</h4>
        <span className="text-[11px] text-muted">bar length = size of each contribution</span>
      </div>
      <div className="space-y-1.5">
        {rows.map((row, i) => {
          const pct = (Math.abs(row.v) / scale) * 50;
          const neg = row.v < 0;
          return (
            <div key={i} className={cx("grid grid-cols-[92px_minmax(0,1fr)_64px] items-center gap-2", row.kind === "total" && "border-t border-line pt-1.5")}>
              <div className="font-mono text-xs leading-tight">
                <div className={row.kind === "total" ? "font-semibold" : undefined}>{row.name}</div>
                <div className="text-[10px] text-muted">{row.expr}</div>
              </div>
              <div className="relative h-5 rounded bg-surface-2">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <motion.div
                  className={cx("absolute inset-y-0.5 rounded", row.kind === "total" ? "bg-accent" : neg ? "bg-err/70" : "bg-ok/70")}
                  style={neg ? { right: "50%" } : { left: "50%" }}
                  initial={{ width: 0 }} animate={{ width: `${pct}%` }}
                  transition={{ delay: row.at * k, duration: (row.kind === "total" ? COUNT : 0.45) * k, ease }}
                />
              </div>
              <div className={cx("text-right font-mono text-xs", row.kind === "total" && "font-semibold")}>
                <AnimatedNumber value={row.v} from={0} delay={row.at * k} duration={0.45 * k} />
              </div>
            </div>
          );
        })}
      </div>
      <motion.p className="mt-2 text-xs leading-relaxed text-muted" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: sumAt(n) * k }}>
        {Math.abs(biggest.v) < 1e-9
          ? "Every input contribution is 0 here, so only the bias decides the sum."
          : `The biggest push comes from ${biggest.name.split(" ")[0]} (${biggest.v > 0 ? "+" : ""}${biggest.v.toFixed(2)}). Green pushes the sum up, red pushes it down.`}
      </motion.p>
    </div>
  );
}

function ActivationPlot({ who, fn, z, a, k }: { who: string; fn: keyof typeof ACTIVATIONS; z: number; a: number; k: number }) {
  const f = ACTIVATIONS[fn].fn;
  const zr = Math.max(4, Math.ceil(Math.abs(z)) + 1);
  const xs = Array.from({ length: 121 }, (_, i) => -zr + (2 * zr * i) / 120);
  const ys = xs.map(f);
  const [y0, y1] = [Math.min(...ys, a, 0), Math.max(...ys, a, 1)];
  const W = 320, H = 180, P = 28;
  const sx = (v: number) => P + ((v + zr) / (2 * zr)) * (W - 2 * P);
  const sy = (v: number) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P);
  const curve = xs.map((v, i) => `${i ? "L" : "M"}${sx(v).toFixed(1)} ${sy(ys[i]).toFixed(1)}`).join(" ");
  const t = (s: number, d = 0.5) => ({ delay: s * k, duration: d * k, ease });
  const dead = fn === "relu" && z <= 0;
  return (
    <div className="grid items-center gap-4 sm:grid-cols-[320px_minmax(0,1fr)]">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full max-w-[320px]" role="img" aria-label={`${fn} curve: input z ${z.toFixed(2)} gives ${a.toFixed(2)}`}>
        <line x1={P} x2={W - P} y1={sy(0)} y2={sy(0)} stroke="var(--line)" />
        <line x1={sx(0)} x2={sx(0)} y1={P / 2} y2={H - P} stroke="var(--line)" />
        <text x={W - P} y={sy(0) + 14} fontSize={10} textAnchor="end" fill="var(--muted)">z (input)</text>
        <text x={sx(0) + 4} y={P / 2 + 8} fontSize={10} fill="var(--muted)">output</text>
        <motion.path d={curve} fill="none" stroke="var(--violet)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={t(ACT.curve, 0.6)} />
        <motion.line x1={sx(z)} x2={sx(z)} y1={sy(0)} y2={sy(a)} stroke="var(--accent)" strokeWidth={2} strokeDasharray="4 3" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={t(ACT.zLine)} />
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={t(ACT.zLine, 0.3)}>
          <rect x={sx(z) - 30} y={sy(0) + 3} width={60} height={15} rx={3} fill="var(--accent)" />
          <text x={sx(z)} y={sy(0) + 14} fontSize={10} textAnchor="middle" fill="var(--surface)" fontFamily="var(--font-mono)">z = {z.toFixed(2)}</text>
        </motion.g>
        <motion.circle cx={sx(z)} cy={sy(a)} r={6} fill="var(--accent)" initial={{ scale: 0 }} animate={{ scale: [0, 1.6, 1] }} transition={t(ACT.dot, 0.5)} />
        <motion.line x1={sx(z)} x2={sx(0)} y1={sy(a)} y2={sy(a)} stroke="var(--ok)" strokeWidth={2} strokeDasharray="4 3" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={t(ACT.aLine)} />
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={t(ACT.aLine + 0.3, 0.3)}>
          <rect x={sx(0) - (z < 0 ? -4 : 64)} y={sy(a) - 8} width={60} height={15} rx={3} fill="var(--ok)" />
          <text x={sx(0) - (z < 0 ? -34 : 34)} y={sy(a) + 3} fontSize={10} textAnchor="middle" fill="var(--surface)" fontFamily="var(--font-mono)">out = {a.toFixed(2)}</text>
        </motion.g>
      </svg>
      <div className="space-y-2 text-sm leading-relaxed">
        <p className="font-semibold">{who}: {fn}(<span className="font-mono">{z.toFixed(2)}</span>) = <span className="font-mono"><AnimatedNumber value={a} from={z} delay={ACT.morph * k} duration={0.5 * k} /></span></p>
        <p className="text-muted">{ACT_TEXT[fn]}</p>
        {dead && <p className="rounded-md bg-warn-soft px-2 py-1 text-xs">The sum is not positive, so {who} outputs 0. It stays silent for this input and its outgoing line carries nothing.</p>}
      </div>
    </div>
  );
}

function Gauge({ y, k }: { y: number; k: number }) {
  const cls = y >= 0.5 ? 1 : 0;
  return (
    <div className="space-y-3">
      <div className="relative h-8 overflow-hidden rounded-md bg-surface-2">
        <motion.div className={cx("absolute inset-y-0 left-0", cls ? "bg-ok/70" : "bg-accent/60")} initial={{ width: 0 }} animate={{ width: `${y * 100}%` }} transition={{ delay: 0.2 * k, duration: 1 * k, ease }} />
        <div className="absolute inset-y-0 left-1/2 w-0.5 bg-fg/60" />
        <span className="absolute top-1/2 left-1/2 ml-1.5 -translate-y-1/2 text-[10px] text-fg/70">threshold 0.5</span>
        <span className="absolute top-1/2 right-2 -translate-y-1/2 font-mono text-sm font-semibold">ŷ = <AnimatedNumber value={y} from={0} delay={0.2 * k} duration={1 * k} /></span>
      </div>
      <div className="flex justify-between font-mono text-[10px] text-muted"><span>0 = class 0</span><span>1 = class 1</span></div>
      <motion.p initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.2 * k }} className="text-sm">
        ŷ is {y >= 0.5 ? "above" : "below"} 0.5, so the prediction is <strong>class {cls}</strong>. The network is {(Math.max(y, 1 - y) * 100).toFixed(0)}% sure.
      </motion.p>
    </div>
  );
}
