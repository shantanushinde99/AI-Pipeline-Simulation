import { motion } from "motion/react";
import { Lock } from "lucide-react";
import { ease, scale } from "../linear-regression/kit";
import { D, H, TOPICS, VOCAB, effectiveW, type Example, type Method, type Snapshot } from "./simulation";

/**
 * One schedule (seconds at 1×) for an epoch step: inputs light up their columns → the gradient sweeps back
 * from the output → weights recolor while the loss curves, probe bars and counters move. Multiply by k = 1/speed.
 */
export const T = { feed: 0.15, feedStagger: 0.09, feedDur: 0.6, grad: 1.05, gradDur: 0.7, update: 1.7, updDur: 0.9 };
const MS = { base: 3000, data: 3200, epoch: 3400, result: 3400 } as const;
export const stepMs = (kind: keyof typeof MS) => MS[kind];

const P = 22, C = 20; // cell pitch and size
const LABEL_H = 74;

/** Positive weights in blue, negative in red; stronger = more opaque. Two layers so opacity can animate. */
const pos = (v: number) => (v > 0 ? Math.min(0.95, 0.06 + Math.abs(v) / 3) : 0);
const neg = (v: number) => (v < 0 ? Math.min(0.95, 0.06 + Math.abs(v) / 3) : 0);

function Cell({ x, y, v, from, at, k, animate, flash }: { x: number; y: number; v: number; from: number; at: number; k: number; animate: boolean; flash: boolean }) {
  const tr = { delay: at * k, duration: T.updDur * k, ease };
  return (
    <g>
      <rect x={x} y={y} width={C} height={C} rx={3} fill="var(--surface-2)" />
      <motion.rect x={x} y={y} width={C} height={C} rx={3} fill="var(--accent)" initial={animate ? { opacity: pos(from) } : false} animate={{ opacity: pos(v) }} transition={tr} />
      <motion.rect x={x} y={y} width={C} height={C} rx={3} fill="var(--err)" initial={animate ? { opacity: neg(from) } : false} animate={{ opacity: neg(v) }} transition={tr} />
      {flash && (
        <motion.rect x={x - 1.5} y={y - 1.5} width={C + 3} height={C + 3} rx={4} fill="none" stroke="var(--fg)" strokeWidth={1.5}
          initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 0] }} transition={{ delay: at * k, duration: T.updDur * 1.2 * k }} />
      )}
    </g>
  );
}

/**
 * The model's weights as a heatmap: rows = 8 hidden units, columns = the 12 input words, plus the output head.
 * In LoRA mode, B (8×r) sits left of W and A (r×12) below it, so B·A visibly lands on W.
 */
export function WeightGrid({ prev, cur, method, batch, animate, k, animKey }: {
  prev: Snapshot; cur: Snapshot; method: Method; batch: Example[]; animate: boolean; k: number; animKey: string;
}) {
  const lora = method === "lora" && cur.lora ? cur.lora : null;
  const r = lora?.A.length ?? 0;
  const bw = lora ? r * P + 12 : 0;
  const x0 = 8 + bw + 24, y0 = LABEL_H;
  const headX = x0 + D * P + 28;
  const W = headX + P + 12, Hh = y0 + H * P + (lora ? 16 + r * P + 22 : 30);
  const noteY = y0 + H * P + (lora ? 16 + r * P + 16 : 16);
  const Wn = effectiveW(cur.params, cur.lora), Wp = effectiveW(prev.params, prev.lora);
  const big = (a: number, b: number) => animate && Math.abs(a - b) > 0.08;
  const trainW = method === "full", trainHead = method !== "lora";
  const at = T.update;
  // The gradient sweeps backwards: from the output head, across W (or only the head when W is frozen).
  const sweepFrom = headX + P, sweepTo = method === "head" ? headX - 6 : lora ? 8 : x0;
  return (
    <svg viewBox={`0 0 ${W} ${Hh}`} className="h-auto w-full" role="img"
      aria-label={`Weight heatmap at epoch ${cur.epoch}: ${H} hidden units by ${D} words${lora ? `, with a rank-${r} LoRA adapter` : ""}`}>
      {VOCAB.map((w, i) => (
        <text key={w} x={x0 + i * P + C / 2} y={y0 - 6} fontSize={10} fill="var(--fg)" fillOpacity={0.8} transform={`rotate(-55 ${x0 + i * P + C / 2} ${y0 - 6})`}
          fontWeight={w === "long" || w === "unpredictable" ? 700 : 400}>{w}</text>
      ))}
      {Array.from({ length: H }, (_, j) => (
        <text key={j} x={x0 - 4} y={y0 + j * P + 14} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">h{j + 1}</text>
      ))}
      {/* Input columns light up as each training example flows in. */}
      {animate && batch.map((e, n) => e.words.map((w) => {
        const i = VOCAB.indexOf(w);
        return (
          <motion.circle key={`${animKey}-${n}-${w}`} cx={x0 + i * P + C / 2} r={4} fill={e.words.some((x) => TOPICS.movie.includes(x)) ? "var(--violet)" : "var(--accent)"}
            initial={{ cy: 4, opacity: 0 }} animate={{ cy: [4, y0 + H * P], opacity: [0, 1, 1, 0] }}
            transition={{ delay: (T.feed + n * T.feedStagger) * k, duration: T.feedDur * k, ease: "easeIn", opacity: { delay: (T.feed + n * T.feedStagger) * k, duration: T.feedDur * k, times: [0, 0.15, 0.8, 1] } }} />
        );
      }))}
      {Wn.map((row, j) => row.map((v, i) => (
        <Cell key={`${animKey}-w${j}-${i}`} x={x0 + i * P} y={y0 + j * P} v={v} from={Wp[j][i]} at={at} k={k} animate={animate} flash={big(v, Wp[j][i])} />
      )))}
      {!trainW && (
        <g>
          <rect x={x0 - 3} y={y0 - 3} width={D * P + 4} height={H * P + 4} rx={5} fill={method === "head" ? "var(--surface)" : "none"} fillOpacity={0.45} stroke="var(--muted)" strokeDasharray="4 3" />
          <g transform={`translate(${x0 + D * P - 14}, ${noteY - 10})`}><Lock size={12} color="var(--muted)" /></g>
          <text x={x0 + D * P - 18} y={noteY} fontSize={10} textAnchor="end" fill="var(--muted)">{lora ? "W frozen · cells show W + B·A" : "W frozen"}</text>
        </g>
      )}
      {/* Output head w₂ */}
      <text x={headX + C / 2} y={y0 - 8} fontSize={10} textAnchor="middle" fill="var(--fg)" fillOpacity={0.8}>head</text>
      {cur.params.w2.map((v, j) => (
        <Cell key={`${animKey}-h${j}`} x={headX} y={y0 + j * P} v={v} from={prev.params.w2[j]} at={at} k={k} animate={animate} flash={big(v, prev.params.w2[j])} />
      ))}
      {!trainHead && <g transform={`translate(${headX + 4}, ${noteY - 10})`}><Lock size={12} color="var(--muted)" /></g>}
      {lora && prev.lora && (
        <g>
          <text x={8} y={y0 - 8} fontSize={10} fill="var(--violet)" fontWeight={600}>B</text>
          {lora.B.map((row, j) => row.map((v, c) => (
            <Cell key={`${animKey}-b${j}-${c}`} x={8 + c * P} y={y0 + j * P} v={v} from={prev.lora!.B[j][c]} at={at} k={k} animate={animate} flash={big(v, prev.lora!.B[j][c])} />
          )))}
          <text x={x0 - 34} y={y0 + H * P + 16 + (r * P) / 2 + 4} fontSize={10} textAnchor="end" fill="var(--violet)" fontWeight={600}>A</text>
          {lora.A.map((row, c) => row.map((v, i) => (
            <Cell key={`${animKey}-a${c}-${i}`} x={x0 + i * P} y={y0 + H * P + 16 + c * P} v={v} from={prev.lora!.A[c][i]} at={at} k={k} animate={animate} flash={big(v, prev.lora!.A[c][i])} />
          )))}
          <rect x={6} y={y0 - 2} width={r * P} height={H * P} rx={4} fill="none" stroke="var(--violet)" strokeWidth={1.5} />
          <rect x={x0 - 2} y={y0 + H * P + 14} width={D * P} height={r * P} rx={4} fill="none" stroke="var(--violet)" strokeWidth={1.5} />
        </g>
      )}
      {(trainW || trainHead) && <rect x={(trainW ? x0 : headX) - 3} y={y0 - 3} width={trainW ? headX - x0 + P + 2 : P + 2} height={H * P + 4} rx={5} fill="none" stroke="var(--accent)" strokeOpacity={0.5} />}
      {animate && (
        <motion.rect key={`${animKey}-sweep`} y={y0 - 4} width={26} height={H * P + (lora ? 16 + r * P : 0) + 8} rx={6} fill="var(--warn)"
          initial={{ x: sweepFrom, opacity: 0 }} animate={{ x: [sweepFrom, sweepTo], opacity: [0, 0.35, 0.35, 0] }}
          transition={{ delay: T.grad * k, duration: T.gradDur * k, ease: "easeInOut", opacity: { delay: T.grad * k, duration: T.gradDur * k, times: [0, 0.15, 0.8, 1] } }} />
      )}
    </svg>
  );
}

export type Curve = { key: "trainLoss" | "valLoss" | "origLoss"; name: string; color: string; dashed?: boolean };
export const CURVES: Curve[] = [
  { key: "trainLoss", name: "phone · train (8 examples)", color: "var(--muted)", dashed: true },
  { key: "valLoss", name: "phone · validation (new task)", color: "var(--accent)" },
  { key: "origLoss", name: "movie · held out (original task)", color: "var(--err)" },
];

/** Three loss curves; on an epoch step the newest segment draws itself in sync with the weight update. */
export function LossChart({ history, from, to, k, animKey, animate, best }: {
  history: Snapshot[]; from: number; to: number; k: number; animKey: string; animate: boolean; best?: number;
}) {
  const Wd = 300, Ht = 230, L = 34, R = 8, Tp = 10, B = 30;
  const N = Math.max(1, history.length - 1);
  const all = history.flatMap((h) => CURVES.map((c) => h.m[c.key])).filter(Number.isFinite);
  const top = Math.min(3, Math.max(0.5, ...all) * 1.05);
  const sx = scale(0, N, L, Wd - R), sy = (v: number) => scale(0, top, Ht - B, Tp)(Math.min(top, Number.isFinite(v) ? v : top));
  const pts = (c: Curve, a: number, b: number) => history.slice(a, b + 1).map((h) => `${sx(h.epoch)},${sy(h.m[c.key])}`).join(" ");
  const ticks = [0, top / 2, top];
  return (
    <svg viewBox={`0 0 ${Wd} ${Ht}`} className="h-auto w-full" role="img" aria-label={`Loss by epoch up to epoch ${to}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={Wd - R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={L - 4} y={sy(t) + 3} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{t.toFixed(1)}</text>
        </g>
      ))}
      <text x={L} y={Ht - 16} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">0</text>
      <text x={Wd - R} y={Ht - 16} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{N}</text>
      <text x={(L + Wd) / 2} y={Ht - 4} fontSize={10} textAnchor="middle" fill="var(--muted)">epoch → loss (lower is better)</text>
      {Math.max(...all) > top && <text x={Wd - R} y={Tp - 2} fontSize={9} textAnchor="end" fill="var(--err)">values above {top.toFixed(0)} clipped</text>}
      {best !== undefined && (
        <g>
          <line x1={sx(best)} x2={sx(best)} y1={Tp} y2={Ht - B} stroke="var(--accent)" strokeDasharray="3 3" />
          <text x={sx(best) + 3} y={Tp + 9} fontSize={9} fill="var(--accent)">best val</text>
        </g>
      )}
      {CURVES.map((c) => {
        const head = history[to], tail = history[from];
        return (
          <g key={c.key}>
            {from > 0 && <polyline points={pts(c, 0, from)} fill="none" stroke={c.color} strokeWidth={2} strokeDasharray={c.dashed ? "4 3" : undefined} />}
            {to > from && (
              <motion.polyline key={animKey} points={pts(c, from, to)} fill="none" stroke={c.color} strokeWidth={2.5} strokeDasharray={c.dashed ? "4 3" : undefined}
                initial={animate ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={{ delay: T.update * k, duration: T.updDur * k, ease: "linear" }} />
            )}
            <motion.circle key={`${animKey}-dot`} r={4} fill={c.color} stroke="var(--surface)" strokeWidth={1.5}
              initial={animate ? { cx: sx(tail.epoch), cy: sy(tail.m[c.key]) } : false} animate={{ cx: sx(head.epoch), cy: sy(head.m[c.key]) }}
              transition={{ delay: T.update * k, duration: T.updDur * k, ease }} />
          </g>
        );
      })}
    </svg>
  );
}
