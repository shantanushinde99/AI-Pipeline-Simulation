import { clamp, sigmoid } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Preset = "separable" | "overlap" | "xor";
export type LogConfig = { preset: Preset; lr: number; lambda: number; epochs: number; seed: number };
export const DEFAULT_LOG: LogConfig = { preset: "separable", lr: 0.5, lambda: 0, epochs: 60, seed: 5 };
export const PRESETS: { id: Preset; label: string }[] = [
  { id: "separable", label: "Separable blobs" },
  { id: "overlap", label: "Overlapping blobs" },
  { id: "xor", label: "XOR corners (not linearly separable)" },
];
/** The plane shown is [-BOX, BOX]². */
export const BOX = 3;

export type Pt = { x: [number, number]; y: 0 | 1 };
export type Params = { w1: number; w2: number; b: number };
/** Fixed, deliberately poor starting weights so the boundary visibly moves. */
export const INIT: Params = { w1: -0.6, w2: 0.9, b: 0.5 };

export function makeData(preset: Preset, seed: number): Pt[] {
  const r = rng(seed);
  const blob = (cx: number, cy: number, sd: number, y: 0 | 1, n: number): Pt[] =>
    Array.from({ length: n }, () => ({ x: [clamp(cx + sd * r.gaussian(), -2.8, 2.8), clamp(cy + sd * r.gaussian(), -2.8, 2.8)] as [number, number], y }));
  if (preset === "xor")
    return [...blob(1.5, 1.5, 0.45, 1, 5), ...blob(-1.5, -1.5, 0.45, 1, 5), ...blob(-1.5, 1.5, 0.45, 0, 5), ...blob(1.5, -1.5, 0.45, 0, 5)];
  const sd = preset === "separable" ? 0.55 : 1.05;
  const d = preset === "separable" ? 1.2 : 0.55;
  return [...blob(d, d * 0.8, sd, 1, 10), ...blob(-d, -d * 0.8, sd, 0, 10)];
}

export const prob = (w: Params, x: readonly number[]) => sigmoid(w.w1 * x[0] + w.w2 * x[1] + w.b);

/** Mean binary cross-entropy + (λ/2)·‖w‖² (the bias is not regularized). */
export function loss(pts: Pt[], w: Params, lambda: number) {
  const eps = 1e-12;
  const ce = pts.reduce((s, p) => {
    const q = prob(w, p.x);
    return s - (p.y ? Math.log(Math.max(q, eps)) : Math.log(Math.max(1 - q, eps)));
  }, 0) / Math.max(1, pts.length);
  return ce + (lambda / 2) * (w.w1 ** 2 + w.w2 ** 2);
}

/** Analytic gradient: ∂L/∂w = mean((p − y)·x) + λw, ∂L/∂b = mean(p − y). */
export function gradient(pts: Pt[], w: Params, lambda: number): Params {
  const n = Math.max(1, pts.length);
  let g1 = 0, g2 = 0, gb = 0;
  for (const p of pts) {
    const e = prob(w, p.x) - p.y;
    g1 += e * p.x[0];
    g2 += e * p.x[1];
    gb += e;
  }
  return { w1: g1 / n + lambda * w.w1, w2: g2 / n + lambda * w.w2, b: gb / n };
}

export const accuracy = (pts: Pt[], w: Params) => pts.filter((p) => (prob(w, p.x) >= 0.5 ? 1 : 0) === p.y).length / Math.max(1, pts.length);

export type Epoch = Params & { epoch: number; loss: number; acc: number; grad: Params };

export function train(pts: Pt[], lr: number, lambda: number, epochs: number): Epoch[] {
  let w = { ...INIT };
  const out: Epoch[] = [];
  for (let e = 0; e <= epochs; e++) {
    const g = gradient(pts, w, lambda);
    out.push({ ...w, epoch: e, loss: loss(pts, w, lambda), acc: accuracy(pts, w), grad: g });
    w = { w1: w.w1 - lr * g.w1, w2: w.w2 - lr * g.w2, b: w.b - lr * g.b };
  }
  return out;
}

/** Epochs shown as steps: dense at first (where the boundary moves most), then sparser. */
export function snapshots(epochs: number) {
  return Array.from(new Set([0, 1, 2, 3, 5, 8, 12, 18, 26, 38, 55, 80, 120, 160, epochs].filter((e) => e <= epochs))).sort((a, b) => a - b);
}

/** Endpoints of w1·x + w2·y + b = 0 clipped to the box, or null if it misses the box. */
export function boundary(w: Params): [[number, number], [number, number]] | null {
  const pts: [number, number][] = [];
  if (Math.abs(w.w2) > 1e-9) for (const x of [-BOX, BOX]) { const y = -(w.w1 * x + w.b) / w.w2; if (Math.abs(y) <= BOX + 1e-9) pts.push([x, y]); }
  if (Math.abs(w.w1) > 1e-9) for (const y of [-BOX, BOX]) { const x = -(w.w2 * y + w.b) / w.w1; if (Math.abs(x) <= BOX + 1e-9) pts.push([x, y]); }
  const uniq = pts.filter((p, i) => pts.findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-6) === i);
  if (uniq.length < 2) return null;
  // Order endpoints left→right so tweens between snapshots don't flip the line.
  const [a, b] = uniq[0][0] <= uniq[1][0] ? [uniq[0], uniq[1]] : [uniq[1], uniq[0]];
  return [a, b];
}

export type LogState = { kind: "data" | "init" | "epoch" | "result"; epoch: number; prevEpoch: number; headline: string; sub: string };

export function simulate(raw: LogConfig) {
  const cfg = { ...raw, lr: clamp(raw.lr, 0, 20), lambda: clamp(raw.lambda, 0, 5), epochs: clamp(Math.round(raw.epochs), 1, 200) };
  const pts = makeData(cfg.preset, cfg.seed);
  const history = train(pts, cfg.lr, cfg.lambda, cfg.epochs);
  const snaps = snapshots(cfg.epochs);
  const f = (v: number, d = 3) => v.toFixed(d);
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "train", message, tex, severity });
  const h0 = history[0];
  const last = history[cfg.epochs];
  // Unstable = the loss rose on several epochs (overshooting), or blew up.
  const rises = history.filter((h, i) => i > 0 && h.loss > history[i - 1].loss + 1e-9).length;
  const diverged = !Number.isFinite(last.loss) || rises >= 3;
  const steps: SimulationStep<LogState>[] = [
    {
      id: "data", title: "Two classes of points",
      description: `${pts.length} labelled points: circles are class 1, squares are class 0. Logistic regression will learn a straight boundary between them.`,
      state: { kind: "data", epoch: 0, prevEpoch: 0, headline: "Two classes of points appear on the plane", sub: "Circles = class 1, squares = class 0. We want P(class 1 | x) for any point." },
      events: [ev("n", `${pts.filter((p) => p.y).length} class-1 and ${pts.filter((p) => !p.y).length} class-0 points (${cfg.preset})`)],
    },
    {
      id: "init", title: "Starting weights",
      description: `Start from w = (${INIT.w1}, ${INIT.w2}), b = ${INIT.b}. The model scores z = w·x + b, and σ(z) turns it into a probability. The line is where σ(z) = 0.5.`,
      state: { kind: "init", epoch: 0, prevEpoch: 0, headline: `A first guess: loss ${f(h0.loss)}, accuracy ${(h0.acc * 100).toFixed(0)}%`, sub: "Shading = predicted probability of class 1. The line is where it equals 0.5." },
      events: [ev("l0", `loss = ${f(h0.loss, 4)}`, `\\mathcal{L}_0 = ${f(h0.loss, 4)}`), ev("g0", `gradient = (${f(h0.grad.w1)}, ${f(h0.grad.w2)}, ${f(h0.grad.b)})`, `\\nabla\\mathcal{L} = (${f(h0.grad.w1)},\\ ${f(h0.grad.w2)},\\ ${f(h0.grad.b)})`)],
    },
    ...snaps.slice(1).map((e, i): SimulationStep<LogState> => {
      const h = history[e], prev = history[snaps[i]];
      const delta = h.loss - prev.loss;
      return {
        id: `epoch-${e}`, title: `Epoch ${e}`,
        description: `${e - snaps[i]} gradient step${e - snaps[i] > 1 ? "s" : ""} (learning rate ${cfg.lr}) moved the weights to w = (${f(h.w1)}, ${f(h.w2)}), b = ${f(h.b)}. Loss ${f(prev.loss)} → ${f(h.loss)}.`,
        state: {
          kind: "epoch", epoch: e, prevEpoch: snaps[i],
          headline: `Epoch ${e}: loss ${f(h.loss)} ${!Number.isFinite(delta) ? "" : delta < -1e-4 ? "↓" : delta > 1e-4 ? "↑" : "→"}, accuracy ${(h.acc * 100).toFixed(0)}%`,
          sub: delta > 1e-4 ? "The loss went UP: the step was too big and overshot." : Math.abs(delta) <= 1e-4 ? "Barely moving: the gradient is (nearly) zero here." : "Each step moves w opposite the gradient, so the boundary turns and slides toward a better split.",
        },
        events: [
          ev(`u${e}`, `w ← w − η·∇L`, `w \\leftarrow w - ${cfg.lr}\\,\\nabla_w\\mathcal{L}`),
          ev(`l${e}`, `loss = ${f(h.loss, 4)}`, `\\mathcal{L}_{${e}} = ${f(h.loss, 4)}`, delta > 1e-4 ? "warning" : undefined),
        ],
      };
    }),
    {
      id: "result", title: "Result",
      description: cfg.preset === "xor"
        ? `After ${cfg.epochs} epochs accuracy is ${(last.acc * 100).toFixed(0)}% and loss ${f(last.loss)} (chance level is ln 2 ≈ 0.693). No straight line separates XOR, so a linear model cannot solve it.`
        : `After ${cfg.epochs} epochs: loss ${f(last.loss)}, accuracy ${(last.acc * 100).toFixed(0)}%.`,
      state: {
        kind: "result", epoch: cfg.epochs, prevEpoch: cfg.epochs,
        headline: diverged ? `Unstable training: the loss went up on ${rises} of ${cfg.epochs} epochs` : cfg.preset === "xor" ? `Stuck at ${(last.acc * 100).toFixed(0)}% accuracy: no straight line can split XOR` : `Trained: ${(last.acc * 100).toFixed(0)}% accuracy, loss ${f(last.loss)}`,
        sub: diverged ? "Lower the learning rate and replay." : cfg.preset === "xor" ? "Diagonal corners share a class. You need a non-linear model (features like x₁·x₂, or a hidden layer)." : "The final line is the model: everything on the blue side is predicted class 1.",
      },
      events: [
        ev("acc", `accuracy = ${(last.acc * 100).toFixed(1)}%`, undefined, cfg.preset === "xor" ? "warning" : undefined),
        ...(diverged ? [ev("div", `Loss rose on ${rises} epochs: learning rate too high (overshooting)`, undefined, "error")] : []),
      ],
    },
  ];
  return { cfg, pts, history, snaps, steps };
}
