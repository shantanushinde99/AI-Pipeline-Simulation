import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type SurfaceId = "bowl" | "saddle";
export type Vec = [number, number];
export type LrPreset = "slow" | "good" | "oscillating" | "diverging";
export type Regime = LrPreset | "noisy";
export type Surface = {
  id: SurfaceId; label: string; tex: string; gradTex: string;
  f: (p: Vec) => number; grad: (p: Vec) => Vec;
  domain: { x: Vec; y: Vec }; start: Vec; minima: Vec[]; saddle?: Vec;
  lr: Record<LrPreset, number>;
};

export const SURFACES: Record<SurfaceId, Surface> = {
  bowl: {
    id: "bowl", label: "Elongated bowl (ill-conditioned)",
    tex: "f(x, y) = \\tfrac{1}{2}\\,(x^2 + 10\\,y^2)", gradTex: "\\nabla f = (x,\\ 10\\,y)",
    f: ([x, y]) => 0.5 * (x * x + 10 * y * y), grad: ([x, y]) => [x, 10 * y],
    domain: { x: [-5, 5], y: [-3, 3] }, start: [-4.5, 2.5], minima: [[0, 0]],
    // Curvatures are 1 and 10, so plain GD is stable only for η < 2/10.
    lr: { slow: 0.02, good: 0.12, oscillating: 0.18, diverging: 0.21 },
  },
  saddle: {
    id: "saddle", label: "Saddle between two valleys",
    tex: "f(x, y) = \\tfrac{1}{2}x^2 + \\tfrac{1}{16}(y^2 - 4)^2", gradTex: "\\nabla f = \\big(x,\\ \\tfrac{1}{4}\\,y\\,(y^2 - 4)\\big)",
    f: ([x, y]) => 0.5 * x * x + ((y * y - 4) ** 2) / 16, grad: ([x, y]) => [x, (y * (y * y - 4)) / 4],
    domain: { x: [-5, 5], y: [-3.5, 3.5] }, start: [-4.5, 0.15], minima: [[0, 2], [0, -2]], saddle: [0, 0],
    // Curvature at the minima is 1 (x) and 2 (y): stable for η < 1. Between 1 and 2 the quartic walls keep the
    // bouncing bounded (y oscillates chaotically); above 2 even the x-direction (curvature 1) blows up.
    lr: { slow: 0.05, good: 0.4, oscillating: 0.9, diverging: 2.2 },
  },
};

export type GDConfig = { surface: SurfaceId; lr: number; momentum: boolean; beta: number; stochastic: boolean; noise: number; steps: number; seed: number };
export const DEFAULT_GD: GDConfig = { surface: "bowl", lr: SURFACES.bowl.lr.good, momentum: false, beta: 0.8, stochastic: false, noise: 3, steps: 25, seed: 4 };

export type Iter = {
  t: number; p: Vec; f: number;
  /** Gradient actually used at p (noisy when stochastic), the true gradient, velocity after this step, and the move applied. */
  g: Vec; gTrue: Vec; v: Vec; move: Vec;
};

const LIMIT = 60;

/** Runs GD from the surface's start. Iteration t holds the point *before* step t+1 and the step taken from it. */
export function run(raw: GDConfig) {
  const cfg = { ...raw, lr: clamp(raw.lr, 0, 5), beta: clamp(raw.beta, 0, 0.99), noise: clamp(raw.noise, 0, 20), steps: clamp(Math.round(raw.steps), 1, 80) };
  const s = SURFACES[cfg.surface];
  const r = rng(cfg.seed);
  let p: Vec = [...s.start];
  let v: Vec = [0, 0];
  const iters: Iter[] = [];
  let diverged = false;
  for (let t = 0; t <= cfg.steps; t++) {
    const gTrue = s.grad(p);
    // Stochastic: a mini-batch estimate = true gradient + seeded Gaussian noise.
    const g: Vec = cfg.stochastic ? [gTrue[0] + cfg.noise * r.gaussian(), gTrue[1] + cfg.noise * r.gaussian()] : gTrue;
    const nv: Vec = cfg.momentum ? [cfg.beta * v[0] + g[0], cfg.beta * v[1] + g[1]] : g;
    const move: Vec = t < cfg.steps ? [-cfg.lr * nv[0], -cfg.lr * nv[1]] : [0, 0];
    iters.push({ t, p, f: s.f(p), g, gTrue, v: nv, move });
    if (!Number.isFinite(s.f(p)) || Math.abs(p[0]) > LIMIT || Math.abs(p[1]) > LIMIT) { diverged = true; iters[iters.length - 1].move = [0, 0]; break; }
    if (t === cfg.steps) break;
    v = nv;
    p = [p[0] + move[0], p[1] + move[1]];
  }
  const last = iters[iters.length - 1];
  const f0 = iters[0].f;
  // Oscillation: the step along y (the steep direction on both surfaces) keeps reversing.
  const flips = iters.slice(1).filter((it, i) => it.move[1] * iters[i].move[1] < 0 && Math.abs(it.move[1]) > 1e-3).length;
  const regime: Regime = diverged || last.f > f0 ? "diverging" : cfg.stochastic ? "noisy" : flips >= Math.min(6, iters.length / 3) ? "oscillating" : last.f > 0.05 * f0 ? "slow" : "good";
  return { cfg, surface: s, iters, diverged, regime, flips };
}

export type GDState = { kind: "start" | "step" | "summary"; t: number; headline: string; sub: string };

const f3 = (v: number) => (Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(3));
const p3 = (v: number) => (v < 0 ? `(${f3(v)})` : f3(v));

export function simulate(cfg: GDConfig) {
  const res = run(cfg);
  const { iters, surface: s } = res;
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "gd", message, tex, severity });
  const steps: SimulationStep<GDState>[] = [{
    id: "start", title: "Drop the ball",
    description: `Start at (${s.start.join(", ")}) where the loss is ${f3(iters[0].f)}. Darker = lower loss. The minimum${s.minima.length > 1 ? "s are" : " is"} marked ★.`,
    state: { kind: "start", t: 0, headline: `The ball starts at (${s.start.join(", ")}), loss ${f3(iters[0].f)}`, sub: "Each step: measure the slope (gradient), then move a little downhill. Darker = lower." },
    events: [ev("f0", `f(start) = ${f3(iters[0].f)}`, `f(${s.start.join(",\\ ")}) = ${f3(iters[0].f)}`)],
  }];
  for (let t = 0; t < iters.length - 1; t++) {
    const a = iters[t], b = iters[t + 1];
    const up = b.f > a.f;
    const fell = b.f < a.f;
    const ratio = a.f > 0 ? b.f / a.f : 1;
    const out = Math.abs(b.p[0]) > LIMIT || Math.abs(b.p[1]) > LIMIT || !Number.isFinite(b.f);
    steps.push({
      id: `step-${t + 1}`, title: `Step ${t + 1}`,
      description: `At (${f3(a.p[0])}, ${f3(a.p[1])}) the gradient is (${f3(a.g[0])}, ${f3(a.g[1])}). Move by −η·${res.cfg.momentum ? "v" : "∇f"} = (${f3(a.move[0])}, ${f3(a.move[1])}). Loss ${f3(a.f)} → ${f3(b.f)}.`,
      state: {
        kind: "step", t: t + 1,
        headline: out ? `Step ${t + 1}: the ball flew off the surface. Diverged!` : `Step ${t + 1}: loss ${f3(a.f)} → ${f3(b.f)} ${up ? "↑ (overshot!)" : fell ? "↓" : "→"}`,
        sub: up ? "The step was so large it landed higher up the opposite wall." : ratio > 0.97 ? "Tiny progress: either the slope is flat here or the learning rate is small." : "Gradient arrow (red) points uphill; the ball moves the opposite way (green).",
      },
      events: [
        ev(`g${t}`, `∇f = (${f3(a.g[0])}, ${f3(a.g[1])})`, `\\nabla f = (${f3(a.g[0])},\\ ${f3(a.g[1])})${res.cfg.stochastic ? "\\ \\text{(noisy)}" : ""}`),
        ev(`m${t}`, `move = (${f3(a.move[0])}, ${f3(a.move[1])})`, `\\Delta = -${res.cfg.lr}\\,${res.cfg.momentum ? "v" : "\\nabla f"} = (${f3(a.move[0])},\\ ${f3(a.move[1])})`),
        ev(`f${t}`, `loss ${f3(a.f)} → ${f3(b.f)}`, undefined, up ? "warning" : undefined),
      ],
    });
  }
  const last = iters[iters.length - 1];
  const verdict: Record<Regime, string> = {
    slow: "Too slow: after all these steps the ball is still far from the bottom.",
    good: "Good learning rate: fast, steady progress to the bottom.",
    oscillating: "Oscillating: the ball zig-zags across the narrow valley but still converges.",
    diverging: "Diverging: each step overshoots more than the last and the loss explodes.",
    noisy: "Noisy gradients: the path jitters but trends downhill, then hovers around the bottom.",
  };
  steps.push({
    id: "summary", title: "Summary",
    description: `${iters.length - 1} steps: loss ${f3(iters[0].f)} → ${f3(last.f)}. ${verdict[res.regime]}`,
    state: { kind: "summary", t: iters.length - 1, headline: verdict[res.regime], sub: `Loss ${f3(iters[0].f)} → ${f3(last.f)} in ${iters.length - 1} steps with η = ${res.cfg.lr}${res.cfg.momentum ? `, momentum β = ${res.cfg.beta}` : ""}${res.cfg.stochastic ? ", noisy gradients" : ""}.` },
    events: [ev("sum", `final loss ${f3(last.f)} at (${f3(last.p[0])}, ${f3(last.p[1])})`, `f_{\\text{final}} = ${f3(last.f)}`, res.regime === "diverging" ? "error" : res.regime === "good" ? undefined : "warning")],
  });
  return { ...res, steps };
}

export { f3, p3 };
