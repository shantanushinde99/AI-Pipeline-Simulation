import { clamp, sigmoid } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type ActName = "relu" | "leaky" | "sigmoid" | "tanh" | "gelu";

export const LEAKY_ALPHA = 0.1;
const C = Math.sqrt(2 / Math.PI);

/** GELU, tanh approximation (the form used by GPT-2/BERT code). */
function gelu(z: number) {
  return 0.5 * z * (1 + Math.tanh(C * (z + 0.044715 * z ** 3)));
}
function geluD(z: number) {
  const t = Math.tanh(C * (z + 0.044715 * z ** 3));
  return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * C * (1 + 3 * 0.044715 * z * z);
}

/** f = activation, d = its derivative f'(z). `tex`/`dtex` are LaTeX. ReLU's derivative at exactly 0 is taken as 0 (as PyTorch does). */
export const ACTS: Record<ActName, { label: string; color: string; f: (z: number) => number; d: (z: number) => number; tex: string; dtex: string; name: string }> = {
  relu: { label: "ReLU", color: "var(--accent)", f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0), name: "\\operatorname{ReLU}", tex: "\\operatorname{ReLU}(z) = \\max(0, z)", dtex: "\\operatorname{ReLU}'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z \\le 0 \\end{cases}" },
  leaky: { label: `Leaky ReLU (α=${LEAKY_ALPHA})`, color: "var(--ok)", f: (z) => (z > 0 ? z : LEAKY_ALPHA * z), d: (z) => (z > 0 ? 1 : LEAKY_ALPHA), name: "\\operatorname{LReLU}", tex: `\\operatorname{LReLU}(z) = \\max(${LEAKY_ALPHA} z, z)`, dtex: `\\operatorname{LReLU}'(z) = \\begin{cases} 1 & z > 0 \\\\ ${LEAKY_ALPHA} & z \\le 0 \\end{cases}` },
  sigmoid: { label: "Sigmoid", color: "var(--violet)", f: sigmoid, d: (z) => sigmoid(z) * (1 - sigmoid(z)), name: "\\sigma", tex: "\\sigma(z) = \\frac{1}{1 + e^{-z}}", dtex: "\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z))" },
  tanh: { label: "Tanh", color: "var(--warn)", f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2, name: "\\tanh", tex: "\\tanh(z) = \\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}", dtex: "\\tanh'(z) = 1 - \\tanh^2(z)" },
  gelu: { label: "GELU", color: "var(--err)", f: gelu, d: geluD, name: "\\operatorname{GELU}", tex: "\\operatorname{GELU}(z) \\approx \\tfrac{z}{2}\\big(1 + \\tanh\\big(\\sqrt{2/\\pi}\\,(z + 0.044715 z^3)\\big)\\big)", dtex: "\\operatorname{GELU}'(z) \\;(\\text{derivative of the line above})" },
};
export const ACT_NAMES = Object.keys(ACTS) as ActName[];

/** Below this |f'(z)|, we call the gradient "vanishing": almost no learning signal passes back. */
export const SATURATION = 0.05;
export type Zone = "healthy" | "vanishing" | "dead";
export function zoneOf(act: ActName, z: number): Zone {
  const d = Math.abs(ACTS[act].d(z));
  return d === 0 ? "dead" : d < SATURATION ? "vanishing" : "healthy";
}

export type NeuronInput = { x: number[]; w: number[]; b: number; act: ActName };
export const LIMITS = { x: 2, w: 3, b: 5 };

export function sanitize(inp: NeuronInput): NeuronInput {
  const n = Math.min(inp.x.length, inp.w.length);
  const fix = (v: number, lim: number) => (Number.isFinite(v) ? clamp(v, -lim, lim) : 0);
  return { x: inp.x.slice(0, n).map((v) => fix(v, LIMITS.x)), w: inp.w.slice(0, n).map((v) => fix(v, LIMITS.w)), b: fix(inp.b, LIMITS.b), act: inp.act in ACTS ? inp.act : "relu" };
}

export type NeuronResult = { products: number[]; z: number; a: number; d: number; zone: Zone };
export function computeNeuron(raw: NeuronInput): NeuronResult {
  const { x, w, b, act } = sanitize(raw);
  const products = x.map((v, i) => v * w[i]);
  const z = products.reduce((s, p) => s + p, b);
  return { products, z, a: ACTS[act].f(z), d: ACTS[act].d(z), zone: zoneOf(act, z) };
}

/** The same z through every activation. */
export const compareAll = (z: number) => ACT_NAMES.map((act) => ({ act, a: ACTS[act].f(z), d: ACTS[act].d(z), zone: zoneOf(act, z) }));

/**
 * Backprop multiplies one f'(z) per layer. If every layer sat at this same z, the gradient reaching layer n back
 * would be scaled by f'(z)^n (weights ignored). A teaching simplification of the vanishing-gradient effect.
 */
export const chainedGradient = (d: number, layers = 5) => Array.from({ length: layers }, (_, i) => d ** (i + 1));

/** Intervals of [lo, hi] where |f'(z)| < SATURATION (sampled), used to shade dead/saturated regions. */
export function flatRegions(act: ActName, lo: number, hi: number, samples = 240): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i <= samples; i++) {
    const z = lo + ((hi - lo) * i) / samples;
    const flat = Math.abs(ACTS[act].d(z)) < SATURATION;
    if (flat && start === null) start = z;
    if ((!flat || i === samples) && start !== null) {
      out.push([start, z]);
      start = null;
    }
  }
  return out;
}

export type Phase = "inputs" | "sum" | "activate" | "output" | "derivative" | "compare";
export type NeuronState = { phase: Phase; input: NeuronInput; r: NeuronResult };
export const PHASES: Phase[] = ["inputs", "sum", "activate", "output", "derivative", "compare"];

const p = (v: number, d = 2) => (v < 0 ? `(${v.toFixed(d)})` : v.toFixed(d));

export function simulate(raw: NeuronInput): SimulationStep<NeuronState>[] {
  const input = sanitize(raw);
  const r = computeNeuron(input);
  const { x, w, b, act } = input;
  const A = ACTS[act];
  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });
  const sumTex = `z = ${x.map((v, i) => `${p(w[i])} \\cdot ${p(v)}`).join(" + ")} + ${p(b)} = ${r.z.toFixed(3)}`;
  const all = compareAll(r.z);
  const step = (phase: Phase, title: string, description: string, events: SimulationEvent[]): SimulationStep<NeuronState> => ({ id: phase, title, description, state: { phase, input, r }, events });

  return [
    step("inputs", "Inputs arrive", `${x.length} input values arrive, each on a connection with its own weight.`, [ev("in", `x = [${x.join(", ")}], w = [${w.join(", ")}], b = ${b}`)]),
    step("sum", "Weighted sum + bias", "Each input is multiplied by its weight; the products and the bias are added into one number z.", [
      ...x.map((v, i) => ev(`p${i}`, `x${i + 1}·w${i + 1} = ${v}·${w[i]} = ${r.products[i].toFixed(3)}`, `x_${i + 1} w_${i + 1} = ${p(v)} \\cdot ${p(w[i])} = ${r.products[i].toFixed(3)}`)),
      ev("z", `z = ${r.z.toFixed(3)}`, sumTex),
    ]),
    step("activate", `Activation: ${A.label}`, `z is passed through ${A.label}. The output is read off the curve at z.`, [
      ev("a", `a = ${act}(${r.z.toFixed(3)}) = ${r.a.toFixed(4)}`, `a = ${A.name}(${r.z.toFixed(3)}) = ${r.a.toFixed(4)}`),
    ]),
    step("output", "Output leaves the neuron", "The activation a is the neuron's output: the next layer receives a, not z.", [ev("out", `output a = ${r.a.toFixed(4)}`)]),
    step("derivative", "Gradient: slope at z", "During training, the learning signal is multiplied by f'(z). Flat parts of the curve pass almost nothing back.", [
      ev("d", `f'(z) = ${r.d.toFixed(4)}`, `${A.name}'(${r.z.toFixed(3)}) = ${r.d.toFixed(4)}`),
      ...(r.zone === "dead" ? [ev("dead", "Gradient is exactly 0: this ReLU is dead for this input and its weights get no update.", undefined, "warning")] : []),
      ...(r.zone === "vanishing" ? [ev("van", `|f'(z)| < ${SATURATION}: the neuron is saturated, so the gradient vanishes.`, undefined, "warning")] : []),
    ]),
    step("compare", "Same z, every activation", "The same weighted sum z through all five activations, with each output and slope.", all.map((c) => ev(`c-${c.act}`, `${c.act}(${r.z.toFixed(2)}) = ${c.a.toFixed(3)}, slope ${c.d.toFixed(3)} (${c.zone})`))),
  ];
}

export const PRESETS: { id: string; label: string; input: NeuronInput }[] = [
  { id: "healthy", label: "Healthy", input: { x: [1, 0.5, -1], w: [0.8, -0.4, 0.3], b: 0.5, act: "relu" } },
  { id: "dead", label: "Dead ReLU (bias −4)", input: { x: [1, 0.5, -1], w: [0.8, -0.4, 0.3], b: -4, act: "relu" } },
  { id: "saturated", label: "Saturated sigmoid (big weights)", input: { x: [1.5, 1, 1], w: [2, 2, 1.5], b: 1, act: "sigmoid" } },
  { id: "negative", label: "Negative z, GELU", input: { x: [-1, 1, 0.5], w: [1.2, -0.8, -0.6], b: 0.2, act: "gelu" } },
];
