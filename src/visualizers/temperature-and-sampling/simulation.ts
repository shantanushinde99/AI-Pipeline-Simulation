import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export const PROMPT = "The weather today is";
/** Fixed, hand-picked logits for the next token (a stand-in for a model's output layer). Sorted high → low. */
export const VOCAB = ["sunny", "cloudy", "rainy", "warm", "cold", "nice", "perfect", "purple"];
export const LOGITS = [3.2, 2.5, 2.1, 1.6, 1.1, 0.8, 0.2, -1.5];

export type SampleConfig = { T: number; topK: number; topP: number; seed: number; n: number };
export const DEFAULTS: SampleConfig = { T: 1, topK: 0, topP: 1, seed: 3, n: 200 };

export function sanitize(c: SampleConfig): SampleConfig {
  const num = (v: number, d: number) => (Number.isFinite(v) ? v : d);
  return {
    T: clamp(num(c.T, 1), 0, 3),
    topK: Math.round(clamp(num(c.topK, 0), 0, VOCAB.length)),
    topP: clamp(num(c.topP, 1), 0.01, 1),
    seed: Math.round(num(c.seed, 7)),
    n: Math.round(clamp(num(c.n, 200), 1, 2000)),
  };
}

/** Softmax with temperature. T = 0 is the limit T → 0: all mass on the argmax (greedy). Uses the max-subtraction trick. */
export function softmaxT(logits: number[], T: number) {
  const m = Math.max(...logits);
  if (T <= 0) {
    const i = logits.indexOf(m);
    return { scaled: null, expd: logits.map((_, j) => (j === i ? 1 : 0)), probs: logits.map((_, j) => (j === i ? 1 : 0)) };
  }
  const scaled = logits.map((z) => z / T);
  const ms = m / T;
  const expd = scaled.map((s) => Math.exp(s - ms));
  const sum = expd.reduce((a, b) => a + b, 0);
  return { scaled, expd, probs: expd.map((e) => e / sum) };
}

/** Keep the k most likely tokens (k = 0 → off). */
export function topKMask(probs: number[], k: number) {
  if (k <= 0 || k >= probs.length) return probs.map(() => true);
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  const keep = new Set(order.slice(0, k).map(([, i]) => i));
  return probs.map((_, i) => keep.has(i));
}

/** Nucleus: the smallest set of most-likely tokens whose probability sums to ≥ p. Always keeps at least one. */
export function topPMask(probs: number[], p: number) {
  const order = probs.map((q, i) => [q, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  const keep = new Set<number>();
  let cum = 0;
  const cumulative: { i: number; p: number; cum: number }[] = [];
  for (const [q, i] of order) {
    const before = cum;
    cum += q;
    cumulative.push({ i, p: q, cum });
    if (before < p - 1e-12 && q > 0) keep.add(i);
  }
  if (!keep.size) keep.add(order[0][1]);
  return { mask: probs.map((_, i) => keep.has(i)), cumulative };
}

export const renormalize = (probs: number[], mask: boolean[]) => {
  const kept = probs.reduce((s, p, i) => s + (mask[i] ? p : 0), 0);
  return { kept, final: probs.map((p, i) => (mask[i] && kept > 0 ? p / kept : 0)) };
};

/** Inverse-CDF sampling: walk the cumulative distribution until it passes u ∈ [0, 1). */
export function sampleIndex(final: number[], u: number) {
  let cum = 0;
  let last = 0;
  for (let i = 0; i < final.length; i++) {
    if (final[i] <= 0) continue;
    last = i;
    cum += final[i];
    if (u < cum) return i;
  }
  return last; // guards against rounding when u is within 1e-16 of 1
}

export const entropyBits = (p: number[]) => -p.reduce((s, q) => s + (q > 0 ? q * Math.log2(q) : 0), 0);
export const tvDistance = (a: number[], b: number[]) => 0.5 * a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);

export const CHECKPOINTS = 20;

export function pipeline(raw: SampleConfig) {
  const cfg = sanitize(raw);
  const { scaled, expd, probs } = softmaxT(LOGITS, cfg.T);
  const keepK = topKMask(probs, cfg.topK);
  const afterK = renormalize(probs, keepK).final;
  const topP = topPMask(afterK, cfg.topP);
  const mask = keepK.map((k, i) => k && topP.mask[i]);
  const { kept, final } = renormalize(probs, mask);
  const r = rng(cfg.seed);
  const u = r.next();
  const pick = sampleIndex(final, u);
  const draws = Array.from({ length: cfg.n }, () => sampleIndex(final, r.next()));
  const checkpoints = Array.from({ length: CHECKPOINTS + 1 }, (_, c) => {
    const upto = Math.round((cfg.n * c) / CHECKPOINTS);
    const counts = VOCAB.map(() => 0);
    for (let i = 0; i < upto; i++) counts[draws[i]]++;
    return { upto, counts, tv: upto ? tvDistance(counts.map((x) => x / upto), final) : 1 };
  });
  return { cfg, scaled, expd, probs, keepK, afterK, topP, mask, kept, final, u, pick, draws, checkpoints, greedy: LOGITS.indexOf(Math.max(...LOGITS)) };
}
export type Pipeline = ReturnType<typeof pipeline>;

export type Phase = "logits" | "scale" | "exp" | "normalize" | "topk" | "topp" | "renorm" | "sample" | "repeat";
export const PHASES: Phase[] = ["logits", "scale", "exp", "normalize", "topk", "topp", "renorm", "sample", "repeat"];
export type SampState = { phase: Phase; pl: Pipeline };

const f3 = (v: number) => v.toFixed(3);

export function simulate(raw: SampleConfig): SimulationStep<SampState>[] {
  const pl = pipeline(raw);
  const { cfg } = pl;
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "sample", message, severity, tex });
  const kOff = cfg.topK === 0 || cfg.topK >= VOCAB.length;
  const pOff = cfg.topP >= 1;
  const removedK = pl.keepK.filter((k) => !k).length;
  const removedP = pl.keepK.filter((k, i) => k && !pl.topP.mask[i]).length;
  const last = pl.checkpoints[CHECKPOINTS];
  const mk = (phase: Phase, title: string, description: string, events: SimulationEvent[]): SimulationStep<SampState> => ({ id: phase, title, description, state: { phase, pl }, events });
  return [
    mk("logits", "Raw logits from the model", `After "${PROMPT}", the model outputs one score (logit) per vocabulary token. Logits are unbounded and can be negative.`, [ev("l", VOCAB.map((v, i) => `${v} ${LOGITS[i]}`).join(", "))]),
    mk("scale", `Divide by temperature T = ${cfg.T}`, cfg.T === 0 ? "T = 0 is the limit of dividing by an ever smaller number: the top logit wins everything (greedy)." : `Each logit is divided by T. T < 1 stretches the gaps, T > 1 shrinks them.`,
      [cfg.T === 0 ? ev("g", "T = 0 → greedy: argmax only", "warning") : ev("s", `z/T: ${pl.scaled!.map((s) => s.toFixed(2)).join(", ")}`, undefined, `\\tfrac{z_{\\text{sunny}}}{T} = \\tfrac{${LOGITS[0]}}{${cfg.T}} = ${pl.scaled![0].toFixed(3)}`)]),
    mk("exp", "Exponentiate", "e^(z/T) makes every score positive. Subtracting the largest score first keeps the numbers small without changing the ratios.", [ev("e", `exp: ${pl.expd.map(f3).join(", ")}`)]),
    mk("normalize", "Normalize: softmax probabilities", "Dividing by the sum makes the scores add up to 1: a probability distribution over the next token.",
      [ev("p", `p: ${pl.probs.map(f3).join(", ")}`, undefined, `p_{\\text{sunny}} = ${f3(pl.probs[0])}`), ev("h", `entropy ${entropyBits(pl.probs).toFixed(2)} bits`)]),
    mk("topk", kOff ? "Top-k: off" : `Top-k: keep the ${cfg.topK} most likely`, kOff ? "Top-k is off, so every token stays a candidate." : `Only the ${cfg.topK} highest-probability tokens survive; the other ${removedK} are cut.`,
      [ev("k", kOff ? "top-k off" : `top-k=${cfg.topK}: removed ${removedK} tokens`)]),
    mk("topp", pOff ? "Top-p: off" : `Top-p (nucleus): p = ${cfg.topP}`, pOff ? "Top-p is off (p = 1), so nothing more is cut." : `Sort by probability and keep the smallest set whose cumulative probability reaches ${cfg.topP}.`,
      [ev("tp", pOff ? "top-p off" : `top-p=${cfg.topP}: removed ${removedP} more tokens`)]),
    mk("renorm", "Renormalize what is left", `The surviving tokens hold ${(pl.kept * 100).toFixed(1)}% of the probability; dividing by that makes them sum to 1 again.`,
      [ev("r", `kept mass ${f3(pl.kept)} → final: ${pl.final.map(f3).join(", ")}`)]),
    mk("sample", `Sample: "${VOCAB[pl.pick]}"`, `A seeded random number u = ${pl.u.toFixed(4)} lands on the cumulative-probability strip. The token whose interval contains u is chosen.`,
      [ev("u", `u = ${pl.u.toFixed(4)} → "${VOCAB[pl.pick]}"${pl.pick === pl.greedy ? " (same as greedy)" : ` (greedy would pick "${VOCAB[pl.greedy]}")`}`)]),
    mk("repeat", `Repeat ${cfg.n} times`, `Draw ${cfg.n} samples with the same seed stream. The histogram converges to the final distribution.`,
      [ev("tv", `total-variation distance after ${cfg.n} draws: ${last.tv.toFixed(3)}`), ...(pl.final.filter((p) => p > 0).length === 1 ? [ev("det", "Only one token has probability > 0: sampling is deterministic (same as greedy).", "warning")] : [])]),
  ];
}
