import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/**
 * A tiny sentiment classifier: bag-of-words x (12 words) → tanh hidden layer (8) → sigmoid "P(positive)".
 * It is pretrained on movie reviews, then fine-tuned on a handful of phone reviews, where two words flip meaning
 * ("an unpredictable plot" is good, "an unpredictable battery" is not). Everything, pretraining included,
 * is computed here with real gradients and a seeded RNG.
 */
export const VOCAB = ["great", "awful", "love", "boring", "long", "unpredictable", "plot", "acting", "ending", "battery", "screen", "camera"] as const;
export type Word = (typeof VOCAB)[number];
export const D = VOCAB.length;
export const H = 8;
export type Domain = "movie" | "phone";

/** Word sentiment per domain. "long" and "unpredictable" flip between movies and phones; the rest agree. */
const SCORE: Record<Domain, Partial<Record<Word, number>>> = {
  movie: { great: 1, awful: -1, love: 1, boring: -1, long: -1, unpredictable: 1 },
  phone: { great: 1, awful: -1, love: 1, boring: -1, long: 1, unpredictable: -1 },
};
const SENTIMENT: Word[] = ["great", "awful", "love", "boring", "long", "unpredictable"];
export const FLIP_WORDS: Word[] = ["long", "unpredictable"];
export const TOPICS: Record<Domain, Word[]> = { movie: ["plot", "acting", "ending"], phone: ["battery", "screen", "camera"] };

export type Example = { words: Word[]; y: 0 | 1; x: number[]; noisy?: boolean };
const vec = (words: Word[]) => VOCAB.map((w) => (words.includes(w) ? 1 : 0));
const ex = (words: Word[], y: 0 | 1, noisy?: boolean): Example => ({ words, y, x: vec(words), ...(noisy ? { noisy } : {}) });

/** Every review "<topic word> + 1 or 2 sentiment words" whose label is unambiguous (score ≠ 0). */
function allReviews(domain: Domain): Example[] {
  const sets: Word[][] = [...SENTIMENT.map((w) => [w]), ...SENTIMENT.flatMap((a, i) => SENTIMENT.slice(i + 1).map((b) => [a, b]))];
  return TOPICS[domain].flatMap((t) => sets.flatMap((s) => {
    const score = s.reduce((sum, w) => sum + (SCORE[domain][w] ?? 0), 0);
    return score === 0 ? [] : [ex([t, ...s], score > 0 ? 1 : 0)];
  }));
}

function shuffle<T>(xs: T[], seed: number): T[] {
  const r = rng(seed);
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

const movies = shuffle(allReviews("movie"), 11);
/** Original task: movie reviews. 70% were used to pretrain; the held-out 30% measure forgetting. */
export const ORIGINAL_TRAIN = movies.slice(0, Math.round(movies.length * 0.7));
export const ORIGINAL_EVAL = movies.slice(ORIGINAL_TRAIN.length);

/** New task: only 8 labelled phone reviews, one of them mislabeled (as tiny real datasets often are). */
export const TARGET_TRAIN: Example[] = [
  ex(["battery", "long"], 1),
  ex(["screen", "unpredictable"], 0),
  ex(["camera", "great", "long"], 1),
  ex(["screen", "love"], 1),
  ex(["battery", "awful"], 0),
  ex(["camera", "boring", "unpredictable"], 0),
  ex(["screen", "love", "long"], 1),
  ex(["battery", "great"], 0, true),
];
const key = (e: Example) => [...e.words].sort().join(" ");
const trainKeys = new Set(TARGET_TRAIN.map(key));
/** Every other (correctly labelled) phone review: the validation set. */
export const TARGET_VAL = allReviews("phone").filter((e) => !trainKeys.has(key(e)));

/** Probe sentences tracked on screen: two unseen phone reviews and two movie reviews, all hinging on a flip word. */
export const PROBES: { label: string; domain: Domain; ex: Example }[] = [
  { label: "camera · long", domain: "phone", ex: ex(["camera", "long"], 1) },
  { label: "battery · unpredictable", domain: "phone", ex: ex(["battery", "unpredictable"], 0) },
  { label: "plot · long", domain: "movie", ex: ex(["plot", "long"], 0) },
  { label: "ending · unpredictable", domain: "movie", ex: ex(["ending", "unpredictable"], 1) },
];

// ---------------------------------------------------------------- model

export type Params = { W: number[][]; b1: number[]; w2: number[]; b2: number };
export type Lora = { A: number[][]; B: number[][] }; // A: r × D, B: H × r, ΔW = B·A
export type Method = "full" | "lora" | "head";
export const METHOD_LABEL: Record<Method, string> = { full: "Full fine-tuning", lora: "LoRA (low-rank adapter)", head: "Frozen base + train head" };

const zeros = (r: number, c: number) => Array.from({ length: r }, () => new Array<number>(c).fill(0));
export const matmul = (A: number[][], B: number[][]) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const addM = (A: number[][], B: number[][]) => A.map((row, i) => row.map((v, j) => v + B[i][j]));

/** The effective first-layer weights: W, plus B·A when a LoRA adapter is attached. */
export const effectiveW = (p: Params, lora?: Lora) => (lora ? addM(p.W, matmul(lora.B, lora.A)) : p.W);

export function forward(p: Params, x: number[], lora?: Lora) {
  const W = effectiveW(p, lora);
  const h = W.map((row, j) => Math.tanh(row.reduce((s, v, i) => s + v * x[i], p.b1[j])));
  const z = p.w2.reduce((s, v, j) => s + v * h[j], p.b2);
  return { h, z, p: 1 / (1 + Math.exp(-z)) };
}

/** Binary cross-entropy from the logit, computed stably: log(1 + e^z) − y·z. */
export const bce = (z: number, y: number) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - y * z;

export function evaluate(p: Params, data: Example[], lora?: Lora) {
  let loss = 0, correct = 0;
  for (const e of data) {
    const f = forward(p, e.x, lora);
    loss += bce(f.z, e.y);
    if ((f.p >= 0.5 ? 1 : 0) === e.y) correct++;
  }
  return { loss: loss / data.length, acc: correct / data.length };
}

type Grads = { W: number[][]; b1: number[]; w2: number[]; b2: number };
/** Mean gradient of the BCE loss over a batch, w.r.t. W (effective), b1, w2, b2. Backprop by hand. */
export function gradients(p: Params, batch: Example[], lora?: Lora): Grads {
  const g: Grads = { W: zeros(H, D), b1: new Array(H).fill(0), w2: new Array(H).fill(0), b2: 0 };
  const n = batch.length;
  for (const e of batch) {
    const f = forward(p, e.x, lora);
    const dz = (f.p - e.y) / n;
    g.b2 += dz;
    for (let j = 0; j < H; j++) {
      g.w2[j] += dz * f.h[j];
      const dz1 = dz * p.w2[j] * (1 - f.h[j] ** 2);
      g.b1[j] += dz1;
      for (let i = 0; i < D; i++) g.W[j][i] += dz1 * e.x[i];
    }
  }
  return g;
}

/** Pretraining on movie reviews (full batch, fixed seed). Memoized: it is the same "base model" for every run. */
let base: Params | null = null;
export function pretrained(): Params {
  if (base) return base;
  const r = rng(7);
  const p: Params = { W: zeros(H, D).map((row) => row.map(() => 0.3 * r.gaussian())), b1: new Array(H).fill(0), w2: Array.from({ length: H }, () => 0.3 * r.gaussian()), b2: 0 };
  for (let t = 0; t < 600; t++) {
    const g = gradients(p, ORIGINAL_TRAIN);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < D; i++) p.W[j][i] -= 0.5 * g.W[j][i];
      p.b1[j] -= 0.5 * g.b1[j];
      p.w2[j] -= 0.5 * g.w2[j];
    }
    p.b2 -= 0.5 * g.b2;
  }
  base = p;
  return p;
}

/** Trainable-parameter counts. LoRA trains r·(H + D) numbers; full trains every weight and bias. */
export const paramCount = (method: Method, rank: number) => ({
  total: H * D + H + H + 1,
  trainable: method === "full" ? H * D + H + H + 1 : method === "head" ? H + 1 : rank * (H + D),
});

// ---------------------------------------------------------------- fine-tuning run

export type FTConfig = { method: Method; lr: number; epochs: number; rank: number; replay: number };
export const DEFAULT_FT: FTConfig = { method: "full", lr: 0.3, epochs: 40, rank: 1, replay: 0 };
export const LR_PRESETS = { low: 0.05, good: 0.3, high: 10 } as const;
export const BATCH = 4;

export type Metrics = { trainLoss: number; trainAcc: number; valLoss: number; valAcc: number; origLoss: number; origAcc: number };
export type Snapshot = { epoch: number; params: Params; lora?: Lora; m: Metrics; probes: number[]; drift: number; updates: number };

const snapshot = (epoch: number, p: Params, lora: Lora | undefined, b: Params, updates: number): Snapshot => {
  const t = evaluate(p, TARGET_TRAIN, lora), v = evaluate(p, TARGET_VAL, lora), o = evaluate(p, ORIGINAL_EVAL, lora);
  const W = effectiveW(p, lora);
  // Drift: how far the effective weights have moved from the pretrained ones (Frobenius norm over all parameters).
  let d = (p.b2 - b.b2) ** 2;
  for (let j = 0; j < H; j++) {
    d += (p.b1[j] - b.b1[j]) ** 2 + (p.w2[j] - b.w2[j]) ** 2;
    for (let i = 0; i < D; i++) d += (W[j][i] - b.W[j][i]) ** 2;
  }
  return {
    epoch, updates, params: structuredClone(p), lora: lora && structuredClone(lora), drift: Math.sqrt(d),
    m: { trainLoss: t.loss, trainAcc: t.acc, valLoss: v.loss, valAcc: v.acc, origLoss: o.loss, origAcc: o.acc },
    probes: PROBES.map((q) => forward(p, q.ex.x, lora).p),
  };
};

export type Regime = "diverged" | "overfit" | "forgot" | "underfit" | "good";

export function run(raw: FTConfig) {
  const cfg: FTConfig = { method: raw.method, lr: clamp(raw.lr, 0.001, 20), epochs: clamp(Math.round(raw.epochs), 1, 120), rank: clamp(Math.round(raw.rank), 1, 4), replay: clamp(raw.replay, 0, 1) };
  const b = pretrained();
  const p = structuredClone(b);
  const r = rng(42);
  const lora: Lora | undefined = cfg.method === "lora"
    ? { A: Array.from({ length: cfg.rank }, () => Array.from({ length: D }, () => r.gaussian() / Math.sqrt(D))), B: zeros(H, cfg.rank) }
    : undefined;
  // Replay: mix this share of the original (movie) training set back in, so the model keeps practising it.
  const nReplay = Math.round(ORIGINAL_TRAIN.length * cfg.replay);
  const replayed = ORIGINAL_TRAIN.slice(0, nReplay);
  const train = [...TARGET_TRAIN, ...replayed];
  const history: Snapshot[] = [snapshot(0, p, lora, b, 0)];
  let updates = 0;
  for (let e = 1; e <= cfg.epochs; e++) {
    const order = shuffle(train, 1000 + e);
    for (let s = 0; s < order.length; s += BATCH) {
      const g = gradients(p, order.slice(s, s + BATCH), lora);
      const η = cfg.lr;
      if (cfg.method === "full") {
        for (let j = 0; j < H; j++) for (let i = 0; i < D; i++) p.W[j][i] -= η * g.W[j][i];
        for (let j = 0; j < H; j++) p.b1[j] -= η * g.b1[j];
      }
      if (cfg.method !== "lora") {
        for (let j = 0; j < H; j++) p.w2[j] -= η * g.w2[j];
        p.b2 -= η * g.b2;
      }
      if (lora) {
        // ∂L/∂B = G·Aᵀ and ∂L/∂A = Bᵀ·G, where G = ∂L/∂W_eff. W itself never changes.
        const gB = matmul(g.W, lora.A[0].map((_, i) => lora.A.map((row) => row[i])));
        const gA = matmul(lora.B[0].map((_, k) => lora.B.map((row) => row[k])), g.W);
        lora.B = lora.B.map((row, j) => row.map((v, k) => v - η * gB[j][k]));
        lora.A = lora.A.map((row, k) => row.map((v, i) => v - η * gA[k][i]));
      }
      updates++;
    }
    history.push(snapshot(e, p, lora, b, updates));
  }
  const first = history[0].m, last = history.at(-1)!.m;
  const bestVal = Math.min(...history.map((h) => h.m.valLoss));
  const bestEpoch = history.findIndex((h) => h.m.valLoss === bestVal);
  const diverged = !Number.isFinite(last.trainLoss) || last.trainLoss > first.trainLoss || history.some((h, i) => i > 0 && h.m.trainLoss > 3 * first.trainLoss);
  const forgetting = last.origLoss - first.origLoss;
  const regime: Regime = diverged ? "diverged"
    : last.valLoss > 1.35 * bestVal && last.valLoss > bestVal + 0.04 && bestEpoch <= cfg.epochs * 0.7 ? "overfit"
    : last.valAcc < 0.8 ? "underfit"
    : last.origAcc < first.origAcc - 0.15 ? "forgot"
    : "good";
  // LoRA never edits W, so detaching the adapter gives back the pretrained model exactly.
  const detached = lora ? evaluate(p, ORIGINAL_EVAL) : null;
  return { cfg, history, train, nReplay, params: paramCount(cfg.method, cfg.rank), bestEpoch, bestVal, forgetting, regime, detached };
}
export type Run = ReturnType<typeof run>;

// ---------------------------------------------------------------- playback steps

export type FTState = { kind: "base" | "data" | "epoch" | "result"; epoch: number; prevEpoch: number; headline: string; sub: string };

/** Which epochs get their own playback step: up to 8, denser at the start where most of the change happens. */
export function checkpoints(epochs: number): number[] {
  if (epochs <= 8) return Array.from({ length: epochs }, (_, i) => i + 1);
  return [...new Set(Array.from({ length: 8 }, (_, i) => Math.max(1, Math.round(epochs * ((i + 1) / 8) ** 1.6))))];
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const f3 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) >= 1e3 ? v.toExponential(2) : v.toFixed(3)) : "∞");

export function simulate(cfg: FTConfig) {
  const sim = run(cfg);
  const { history: hs, cfg: c } = sim;
  const h0 = hs[0];
  const steps: SimulationStep<FTState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "train", message, severity, tex });
  const wrongPhone = TARGET_VAL.filter((e) => (forward(h0.params, e.x).p >= 0.5 ? 1 : 0) !== e.y).length;

  steps.push({
    id: "base", title: "Base model on the new task",
    description: `The pretrained model scores ${pct(h0.m.origAcc)} on held-out movie reviews but only ${pct(h0.m.valAcc)} on phone reviews: it gets ${wrongPhone} of ${TARGET_VAL.length} wrong, because "long" and "unpredictable" mean the opposite for phones.`,
    state: { kind: "base", epoch: 0, prevEpoch: 0, headline: `The pretrained model is ${pct(h0.m.origAcc)} right on movies, but only ${pct(h0.m.valAcc)} on phones`, sub: "It learned \"long\" = bad and \"unpredictable\" = good from movie reviews. For a phone, a long battery is good." },
    events: [ev("b-orig", `Original task (movies): loss ${f3(h0.m.origLoss)}, accuracy ${pct(h0.m.origAcc)}`), ev("b-val", `New task (phones): loss ${f3(h0.m.valLoss)}, accuracy ${pct(h0.m.valAcc)}`, h0.m.valAcc < 0.8 ? "warning" : undefined)],
  });
  steps.push({
    id: "data", title: "A small labelled dataset",
    description: `Fine-tuning uses ${TARGET_TRAIN.length} labelled phone reviews${sim.nReplay ? ` plus ${sim.nReplay} replayed movie reviews (${pct(c.replay)} of the original training set)` : ""}, in batches of ${BATCH}. One label is wrong ("battery · great" marked negative): tiny datasets are easy to memorise, mistakes included.`,
    state: { kind: "data", epoch: 0, prevEpoch: 0, headline: `${sim.train.length} labelled examples are all the new task gets`, sub: `${METHOD_LABEL[c.method]}: ${sim.params.trainable} of ${sim.params.total} numbers will be allowed to change.` },
    events: [ev("d-n", `Training set: ${TARGET_TRAIN.length} phone reviews${sim.nReplay ? ` + ${sim.nReplay} movie reviews (replay)` : ""}`), ev("d-p", `Trainable parameters: ${sim.params.trainable} / ${sim.params.total}`)],
  });

  let prev = 0;
  for (const e of checkpoints(c.epochs)) {
    const a = hs[prev], b = hs[e];
    const up = (v: number, w: number) => (w > v + 1e-3 ? "rose" : w < v - 1e-3 ? "fell" : "held");
    const headline = !Number.isFinite(b.m.trainLoss) || b.m.trainLoss > 2 * h0.m.trainLoss
      ? `Epoch ${e}: steps are too big, the loss jumps around instead of falling`
      : `Epoch ${e}: phone-review loss ${up(a.m.valLoss, b.m.valLoss)} to ${b.m.valLoss.toFixed(2)}, movie-review loss ${up(a.m.origLoss, b.m.origLoss)} to ${b.m.origLoss.toFixed(2)}`;
    const sub = c.method === "lora"
      ? "W stays frozen. Only the two thin matrices B and A learn; their product B·A is added to W."
      : c.method === "head"
        ? `The first layer is frozen (grey). Only the ${H + 1} output weights move.`
        : "Every weight that touched a training example gets nudged against its gradient.";
    steps.push({
      id: `epoch-${e}`, title: `Epoch ${e}`,
      description: `Epochs ${prev + 1}–${e}: ${b.updates - a.updates} gradient updates (lr ${c.lr}). Train loss ${f3(a.m.trainLoss)} → ${f3(b.m.trainLoss)}, new-task val loss ${f3(a.m.valLoss)} → ${f3(b.m.valLoss)}, original-task loss ${f3(a.m.origLoss)} → ${f3(b.m.origLoss)}.`,
      state: { kind: "epoch", epoch: e, prevEpoch: prev, headline, sub },
      events: [
        ev(`e${e}-u`, `${b.updates - a.updates} updates of size lr × gradient`, undefined, `\\theta \\leftarrow \\theta - ${c.lr}\\,\\nabla_\\theta \\mathcal{L}_{\\text{batch}}`),
        ev(`e${e}-v`, `New-task val: loss ${f3(b.m.valLoss)}, accuracy ${pct(b.m.valAcc)}`),
        ev(`e${e}-o`, `Original task: loss ${f3(b.m.origLoss)}, accuracy ${pct(b.m.origAcc)}`, b.m.origAcc < h0.m.origAcc - 0.15 ? "warning" : undefined),
        ...(e === sim.bestEpoch ? [ev(`e${e}-best`, `Lowest new-task validation loss so far: a good place to stop.`)] : []),
      ],
    });
    prev = e;
  }

  const last = hs.at(-1)!;
  const result: Record<Regime, [string, string]> = {
    diverged: ["The learning rate was too high: training blew up", `Train loss ended at ${f3(last.m.trainLoss)} (it started at ${f3(h0.m.trainLoss)}). Try lr ${LR_PRESETS.good}.`],
    overfit: [`Too many epochs: validation loss was lowest at epoch ${sim.bestEpoch}, then rose`, "The model memorised the 8 examples (mislabeled one included). Stop early or use fewer epochs."],
    underfit: [`The new task is still not learned: ${pct(last.m.valAcc)} on phone reviews`, c.method === "head" ? "The frozen first layer mixes \"long\" with movie meaning; the head alone cannot undo that." : "Too few epochs, too low a learning rate, or too little capacity (rank)."],
    forgot: [`New task learned (${pct(last.m.valAcc)}), but movie accuracy fell ${pct(h0.m.origAcc)} → ${pct(last.m.origAcc)}`, "Catastrophic forgetting: nothing in the loss asked the model to keep the old task. Try LoRA or replay."],
    good: [`New task ${pct(h0.m.valAcc)} → ${pct(last.m.valAcc)}, movies kept at ${pct(last.m.origAcc)}`, sim.forgetting > 0.05 ? `Some forgetting remains (original loss +${f3(sim.forgetting)}), but accuracy held.` : "The model adapted without losing the original task."],
  };
  steps.push({
    id: "result", title: "Result",
    description: `After ${c.epochs} epochs with ${METHOD_LABEL[c.method].toLowerCase()}: new task ${pct(h0.m.valAcc)} → ${pct(last.m.valAcc)}, original task ${pct(h0.m.origAcc)} → ${pct(last.m.origAcc)}. Weights moved ${f3(last.drift)} from the pretrained ones.`,
    state: { kind: "result", epoch: c.epochs, prevEpoch: 0, headline: result[sim.regime][0], sub: result[sim.regime][1] },
    events: [
      ev("r-v", `New task: ${pct(h0.m.valAcc)} → ${pct(last.m.valAcc)}`),
      ev("r-o", `Original task: ${pct(h0.m.origAcc)} → ${pct(last.m.origAcc)} (loss ${sim.forgetting >= 0 ? "+" : ""}${f3(sim.forgetting)})`, sim.regime === "forgot" ? "warning" : undefined),
      ev("r-p", `Trainable parameters: ${sim.params.trainable} of ${sim.params.total}`),
      ...(sim.detached ? [ev("r-x", `Adapter detached: movie loss back to ${f3(sim.detached.loss)} (exactly the base model, W was never edited)`)] : []),
      ...(sim.regime === "diverged" ? [ev("r-d", "Training diverged: loss ended higher than it started.", "error")] : []),
    ],
  });
  return { ...sim, steps };
}
export type Sim = ReturnType<typeof simulate>;
