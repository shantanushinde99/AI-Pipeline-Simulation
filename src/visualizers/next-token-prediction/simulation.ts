import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** Built-in training corpus (word-level). Small on purpose: you can read every count. */
export const CORPUS =
  "the cat sat on the mat . the cat ate the fish . the dog sat on the rug . the dog chased the cat . " +
  "a cat sat on a hat . the bird sang a song . the cat sat on the hat .";

/** Lower-case words and sentence punctuation; everything else is dropped. */
export const tokenize = (text: string) => text.toLowerCase().match(/[a-z]+|[.,!?]/g) ?? [];

export type Order = 2 | 3;
export type Smoothing = "backoff" | "laplace" | "none";
export const SMOOTHING_LABEL: Record<Smoothing, string> = { backoff: "Back off to a shorter context", laplace: "Add-one (Laplace) smoothing", none: "None (break it)" };

export type Model = { tokens: string[]; vocab: string[]; counts: Map<string, Map<string, number>> };
const key = (ctx: string[]) => ctx.join(" ");

/** Count every n-gram for n = 1..3. counts.get(context)?.get(next). Unigrams use the empty context "". */
export function train(corpus = CORPUS): Model {
  const tokens = tokenize(corpus);
  const vocab = [...new Set(tokens)];
  const counts = new Map<string, Map<string, number>>();
  const add = (ctx: string, next: string) => {
    const m = counts.get(ctx) ?? new Map<string, number>();
    m.set(next, (m.get(next) ?? 0) + 1);
    counts.set(ctx, m);
  };
  tokens.forEach((t, i) => {
    add("", t);
    if (i >= 1) add(key(tokens.slice(i - 1, i)), t);
    if (i >= 2) add(key(tokens.slice(i - 2, i)), t);
  });
  return { tokens, vocab, counts };
}

export type Candidate = { token: string; count: number; p: number };
export type Prediction = {
  context: string[]; // the context actually used
  wanted: string[]; // the context the chosen order asked for
  usedOrder: number; // 1 = unigram
  total: number;
  candidates: Candidate[]; // sorted: probability desc, then first appearance in corpus
  smoothed: boolean;
  unseen: boolean; // the wanted context never occurred
};

/** Next-token distribution for the last (order − 1) tokens of `seq`. Returns null candidates list when there is no data (smoothing "none"). */
export function predict(model: Model, seq: string[], order: Order, smoothing: Smoothing): Prediction {
  const wanted = seq.slice(Math.max(0, seq.length - (order - 1)));
  const rank = (t: string) => model.vocab.indexOf(t);
  const sort = (c: Candidate[]) => c.sort((a, b) => b.p - a.p || rank(a.token) - rank(b.token));
  const fromCounts = (ctx: string[], usedOrder: number, unseen: boolean): Prediction => {
    const m = model.counts.get(key(ctx)) ?? new Map<string, number>();
    const total = [...m.values()].reduce((s, v) => s + v, 0);
    if (smoothing === "laplace") {
      const V = model.vocab.length;
      return { context: ctx, wanted, usedOrder, total, unseen, smoothed: true, candidates: sort(model.vocab.map((token) => ({ token, count: m.get(token) ?? 0, p: ((m.get(token) ?? 0) + 1) / (total + V) }))) };
    }
    return { context: ctx, wanted, usedOrder, total, unseen, smoothed: false, candidates: total ? sort([...m].map(([token, count]) => ({ token, count, p: count / total }))) : [] };
  };
  const seen = (ctx: string[]) => ctx.length === wanted.length && model.counts.has(key(ctx));
  if (seen(wanted) || smoothing !== "backoff") return fromCounts(wanted, order, !seen(wanted));
  // Back off: drop the oldest context word until the context has been seen (unigram always has).
  for (let n = wanted.length - 1; n >= 0; n--) {
    const ctx = wanted.slice(wanted.length - n);
    if (model.counts.has(key(ctx))) return fromCounts(ctx, n + 1, true);
  }
  return fromCounts([], 1, true);
}

/** Every corpus position where `ctx` occurs and is followed by a token (used to highlight where the counts come from). */
export function occurrences(model: Model, ctx: string[]): number[] {
  const out: number[] = [];
  const n = ctx.length;
  for (let i = 0; i + n < model.tokens.length; i++) if (ctx.every((t, j) => model.tokens[i + j] === t)) out.push(i);
  return out;
}

export type GenState = {
  phase: "count" | "predict" | "stuck" | "done";
  prompt: string[];
  seq: string[]; // sequence before this step's pick
  i: number; // generation index
  pred: Prediction | null;
  pick: string | null;
  loopAt: number | null; // index in the final sequence where a repeating cycle starts
};

export type GenConfig = { prompt: string; order: Order; smoothing: Smoothing; n: number };

export function generate(model: Model, cfg: GenConfig) {
  const prompt = tokenize(cfg.prompt).slice(-12);
  const n = Math.max(1, Math.min(16, Math.floor(cfg.n)));
  const seq = [...prompt];
  const out: { seq: string[]; pred: Prediction; pick: string | null }[] = [];
  const seenStates = new Map<string, number>();
  let loopAt: number | null = null;
  if (prompt.length >= cfg.order - 1) seenStates.set(key(prompt.slice(-(cfg.order - 1))), prompt.length);
  for (let i = 0; i < n; i++) {
    const pred = predict(model, seq, cfg.order, cfg.smoothing);
    const pick = pred.candidates[0]?.token ?? null; // greedy
    out.push({ seq: [...seq], pred, pick });
    if (!pick) break;
    seq.push(pick);
    // Greedy decoding is deterministic: once the same context state recurs, the text cycles forever.
    const state = key(seq.slice(-(cfg.order - 1)));
    if (loopAt === null && seenStates.has(state)) loopAt = seenStates.get(state)!;
    if (!seenStates.has(state)) seenStates.set(state, seq.length);
  }
  return { prompt, steps: out, final: seq, loopAt, stuck: out.length > 0 && out[out.length - 1].pick === null };
}

const tx = (s: string) => `\\text{${s.replace(/[^a-z.,!? ]/g, "")}}`;

export function simulate(cfg: GenConfig, model: Model = train()): SimulationStep<GenState>[] {
  const g = generate(model, cfg);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "predict", message, severity, tex });
  const steps: SimulationStep<GenState>[] = [
    {
      id: "count",
      title: `Count ${cfg.order === 2 ? "bigrams" : "trigrams"} in the corpus`,
      description: `The corpus has ${model.tokens.length} tokens and ${model.vocab.length} distinct words. Training an n-gram model is just counting which word follows which ${cfg.order === 2 ? "word" : "pair of words"}.`,
      state: { phase: "count", prompt: g.prompt, seq: g.prompt, i: -1, pred: null, pick: null, loopAt: null },
      events: [ev("t", `${model.tokens.length} tokens, vocabulary ${model.vocab.length}, ${[...model.counts.keys()].filter((k) => k.split(" ").length === cfg.order - 1 && k).length} distinct contexts of length ${cfg.order - 1}`)],
    },
  ];
  g.steps.forEach((s, i) => {
    const { pred, pick } = s;
    const ctx = pred.context.join(" ") || "∅";
    const events: SimulationEvent[] = [];
    if (pred.unseen) events.push(ev(`u${i}`, `Context "${pred.wanted.join(" ")}" never appears in the corpus.`, "warning"));
    if (pred.unseen && cfg.smoothing === "backoff") events.push(ev(`b${i}`, `Backed off to ${pred.usedOrder === 1 ? "unigram (no context)" : `"${ctx}"`}.`, "warning"));
    if (pick) {
      const top = pred.candidates[0];
      events.push(ev(`p${i}`, `P(${pick} | ${ctx}) = ${pred.smoothed ? `(${top.count}+1)/(${pred.total}+${model.vocab.length})` : `${top.count}/${pred.total}`} = ${top.p.toFixed(3)} → pick "${pick}"`, undefined,
        `P(${tx(pick)} \\mid ${pred.context.length ? tx(ctx) : "\\varnothing"}) = ${pred.smoothed ? `\\frac{${top.count}+1}{${pred.total}+${model.vocab.length}}` : `\\frac{${top.count}}{${pred.total}}`} = ${top.p.toFixed(3)}`));
    } else events.push(ev(`x${i}`, "No counts for this context and no smoothing: the model has no prediction. Generation stops.", "error"));
    steps.push({
      id: `gen-${i}`,
      title: pick ? `Token ${i + 1}: "${pick}"` : `Token ${i + 1}: no prediction`,
      description: pick
        ? `Look up what followed "${ctx}" in the corpus, turn counts into probabilities, pick the most likely word (greedy).`
        : `"${pred.wanted.join(" ")}" was never seen, and with no smoothing or backoff every count is 0. The model cannot continue.`,
      state: { phase: pick ? "predict" : "stuck", prompt: g.prompt, seq: s.seq, i, pred, pick, loopAt: null },
      events,
    });
  });
  steps.push({
    id: "done",
    title: g.loopAt !== null ? "Done: stuck in a loop" : "Done",
    description: g.loopAt !== null
      ? `Greedy decoding always picks the same word for the same context, so once a context repeats the text cycles forever (from word ${g.loopAt + 1}).`
      : `Generated ${g.final.length - g.prompt.length} tokens.`,
    state: { phase: "done", prompt: g.prompt, seq: g.final, i: g.steps.length, pred: null, pick: null, loopAt: g.loopAt },
    events: [ev("d", `"${g.final.join(" ")}"`, g.loopAt !== null ? "warning" : undefined), ...(g.loopAt !== null ? [ev("loop", "Repetition loop: the same context came back, and greedy decoding has no randomness to escape it.", "warning")] : [])],
  });
  return steps;
}

export const PROMPT_PRESETS = [
  { label: "the cat", prompt: "the cat" },
  { label: "the dog chased", prompt: "the dog chased" },
  { label: "a bird", prompt: "a bird" },
  { label: "unseen: the zebra", prompt: "the zebra" },
];
