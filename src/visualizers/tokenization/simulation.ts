import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** Built-in training corpus: small, repetitive on purpose so merges are visible. */
export const CORPUS =
  "the cat sat on the mat. the rat sat on the hat. the fat cat is the fattest cat. the rat is faster than the cat. " +
  "the cat is sitting and the rat is eating. a hat, a mat and a cat. newer hats, newest mats, slower rats, slowest cats. " +
  "the best cat is resting. the rat is resting and eating.";

/** Base vocabulary: the 95 printable ASCII characters (ids 0–94), then <unk> (95). Merged tokens start at 96. */
export const BASE = Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i));
export const UNK = "<unk>";
export const UNK_ID = 95;
export const FIRST_MERGE_ID = 96;
export const MAX_TEXT = 80;

/**
 * GPT-2-style pre-tokenization: a word keeps its leading space (" cat"), digits and punctuation form their own
 * chunks, and in a run of spaces only the last one attaches to the next word.
 */
export const pretokenize = (text: string) => text.match(/ ?[A-Za-z]+| ?[0-9]+| ?[^ A-Za-z0-9]+| (?= )| +$/g) ?? [];

const toSymbols = (chunk: string) => [...chunk].map((c) => (BASE.includes(c) ? c : UNK));

export type Pair = [string, string];
export type PairCount = { pair: Pair; count: number };
export type CorpusWord = { text: string; count: number };
export type Merge = { rank: number; id: number; pair: Pair; token: string; count: number; top: PairCount[]; contributions: { word: string; freq: number; occ: number }[] };

export function corpusWords(corpus = CORPUS): CorpusWord[] {
  const counts = new Map<string, number>();
  for (const w of pretokenize(corpus)) counts.set(w, (counts.get(w) ?? 0) + 1);
  return [...counts].map(([text, count]) => ({ text, count }));
}

/** Count every adjacent symbol pair, weighted by word frequency. Ties: higher count first, then alphabetical. */
export function countPairs(words: CorpusWord[], seg: string[][]): PairCount[] {
  const m = new Map<string, PairCount>();
  seg.forEach((s, w) => {
    for (let i = 0; i < s.length - 1; i++) {
      const key = `${s[i]}\u0000${s[i + 1]}`;
      const e = m.get(key) ?? { pair: [s[i], s[i + 1]] as Pair, count: 0 };
      e.count += words[w].count;
      m.set(key, e);
    }
  });
  return [...m.values()].sort((a, b) => b.count - a.count || (a.pair.join("\u0000") < b.pair.join("\u0000") ? -1 : 1));
}

/** Merge every occurrence of `pair` (left to right, non-overlapping). `fresh[i]` marks tokens created by this merge. */
export function applyMerge(symbols: string[], pair: Pair): { out: string[]; fresh: boolean[] } {
  const out: string[] = [];
  const fresh: boolean[] = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === pair[0] && symbols[i + 1] === pair[1]) {
      out.push(pair[0] + pair[1]);
      fresh.push(true);
      i++;
    } else {
      out.push(symbols[i]);
      fresh.push(false);
    }
  }
  return { out, fresh };
}

export type MergeSnapshot = { before: string[][]; after: string[][]; fresh: boolean[][] };
export type Trained = { words: CorpusWord[]; merges: Merge[]; snapshots: MergeSnapshot[]; initial: string[][]; exhaustedAt: number | null };

/** Train byte-pair encoding for up to `numMerges` merges. Stops early when no pair occurs at least twice. */
export function trainBPE(numMerges: number, corpus = CORPUS): Trained {
  const words = corpusWords(corpus);
  const initial = words.map((w) => toSymbols(w.text));
  let seg = initial;
  const merges: Merge[] = [];
  const snapshots: MergeSnapshot[] = [];
  let exhaustedAt: number | null = null;
  const n = Math.max(0, Math.min(200, Math.floor(numMerges)));
  for (let r = 0; r < n; r++) {
    const pairs = countPairs(words, seg);
    if (!pairs.length || pairs[0].count < 2) {
      exhaustedAt = r;
      break;
    }
    const best = pairs[0];
    const contributions = words
      .map((w, i) => ({ word: w.text, freq: w.count, occ: seg[i].reduce((c, s, j) => c + (s === best.pair[0] && seg[i][j + 1] === best.pair[1] ? 1 : 0), 0) }))
      .filter((c) => c.occ > 0)
      .sort((a, b) => b.freq * b.occ - a.freq * a.occ);
    const results = seg.map((s) => applyMerge(s, best.pair));
    const after = results.map((x) => x.out);
    merges.push({ rank: r, id: FIRST_MERGE_ID + r, pair: best.pair, token: best.pair.join(""), count: best.count, top: pairs.slice(0, 6), contributions });
    snapshots.push({ before: seg, after, fresh: results.map((x) => x.fresh) });
    seg = after;
  }
  return { words, merges, snapshots, initial, exhaustedAt };
}

export function vocabFor(merges: Merge[]) {
  const id = new Map<string, number>(BASE.map((c, i) => [c, i]));
  id.set(UNK, UNK_ID);
  for (const m of merges) if (!id.has(m.token)) id.set(m.token, m.id);
  return id;
}

export type Token = { text: string; id: number; chunk: number; start: number };
export type EncodeFrame = { merge: Merge | null; tokens: Token[] };
export type Encoded = { chunks: string[]; frames: EncodeFrame[]; tokens: Token[]; unseen: number[] };

/** Encode text: pre-tokenize, split to characters, then apply learned merges in the order they were learned. */
export function encode(text: string, merges: Merge[], corpusChunks: Set<string> = new Set(corpusWords().map((w) => w.text))): Encoded {
  const chunks = pretokenize(text.slice(0, MAX_TEXT));
  const vocab = vocabFor(merges);
  let seg = chunks.map(toSymbols);
  const snap = (merge: Merge | null): EncodeFrame => ({
    merge,
    tokens: seg.flatMap((s, c) => {
      let start = 0;
      return s.map((t) => {
        const tok = { text: t, id: vocab.get(t) ?? UNK_ID, chunk: c, start };
        start += t === UNK ? 1 : t.length;
        return tok;
      });
    }),
  });
  const frames: EncodeFrame[] = [snap(null)];
  for (const m of merges) {
    const next = seg.map((s) => applyMerge(s, m.pair).out);
    if (next.some((s, i) => s.length !== seg[i].length)) {
      seg = next;
      frames.push(snap(m));
    }
  }
  const unseen = chunks.map((c, i) => (corpusChunks.has(c) ? -1 : i)).filter((i) => i >= 0);
  return { chunks, frames, tokens: frames[frames.length - 1].tokens, unseen };
}

export const countWords = (text: string) => text.trim().split(/\s+/).filter(Boolean).length;

export type TokPhase = "corpus" | "merge" | "split" | "encode" | "compare";
export type TokState = { phase: TokPhase; mergeIndex: number; trained: Trained; encoded: Encoded; text: string };

export function simulate(numMerges: number, text: string): SimulationStep<TokState>[] {
  const trained = trainBPE(numMerges);
  const clean = text.slice(0, MAX_TEXT);
  const encoded = encode(clean, trained.merges);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "tokenize", message, severity });
  const show = (s: string) => s.replace(/ /g, "␣");
  const base = { trained, encoded, text: clean };
  const totalWords = trained.words.reduce((s, w) => s + w.count, 0);
  const steps: SimulationStep<TokState>[] = [
    {
      id: "corpus",
      title: "Split the corpus into characters",
      description: `The corpus has ${totalWords} words (${trained.words.length} distinct). Each starts as single characters; the vocabulary starts as 95 printable characters + <unk>.`,
      state: { ...base, phase: "corpus", mergeIndex: -1 },
      events: [ev("c", `${trained.words.length} distinct pre-tokenized words, ${trained.initial.reduce((s, w, i) => s + w.length * trained.words[i].count, 0)} character tokens in total`)],
    },
  ];
  trained.merges.forEach((m, i) =>
    steps.push({
      id: `merge-${i}`,
      title: `Merge ${i + 1}: "${show(m.pair[0])}" + "${show(m.pair[1])}"`,
      description: `The most frequent adjacent pair "${show(m.pair[0])}" + "${show(m.pair[1])}" appears ${m.count} times. It becomes new token #${m.id} "${show(m.token)}".`,
      state: { ...base, phase: "merge", mergeIndex: i },
      events: [ev(`m${i}`, `merge #${i + 1}: (${show(m.pair[0])}, ${show(m.pair[1])}) → ${show(m.token)} · count ${m.count} · id ${m.id}`)],
    }),
  );
  if (trained.exhaustedAt !== null) steps[steps.length - 1].events.push(ev("ex", `Stopped after ${trained.exhaustedAt} merges: no pair occurs twice any more.`, "warning"));
  const fin = encoded.tokens;
  const unk = fin.filter((t) => t.id === UNK_ID).length;
  steps.push(
    {
      id: "split",
      title: "Tokenize your text: split",
      description: `Your text is cut into ${encoded.chunks.length} pre-tokens (a word keeps its leading space), then into ${encoded.frames[0].tokens.length} characters.`,
      state: { ...base, phase: "split", mergeIndex: trained.merges.length - 1 },
      events: [ev("s", `pre-tokens: ${encoded.chunks.map((c) => `"${show(c)}"`).join(" ")}`), ...(unk ? [ev("unk", `${unk} character(s) are not in the base vocabulary and become <unk>.`, "warning")] : [])],
    },
    {
      id: "encode",
      title: "Tokenize your text: apply merges",
      description: `The learned merges are replayed in training order. ${encoded.frames.length - 1} of them apply to this text, leaving ${fin.length} tokens.`,
      state: { ...base, phase: "encode", mergeIndex: trained.merges.length - 1 },
      events: encoded.frames.slice(1).map((f, i) => ev(`e${i}`, `apply merge #${f.merge!.rank + 1} (${show(f.merge!.token)}) → ${f.tokens.length} tokens`)),
    },
    {
      id: "compare",
      title: "Compare token counts",
      description: `${clean.length} characters, ${countWords(clean)} words, ${fin.length} BPE tokens. Unseen words split into known sub-word pieces.`,
      state: { ...base, phase: "compare", mergeIndex: trained.merges.length - 1 },
      events: [
        ev("n", `chars ${clean.length} · words ${countWords(clean)} · BPE tokens ${fin.length}`),
        ...encoded.unseen.map((c) => ev(`u${c}`, `unseen "${show(encoded.chunks[c])}" → ${fin.filter((t) => t.chunk === c).map((t) => show(t.text)).join(" | ")}`)),
      ],
    },
  );
  return steps;
}

export const TEXT_PRESETS = [
  { label: "Unseen word", text: "the fastest rats were sitting" },
  { label: "Case", text: "The CAT sat on the Mat" },
  { label: "Spaces", text: "the  cat   sat" },
  { label: "Numbers", text: "the cat ate 2024 rats in 365 days" },
  { label: "Not ASCII", text: "the café cat" },
];
