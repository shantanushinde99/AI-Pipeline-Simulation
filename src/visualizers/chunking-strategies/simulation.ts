import { dot, mean } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { embed } from "../rag/simulation";

export type Strategy = "fixed" | "overlap" | "sentence" | "heading" | "semantic";
export const STRATEGIES: Record<Strategy, { name: string; rule: string }> = {
  fixed: { name: "Fixed size", rule: "Cut every N words, wherever that lands." },
  overlap: { name: "Fixed size + overlap", rule: "Cut every N − overlap words; each chunk repeats the last words of the previous one." },
  sentence: { name: "Sentence-based", rule: "Pack whole sentences into a chunk until the next one would exceed N words." },
  heading: { name: "Heading / section-aware", rule: "Start a new chunk at every heading; inside a section, pack sentences up to N words." },
  semantic: { name: "Semantic (similarity drop)", rule: "Embed each sentence; cut where similarity to the previous sentence falls below a threshold (or the chunk would exceed N)." },
};

export type ChunkParams = { strategy: Strategy; size: number; overlap: number; threshold: number };
export const DEFAULT_PARAMS: ChunkParams = { strategy: "fixed", size: 30, overlap: 10, threshold: 0.15 };

/** A fictional handbook. Sections are separated by headings; within a section, sentences share key terms. */
const SECTIONS: { heading: string; sentences: string[] }[] = [
  { heading: "Access control", sentences: [
    "Every workspace has three roles: owner, admin and member.",
    "The owner role can delete the workspace and manage billing.",
    "The admin role can invite members and configure single sign-on.",
    "A member can create projects but cannot change workspace security settings.",
  ] },
  { heading: "Passwords and sign-in", sentences: [
    "Passwords must be at least 12 characters long.",
    "After five failed password attempts the account is locked for 15 minutes.",
    "Two-factor authentication is required for every admin account.",
    "Recovery codes for two-factor authentication are shown only once.",
  ] },
  { heading: "Data retention", sentences: [
    "Request logs are retained for 30 days on Starter and Pro plans.",
    "Enterprise plans keep request logs for 365 days.",
    "Deleted projects are retained for 7 days before permanent removal.",
    "Backups of projects are encrypted with AES-256.",
  ] },
  { heading: "Incident response", sentences: [
    "A security incident is triaged within one hour by the on-call engineer.",
    "Customers are notified within 72 hours of a confirmed incident involving their data.",
    "A written post-mortem for every incident is published within 14 days.",
  ] },
];

export type Word = { text: string; sentence: number; section: number; heading: boolean };
export type Sentence = { index: number; section: number; heading: boolean; start: number; end: number; text: string };

export function buildDoc(): { words: Word[]; sentences: Sentence[] } {
  const words: Word[] = [];
  const sentences: Sentence[] = [];
  const add = (text: string, section: number, heading: boolean) => {
    const start = words.length;
    const index = sentences.length;
    for (const w of text.split(/\s+/)) words.push({ text: w, sentence: index, section, heading });
    sentences.push({ index, section, heading, start, end: words.length, text });
  };
  SECTIONS.forEach((s, si) => {
    add(s.heading, si, true);
    s.sentences.forEach((t) => add(t, si, false));
  });
  return { words, sentences };
}

/** Facts a user might ask about. Some need two sentences: the answer only makes sense if both land in one chunk. */
export const FACTS = [
  { id: "f1", label: "Admins invite members and configure SSO", sentences: ["The admin role can invite members and configure single sign-on."] },
  { id: "f2", label: "Lockout: 5 failed attempts → 15 min", sentences: ["After five failed password attempts the account is locked for 15 minutes."] },
  { id: "f3", label: "2FA for admins; recovery codes shown once", sentences: ["Two-factor authentication is required for every admin account.", "Recovery codes for two-factor authentication are shown only once."] },
  { id: "f4", label: "Log retention: 30 days vs 365 on Enterprise", sentences: ["Request logs are retained for 30 days on Starter and Pro plans.", "Enterprise plans keep request logs for 365 days."] },
  { id: "f5", label: "Incident: notify in 72 h, post-mortem in 14 days", sentences: ["Customers are notified within 72 hours of a confirmed incident involving their data.", "A written post-mortem for every incident is published within 14 days."] },
];

export type Fact = { id: string; label: string; start: number; end: number };
export function factSpans(sentences: Sentence[]): Fact[] {
  return FACTS.map((f) => {
    const ss = f.sentences.map((t) => sentences.find((x) => x.text === t)!);
    return { id: f.id, label: f.label, start: ss[0].start, end: ss[ss.length - 1].end };
  });
}

export type Chunk = { index: number; start: number; end: number };

export function sanitize(p: ChunkParams): ChunkParams {
  const size = Math.max(5, Math.min(200, Math.round(p.size)));
  return { ...p, size, overlap: Math.max(0, Math.min(Math.round(p.overlap), size - 1)), threshold: Math.max(0, Math.min(1, p.threshold)) };
}

/** Similarity of each sentence to the one before it (index i compares i-1 and i). Headings are glued to the next sentence. */
export function adjacentSimilarity(sentences: Sentence[]): { from: number; to: number; sim: number }[] {
  const units = sentenceUnits(sentences);
  const vecs = units.map((u) => embed(u.text, "embed-v1"));
  return units.slice(1).map((u, i) => ({ from: units[i].start, to: u.start, sim: dot(vecs[i], vecs[i + 1]) }));
}

/** A heading on its own is not a meaningful sentence; merge it into the sentence that follows. */
function sentenceUnits(sentences: Sentence[]) {
  const out: { start: number; end: number; text: string }[] = [];
  let pending: Sentence | null = null;
  for (const s of sentences) {
    if (s.heading) { pending = s; continue; }
    out.push(pending ? { start: pending.start, end: s.end, text: `${pending.text}. ${s.text}` } : { start: s.start, end: s.end, text: s.text });
    pending = null;
  }
  return out;
}

/** Pack consecutive [start,end) units into chunks of at most `size` words; `hardBreak(i)` forces a cut before unit i. */
function pack(units: { start: number; end: number }[], size: number, hardBreak: (i: number) => boolean): Chunk[] {
  const out: Chunk[] = [];
  let cur: Chunk | null = null;
  units.forEach((u, i) => {
    if (cur && (hardBreak(i) || u.end - cur.start > size)) { out.push(cur); cur = null; }
    if (!cur) cur = { index: out.length, start: u.start, end: u.end };
    else cur.end = u.end;
  });
  if (cur) out.push(cur);
  return out.map((c, i) => ({ ...c, index: i }));
}

export function chunkDoc(pIn: ChunkParams, doc = buildDoc()): Chunk[] {
  const p = sanitize(pIn);
  const n = doc.words.length;
  switch (p.strategy) {
    case "fixed":
    case "overlap": {
      const step = p.strategy === "overlap" ? p.size - p.overlap : p.size;
      const out: Chunk[] = [];
      for (let s = 0; s < n; s += step) {
        out.push({ index: out.length, start: s, end: Math.min(n, s + p.size) });
        if (s + p.size >= n) break;
      }
      return out;
    }
    case "sentence":
      return pack(doc.sentences, p.size, () => false);
    case "heading": {
      const units = doc.sentences;
      return pack(units, p.size, (i) => units[i].heading);
    }
    case "semantic": {
      const units = sentenceUnits(doc.sentences);
      const sims = adjacentSimilarity(doc.sentences);
      return pack(units, p.size, (i) => i > 0 && sims[i - 1].sim < p.threshold);
    }
  }
}

export type Stats = { count: number; sizes: number[]; mean: number; min: number; max: number; std: number; split: string[]; splitPct: number; overlapWords: number };
export function stats(chunks: Chunk[], facts: Fact[]): Stats {
  const sizes = chunks.map((c) => c.end - c.start);
  const m = mean(sizes);
  const split = facts.filter((f) => !chunks.some((c) => c.start <= f.start && c.end >= f.end)).map((f) => f.id);
  const covered = chunks.reduce((s, c) => s + (c.end - c.start), 0);
  const total = chunks.length ? Math.max(...chunks.map((c) => c.end)) - Math.min(...chunks.map((c) => c.start)) : 0;
  return {
    count: chunks.length, sizes, mean: m, min: Math.min(...sizes), max: Math.max(...sizes),
    std: Math.sqrt(mean(sizes.map((s) => (s - m) ** 2))), split, splitPct: facts.length ? split.length / facts.length : 0, overlapWords: covered - total,
  };
}

export type ChunkPhase = "doc" | "candidates" | "cut" | "overlap" | "cards" | "sizes" | "facts" | "compare";
export type ChunkState = { phase: ChunkPhase };
const PHASES: { phase: ChunkPhase; title: string }[] = [
  { phase: "doc", title: "The document" },
  { phase: "candidates", title: "Where may we cut?" },
  { phase: "cut", title: "Cut lines go in" },
  { phase: "overlap", title: "Overlap" },
  { phase: "cards", title: "Chunks become cards" },
  { phase: "sizes", title: "Chunk size distribution" },
  { phase: "facts", title: "Facts across boundaries" },
  { phase: "compare", title: "Compare all strategies" },
];

export function candidates(p: ChunkParams, doc = buildDoc()): number[] {
  const s = sanitize(p);
  if (s.strategy === "fixed" || s.strategy === "overlap") {
    const step = s.strategy === "overlap" ? s.size - s.overlap : s.size;
    return Array.from({ length: Math.ceil(doc.words.length / step) - 1 }, (_, i) => (i + 1) * step).filter((x) => x < doc.words.length);
  }
  if (s.strategy === "heading") return doc.sentences.filter((x) => x.heading && x.start > 0).map((x) => x.start);
  if (s.strategy === "semantic") return adjacentSimilarity(doc.sentences).filter((x) => x.sim < s.threshold).map((x) => x.to);
  return doc.sentences.slice(1).map((x) => x.start);
}

export function simulate(pIn: ChunkParams) {
  const p = sanitize(pIn);
  const doc = buildDoc();
  const chunks = chunkDoc(p, doc);
  const facts = factSpans(doc.sentences);
  const st = stats(chunks, facts);
  const cand = candidates(p, doc);
  let e = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id: `e${e++}`, type: "log", message, severity, tex });
  const desc: Record<ChunkPhase, string> = {
    doc: `The handbook has ${doc.words.length} words in ${SECTIONS.length} sections. Underlined sentences are facts users will ask about.`,
    candidates: STRATEGIES[p.strategy].rule,
    cut: `The chunker places ${chunks.length - 1} cut${chunks.length === 2 ? "" : "s"}, producing ${chunks.length} chunks.`,
    overlap: p.strategy === "overlap" ? `Each chunk starts ${p.overlap} words before the previous one ended, so ${st.overlapWords} words are stored twice.` : "This strategy has no overlap: every word belongs to exactly one chunk.",
    cards: "Each chunk is what gets embedded and retrieved as one unit. A chunk that starts mid-sentence carries no context about what it refers to.",
    sizes: `Sizes range from ${st.min} to ${st.max} words (mean ${st.mean.toFixed(1)}). Very small chunks lack context; very large ones dilute the embedding.`,
    facts: `${st.split.length} of ${facts.length} fact sentences are cut by a boundary. A split fact can't be retrieved whole.`,
    compare: "Same document, same size budget, five strategies. Structure-aware cuts keep facts whole at similar chunk counts.",
  };
  const events = (phase: ChunkPhase): SimulationEvent[] => {
    switch (phase) {
      case "doc": return [ev(`${doc.words.length} words, ${doc.sentences.length} sentences (incl. ${SECTIONS.length} headings)`)];
      case "candidates": return [ev(`${cand.length} candidate cut positions`)];
      case "cut": return chunks.slice(1).map((c) => ev(`cut before word ${c.start} (“${doc.words[c.start].text}”)`));
      case "overlap": return [ev(`${st.overlapWords} words duplicated across chunks`)];
      case "cards": return [ev(`${chunks.length} chunks created`)];
      case "sizes": return [ev(`sizes: ${st.sizes.join(", ")}`, "info", `\\bar{s} = ${st.mean.toFixed(1)},\\ \\sigma = ${st.std.toFixed(1)}`)];
      case "facts": return facts.map((f) => ev(`${st.split.includes(f.id) ? "SPLIT" : "intact"}: ${f.label}`, st.split.includes(f.id) ? "warning" : "info"));
      case "compare": return [ev(`${(st.splitPct * 100).toFixed(0)}% of facts split with ${STRATEGIES[p.strategy].name}`)];
    }
  };
  const steps: SimulationStep<ChunkState>[] = PHASES.map(({ phase, title }) => ({ id: phase, title, description: desc[phase], state: { phase }, events: events(phase) }));
  return { steps, doc, chunks, facts, stats: st, candidates: cand, params: p };
}

export function compareAll(p: ChunkParams) {
  const doc = buildDoc();
  const facts = factSpans(doc.sentences);
  return (Object.keys(STRATEGIES) as Strategy[]).map((s) => ({ strategy: s, stats: stats(chunkDoc({ ...p, strategy: s }, doc), facts) }));
}
