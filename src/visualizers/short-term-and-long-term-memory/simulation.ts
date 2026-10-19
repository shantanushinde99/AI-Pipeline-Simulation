import { clamp, cosine } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Slot = "diet" | "date" | "city";
export const SLOTS: Slot[] = ["diet", "city", "date"];

/** Hand-picked concept dimensions: a readable stand-in for a learned embedding. */
export const DIMS = ["food", "date", "place", "money", "person", "lodging"] as const;
const KEYWORDS: Record<string, number> = {
  vegetarian: 0, vegan: 0, meat: 0, eat: 0, dinner: 0, restaurant: 0, food: 0,
  may: 1, date: 1, arrive: 1, arriving: 1, flight: 1, moved: 1, when: 1, fly: 1, flying: 1,
  lisbon: 2, city: 2, portugal: 2,
  budget: 3, "€150": 3, price: 3, cost: 3,
  name: 4, priya: 4,
  hotel: 5, night: 5, room: 5, stay: 5,
};
export function words(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9€]+/g) ?? []).map((w) => (w.length > 3 && w.endsWith("s") && !w.endsWith("ss") ? w.slice(0, -1) : w));
}
/** Keyword-count vector over DIMS. */
export function vectorize(text: string): number[] {
  const v = DIMS.map(() => 0);
  for (const w of words(text)) if (w in KEYWORDS) v[KEYWORDS[w]]++;
  return v;
}
/** Toy tokenizer: one token per whitespace-separated word. */
export const countTokens = (text: string) => text.split(/\s+/).filter(Boolean).length;

type FactDef = { id: string; text: string; values: Partial<Record<Slot, string>>; summary: string; summaryValues: Partial<Record<Slot, string>> };
type ExchangeDef = { user: string; assistant: string; fact?: FactDef; conflictOnly?: boolean };

export const EXCHANGES: ExchangeDef[] = [
  { user: "Hi, I'm Priya.", assistant: "Hi Priya! How can I help with your trip?", fact: { id: "name", text: "Name: Priya", values: {}, summary: "User is Priya.", summaryValues: {} } },
  { user: "I'm flying to Lisbon and I arrive on 12 May.", assistant: "Lovely, Lisbon in May is great.", fact: { id: "trip", text: "Flight to Lisbon, arrive 12 May", values: { city: "Lisbon", date: "12 May" }, summary: "Trip to Lisbon planned.", summaryValues: { city: "Lisbon" } } },
  { user: "I'm vegetarian, so no meat please.", assistant: "Noted: vegetarian places only.", fact: { id: "diet", text: "Diet: vegetarian, eat no meat", values: { diet: "vegetarian" }, summary: "Vegetarian.", summaryValues: { diet: "vegetarian" } } },
  { user: "Change of plan: my flight moved, I now arrive 19 May.", assistant: "Got it, arriving 19 May.", conflictOnly: true, fact: { id: "update", text: "Flight moved: arrive 19 May", values: { date: "19 May" }, summary: "Flight was moved.", summaryValues: {} } },
  { user: "My hotel budget is €150 per night.", assistant: "I'll keep hotels under €150.", fact: { id: "hotel", text: "Hotel budget €150 per night", values: {}, summary: "Hotel budget set.", summaryValues: {} } },
  { user: "What's the weather like there in spring?", assistant: "Usually sunny, around 20°C." },
];
export const QUESTION = "Book dinner for my first night in Lisbon: when do I arrive, and what can I eat?";

export type MemoryConfig = { windowTokens: number; summarize: boolean; longTerm: boolean; conflict: boolean; topK: number; threshold: number };
export const DEFAULT_MEMORY: MemoryConfig = { windowTokens: 44, summarize: true, longTerm: true, conflict: false, topK: 3, threshold: 0.3 };
export function sanitize(c: MemoryConfig): MemoryConfig {
  return { ...c, windowTokens: Math.round(clamp(c.windowTokens, 36, 200)), topK: Math.round(clamp(c.topK, 1, 5)), threshold: clamp(c.threshold, 0, 1) };
}

export type Turn = { id: string; role: "user" | "assistant"; text: string; tokens: number; exchange: number; fact?: FactDef };
export type SummaryLine = { id: string; text: string; tokens: number; values: Partial<Record<Slot, string>> };
export type StoredFact = { id: string; text: string; values: Partial<Record<Slot, string>>; vector: number[]; turn: number };
export type Hit = { fact: StoredFact; score: number; retrieved: boolean };
export type SlotAnswer = { slot: Slot; value: string | null; source: "short-term" | "summary" | "long-term" | "missing"; correct: boolean; conflict?: string };
export type Phase = "intro" | "exchange" | "question" | "retrieve" | "answer";
export type MemState = {
  phase: Phase;
  buffer: Turn[];
  summary: SummaryLine[];
  store: StoredFact[];
  entering: string[];
  evicted: { turn: Turn; summarized: boolean }[];
  written: string[];
  used: number;
  hits: Hit[];
  answer?: { slots: SlotAnswer[]; text: string; correct: number };
};

export const expected = (c: MemoryConfig): Record<Slot, string> => ({ diet: "vegetarian", city: "Lisbon", date: c.conflict ? "19 May" : "12 May" });

export function simulate(cfgIn: MemoryConfig): SimulationStep<MemState>[] {
  const c = sanitize(cfgIn);
  const exchanges = EXCHANGES.filter((e) => c.conflict || !e.conflictOnly);
  const steps: SimulationStep<MemState>[] = [];
  let buffer: Turn[] = [];
  let summary: SummaryLine[] = [];
  const store: StoredFact[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "memory", message, severity, tex });
  const used = () => buffer.reduce((a, t) => a + t.tokens, 0) + summary.reduce((a, s) => a + s.tokens, 0);
  const snap = (phase: Phase, extra: Partial<MemState>): MemState => structuredClone({
    phase, buffer, summary, store, entering: [], evicted: [], written: [], used: used(), hits: [], ...extra,
  });

  /** Evict oldest turns (never the ones just added) until the window fits. */
  const fit = (protect: Set<string>) => {
    const evicted: { turn: Turn; summarized: boolean }[] = [];
    while (used() > c.windowTokens) {
      const victim = buffer.find((t) => !protect.has(t.id));
      if (!victim) break;
      buffer = buffer.filter((t) => t !== victim);
      const line = c.summarize && victim.role === "user" ? { id: victim.id, text: victim.fact?.summary ?? "Small talk.", tokens: countTokens(victim.fact?.summary ?? "Small talk."), values: victim.fact?.summaryValues ?? {} } : null;
      if (line) summary = [...summary, line];
      evicted.push({ turn: victim, summarized: line !== null });
    }
    return evicted;
  };

  steps.push({
    id: "intro", title: "Empty memory",
    description: `The context window holds ${c.windowTokens} tokens. ${c.summarize ? "Evicted user turns are compressed into a running summary." : "Evicted turns are simply dropped."} ${c.longTerm ? "Salient facts are also written to a long-term vector store." : "There is no long-term store."}`,
    state: snap("intro", {}),
    events: [ev("cfg", `window ${c.windowTokens} tokens · summary ${c.summarize ? "on" : "off"} · long-term ${c.longTerm ? "on" : "off"}`)],
  });

  exchanges.forEach((e, i) => {
    const n = i + 1;
    const u: Turn = { id: `u${n}`, role: "user", text: e.user, tokens: countTokens(e.user), exchange: n, fact: e.fact };
    const a: Turn = { id: `a${n}`, role: "assistant", text: e.assistant, tokens: countTokens(e.assistant), exchange: n };
    buffer = [...buffer, u, a];
    const evicted = fit(new Set([u.id, a.id]));
    const written: string[] = [];
    if (c.longTerm && e.fact) {
      store.push({ id: e.fact.id, text: e.fact.text, values: e.fact.values, vector: vectorize(e.fact.text), turn: n });
      written.push(e.fact.id);
    }
    steps.push({
      id: `ex${n}`, title: `Turn ${n}: "${e.user.length > 34 ? `${e.user.slice(0, 32)}…` : e.user}"`,
      description: `The exchange adds ${u.tokens + a.tokens} tokens.${evicted.length ? ` The window overflows, so the ${evicted.length} oldest turn${evicted.length > 1 ? "s are" : " is"} ${c.summarize ? "evicted and summarised" : "evicted and lost"}.` : " Everything still fits."}${written.length ? ` The fact "${e.fact!.text}" is written to long-term memory.` : ""}`,
      state: snap("exchange", { entering: [u.id, a.id], evicted, written }),
      events: [
        ev(`in${n}`, `+${u.tokens + a.tokens} tokens → ${used()} / ${c.windowTokens}`),
        ...evicted.map((x) => ev(`ev-${x.turn.id}`, `evicted ${x.turn.role} turn "${x.turn.text}"${x.summarized && x.turn.role === "user" ? " → summary" : ""}`, "warning")),
        ...written.map((w) => ev(`w-${w}`, `long-term write: ${e.fact!.text}`)),
      ],
    });
  });

  const q: Turn = { id: "q", role: "user", text: QUESTION, tokens: countTokens(QUESTION), exchange: exchanges.length + 1 };
  buffer = [...buffer, q];
  const evQ = fit(new Set(["q"]));
  steps.push({
    id: "question", title: "A question that needs old facts",
    description: `The user asks: "${QUESTION}" Answering needs the diet, the city and the arrival date, all mentioned turns ago.`,
    state: snap("question", { entering: ["q"], evicted: evQ }),
    events: [ev("q", `question added: ${q.tokens} tokens`), ...evQ.map((x) => ev(`evq-${x.turn.id}`, `evicted "${x.turn.text}"`, "warning"))],
  });

  const qv = vectorize(QUESTION);
  const scored = store.map((f) => ({ fact: f, score: cosine(qv, f.vector) || 0 })).sort((x, y) => y.score - x.score);
  const hits: Hit[] = scored.map((h, r) => ({ ...h, retrieved: r < c.topK && h.score >= c.threshold }));
  steps.push({
    id: "retrieve", title: c.longTerm ? "Retrieve from long-term memory" : "No long-term memory to search",
    description: c.longTerm
      ? `The question is embedded and compared with every stored fact by cosine similarity. The top ${c.topK} with score ≥ ${c.threshold} are added to the context: ${hits.filter((h) => h.retrieved).map((h) => `"${h.fact.text}" (${h.score.toFixed(2)})`).join(", ") || "none"}.`
      : "Long-term memory is off, so the model can only use what is still in the window and the summary.",
    state: snap("retrieve", { hits }),
    events: hits.map((h) => ev(`h-${h.fact.id}`, `${h.fact.text}: ${h.score.toFixed(3)}`, h.retrieved ? "info" : undefined, `\\cos(q, \\text{${h.fact.id}}) = ${h.score.toFixed(3)}${h.retrieved ? "\\ \\checkmark" : ""}`)),
  });

  const exp = expected(c);
  const slots: SlotAnswer[] = SLOTS.map((slot) => {
    const inBuf = [...buffer].reverse().find((t) => t.fact?.values[slot]);
    if (inBuf) return { slot, value: inBuf.fact!.values[slot]!, source: "short-term" as const, correct: inBuf.fact!.values[slot] === exp[slot] };
    const inSum = summary.find((s) => s.values[slot]);
    if (inSum) return { slot, value: inSum.values[slot]!, source: "summary" as const, correct: inSum.values[slot] === exp[slot] };
    const mem = hits.filter((h) => h.retrieved && h.fact.values[slot]);
    if (mem.length) {
      const distinct = [...new Set(mem.map((h) => h.fact.values[slot]!))];
      const v = mem[0].fact.values[slot]!;
      return { slot, value: v, source: "long-term" as const, correct: v === exp[slot], conflict: distinct.length > 1 ? `memory holds ${distinct.join(" and ")}; the higher-similarity fact wins` : undefined };
    }
    return { slot, value: null, source: "missing" as const, correct: false };
  });
  const val = (s: Slot) => slots.find((x) => x.slot === s)!.value;
  const missing = slots.filter((s) => !s.value).map((s) => s.slot);
  const text = `${val("diet") ? `Booked a ${val("diet")} dinner` : "Booked a dinner"}${val("city") ? ` in ${val("city")}` : ""}${val("date") ? ` on ${val("date")}` : ""}.${missing.length ? ` (I don't know your ${missing.join(" or ")}: could you remind me?)` : ""}`;
  const correct = slots.filter((s) => s.correct).length;
  steps.push({
    id: "answer", title: correct === 3 ? "Answer: every fact recalled" : `Answer: ${3 - correct} fact${3 - correct > 1 ? "s" : ""} wrong or missing`,
    description: `Each needed fact is looked up in priority order: short-term window (newest first), then the summary, then retrieved long-term facts. ${slots.map((s) => `${s.slot}: ${s.value ?? "missing"} (${s.source})`).join("; ")}.`,
    state: snap("answer", { hits, answer: { slots, text, correct } }),
    events: slots.map((s) => ev(`s-${s.slot}`, `${s.slot} = ${s.value ?? "?"} from ${s.source}${s.conflict ? ` (conflict: ${s.conflict})` : ""}`, s.correct ? "info" : "error")),
  });
  return steps;
}
