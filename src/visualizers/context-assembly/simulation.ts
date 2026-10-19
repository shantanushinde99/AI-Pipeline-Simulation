import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { tokens } from "../rag/simulation";
import { estimateTokens } from "../top-k-retrieval/simulation";

export type Chunk = { id: string; title: string; text: string; rel: number; tokens: number };
export const QUERY = "Can I get a refund on an annual plan after three weeks?";
/** The fact the answer needs. It survives only if a selected chunk contains it in full. */
export const FACT = "30 days of purchase";

const raw: [string, string, number][] = [
  ["Refund policy (v2)", "Refund policy. Northwind Cloud bills monthly or annually. This page explains when a charge can be refunded and how to request one through the billing portal or by contacting the support team.", 0.92],
  ["Refund policy (v1, older copy)", "Refund policy. Northwind Cloud bills monthly or annually. This page explains when a charge can be refunded and how to request one through the billing portal or support.", 0.91],
  ["Monthly refunds", "Monthly plans can be refunded within 14 days of the charge if usage stayed under the free tier limits.", 0.88],
  ["Monthly refunds (FAQ copy)", "Monthly plans can be refunded within 14 days of the charge when usage stayed under free tier limits.", 0.86],
  ["Annual plans", "Annual plans are billed once per year and renew automatically unless you cancel in the billing portal before the renewal date. Annual plans can be refunded within 30 days of purchase, prorated for any usage above the free tier.", 0.84],
  ["Cancel renewal", "After the refund window closes you can still cancel renewal from the billing portal.", 0.8],
  ["Annual discount", "Annual billing gives you two months free on any paid plan.", 0.77],
  ["Invoices", "Invoices are emailed to the billing contact three days after each charge.", 0.7],
];
export const CHUNKS: Chunk[] = raw.map(([title, text, rel], i) => ({ id: `C${i + 1}`, title, text, rel, tokens: estimateTokens(text) }));

/** Jaccard similarity of normalized token sets: |A ∩ B| / |A ∪ B|. */
export function jaccard(a: string, b: string) {
  const A = new Set(tokens(a)), B = new Set(tokens(b));
  const inter = [...A].filter((x) => B.has(x)).length;
  const union = new Set([...A, ...B]).size;
  return union ? inter / union : 0;
}
export const SIM: number[][] = CHUNKS.map((a) => CHUNKS.map((b) => (a === b ? 1 : jaccard(a.text, b.text))));
const sim = (a: string, b: string) => SIM[CHUNKS.findIndex((c) => c.id === a)][CHUNKS.findIndex((c) => c.id === b)];

export type Strategy = "greedy" | "dedupe" | "mmr";
export type Ordering = "relevance" | "ends";
export type Overflow = "truncate" | "drop";
export type AssemblyInput = { strategy: Strategy; budget: number; lambda: number; dupThreshold: number; ordering: Ordering; overflow: Overflow };
export const DEFAULT_INPUT: AssemblyInput = { strategy: "greedy", budget: 130, lambda: 0.5, dupThreshold: 0.6, ordering: "relevance", overflow: "truncate" };

/** Keep the first `room` tokens' worth of words (toy tokenizer: 1.3 tokens per word). */
export function truncate(text: string, room: number) {
  const words = text.split(/\s+/).filter(Boolean);
  let n = Math.floor(room / 1.3);
  while (n > 0 && estimateTokens(words.slice(0, n).join(" ")) > room) n--;
  const kept = words.slice(0, Math.max(0, n)).join(" ");
  return { text: kept, tokens: estimateTokens(kept) };
}

export type Picked = { id: string; tokens: number; text: string; truncated: boolean };
export type Dropped = { id: string; reason: string; kind: "duplicate" | "budget" | "unused" };
export type MmrRow = { id: string; rel: number; maxSim: number; maxSimWith: string | null; score: number };
export type Decision = {
  id: string; action: "add" | "truncate" | "drop-dup" | "drop-budget";
  used: number; room: number; maxSim?: number; maxSimWith?: string | null; mmr?: MmrRow[];
};
export type AsmState = {
  phase: "candidates" | "consider" | "order" | "result";
  picked: Picked[]; dropped: Dropped[]; used: number; decision?: Decision; order?: string[];
};

function maxSimTo(id: string, picked: Picked[]) {
  let best = 0, with_: string | null = null;
  for (const p of picked) { const s = sim(id, p.id); if (s > best) { best = s; with_ = p.id; } }
  return { maxSim: best, maxSimWith: with_ };
}

/** "Best at both ends": ranks 1, 3, 5… fill from the front, ranks 2, 4, 6… fill from the back. */
export function bothEnds<T>(items: T[]): T[] {
  const front: T[] = [], back: T[] = [];
  items.forEach((it, i) => (i % 2 === 0 ? front.push(it) : back.unshift(it)));
  return [...front, ...back];
}

export function assemble(inputRaw: AssemblyInput) {
  const input = { ...inputRaw, budget: Math.max(10, Math.round(inputRaw.budget)), lambda: Math.min(1, Math.max(0, inputRaw.lambda)), dupThreshold: Math.min(1, Math.max(0, inputRaw.dupThreshold)) };
  const picked: Picked[] = [], dropped: Dropped[] = [], decisions: Decision[] = [];
  let used = 0, stopped = false;
  const remaining = [...CHUNKS];
  while (remaining.length && !stopped) {
    let c: Chunk, mmr: MmrRow[] | undefined;
    if (input.strategy === "mmr") {
      mmr = remaining.map((r) => { const m = maxSimTo(r.id, picked); return { id: r.id, rel: r.rel, ...m, score: input.lambda * r.rel - (1 - input.lambda) * m.maxSim }; });
      const best = mmr.reduce((a, b) => (b.score > a.score ? b : a));
      c = remaining.find((r) => r.id === best.id)!;
    } else c = remaining[0];
    remaining.splice(remaining.indexOf(c), 1);
    const room = input.budget - used;
    const { maxSim, maxSimWith } = maxSimTo(c.id, picked);
    const base = { id: c.id, used, room, maxSim, maxSimWith, mmr };
    if (input.strategy === "dedupe" && maxSim >= input.dupThreshold) {
      dropped.push({ id: c.id, kind: "duplicate", reason: `near-duplicate of ${maxSimWith} (Jaccard ${maxSim.toFixed(2)} ≥ ${input.dupThreshold})` });
      decisions.push({ ...base, action: "drop-dup" });
    } else if (c.tokens <= room) {
      picked.push({ id: c.id, tokens: c.tokens, text: c.text, truncated: false });
      used += c.tokens;
      decisions.push({ ...base, action: "add" });
    } else if (input.overflow === "truncate" && room > 0) {
      const t = truncate(c.text, room);
      if (t.tokens > 0) { picked.push({ id: c.id, tokens: t.tokens, text: t.text, truncated: true }); used += t.tokens; }
      decisions.push({ ...base, action: "truncate" });
      dropped.push({ id: c.id, kind: "budget", reason: `truncated: kept ${t.tokens} of ${c.tokens} tokens` });
      stopped = true;
    } else {
      dropped.push({ id: c.id, kind: "budget", reason: `${c.tokens} tokens > ${room} left` });
      decisions.push({ ...base, action: "drop-budget" });
    }
  }
  for (const r of remaining) dropped.push({ id: r.id, kind: "unused", reason: "never considered: budget filled by a truncated chunk" });
  const ranked = [...picked].sort((a, b) => CHUNKS.find((c) => c.id === b.id)!.rel - CHUNKS.find((c) => c.id === a.id)!.rel);
  const order = (input.ordering === "ends" ? bothEnds(ranked) : ranked).map((p) => p.id);
  const factChunk = picked.find((p) => p.text.includes(FACT));
  const factPos = factChunk ? order.indexOf(factChunk.id) : -1;
  const middle = factPos > 0 && factPos < order.length - 1 && order.length >= 3;
  return { input, picked, dropped, decisions, used, order, factChunk, factPos, middle };
}

export type Assembly = ReturnType<typeof assemble>;

export function simulate(input: AssemblyInput): { asm: Assembly; steps: SimulationStep<AsmState>[] } {
  const asm = assemble(input);
  const B = asm.input.budget;
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "context", message, severity, tex });
  const steps: SimulationStep<AsmState>[] = [{
    id: "candidates", title: "Candidates and a token budget",
    description: `${CHUNKS.length} reranked chunks (${CHUNKS.reduce((s, c) => s + c.tokens, 0)} tokens in total) compete for a ${B}-token context budget. Two pairs are near-duplicates.`,
    state: { phase: "candidates", picked: [], dropped: [], used: 0 },
    events: [ev("b", `budget B = ${B} tokens, strategy = ${asm.input.strategy}`)],
  }];
  const picked: Picked[] = [], dropped: Dropped[] = [];
  asm.decisions.forEach((d, i) => {
    const c = CHUNKS.find((x) => x.id === d.id)!;
    const p = asm.picked.find((x) => x.id === d.id);
    if (p) picked.push(p);
    const dr = asm.dropped.find((x) => x.id === d.id);
    if (dr) dropped.push(dr);
    const after = picked.reduce((s, x) => s + x.tokens, 0);
    const what = d.action === "add" ? `fits (${d.used} + ${c.tokens} = ${after} ≤ ${B}), added`
      : d.action === "truncate" ? `does not fit (${c.tokens} > ${d.room} left): cut to ${p?.tokens ?? 0} tokens, assembly stops`
        : d.action === "drop-dup" ? `skipped: near-duplicate of ${d.maxSimWith} (similarity ${d.maxSim!.toFixed(2)})`
          : `does not fit (${c.tokens} > ${d.room} left): dropped, try the next one`;
    steps.push({
      id: `consider-${d.id}`, title: `${asm.input.strategy === "mmr" ? `MMR pick ${i + 1}` : `Consider ${d.id}`}: ${d.action === "add" ? "add" : d.action === "truncate" ? "truncate" : "drop"}`,
      description: `${d.id} “${c.title}” (${c.tokens} tokens) ${what}.`,
      state: { phase: "consider", picked: [...picked], dropped: [...dropped], used: after, decision: d },
      events: [
        ...(d.mmr ? [ev(`mmr-${d.id}`, `MMR(${d.id}) = ${asm.input.lambda}·${c.rel} − ${(1 - asm.input.lambda).toFixed(2)}·${(d.maxSim ?? 0).toFixed(2)}`, undefined,
          `\\text{MMR}(${d.id}) = ${asm.input.lambda}\\cdot${c.rel} - ${(1 - asm.input.lambda).toFixed(2)}\\cdot${(d.maxSim ?? 0).toFixed(2)} = ${d.mmr.find((m) => m.id === d.id)!.score.toFixed(3)}`)] : []),
        ev(`d-${d.id}`, `${d.id}: ${what}`, d.action === "add" ? undefined : "warning"),
      ],
    });
  });
  steps.push({
    id: "order", title: asm.input.ordering === "ends" ? "Order: best at both ends" : "Order: most relevant first",
    description: asm.input.ordering === "ends"
      ? `Ranks 1, 3, 5… go to the front and ranks 2, 4… to the back, so the strongest chunks sit where models attend most. Order: ${asm.order.join(", ")}.`
      : `Chunks are placed by relevance, best first. Order: ${asm.order.join(", ")}.`,
    state: { phase: "order", picked: asm.picked, dropped: asm.dropped, used: asm.used, order: asm.order },
    events: [ev("o", `final order: ${asm.order.join(" → ") || "empty"}`, asm.middle ? "warning" : undefined), ...(asm.middle ? [ev("mid", `answer chunk ${asm.factChunk!.id} sits in the middle (position ${asm.factPos + 1} of ${asm.order.length})`, "warning")] : [])],
  });
  const lost = !asm.factChunk;
  steps.push({
    id: "result", title: lost ? "The answer fact did not survive" : "The answer fact survived",
    description: lost
      ? `“${FACT}” is not in the assembled context${asm.dropped.find((d) => d.id === "C5") ? `: C5 was ${asm.dropped.find((d) => d.id === "C5")!.reason}` : ""}. The model can only guess or refuse.`
      : `“${FACT}” is in ${asm.factChunk!.id} at position ${asm.factPos + 1} of ${asm.order.length}. ${asm.used} of ${B} tokens used.`,
    state: { phase: "result", picked: asm.picked, dropped: asm.dropped, used: asm.used, order: asm.order },
    events: [ev("fact", lost ? "answer fact lost" : "answer fact present", lost ? "error" : asm.middle ? "warning" : undefined)],
  });
  return { asm, steps };
}
