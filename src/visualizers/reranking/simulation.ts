import { sigmoid } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { tokens } from "../rag/simulation";
import { estimateTokens, ndcgAtK, type Grade } from "../top-k-retrieval/simulation";

export type Passage = { id: string; text: string; firstScore: number; grade: Grade; firstRank: number; tokens: number };
export type Intent = "permission" | "quantity" | "procedure" | "lookup";
export type RerankPreset = { id: string; name: string; query: string; passages: Passage[] };

type Raw = [text: string, score: number, grade: Grade];
const build = (id: string, rows: Raw[]): Passage[] =>
  rows.map(([text, firstScore, grade], i) => ({ id: `${id}${i + 1}`, text, firstScore, grade, firstRank: i + 1, tokens: estimateTokens(text) }));

export const PRESETS: RerankPreset[] = [
  {
    id: "refund", name: "Annual refund (answer at first-stage rank 7)", query: "Can I get a refund on an annual plan?",
    passages: build("R", [
      ["Annual billing gives you two months free on any paid plan.", 0.89, 0],
      ["Refunds are not available for annual plans bought through a reseller.", 0.87, 1],
      ["Monthly plans can be refunded within 14 days of the charge.", 0.85, 1],
      ["Northwind Cloud bills monthly or annually and this page explains refunds.", 0.83, 0],
      ["Upgrading to an annual plan charges the prorated difference immediately.", 0.8, 0],
      ["After the refund window closes you can still cancel renewal.", 0.78, 0],
      ["Annual plans can be refunded within 30 days of purchase, prorated for usage.", 0.75, 2],
      ["Invoices are emailed to the billing contact three days after each charge.", 0.7, 0],
      ["To request a refund on an annual plan open Billing then Refunds and submit the form.", 0.68, 1],
      ["The free tier includes 10 GB of storage and one million requests.", 0.6, 0],
    ]),
  },
  {
    id: "deleted", name: "Deleted projects (negation trap at rank 1)", query: "How long are deleted projects kept?",
    passages: build("P", [
      ["Deleted projects are not restored by support once the recovery window ends.", 0.9, 1],
      ["Archive a project to hide it without deleting any data.", 0.86, 0],
      ["Request logs are kept for 30 days on Starter and Pro plans.", 0.84, 0],
      ["Projects can be transferred between organizations by an admin.", 0.8, 0],
      ["Deleted projects can be recovered for 7 days before they are permanently removed.", 0.78, 2],
      ["Deleting a project removes its API keys immediately.", 0.74, 0],
      ["Backups of deleted projects are kept for 30 days in cold storage and cannot be restored by customers.", 0.71, 1],
      ["Project names must be unique within an organization.", 0.66, 0],
      ["Deleted workspace data is kept for 90 days.", 0.62, 1],
      ["Deleted files stay in the trash for 14 days.", 0.58, 0],
    ]),
  },
];

// ---------- simulated cross-encoder ----------
export const WEIGHTS = { bias: -2, overlap: 2.5, phrase: 1, intent: 1.5, negation: -2.5 };
export const LATENCY = { fixedMs: 4, perPairMs: 6, perTokenMs: 0.3 };

const INTENT_CUES: Record<Intent, (p: string) => boolean> = {
  permission: (p) => /\b(can be|allowed|within|eligible)\b/.test(p),
  quantity: (p) => /\d/.test(p),
  procedure: (p) => /\b(open|click|go to|then|upload|submit)\b/.test(p),
  lookup: () => false,
};
export function detectIntent(query: string): Intent {
  const q = query.toLowerCase().trim();
  if (/^how (long|many|much)\b/.test(q)) return "quantity";
  if (/^(how (do|can) i|how to)\b/.test(q)) return "procedure";
  if (/^(can|may|is it possible)\b/.test(q)) return "permission";
  return "lookup";
}
/** Question words that only signal intent; they are scored by the intent feature, not by overlap. */
const INTENT_WORDS = new Set(["long", "many", "much"]);
const NEG = /\b(not|no|never|cannot|can't|isn't|aren't|won't)\b/;

export type Features = { overlap: number; matched: string[]; qTerms: string[]; phrase: string | null; intent: boolean; negation: string | null };
export function features(query: string, passage: string, useNegation = true): Features {
  const qSeq = tokens(query).filter((t) => !INTENT_WORDS.has(t));
  const qTerms = [...new Set(qSeq)];
  const pToks = tokens(passage);
  const pSet = new Set(pToks);
  const matched = qTerms.filter((t) => pSet.has(t));
  let phrase: string | null = null;
  for (let i = 0; i + 1 < qSeq.length && !phrase; i++)
    for (let j = 0; j + 1 < pToks.length; j++) if (pToks[j] === qSeq[i] && pToks[j + 1] === qSeq[i + 1]) { phrase = `${qSeq[i]} ${qSeq[i + 1]}`; break; }
  const p = passage.toLowerCase();
  const neg = useNegation && !NEG.test(query.toLowerCase()) ? (p.match(NEG)?.[0] ?? null) : null;
  return { overlap: qTerms.length ? matched.length / qTerms.length : 0, matched, qTerms, phrase, intent: INTENT_CUES[detectIntent(query)](p), negation: neg };
}
export function crossScore(f: Features) {
  const parts = { bias: WEIGHTS.bias, overlap: WEIGHTS.overlap * f.overlap, phrase: f.phrase ? WEIGHTS.phrase : 0, intent: f.intent ? WEIGHTS.intent : 0, negation: f.negation ? WEIGHTS.negation : 0 };
  const z = parts.bias + parts.overlap + parts.phrase + parts.intent + parts.negation;
  return { parts, z, p: sigmoid(z) };
}
export const pairLatency = (p: Passage) => LATENCY.perPairMs + LATENCY.perTokenMs * p.tokens;

export type Scored = Passage & { f: Features; z: number; p: number; parts: ReturnType<typeof crossScore>["parts"]; latency: number };
export type RerankInput = { presetId: string; n: number; useNegation: boolean };

export function rerank(input: RerankInput) {
  const preset = PRESETS.find((p) => p.id === input.presetId) ?? PRESETS[0];
  const N = Math.max(1, Math.min(preset.passages.length, Math.round(input.n)));
  const scored: Scored[] = preset.passages.slice(0, N).map((p) => {
    const f = features(preset.query, p.text, input.useNegation);
    const s = crossScore(f);
    return { ...p, f, z: s.z, p: s.p, parts: s.parts, latency: pairLatency(p) };
  });
  const reranked = [...scored].sort((a, b) => b.z - a.z || a.firstRank - b.firstRank);
  const pool = preset.passages.map((p) => p.grade);
  const before = ndcgAtK(pool, 3, pool);
  const after = ndcgAtK(reranked.map((r) => r.grade), 3, pool);
  const latencyMs = LATENCY.fixedMs + scored.reduce((s, r) => s + r.latency, 0);
  return { preset, N, scored, reranked, before, after, latencyMs, intent: detectIntent(preset.query) };
}

/** Quality/latency trade-off for every N, for the chart. */
export const tradeoff = (presetId: string, useNegation: boolean) =>
  PRESETS.find((p) => p.id === presetId)!.passages.map((_, i) => { const r = rerank({ presetId, n: i + 1, useNegation }); return { n: i + 1, ndcg: r.after, latency: r.latencyMs }; });

export type Phase = "first" | "score" | "sort" | "compare";
export type RerankState = { phase: Phase; i: number; elapsedMs: number };

export function simulate(input: RerankInput) {
  const run = rerank(input);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "rerank", message, severity, tex });
  const steps: SimulationStep<RerankState>[] = [{
    id: "first", title: "First stage: top candidates",
    description: `A fast bi-encoder returned ${run.preset.passages.length} candidates for “${run.preset.query}”. Only the top N = ${run.N} are sent to the slower cross-encoder; the rest are cut.`,
    state: { phase: "first", i: -1, elapsedMs: LATENCY.fixedMs },
    events: [ev("n", `rerank budget N = ${run.N}; query intent detected: ${run.intent}`), ...(run.preset.passages.slice(run.N).some((p) => p.grade === 2) ? [ev("cut", "The passage with the answer is below N and will never be reranked.", "warning")] : [])],
  }];
  let elapsed = LATENCY.fixedMs;
  run.scored.forEach((s, i) => {
    elapsed += s.latency;
    steps.push({
      id: `pair-${s.id}`, title: `Score pair ${i + 1}: (query, ${s.id})`,
      description: `The cross-encoder reads the query and passage ${s.id} together. Overlap ${s.f.matched.length}/${s.f.qTerms.length}${s.f.phrase ? `, phrase “${s.f.phrase}”` : ""}${s.f.intent ? ", answers the intent" : ""}${s.f.negation ? `, negation “${s.f.negation}”` : ""} → relevance ${s.p.toFixed(2)}.`,
      state: { phase: "score", i, elapsedMs: elapsed },
      events: [
        ev(`z-${s.id}`, `z = ${s.z.toFixed(2)}, p = σ(z) = ${s.p.toFixed(3)}`, s.f.negation ? "warning" : undefined, `z = ${s.z.toFixed(2)},\\; \\sigma(z) = ${s.p.toFixed(3)}`),
        ev(`l-${s.id}`, `+${s.latency.toFixed(1)} ms (${s.tokens} tokens) → ${elapsed.toFixed(1)} ms total`),
      ],
    });
  });
  const moves = run.reranked.map((r, i) => `${r.id}: ${r.firstRank} → ${i + 1}`);
  steps.push({
    id: "sort", title: "Re-sort by cross-encoder score",
    description: `Rows re-order by the new scores. ${run.reranked[0].id} (first-stage rank ${run.reranked[0].firstRank}) is now first.`,
    state: { phase: "sort", i: -1, elapsedMs: elapsed },
    events: [ev("moves", moves.join(", "))],
  });
  steps.push({
    id: "compare", title: "Quality vs latency",
    description: `nDCG@3 goes from ${run.before.toFixed(3)} (first stage) to ${run.after.toFixed(3)} (reranked) for ${run.latencyMs.toFixed(1)} ms of simulated reranker time.`,
    state: { phase: "compare", i: -1, elapsedMs: elapsed },
    events: [ev("ndcg", `nDCG@3 ${run.before.toFixed(3)} → ${run.after.toFixed(3)}`, run.after < run.before ? "warning" : undefined)],
  });
  return { run, steps };
}
