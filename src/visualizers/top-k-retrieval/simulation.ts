import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** 0 = not relevant, 1 = partially relevant (on topic, no answer), 2 = fully relevant (contains the answer). */
export type Grade = 0 | 1 | 2;
export type Candidate = { id: string; title: string; text: string; score: number; grade: Grade; tokens: number };
export type Preset = { id: string; name: string; query: string; candidates: Candidate[]; missed: { title: string; grade: Grade }[]; note: string };

/** Toy token estimate: ~1.3 tokens per English word (real tokenizers vary). */
export const estimateTokens = (text: string) => Math.round(text.split(/\s+/).filter(Boolean).length * 1.3);

type Raw = [title: string, text: string, score: number, grade: Grade];
const build = (id: string, rows: Raw[]): Candidate[] =>
  rows.map(([title, text, score, grade], i) => ({ id: `${id}-${i + 1}`, title, text, score, grade, tokens: estimateTokens(text) }));

export const PRESETS: Preset[] = [
  {
    id: "easy",
    name: "Easy query: answer at rank 1",
    query: "How long are request logs kept on the Enterprise plan?",
    note: "The retriever did its job: the answer is the top hit, so small k is enough.",
    missed: [],
    candidates: build("easy", [
      ["Data retention", "Request logs are retained for 30 days on Starter and Pro plans and for 365 days on the Enterprise plan.", 0.91, 2],
      ["Deleted projects", "Deleted projects can be recovered for 7 days before they are permanently removed.", 0.74, 0],
      ["Plans and pricing", "Northwind Cloud offers three plans. Every plan includes unlimited projects and the web dashboard.", 0.69, 0],
      ["Audit log export", "Enterprise admins can export the audit log as CSV. Exports cover the same retention window as request logs.", 0.66, 1],
      ["Enterprise plan", "Enterprise includes custom pricing, unlimited seats and 24/7 phone support.", 0.61, 0],
      ["API rate limits", "The Pro plan allows 600 requests per minute per API key. The Starter plan allows 60 requests per minute.", 0.58, 0],
      ["Status page", "Incident history on the status page is kept for 90 days.", 0.55, 0],
      ["SSO setup", "SSO is available on the Enterprise plan only. An admin uploads the identity provider metadata XML file.", 0.51, 0],
      ["Refund policy", "Annual plans can be refunded within 30 days of purchase, prorated for usage above the free tier.", 0.47, 0],
      ["Billing portal", "After the refund window closes you can still cancel renewal from the billing portal.", 0.44, 0],
    ]),
  },
  {
    id: "rank7",
    name: "Hard query: answer buried at rank 7",
    query: "Can I get a refund on an annual plan after three weeks?",
    note: "Look-alike chunks about billing outrank the one that answers. Only k ≥ 7 retrieves the answer.",
    missed: [],
    candidates: build("rank7", [
      ["Annual billing", "Annual billing gives you two months free on any paid plan.", 0.88, 0],
      ["Monthly refunds", "Monthly plans can be refunded within 14 days of the charge if usage stayed under the free tier limits.", 0.86, 1],
      ["Refund policy intro", "Northwind Cloud bills monthly or annually. This page explains when a charge can be refunded and how to request one.", 0.84, 1],
      ["Plans and pricing", "Northwind Cloud offers three plans. Every plan includes unlimited projects and the web dashboard.", 0.79, 0],
      ["Cancel renewal", "After the refund window closes you can still cancel renewal from the billing portal.", 0.77, 0],
      ["Invoices", "Invoices are emailed to the billing contact three days after each charge.", 0.74, 0],
      ["Annual refunds", "Annual plans can be refunded within 30 days of purchase, prorated for any usage above the free tier.", 0.72, 2],
      ["Plan upgrades", "Upgrading mid-cycle charges the prorated difference immediately.", 0.69, 0],
      ["Free tier", "The free tier includes 10 GB of storage and 1 million requests per month.", 0.65, 0],
      ["Data retention", "Request logs are retained for 30 days on Starter and Pro plans.", 0.6, 0],
    ]),
  },
  {
    id: "missed",
    name: "Retriever missed a relevant chunk",
    query: "How do I enable SSO?",
    note: "Two chunks are needed (upload XML, then verify DNS). The DNS one never made the candidate list, so recall can't reach 1 at any k.",
    missed: [{ title: "Verify domain: add a DNS TXT record (ranked 14th, not in the list)", grade: 2 }],
    candidates: build("missed", [
      ["SSO overview", "Single sign-on lets your team log in with your company identity provider. SSO is available on the Enterprise plan only.", 0.9, 1],
      ["Security settings", "The Security page lists active sessions, API keys and two-factor settings for each member.", 0.82, 0],
      ["Upload IdP metadata", "To enable SSO an organization admin opens Settings then Security then Single sign-on and uploads the identity provider metadata XML file.", 0.8, 2],
      ["Two-factor auth", "Members can enable two-factor authentication with any TOTP app from their profile.", 0.73, 0],
      ["Enterprise plan", "Enterprise includes custom pricing, unlimited seats and 24/7 phone support.", 0.7, 0],
      ["Roles", "Organization admins can invite members, change roles and remove users.", 0.66, 0],
      ["Login troubleshooting", "If you cannot log in, clear cookies and check that your company identity provider is reachable.", 0.62, 1],
      ["API keys", "API keys are created per project and can be rotated at any time.", 0.57, 0],
      ["Password reset", "Use the reset link on the sign-in page to choose a new password.", 0.53, 0],
      ["Audit log", "Enterprise admins can export the audit log as CSV.", 0.49, 0],
    ]),
  },
];

export type Threshold = 1 | 2;
const isRel = (g: Grade, thr: Threshold) => g >= thr;

// ---------- ranking metrics (exported; the reranking visualizer reuses them) ----------
export const precisionAtK = (grades: Grade[], k: number, thr: Threshold = 1) =>
  k <= 0 ? 0 : grades.slice(0, k).filter((g) => isRel(g, thr)).length / k;
export const recallAtK = (grades: Grade[], k: number, totalRelevant: number, thr: Threshold = 1) =>
  totalRelevant <= 0 ? 0 : grades.slice(0, k).filter((g) => isRel(g, thr)).length / totalRelevant;
/** 1 / rank of the first relevant item within the top k (0 if none). Averaged over queries this is MRR. */
export function reciprocalRank(grades: Grade[], k: number, thr: Threshold = 1) {
  const i = grades.slice(0, k).findIndex((g) => isRel(g, thr));
  return i < 0 ? 0 : 1 / (i + 1);
}
/** Gain 2^g − 1 discounted by log2(rank + 1). */
export const gain = (g: number) => 2 ** g - 1;
export const discount = (rank: number) => Math.log2(rank + 1);
export const dcgAtK = (grades: number[], k: number) => grades.slice(0, k).reduce((s, g, i) => s + gain(g) / discount(i + 1), 0);
/** `pool` = every graded item for this query (including relevant items the retriever missed) for the ideal ordering. */
export function ndcgAtK(grades: number[], k: number, pool: number[] = grades) {
  const ideal = dcgAtK([...pool].sort((a, b) => b - a), k);
  return ideal === 0 ? 0 : dcgAtK(grades, k) / ideal;
}

export type KMetrics = { k: number; precision: number; recall: number; rr: number; ndcg: number; dcg: number; idcg: number; tokens: number; relInK: number };
export function metricsAt(p: Preset, k: number, thr: Threshold): KMetrics {
  const grades = p.candidates.map((c) => c.grade);
  const pool = [...grades, ...p.missed.map((m) => m.grade)];
  const total = pool.filter((g) => isRel(g as Grade, thr)).length;
  const kk = Math.max(0, Math.min(k, grades.length));
  return {
    k: kk,
    precision: precisionAtK(grades, kk, thr),
    recall: recallAtK(grades, kk, total, thr),
    rr: reciprocalRank(grades, kk, thr),
    ndcg: ndcgAtK(grades, kk, pool),
    dcg: dcgAtK(grades, kk),
    idcg: dcgAtK([...pool].sort((a, b) => b - a), kk),
    tokens: p.candidates.slice(0, kk).reduce((s, c) => s + c.tokens, 0),
    relInK: grades.slice(0, kk).filter((g) => isRel(g, thr)).length,
  };
}
export const totalRelevant = (p: Preset, thr: Threshold) => [...p.candidates.map((c) => c.grade), ...p.missed.map((m) => m.grade)].filter((g) => isRel(g, thr)).length;

export type TopKState = { phase: "list" | "grow" | "pick"; k: number; m: KMetrics; added?: Candidate };
export type TopKInput = { presetId: string; threshold: Threshold; chosenK: number };

export function simulate(input: TopKInput): SimulationStep<TopKState>[] {
  const p = PRESETS.find((x) => x.id === input.presetId) ?? PRESETS[0];
  const thr: Threshold = input.threshold === 2 ? 2 : 1;
  const n = p.candidates.length;
  const chosenK = Math.max(1, Math.min(n, Math.round(input.chosenK)));
  const R = totalRelevant(p, thr);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "metric", message, severity, tex });
  const steps: SimulationStep<TopKState>[] = [{
    id: "list", title: "Retriever returns a ranked list",
    description: `The retriever scored the collection and returned its ${n} best candidates for “${p.query}”, sorted by score. Nothing has been sent to the LLM yet (k = 0).`,
    state: { phase: "list", k: 0, m: metricsAt(p, 0, thr) },
    events: [ev("list", `${n} candidates, ${R} relevant chunk(s) exist for this query${p.missed.length ? ` (${p.missed.length} not in the list)` : ""}.`)],
  }];
  for (let k = 1; k <= n; k++) {
    const c = p.candidates[k - 1];
    const m = metricsAt(p, k, thr);
    const rel = isRel(c.grade, thr);
    steps.push({
      id: `k${k}`, title: `k = ${k}: add rank ${k}`,
      description: `Rank ${k} (“${c.title}”, ${c.tokens} tokens) joins the retrieved set. It is ${c.grade === 2 ? "fully relevant" : c.grade === 1 ? "partially relevant" : "not relevant"}${rel ? "" : thr === 2 && c.grade === 1 ? ", which does not count under the strict threshold" : ""}.`,
      state: { phase: "grow", k, m, added: c },
      events: [
        ev(`p${k}`, `P@${k} = ${m.relInK}/${k} = ${m.precision.toFixed(3)}`, undefined, `P@${k} = \\tfrac{${m.relInK}}{${k}} = ${m.precision.toFixed(3)}`),
        ev(`r${k}`, `R@${k} = ${m.relInK}/${R} = ${m.recall.toFixed(3)}`, undefined, `R@${k} = \\tfrac{${m.relInK}}{${R}} = ${m.recall.toFixed(3)}`),
        ...(rel && m.relInK === 1 ? [ev(`rr${k}`, `First relevant chunk at rank ${k}: RR = 1/${k}`, k > 3 ? "warning" : undefined, `\\text{RR} = \\tfrac{1}{${k}} = ${m.rr.toFixed(3)}`)] : []),
        ev(`t${k}`, `Context grows to ${m.tokens} tokens (+${c.tokens})`),
      ],
    });
  }
  const pick = metricsAt(p, chosenK, thr);
  const full = steps.findIndex((s) => s.state.k > 0 && s.state.m.recall >= 1 - 1e-9);
  steps.push({
    id: "pick", title: `Your choice: k = ${chosenK}`,
    description: `At k = ${chosenK}: precision ${pick.precision.toFixed(2)}, recall ${pick.recall.toFixed(2)}, nDCG ${pick.ndcg.toFixed(2)}, ${pick.tokens} context tokens. ${full > 0 ? `Recall first reaches 1 at k = ${steps[full].state.k}.` : "Recall never reaches 1: a relevant chunk is missing from the candidate list."}`,
    state: { phase: "pick", k: chosenK, m: pick },
    events: [ev("pick", `k = ${chosenK} → ${pick.tokens} tokens`, pick.recall < 1 ? "warning" : undefined)],
  });
  return steps;
}
