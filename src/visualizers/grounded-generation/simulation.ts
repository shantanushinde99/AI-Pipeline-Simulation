import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { QUESTIONS, type Question } from "../rag/corpus";
import { DEFAULT_RAG, runRag, tokens } from "../rag/simulation";
import { bestSupport, sentenceSpans, type Span, type SupportCheck } from "./claims";

/** What one answer sentence should cover. `phrase` is turned into search terms; `guess` is what the model says without evidence. */
export type Intent = { label: string; phrase: string; guess: string };
export type Scenario = { id: string; question: Question; intents: Intent[]; prior: string };

const q = (id: string) => QUESTIONS.find((x) => x.id === id)!;

export const SCENARIOS: Scenario[] = [
  {
    id: "rate", question: q("q-rate"),
    intents: [
      { label: "the Pro plan's limit", phrase: "Pro requests per minute", guess: "The Pro plan allows 300 requests per minute." },
      { label: "what happens above the limit", phrase: "above limit", guess: "Requests above the limit are queued and retried automatically." },
    ],
    prior: "Enterprise customers can request a custom rate limit from their account manager.",
  },
  {
    id: "refund", question: q("q-refund"),
    intents: [
      { label: "the annual-plan refund window", phrase: "annual refund", guess: "Annual plans can be refunded within 60 days of purchase." },
      { label: "how monthly plans compare", phrase: "monthly refund", guess: "Monthly plans are refundable within 7 days." },
    ],
    prior: "Refunds usually reach your card within 5 to 10 business days.",
  },
  {
    id: "sso", question: q("q-sso"),
    intents: [
      { label: "which plan includes SSO", phrase: "SSO plan", guess: "SSO is available on the Pro and Enterprise plans." },
      { label: "the setup steps", phrase: "enable SSO", guess: "An admin enables SSO in the Admin console by pasting the identity provider URL." },
      { label: "the final verification", phrase: "verifies domain", guess: "The admin then confirms a verification email." },
    ],
    prior: "SSO changes usually take effect within one hour.",
  },
];

export type GGParams = { scenarioId: string; tau: number; allowPrior: boolean; missingFact: boolean; abstain: boolean };
export const DEFAULT_GG: GGParams = { scenarioId: "rate", tau: 0.75, allowPrior: false, missingFact: false, abstain: true };

export type Passage = { label: number; chunkId: string; file: string; text: string; original: string; removed: string[] };
/** Evidence for one intent: coverage of the intent's search terms by every span. */
export type Evidence = { terms: string[]; scores: { spanId: string; coverage: number; matched: string[] }[]; best: { spanId: string; coverage: number } | null };
export type AnswerSentence = {
  id: string;
  intent: number | null; // null = extra claim from parametric memory
  label: string;
  text: string;
  kind: "claim" | "abstain";
  origin: "context" | "guess" | "prior" | "none";
  evidence: Evidence | null;
  check: SupportCheck | null; // verifier result (claims only)
};
export type GGResult = {
  params: GGParams; scenario: Scenario; passages: Passage[]; spans: Span[];
  sentences: AnswerSentence[]; claims: number; supported: number; abstentions: number; groundedness: number | null;
};

export function sanitize(p: GGParams): GGParams {
  const tau = Number.isFinite(p.tau) ? Math.min(1, Math.max(0.25, p.tau)) : DEFAULT_GG.tau;
  return { ...p, tau, scenarioId: SCENARIOS.some((s) => s.id === p.scenarioId) ? p.scenarioId : DEFAULT_GG.scenarioId };
}

export function evidenceFor(phrase: string, spans: Span[]): Evidence {
  const terms = [...new Set(tokens(phrase))];
  const scores = spans.map((s) => {
    const have = new Set(tokens(s.text));
    const matched = terms.filter((t) => have.has(t));
    return { spanId: s.id, coverage: terms.length ? matched.length / terms.length : 0, matched };
  });
  const best = scores.reduce<Evidence["best"]>((a, b) => (b.coverage > (a?.coverage ?? 0) ? { spanId: b.spanId, coverage: b.coverage } : a), null);
  return { terms, scores, best };
}

/** Retrieval (from the RAG simulation) → optional removal of the fact → one sentence per intent → verification. */
export function generate(pIn: GGParams): GGResult {
  const params = sanitize(pIn);
  const scenario = SCENARIOS.find((s) => s.id === params.scenarioId)!;
  const fact = scenario.question.expect.toLowerCase();
  const run = runRag(DEFAULT_RAG, scenario.question.id);
  const passages: Passage[] = run.context.map((c) => {
    const sentences = c.text.split(/(?<=[.!?])\s+/);
    const removed = params.missingFact ? sentences.filter((s) => s.toLowerCase().includes(fact)) : [];
    return { label: c.label, chunkId: c.entry.id, file: run.index.sources.find((d) => d.id === c.entry.docId)?.file ?? c.entry.docId, text: sentences.filter((s) => !removed.includes(s)).join(" "), original: c.text, removed };
  });
  const spans = sentenceSpans(passages);
  const spanText = (id: string) => spans.find((s) => s.id === id)!.text;

  const sentences: AnswerSentence[] = scenario.intents.map((intent, i) => {
    const evidence = evidenceFor(intent.phrase, spans);
    const found = evidence.best && evidence.best.coverage >= params.tau ? evidence.best : null;
    if (found) {
      const text = spanText(found.spanId);
      return { id: `s${i}`, intent: i, label: intent.label, text, kind: "claim", origin: "context", evidence, check: bestSupport(text, spans) };
    }
    if (params.abstain) {
      return { id: `s${i}`, intent: i, label: intent.label, text: `The provided documents don't state ${intent.label}.`, kind: "abstain", origin: "none", evidence, check: null };
    }
    return { id: `s${i}`, intent: i, label: intent.label, text: intent.guess, kind: "claim", origin: "guess", evidence, check: bestSupport(intent.guess, spans) };
  });
  if (params.allowPrior) {
    sentences.push({ id: "prior", intent: null, label: "extra detail from memory", text: scenario.prior, kind: "claim", origin: "prior", evidence: null, check: bestSupport(scenario.prior, spans) });
  }
  const claimList = sentences.filter((s) => s.kind === "claim");
  const supported = claimList.filter((s) => s.check!.supported).length;
  return {
    params, scenario, passages, spans, sentences,
    claims: claimList.length, supported, abstentions: sentences.length - claimList.length,
    groundedness: claimList.length ? supported / claimList.length : null,
  };
}

export type GGState = { phase: "context" | "plan" | "write" | "score"; shown: number; focus: number | null };

export function simulate(pIn: GGParams): { result: GGResult; steps: SimulationStep<GGState>[] } {
  const result = generate(pIn);
  const { scenario, passages, spans, sentences, params } = result;
  let n = 0;
  const ev = (message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id: `e${n++}`, type: "generate", message, severity, tex });
  const steps: SimulationStep<GGState>[] = [];
  const removed = passages.flatMap((p) => p.removed);

  steps.push({
    id: "context", title: "Question + retrieved context",
    description: `The generator receives the question and ${passages.length} numbered passages (${spans.length} sentences). It may only state what these passages support.`,
    state: { phase: "context", shown: 0, focus: null },
    events: [
      ev(`question: "${scenario.question.text}"`),
      ev(`context: ${passages.map((p) => `[${p.label}] ${p.chunkId}`).join(", ")}`),
      ...(removed.length ? [ev(`fault: the sentence holding "${scenario.question.expect}" is missing from the context`, "warning")] : []),
    ],
  });
  steps.push({
    id: "plan", title: "Plan what the answer must cover",
    description: `The (scripted) answer outline has ${scenario.intents.length} points. For each, the generator searches the context for a sentence covering at least τ = ${params.tau.toFixed(2)} of its terms.`,
    state: { phase: "plan", shown: 0, focus: null },
    events: scenario.intents.map((it, i) => ev(`point ${i + 1}: ${it.label} → terms {${[...new Set(tokens(it.phrase))].join(", ")}}`)),
  });
  sentences.forEach((s, i) => {
    const e = s.evidence;
    const best = e?.best;
    const events: SimulationEvent[] = [];
    if (e) events.push(ev(`best evidence ${best ? `${best.spanId} covers ${(best.coverage * 100).toFixed(0)}%` : "none"} of {${e.terms.join(", ")}}`, undefined,
      `\\text{evidence} = \\frac{|\\text{terms} \\cap \\text{span}|}{|\\text{terms}|} = ${best ? best.coverage.toFixed(2) : "0"} ${best && best.coverage >= params.tau ? "\\ge" : "<"} \\tau = ${params.tau.toFixed(2)}`));
    if (s.kind === "abstain") events.push(ev("evidence below τ and abstention is on → the generator says it doesn't know"));
    else if (s.origin === "guess") events.push(ev("evidence below τ but abstention is off → confident guess", "warning"));
    else if (s.origin === "prior") events.push(ev("prior knowledge allowed → an extra claim from parametric memory", "warning"));
    if (s.check) events.push(ev(s.check.supported ? `verified: supported by ${s.check.spanId} (${(s.check.coverage * 100).toFixed(0)}% of claim words)` : `unsupported: best span ${s.check.spanId ?? "none"} covers ${(s.check.coverage * 100).toFixed(0)}%${s.check.missingNumbers.length ? `, numbers ${s.check.missingNumbers.join(", ")} absent` : ""}`, s.check.supported ? undefined : "error"));
    steps.push({
      id: `write-${s.id}`,
      title: s.intent === null ? "Extra claim from memory" : `Sentence ${i + 1}: ${s.label}`,
      description: s.kind === "abstain"
        ? `No sentence covers enough of {${e!.terms.join(", ")}}, so the generator abstains on ${s.label} instead of inventing it.`
        : s.origin === "context"
          ? `The best sentence (${best!.spanId}) covers ${(best!.coverage * 100).toFixed(0)}% of the terms, so it is written into the answer and linked to that span.`
          : s.origin === "guess"
            ? `Evidence is too weak (${((best?.coverage ?? 0) * 100).toFixed(0)}% < ${(params.tau * 100).toFixed(0)}%), yet abstention is off: the generator writes a fluent guess that no passage supports.`
            : "With prior knowledge allowed, the generator adds a plausible detail it 'remembers'. Nothing in the context supports it.",
      state: { phase: "write", shown: i + 1, focus: i },
      events,
    });
  });
  const g = result.groundedness;
  steps.push({
    id: "score", title: "Groundedness score",
    description: g === null
      ? "The answer makes no claims (it abstained everywhere), so groundedness is undefined; nothing unsupported was said."
      : `${result.supported} of ${result.claims} claims are supported by a context span: groundedness = ${(g * 100).toFixed(0)}%.`,
    state: { phase: "score", shown: sentences.length, focus: null },
    events: [ev(g === null ? "no claims → groundedness n/a" : `groundedness = ${result.supported}/${result.claims} = ${g.toFixed(2)}`, g !== null && g < 1 ? "warning" : undefined,
      g === null ? undefined : `\\text{groundedness} = \\frac{${result.supported}}{${result.claims}} = ${g.toFixed(2)}`)],
  });
  return { result, steps };
}
