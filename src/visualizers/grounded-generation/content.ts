import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch an answer get written claim by claim, each linked to the span that supports it, or flagged.", minutes: 12, hasMetrics: false };

const t = "grounded-generation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Grounded generation means every statement in an answer is backed by the retrieved context. The generator is told to use only the numbered passages, to cite them, and to say it doesn't know when they are silent. A grounded answer can be traced claim by claim to a span in a source document." },
    { heading: "Why it matters", body: "Users trust RAG answers because they look sourced. A single sentence from model memory (a stale price, an invented policy) breaks that trust while reading exactly like the grounded sentences around it. Groundedness is measured per claim: supported claims ÷ total claims." },
    { heading: "Abstention is a feature", body: "When the context is missing the fact, the right output is 'the documents don't say'. A generator that always answers will produce a confident guess instead. Toggle 'context missing the fact' and then 'abstain when evidence is weak' to see both behaviours on the same question." },
    { heading: "Grounded is not the same as correct", body: "Lower the evidence threshold τ with the fact removed: the generator happily quotes the Starter plan's limit for a Pro plan question. Every word is supported by the context, so groundedness stays at 100%, but the answer addresses the wrong thing. That is why relevance and correctness are evaluated separately." },
    { heading: "Assumptions in this visualizer", body: "The generator is rule-based, not a language model: the answer outline (what each sentence should cover) is scripted per question, and each sentence is copied from the context sentence with the highest share of the point's search terms. Guesses and 'memory' facts are hand-written strings. The verifier is a word-overlap check (≥ 60% of claim words, numbers must match) standing in for an NLI model or LLM judge. Retrieval reuses the simulated RAG pipeline (hashed bag-of-words embeddings)." },
  ],
  code: [
    {
      title: "Grounded prompt + per-claim check (Python)",
      source: `SYSTEM = """Answer using ONLY the numbered passages.
After every sentence, cite the passage it came from as [n].
If the passages do not contain the answer, reply exactly:
"The provided documents don't say." Do not use outside knowledge."""

def groundedness(answer_sentences, passages, judge):
    """judge(claim, passage) -> True if the passage entails the claim (NLI model or LLM judge)."""
    supported = 0
    for claim in answer_sentences:
        if any(judge(claim, p) for p in passages):
            supported += 1
    return supported / len(answer_sentences) if answer_sentences else None`,
    },
    {
      title: "Abstain on weak evidence (TypeScript)",
      source: `const best = rankSpans(question, contextSpans)[0];
if (!best || best.score < EVIDENCE_THRESHOLD) {
  return { text: "The provided documents don't say.", citations: [] };
}
return generateWithCitations(question, contextSpans);`,
    },
  ],
  failureModes: [
    { title: "Answer drawn from model memory", symptom: "A specific, outdated number (300 req/min) appears even though the context says 600.", detect: "Claim-level support check: the claim's number is absent from every context span.", fix: "Grounding instruction + abstention rule in the prompt; block answers whose claims fail verification.", scenarioId: "rag-unfaithful" },
    { title: "Confident guess when the fact is missing", symptom: "The answer sounds certain but no passage contains it.", detect: "Evidence for that point is below threshold, yet a claim was written; groundedness drops.", fix: "Require an explicit 'don't know' path and evaluate abstention rate on questions whose answer is absent." },
    { title: "Grounded but off-target", symptom: "Every sentence is supported, yet the user's question is not answered (Starter limit given for a Pro question).", detect: "Groundedness 100% but answer correctness or relevance low.", fix: "Raise the evidence threshold, add a relevance check, retrieve better context." },
    { title: "Unsupported filler sentences", symptom: "Extra helpful-sounding details (processing times, escalation paths) with no source.", detect: "Claims with no linked span in the verifier output.", fix: "Strip or flag unsupported sentences before showing the answer; ask for citations per sentence." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "On the rate-limit question, you turn on 'allow prior knowledge'. What happens to groundedness?", options: ["Stays 100%", "Drops to 67%: 2 of 3 claims are supported", "Drops to 0%", "Becomes undefined"], answer: 1, explanation: "The extra memory claim has no supporting span, so it becomes 1 unsupported claim out of 3: 2/3 ≈ 67%." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "With the fact removed and abstention off, the answer says 'The Pro plan allows 300 requests per minute.' Why does the verifier reject it even though 5 of 6 words match a span?", options: ["The sentence is too short", "The number 300 does not appear in the span (it says 600)", "It cites the wrong passage", "Coverage must be 100%"], answer: 1, explanation: "Word coverage is 83%, above the 60% bar, but strict mode requires every number in the claim to appear in the span. Numbers are where hallucinations hurt most." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "Fact removed, abstention on. You lower τ from 0.75 to 0.5. What changes?", options: ["Nothing", "The generator now writes the Starter plan's limit as the answer: grounded but wrong", "It abstains on every point", "Groundedness drops to 0%"], answer: 1, explanation: "The Starter sentence covers 2/3 of {pro, request, minute} = 0.67 ≥ 0.5, so it passes. It is supported by the context, so groundedness stays 100%, but it answers a different question." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "An answer abstains on every point. What is its groundedness?", options: ["0%", "100%", "Undefined: there are no claims to score", "50%"], answer: 2, explanation: "Groundedness = supported / claims. With zero claims the ratio is undefined; report abstention rate separately instead of pretending it is 0% or 100%." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Which change most directly prevents confident guesses when retrieval misses the fact?", options: ["Raise temperature", "Add more passages regardless of relevance", "An explicit abstain path plus a minimum-evidence threshold", "Remove citations"], answer: 2, explanation: "If no span clears the evidence bar, the generator should say it doesn't know. That is exactly the 'abstain when evidence is weak' toggle." },
  ],
};
