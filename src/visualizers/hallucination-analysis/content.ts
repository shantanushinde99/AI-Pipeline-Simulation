import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Split an answer into atomic claims, check each against sources and compute a hallucination rate.", minutes: 12, hasMetrics: false };

const t = "hallucination-analysis";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Hallucination analysis measures how much of a model's answer is not backed by the evidence it had. Instead of grading the whole answer as right or wrong, you decompose it into atomic claims (one attribute, one value each), check every claim against the sources, and label it supported, contradicted or unverifiable." },
    { heading: "Why claim-level checking", body: "A paragraph with five correct facts and one invented number looks 'mostly right' to a reader, and an answer-level grade hides it. Claim-level labels show exactly which statement failed and why, and the rate (hallucinated claims ÷ all claims) is comparable across answers of different lengths." },
    { heading: "Hallucination types", body: "Fabricated citation: a reference to a source that does not exist. Entity swap: the right kind of fact with the wrong name. Number drift: a measurement or date that moved. Unsupported generalization: the source says 'some', the answer says 'all'. Each type points to a different fix: citation validation, entity grounding, numeric checks, or prompting for hedged language." },
    { heading: "Unverifiable is not the same as false", body: "If a source is missing from retrieval, true claims become unverifiable. Strict graders count them as hallucinations (the answer asserted something the evidence cannot back); lenient graders only count contradictions. Pick one convention and report it next to the number." },
    { heading: "Assumptions in this visualizer", body: "The observatory, its sources and the four answers are fictional and hand-written. The claim decomposition is authored by hand, standing in for an LLM decomposer. The checker is a real but simple rule set over structured facts: citation exists, attribute match, exact match for years, relative tolerance for measurements, a some/all quantifier rule, and string equality for names. Real systems use an NLI model or an LLM judge for matching, which adds its own error rate." },
  ],
  code: [
    {
      title: "claim_check.py — LLM-judged claim verification (sketch)",
      source: `from dataclasses import dataclass

@dataclass
class Verdict:
    claim: str
    label: str          # "supported" | "contradicted" | "unverifiable"
    evidence_id: str | None

def decompose(answer: str, llm) -> list[str]:
    prompt = "Split into atomic factual claims, one per line:\\n" + answer
    return [c.strip() for c in llm(prompt).splitlines() if c.strip()]

def verify(claim: str, sources: dict[str, str], nli) -> Verdict:
    best = None
    for sid, text in sources.items():
        label, score = nli(premise=text, hypothesis=claim)  # entail / contradict / neutral
        if label != "neutral" and (best is None or score > best[2]):
            best = (sid, label, score)
    if best is None:
        return Verdict(claim, "unverifiable", None)
    return Verdict(claim, "supported" if best[1] == "entail" else "contradicted", best[0])

def hallucination_rate(verdicts, count_unverifiable=True):
    bad = sum(v.label == "contradicted" or (count_unverifiable and v.label == "unverifiable") for v in verdicts)
    return bad / max(1, len(verdicts))`,
    },
    {
      title: "Cheap pre-checks before any judge (TypeScript)",
      source: `const CITE = /\\[(S\\d+)\\]/g;

export function fabricatedCitations(answer: string, retrievedIds: Set<string>) {
  return [...answer.matchAll(CITE)].map((m) => m[1]).filter((id) => !retrievedIds.has(id));
}

export function numbersNotInSources(answer: string, sources: string[], tol = 0.05) {
  const nums = (s: string) => [...s.matchAll(/\\d[\\d,]*(\\.\\d+)?/g)].map((m) => Number(m[0].replace(/,/g, "")));
  const known = sources.flatMap(nums);
  return nums(answer).filter((n) => !known.some((k) => Math.abs(n - k) <= tol * Math.abs(k)));
}`,
    },
  ],
  failureModes: [
    { title: "Fabricated citations pass review", symptom: "Answers cite [S4] or [S5]; reviewers trust the brackets.", detect: "Check every cited id against the retrieved set before anything else; the 'Citation-happy' preset shows two.", fix: "Validate citation ids programmatically and only let the model cite ids you injected.", scenarioId: "rag-citation" },
    { title: "Fluent answer, drifted numbers", symptom: "A 3.6 m mirror where the source says 2.4 m; the prose still reads well.", detect: "Claim-level numeric checks with a tolerance per unit; years and ids need exact matches.", fix: "Quote numbers from context verbatim, or post-check numbers against sources and regenerate.", scenarioId: "rag-unfaithful" },
    { title: "Tolerance set too loose", symptom: "Hallucination rate looks great, yet readers find wrong numbers.", detect: "Raise the tolerance slider to 30%: the 'Subtle drift' mirror claim and more slip through as supported.", fix: "Set tolerances per unit from what a human would call the same fact; audit a sample by hand." },
    { title: "Missing evidence counted as model error", symptom: "Rate jumps after a retrieval change although the generator is unchanged.", detect: "Drop S3 from retrieval: true claims flip to unverifiable, not contradicted.", fix: "Report contradicted and unverifiable separately; diagnose retrieval recall before blaming generation.", scenarioId: "rag-stale-index" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "The 'Confident but wrong' answer has 6 claims: 2 supported and 4 contradicted. What is its hallucination rate?", options: ["33%", "50%", "67%", "100%"], answer: 2, explanation: "Rate = (contradicted + unverifiable) / claims = (4 + 0) / 6 ≈ 67%. It doesn't matter whether unverifiable claims are counted, because there are none." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "In 'Subtle drift', the answer says the mirror is 2.5 m; the source says 2.4 m. At what tolerance does that claim flip from supported to contradicted?", options: ["Below about 4.2%", "Below about 10%", "Only at 0%", "It is always contradicted because it is a number"], answer: 0, explanation: "|2.5 − 2.4| / 2.4 ≈ 0.0417. With a tolerance below 4.17% the drift counts as contradicted; at 5% it passes." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "After a retrieval change, the 'Grounded answer' shows two unverifiable claims and zero contradicted ones. Most likely cause?", options: ["The model started inventing numbers", "A source that backed those claims is no longer retrieved", "The tolerance is too strict", "Citations were fabricated"], answer: 1, explanation: "Drop S3 in the visualizer: altitude and clear-nights claims lose their evidence and become unverifiable, while nothing is contradicted. That is a retrieval problem." },
    { id: `${t}-4`, topicId: t, skill: "prediction", question: "The 'Citation-happy' answer scores 60% with unverifiable claims counted. What does it score if you only count contradictions?", options: ["60%", "40%", "20%", "0%"], answer: 3, explanation: "All three problems (two fabricated citations, one over-generalization) are unverifiable rather than contradicted, so a lenient grader reports 0%. That is why the convention must be stated." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Which fix targets fabricated citations most directly?", options: ["Lower temperature", "Validate cited ids against the retrieved set and reject unknown ids", "Add more documents to the index", "Raise the numeric tolerance"], answer: 1, explanation: "A citation is fabricated when it points outside the evidence. A deterministic id check catches every one of them before any judge model runs." },
  ],
};
