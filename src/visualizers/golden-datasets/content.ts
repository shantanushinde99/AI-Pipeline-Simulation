import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Build a golden set from logs: dedupe, double-label with κ, fill coverage gaps, and version it.", minutes: 12, hasMetrics: false };

const t = "golden-datasets";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A golden dataset is a curated, labelled, versioned set of examples that every release is evaluated against. It is the fixed yardstick that makes scores comparable over time. Good golden sets are built from real traffic, cleaned of duplicates, labelled carefully, and deliberately cover the categories that matter even when they are rare." },
    { heading: "Label quality: inter-annotator agreement", body: "Two reviewers label each example independently. Cohen's κ = (pₒ − pₑ)/(1 − pₑ) measures their agreement beyond chance; low κ means the labeling guide is ambiguous. Disagreements go to adjudication. Agreement is not correctness: two reviewers can agree on the same wrong label, which only an audit catches." },
    { heading: "Coverage and the blind-spot problem", body: "A golden set can only detect regressions in what it contains. Random sampling copies production's skew, so rare but important categories (other languages, safety) get few or no examples. A gate that ignores tiny categories (to avoid noise) then cannot see a regression there at all. Set explicit per-category coverage targets and fill gaps by stratified or targeted sampling." },
    { heading: "Versioning", body: "Freeze each version with a fingerprint (a hash of item ids and labels). Report scores with the version they were computed on; when you add examples or fix labels you create v2, and old scores stay comparable only to v1. Keep a changelog: what was added, removed and relabelled, and why." },
    { heading: "Assumptions in this visualizer", body: "The production log is generated from hand-written query templates with seeded randomness (category frequencies, order numbers, filler words). Reviewers are simulated: each gives the correct label with the chosen probability, otherwise a random wrong one; adjudication always recovers the true label. The candidate model's pass/fail per example is simulated (≈3% random baseline failures, plus every example in the regressed category fails). Normalization, Jaccard dedupe, Cohen's κ, the FNV-1a fingerprint and the release gate are real computations." },
  ],
  code: [
    {
      title: "build_golden.py — dedupe, κ, coverage",
      source: `import re, hashlib
from collections import Counter
from sklearn.metrics import cohen_kappa_score

def norm(t): return re.sub(r"\\s+", " ", re.sub(r"[^\\w\\s]|\\b(please|thanks)\\b", " ", t.lower())).strip()
def jaccard(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / len(A | B)

kept = []
for q in candidates:                                  # near-duplicate removal
    if all(jaccard(q.text, k.text) < 0.85 for k in kept):
        kept.append(q)

kappa = cohen_kappa_score([q.r1 for q in kept], [q.r2 for q in kept])
assert kappa >= 0.6, "labeling guide is ambiguous"

coverage = Counter(q.category for q in kept)
gaps = {c: TARGET - coverage[c] for c in CATEGORIES if coverage[c] < TARGET}
fingerprint = hashlib.sha256("|".join(f"{q.id}:{q.label}" for q in kept).encode()).hexdigest()[:8]`,
    },
    {
      title: "Release gate with per-category rule (TypeScript)",
      source: `type Result = { category: string; pass: boolean };

export function gate(results: Result[], minSupport = 3) {
  const overall = results.filter((r) => r.pass).length / results.length;
  const byCat = Map.groupBy(results, (r) => r.category);
  const failing = [...byCat].filter(([, rs]) => rs.length >= minSupport && rs.filter((r) => r.pass).length / rs.length < 0.8);
  return { pass: overall >= 0.9 && failing.length === 0, overall, failing: failing.map(([c]) => c) };
}
// The gate is only as good as the coverage: a category with < minSupport examples is never checked.`,
    },
  ],
  failureModes: [
    { title: "Coverage gap hides a regression", symptom: "A release passes the golden set, then a whole user segment (e.g. Spanish speakers) reports failures.", detect: "Per-category coverage report against explicit targets; categories with fewer than the gate's minimum support are blind spots.", fix: "Stratified or targeted sampling to fill gaps; make coverage targets part of the dataset's definition of done." },
    { title: "Duplicates inflate scores", symptom: "Golden-set accuracy is high, but driven by a few popular questions repeated many times.", detect: "Normalize and near-dedupe (Jaccard/MinHash or embedding similarity); count repeats per cluster.", fix: "Keep one representative per cluster; weight by traffic separately if needed." },
    { title: "Noisy labels", symptom: "Models 'fail' examples whose labels are wrong; scores swing with reviewer changes.", detect: "Double-label a sample and track Cohen's κ; audit agreed labels periodically.", fix: "Tighten the labeling guide, adjudicate disagreements, version label fixes." },
    { title: "Silent edits to the golden set", symptom: "Scores jump between releases with no model change.", detect: "Fingerprint/hash the dataset and store it with every eval run.", fix: "Immutable versions (v1, v2…) with changelogs; compare runs only within a version." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "In the default run the candidate is broken on every non-English query, yet the gate passes on v1. Why?", options: ["Non-English queries are easy", "v1 has fewer than 3 non-English examples, so the per-category rule never checks that category, and the overall rate stays above 90%", "The reviewers mislabelled them", "Dedupe removed the regression"], answer: 1, explanation: "Random sampling produced only 2 non-English examples. Both fail, but the gate ignores categories below minimum support, and 2 failures barely dent the overall rate." },
    { id: `${t}-2`, topicId: t, skill: "fix", question: "Which change lets v1 itself catch the non-English regression?", options: ["More bootstrap resamples", "Stratified sampling by category", "Raising reviewer accuracy", "Lowering the overall threshold to 80%"], answer: 1, explanation: "With a per-category quota, non-English gets enough examples (≥ 3) for the per-category rule to fire. Try the 'Stratified sampling' preset." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Two reviewers agree on 90% of items, but one label dominates (~70% 'answer' for both). Why can κ still be modest?", options: ["κ ignores agreement", "Much of the agreement is expected by chance when both mostly pick the same label, and κ subtracts that chance agreement", "κ only works for binary labels", "κ rewards disagreement"], answer: 1, explanation: "κ = (pₒ − pₑ)/(1 − pₑ). With skewed marginals pₑ is large, so the same pₒ gives a smaller κ." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "Lower reviewer accuracy to 60%. What happens in the labeling and adjudication steps?", options: ["κ rises", "κ falls and many more items need adjudication; some agreed labels are wrong", "Nothing changes", "Duplicates increase"], answer: 1, explanation: "Less accurate reviewers disagree more (lower κ, more adjudication), and they occasionally agree on the same wrong label, which adjudication never sees." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Why does the visualizer give v1 and v2 different fingerprints, and why does that matter?", options: ["It's cosmetic", "The fingerprint hashes item ids and labels, so any added item or label fix changes it; scores must be reported with the version they were computed on", "v2 is encrypted", "Hashes speed up evaluation"], answer: 1, explanation: "A dataset fingerprint makes silent edits detectable and keeps score comparisons honest: a 92% on v1 and 85% on v2 are not the same yardstick." },
  ],
};
