import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Score two models on a sliced test set, bootstrap a CI, and catch a slice the aggregate hides.", minutes: 12, hasMetrics: false };

const t = "offline-evaluation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Offline evaluation runs a model over a fixed, labelled dataset before any user sees it and grades every output. Comparing a baseline (A) and a candidate (B) on the same examples gives a paired comparison: the only thing that changed is the model." },
    { heading: "Uncertainty: the bootstrap", body: "A test set is a sample, so its accuracy is an estimate. The bootstrap resamples the examples with replacement many times, recomputes the metric difference each time and reads the 2.5th and 97.5th percentiles as a 95% confidence interval. Resampling A's and B's outcomes for the same example together (paired) keeps the interval honest about their correlation." },
    { heading: "Slices and the aggregate trap", body: "An aggregate is a weighted average of slice metrics, weighted by how many examples each slice has in the eval set. A large slice improving can outweigh a small slice collapsing, so the headline number rises while one group of users gets a much worse product. If the eval mix differs from production traffic, re-weighting by the production mix can even flip the winner. This is the same mechanism behind Simpson's paradox: the mix of groups changes the conclusion." },
    { heading: "When it matters", body: "Every model, prompt or retrieval change. Offline evals are cheap and repeatable, so they should gate releases; online A/B tests then confirm the effect on real behaviour. Report per-slice metrics with intervals for any slice you would be embarrassed to break (languages, tiers, safety categories)." },
    { heading: "Assumptions in this visualizer", body: "No real model is called. Each model's per-slice correct counts are fixed by the scenario, and a seeded random 'difficulty' decides which specific examples each model gets right (so A and B are correlated, like real models). Grading is exact-match against the label. The bootstrap, percentiles and re-weighting are real computations on those 80 simulated outcomes." },
  ],
  code: [
    {
      title: "paired_bootstrap.py — NumPy",
      source: `import numpy as np

def paired_bootstrap(a, b, n_resamples=1000, seed=0):
    """a, b: 0/1 arrays, one entry per example, same order."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_resamples, n))   # resample examples
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)  # keep pairs together
    return np.percentile(diffs, [2.5, 97.5])

for name, mask in slices.items():                     # per-slice report
    lo, hi = paired_bootstrap(a[mask], b[mask])
    flag = "REGRESSED" if hi < 0 else ""
    print(f"{name:16} Δ={b[mask].mean()-a[mask].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {flag}")`,
    },
    {
      title: "Re-weight slice metrics to production traffic (TypeScript)",
      source: `type SliceMetric = { slice: string; acc: number };

function weightedAccuracy(metrics: SliceMetric[], trafficShare: Record<string, number>) {
  const total = Object.values(trafficShare).reduce((s, v) => s + v, 0);
  return metrics.reduce((s, m) => s + (trafficShare[m.slice] ?? 0) / total * m.acc, 0);
}

// Gate: no slice may regress beyond its CI, AND production-weighted accuracy must not drop.`,
    },
  ],
  failureModes: [
    { title: "Aggregate improves, a slice collapses", symptom: "Headline accuracy is up, but one language or customer segment complains after launch.", detect: "Per-slice table with confidence intervals; flag any slice whose CI is entirely below 0.", fix: "Add slice-level release gates; ship only when no protected slice regresses." },
    { title: "Eval mix differs from production", symptom: "Offline win, online loss.", detect: "Compare the slice shares of the eval set against recent traffic; re-weight metrics to the production mix.", fix: "Stratify the eval set to production (or report both weightings)." },
    { title: "Reading noise as a win", symptom: "A +2 pt 'improvement' disappears on the next eval run.", detect: "Bootstrap CI for the difference includes 0.", fix: "Use paired comparisons, more examples, and require the CI to exclude 0 before claiming a gain." },
    { title: "Test set leaks into training or prompt tuning", symptom: "Offline scores keep rising while real quality is flat.", detect: "Hash and de-duplicate eval items against training/few-shot data; keep a sealed holdout.", fix: "Freeze and version the eval set; tune on a separate dev split.", scenarioId: "ml-leakage" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "In the default scenario, B beats A overall by +15 pts with a CI that excludes 0. What does the per-slice step reveal?", options: ["Every slice improves", "Non-English drops from 80% to 30% with a CI entirely below 0", "Code questions are the only regression", "The CI is too wide to say anything"], answer: 1, explanation: "The large Short and Long slices improve enough to lift the aggregate, while the 10-example Non-English slice falls by 50 pts. The aggregate's weights (eval-set shares) hide it." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "Keep the trap scenario and move 'Non-English share of production traffic' from 35% up to 60%. What happens to the production-weighted comparison?", options: ["B's advantage grows", "Nothing: weights don't affect accuracy", "B falls further behind A, because the slice where B collapsed now carries more weight", "Both accuracies become equal"], answer: 2, explanation: "Production-weighted accuracy is Σ w_s · acc_s. Raising the weight of the slice where B is 50 pts worse pulls B's weighted score down much faster than A's." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Why does the bootstrap resample A's and B's outcomes for the same example together (paired) instead of resampling each model independently?", options: ["It is faster", "Because both models tend to fail on the same hard examples; pairing removes that shared difficulty and gives a tighter, correct interval for the difference", "Independent resampling is biased toward A", "It guarantees the CI excludes 0"], answer: 1, explanation: "Outcomes on the same example are correlated. The variance of the difference is var(A) + var(B) − 2cov(A, B); pairing keeps the covariance term, independent resampling throws it away and overstates uncertainty." },
    { id: `${t}-4`, topicId: t, skill: "prediction", question: "Switch to the 'No real difference' scenario. What should the bootstrap step show?", options: ["A CI entirely above 0", "A CI that straddles 0", "A CI of exactly zero width", "No histogram"], answer: 1, explanation: "With B only one example better overall, the resampled differences spread on both sides of 0, so the 95% interval includes 0: no evidence of a real difference." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Which release rule would have blocked the trap scenario?", options: ["Ship if aggregate accuracy improves", "Ship if the aggregate CI excludes 0", "Block if any protected slice's CI for Δ lies entirely below 0", "Ship if the eval ran without errors"], answer: 2, explanation: "Both aggregate rules pass in the trap scenario. Only a slice-level gate (or production-weighted gate) catches the Non-English collapse." },
  ],
};
