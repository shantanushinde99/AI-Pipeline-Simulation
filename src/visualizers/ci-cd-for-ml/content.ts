import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Push a change through tests, data checks, training, gated evaluation and a canary before it ships.", minutes: 15, hasMetrics: false };

const t = "ci-cd-for-ml";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "CI/CD for ML is an automated pipeline that every change (code, data or config) must pass before a model reaches users: unit tests, data validation (schema and drift), training, evaluation against the current production baseline with explicit gates, a canary deployment on a small share of live traffic, and finally promotion. Each gate compares a measured value with a threshold and stops the pipeline when it is crossed." },
    { heading: "Why ML needs more than software CI", body: "A model can break without any code change: an upstream table renames a column, a feature's distribution drifts, or a retrain improves the average while hurting a subgroup. Unit tests run on fixtures and cannot see these. Data checks, slice-level evaluation gates and a canary each catch a different class of failure, at increasing cost and closer to users." },
    { heading: "Choosing gates and thresholds", body: "Gate on what you care about: no regression beyond a tolerance versus the baseline (not an absolute target), latency budgets, and the worst slice rather than only the average. Thresholds trade false alarms for missed regressions; a canary needs enough requests before its error rate means anything, and a smaller canary exposes fewer users but takes longer to judge." },
    { heading: "Assumptions in this visualizer", body: "Nothing is trained or deployed. Feature histograms, slice counts and the loss curve are fixed numbers chosen to illustrate each preset; PSI, accuracies, p95 (from 200 seeded log-normal latencies) and canary error rates are computed from them. Canary traffic is 1,000 simulated requests per minute with seeded Bernoulli errors (0.5% normally, 4% for the fp16 preset); the verdict needs at least 200 candidate requests unless the 10-minute window ends." },
  ],
  code: [
    {
      title: "Evaluation gates in a CI job (Python)",
      source: `import json, sys

cand, base = json.load(open("eval/candidate.json")), json.load(open("eval/baseline.json"))
gates = {
    "accuracy":   cand["accuracy"] - base["accuracy"] >= -0.005,
    "p95_ms":     cand["p95_ms"] <= 120,
    "worst_slice": max(base["slices"][s] - cand["slices"][s] for s in base["slices"]) <= 0.03,
}
for name, ok in gates.items():
    print(f"{name:12s} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if all(gates.values()) else 1)   # non-zero exit blocks the pipeline`,
    },
    {
      title: "GitHub Actions pipeline skeleton",
      source: `jobs:
  ml-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/unit
      - run: python validate_data.py --schema schema.yaml --max-psi 0.2
      - run: python train.py --out artifacts/candidate
      - run: python evaluate.py --candidate artifacts/candidate --baseline models:/churn@production
      - run: python gates.py
      - run: ./deploy.sh canary --traffic 10 --watch 10m --max-error-rate 0.02
      - run: ./deploy.sh promote`,
    },
  ],
  failureModes: [
    { title: "Upstream schema change", symptom: "Predictions shift overnight with no code change; a feature is all nulls.", detect: "Schema validation on incoming data (expected columns and types) before training and at serving.", fix: "Fail the pipeline, fix the mapping, and version the data contract with the upstream team." },
    { title: "Average up, slice down", symptom: "Overall accuracy improves but complaints rise from one customer group.", detect: "Per-slice evaluation with a 'worst slice drop' gate. Try the slice preset with the gate off.", fix: "Keep slice gates on; investigate the training change (here a loss reweighting) and rebalance." },
    { title: "Serving-path failure invisible offline", symptom: "Offline metrics pass, production errors spike after deploy.", detect: "Canary on a small traffic share with an error-rate gate and automatic rollback.", fix: "Roll back, reproduce the serving path (here fp16 inference) in pre-production tests." },
    { title: "Gates that always pass", symptom: "Pipeline is green for months while quality erodes.", detect: "Track how often each gate fails; zero failures can mean thresholds are too loose.", fix: "Set thresholds relative to the baseline and review them when metrics or traffic change.", scenarioId: "prod-everything-green" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "With the 'Upstream schema change' preset, which stage stops the pipeline, and why didn't unit tests catch it?", options: ["Unit tests; they check every column", "Data validation; unit tests run on fixed fixtures, not the new upstream data", "Evaluation; accuracy drops", "Canary; errors spike"], answer: 1, explanation: "Unit tests pass 48/48. The schema check compares the expected columns with what arrived and finds income missing." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "In the slice preset, overall accuracy goes from 0.871 to 0.884. Why is the candidate still blocked?", options: ["Latency is too high", "The 65+ slice drops from 0.850 to 0.760, a 9-point drop above the 3-point limit", "PSI is too high", "The canary failed"], answer: 1, explanation: "The worst-slice gate measures the largest per-slice drop: (255 − 228) / 300 = 0.09 > 0.03." },
    { id: `${t}-3`, topicId: t, skill: "fix", question: "You disable the fairness slice gate with the slice preset. What happens?", options: ["The pipeline still blocks at evaluation", "The candidate is promoted to 100% with the 65+ regression", "The canary rolls it back", "Data validation fails"], answer: 1, explanation: "Every enabled gate passes and the canary's error rate is normal, because a quality drop on one slice is not an error. It ships." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "In the canary preset, lower the canary share from 10% to 2%. What changes?", options: ["Nothing", "Fewer users see errors per minute, but it takes longer to collect the 200 requests needed for a verdict", "The canary passes", "Data validation fails"], answer: 1, explanation: "At 2% only 20 requests per minute go to the candidate, so the verdict waits for enough evidence. Exposure per minute is lower, detection is slower." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "The accuracy gate requires 'change vs baseline ≥ −0.5 pts' instead of 'accuracy ≥ 0.85'. Why compare with the baseline?", options: ["It is easier to compute", "It catches regressions relative to what is serving now, regardless of the absolute level", "Absolute thresholds are illegal", "Baselines never change"], answer: 1, explanation: "A relative gate blocks anything worse than production by more than the tolerance. An absolute bar can pass a model that is much worse than the current one." },
  ],
};
