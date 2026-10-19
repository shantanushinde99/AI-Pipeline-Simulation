import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Run a hyperparameter sweep, log every run, compare runs and catch the best-by-chance trap.", minutes: 12, hasMetrics: false };

const t = "experiment-tracking";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Experiment tracking records every training run's parameters (learning rate, depth, seed, data version), metrics (train and validation accuracy) and artifacts under a run id. The tracker then lets you sort runs, plot metrics against parameters, draw parallel-coordinates views, and diff two runs side by side." },
    { heading: "Why it matters", body: "A sweep produces dozens of runs. Without a tracker, 'which settings gave 0.91?' becomes guesswork, and the comparison you base a launch on may mix different data or code versions. Logged runs make results searchable, comparable and auditable." },
    { heading: "The best-by-chance trap", body: "Every validation score is a noisy estimate: with n examples its standard error is √(p(1−p)/n). Picking the maximum of many noisy scores favours runs that got lucky, so the winner's validation score is biased upward and it can lose to a runner-up on fresh data. Seed variance adds a second layer of noise: the same config retrained with a new seed lands somewhere else. Use a larger validation set, repeat seeds, and confirm on a held-out test set touched only once." },
    { heading: "Assumptions in this visualizer", body: "No model is trained. Each configuration has a hidden 'true' accuracy from a smooth made-up surface (peak near lr 0.006, depth 7). A run adds seeded training noise (sd 0.008) and then measures accuracy on n validation examples with binomial-sized noise, rounded to a whole number of correct answers. Test accuracy uses 10,000 examples. Train accuracy is a stand-in that grows with depth to mimic overfitting." },
  ],
  code: [
    {
      title: "Log a sweep with MLflow",
      source: `import mlflow, numpy as np

rng = np.random.default_rng(2)
for i in range(12):
    params = {"lr": float(10 ** rng.uniform(-4, -0.5)), "depth": int(rng.integers(2, 13)), "seed": 100 + i}
    with mlflow.start_run(run_name=f"run-{i+1}"):
        mlflow.log_params(params)
        model = train(**params)
        mlflow.log_metrics({"train_acc": acc(model, X_train, y_train), "val_acc": acc(model, X_val, y_val)})

runs = mlflow.search_runs(order_by=["metrics.val_acc DESC"])
print(runs[["run_id", "params.lr", "params.depth", "metrics.val_acc"]].head())`,
    },
    {
      title: "Is the winner's lead real? Compare against the standard error",
      source: `import math

def se(p, n):
    return math.sqrt(p * (1 - p) / n)

a, b, n = 0.915, 0.905, 200
diff = a - b
se_diff = math.sqrt(se(a, n) ** 2 + se(b, n) ** 2)
print(f"lead {diff:.3f}, {diff / se_diff:.2f} standard errors")  # < 2 → not distinguishable

# Confirm only the final choice on an untouched test set, once.`,
    },
  ],
  failureModes: [
    { title: "Winner's curse on a small validation set", symptom: "The chosen run scores several points lower in production than in the sweep.", detect: "Compare its validation and test scores; check whether the top runs' ±2 SE intervals overlap.", fix: "Use a larger validation set, cross-validation, or select among the top-k by repeated seeds before the final test." },
    { title: "Seed variance mistaken for improvement", symptom: "A 'better' config does not reproduce when rerun.", detect: "Seed-variance mode: retrain one config with several seeds and look at the spread.", fix: "Report mean ± sd over seeds; require a lead larger than the seed spread." },
    { title: "Untracked changes between runs", symptom: "Two runs differ in data or code, but only the hyperparameters were logged.", detect: "Run diff shows identical params yet different metrics.", fix: "Log data version, commit and seed with every run, not just hyperparameters." },
    { title: "Tuning on the test set", symptom: "Test accuracy creeps up as you iterate, then production is worse.", detect: "Count how many times the test set was used for a decision.", fix: "Keep the test set for one final check; tune on validation only.", scenarioId: "ml-leakage" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "With 200 validation examples and accuracy around 0.9, what is one standard error of the validation accuracy?", options: ["about 0.002", "about 0.021", "about 0.09", "about 0.2"], answer: 1, explanation: "√(0.9·0.1/200) = √0.00045 ≈ 0.021. Two runs 1 point apart are well within noise." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Switch the validation set from 200 to 10,000 examples. What happens to the gap between the winner's validation and test accuracy?", options: ["It grows", "It shrinks, because each validation score is a much less noisy estimate", "It stays exactly the same", "The winner always changes"], answer: 1, explanation: "Smaller measurement noise means less room for a run to win by luck, so the max is less optimistic." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "In the default sweep, the validation winner is beaten on the test set by another run. What is the most likely explanation?", options: ["The test set is broken", "The winner's validation score was inflated by noise, and picking the max selected that luck", "The runner-up was trained longer", "Learning rate does not matter"], answer: 1, explanation: "Selecting the maximum of noisy estimates is biased upward. The test set reveals the winner's true level." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "In seed-variance mode the same config spans roughly 4 points of validation accuracy. How should you judge a new config that beats the old one by 1 point on one seed?", options: ["Ship it: it is better", "Rerun both configs over several seeds and compare means relative to the spread", "Reduce the validation set", "Pick the seed with the best score"], answer: 1, explanation: "A 1-point lead is smaller than the seed-to-seed spread, so it says little. Compare mean ± sd across seeds." },
    { id: `${t}-5`, topicId: t, skill: "parameter", question: "In the compare view, two runs differ only in seed but their validation accuracy differs. What does that difference measure?", options: ["The effect of learning rate", "Pure noise: training randomness plus finite validation size", "Data leakage", "Overfitting from depth"], answer: 1, explanation: "With every other setting equal, only the seed (and validation sampling) differs, so the gap is noise." },
  ],
};
