import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Run a model through 12 production weeks: PSI and KS per batch, delayed labels, and retraining.", minutes: 14, hasMetrics: false };

const t = "model-drift";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Model drift is the slow or sudden loss of a deployed model's fit to the world. Covariate drift moves the inputs P(x); concept drift changes the rule P(y|x) that maps inputs to labels; real systems often see both, gradually. Monitoring compares each production batch to a fixed reference (usually the training data) and tracks outcome metrics once labels arrive." },
    { heading: "PSI and KS", body: "The Population Stability Index bins the feature and sums (a − e)·ln(a/e) over bins, where e is the reference share and a this batch's share. Rules of thumb: < 0.1 stable, 0.1–0.2 watch, > 0.2 act. The two-sample Kolmogorov–Smirnov statistic D is the largest vertical gap between the two empirical CDFs; at α = 0.05 it is significant above 1.36·√((n+m)/(n·m)). PSI depends on the binning; KS needs none but becomes 'significant' for tiny shifts when n is large." },
    { heading: "Why input monitors are not enough", body: "PSI and KS only see inputs. Under covariate drift they fire even when accuracy is fine (the rule still holds). Under concept drift they stay silent while accuracy collapses, because the inputs look the same. Only labelled outcomes reveal concept drift, and labels usually arrive late: chargebacks, churn, loan defaults take weeks or months." },
    { heading: "Responding: retrain on recent data", body: "When labels confirm a drop, retrain on recent labelled data and reset the reference to it. Retraining too early (on data from before the drift) teaches nothing new; retraining on every input alert wastes effort when the shift is harmless. In practice: shadow-evaluate the new model on a holdout before promoting it." },
    { heading: "Assumptions in this visualizer", body: "One synthetic feature x ~ N(μ, 1) with true P(y=1|x) = sigmoid(4(x − b)). Covariate drift moves μ by 0.6 × magnitude; concept drift moves b by −magnitude; gradual ramps both linearly from the start week to week 12. 400 training rows, 200 rows per week, all from a seeded generator. The model is real 1-feature logistic regression trained in your browser. PSI uses 11 fixed bins (floor 1e-4); KS is exact. Retraining uses only the most recent labelled week. Each week's 'true accuracy' is computed immediately but shown only after the label delay." },
  ],
  code: [
    {
      title: "PSI and KS per production batch (Python)",
      source: `import numpy as np
from scipy.stats import ks_2samp

EDGES = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])

def shares(x):
    counts = np.bincount(np.searchsorted(EDGES, x, side="right"), minlength=len(EDGES) + 1)
    return counts / len(x)

def psi(ref, cur, floor=1e-4):
    e, a = np.maximum(shares(ref), floor), np.maximum(shares(cur), floor)
    return float(np.sum((a - e) * np.log(a / e)))

for week, batch in enumerate(batches, 1):
    p = psi(reference, batch)
    ks = ks_2samp(reference, batch)          # .statistic = D, .pvalue
    if p > 0.2 or ks.pvalue < 0.05:
        alert(f"week {week}: input drift PSI={p:.3f} D={ks.statistic:.3f}")`,
    },
    {
      title: "Label-delayed accuracy check with retraining",
      source: `# labels for week w arrive at week w + LABEL_DELAY
for now in range(LABEL_DELAY, n_weeks):
    w = now - LABEL_DELAY
    acc = (model.predict(X[w]) == y[w]).mean()
    if acc < ACC_THRESHOLD:
        candidate = clone(model).fit(X[w], y[w])          # most recent labelled data
        if holdout_score(candidate) > holdout_score(model):
            model, reference = candidate, X[w]               # reset the drift reference too`,
    },
  ],
  failureModes: [
    { title: "Concept drift behind green input dashboards", symptom: "PSI and KS are flat, but complaints rise and, weeks later, labelled accuracy has dropped.", detect: "Track outcome metrics on labelled data and proxy signals (prediction distribution, override rates) while labels are pending.", fix: "Retrain on recent labelled data; shorten the label delay with faster feedback where possible.", scenarioId: "prod-everything-green" },
    { title: "Retraining on every input alert", symptom: "Frequent retrains, no accuracy gain, occasional regressions.", detect: "Input alerts with stable labelled accuracy (the covariate preset).", fix: "Treat input drift as a signal to investigate; retrain when outcome metrics confirm harm, and gate promotion on a holdout." },
    { title: "Comparing to last week instead of a fixed reference", symptom: "Gradual drift never triggers: each week differs only a little from the previous one.", detect: "Cumulative shift vs the training reference is large while week-over-week PSI is small.", fix: "Compare to a fixed reference (training data or a frozen baseline window) and reset it only on retrain." },
    { title: "KS alarms on huge batches", symptom: "KS is 'significant' every day on millions of rows for shifts nobody cares about.", detect: "D is tiny but above the critical value because n is enormous.", fix: "Alert on effect size (D or PSI) with a business threshold, not on the p-value alone." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "Choose 'Concept: silent failure'. Which monitor fires first?", options: ["PSI in the drift start week", "KS in the drift start week", "The accuracy alert, label-delay weeks after drift starts", "None ever fires"], answer: 2, explanation: "Concept drift changes P(y|x) but not P(x), so the histogram and CDFs match the reference and PSI/KS stay low. Accuracy reveals the drop once labels for the drift week arrive." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "With covariate drift (magnitude 1), what happens to accuracy after PSI goes over 0.2?", options: ["It collapses below 60%", "It barely changes, because the input→label rule still holds", "It becomes undefined", "It rises to 100%"], answer: 1, explanation: "The inputs moved but the true boundary did not, and the model's boundary is still correct. An input alert is a reason to investigate, not proof of harm." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "In the concept preset, you raise label delay from 2 to 4 weeks. What changes?", options: ["PSI values change", "The accuracy alert (and any retrain) arrives 2 weeks later", "True accuracy improves", "Nothing"], answer: 1, explanation: "The model's real accuracy is identical; you just learn about it later. Every week of delay is a week of bad predictions before you can react." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "A bin has reference share 10% and this week 20%. What is its PSI term?", options: ["0.010", "0.069", "0.100", "0.693"], answer: 1, explanation: "(0.20 − 0.10) · ln(0.20 / 0.10) = 0.10 · 0.693 ≈ 0.069. The PSI card computes exactly this for every bin and sums them." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Turn on 'Retrain on recent data' with concept drift. Why does accuracy recover?", options: ["The reference histogram was reset", "The new model learns the moved boundary from labelled post-drift data", "PSI went down", "The threshold was lowered"], answer: 1, explanation: "The retrained model's boundary moves toward the true one (see the solid line jump in the histogram). Resetting the reference only changes what future PSI compares against." },
  ],
};
