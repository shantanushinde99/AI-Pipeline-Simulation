import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch inputs slide (covariate shift) or labels flip (concept drift) and see which detector notices.", minutes: 14, hasMetrics: false };

const t = "data-drift-and-concept-drift";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A model learns P(y|x) from data drawn from some P(x). In production either can change. Covariate shift (data drift) means P(x) moves: new kinds of users, a new region, a changed upstream form. The rule linking inputs to answers stays the same. Concept drift means P(y|x) changes: the same inputs now deserve different answers (fraudsters adapt, customer preferences change, a policy changes)." },
    { heading: "Why the difference matters", body: "Covariate shift is visible in the inputs alone, so you can catch it immediately with distribution tests such as PSI (Population Stability Index) or the Kolmogorov–Smirnov statistic, with no labels needed. It only hurts when the model has to extrapolate into regions it never saw. Concept drift leaves the inputs unchanged, so input tests are blind to it; only labels (ground truth) reveal it, and labels often arrive days or weeks late." },
    { heading: "The detectors", body: "PSI bins the reference data into quantile bins and compares the share of production rows per bin: PSI = Σ (a − e)·ln(a / e). A common rule of thumb is < 0.1 stable, 0.1–0.2 watch, > 0.2 act. KS takes the largest gap between the two empirical CDFs and compares it with a critical value for the sample sizes. A label-based monitor computes accuracy (or calibration, or business KPIs) on batches whose labels have arrived." },
    { heading: "What to do about it", body: "Treat an input-drift alert as an early warning and check whether the shifted region is one the model handles well. Treat a label-based alert as confirmed harm: retrain on recent labelled data, weight recent data more, or roll back. Shorten label delay where you can (proxy labels, human review samples)." },
    { heading: "Assumptions in this visualizer", body: "Data is synthetic and 2-D, generated with a seeded random generator: 240 reference points and 8 production batches of 120 points (the same 'users' with small per-batch jitter, so points visibly slide). The true rule is a slightly curved boundary; the model is a real logistic regression trained on the reference data (from the shared ml-model scenario code). Covariate shift moves the input mean linearly up to (2.0, 1.4); concept drift rotates the true boundary up to 55°. PSI uses 5 quantile bins (10 is more common; 5 keeps sampling noise low at these sample sizes). Every number shown is computed on the points shown." },
  ],
  code: [
    {
      title: "drift_checks.py — PSI, KS and a delayed accuracy monitor",
      source: `import numpy as np
from scipy.stats import ks_2samp

def psi(reference, current, bins=10):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])
    e = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=bins) / len(reference)
    a = np.bincount(np.searchsorted(edges, current, side="right"), minlength=bins) / len(current)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

for feature in ["x1", "x2"]:
    score = psi(ref_df[feature], prod_df[feature])
    stat, p = ks_2samp(ref_df[feature], prod_df[feature])
    if score > 0.2 or p < 0.05:
        alert(f"input drift on {feature}: PSI={score:.2f}, KS={stat:.2f}")

# Concept drift needs labels: join predictions with ground truth once it lands.
labelled = preds.merge(labels, on="request_id")          # only batches old enough
acc = (labelled.pred == labelled.label).groupby(labelled.batch).mean()
if (acc < 0.85).any():
    alert("accuracy dropped: possible concept drift")`,
    },
  ],
  failureModes: [
    { title: "Concept drift with a green drift dashboard", symptom: "Business KPIs fall while every input-drift check reports 'stable'.", detect: "Accuracy or calibration on newly labelled data drops; PSI/KS stay low because P(x) did not move.", fix: "Monitor label-based metrics, shorten label delay with proxy labels or reviewed samples, retrain on recent data." },
    { title: "Alarm fatigue from input drift", symptom: "PSI alerts fire every week but accuracy is fine.", detect: "Drift happens in regions where the model still generalizes; label-based metrics stay flat.", fix: "Use input drift as a warning that triggers a label check, not as a page; weight features by importance." },
    { title: "Label delay hides the damage", symptom: "The accuracy alert finally fires weeks after the model went bad.", detect: "Compare when the true accuracy dropped with when the monitor knew (here: the label-delay gap).", fix: "Collect faster proxy labels, sample for human review, alert on prediction-distribution shifts as an early signal." },
    { title: "Reference window chosen badly", symptom: "Drift tests fire constantly (or never) because the reference set is stale or tiny.", detect: "PSI between two reference slices is already high; bins hold very few rows.", fix: "Use a large, recent, representative reference window and enough rows per bin." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Run 'Concept drift' to the end. What does the PSI detector report?", options: ["PSI rises steadily above 0.2", "PSI stays near its noise level because P(x) does not change", "PSI becomes negative", "PSI fires once labels arrive"], answer: 1, explanation: "Concept drift rotates the true boundary while the points stay put. The histograms match the reference, so PSI only shows sampling noise. Only a label-based monitor can see it." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "In the concept-drift scenario, accuracy first drops below the threshold at batch 5. With a label delay of 2 batches, when does the label monitor fire?", options: ["Batch 3", "Batch 5", "Batch 7", "Never"], answer: 2, explanation: "Labels for batch 5 arrive two batches later, so the monitor learns about it at batch 7. Set the delay to 0 and it fires at batch 5." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "Under covariate shift, PSI fires at batch 2 while accuracy is still above 90%. Why is the model still fine at that point?", options: ["PSI is wrong", "The inputs moved only slightly, into a region where the linear model still matches the curved true boundary", "Labels flipped back", "The model retrained itself"], answer: 1, explanation: "Covariate shift only hurts where the model must extrapolate. Early batches stay near the training region; later batches reach the part of the curve the straight line cannot follow." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "Reference bin shares are 0.2 each. A batch has shares [0.05, 0.10, 0.15, 0.30, 0.40]. Roughly what is the PSI?", options: ["About 0.01", "About 0.08", "About 0.40", "About 2.0"], answer: 2, explanation: "Σ(a−e)ln(a/e) = (−0.15)ln(0.25) + (−0.1)ln(0.5) + (−0.05)ln(0.75) + 0.1·ln(1.5) + 0.2·ln(2) ≈ 0.208 + 0.069 + 0.014 + 0.041 + 0.139 ≈ 0.47, well above the 0.2 'act' level." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Your fraud model faces concept drift and labels (chargebacks) take 6 weeks. What helps most?", options: ["Lower the PSI threshold", "Add faster proxy labels or reviewed samples and monitor accuracy on them", "Add more input features to the drift check", "Increase the batch size"], answer: 1, explanation: "Input tests cannot see concept drift. The bottleneck is label delay, so reduce it with proxy or human-reviewed labels and alert on those." },
  ],
};
