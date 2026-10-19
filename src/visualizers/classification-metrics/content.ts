import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Drag a threshold over model scores and watch precision, recall, F1, ROC and PR respond.", minutes: 12, hasMetrics: false };

const t = "classification-metrics";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A classifier usually outputs a score. A threshold turns scores into yes/no decisions, which sorts every example into one of four outcomes: true positive, false positive, false negative, true negative. Precision (TP / predicted positives), recall (TP / actual positives), F1 (their harmonic mean), accuracy ((TP + TN) / n) and specificity (TN / actual negatives) are all ratios of those four counts." },
    { heading: "Threshold-free views: ROC and PR", body: "Sweeping the threshold from 1 to 0 traces a curve. ROC plots false-positive rate against true-positive rate; its area (AUC) is the probability that a random positive scores above a random negative. The precision-recall curve plots precision against recall; its area (average precision) starts from the prevalence for a random model, which makes it the more honest view when positives are rare." },
    { heading: "Why accuracy misleads under imbalance", body: "With 8% positives, a model that says 'negative' to everyone is 92% accurate and catches nothing. Always compare against that baseline, and report recall, precision or PR-AUC for the class you care about." },
    { heading: "Choosing a threshold", body: "The threshold is a product decision, not a model property. Pick it from the cost of a false alarm versus a miss, then report the metrics at that operating point. The black dot on the ROC and PR charts is that point." },
    { heading: "Assumptions in this visualizer", body: "Scores are synthetic: positives get sigmoid(+separation/2 + noise) and negatives sigmoid(−separation/2 + noise) from a seeded generator, rounded to 3 decimals. No real model is trained. Every metric, curve point, AUC (trapezoid rule) and average precision (sklearn's step-wise sum) is computed exactly from the dots on screen. Precision is shown as undefined (not 0) when nothing is predicted positive." },
  ],
  code: [
    {
      title: "metrics.py — scikit-learn",
      source: `from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, average_precision_score, roc_curve, precision_recall_curve)

threshold = 0.5
y_pred = (scores >= threshold).astype(int)
print("precision", precision_score(y_true, y_pred, zero_division=0))
print("recall   ", recall_score(y_true, y_pred))
print("F1       ", f1_score(y_true, y_pred))
print("accuracy ", accuracy_score(y_true, y_pred))
print("ROC AUC  ", roc_auc_score(y_true, scores))          # threshold-free
print("AP       ", average_precision_score(y_true, scores)) # area under PR, step-wise

fpr, tpr, thr = roc_curve(y_true, scores)
prec, rec, thr_pr = precision_recall_curve(y_true, scores)`,
    },
    {
      title: "Pick the threshold from costs",
      source: `import numpy as np
cost_fp, cost_fn = 1.0, 20.0           # a miss costs 20x a false alarm
grid = np.unique(scores)
costs = [cost_fp * ((scores >= t) & (y_true == 0)).sum() +
         cost_fn * ((scores <  t) & (y_true == 1)).sum() for t in grid]
best_t = grid[int(np.argmin(costs))]`,
    },
  ],
  failureModes: [
    { title: "Accuracy hides a useless model", symptom: "95% accuracy, but the positive class is almost never caught.", detect: "Compare with the always-negative baseline (last step); check recall and the PR curve.", fix: "Report per-class recall/precision or PR-AUC; reweight or resample during training; tune the threshold.", scenarioId: "ml-imbalance" },
    { title: "Default 0.5 threshold shipped blindly", symptom: "Recall far lower than expected in production even though AUC is good.", detect: "Look at where the operating point sits on the ROC/PR curves; AUC says nothing about it.", fix: "Choose the threshold from the cost of FP vs FN on a validation set and re-check it after retraining." },
    { title: "Reporting ROC AUC on rare events", symptom: "AUC 0.95 but most alerts are false alarms.", detect: "Average precision is close to the prevalence line; precision at your threshold is low.", fix: "Report precision at the operating point and PR-AUC alongside ROC AUC." },
    { title: "Precision of an empty prediction set", symptom: "Precision shows 0 (or NaN) and dashboards break when the threshold is high.", detect: "TP + FP = 0 at that threshold.", fix: "Treat it as undefined, set zero_division explicitly, and alert on 'no positives predicted'." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "Starting from the balanced preset, you raise the threshold from 0.5 to 0.8. What usually happens?", options: ["Precision and recall both rise", "Precision rises, recall falls", "Recall rises, precision falls", "Nothing: AUC is unchanged so metrics are unchanged"], answer: 1, explanation: "Fewer examples cross a higher threshold. Those that do are mostly real positives (precision up), but more positives are left below it (recall down). AUC stays the same because the scores did not change." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "Load the imbalanced preset and press 'always say negative'. Accuracy is about 92%. What does that tell you?", options: ["The model is excellent", "Accuracy here mostly measures the share of negatives; recall is 0", "Precision is 92%", "The ROC curve is broken"], answer: 1, explanation: "With 8% positives, predicting the majority class gives 92% accuracy for free. Recall is 0 and precision is undefined. Always compare against this baseline." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Which metric does NOT change when you drag the threshold?", options: ["F1", "Specificity", "ROC AUC", "Precision"], answer: 2, explanation: "AUC summarizes the whole sweep of thresholds, so it depends only on how scores rank positives against negatives. Only the black operating point moves." },
    { id: `${t}-4`, topicId: t, skill: "prediction", question: "Set class separation to 0. Where does the ROC curve go?", options: ["To the top-left corner", "Close to the diagonal, AUC ≈ 0.5", "Below the diagonal, AUC ≈ 0", "It disappears"], answer: 1, explanation: "With no separation, positives and negatives get scores from the same distribution, so ranking is random: TPR ≈ FPR at every threshold." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Your fraud model has ROC AUC 0.93 but positives are 2% of traffic. What should you check before celebrating?", options: ["Nothing; 0.93 is high", "Average precision vs the 0.02 random baseline, and precision at the chosen threshold", "The training loss", "Specificity only"], answer: 1, explanation: "ROC AUC can stay high while precision is poor when negatives vastly outnumber positives. The PR curve and precision at the operating point show how many alerts will be false." },
  ],
};
