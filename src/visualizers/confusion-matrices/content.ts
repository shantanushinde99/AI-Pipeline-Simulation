import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch examples fly into a 2×2 or 3×3 grid, then read precision, recall and cost off it.", minutes: 10, hasMetrics: false };

const t = "confusion-matrices";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A confusion matrix tallies (actual, predicted) pairs. Rows are the true class, columns the predicted class, so cell (i, j) counts examples of class i that the model called j. The diagonal holds correct predictions; every off-diagonal cell is a specific kind of mistake. It works the same for 2 classes (TP, FN, FP, TN) and for many." },
    { heading: "Reading metrics off it", body: "Accuracy is the diagonal divided by the total. Precision for a class is its diagonal cell divided by its column total (of everything predicted as that class, how much was right). Recall is the diagonal cell divided by its row total (of everything that truly is that class, how much was found)." },
    { heading: "Normalization and cost", body: "Row-normalizing divides each row by its total, so every row sums to 1 and the diagonal shows per-class recall. That makes rare classes visible next to common ones. A cost matrix assigns a price to each kind of mistake; total cost = Σ count × cost. When missing a critical case is expensive, the model with the lowest cost may not be the one with the highest accuracy." },
    { heading: "Assumptions in this visualizer", body: "The examples are synthetic patient-message triage labels from a seeded generator. Each prediction is correct with probability 'model quality'; wrong predictions pick another class (in 3-class mode neighbouring classes are confused twice as often). No real model is run. Class counts are exact shares of n. Every matrix cell, ratio and cost shown is computed from the examples on screen." },
  ],
  code: [
    {
      title: "confusion.py — scikit-learn",
      source: `from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

labels = ["urgent", "routine", "self-care"]
M = confusion_matrix(y_true, y_pred, labels=labels)          # rows = actual, cols = predicted
M_row = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")  # rows sum to 1
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

C = np.array([[0, 10, 10],   # missing an urgent message costs 10
              [1,  0,  1],
              [1,  1,  0]])
total_cost = (M * C).sum()`,
    },
    {
      title: "Per-class precision/recall by hand",
      source: `precision = np.diag(M) / M.sum(axis=0)   # column totals
recall    = np.diag(M) / M.sum(axis=1)   # row totals
accuracy  = np.trace(M) / M.sum()`,
    },
  ],
  failureModes: [
    { title: "Transposed matrix", symptom: "Precision and recall look swapped compared with another report.", detect: "Check the axis convention: sklearn uses rows = actual, columns = predicted; some tools use the opposite.", fix: "Label both axes explicitly in every chart and log the convention with the numbers." },
    { title: "Majority-class model looks accurate", symptom: "High accuracy, but one column holds every prediction.", detect: "A single non-zero column; recall 0 for the other classes. Toggle 'always predicts the majority class' to see it.", fix: "Track per-class recall and macro F1; rebalance or reweight training.", scenarioId: "ml-imbalance" },
    { title: "Raw counts hide a failing rare class", symptom: "The matrix 'looks diagonal' yet urgent cases are missed.", detect: "Row-normalize: the rare class's diagonal share drops well below the others.", fix: "Report row-normalized matrices and per-class metrics alongside counts." },
    { title: "Optimizing accuracy instead of cost", symptom: "A model with better accuracy causes more expensive incidents.", detect: "Compute total cost with a cost matrix agreed with stakeholders.", fix: "Select models and thresholds on expected cost; weight the loss by class cost." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "In a 3-class matrix (rows = actual), how do you compute recall for 'urgent'?", options: ["Urgent diagonal cell ÷ urgent column total", "Urgent diagonal cell ÷ urgent row total", "Sum of the diagonal ÷ total", "Urgent row total ÷ total"], answer: 1, explanation: "Recall asks how much of the true class was found, so it divides by everything that truly is urgent: the row total." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "Turn on imbalanced classes and 'always predicts the majority class'. What does the matrix show?", options: ["A perfect diagonal", "All predictions in one column; urgent recall is 0 while accuracy stays high", "Empty cells everywhere", "Precision for urgent is 1"], answer: 1, explanation: "Every example lands in the majority column. Accuracy equals the majority share (about 85%), yet no urgent message is found and precision for urgent is undefined." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "After row-normalizing, what does each diagonal cell equal?", options: ["Precision of that class", "Recall of that class", "Accuracy", "The class prior"], answer: 1, explanation: "Dividing by the row total turns the diagonal cell into M_ii / Σ_j M_ij, which is exactly that class's recall." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "With miss cost 10 and other-error cost 1, a model has 2 missed urgent and 6 other errors. What is the total cost?", options: ["8", "26", "62", "20"], answer: 1, explanation: "2 × 10 + 6 × 1 = 26. The two missed urgent messages cost more than all six other errors combined." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Your matrix cells add up to 994 but you evaluated 1,000 examples. What should you suspect?", options: ["Nothing; rounding", "Some examples were dropped, e.g. labels outside the label list or failed predictions", "The model is overfitting", "Precision is wrong but recall is fine"], answer: 1, explanation: "Every example must land in exactly one cell. A shortfall means rows were filtered out, often unseen labels or null predictions, which can bias every metric." },
  ],
};
