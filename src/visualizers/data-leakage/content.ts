import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Toggle three classic leaks and watch the offline score inflate while production stays low.", minutes: 12, hasMetrics: false };

const t = "data-leakage";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Data leakage is any path by which information that will not be available at prediction time reaches model training or model selection. The model learns a shortcut, the held-out score looks great, and production performance is far worse. The test set only estimates production if nothing about it influenced training." },
    { heading: "Three classic leaks", body: "Target leakage: a feature that is a consequence of the label (a cancellation ticket logged after the customer cancels). Preprocessing leakage: a scaler, imputer, encoder or feature selector fitted on all rows before the split, so test statistics shape training. Duplicate leakage: the same entity appears on both sides of the split, so a model that memorizes looks like it generalizes." },
    { heading: "How to prevent it", body: "Split first (by entity or by time), then fit every transform inside a pipeline on training rows only. Audit each feature for when it is written relative to the label. Deduplicate before splitting, or group by entity. Keep a final check that cannot leak: score fresh, later data before trusting the offline number." },
    { heading: "Why the size of the leak differs", body: "A leaked label column is catastrophic: it can carry almost all of the signal. Duplicates hurt in proportion to how much the model memorizes (1-NN memorizes everything; logistic regression barely). A scaler fitted on 30% more rows changes μ and σ only slightly, so with this model the effect is tiny, but the same mistake with target encoding or feature selection is large." },
    { heading: "Assumptions in this visualizer", body: "Data is the synthetic churn generator shared across the site (seeded). The logistic model is trained by real gradient descent and 1-NN is exact, both in your browser. 'Production' is a fresh draw of customers from the same generator, where cancel_ticket_logged is always 0 because it does not exist yet at prediction time. 30% of rows are duplicated when that leak is on; 30% of rows are held out. The canvas shows a sample of rows; every number is computed on the full data." },
  ],
  code: [
    {
      title: "Leak-free scikit-learn pipeline",
      source: `from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = df.drop(columns=["cancel_ticket_logged"])      # written after the label: target leak
df = df.drop_duplicates()                            # or split by customer_id below

split = GroupShuffleSplit(test_size=0.3, random_state=0)
train_idx, test_idx = next(split.split(df, groups=df["customer_id"]))
train, test = df.iloc[train_idx], df.iloc[test_idx]

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(train[FEATURES], train["churned"])        # scaler is fit on train rows only
print("test accuracy", model.score(test[FEATURES], test["churned"]))`,
    },
    {
      title: "The leaky version (do not do this)",
      source: `X = StandardScaler().fit_transform(df[FEATURES])     # fit on ALL rows, before the split
X_train, X_test, y_train, y_test = train_test_split(X, df["churned"], test_size=0.3)
# + cancel_ticket_logged still in FEATURES, + duplicate rows on both sides`,
    },
  ],
  failureModes: [
    { title: "Label written into a feature", symptom: "Offline accuracy 95%+, production recall near zero.", detect: "One feature has a suspiciously high correlation with the label and a dominant weight; check when that column is written.", fix: "Drop features that are only known after the outcome; build features as of the prediction timestamp.", scenarioId: "ml-leakage" },
    { title: "Transform fitted before the split", symptom: "Small but systematic optimism; large with target encoding or feature selection.", detect: "Code calls fit or fit_transform on the full dataset before train_test_split.", fix: "Put every fitted step inside a Pipeline and fit it on training folds only." },
    { title: "Duplicates or the same entity across the split", symptom: "High-capacity models score far better offline than on new users.", detect: "Count test rows with an exact or same-entity twin in train; score twins and non-twins separately.", fix: "Deduplicate first and split by group (customer, patient, device) or by time." },
    { title: "Tuning on the test set", symptom: "The reported test score slowly creeps up across experiments but production does not.", detect: "The same test split has been used to pick hyperparameters or features many times.", fix: "Use a validation split or cross-validation for selection and keep a locked final test set." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Turn on 'keep cancel_ticket_logged'. What happens to production recall?", options: ["It rises with offline recall", "It collapses toward 0, because the column is always 0 for customers who have not cancelled yet", "It is unchanged", "It becomes undefined"], answer: 1, explanation: "The model learned 'cancel ticket = churn'. In production nobody has a cancel ticket at prediction time, so the model predicts almost nobody churns." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "With duplicates on, which model shows the bigger offline/production gap?", options: ["Logistic regression", "1-nearest-neighbour", "Both identical", "Neither shows a gap"], answer: 1, explanation: "1-NN returns the label of the closest training row; an exact twin is distance 0, so every twinned test row is scored correctly for free. Logistic regression cannot memorize individual rows." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "With only 'fit scaler before split' on, why does the score barely move here?", options: ["The leak is fake", "μ and σ computed on 30% more rows from the same distribution are almost identical, and they contain no label information", "The scaler is never applied", "Accuracy ignores scaling"], answer: 1, explanation: "The scale step's table shows the train-only and leaked statistics differ only in the 2nd-3rd decimal. The leak is still a bug: with transforms that use labels or future data it becomes large." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "What is the correct order of operations?", options: ["Scale, deduplicate, split, train", "Deduplicate, split by entity, fit scaler on train, train, evaluate on test", "Split, train, scale test only", "Train on everything, then split to evaluate"], answer: 1, explanation: "Remove duplicates and split first so nothing downstream sees test rows; every fitted transform learns from training rows only." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Which check would have caught all three leaks in this visualizer?", options: ["A higher learning rate", "Scoring a fresh batch of later customers and comparing it to the offline score", "More epochs", "Reporting F1 instead of accuracy"], answer: 1, explanation: "Fresh data cannot leak into training. A large gap between offline and fresh-data scores is the symptom shared by every leak." },
  ],
};
