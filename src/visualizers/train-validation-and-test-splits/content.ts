import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Deal rows into train, validation and test, tune on validation, test once, and see what peeking costs.", minutes: 10, hasMetrics: false };

const t = "train-validation-and-test-splits";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Before training, labelled data is split three ways. The training set fits each candidate model. The validation set compares candidates (hyperparameters, features, architectures) on rows they never saw. The test set stays locked until one final model is chosen, then is used exactly once to estimate how it will do on new data." },
    { heading: "Why three sets, not two", body: "Every time you choose something based on a score, that score becomes optimistic: you picked the candidate that happened to get lucky on those rows. If you choose on the test set, the test score is no longer an honest estimate. Toggle 'tune on the test set' and compare the reported score with the same model on 600 fresh rows." },
    { heading: "k-fold cross-validation", body: "With little data, one fixed validation set is noisy. k-fold CV deals the non-test rows into K folds, trains K times leaving one fold out, and averages the K validation scores. Every row is used for validation once, so the estimate is steadier, at K times the training cost. The test set is still held out separately." },
    { heading: "Practical rules", body: "Shuffle before splitting unless rows are ordered in time (then split by time). Keep groups (same user, same patient) on one side of every split. Stratify by class for imbalanced labels. After selection, refit on train + validation and report the single test score." },
    { heading: "Assumptions in this visualizer", body: "The data is synthetic: two features drawn uniformly, a wavy true boundary, and a chosen fraction of labels flipped at random. Candidates are k-nearest-neighbour classifiers with different k, computed exactly in simulation.ts. 'Fresh data' is 600 extra rows from the same generator, standing in for production. All randomness is seeded." },
  ],
  code: [
    {
      title: "scikit-learn — holdout, then test once",
      source: `from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=0, stratify=y_dev)

best_k = max([1, 3, 5, 9, 15, 25, 41],
             key=lambda k: KNeighborsClassifier(k).fit(X_tr, y_tr).score(X_val, y_val))
# or: key=lambda k: cross_val_score(KNeighborsClassifier(k), X_dev, y_dev, cv=5).mean()

final = KNeighborsClassifier(best_k).fit(X_dev, y_dev)
print("test accuracy (report once):", final.score(X_test, y_test))`,
    },
  ],
  failureModes: [
    { title: "Tuning on the test set", symptom: "Great test score, disappointing production performance.", detect: "Test set used in more than one decision; compare against a fresh holdout or a later time window.", fix: "Select on validation or CV only; keep a sealed test set and evaluate once.", scenarioId: "ml-leakage" },
    { title: "Training accuracy used as the estimate", symptom: "k = 1 or a deep tree looks perfect.", detect: "Large gap between training and validation accuracy.", fix: "Always score on held-out rows; treat train accuracy only as an underfitting check." },
    { title: "Random split on time-ordered or grouped data", symptom: "Offline scores far above production.", detect: "Same user/session/day appears on both sides; future rows in training.", fix: "Group-aware or time-based splits (GroupKFold, TimeSeriesSplit)." },
    { title: "Test set too small", symptom: "Reported score swings by many points between seeds.", detect: "Re-split with different seeds; compute a confidence interval.", fix: "Larger test set, repeated CV for model selection, report intervals." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "Why does k = 1 always show 100% training accuracy here?", options: ["Each training row is its own nearest neighbour", "k = 1 is the best model", "The data has no noise", "The test set leaked"], answer: 0, explanation: "Graded on the rows it memorised, 1-NN can't be wrong. Validation accuracy reveals it is overfitting the label noise.", skill: "diagnosis" },
    { id: `${t}-2`, topicId: t, question: "Turn on 'tune on the test set' (seed 4). How does the reported score compare with fresh data?", options: ["Reported is much higher than fresh", "They are identical", "Reported is lower", "Fresh data can't be scored"], answer: 0, explanation: "Picking the candidate with the best test score selects for luck on those 18 rows: ~94% reported vs ~77% fresh.", skill: "prediction" },
    { id: `${t}-3`, topicId: t, question: "In 5-fold CV, how many times is each non-test row used for validation?", options: ["Exactly once", "Five times", "Never", "Four times"], answer: 0, explanation: "Folds rotate: each fold is the validation fold once and a training fold K − 1 times.", skill: "parameter" },
    { id: `${t}-4`, topicId: t, question: "You picked k on validation. What should you train the final model on before testing?", options: ["Train + validation rows", "Test rows", "Validation rows only", "All rows including test"], answer: 0, explanation: "Refitting on all non-test data uses more information; the test set must stay untouched.", skill: "fix" },
    { id: `${t}-5`, topicId: t, question: "Which check best tells you a reported test score was over-tuned?", options: ["Score the same model on a fresh holdout and compare", "Look at training accuracy", "Increase k", "Re-run on the same test set"], answer: 0, explanation: "A fresh holdout that was never used for any decision gives an unbiased estimate to compare against.", skill: "verification" },
  ],
};
