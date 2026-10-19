import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch a tree sweep every threshold, pick the purest split and grow until it overfits.", minutes: 10, hasMetrics: false };

const t = "decision-trees";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A decision tree classifies by asking a chain of yes/no questions about one feature at a time (\"is x₁ ≤ 4.5?\"). Each question cuts the current region of feature space in two with an axis-aligned line, so the finished tree is a set of rectangles, each predicting the majority class of the training points inside it." },
    { heading: "How a split is chosen", body: "CART is greedy. For the node being split it tries every candidate threshold on every feature (midpoints between neighbouring values), computes the weighted Gini impurity of the two children, and keeps the lowest. Gini = 1 − Σ p_c² is 0 for a pure node and 0.5 for a 50/50 two-class node. The tree then repeats the search inside each child." },
    { heading: "Why trees overfit, and the knobs that stop it", body: "Left alone, a tree keeps splitting until every leaf is pure, which means drawing tiny rectangles around single noisy points. Training accuracy hits 100% while test accuracy drops. Max depth, min samples to split, min samples per leaf and cost-complexity pruning all limit that. Ensembles (random forests, gradient boosting) average many trees to cut the variance further." },
    { heading: "Diagonal boundaries are hard", body: "Because every split is axis-aligned, a boundary like x₂ > x₁ needs a staircase of many splits. Try the Diagonal dataset: the tree needs depth to approximate the slope, and each extra step is another chance to fit noise." },
    { heading: "Assumptions in this visualizer", body: "The data are synthetic: 40 training and 40 test points drawn uniformly in [0, 10]², labelled by a known rule, then a chosen share of labels is flipped at random (seeded). The tree is a real CART implementation with Gini impurity, grown breadth-first so you can watch it level by level; it only splits when the split lowers impurity, and leaves need at least one point. No pruning, no class weights." },
  ],
  code: [
    {
      title: "scikit-learn",
      source: `from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=4)
for depth in [1, 3, 8]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, min_samples_split=2, random_state=0)
    tree.fit(X_tr, y_tr)
    print(depth, tree.score(X_tr, y_tr), tree.score(X_te, y_te))

print(export_text(tree, feature_names=["x1", "x2"]))`,
    },
    {
      title: "best_split.py: the greedy search by hand",
      source: `import numpy as np

def gini(y):
    if len(y) == 0: return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - (p ** 2).sum()

def best_split(X, y):
    best = (None, None, gini(y))            # (feature, threshold, weighted gini)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        for t in (vals[:-1] + vals[1:]) / 2:  # midpoints
            left = X[:, f] <= t
            g = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if g < best[2]:
                best = (f, t, g)
    return best`,
    },
  ],
  failureModes: [
    { title: "Overfitting with a deep tree", symptom: "Training accuracy is 100% but test accuracy is much lower; the plane is full of thin slivers around single points.", detect: "Compare train and test accuracy across depths (the chart on the last step); the gap widens with depth.", fix: "Limit max depth, raise min samples to split/leaf, prune, or use a random forest / boosting." },
    { title: "Label noise becomes structure", symptom: "Small leaves that predict the 'wrong' class in the middle of a region.", detect: "Leaves with very few samples; turn label noise up and watch new tiny rectangles appear.", fix: "Require a minimum leaf size and clean labels where possible." },
    { title: "Unstable trees", symptom: "Changing the seed (a slightly different sample) produces a completely different tree.", detect: "Refit on bootstrap samples and compare the root splits.", fix: "Use bagging / random forests, or treat single-tree explanations with caution." },
    { title: "Leakage through an ID-like feature", symptom: "A tree splits on a feature that perfectly separates training rows (an ID, a timestamp) and fails in production.", detect: "Feature importance is dominated by a column that should be meaningless.", fix: "Drop identifiers and post-outcome fields before training.", scenarioId: "ml-leakage" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "A node holds 30 points of class A and 10 of class B. What is its Gini impurity?", options: ["0.25", "0.375", "0.5", "0.75"], answer: 1, explanation: "1 − (30/40)² − (10/40)² = 1 − 0.5625 − 0.0625 = 0.375." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Use the 'Break it' preset (depth 8, 20% noise). What happens to training and test accuracy compared with depth 3?", options: ["Both go up", "Training rises to about 100%, test accuracy falls or stalls", "Both go down", "Nothing changes; depth only affects speed"], answer: 1, explanation: "Extra depth lets the tree carve rectangles around individual flipped labels. That memorises the training set but those rules do not hold for new points." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "Keeping depth 8 and 20% noise, which single change most directly shrinks the train–test gap?", options: ["Increase the seed", "Raise 'min samples to split' to around 12", "Lower the number of test points", "Switch to the blocks dataset colours"], answer: 1, explanation: "Nodes with fewer than 12 points can no longer split, so the tree cannot isolate single noisy points. Try the 'Regularised' preset." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "On the Diagonal dataset the tree needs many splits even with no noise. Why?", options: ["Gini cannot handle diagonal data", "Every split is axis-aligned, so a slanted boundary must be approximated by a staircase", "The test set is too small", "The thresholds are rounded"], answer: 1, explanation: "Each split tests one feature against a threshold. A line like x₂ = x₁ can only be approximated by many small horizontal and vertical steps." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "During a split step, the x₁ curve's lowest point is 0.21 and the x₂ curve's lowest is 0.18. Which split is chosen?", options: ["The x₁ split, because x₁ is searched first", "The x₂ split, because 0.18 < 0.21", "Neither: both must be below 0.1", "The split with more candidates"], answer: 1, explanation: "The greedy rule keeps the candidate with the lowest weighted child Gini across all features; ties go to x₁ only when the values are equal." },
  ],
};
