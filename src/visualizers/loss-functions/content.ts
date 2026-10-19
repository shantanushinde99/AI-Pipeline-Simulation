import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Price the same misses with MSE, MAE, Huber and cross-entropy, and watch one outlier drag only some fits.", minutes: 10, hasMetrics: false };

const t = "loss-functions";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A loss function turns each prediction error into a cost; training finds the parameters with the lowest total cost. For regression the error is the residual r = y − ŷ. MSE charges r², MAE charges |r|, and Huber charges ½r² for small misses and a linear δ(|r| − ½δ) for big ones. Different price lists lead to different 'best' models." },
    { heading: "Why the shape matters", body: "The gradient dℓ/dr is how hard a point pulls on the model. Under MSE the pull is 2r, so a point twice as far pulls twice as hard and an outlier can dominate. Under MAE every point pulls with strength 1 (the fit behaves like a median). Huber caps the pull at δ: smooth near zero like MSE, robust far away like MAE." },
    { heading: "Classification: cross-entropy", body: "For a probability output, binary cross-entropy costs −log q, where q is the probability given to the true class. A confident mistake (q → 0) costs enormously, and through a sigmoid its gradient with respect to the logit is 1 − q, which stays large. MSE on probabilities is capped at 1 and its gradient 2q(1 − q)² vanishes for confident mistakes, so training barely corrects them." },
    { heading: "Choosing one", body: "MSE: Gaussian-ish noise, you care about large errors, and you want the mean. MAE: heavy-tailed noise or outliers, and you want the median. Huber: the best of both with a tunable δ (common for regression with occasional bad labels). Cross-entropy: any classifier that outputs probabilities." },
    { heading: "Assumptions in this visualizer", body: "Regression data is synthetic: 8 points from y = 1 + 0.5x with seeded noise, plus an optional hand-placed outlier at x = 9. The candidate line is the least-squares fit to the clean points. MSE fits use the closed form; MAE and Huber fits use iteratively reweighted least squares, which converges to their minimizers (checked in tests). Classification predictions are hand-set probabilities, not a trained model." },
  ],
  code: [
    {
      title: "losses.py — NumPy",
      source: `import numpy as np

def mse(r):   return r ** 2
def mae(r):   return np.abs(r)
def huber(r, delta=1.0):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))

def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

r = np.array([0.2, -0.4, -6.0])           # the last one is an outlier
print(mse(r).mean(), mae(r).mean(), huber(r).mean())`,
    },
    {
      title: "PyTorch / scikit-learn",
      source: `import torch.nn as nn
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

loss_fn = nn.HuberLoss(delta=1.0)          # or nn.MSELoss(), nn.L1Loss()
clf_loss = nn.BCEWithLogitsLoss()          # sigmoid + cross-entropy, numerically stable

ols = LinearRegression().fit(X, y)                        # MSE
robust = HuberRegressor(epsilon=1.35).fit(X, y)           # Huber
median = QuantileRegressor(quantile=0.5, alpha=0).fit(X, y)  # MAE`,
    },
  ],
  failureModes: [
    { title: "Outliers hijack an MSE model", symptom: "The fitted line tilts toward a few extreme points; typical-case error gets worse.", detect: "Per-point loss share: one point owns most of the total MSE (bars step).", fix: "Clean or cap bad labels, or switch to Huber/MAE." },
    { title: "MSE on probabilities", symptom: "Classifier learns slowly and stays confidently wrong on some examples.", detect: "Gradient for saturated wrong predictions is near zero (2q(1 − q)²).", fix: "Use cross-entropy (BCEWithLogitsLoss) on logits." },
    { title: "log(0) in cross-entropy", symptom: "Loss is inf or NaN as soon as a prediction hits exactly 0 or 1.", detect: "Loss spikes to inf; probabilities of exactly 0.0/1.0 appear.", fix: "Compute from logits (log-sum-exp) or clip probabilities to [ε, 1 − ε]." },
    { title: "Optimizing the wrong loss for the metric", symptom: "Validation loss improves but the business metric (e.g. median error) does not.", detect: "Compare the metric you report against the loss you train.", fix: "Train on (or tune toward) a loss aligned with the metric: MAE for median error, quantile loss for percentiles." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Turn the outlier on. Which fitted line moves the most?", options: ["MAE", "Huber", "MSE", "They all move equally"], answer: 2, explanation: "MSE's pull grows with the residual (2r), so the outlier dominates. MAE's pull is capped at 1 and Huber's at δ." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "A residual of −4 with δ = 1: what are the MSE, MAE and Huber losses?", options: ["16, 4, 3.5", "16, 4, 8", "8, 4, 3.5", "16, −4, 3.5"], answer: 0, explanation: "r² = 16, |r| = 4, Huber = δ(|r| − δ/2) = 1·(4 − 0.5) = 3.5." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "What happens to the Huber fit as you increase δ toward a value larger than every residual?", options: ["It approaches the MAE fit", "It approaches the MSE fit", "It becomes a flat line", "Nothing changes"], answer: 1, explanation: "If every |r| ≤ δ, Huber is ½r² for every point, which has the same minimizer as MSE." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "Example G has true class 1 and p = 0.02. Why does a classifier trained with MSE + sigmoid fix it slowly?", options: ["MSE is undefined at 0.02", "Its gradient w.r.t. the logit, 2q(1 − q)² ≈ 0.04, is tiny, while cross-entropy's is 1 − q = 0.98", "MSE ignores class 1", "The learning rate is too high"], answer: 1, explanation: "The sigmoid is saturated, and MSE's gradient includes the sigmoid's slope q(1 − q), which kills it. Cross-entropy cancels that factor." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Your house-price data has a few mislabeled sales at 10× the real price. Which loss is the safest default?", options: ["MSE", "Huber (or MAE)", "Cross-entropy", "No loss; use accuracy"], answer: 1, explanation: "Robust losses cap each point's influence, so a few wild labels can't drag the model. Cross-entropy is for classification." },
  ],
};
