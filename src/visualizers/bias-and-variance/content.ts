import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Fit one model class to many resampled datasets and split its test error into bias², variance and noise.", minutes: 12, hasMetrics: false };

const t = "bias-and-variance";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Imagine rerunning your whole training process on many fresh datasets drawn from the same source. Each run gives a slightly different model. Bias is how far the average of those models is from the truth. Variance is how much the individual models scatter around their average. For squared error, expected test error = bias² + variance + σ², where σ² is the noise in the labels that no model can predict." },
    { heading: "Why complexity trades one for the other", body: "A rigid model (a straight line) gives nearly the same answer on every sample, so its variance is low, but it cannot bend like sin(3x), so its bias is high. A very flexible model (degree 10) can match the truth on average, so its bias is low, but it also chases each sample's noise, so its variance is high. The total is U-shaped in complexity, and the sweet spot moves with the data size and noise level." },
    { heading: "What actually lowers each term", body: "Bias: a more flexible model, better features, less regularisation. Variance: more training data, a simpler model, regularisation (ridge, dropout, early stopping), or averaging many models (bagging, random forests). Noise: only better measurements or labels. Try 'More data tames variance': degree 10 with 40 points behaves far better than with 12." },
    { heading: "How to see it in practice", body: "You only ever get one training set, so you can't compute bias and variance directly. Learning curves (training vs. validation error as data grows) and cross-validation across model sizes reveal the same story: a large train/validation gap signals variance, and both errors high and close together signals bias." },
    { heading: "Assumptions in this visualizer", body: "The truth is known (y = sin(3x)) and labels get Gaussian noise with the σ you choose, which is what makes the decomposition computable. Each training set draws one x uniformly inside each of n equal bins of [−1, 1] (stratified, so no set has a big gap to extrapolate across). Models are least-squares polynomials (the same solver as the Overfitting visualizer, no regularisation). Bias² and variance are averaged over 81 evenly spaced x values and estimated from a finite number of training sets, so they wobble slightly with the seed. The 'measured' check compares every fitted curve with fresh noisy labels on the same grid." },
  ],
  code: [
    {
      title: "bias_variance.py: the experiment by hand",
      source: `import numpy as np

rng = np.random.default_rng(1)
f = lambda x: np.sin(3 * x)
grid = np.linspace(-1, 1, 81)
sigma, n, M, degree = 0.3, 15, 20, 3

preds = []
for _ in range(M):
    x = -1 + 2 * (np.arange(n) + rng.random(n)) / n      # stratified draws
    y = f(x) + sigma * rng.standard_normal(n)
    w = np.polyfit(x, y, degree)
    preds.append(np.polyval(w, grid))
preds = np.array(preds)                                   # (M, 81)

avg = preds.mean(0)
bias2 = ((avg - f(grid)) ** 2).mean()
variance = preds.var(0).mean()
print(bias2, variance, sigma**2, bias2 + variance + sigma**2)`,
    },
    {
      title: "In practice: learning curves with scikit-learn",
      source: `from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = make_pipeline(PolynomialFeatures(10), LinearRegression())
sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="neg_mean_squared_error", train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0])
# big gap between train and validation error → variance; both high → bias`,
    },
  ],
  failureModes: [
    { title: "High variance mistaken for a good model", symptom: "Training error is tiny, but retraining on a new data pull gives a very different model and worse results.", detect: "Large train/validation gap; predictions change a lot across CV folds or retrains.", fix: "More data, a simpler model, regularisation, or bagging." },
    { title: "High bias blamed on the data", symptom: "Training and validation errors are both high and close together; more data does not help.", detect: "Learning curve flattens early at a high error.", fix: "Add capacity or better features (the model cannot represent the pattern)." },
    { title: "Chasing the noise floor", symptom: "Endless tuning cannot push error below a stubborn level.", detect: "Estimate label noise (e.g. disagreement between repeated labels); compare it with your error.", fix: "Stop tuning; improve the labels or measurement instead." },
    { title: "Leakage hides variance", symptom: "Cross-validation looks stable and accurate, production is noisy and worse.", detect: "Features that encode the target or rows duplicated across folds.", fix: "Remove leaking features and group-split duplicated entities.", scenarioId: "ml-leakage" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "With degree 10 and 12 points per set, which single change most reduces variance?", options: ["Increase noise σ", "Increase points per set to 40", "Increase the number of training sets", "Change the seed"], answer: 1, explanation: "Variance comes from fitting each sample's noise. More points per set pin the curve down. More training sets only make the estimate of variance steadier; they don't change the variance itself." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Set the degree to 1. Which term dominates the expected error?", options: ["Variance", "Bias²", "Noise σ²", "They are all equal"], answer: 1, explanation: "A straight line cannot follow sin(3x). Every training set gives nearly the same wrong line, so the average is far from the truth: high bias², low variance." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "bias² = 0.004, variance = 0.012, σ = 0.3. What is the expected squared test error?", options: ["0.016", "0.106", "0.316", "0.09"], answer: 1, explanation: "Add the noise variance: 0.004 + 0.012 + 0.3² = 0.004 + 0.012 + 0.09 = 0.106." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "Average training MSE is 0.02 but expected test error is 0.4. Which picture did the 'many curves' step most likely show?", options: ["Curves bunched tightly but far from the truth", "Curves fanning out widely around the truth", "A single straight line", "Curves identical to the truth"], answer: 1, explanation: "Low training error with high test error is variance: each flexible curve fits its own sample closely, and the curves disagree with each other." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How does the visualizer check that bias² + variance + σ² is right?", options: ["It trusts the formula", "It compares the sum with the error of every fitted curve against fresh noisy labels", "It uses the training error", "It averages the coefficients"], answer: 1, explanation: "The 'measured' value draws new noisy labels on the grid and averages each model's squared error. It matches the sum up to sampling noise." },
  ],
};
