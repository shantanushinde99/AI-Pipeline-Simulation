import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Train a logistic classifier epoch by epoch and watch the boundary turn, the shading sharpen and loss fall.", minutes: 12, hasMetrics: false };

const t = "logistic-regression";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Logistic regression is a linear classifier. It computes a score z = w·x + b and squashes it with the sigmoid σ(z) = 1/(1 + e^−z) into a probability P(class 1 | x). The decision boundary, where P = 0.5, is the straight line w·x + b = 0; the weight vector w points toward class 1 and its length sets how sharp the transition is." },
    { heading: "How it learns", body: "Training minimizes the mean log loss (binary cross-entropy). Its gradient is beautifully simple: ∇w = mean((p − y)·x). Each point pulls the weights by its error p − y times its features, so confidently wrong points pull hardest. Gradient descent repeats w ← w − η·∇L, which rotates and slides the boundary." },
    { heading: "Learning rate and regularization", body: "Too small a learning rate crawls; too large overshoots, so the loss bounces up and down (try η = 15 on overlapping blobs). L2 regularization adds (λ/2)‖w‖², which keeps weights small: the boundary stays softer and probabilities less extreme. On perfectly separable data, without regularization the weights grow forever, pushing probabilities to 0 and 1." },
    { heading: "Limits", body: "A single straight boundary cannot separate XOR-like data: the loss stalls near ln 2 ≈ 0.693, the score of a coin flip. Fix it with engineered features (for example x₁·x₂) or a non-linear model such as a neural network with a hidden layer." },
    { heading: "Assumptions in this visualizer", body: "Points are synthetic (seeded Gaussian blobs, clamped to the plotted box). Training is full-batch gradient descent from fixed starting weights (w = (−0.6, 0.9), b = 0.5) so the boundary visibly moves. Every probability, loss, gradient and accuracy shown is computed exactly on the plotted points; there is no test set, so accuracy here is training accuracy." },
  ],
  code: [
    {
      title: "logreg.py — NumPy gradient descent",
      source: `import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.5, lam=0.0, epochs=60):
    w, b = np.array([-0.6, 0.9]), 0.5
    for _ in range(epochs):
        p = sigmoid(X @ w + b)             # P(class 1) per point
        grad_w = X.T @ (p - y) / len(y) + lam * w
        grad_b = (p - y).mean()
        w, b = w - lr * grad_w, b - lr * grad_b
    return w, b`,
    },
    {
      title: "scikit-learn",
      source: `from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0)   # C = 1/λ: smaller C = stronger L2 regularization
clf.fit(X, y)
print(clf.coef_, clf.intercept_)
print(clf.predict_proba(X[:3])[:, 1])  # P(class 1)`,
    },
  ],
  failureModes: [
    { title: "Learning rate too high", symptom: "Loss jumps up and down between epochs instead of falling smoothly.", detect: "The loss curve zig-zags; the step headline shows ↑ on some epochs.", fix: "Lower η, standardize features, or use a line search / adaptive optimizer." },
    { title: "Non-linear classes", symptom: "Loss plateaus near 0.693 and accuracy near 50% no matter how long you train.", detect: "Try the XOR preset: the boundary wanders but never splits the corners.", fix: "Add interaction or polynomial features, or use a non-linear model (trees, neural networks)." },
    { title: "Overconfident weights on separable data", symptom: "Weights keep growing; probabilities become 0.000 or 1.000.", detect: "‖w‖ in the inspector rises every epoch while loss creeps toward 0.", fix: "Add L2 regularization (λ > 0) or early stopping; calibrate probabilities on held-out data." },
    { title: "Class imbalance hides a useless model", symptom: "High accuracy, but the minority class is never predicted.", detect: "Check per-class recall and the confusion matrix, not accuracy alone.", fix: "Re-weight classes, resample, and tune the threshold with precision/recall.", scenarioId: "ml-imbalance" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Which way does the weight vector w point relative to the decision boundary?", options: ["Along the boundary", "Perpendicular to it, toward the class-1 side", "Perpendicular to it, toward the class-0 side", "Toward the origin"], answer: 1, explanation: "The boundary is w·x + b = 0, so w is its normal. Moving along +w increases z and therefore P(class 1). The arrow on the plane shows this." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "With the XOR preset, the loss flattens at about 0.693. What does that number mean?", options: ["The model is 69% accurate", "It equals ln 2: the model predicts ~0.5 for every point, like a coin flip", "The learning rate is too high", "Regularization is too strong"], answer: 1, explanation: "Log loss of predicting 0.5 is −ln 0.5 = ln 2 ≈ 0.693. A linear boundary can't beat chance on XOR, so the best it finds is near-uniform uncertainty." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "Set η = 15 on overlapping blobs. What do you see in the loss curve?", options: ["A smooth, faster decrease", "It bounces up and down because each step overshoots", "It is flat at zero", "It is identical to η = 0.5"], answer: 1, explanation: "A step much larger than the curvature allows jumps past the minimum. The loss rises on several epochs, and the result step reports unstable training." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "A point has label 1 and predicted probability 0.2. What is its contribution to ∂L/∂w₁ (before averaging), if x₁ = 2?", options: ["−1.6", "0.4", "1.6", "−0.4"], answer: 0, explanation: "(p − y)·x₁ = (0.2 − 1)·2 = −1.6. A negative gradient means increasing w₁ lowers the loss, so the update w₁ ← w₁ − η·(−1.6) raises it." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "On separable blobs, ‖w‖ keeps growing every epoch. What keeps the weights bounded?", options: ["A larger learning rate", "More epochs", "L2 regularization (λ > 0)", "Removing the bias"], answer: 2, explanation: "Without a penalty, scaling w up always lowers the loss on separable data. The λw term pulls back, so training settles at a finite ‖w‖ with softer probabilities." },
  ],
};
