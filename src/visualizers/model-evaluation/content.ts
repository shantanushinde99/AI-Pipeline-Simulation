import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Fit polynomials of growing complexity and watch training and test error diverge.", minutes: 15, hasMetrics: true };

const t = "overfitting";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Overfitting means a model has learned the noise in its training data rather than the underlying pattern. It scores well on data it has seen and poorly on data it hasn't. Underfitting is the opposite: the model is too simple to capture the pattern at all, so both errors stay high." },
    { heading: "Why hold out data", body: "Training error is optimistically biased: the parameters were chosen to minimize it. A test split that never touches fitting gives an unbiased estimate of error on new data. In practice you tune on a validation split and report the test split once." },
    { heading: "Bias and variance", body: "Low-degree polynomials have high bias (systematically wrong shape). High-degree polynomials have high variance (the curve changes wildly with a different sample). Try changing the seed with degree 14: the curve swings, while degree 3 barely moves." },
    { heading: "Regularization", body: "Ridge regression adds λ·Σw² to the loss, shrinking coefficients. It trades a little training error for much lower variance. Set degree 14 and raise λ: the curve calms down and test error drops." },
    { heading: "Assumptions in this visualizer", body: "Data is 1-D and synthetic (y = sin(3x) + Gaussian noise), so the true function and the noise floor are known, which is never the case in production. The underfit/good/overfit label is a heuristic relative to the best test error in the degree sweep." },
  ],
  code: [
    {
      title: "scikit-learn",
      source: `from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=11)

for degree in range(1, 16):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model.fit(X_tr, y_tr)
    print(degree,
          mean_squared_error(y_tr, model.predict(X_tr)),
          mean_squared_error(y_te, model.predict(X_te)))`,
    },
  ],
  failureModes: [
    { title: "Reporting training accuracy as model quality", symptom: "Great offline numbers, disappointing production results.", detect: "Compare training and held-out error; a large gap is the warning sign.", fix: "Always evaluate on data that played no part in fitting or tuning." },
    { title: "Data leakage", symptom: "Held-out score is suspiciously high.", detect: "A feature or preprocessing step used information from the label or the test set.", fix: "Fit preprocessing on the training split only; remove features unavailable at prediction time.", scenarioId: "ml-leakage" },
    { title: "Tuning on the test set", symptom: "Test error creeps down across many experiments, production doesn't improve.", detect: "Count how many times the test set influenced a decision.", fix: "Use a separate validation split or cross-validation for tuning." },
    { title: "Tiny test set", symptom: "Test error jumps around between seeds.", detect: "Change the seed with a 10% split and few points.", fix: "Use more held-out data or k-fold cross-validation and report variance." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "You raise degree from 4 to 14 with λ off. Training MSE drops by 60%. What most likely happens to test MSE?", options: ["It drops by a similar amount", "It rises: the extra flexibility fits noise", "It stays exactly the same", "It becomes zero"], answer: 1, explanation: "Extra parameters let the curve pass near individual noisy training points. That lowers training error but makes predictions between points worse." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "Train MSE = 0.31, test MSE = 0.33, noise σ² = 0.09. What is the best description?", options: ["Overfitting", "Underfitting", "Good fit", "Data leakage"], answer: 1, explanation: "Both errors are similar (small gap) but far above the noise floor, so the model is too simple to capture the signal." },
    { id: `${t}-3`, topicId: t, skill: "fix", question: "A degree-14 model overfits and you cannot collect more data. Which change most directly addresses it?", options: ["Increase the test split to 50%", "Add ridge regularization or reduce the degree", "Train for more epochs", "Report R² instead of MSE"], answer: 1, explanation: "Regularization and lower complexity both reduce variance. Changing the split or the metric only changes how you measure it." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "How should you verify that a chosen degree generalizes?", options: ["Check training MSE is below 0.01", "Evaluate once on a held-out test set after tuning on validation data", "Retrain with the test set included and re-measure", "Compare R² on the training set across degrees"], answer: 1, explanation: "Tuning must not see the final test set; otherwise the reported error is biased low." },
  ],
};
