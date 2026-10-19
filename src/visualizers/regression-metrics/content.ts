import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Build MAE, MSE, RMSE, R² and MAPE from residual bars, then break them with outliers and zeros.", minutes: 12, hasMetrics: false };

const t = "regression-metrics";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Regression metrics summarize the residuals e = y − ŷ. MAE averages |e| (robust, in the target's units). MSE averages e² (what most models minimize; squared units; dominated by big misses). RMSE = √MSE (back in the target's units, still outlier-sensitive). R² = 1 − SS_res/SS_tot compares the model with always predicting the mean (1 is perfect, 0 is no better than the mean, negative is worse). MAPE averages |e|/|y| as a percentage (unit-free, but undefined at y = 0)." },
    { heading: "Which metric reacts to what", body: "One wild prediction multiplies MSE and RMSE much more than MAE, because squaring amplifies it; R² can collapse. Changing units (k$ → $) multiplies MAE and RMSE by the factor and MSE by its square, while R² and MAPE do not change. Targets near zero make MAPE explode even when absolute errors are tiny." },
    { heading: "How to report", body: "Pair a robust metric with a sensitive one (MAE with RMSE): a large RMSE/MAE ratio flags a few big misses. Always state units. Avoid MAPE when targets can be zero or tiny; use WAPE (Σ|e| / Σ|y|) or MAE instead." },
    { heading: "Assumptions in this visualizer", body: "Data is synthetic and seeded: house prices (k$) or hourly energy use (kWh, with many readings near zero). Predictions are the true value plus Gaussian noise scaled by 'typical model error'; the outlier toggle replaces example #7's prediction with a large overestimate. No model is trained. Every metric and running total is computed exactly from the points shown." },
  ],
  code: [
    {
      title: "metrics.py — scikit-learn",
      source: `import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error)

mae  = mean_absolute_error(y, y_hat)
mse  = mean_squared_error(y, y_hat)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_hat)
mape = mean_absolute_percentage_error(y, y_hat) * 100   # explodes when y ~ 0
wape = np.abs(y - y_hat).sum() / np.abs(y).sum() * 100  # safer alternative
print(f"RMSE/MAE = {rmse / mae:.2f}  (>> 1 means a few large misses)")`,
    },
  ],
  failureModes: [
    { title: "One outlier dominates MSE/RMSE", symptom: "RMSE jumps after a deploy while MAE barely moves.", detect: "RMSE/MAE ratio rises; the MSE running total is mostly one example.", fix: "Inspect the largest residuals, fix data errors, report MAE alongside RMSE, or use a robust loss (Huber)." },
    { title: "MAPE explodes on tiny targets", symptom: "MAPE in the hundreds of percent although absolute errors are small.", detect: "Sort per-example |e|/|y|: the worst ones have y near 0.", fix: "Use WAPE, MAE or sMAPE, or exclude/floor near-zero targets explicitly and say so." },
    { title: "Comparing errors across units", symptom: "A model looks 1000× worse after a data pipeline change.", detect: "MAE/RMSE scale by exactly the unit factor while R² is unchanged.", fix: "Pin target units in the feature/label schema and print units next to every metric." },
    { title: "Trusting R² alone", symptom: "R² = 0.97 but errors on small values are unacceptable.", detect: "Check MAE per segment or MAPE; R² is dominated by large-variance targets.", fix: "Report errors in units for the segments that matter." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Turn on the outlier. Which metric grows by the largest factor?", options: ["MAE", "MSE", "MAPE", "R²"], answer: 1, explanation: "Squaring a residual that is ~10× the others makes it ~100× bigger, so MSE jumps the most. MAE grows only by the outlier's absolute size." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "You switch the target from k$ to $. What happens?", options: ["All metrics ×1000", "MAE and RMSE ×1000, MSE ×1,000,000, R² and MAPE unchanged", "Only MSE changes", "R² drops"], answer: 1, explanation: "MAE and RMSE are in target units, MSE in squared units. R² is a ratio of squared errors and MAPE a ratio of errors to targets, so both are unit-free." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "On the energy dataset MAE is 0.14 kWh but MAPE is about 60%. Why?", options: ["The model is terrible", "Several targets are near zero, so tiny misses become huge percentages", "MAPE is computed wrong", "Units are mixed"], answer: 1, explanation: "A 0.14 kWh miss on a 0.03 kWh reading is hundreds of percent. MAPE is driven by the smallest targets." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "What does R² < 0 mean?", options: ["The data is negative", "The model's squared error is larger than that of always predicting the mean", "The model is perfect", "Precision is low"], answer: 1, explanation: "R² = 1 − SS_res/SS_tot. If SS_res > SS_tot, the model is worse than a constant prediction of ȳ." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "RMSE is 3× MAE. What should you check?", options: ["Nothing", "The largest residuals: a few big misses (or bad labels) are inflating RMSE", "The units", "Whether R² is 1"], answer: 1, explanation: "With evenly sized errors RMSE ≈ MAE. A large ratio means the error is concentrated in a few examples worth inspecting." },
  ],
};
