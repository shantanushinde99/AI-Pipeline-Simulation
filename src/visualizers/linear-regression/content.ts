import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Fit a least-squares line step by step and see residuals become squares whose area is SSE.", minutes: 10, hasMetrics: false };

const t = "linear-regression";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Linear regression predicts a number with a straight line, ŷ = a + b·x. Ordinary least squares (OLS) picks the slope b and intercept a that make the sum of squared residuals (SSE) as small as possible. For one input the answer has a closed form: b = Sxy / Sxx and a = ȳ − b·x̄, so the line always passes through the point of means (x̄, ȳ)." },
    { heading: "Why squares?", body: "Squaring makes every miss positive and punishes big misses much more than small ones. Geometrically, each residual becomes a square with area e²; OLS minimizes the total area. That is also why a single outlier, whose square is huge, can drag the whole line toward it." },
    { heading: "Reading R²", body: "R² = 1 − SSE/SST compares your line with the laziest model: predicting ȳ for everyone (SST is that model's squared error). R² = 0.9 means the line removes 90% of that error. It does not say the relationship is causal, linear, or that the model will generalize." },
    { heading: "When it matters", body: "Linear regression is the baseline for every regression task, the building block of many models (a neuron without its activation is a linear model), and its residual plots are the first diagnostic for any regressor." },
    { heading: "Assumptions in this visualizer", body: "The data is synthetic: points are drawn from the hidden line y = 2 + 0.7x plus seeded Gaussian noise and rounded to one decimal. Every number shown (means, Sxy, Sxx, slope, intercept, residuals, SSE, R²) is computed exactly from those points. The squares use the y-axis scale for both sides, so their on-screen area is proportional to e²." },
  ],
  code: [
    {
      title: "ols.py — NumPy, from the formulas",
      source: `import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 5.0])

xm, ym = x.mean(), y.mean()
b = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()  # Sxy / Sxx
a = ym - b * xm
residuals = y - (a + b * x)
sse = (residuals ** 2).sum()
r2 = 1 - sse / ((y - ym) ** 2).sum()
print(b, a, sse, r2)  # 1.5 0.333 0.1667 0.964`,
    },
    {
      title: "Same fit with scikit-learn",
      source: `from sklearn.linear_model import LinearRegression, HuberRegressor

model = LinearRegression().fit(x.reshape(-1, 1), y)
print(model.coef_[0], model.intercept_, model.score(x.reshape(-1, 1), y))  # slope, intercept, R²

# Outliers? A robust loss down-weights them instead of squaring them:
robust = HuberRegressor().fit(x.reshape(-1, 1), y)`,
    },
  ],
  failureModes: [
    { title: "One outlier drags the line", symptom: "The slope changes a lot when a single point is added or removed.", detect: "Turn on the outlier: its square dominates SSE (see the stacked SSE bar) and the slope drops.", fix: "Investigate the point (data error?), use a robust loss (Huber/MAE) or robust regression, or report the fit with and without it." },
    { title: "Mistaking R² for correctness", symptom: "High R² on training data, poor predictions in production.", detect: "Evaluate on held-out data; plot residuals against x and look for curves or funnels.", fix: "Validate out of sample; add features or transform x when residuals show structure." },
    { title: "No spread in x", symptom: "Slope is undefined or wildly unstable.", detect: "Sxx is zero or tiny compared with the noise.", fix: "Collect data over a wider x range; regularize when features are nearly constant or collinear." },
    { title: "Extrapolating far from the data", symptom: "Predictions at x outside the observed range are nonsense (e.g. negative prices).", detect: "Check whether the query x lies inside the training x range; the intercept is itself an extrapolation if x = 0 is far away.", fix: "Refuse or flag out-of-range inputs; model the domain with a more appropriate function." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Before computing the slope, which point is the OLS line guaranteed to pass through?", options: ["The origin (0, 0)", "The first data point", "The point of means (x̄, ȳ)", "The point with the largest residual"], answer: 2, explanation: "Because a = ȳ − b·x̄, plugging x = x̄ into ŷ = a + b·x gives ȳ. Watch the slope step: the line rotates around the crosshair." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "Turn on the outlier at (8.5, 0.5). What happens to the slope and why?", options: ["It rises, because more points mean a steeper line", "It drops, because the outlier's huge square dominates SSE and the line tilts toward it", "It is unchanged, because OLS ignores outliers", "It becomes undefined"], answer: 1, explanation: "Squared error grows quadratically with the miss, so the line gives up a little on every other point to shrink the one enormous square. The stacked SSE bar shows the outlier's share." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "SST = 40 and SSE = 10. What is R²?", options: ["0.25", "0.75", "4", "30"], answer: 1, explanation: "R² = 1 − SSE/SST = 1 − 10/40 = 0.75: the line removes 75% of the squared error of always predicting the mean." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "In 'Your turn', you set the OLS slope but an intercept 1 unit too high. Compared with the minimum, SSE…", options: ["stays the same", "increases by exactly n (one per point)", "decreases", "doubles regardless of the data"], answer: 1, explanation: "Every residual shifts by −1. Σ(e−1)² = Σe² − 2Σe + n, and Σe = 0 at the OLS fit, so SSE grows by exactly n." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How can you check from the numbers alone that a line is the OLS fit?", options: ["The residuals sum to zero and nudging slope or intercept only increases SSE", "Every residual is zero", "R² equals 1", "The intercept is positive"], answer: 0, explanation: "At the least-squares minimum the gradient of SSE is zero, which implies Σe = 0 and Σx·e = 0. The residual step shows Σe ≈ 0, and the manual panel shows any change raises SSE." },
  ],
};
