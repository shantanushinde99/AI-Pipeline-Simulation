import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Transform a house-price table column by column and watch held-out R² respond to each step.", minutes: 12, hasMetrics: false };

const t = "feature-engineering";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Feature engineering turns raw columns into numbers a model can use well: parsing dates into ages or durations, encoding categories, taking logs of skewed quantities, rescaling, and building interaction terms. A linear model can only add up weight × column, so giving it the right columns is often worth more than switching models." },
    { heading: "The transforms in this table", body: "Parse built_date → age, because a string carries no numeric meaning. log(sqft), because price grows with each doubling of size, not each extra square foot. One-hot neighbourhood (dropping Riverside as the baseline), so each area gets its own premium. Standardise, so ridge's penalty treats every coefficient on the same scale. Interaction size × Downtown, so size can be worth more in one neighbourhood than another." },
    { heading: "Fit transforms on training rows only", body: "Anything learned from data (a scaler's mean and σ, an imputation value, a category list, a target encoding) is part of the model. It must be fit on the training rows and then applied, unchanged, to validation, test and production rows. Fitting it on all rows lets held-out information shape the features: data leakage. With a mean/std scaler the effect is usually small, which is exactly why it slips through; with target encoding or feature selection on all rows it can inflate scores dramatically. Use a Pipeline so the fit happens inside each split." },
    { heading: "Measuring a transform's contribution", body: "The change in held-out R² when a transform is added depends on what is already in the model and on the particular held-out rows. Turn one transform off and compare the final score to measure its marginal value. With only 12 held-out rows, small differences (±0.02) are noise; use cross-validation before trusting them." },
    { heading: "Assumptions in this visualizer", body: "The 40 houses are synthetic and seeded. Their true price is 160 + 150·ln(sqft/1000) + area premium − 2.5·age + 110·ln(sqft/1000)·Downtown + Gaussian noise (σ = 25, in thousands of dollars), so every transform has a real signal to recover; real data is rarely this tidy. 28 rows train the model and 12 are held out. The model is ridge regression solved exactly with the normal equations; λ is scaled by the number of training rows and the intercept is not penalised. Ages are measured at 2025-01-01." },
  ],
  code: [
    {
      title: "scikit-learn Pipeline: transforms are fit inside the split",
      source: `import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

df["age"] = (pd.Timestamp("2025-01-01") - pd.to_datetime(df["built_date"])).dt.days / 365.25
df["log_sqft"] = np.log(df["sqft"])
df["size_x_dt"] = df["log_sqft"] * (df["neighbourhood"] == "Downtown")

pre = ColumnTransformer([
    ("num", StandardScaler(), ["age", "log_sqft", "size_x_dt"]),
    ("cat", OneHotEncoder(drop=["Riverside"]), ["neighbourhood"]),
])
model = make_pipeline(pre, Ridge(alpha=1.0))
model.fit(X_train, y_train)            # scaler sees training rows only
print(model.score(X_test, y_test))     # held-out R²`,
    },
    {
      title: "The leak, and the fix",
      source: `# WRONG: statistics include the test rows
scaler = StandardScaler().fit(X)              # all rows
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# RIGHT: fit on train, apply to test
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)`,
    },
  ],
  failureModes: [
    { title: "Preprocessing fit on all rows", symptom: "Offline scores are a little (or a lot) better than production.", detect: "Search the code for fit/fit_transform called before the train/test split, or outside the CV loop.", fix: "Put every learned transform in a Pipeline and fit it per split.", scenarioId: "ml-leakage" },
    { title: "Training/serving skew", symptom: "The same house gets a different prediction online than in the notebook.", detect: "Log the transformed feature vector at serving time and diff it against the training pipeline's output.", fix: "Ship the fitted transformer with the model; never re-implement transforms by hand in the service." },
    { title: "Column order or category drift", symptom: "A new neighbourhood appears, or one-hot columns arrive in a different order; predictions go haywire.", detect: "Validate the schema and category set at the serving boundary.", fix: "Pin the encoder's category list, use handle_unknown='ignore', and check column names, not positions.", scenarioId: "ml-feature-order" },
    { title: "Unparsed or mis-parsed dates", symptom: "Age is negative or NaN for some rows (wrong format, time zone, two-digit years).", detect: "Range-check engineered columns (0 ≤ age ≤ 200) and count NaNs.", fix: "Parse with an explicit format and fail loudly on unparseable values." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "In the raw step, why does the model ignore built_date and neighbourhood?", options: ["They are not correlated with price", "They are strings/categories, and a linear model can only multiply numbers by weights", "Ridge removed them", "They contain missing values"], answer: 1, explanation: "A linear model computes w₀ + Σ wⱼ·xⱼ. A date string or a neighbourhood name has no numeric value until it is parsed or encoded." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "Set λ = 0 (plain least squares) and step to 'Standardise'. What happens to held-out R²?", options: ["It jumps up", "It stays the same: rescaling a column just rescales its weight", "It drops to 0", "The model cannot be fit"], answer: 1, explanation: "Least squares is invariant to rescaling and shifting a column: the fitted weight compensates exactly. Scaling matters for penalised models (ridge, lasso), distance-based models and gradient descent." },
    { id: `${t}-3`, topicId: t, skill: "fix", question: "Which change removes the leak shown by the 'Break it' preset?", options: ["Use a larger λ", "Compute the scaler's μ and σ from training rows only and apply them to held-out rows", "Standardise the target as well", "Drop the held-out rows"], answer: 1, explanation: "Anything learned from data belongs to the model and must be fit on training rows only. A Pipeline enforces that automatically." },
    { id: `${t}-4`, topicId: t, skill: "prediction", question: "Turn off one-hot encoding. What happens to the interaction step?", options: ["It still adds size × Downtown", "It is skipped, because the interaction needs the is_Downtown column", "It replaces one-hot encoding", "R² becomes negative"], answer: 1, explanation: "The interaction multiplies size by is_Downtown. Without the one-hot columns that indicator does not exist, so the step is skipped and its bar stays flat." },
    { id: `${t}-5`, topicId: t, skill: "metric", question: "With 12 held-out rows, a transform changes R² from 0.882 to 0.889. What is the right conclusion?", options: ["The transform clearly helps", "The change is within noise for 12 rows; confirm with cross-validation or more data", "The transform hurts", "R² cannot change by less than 0.01"], answer: 1, explanation: "Small held-out sets give noisy scores. Try a few seeds: tiny differences flip sign, while real gains (like one-hot encoding here) show up every time." },
  ],
};
