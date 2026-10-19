import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Compute the same features in batch SQL and an online service, spot the diffs and their effect.", minutes: 12, hasMetrics: false };

const t = "feature-and-training-serving-skew";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Training-serving skew is any difference between the features a model was trained on and the features it receives in production. It usually happens because the same feature is written twice: once in batch SQL that builds the training set, and again in a low-latency online service. The model is unchanged, but it is fed different numbers." },
    { heading: "Why it is dangerous", body: "Nothing crashes. The service returns a valid number, the model returns a valid probability, and offline metrics still look great because they were computed on training-path features. The only symptom is that production quality is worse than the offline evaluation promised." },
    { heading: "Typical causes", body: "Unit mismatches (cents vs dollars), timezone handling (UTC in the warehouse, server-local time online), different null handling (training imputed the mean, serving fills 0), and stale joins (the online path reads an aggregate table that lags behind the events). Each is a one-line bug that passes code review." },
    { heading: "The fix", body: "Define each feature once and run that definition in both places (a feature store or a shared feature library), compute training features point-in-time correctly, and continuously compare logged serving features with a batch recompute for the same entities and timestamps." },
    { heading: "Assumptions in this visualizer", body: "Six customers, their orders and ratings are hand-made; both 'pipelines' are TypeScript functions that mimic the SQL and Python shown, and the 'model' is a hand-set logistic regression, not a trained one. No database or service is called. The four bugs are injected deliberately so each skews exactly one feature; tenure_days is a bug-free control." },
  ],
  code: [
    {
      title: "One feature definition, two runtimes (Python)",
      source: `from datetime import datetime, timedelta, timezone

def avg_order_value_usd(orders) -> float:
    return sum(o.amount_cents for o in orders) / len(orders) / 100 if orders else 0.0

def last_order_hour_utc(orders) -> int:
    return max(o.ts for o in orders).astimezone(timezone.utc).hour

def orders_7d(orders, as_of: datetime) -> int:
    return sum(1 for o in orders if as_of - timedelta(days=7) <= o.ts < as_of)

TRAIN_MEAN_RATING = 4.2  # stored with the model artifact, not re-derived
def avg_rating(ratings) -> float:
    return sum(ratings) / len(ratings) if ratings else TRAIN_MEAN_RATING

# The batch job and the online service both import this module.`,
    },
    {
      title: "Skew check: logged serving features vs batch recompute",
      source: `import pandas as pd

def skew_report(served: pd.DataFrame, recomputed: pd.DataFrame, tol=1e-6):
    # both: one row per (entity_id, as_of), one column per feature
    m = served.merge(recomputed, on=["entity_id", "as_of"], suffixes=("_serve", "_train"))
    out = {}
    for f in [c[:-6] for c in m.columns if c.endswith("_serve")]:
        diff = (m[f + "_serve"] - m[f + "_train"]).abs()
        out[f] = {"mismatch_rate": (diff > tol).mean(), "max_abs_diff": diff.max()}
    return pd.DataFrame(out).T.sort_values("mismatch_rate", ascending=False)`,
    },
  ],
  failureModes: [
    { title: "Unit mismatch", symptom: "Predictions saturate near 0 or 1 for almost every request after a deploy.", detect: "Compare the serving feature distribution with training: the mean of avg_order_value is ~100× larger.", fix: "Put units in feature names (amount_usd), share the definition, and add range checks on serving features.", scenarioId: "ml-feature-order" },
    { title: "Timezone offset", symptom: "Time-of-day features look shifted; the model under-performs for some regions or hours.", detect: "Recompute features for logged requests in batch and diff them; hour features differ by a constant.", fix: "Store and compute in UTC with timezone-aware datetimes; never rely on the server's local clock." },
    { title: "Different null handling", symptom: "New users get much worse predictions online than in offline evaluation.", detect: "Slice the skew report by 'has history'; serving shows 0 where training had the imputed mean.", fix: "Save imputation values with the model artifact and apply them through the shared definition." },
    { title: "Stale join", symptom: "Recency features lag reality; the model ignores customers' latest activity.", detect: "Track the freshness (last refresh time) of every online feature table and alert on lag.", fix: "Compute from fresh events or a streaming aggregate, and build training features point-in-time with the same freshness." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "Only the 'cents vs dollars' bug is on. How many customers have a mismatched avg_order_value?", options: ["All 6", "Only customers with no orders", "None, because the model rescales inputs", "Exactly 1"], answer: 0, explanation: "Every customer with orders gets a value 100× too large online, so all six differ. Turn off the other bugs and watch only that column turn red.", skill: "parameter" },
    { id: `${t}-2`, topicId: t, question: "The batch path says last_order_hour = 21 for a customer. With the timezone bug on (server clock UTC−5), what does serving send?", options: ["16", "2", "21", "0"], answer: 0, explanation: "The naive conversion applies the server's −5 h offset: (21 − 5) mod 24 = 16.", skill: "prediction" },
    { id: `${t}-3`, topicId: t, question: "Offline AUC is excellent, but production conversions predicted for new customers are badly off. Which bug best explains it?", options: ["Null → 0 instead of the training mean for avg_rating", "tenure_days computed twice", "A stale join on a table nobody reads", "Batch SQL uses UTC"], answer: 0, explanation: "New customers have no ratings. Training imputed 4.2; serving sends 0, which the model reads as very low satisfaction, dragging their scores down.", skill: "diagnosis" },
    { id: `${t}-4`, topicId: t, question: "Which measurement directly shows the effect of skew on the model's output?", options: ["|p_serve − p_train| per entity for the same model", "Training loss", "The number of features", "Serving latency"], answer: 0, explanation: "Scoring the same model on both feature vectors isolates the skew: any prediction gap comes only from the features.", skill: "metric" },
    { id: `${t}-5`, topicId: t, question: "What makes the diffs disappear by construction rather than by careful re-implementation?", options: ["Both paths call one shared feature definition (feature store / shared library)", "Retraining the model more often", "Adding more features", "Raising the decision threshold"], answer: 0, explanation: "With one definition there is nothing to drift apart. Toggle the fix and every red cell and every Δp goes to zero.", skill: "fix" },
  ],
};
