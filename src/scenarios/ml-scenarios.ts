import { Archive, BarChart3, Boxes, Cpu, Database, Gauge, Server, Workflow } from "lucide-react";
import type { SimulationEvent, TraceSpan } from "../types/simulation";
import { churnData, dropFeature, evaluateModel, fraudData, reorder, trainLogistic, type Dataset, type Model } from "./ml-model";
import type { Evidence, HealthMetric, Outcome, Scenario } from "./types";

export const ML_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "data", label: "Training data", Icon: Database, x: 0, y: 30, match: (s) => s.id === "load" },
    { id: "features", label: "Feature pipeline", Icon: Workflow, x: 170, y: 30, match: (s) => s.id === "features" },
    { id: "train", label: "Train", Icon: Cpu, x: 340, y: 30, match: (s) => s.id === "train" },
    { id: "offline", label: "Offline eval", Icon: BarChart3, x: 510, y: 30, match: (s) => s.id === "offline" },
    { id: "registry", label: "Model registry", Icon: Archive, x: 680, y: 30, match: (s) => s.id === "register" },
    { id: "requests", label: "Prod requests", Icon: Boxes, x: 170, y: 170, match: () => false },
    { id: "serving", label: "Serving features", Icon: Workflow, x: 340, y: 170, match: (s) => s.id === "serve.features" },
    { id: "server", label: "Model server", Icon: Server, x: 510, y: 170, match: (s) => s.id === "serve.predict" },
    { id: "monitor", label: "Online eval", Icon: Gauge, x: 680, y: 170, match: (s) => s.id === "online" },
  ],
  edges: [
    { source: "data", target: "features" }, { source: "features", target: "train" }, { source: "train", target: "offline" }, { source: "offline", target: "registry" },
    { source: "registry", target: "server", sourceHandle: "b", targetHandle: "t" },
    { source: "requests", target: "serving" }, { source: "serving", target: "server" }, { source: "server", target: "monitor" },
  ],
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
let evn = 0;
const ev = (message: string, severity: SimulationEvent["severity"] = "info"): SimulationEvent => ({ id: `m${evn++}`, type: "log", message, severity });

export function pipelineSpans(o: { trainRows: number; features: string[]; offlineAcc: number; onlineAcc: number; servingOrder: string[]; prodRows: number; version?: string }): TraceSpan[] {
  const s = (id: string, name: string, startMs: number, durationMs: number, extra: Partial<TraceSpan> = {}): TraceSpan => ({ id, name, component: "ml-pipeline", startMs, durationMs, status: "ok", events: [], ...extra });
  return [
    s("load", "load training data", 0, 1200, { output: { rows: o.trainRows } }),
    s("features", "build features", 1200, 800, { output: { features: o.features.join(", ") } }),
    s("train", "train logistic regression", 2000, 4200, { input: { features: o.features.join(", ") } }),
    s("offline", "evaluate on held-out test set", 6200, 600, { metrics: { accuracy: +o.offlineAcc.toFixed(3) }, events: [ev(`offline accuracy ${pct(o.offlineAcc)}`)] }),
    s("register", `register model ${o.version ?? "churn-v7"}`, 6800, 200, { output: { version: o.version ?? "churn-v7" } }),
    { ...s("serve.features", "serving: assemble feature vectors", 7000, 300, { output: { column_order: o.servingOrder.join(", ") } }), component: "serving" },
    { ...s("serve.predict", "serving: predict (batch)", 7300, 900, { output: { requests: o.prodRows, http_200: "100%" } }), component: "serving" },
    { ...s("online", "online evaluation (labels after 30 days)", 8200, 400, { metrics: { accuracy: +o.onlineAcc.toFixed(3) }, events: [ev(`online accuracy ${pct(o.onlineAcc)}`, o.offlineAcc - o.onlineAcc > 0.05 ? "warning" : "info")] }), component: "monitoring" },
  ];
}

export const weightsLines = (m: Model) => m.features.map((f, j) => `${f}: ${m.weights[j] >= 0 ? "+" : ""}${m.weights[j].toFixed(2)} (standardized)`).sort((a, b) => Math.abs(parseFloat(b.split(": ")[1])) - Math.abs(parseFloat(a.split(": ")[1])));
export const meanOf = (ds: Dataset, j: number) => ds.rows.reduce((s, r) => s + r.x[j], 0) / ds.rows.length;

export type Act = { id: string; label: string; lines: string[] };
export const toEvidence = (acts: Act[], relevant: string[]): Evidence[] => acts.map((a) => ({ id: a.id, action: a.label, title: a.label, lines: a.lines, relevant: relevant.includes(a.id) }));
export const healthy: HealthMetric[] = [
  { label: "Model server uptime", value: "100%", status: "ok", kind: "health" },
  { label: "p95 inference latency", value: "12 ms", status: "ok", kind: "health" },
  { label: "HTTP errors", value: "0%", status: "ok", kind: "health" },
];

// ---------- leakage ----------
const LEAK = "cancel_ticket_logged";
function leakageRun(fixId?: string): Outcome {
  const train = churnData(1, 800), test = churnData(2, 400), prod = churnData(3, 400, true);
  const drop = fixId === "f-drop";
  const big = fixId === "f-more" ? churnData(9, 3000) : train;
  const tr = drop ? dropFeature(big, LEAK) : big, te = drop ? dropFeature(test, LEAK) : test, pr = drop ? dropFeature(prod, LEAK) : prod;
  const m = trainLogistic(tr);
  const thr = fixId === "f-threshold" ? 0.2 : 0.5;
  const off = evaluateModel(m, te.rows, thr), on = evaluateModel(m, pr.rows, thr);
  const resolved = on.recall >= 0.5 && Math.abs(off.accuracy - on.accuracy) < 0.08;
  const acts: Act[] = [
    { id: "offline", label: "Review offline evaluation", lines: [`held-out test accuracy ${pct(off.accuracy)}, recall ${pct(off.recall)}, precision ${pct(off.precision)}`, `train/test split: random 80/20 of historical customers`] },
    { id: "online", label: "Review online evaluation (labels arrived after 30 days)", lines: [`accuracy ${pct(on.accuracy)}, recall ${pct(on.recall)} (${on.tp} of ${on.tp + on.fn} churners flagged)`, `confusion: TP ${on.tp} · FP ${on.fp} · TN ${on.tn} · FN ${on.fn}`] },
    { id: "weights", label: "Inspect learned feature weights", lines: weightsLines(m) },
    { id: "dist", label: "Compare feature distributions: training vs production", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(2)} · prod mean ${meanOf(pr, j).toFixed(2)}`) },
    { id: "lineage", label: "Read feature documentation (lineage)", lines: [`${LEAK}: set by the billing system when a cancellation ticket is filed (i.e. after the customer decides to churn).`, "tenure_months, monthly_spend: from the accounts table, available at prediction time.", "support_tickets: rolling 90-day count, available at prediction time."] },
    { id: "serving", label: "Check serving payload and feature order", lines: ["request {tenure_months, monthly_spend, support_tickets, cancel_ticket_logged} → vector in training order ✓", "schema hash matches training"] },
    { id: "deploys", label: "Read the change log", lines: ["Sep 02 data-eng: added billing-system columns to the churn feature table (incl. cancel_ticket_logged)", "Sep 05 ml: churn-v7 trained on new feature table; offline accuracy 98% → shipped", "Sep 20 crm: retention campaign targets customers with churn score > 0.5"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: off.accuracy, onlineAcc: on.accuracy, servingOrder: tr.features, prodRows: pr.rows.length }),
    metrics: [...healthy, { label: "Offline accuracy", value: pct(off.accuracy), status: "ok", kind: "quality" }, { label: "Online accuracy", value: pct(on.accuracy), status: Math.abs(off.accuracy - on.accuracy) < 0.08 ? "ok" : "error", kind: "quality" }, { label: "Online recall (churners caught)", value: pct(on.recall), status: on.recall >= 0.5 ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Model card", text: `churn model: offline accuracy ${pct(off.accuracy)}` }, { label: "Production (30 days later)", text: `flagged ${on.tp + on.fp} customers; caught ${on.tp} of ${on.tp + on.fn} actual churners`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["weights", "dist", "lineage", "deploys"]),
  };
}

// ---------- imbalance ----------
function imbalanceRun(fixId?: string): Outcome {
  const tr = fraudData(4, 3000), te = fraudData(5, 2000);
  const m = trainLogistic(tr, { positiveWeight: fixId === "f-weights" ? 30 : 1, epochs: fixId === "f-epochs" ? 1600 : 400 });
  const e = evaluateModel(m, te.rows);
  const at = (t: number) => evaluateModel(m, te.rows, t);
  const resolved = e.recall >= 0.6;
  const acts: Act[] = [
    { id: "accuracy", label: "Review the accuracy dashboard", lines: [`accuracy ${pct(e.accuracy)} on ${e.n} test transactions`] },
    { id: "classes", label: "Check class distribution", lines: [`fraud ${e.positives} of ${e.n} (${pct(e.positives / e.n)}) · legitimate ${e.n - e.positives}`, `predicting “legitimate” for everything would score ${pct(1 - e.positives / e.n)} accuracy`] },
    { id: "confusion", label: "Inspect the confusion matrix", lines: [`TP ${e.tp} · FN ${e.fn} (fraud missed) · FP ${e.fp} · TN ${e.tn}`] },
    { id: "perclass", label: "Compute per-class precision and recall", lines: [`fraud recall ${pct(e.recall)}, precision ${pct(e.precision)}, F1 ${e.f1.toFixed(2)}`] },
    { id: "thresholds", label: "Sweep the decision threshold", lines: [0.5, 0.3, 0.1].map((t) => { const x = at(t); return `threshold ${t}: recall ${pct(x.recall)}, precision ${pct(x.precision)}, accuracy ${pct(x.accuracy)}`; }) },
    { id: "drift", label: "Compare train vs test feature distributions", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(2)} · test mean ${meanOf(te, j).toFixed(2)}`) },
    { id: "deploys", label: "Read the change log", lines: ["Aug 30 ml: fraud-v3 trained with default (unweighted) log loss, threshold 0.5", "Aug 30 ml: model selected by highest validation accuracy", "Sep 15 payments: chargeback volume +40% QoQ"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: e.accuracy, onlineAcc: e.accuracy, servingOrder: tr.features, prodRows: te.rows.length }),
    metrics: [...healthy, { label: "Accuracy", value: pct(e.accuracy), status: "ok", kind: "quality" }, { label: "Fraud recall", value: pct(e.recall), status: e.recall >= 0.6 ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Dashboard", text: `fraud model accuracy ${pct(e.accuracy)}` }, { label: "Finance", text: `${e.fn} of ${e.positives} fraudulent test transactions were approved`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["classes", "confusion", "perclass", "thresholds"]),
  };
}

// ---------- feature order ----------
function featureOrderRun(fixId?: string): Outcome {
  const tr = dropFeature(churnData(fixId === "f-retrain" ? 7 : 1, 800), LEAK), te = dropFeature(churnData(2, 400), LEAK), pr = dropFeature(churnData(3, 400, true), LEAK);
  const m = trainLogistic(tr);
  const fixed = fixId === "f-schema";
  const order = fixed ? [0, 1, 2] : [1, 0, 2];
  const off = evaluateModel(m, te.rows), on = evaluateModel(m, reorder(pr.rows, order));
  const servingNames = order.map((j) => tr.features[j]);
  const resolved = Math.abs(off.accuracy - on.accuracy) < 0.06;
  const servedMeans = order.map((j) => meanOf(pr, j));
  const acts: Act[] = [
    { id: "offline", label: "Review offline evaluation", lines: [`held-out accuracy ${pct(off.accuracy)} (model file sha 9f3c…, unchanged since Sep 05)`] },
    { id: "online", label: "Review online evaluation", lines: [`online accuracy ${pct(on.accuracy)}, recall ${pct(on.recall)}`, "every request returned HTTP 200 with a probability in [0, 1]"] },
    { id: "payload", label: "Trace one request through serving", lines: ["raw request: {tenure_months: 4, monthly_spend: 182.5, support_tickets: 3}", `vector sent to model: [${order.map((j) => [4, 182.5, 3][j]).join(", ")}]`, `model expects: [${tr.features.join(", ")}]`] },
    { id: "positional", label: "Compare per-position input statistics: training vs serving", lines: tr.features.map((f, j) => `position ${j} (${f} in training): train mean ${meanOf(tr, j).toFixed(1)} · served mean ${servedMeans[j].toFixed(1)}`) },
    { id: "named", label: "Compare raw feature distributions by name", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(1)} · prod mean ${meanOf(pr, j).toFixed(1)}`) },
    { id: "deploys", label: "Read the change log", lines: ["Oct 14 feature-service 3.0: rewrote request→vector mapping using dict iteration order", "Oct 14 model-server: no change (churn-v6 still serving)", "Oct 01 crm: new signup flow"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: off.accuracy, onlineAcc: on.accuracy, servingOrder: servingNames, prodRows: pr.rows.length }),
    metrics: [...healthy, { label: "Offline accuracy", value: pct(off.accuracy), status: "ok", kind: "quality" }, { label: "Online accuracy", value: pct(on.accuracy), status: resolved ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Weekly report", text: `online accuracy ${pct(on.accuracy)} (offline ${pct(off.accuracy)})`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["payload", "positional", "deploys"]),
  };
}

export const ML_SCENARIOS: Scenario[] = [
  {
    id: "ml-leakage", title: "Churn model: 98% offline, useless in production", system: "ML", difficulty: "Intermediate", relatedTopic: "overfitting",
    symptom: "Offline accuracy far above online", objective: "Recognize label leakage from feature weights and feature availability.",
    incident: "The new churn model scored 98% accuracy offline. Four weeks after launch, the retention campaign targeted almost nobody and churn didn't move.",
    architecture: ML_ARCH, run: leakageRun,
    hypotheses: [
      { id: "h-leak", text: `Label leakage: ${LEAK} is only set after a customer churns, so it's always 0 at prediction time.`, isRootCause: true, evidenceIds: ["weights", "dist", "lineage", "deploys"], explanation: `${LEAK} has by far the largest weight; it averages ~0.3 in training and exactly 0 in production; lineage says it's written after cancellation.` },
      { id: "h-drift", text: "Customer behaviour drifted after launch.", isRootCause: false, evidenceIds: ["dist"], explanation: "tenure, spend and ticket distributions are similar between training and production. Only the leaked column differs." },
      { id: "h-overfit", text: "The model is overfitting the training set.", isRootCause: false, evidenceIds: ["offline"], explanation: "It scores just as well on a held-out test set. The test set has the same leak, so the split can't reveal it." },
      { id: "h-serving", text: "Serving sends features in the wrong order.", isRootCause: false, evidenceIds: ["serving"], explanation: "Payload maps to the training order and the schema hash matches." },
    ],
    fixes: [
      { id: "f-drop", label: `Remove ${LEAK} and retrain on features available at prediction time`, explanation: "Offline accuracy drops to a realistic level and matches production." },
      { id: "f-more", label: "Retrain on 3,000 more recent customers", explanation: "More data with the same leak gives the same result." },
      { id: "f-threshold", label: "Lower the decision threshold to 0.2", explanation: "The model's signal in production is near zero; a lower threshold flags a few more people at random." },
    ],
    verifications: [
      { id: "v-time", label: "Evaluate with a time-based split using only features available at the prediction timestamp, and compare to online metrics", sound: true, explanation: "Mirrors production conditions, which a random split with leaked features cannot." },
      { id: "v-offline", label: "Confirm offline accuracy is still above 95%", sound: false, explanation: "That number was the symptom." },
      { id: "v-uptime", label: "Confirm the model server is healthy", sound: false, explanation: "It always was." },
    ],
    rootCause: `A billing column written after cancellation (${LEAK}) entered the feature table. It nearly encodes the label, so offline metrics were excellent; in production it's always 0 and the model loses its main signal.`,
  },
  {
    id: "ml-imbalance", title: "Fraud model reports 97% accuracy while losses rise", system: "ML", difficulty: "Beginner", relatedTopic: "overfitting",
    symptom: "High accuracy, rising chargebacks", objective: "See why accuracy hides minority-class failure and pick the right metric.",
    incident: "Chargebacks are up 40% this quarter, but the fraud model's dashboard shows ~97% accuracy, the same as at launch.",
    architecture: ML_ARCH, run: imbalanceRun,
    hypotheses: [
      { id: "h-imb", text: "Class imbalance: an unweighted model at threshold 0.5 rarely predicts fraud; accuracy hides it.", isRootCause: true, evidenceIds: ["classes", "confusion", "perclass", "thresholds"], explanation: "Fraud is ~3% of transactions; most fraud is a false negative while accuracy stays near the all-legitimate baseline." },
      { id: "h-drift", text: "Fraud patterns drifted.", isRootCause: false, evidenceIds: ["drift"], explanation: "Train and test feature distributions match; the model was weak on fraud from the start." },
      { id: "h-features", text: "The features carry no fraud signal.", isRootCause: false, evidenceIds: ["thresholds"], explanation: "Lowering the threshold raises recall sharply, so the scores do separate fraud; the operating point is wrong." },
    ],
    fixes: [
      { id: "f-weights", label: "Retrain with class weights (fraud weighted ~30×) and select by recall/F1", explanation: "The loss stops ignoring the minority class." },
      { id: "f-epochs", label: "Train 4× longer", explanation: "Converges to the same accuracy-optimal, low-recall solution." },
      { id: "f-testset", label: "Evaluate on a larger test set for a more precise accuracy number", explanation: "Measures the wrong metric more precisely." },
    ],
    verifications: [
      { id: "v-recall", label: "Track fraud recall and precision (or PR-AUC) per release and alert on recall drops", sound: true, explanation: "Monitors the class that matters." },
      { id: "v-acc", label: "Confirm accuracy stays above 95%", sound: false, explanation: "It did throughout the incident." },
      { id: "v-latency", label: "Check inference latency", sound: false, explanation: "Unrelated to quality." },
    ],
    rootCause: "Fraud is ~3% of traffic. Trained with unweighted log loss and judged by accuracy, the model learned to approve almost everything: 97% accurate, missing most fraud.",
  },
  {
    id: "ml-feature-order", title: "Predictions degraded after a serving refactor", system: "ML", difficulty: "Intermediate", relatedTopic: "forward-propagation",
    symptom: "Online accuracy dropped; no errors", objective: "Find a training-serving skew that produces valid-looking but wrong inputs.",
    incident: "After the feature-service refactor (no model change), the churn model's weekly online accuracy dropped well below its offline number. Every request returns HTTP 200.",
    architecture: ML_ARCH, run: featureOrderRun,
    hypotheses: [
      { id: "h-order", text: "Serving assembles the feature vector in a different column order than training.", isRootCause: true, evidenceIds: ["payload", "positional", "deploys"], explanation: "The traced request puts monthly_spend in position 0 where the model expects tenure_months; per-position stats shift while named stats don't." },
      { id: "h-drift", text: "Customer data drifted.", isRootCause: false, evidenceIds: ["named"], explanation: "Raw features by name have the same distributions as training." },
      { id: "h-corrupt", text: "The model file was corrupted during deploy.", isRootCause: false, evidenceIds: ["offline"], explanation: "Model hash unchanged; the same file still scores well offline." },
    ],
    fixes: [
      { id: "f-schema", label: "Build serving vectors from the model's named feature schema and validate it at the boundary", explanation: "Each value lands in the position the model was trained on." },
      { id: "f-retrain", label: "Retrain the model on last month's data", explanation: "The new model still expects training order; serving still swaps columns." },
      { id: "f-rollback", label: "Roll back the model to the previous version", explanation: "The model didn't change; the feature service did." },
    ],
    verifications: [
      { id: "v-contract", label: "Add a serving test that sends named fixtures and compares predictions to the training pipeline's, then confirm online ≈ offline accuracy", sound: true, explanation: "Catches any future skew between the two code paths." },
      { id: "v-200", label: "Confirm all requests return HTTP 200", sound: false, explanation: "They did throughout." },
      { id: "v-retrain", label: "Retrain weekly to keep the model fresh", sound: false, explanation: "Doesn't touch the serving path." },
    ],
    rootCause: "The refactored feature service built vectors using dictionary order, swapping tenure_months and monthly_spend. Shapes and types still match, so nothing errors; the model just sees the wrong numbers.",
  },
];
