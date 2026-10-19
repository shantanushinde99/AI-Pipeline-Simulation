import { sigmoid } from "../lib/math";
import { rng } from "../lib/random";
import { churnData, dropFeature, evaluateModel, trainLogistic, type Dataset, type Row } from "./ml-model";
import { ML_ARCH, healthy, meanOf, pipelineSpans, toEvidence, weightsLines, type Act } from "./ml-scenarios";
import type { Outcome, Scenario } from "./types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const col = (ds: Dataset | Row[], j: number) => ("rows" in ds ? ds.rows : ds).map((r) => r.x[j]);

/** Population stability index of `actual` against `expected`, using the expected sample's deciles as bins. */
export function psi(expected: number[], actual: number[]) {
  const sorted = [...expected].sort((a, b) => a - b);
  const edges = [...new Set([1, 2, 3, 4, 5, 6, 7, 8, 9].map((k) => sorted[Math.floor((k / 10) * sorted.length)]))];
  const share = (xs: number[]) => {
    const c = new Array<number>(edges.length + 1).fill(0);
    xs.forEach((v) => c[edges.filter((e) => v > e).length]++);
    return c.map((n) => Math.max(n / xs.length, 1e-4));
  };
  const e = share(expected), a = share(actual);
  return e.reduce((s, ei, i) => s + (a[i] - ei) * Math.log(a[i] / ei), 0);
}

/** Expected calibration error over 10 equal-width probability bins, plus the per-bin table. */
export function calibration(probs: number[], labels: number[]) {
  const bins = Array.from({ length: 10 }, (_, b) => {
    const idx = probs.map((p, i) => [p, i] as const).filter(([p]) => Math.min(9, Math.floor(p * 10)) === b).map(([, i]) => i);
    const conf = idx.reduce((s, i) => s + probs[i], 0) / (idx.length || 1);
    const obs = idx.reduce((s, i) => s + labels[i], 0) / (idx.length || 1);
    return { lo: b / 10, n: idx.length, conf, obs };
  });
  return { bins, ece: bins.reduce((s, b) => s + (b.n / probs.length) * Math.abs(b.conf - b.obs), 0) };
}

// ---------- data drift ----------
/**
 * Late-delivery data. Beyond ~6 km, suburban orders get distance-scaled delivery promises and highway routes,
 * so lateness stops rising with distance. A linear model fit on urban orders only sees the rising part.
 */
function deliveryData(seed: number, n: number, suburbanShare: number): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const suburban = r.next() < suburbanShare;
    const d = suburban ? 8 + r.next() * 14 : 0.5 + r.next() * 7.5;
    const items = 1 + Math.floor(r.next() * 8);
    const rush = r.next() < 0.3 ? 1 : 0;
    const z = 0.9 * Math.min(d, 6) - 0.45 * Math.max(0, d - 6) + 0.35 * (items - 4) + 1.3 * rush - 3.2;
    return { x: [Math.round(d * 10) / 10, items, rush], y: r.next() < sigmoid(z) ? 1 : 0 };
  });
  return { features: ["distance_km", "basket_items", "rush_hour"], rows };
}

function driftRun(fixId?: string): Outcome {
  const retrain = fixId === "f-retrain";
  const launch = deliveryData(11, 1500, 0), prod = deliveryData(13, 600, 0.65);
  const tr = retrain ? deliveryData(14, 1500, 0.65) : launch, test = retrain ? deliveryData(15, 600, 0.65) : deliveryData(12, 600, 0);
  const m = trainLogistic(tr, { epochs: fixId === "f-epochs" ? 1600 : 400 });
  const thr = fixId === "f-threshold" ? 0.35 : 0.5;
  const off = evaluateModel(m, test.rows, thr), on = evaluateModel(m, prod.rows, thr);
  const resolved = on.accuracy >= 0.7 && on.accuracy >= off.accuracy - 0.05;
  const psis = launch.features.map((f, j) => [f, psi(col(launch, j), col(prod, j))] as const);
  const sub = prod.rows.filter((r) => r.x[0] >= 8), urb = prod.rows.filter((r) => r.x[0] < 8);
  const acts: Act[] = [
    { id: "online", label: "Review online accuracy (labels after delivery)", lines: [`online accuracy ${pct(on.accuracy)} vs offline ${pct(off.accuracy)}`, `predicted late: ${on.tp + on.fp} of ${on.n} · actually late: ${on.positives}`, `confusion: TP ${on.tp} · FP ${on.fp} · TN ${on.tn} · FN ${on.fn}`] },
    { id: "psi", label: "Compute PSI per feature (training vs last 7 days)", lines: [...psis.map(([f, v]) => `${f}: PSI ${v.toFixed(2)}${v > 0.25 ? "  ← major shift" : v > 0.1 ? "  ← moderate shift" : ""}`), "rule of thumb: < 0.1 stable · 0.1–0.25 moderate · > 0.25 major"] },
    { id: "dist", label: "Compare feature ranges: training vs production", lines: launch.features.map((f, j) => { const a = col(launch, j), b = col(prod, j); return `${f}: train ${Math.min(...a)}–${Math.max(...a)} (mean ${meanOf(launch, j).toFixed(1)}) · prod ${Math.min(...b)}–${Math.max(...b)} (mean ${meanOf(prod, j).toFixed(1)})`; }) },
    { id: "slices", label: "Break online accuracy down by distance", lines: [[urb, "< 8 km (inside training range)"], [sub, "≥ 8 km (outside training range)"]].map(([rows, name]) => { const e = evaluateModel(m, rows as Row[], thr); return `${name}: ${e.n} orders, accuracy ${pct(e.accuracy)}, predicted late ${pct((e.tp + e.fp) / e.n)}, actually late ${pct(e.positives / e.n)}`; }) },
    { id: "thresholds", label: "Sweep the decision threshold on production data", lines: [0.5, 0.35, 0.7].map((t) => `threshold ${t}: online accuracy ${pct(evaluateModel(m, prod.rows, t).accuracy)}`) },
    { id: "weights", label: "Inspect learned feature weights", lines: weightsLines(m) },
    { id: "serving", label: "Check serving payload and schema", lines: ["request {distance_km, basket_items, rush_hour} → vector in training order ✓", "units: distance_km in km at training and serving ✓", "schema hash matches training"] },
    { id: "deploys", label: "Read the change log", lines: ["Sep 01 ml: eta-late-v4 trained on city-hub orders (Jun–Aug), offline accuracy above 80%", "Sep 22 ops: Northgate suburban hub launched (orders 8–22 km)", "Oct 02 dispatch: new courier app release", "Oct 10 model-server: autoscaling min replicas 2 → 3"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: off.accuracy, onlineAcc: on.accuracy, servingOrder: tr.features, prodRows: prod.rows.length, version: "eta-late-v4" }),
    metrics: [...healthy, { label: "Offline accuracy", value: pct(off.accuracy), status: "ok", kind: "quality" }, { label: "Online accuracy", value: pct(on.accuracy), status: resolved ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Ops lead", text: `the model flags ${pct((on.tp + on.fp) / on.n)} of orders as "will be late"; ${pct(on.positives / on.n)} actually were. Couriers ignore the alerts.`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["psi", "dist", "slices", "deploys"]),
  };
}

// ---------- calibration ----------
/** Refund-fraud data. `agreedOnly` keeps only cases all three reviewers agreed on, i.e. the clear-cut ones. */
function refundData(seed: number, n: number, agreedOnly = false): Dataset {
  const r = rng(seed);
  const rows: Row[] = [];
  while (rows.length < n) {
    const risk = r.gaussian(), tenure = r.next() * 6, recent = Math.floor(r.next() * r.next() * 6);
    const z = 1.2 * risk - 0.45 * tenure + 0.55 * recent - 1.2;
    // Consensus labels are reviewers' judgment of clear-cut cases, so they follow the evidence with no outcome noise.
    const y = agreedOnly ? (z > 0 ? 1 : 0) : r.next() < sigmoid(z) ? 1 : 0;
    if (!agreedOnly || Math.abs(z) > 1.5) rows.push({ x: [+risk.toFixed(2), +tenure.toFixed(1), recent], y });
  }
  return { features: ["account_risk_score", "tenure_years", "refunds_90d"], rows };
}

const AUTO = 0.05;
function calibrationRun(fixId?: string): Outcome {
  const train = refundData(fixId === "f-more" ? 31 : 21, fixId === "f-more" ? 3000 : 1000, true), val = refundData(22, 800), prod = refundData(23, 1500), launchMonth = refundData(25, 1500);
  const m = trainLogistic(train, { epochs: fixId === "f-epochs" ? 2000 : 400 });
  const logit = (p: number) => Math.log(Math.max(p, 1e-9) / Math.max(1 - p, 1e-9));
  const platt = fixId === "f-platt" ? trainLogistic({ features: ["logit"], rows: val.rows.map((r) => ({ x: [logit(m.prob(r.x))], y: r.y })) }) : null;
  const prob = (x: number[]) => (platt ? platt.prob([logit(m.prob(x))]) : m.prob(x));
  const probs = prod.rows.map((r) => prob(r.x)), labels = prod.rows.map((r) => r.y);
  const acc = probs.filter((p, i) => (p >= 0.5 ? 1 : 0) === labels[i]).length / probs.length;
  const cal = calibration(probs, labels);
  const auto = probs.map((p, i) => [p, labels[i]] as const).filter(([p]) => p < AUTO);
  const autoFraud = auto.filter(([, y]) => y).length / Math.max(1, auto.length);
  const resolved = cal.ece <= 0.05 && autoFraud <= 0.06;
  const trainCal = calibration(train.rows.map((r) => prob(r.x)), train.rows.map((r) => r.y));
  const acts: Act[] = [
    { id: "accuracy", label: "Review the accuracy dashboard", lines: [`production accuracy ${pct(acc)} at threshold 0.5`, `reference: a model trained on unfiltered outcomes scores ${pct(evaluateModel(trainLogistic(refundData(26, 1000)), prod.rows).accuracy)} on the same traffic`, `fraud base rate in production ${pct(labels.filter(Boolean).length / labels.length)}`] },
    { id: "rule", label: "Audit the auto-approve rule", lines: [`rule: auto-approve when P(fraud) < ${AUTO}; finance budget assumes ≤ ${AUTO * 100}% fraud in that bucket`, `auto-approved ${auto.length} of ${probs.length} refunds (${pct(auto.length / probs.length)})`, `observed fraud among auto-approved: ${pct(autoFraud)} (mean predicted ${pct(auto.reduce((s, [p]) => s + p, 0) / Math.max(1, auto.length))})`] },
    { id: "reliability", label: "Plot a reliability diagram (production)", lines: [...cal.bins.filter((b) => b.n).map((b) => `P ${b.lo.toFixed(1)}–${(b.lo + 0.1).toFixed(1)}: n=${b.n}, mean predicted ${b.conf.toFixed(2)}, observed ${b.obs.toFixed(2)}`), `ECE = ${cal.ece.toFixed(3)}`] },
    { id: "trainset", label: "Inspect how the training set was built", lines: [`${train.rows.length} labelled refunds, all with 3/3 reviewer agreement (disputed cases removed)`, `accuracy on a consensus holdout: ${pct(evaluateModel(m, refundData(24, 800, true).rows).accuracy)}`, `ECE on the training set itself: ${trainCal.ece.toFixed(3)}`, `share of production refunds that reviewers would have disputed: ${pct(prod.rows.filter((r) => { const p = sigmoid(1.2 * r.x[0] - 0.45 * r.x[1] + 0.55 * r.x[2] - 1.2); return Math.abs(Math.log(p / (1 - p))) <= 1.5; }).length / prod.rows.length)}`] },
    { id: "weights", label: "Inspect learned feature weights", lines: weightsLines(m) },
    { id: "drift", label: "Compute PSI per feature (production at launch vs this month)", lines: prod.features.map((f, j) => `${f}: PSI ${psi(col(launchMonth, j), col(prod, j)).toFixed(2)}`) },
    { id: "deploys", label: "Read the change log", lines: ["Aug 14 labeling: vendor switched to 3-reviewer consensus; disputed cases excluded from training exports", "Aug 20 ml: refund-risk-v2 trained on consensus export; holdout accuracy excellent → shipped", "Sep 01 finance: auto-approve refunds with P(fraud) < 0.05", "Sep 18 support: new refund form"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: train.rows.length, features: train.features, offlineAcc: acc, onlineAcc: acc, servingOrder: train.features, prodRows: prod.rows.length, version: "refund-risk-v2" }),
    metrics: [...healthy, { label: "Accuracy", value: pct(acc), status: "ok", kind: "quality" }, { label: "ECE", value: cal.ece.toFixed(3), status: cal.ece <= 0.05 ? "ok" : "error", kind: "quality" }, { label: "Fraud in auto-approved", value: pct(autoFraud), status: autoFraud <= 0.06 ? "ok" : "error", kind: "quality", hint: `budget ${AUTO * 100}%` }],
    transcript: [{ label: "Finance", text: `${auto.filter(([, y]) => y).length} of ${auto.length} auto-approved refunds this month were fraudulent (${pct(autoFraud)}; budget ${AUTO * 100}%).`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["rule", "reliability", "trainset", "deploys"]),
  };
}

// ---------- training-serving skew (units) ----------
const LEAK = "cancel_ticket_logged";
function skewRun(fixId?: string): Outcome {
  const tr = dropFeature(churnData(fixId === "f-retrain" ? 7 : fixId === "f-rollback" ? 5 : 1, 800), LEAK), te = dropFeature(churnData(2, 400), LEAK), pr = dropFeature(churnData(3, 400, true), LEAK);
  const m = trainLogistic(tr);
  // Online feature service reads monthly spend from the payments API, which reports cents.
  const serve = (rows: Row[]) => rows.map((r) => {
    return { ...r, x: [r.x[0], fixId === "f-shared" ? r.x[1] : r.x[1] * 100, r.x[2]] };
  });
  const served = serve(pr.rows);
  const off = evaluateModel(m, te.rows), on = evaluateModel(m, served);
  const resolved = Math.abs(off.accuracy - on.accuracy) < 0.06;
  const acts: Act[] = [
    { id: "offline", label: "Review offline evaluation", lines: [`held-out accuracy ${pct(off.accuracy)} (model churn-v8, trained from the warehouse feature table)`] },
    { id: "online", label: "Review online evaluation", lines: [`online accuracy ${pct(on.accuracy)}, flagged ${pct((on.tp + on.fp) / on.n)} of customers as churn risks (actual churn ${pct(on.positives / on.n)})`, "every request returned HTTP 200 with a probability in [0, 1]"] },
    { id: "logged", label: "Compare logged serving features with training, by name", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(1)} · served mean ${(served.reduce((s, r) => s + r.x[j], 0) / served.length).toFixed(1)}`) },
    { id: "warehouse", label: "Compare warehouse data: training period vs this month", lines: tr.features.map((f, j) => `${f}: PSI ${psi(col(tr, j), col(pr, j)).toFixed(2)} (train mean ${meanOf(tr, j).toFixed(1)} · this month ${meanOf(pr, j).toFixed(1)})`) },
    { id: "definitions", label: "Diff the feature definitions (training vs online)", lines: ["training (warehouse SQL): monthly_spend = SUM(invoice.amount_usd) over last 30 days", `online (feature-service): monthly_spend = payments_api.monthly_total${fixId === "f-shared" ? " → shared definition: amount_usd" : "  (field documented as integer cents)"}`, "tenure_months, support_tickets: identical definitions"] },
    { id: "payload", label: "Trace one request through serving", lines: [`raw account: tenure 4 months, spend $182.50, 3 tickets`, `vector sent to model: [4, ${serve([{ x: [4, 182.5, 3], y: 0 }])[0].x[1]}, 3]`, `model expects: [${tr.features.join(", ")}]  (column order ✓)`] },
    { id: "deploys", label: "Read the change log", lines: ["Oct 09 feature-service 4.1: monthly_spend now read live from payments API (was nightly batch)", "Oct 09 model-server: no change (churn-v8)", "Oct 03 crm: spring pricing campaign", "Sep 28 data-eng: warehouse migrated to new cluster"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: off.accuracy, onlineAcc: on.accuracy, servingOrder: tr.features, prodRows: pr.rows.length, version: "churn-v8" }),
    metrics: [...healthy, { label: "Offline accuracy", value: pct(off.accuracy), status: "ok", kind: "quality" }, { label: "Online accuracy", value: pct(on.accuracy), status: resolved ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Retention team", text: `the model flagged ${on.tp + on.fp} of ${on.n} customers as likely to churn; only ${on.positives} churned`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["logged", "definitions", "payload", "deploys"]),
  };
}

export const ML_EXTRA_SCENARIOS: Scenario[] = [
  {
    id: "ml-data-drift", title: "Late-delivery alerts fire on almost every order", system: "ML", difficulty: "Intermediate", relatedTopic: "model-drift",
    symptom: "Online accuracy fell; no code changed", objective: "Use PSI and per-slice accuracy to recognize covariate drift outside the training range.",
    incident: "Couriers have started ignoring the 'likely late' warnings: the model now flags most orders. Online accuracy dropped well below its offline number over the last month. The model, features and serving code haven't changed.",
    architecture: ML_ARCH, run: driftRun,
    hypotheses: [
      { id: "h-drift", text: "Production inputs moved outside the training range (longer distances), where the linear model's learned slope no longer holds.", isRootCause: true, evidenceIds: ["psi", "dist", "slices", "deploys"], explanation: "distance_km has a major PSI shift, production distances reach far beyond the training maximum, and accuracy is fine below 8 km but collapses above it, starting when the suburban hub launched." },
      { id: "h-threshold", text: "The decision threshold is miscalibrated.", isRootCause: false, evidenceIds: ["thresholds"], explanation: "No threshold restores accuracy: the model's scores are wrong for long-distance orders, not just shifted." },
      { id: "h-skew", text: "Serving computes features differently from training.", isRootCause: false, evidenceIds: ["serving"], explanation: "Schema, order and units match. The inputs are correct; they are just new." },
      { id: "h-concept", text: "Couriers got slower everywhere (concept drift).", isRootCause: false, evidenceIds: ["slices"], explanation: "Accuracy on orders inside the training range is unchanged; only out-of-range orders fail." },
    ],
    fixes: [
      { id: "f-retrain", label: "Retrain on the last 4 weeks of orders (including the suburban hub) and add a PSI alert", explanation: "The model learns the relationship in the region it now serves." },
      { id: "f-threshold", label: "Lower the decision threshold to 0.35", explanation: "Flags even more long-distance orders as late; the problem is the score, not the cut-off." },
      { id: "f-epochs", label: "Train the current model 4× longer", explanation: "More epochs on the same urban-only data fit the same slope more precisely." },
    ],
    verifications: [
      { id: "v-slice", label: "Evaluate the retrained model on a recent time window, sliced by distance band, and alert when any feature's PSI exceeds 0.25", sound: true, explanation: "Checks the region that broke and detects the next shift before users do." },
      { id: "v-offline", label: "Confirm offline accuracy on the original test set is unchanged", sound: false, explanation: "The original test set has no long-distance orders; it can't see this failure." },
      { id: "v-uptime", label: "Confirm the model server is healthy", sound: false, explanation: "It was healthy throughout." },
    ],
    rootCause: "The model was trained only on city-hub orders (under 8 km). When the suburban hub launched, distances moved far outside that range, where lateness no longer rises with distance. The linear model extrapolated its urban slope and flagged nearly every long-distance order.",
  },
  {
    id: "ml-calibration", title: "Auto-approved refunds blow through the fraud budget", system: "ML", difficulty: "Advanced", relatedTopic: "classification-metrics",
    symptom: "High accuracy, but the probability-based rule leaks fraud", objective: "Measure calibration (reliability diagram, ECE) separately from accuracy and fix it post hoc.",
    incident: "Finance auto-approves refunds when the model says P(fraud) < 5%. This month far more of those approvals turned out fraudulent than the 5% budget. The model's accuracy dashboard is green.",
    architecture: ML_ARCH, run: calibrationRun,
    hypotheses: [
      { id: "h-calib", text: "The model is overconfident: trained only on clear-cut consensus labels, its probabilities are too extreme for real traffic.", isRootCause: true, evidenceIds: ["rule", "reliability", "trainset", "deploys"], explanation: "In the low-probability bins observed fraud is several times the predicted rate (high ECE), the training set itself looks calibrated, and the export excluded every ambiguous case." },
      { id: "h-acc", text: "The model's accuracy has degraded.", isRootCause: false, evidenceIds: ["accuracy"], explanation: "Accuracy matches a reference model trained on real outcomes. Calibration is a different property from accuracy." },
      { id: "h-drift", text: "Fraudsters changed behaviour (input drift).", isRootCause: false, evidenceIds: ["drift"], explanation: "Feature PSIs are small; production inputs look like training inputs." },
      { id: "h-undertrained", text: "The model is under-trained and needs more epochs.", isRootCause: false, evidenceIds: ["trainset", "reliability"], explanation: "On its own clean training set the probabilities look fine; more fitting to that set makes them more extreme, not less." },
    ],
    fixes: [
      { id: "f-platt", label: "Fit Platt scaling on a held-out sample of real (unfiltered) traffic and apply it before the rule", explanation: "Probabilities are rescaled to match observed frequencies; ranking and accuracy barely change." },
      { id: "f-epochs", label: "Train for 5× more epochs", explanation: "On near-separable consensus data, longer training pushes probabilities further toward 0 and 1." },
      { id: "f-more", label: "Retrain on 3× more consensus-labelled refunds", explanation: "More clear-cut examples teach the same overconfidence." },
    ],
    verifications: [
      { id: "v-ece", label: "Track ECE and observed fraud rate in the auto-approve bucket on fresh labelled traffic; block releases above the budget", sound: true, explanation: "Measures the exact property the business rule depends on." },
      { id: "v-acc", label: "Confirm production accuracy is unchanged", sound: false, explanation: "Accuracy was normal throughout the incident; it cannot see miscalibration." },
      { id: "v-train", label: "Check the reliability diagram on the training set", sound: false, explanation: "The training set is the filtered, clear-cut data; it looks calibrated, which is how this slipped through." },
    ],
    rootCause: "The labeling vendor's switch to 3/3 consensus removed every ambiguous refund from training. On clear-cut data the model learned extreme probabilities. On real traffic, where many cases are ambiguous, its 'under 5%' bucket contains far more fraud. Accuracy can't show this; calibration can.",
  },
  {
    id: "ml-training-serving-skew", title: "Churn model flags nearly every customer", system: "ML", difficulty: "Intermediate", relatedTopic: "feature-and-training-serving-skew",
    symptom: "Churn-risk list exploded overnight", objective: "Find a feature computed differently in training and serving, even though column order and schema match.",
    incident: "Since Thursday the churn model flags most active customers as likely to churn, overwhelming the retention team. No model was deployed. Requests all succeed and the serving schema check passes.",
    architecture: ML_ARCH, run: skewRun,
    hypotheses: [
      { id: "h-units", text: "Online monthly_spend is computed in cents while the model was trained on dollars.", isRootCause: true, evidenceIds: ["logged", "definitions", "payload", "deploys"], explanation: "Served monthly_spend is ~100× the training mean, the online definition reads an integer-cents field, and the traced request sends 18250 for a $182.50 customer." },
      { id: "h-drift", text: "Customer spending really changed (data drift).", isRootCause: false, evidenceIds: ["warehouse"], explanation: "Warehouse data for this month has the same distribution as training (small PSI). Only the served value differs." },
      { id: "h-order", text: "Serving sends features in the wrong column order.", isRootCause: false, evidenceIds: ["payload"], explanation: "Column order matches; tenure and tickets arrive with the right values." },
      { id: "h-model", text: "A bad model version was deployed.", isRootCause: false, evidenceIds: ["offline", "deploys"], explanation: "churn-v8 hasn't changed and still scores well offline." },
    ],
    fixes: [
      { id: "f-shared", label: "Compute monthly_spend from one shared feature definition (USD) used by both training and serving", explanation: "Serving values match training values again." },
      { id: "f-retrain", label: "Retrain the model on the latest warehouse data", explanation: "The warehouse is in dollars; the new model still receives cents online." },
      { id: "f-rollback", label: "Roll back to the previous model, churn-v7", explanation: "churn-v7 was also trained on dollars; it receives the same cents." },
    ],
    verifications: [
      { id: "v-parity", label: "Add a parity test: compute features for the same accounts via the training and serving paths and require equal values; then confirm online ≈ offline accuracy", sound: true, explanation: "Directly tests the two code paths that diverged." },
      { id: "v-schema", label: "Confirm the serving schema check passes", sound: false, explanation: "It passed throughout: names and types matched, units didn't." },
      { id: "v-200", label: "Confirm all requests return HTTP 200", sound: false, explanation: "They always did." },
    ],
    rootCause: "Feature-service 4.1 began reading monthly spend live from the payments API, which reports integer cents. Training used dollars from the warehouse. Every served spend is 100× too large, so the model sees every customer as an extreme spender.",
  },
];
