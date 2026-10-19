import { sigmoid } from "../lib/math";
import { rng } from "../lib/random";
import { evaluateModel, trainLogistic, type Dataset, type Row } from "./ml-model";
import { ML_ARCH, healthy, meanOf, pipelineSpans, toEvidence, weightsLines, type Act } from "./ml-scenarios";
import type { Outcome, Scenario } from "./types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const num = (v: number, d = 2) => (Number.isFinite(v) ? (Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(d)) : String(v));
const col = (rows: Row[], j: number) => rows.map((r) => r.x[j]);
const sdOf = (xs: number[]) => { const m = xs.reduce((s, v) => s + v, 0) / xs.length; return Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / xs.length) || 1; };
const accuracy = (pred: (x: number[]) => number, rows: Row[]) => rows.filter((r) => (pred(r.x) >= 0.5 ? 1 : 0) === r.y).length / rows.length;

/** Full-batch gradient descent for linear (MSE) or logistic (log loss) models, with optional standardization. */
function fitGD(rows: Row[], o: { loss: "mse" | "log"; lr: number; epochs: number; scale: boolean }) {
  const d = rows[0].x.length, n = rows.length;
  const mu = Array.from({ length: d }, (_, j) => (o.scale ? col(rows, j).reduce((s, v) => s + v, 0) / n : 0));
  const sd = Array.from({ length: d }, (_, j) => (o.scale ? sdOf(col(rows, j)) : 1));
  const z = (x: number[]) => x.map((v, j) => (v - mu[j]) / sd[j]);
  const Z = rows.map((r) => z(r.x));
  let w = new Array<number>(d).fill(0), b = 0;
  const out = (zx: number[]) => { const s = zx.reduce((a, v, j) => a + v * w[j], b); return o.loss === "log" ? sigmoid(s) : s; };
  const curve: number[] = [];
  let grad0: number[] = [];
  for (let e = 0; e < o.epochs; e++) {
    const gw = new Array<number>(d).fill(0);
    let gb = 0, L = 0;
    Z.forEach((zx, i) => {
      const p = out(zx), y = rows[i].y, err = p - y;
      if (o.loss === "mse") L += err * err;
      else { const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12); L -= y * Math.log(q) + (1 - y) * Math.log(1 - q); }
      const g = o.loss === "mse" ? 2 * err : err;
      zx.forEach((v, j) => (gw[j] += g * v));
      gb += g;
    });
    if (e === 0) grad0 = gw.map((g) => g / n);
    w = w.map((wj, j) => wj - (o.lr * gw[j]) / n);
    b -= (o.lr * gb) / n;
    curve.push(L / n);
  }
  return { w, b, mu, sd, curve, grad0, predict: (x: number[]) => out(z(x)) };
}

/** CART classification tree (Gini). maxDepth = Infinity grows until every leaf is pure. */
type TreeNode = { p: number; n: number; j?: number; t?: number; l?: TreeNode; r?: TreeNode };
function growTree(rows: Row[], maxDepth: number, minLeaf = 1, depth = 0): TreeNode {
  const n = rows.length, pos = rows.reduce((s, r) => s + r.y, 0);
  const node: TreeNode = { p: pos / n, n };
  if (depth >= maxDepth || pos === 0 || pos === n || n < 2 * minLeaf) return node;
  let best = (2 * pos * (n - pos)) / n - 1e-9, bj = -1, bt = 0;
  for (let j = 0; j < rows[0].x.length; j++) {
    const s = [...rows].sort((a, c) => a.x[j] - c.x[j]);
    let pl = 0;
    for (let i = 1; i < n; i++) {
      pl += s[i - 1].y;
      if (s[i].x[j] === s[i - 1].x[j] || i < minLeaf || n - i < minLeaf) continue;
      const pr = pos - pl, nr = n - i;
      const g = (2 * pl * (i - pl)) / i + (2 * pr * (nr - pr)) / nr;
      if (g < best) { best = g; bj = j; bt = (s[i].x[j] + s[i - 1].x[j]) / 2; }
    }
  }
  if (bj < 0) return node;
  return { ...node, j: bj, t: bt, l: growTree(rows.filter((r) => r.x[bj] <= bt), maxDepth, minLeaf, depth + 1), r: growTree(rows.filter((r) => r.x[bj] > bt), maxDepth, minLeaf, depth + 1) };
}
const treeProb = (t: TreeNode, x: number[]): number => (t.l && t.r ? treeProb(x[t.j!] <= t.t! ? t.l : t.r, x) : t.p);
const treeStats = (t: TreeNode): { depth: number; leaves: number } => {
  if (!t.l || !t.r) return { depth: 0, leaves: 1 };
  const a = treeStats(t.l), b = treeStats(t.r);
  return { depth: 1 + Math.max(a.depth, b.depth), leaves: a.leaves + b.leaves };
};
const depthLabel = (d: number) => (Number.isFinite(d) ? String(d) : "None (unlimited)");

// ======================= Beginner: learning rate too high =======================
/** Delivery minutes from distance, basket size, rush hour (+ weekday, used only by the "more features" fix). */
function deliveryMinutes(seed: number, n: number): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const d = Math.round((0.5 + r.next() * 9.5) * 10) / 10, items = 1 + Math.floor(r.next() * 10), rush = r.next() < 0.3 ? 1 : 0, weekday = Math.floor(r.next() * 7);
    return { x: [d, items, rush, weekday], y: 12 + 2.6 * d + 1.1 * items + 9 * rush + r.gaussian() * 4 };
  });
  return { features: ["distance_km", "basket_items", "rush_hour", "weekday"], rows };
}
const LR_BAD = 1.5;
function lrRun(fixId?: string): Outcome {
  const k = fixId === "f-features" ? 4 : 3;
  const cut = (ds: Dataset): Dataset => ({ features: ds.features.slice(0, k), rows: ds.rows.map((r) => ({ ...r, x: r.x.slice(0, k) })) });
  const tr = cut(deliveryMinutes(fixId === "f-data" ? 52 : 51, fixId === "f-data" ? 2400 : 600)), te = cut(deliveryMinutes(53, 400));
  const lr = fixId === "f-lr" ? 0.1 : LR_BAD, epochs = fixId === "f-epochs" ? 800 : 200;
  const fit = (rate: number, ep = 200) => fitGD(tr.rows, { loss: "mse", lr: rate, epochs: ep, scale: true });
  const m = fit(lr, epochs);
  const errs = te.rows.map((r) => Math.abs(m.predict(r.x) - r.y));
  const mae = errs.reduce((s, v) => s + v, 0) / errs.length;
  const within = errs.filter((e) => e <= 10).length / errs.length;
  const resolved = mae < 6;
  const at = [1, 2, 5, 10, 50, epochs].map((e) => `epoch ${e}: train MSE ${num(m.curve[e - 1])}`);
  const sample = te.rows.slice(0, 3).map((r) => `distance ${r.x[0]} km, ${r.x[1]} items, rush ${r.x[2]} → predicted ${num(m.predict(r.x), 1)} min (actual ${r.y.toFixed(1)})`);
  const acts: Act[] = [
    { id: "offline", label: "Review offline evaluation", lines: [`test MAE ${num(mae, 1)} min (target < 6 min)`, `deliveries predicted within ±10 min: ${pct(within)}`] },
    { id: "curve", label: "Plot the training loss per epoch", lines: [...at, m.curve[epochs - 1] > m.curve[0] || !Number.isFinite(m.curve[epochs - 1]) ? "loss grows every epoch instead of falling" : "loss falls and flattens: converged"] },
    { id: "preds", label: "Look at a few predictions", lines: sample },
    { id: "weights", label: "Inspect learned weights", lines: tr.features.map((f, j) => `${f}: ${num(m.w[j])} min per standard deviation`) },
    { id: "lrsweep", label: "Train the same model with different learning rates (200 epochs)", lines: [0.01, 0.1, 0.5, 0.9, LR_BAD].map((a) => `lr ${a}: final train MSE ${num(fit(a).curve[199])}`) },
    { id: "data", label: "Check the training data for bad values", lines: tr.features.map((f, j) => { const c = col(tr.rows, j); return `${f}: min ${Math.min(...c)}, max ${Math.max(...c)}, no NaN / null values`; }).concat([`delivery_minutes: min ${Math.min(...tr.rows.map((r) => r.y)).toFixed(1)}, max ${Math.max(...tr.rows.map((r) => r.y)).toFixed(1)}`]) },
    { id: "scaling", label: "Check feature scaling", lines: ["features are standardized inside the trainer (mean 0, std 1) ✓", ...tr.features.map((f, j) => `${f}: mean ${m.mu[j].toFixed(2)}, std ${m.sd[j].toFixed(2)}`)] },
    { id: "serving", label: "Check serving payload and schema", lines: [`request → [${tr.features.join(", ")}] in training order ✓`, "schema hash matches training"] },
    { id: "deploys", label: "Read the change log", lines: [`Oct 12 ml: eta-v3 training config: learning_rate 0.1 → ${LR_BAD} ("train faster")`, "Oct 12 ml: eta-v3 trained for 200 epochs, registered automatically", "Oct 08 ops: new courier shift schedule"] },
  ];
  const spans = pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: within, onlineAcc: within, servingOrder: tr.features, prodRows: te.rows.length, version: "eta-v3" })
    .map((s) => (s.id === "train" && !resolved ? { ...s, status: "error" as const, metrics: { final_mse: num(m.curve[epochs - 1]) } } : s));
  return {
    spans,
    metrics: [...healthy, { label: "Test MAE", value: `${num(mae, 1)} min`, status: resolved ? "ok" : "error", kind: "quality", hint: "target < 6 min" }, { label: "Within ±10 min", value: pct(within), status: within > 0.8 ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Courier app", text: `"${sample[0]}"`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["curve", "preds", "lrsweep", "deploys"]),
  };
}

// ======================= Beginner: unlimited tree depth =======================
/** Coupon-response data with real label noise: the best possible accuracy is well under 100%. */
function couponData(seed: number, n: number): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const visits = Math.floor(r.next() * 13), basket = Math.round(10 + r.next() * 140), days = Math.floor(r.next() * 90), zip = 10000 + Math.floor(r.next() * 90000), signupDay = Math.floor(r.next() * 7);
    const z = 1.6 * (visits >= 5 ? 1 : 0) + 0.025 * (basket - 70) - 0.03 * (days - 30) - 0.5;
    return { x: [visits, basket, days, zip, signupDay], y: r.next() < sigmoid(z) ? 1 : 0 };
  });
  return { features: ["visits_30d", "basket_value", "days_since_order", "zip_code", "signup_weekday"], rows };
}
function treeRun(fixId?: string): Outcome {
  const k = fixId === "f-features" ? 5 : 3;
  const cut = (ds: Dataset): Dataset => ({ features: ds.features.slice(0, k), rows: ds.rows.map((r) => ({ ...r, x: r.x.slice(0, k) })) });
  const tr = cut(couponData(fixId === "f-data" ? 62 : 61, fixId === "f-data" ? 1000 : 500)), te = cut(couponData(63, 400)), pr = cut(couponData(64, 600));
  const depth = fixId === "f-depth" ? 4 : Infinity, minLeaf = fixId === "f-depth" ? 15 : 1;
  const t = growTree(tr.rows, depth, minLeaf);
  const p = (x: number[]) => treeProb(t, x);
  const trainAcc = accuracy(p, tr.rows), testAcc = accuracy(p, te.rows), onAcc = accuracy(p, pr.rows);
  const st = treeStats(t);
  const resolved = onAcc >= 0.7 && trainAcc - testAcc <= 0.1;
  const acts: Act[] = [
    { id: "offline", label: "Review offline evaluation", lines: [`held-out test accuracy ${pct(testAcc)}`, `online accuracy (last 2 weeks of redemptions) ${pct(onAcc)}`] },
    { id: "gap", label: "Compare training accuracy with test accuracy", lines: [`training accuracy ${pct(trainAcc)} · test accuracy ${pct(testAcc)} · gap ${((trainAcc - testAcc) * 100).toFixed(1)} points`] },
    { id: "tree", label: "Inspect the trained tree", lines: [`max_depth: ${depthLabel(depth)}, min_samples_leaf: ${minLeaf}`, `actual depth ${st.depth}, ${st.leaves} leaves for ${tr.rows.length} training rows (${(tr.rows.length / st.leaves).toFixed(1)} rows per leaf)`] },
    { id: "depthsweep", label: "Sweep max_depth and score each tree on the validation set", lines: [2, 3, 4, 6, 10, Infinity].map((d) => { const x = growTree(tr.rows, d, Number.isFinite(d) ? 15 : 1); return `max_depth ${depthLabel(d)}: train ${pct(accuracy((v) => treeProb(x, v), tr.rows))} · validation ${pct(accuracy((v) => treeProb(x, v), te.rows))}`; }) },
    { id: "dist", label: "Compare feature distributions: training vs production", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(1)} · prod mean ${meanOf(pr, j).toFixed(1)}`) },
    { id: "classes", label: "Check class balance", lines: [`redeemed: ${pct(tr.rows.filter((r) => r.y).length / tr.rows.length)} of training rows, ${pct(pr.rows.filter((r) => r.y).length / pr.rows.length)} in production`] },
    { id: "serving", label: "Check serving payload and schema", lines: [`request → [${tr.features.join(", ")}] in training order ✓`, "schema hash matches training"] },
    { id: "deploys", label: "Read the change log", lines: ["Oct 06 ml: coupon-v2 config cleanup removed max_depth: 4 and min_samples_leaf: 15 (now library defaults: unlimited, 1)", "Oct 06 ml: coupon-v2 registered; training accuracy reported in the model card", "Oct 09 marketing: new coupon design"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: testAcc, onlineAcc: onAcc, servingOrder: tr.features, prodRows: pr.rows.length, version: "coupon-v2" }),
    metrics: [...healthy, { label: "Training accuracy", value: pct(trainAcc), status: "ok", kind: "quality" }, { label: "Online accuracy", value: pct(onAcc), status: onAcc >= 0.7 ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Model card", text: `coupon-v2: training accuracy ${pct(trainAcc)}` }, { label: "Marketing", text: `coupons went to the wrong customers: only ${pct(onAcc)} of predictions were right`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["gap", "tree", "depthsweep", "deploys"]),
  };
}

// ======================= Beginner: -999 sentinel for missing values =======================
const SENTINEL = -999;
/** Loan-default data. Since the warehouse migration, unknown years_at_job is written as -999. */
function loanData(seed: number, n: number): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const income = Math.round(30 + r.next() * 120), debt = +(r.next() * 0.8).toFixed(2), years = Math.floor(r.next() * r.next() * 25);
    const z = -0.28 * (years - 5) + 2.2 * (debt - 0.4) - 0.01 * (income - 90) - 0.4;
    const y = r.next() < sigmoid(z) ? 1 : 0;
    return { x: [income, debt, r.next() < 0.22 ? SENTINEL : years], y };
  });
  return { features: ["income_k", "debt_ratio", "years_at_job"], rows };
}
function sentinelRun(fixId?: string): Outcome {
  const tr = loanData(fixId === "f-data" ? 72 : 71, fixId === "f-data" ? 3000 : 1000), te = loanData(73, 600), pr = loanData(74, 600);
  const known = tr.rows.filter((r) => r.x[2] !== SENTINEL).map((r) => r.x[2]).sort((a, b) => a - b);
  const median = known[Math.floor(known.length / 2)];
  const yc = col(tr.rows, 2), yMu = yc.reduce((s, v) => s + v, 0) / yc.length, ySd = sdOf(yc);
  const clipped = yc.filter((v) => Math.abs(v - yMu) > 3 * ySd).length;
  const prep = (ds: Dataset): Dataset => {
    if (fixId === "f-sentinel") return { features: [...ds.features, "years_missing"], rows: ds.rows.map((r) => ({ ...r, x: [r.x[0], r.x[1], r.x[2] === SENTINEL ? median : r.x[2], r.x[2] === SENTINEL ? 1 : 0] })) };
    if (fixId === "f-clip") return { ...ds, rows: ds.rows.map((r) => ({ ...r, x: [r.x[0], r.x[1], Math.min(Math.max(r.x[2], yMu - 3 * ySd), yMu + 3 * ySd)] })) };
    return ds;
  };
  const m = trainLogistic(prep(tr), { epochs: fixId === "f-epochs" ? 1600 : 400 });
  const off = evaluateModel(m, prep(te).rows), on = evaluateModel(m, prep(pr).rows);
  // loan-risk-v4: same pipeline before the migration, when unknowns arrived as NULL and were median-imputed.
  const v4 = trainLogistic({ ...tr, rows: tr.rows.map((r) => ({ ...r, x: [r.x[0], r.x[1], r.x[2] === SENTINEL ? median : r.x[2]] })) });
  const v4Acc = evaluateModel(v4, pr.rows.map((r) => ({ ...r, x: [r.x[0], r.x[1], r.x[2] === SENTINEL ? median : r.x[2]] }))).accuracy;
  const resolved = on.accuracy >= 0.7 && on.recall >= 0.5;
  const share = tr.rows.filter((r) => r.x[2] === SENTINEL).length / tr.rows.length;
  const acts: Act[] = [
    { id: "offline", label: "Review offline and online accuracy", lines: [`loan-risk-v5 test accuracy ${pct(off.accuracy)} · online ${pct(on.accuracy)}`, `loan-risk-v4 (before the migration) scores ${pct(v4Acc)} on the same traffic`, `defaults caught (recall) ${pct(on.recall)}`] },
    { id: "summary", label: "Summarize each feature (min / mean / max)", lines: tr.features.map((f, j) => { const c = col(tr.rows, j); return `${f}: min ${Math.min(...c)}, mean ${meanOf(tr, j).toFixed(1)}, max ${Math.max(...c)}`; }) },
    { id: "counts", label: "Count the most frequent values in years_at_job", lines: [`${SENTINEL}: ${pct(share)} of rows`, ...[0, 1, 2].map((v) => `${v}: ${pct(yc.filter((x) => x === v).length / yc.length)} of rows`)] },
    { id: "weights", label: "Inspect learned feature weights", lines: weightsLines(m) },
    { id: "outliers", label: "Run the standard 3-standard-deviation outlier check", lines: [`years_at_job: mean ${yMu.toFixed(1)}, std ${ySd.toFixed(1)} → allowed range ${(yMu - 3 * ySd).toFixed(0)} to ${(yMu + 3 * ySd).toFixed(0)}`, `${clipped} values outside the range`] },
    { id: "drift", label: "Compare feature means: training vs production", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(1)} · prod mean ${meanOf(pr, j).toFixed(1)}`) },
    { id: "serving", label: "Check serving payload and feature order", lines: [`request → [${tr.features.join(", ")}] in training order ✓`, "schema hash matches training"] },
    { id: "deploys", label: "Read the change log", lines: [`Sep 29 data-eng: warehouse migrated; unknown numeric fields now exported as ${SENTINEL} instead of NULL`, "Oct 02 ml: loan-risk-v5 retrained on the new export (pipeline no longer sees NULLs, so the impute step does nothing)", "Oct 05 risk: approvals use loan-risk-v5"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: m.features, offlineAcc: off.accuracy, onlineAcc: on.accuracy, servingOrder: m.features, prodRows: pr.rows.length, version: "loan-risk-v5" }),
    metrics: [...healthy, { label: "Online accuracy", value: pct(on.accuracy), status: resolved ? "ok" : "error", kind: "quality", hint: `v4: ${pct(v4Acc)}` }, { label: "Defaults caught", value: pct(on.recall), status: resolved ? "ok" : "warning", kind: "quality" }],
    transcript: [{ label: "Risk team", text: `since loan-risk-v5 shipped, ${on.fn} of ${on.positives} loans that defaulted were rated low risk`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["summary", "counts", "weights", "outliers", "deploys"]),
  };
}

// ======================= Advanced: same customers on both sides of a random row split =======================
/**
 * Upgrade-propensity export with one row per session event. Each customer appears 1–8 times with identical
 * account features (only event_hour differs). `enterprise` adds the new large-seat plan (a harmless shift).
 */
function upgradeEvents(seed: number, customers: number, enterprise: number, oneRowEach = false) {
  const r = rng(seed);
  const rows: (Row & { cid: number })[] = [];
  for (let c = 0; c < customers; c++) {
    const usage = Math.round(r.next() * 30), tenure = Math.floor(r.next() * 48), ent = r.next() < enterprise;
    const seats = ent ? 25 + Math.floor(r.next() * 40) : 1 + Math.floor(r.next() * 20);
    const z = 0.25 * (usage - 15) + 0.04 * (tenure - 24) + 0.12 * (Math.min(seats, 25) - 10);
    const y = r.next() < sigmoid(z) ? 1 : 0;
    const k = oneRowEach ? 1 : 1 + Math.floor(r.next() * r.next() * 8);
    for (let e = 0; e < k; e++) rows.push({ cid: seed * 100000 + c, x: [usage, tenure, seats, Math.floor(r.next() * 24)], y });
  }
  return rows;
}
const UP_FEATURES = ["usage_hours_30d", "tenure_months", "seats", "event_hour"];
const DEPTHS = [2, 3, 4, 6, 10, Infinity];
function dupRun(fixId?: string): Outcome {
  const all = upgradeEvents(fixId === "f-retrain" ? 82 : 81, 800, fixId === "f-retrain" ? 0.25 : 0);
  const seen = new Set<string>();
  const deduped = all.filter((x) => { const key = `${x.cid}:${x.x.join(",")}`; if (seen.has(key)) return false; seen.add(key); return true; });
  const exactDupes = all.length - deduped.length;
  const exportRows = fixId === "f-dedupe" ? deduped : all;
  const r = rng(85);
  const group = fixId === "f-group";
  const inTest = new Map<number, boolean>();
  const testFlags = exportRows.map((row) => {
    if (!group) return r.next() < 0.25;
    if (!inTest.has(row.cid)) inTest.set(row.cid, r.next() < 0.25);
    return inTest.get(row.cid)!;
  });
  const train = exportRows.filter((_, i) => !testFlags[i]), test = exportRows.filter((_, i) => testFlags[i]);
  const prod = upgradeEvents(84, 600, 0.25, true);
  const scored = DEPTHS.map((d) => { const t = growTree(train, d, Number.isFinite(d) ? 10 : 1); return { d, t, acc: accuracy((x) => treeProb(t, x), test) }; });
  const chosen = scored.reduce((a, b) => (b.acc > a.acc ? b : a));
  const p = (x: number[]) => treeProb(chosen.t, x);
  const off = chosen.acc, on = accuracy(p, prod);
  const resolved = on >= 0.76 && Math.abs(off - on) <= 0.06;
  // upgrade-v3: the old one-row-per-customer export, random split, max_depth 4.
  const v3Rows = upgradeEvents(81, 800, 0, true), v3Train = v3Rows.filter((_, i) => i % 4), v3Test = v3Rows.filter((_, i) => !(i % 4));
  const v3 = growTree(v3Train, 4, 10);
  const trainIds = new Set(train.map((x) => x.cid));
  const overlap = test.filter((x) => trainIds.has(x.cid)).length / test.length;
  const perCustomer = exportRows.length / new Set(exportRows.map((x) => x.cid)).size;
  // What an honest (customer-grouped) split says about the same candidates.
  const gr = rng(86), gTest = new Map<number, boolean>();
  all.forEach((x) => { if (!gTest.has(x.cid)) gTest.set(x.cid, gr.next() < 0.25); });
  const gTrain = all.filter((x) => !gTest.get(x.cid)), gHold = all.filter((x) => gTest.get(x.cid));
  const seatsTrain = col(train, 2), seatsProd = col(prod, 2);
  const ent = prod.filter((x) => x.x[2] >= 25), small = prod.filter((x) => x.x[2] < 25);
  const acts: Act[] = [
    { id: "online", label: "Compare offline and online accuracy", lines: [`upgrade-v4 offline (test split) ${pct(off)} · online (new customers, 30 days) ${pct(on)}`, `previous model upgrade-v3 (one row per customer, max_depth 4): offline ${pct(accuracy((x) => treeProb(v3, x), v3Test))} · online ${pct(accuracy((x) => treeProb(v3, x), prod))}`] },
    { id: "selection", label: "Read the model-selection report", lines: [...scored.map((s) => `max_depth ${depthLabel(s.d)}: test accuracy ${pct(s.acc)}${s === chosen ? "  ← selected" : ""}`), "selection rule: highest accuracy on the test split"] },
    { id: "overlap", label: "Check whether test rows belong to customers seen in training", lines: [`${pct(overlap)} of test rows come from a customer_id that also appears in the training split`, `split method: ${group ? "by customer_id (GroupShuffleSplit)" : "random 75/25 over rows"}`] },
    { id: "export", label: "Profile the training export", lines: [`${exportRows.length} rows for ${new Set(exportRows.map((x) => x.cid)).size} customers (${perCustomer.toFixed(1)} rows per customer on average)`, "one row per session event; account features repeat across a customer's rows, only event_hour changes", "label (upgraded within 30 days) is per customer, so it repeats too"] },
    { id: "exact", label: "Search for exact duplicate rows", lines: [`${exactDupes} exact duplicate rows (${pct(exactDupes / all.length)}), where two events of a customer happened in the same hour`, "every other repeated row differs in event_hour"] },
    { id: "grouped", label: "Re-score the same candidates with a split by customer_id", lines: DEPTHS.map((d) => { const t = growTree(gTrain, d, Number.isFinite(d) ? 10 : 1); return `max_depth ${depthLabel(d)}: held-out-customer accuracy ${pct(accuracy((x) => treeProb(t, x), gHold))}`; }) },
    { id: "psi", label: "Compare seats distribution: training vs production", lines: [`seats: train mean ${(seatsTrain.reduce((s, v) => s + v, 0) / seatsTrain.length).toFixed(1)}, max ${Math.max(...seatsTrain)} · prod mean ${(seatsProd.reduce((s, v) => s + v, 0) / seatsProd.length).toFixed(1)}, max ${Math.max(...seatsProd)}  ← shift`, "usage_hours_30d, tenure_months, event_hour: stable"] },
    { id: "slices", label: "Break online accuracy down by plan", lines: [`enterprise (seats ≥ 25): ${ent.length} customers, accuracy ${pct(accuracy(p, ent))}`, `self-serve (seats < 25): ${small.length} customers, accuracy ${pct(accuracy(p, small))}`] },
    { id: "deploys", label: "Read the change log", lines: ["Sep 26 data-eng: upgrade export switched from one row per customer to one row per session event (richer data)", "Sep 30 ml: upgrade-v4 picked by test accuracy over a max_depth grid; offline jump → shipped", "Oct 03 sales: enterprise plan launched (25+ seats)"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: train.length, features: UP_FEATURES, offlineAcc: off, onlineAcc: on, servingOrder: UP_FEATURES, prodRows: prod.length, version: "upgrade-v4" }),
    metrics: [...healthy, { label: "Offline accuracy", value: pct(off), status: "ok", kind: "quality" }, { label: "Online accuracy", value: pct(on), status: on >= 0.76 ? "ok" : "error", kind: "quality" }, { label: "Selected max_depth", value: depthLabel(chosen.d), status: "ok", kind: "quality" }],
    transcript: [{ label: "Model card", text: `upgrade-v4: offline accuracy ${pct(off)} (selected max_depth ${depthLabel(chosen.d)})` }, { label: "Sales ops", text: `the upsell list was right for only ${pct(on)} of new customers`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["online", "selection", "overlap", "export", "grouped", "deploys"]),
  };
}

// ======================= Advanced: unscaled features under gradient descent =======================
function creditData(seed: number, n: number): Dataset {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => {
    const income = Math.max(12000, Math.round(65000 + r.gaussian() * 22000)), debt = +Math.min(0.95, Math.max(0.02, 0.35 + r.gaussian() * 0.15)).toFixed(2), late = Math.min(6, Math.floor(r.next() * r.next() * 7));
    const z = -0.5 * ((income - 65000) / 22000) + 1.6 * ((debt - 0.35) / 0.15) + 1.0 * ((late - 1.5) / 1.2) - 0.9;
    return { x: [income, debt, late], y: r.next() < sigmoid(z) ? 1 : 0 };
  });
  return { features: ["income_usd", "debt_ratio", "late_payments_12m"], rows };
}
const LR_UNSCALED = 1e-9;
function scaleRun(fixId?: string): Outcome {
  const keep = fixId === "f-drop" ? [0] : [0, 1, 2];
  const pick = (ds: Dataset): Dataset => ({ features: keep.map((j) => ds.features[j]), rows: ds.rows.map((r) => ({ ...r, x: keep.map((j) => r.x[j]) })) });
  const tr = pick(creditData(91, 600)), te = pick(creditData(92, 600)), pr = pick(creditData(93, 600));
  const scale = fixId === "f-scale";
  const lr = scale ? 0.5 : fixId === "f-lr" ? LR_UNSCALED * 100 : LR_UNSCALED, epochs = fixId === "f-epochs" ? 4000 : 400;
  const m = fitGD(tr.rows, { loss: "log", lr, epochs, scale });
  const evalOn = (rows: Row[]) => { const preds = rows.map((r) => (m.predict(r.x) >= 0.5 ? 1 : 0)); const tp = rows.filter((r, i) => r.y && preds[i]).length, pos = rows.filter((r) => r.y).length; return { acc: rows.filter((r, i) => r.y === preds[i]).length / rows.length, recall: tp / Math.max(1, pos), flagged: preds.filter(Boolean).length / rows.length }; };
  const off = evalOn(te.rows), on = evalOn(pr.rows);
  const resolved = on.acc >= 0.78 && on.recall >= 0.5;
  const base = pr.rows.filter((r) => !r.y).length / pr.rows.length, prior = tr.rows.filter((r) => r.y).length / tr.rows.length;
  const full = creditData(91, 600);
  const sweep = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6].map((a) => { const x = fitGD(full.rows, { loss: "log", lr: a, epochs: 400, scale: false }); return `lr ${a.toExponential(0)}: final log loss ${num(x.curve[399], 3)} (min ${num(Math.min(...x.curve), 3)}), debt_ratio weight ${num(x.w[1], 6)}`; });
  const lib = trainLogistic(full);
  const at = [1, 10, 100, epochs].map((e) => `epoch ${e}: log loss ${num(m.curve[e - 1], 4)}`);
  const acts: Act[] = [
    { id: "online", label: "Review online evaluation", lines: [`online accuracy ${pct(on.acc)} · defaults caught ${pct(on.recall)} · flagged ${pct(on.flagged)} of applicants`, `always predicting "no default" would score ${pct(base)}`, `offline test accuracy ${pct(off.acc)}`] },
    { id: "curve", label: "Plot the training loss per epoch", lines: [...at, `log loss of always predicting the base rate: ${num(-(prior * Math.log(prior) + (1 - prior) * Math.log(1 - prior)), 4)}`] },
    { id: "weights", label: "Inspect learned weights (raw units)", lines: tr.features.map((f, j) => `${f}: ${num(m.w[j], 9)} per unit · ${num(m.w[j] * sdOf(col(tr.rows, j)), 3)} per std${scale ? " (standardized inputs)" : ""}`).concat([`bias: ${num(m.b, 6)}`]) },
    { id: "scales", label: "Compare feature scales", lines: tr.features.map((f, j) => { const c = col(tr.rows, j); return `${f}: mean ${meanOf(tr, j).toFixed(2)}, std ${sdOf(c).toFixed(2)}, range ${Math.min(...c)}–${Math.max(...c)}`; }).concat([`inputs reach the trainer ${scale ? "standardized" : "as raw values (no scaler step)"}`]) },
    { id: "grads", label: "Log the gradient per weight on the first step", lines: tr.features.map((f, j) => `∂loss/∂${f}: ${num(m.grad0[j], 6)}`) },
    { id: "lrsweep", label: "Sweep the learning rate (400 epochs, current pipeline)", lines: sweep },
    { id: "bureau", label: "Check the credit-bureau feed for debt_ratio and late_payments", lines: ["feed delivered daily, 0% nulls, values in expected ranges", `default rate when late_payments_12m ≥ 3: ${pct(full.rows.filter((r) => r.x[2] >= 3 && r.y).length / Math.max(1, full.rows.filter((r) => r.x[2] >= 3).length))} vs ${pct(full.rows.filter((r) => r.x[2] < 3 && r.y).length / Math.max(1, full.rows.filter((r) => r.x[2] < 3).length))} otherwise`] },
    { id: "library", label: "Train the old library model on the same data", lines: [`sklearn-style pipeline (StandardScaler + LogisticRegression): online accuracy ${pct(evaluateModel(lib, creditData(93, 600).rows).accuracy)}`, ...weightsLines(lib)] },
    { id: "deploys", label: "Read the change log", lines: ["Oct 01 data-eng: debt_ratio and late_payments_12m added from the credit-bureau feed", `Oct 04 ml: replaced library trainer with in-house GPU gradient-descent trainer; lr tuned down to ${LR_UNSCALED} after higher values made the loss jump around`, "Oct 06 risk: credit-v9 promoted"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: off.acc, onlineAcc: on.acc, servingOrder: tr.features, prodRows: pr.rows.length, version: "credit-v9" }),
    metrics: [...healthy, { label: "Online accuracy", value: pct(on.acc), status: on.acc >= 0.78 ? "ok" : "error", kind: "quality" }, { label: "Defaults caught", value: pct(on.recall), status: on.recall >= 0.5 ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Credit risk", text: `credit-v9 flagged ${pct(on.flagged)} of applicants and caught ${pct(on.recall)} of the loans that defaulted`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["weights", "scales", "grads", "lrsweep", "deploys"]),
  };
}

export const ML_MORE_SCENARIOS: Scenario[] = [
  {
    id: "ml-lr-diverge", title: "Delivery ETAs come out as absurd numbers", system: "ML", difficulty: "Beginner", relatedTopic: "gradient-descent",
    symptom: "Retrained ETA model predicts nonsense", objective: "Read a training-loss curve and recognize a learning rate that makes gradient descent diverge.",
    incident: "The courier app started showing delivery estimates like \"-3e+28 minutes\" right after this week's ETA model retrain. Nothing about the data or the app changed.",
    architecture: ML_ARCH, run: lrRun,
    hypotheses: [
      { id: "h-lr", text: `The learning rate (${LR_BAD}) is too high: every gradient step overshoots the minimum, so the loss grows instead of shrinking.`, isRootCause: true, evidenceIds: ["curve", "preds", "lrsweep", "deploys"], explanation: "Training MSE rises every epoch to astronomically large values, smaller learning rates converge on the same data, and the change log shows the learning rate was raised." },
      { id: "h-short", text: "The model wasn't trained long enough to converge.", isRootCause: false, evidenceIds: ["curve"], explanation: "The loss isn't slowly falling; it is getting worse every epoch. More epochs make it worse." },
      { id: "h-data", text: "The training data contains corrupted or extreme values.", isRootCause: false, evidenceIds: ["data"], explanation: "All features and labels are in normal ranges with no missing values." },
      { id: "h-scale", text: "The features aren't scaled.", isRootCause: false, evidenceIds: ["scaling"], explanation: "The trainer standardizes every feature to mean 0, std 1." },
      { id: "h-simple", text: "A linear model is too simple for delivery times.", isRootCause: false, evidenceIds: ["lrsweep"], explanation: "The same linear model with lr 0.1 fits well. A too-simple model would plateau at a high loss, not explode." },
    ],
    fixes: [
      { id: "f-lr", label: "Set the learning rate back to 0.1 and retrain", explanation: "Each step now moves part of the way to the minimum, and the loss falls smoothly." },
      { id: "f-epochs", label: "Train 4× longer (800 epochs)", explanation: "Each extra epoch multiplies the error again; the weights overflow." },
      { id: "f-features", label: "Use a bigger model: add the weekday feature", explanation: "More weights, same overshooting steps." },
      { id: "f-data", label: "Retrain on 4× more deliveries", explanation: "Full-batch gradient steps have the same size regardless of row count." },
    ],
    verifications: [
      { id: "v-curve", label: "Check that training loss decreases and flattens, then confirm test MAE is under 6 minutes; fail the training job if the loss rises or becomes NaN", sound: true, explanation: "Directly tests convergence and stops a diverged model from being registered again." },
      { id: "v-uptime", label: "Confirm the model server returns HTTP 200", sound: false, explanation: "It returned 200 with nonsense predictions throughout." },
      { id: "v-schema", label: "Re-run the schema check", sound: false, explanation: "Schema was never the problem." },
    ],
    rootCause: `The ETA retrain raised the learning rate from 0.1 to ${LR_BAD}. On standardized features the curvature of the squared-error loss is about 2, so any learning rate around 1 or above overshoots: each step lands further from the minimum, the loss explodes, and the registered weights produce absurd ETAs.`,
  },
  {
    id: "ml-deep-tree", title: "Coupon model: 100% on training, coin-flip in production", system: "ML", difficulty: "Beginner", relatedTopic: "decision-trees",
    symptom: "Perfect training accuracy, poor real accuracy", objective: "Spot overfitting from the train/test gap and limit tree complexity.",
    incident: "The model card for coupon-v2 shows a perfect training score, so marketing switched to it. Two weeks later, coupons are going to customers who don't redeem them and skipping those who would.",
    architecture: ML_ARCH, run: treeRun,
    hypotheses: [
      { id: "h-depth", text: "The tree has no depth limit, so it memorized the training rows (including their noise) instead of learning the general pattern.", isRootCause: true, evidenceIds: ["gap", "tree", "depthsweep", "deploys"], explanation: "Training accuracy is 100% with roughly one row per leaf, test accuracy is far lower, a shallow tree validates better, and the config cleanup removed max_depth." },
      { id: "h-drift", text: "Customer behaviour changed after launch.", isRootCause: false, evidenceIds: ["dist", "offline"], explanation: "Production feature distributions match training, and the held-out test set (same period as training) is just as bad." },
      { id: "h-imbalance", text: "Too few redeemers in the data (class imbalance).", isRootCause: false, evidenceIds: ["classes"], explanation: "Redeemers are a large share of rows; the classes are reasonably balanced." },
      { id: "h-small", text: "The model is too simple and needs more features.", isRootCause: false, evidenceIds: ["gap"], explanation: "A too-simple model scores poorly on training data too. This one is perfect on training data, the opposite problem." },
      { id: "h-serving", text: "Serving sends features in the wrong order.", isRootCause: false, evidenceIds: ["serving"], explanation: "Payload matches the training schema." },
    ],
    fixes: [
      { id: "f-depth", label: "Restore max_depth 4 and min_samples_leaf 15, then retrain", explanation: "Each leaf now averages many customers, so it captures the pattern instead of individual noise." },
      { id: "f-data", label: "Collect twice as much training data and retrain the same unlimited tree", explanation: "An unlimited tree memorizes 1,000 rows as easily as 500." },
      { id: "f-features", label: "Use a bigger model: add zip_code and signup_weekday", explanation: "More features give the tree more ways to memorize noise." },
    ],
    verifications: [
      { id: "v-val", label: "Choose max_depth by validation accuracy, and require the train/validation gap to stay under 10 points before registering", sound: true, explanation: "Measures generalization instead of memorization." },
      { id: "v-train", label: "Confirm training accuracy is still 100%", sound: false, explanation: "That perfect score was the warning sign." },
      { id: "v-latency", label: "Check inference latency", sound: false, explanation: "Unrelated to prediction quality." },
    ],
    rootCause: "A config cleanup dropped max_depth and min_samples_leaf, so the tree kept splitting until every leaf held a single training row. It memorized noise (100% on training), which doesn't carry over to new customers.",
  },
  {
    id: "ml-sentinel-999", title: "Loan model got worse after the warehouse migration", system: "ML", difficulty: "Beginner", relatedTopic: "feature-engineering",
    symptom: "Retrained risk model misses defaults", objective: "Find a placeholder value that the model treats as a real number.",
    incident: "After the data warehouse migration, loan-risk-v5 was retrained on the new export. Since then, noticeably more risky loans are being approved. Every pipeline job reports success.",
    architecture: ML_ARCH, run: sentinelRun,
    hypotheses: [
      { id: "h-sentinel", text: `Unknown years_at_job is now ${SENTINEL}; the model treats it as a real number, which wrecks the feature's scale and hides its signal.`, isRootCause: true, evidenceIds: ["summary", "counts", "weights", "outliers", "deploys"], explanation: `About a fifth of rows hold exactly ${SENTINEL}; the feature's mean is negative and its std is in the hundreds, so real job tenure barely moves the standardized value and its weight shrank. The 3σ check misses it because the sentinel inflates σ.` },
      { id: "h-drift", text: "Applicants changed after the migration (data drift).", isRootCause: false, evidenceIds: ["drift"], explanation: "Training and production means match; both contain the same placeholder." },
      { id: "h-short", text: "The model needs to train longer.", isRootCause: false, evidenceIds: ["weights"], explanation: "The model converged; it simply can't use a feature whose real values are squeezed into a tiny range." },
      { id: "h-outlier", text: "There are no data-quality issues: the outlier check passed.", isRootCause: false, evidenceIds: ["outliers", "counts"], explanation: `The check uses mean ± 3σ computed with the ${SENTINEL} values included, so the range stretches to cover them.` },
      { id: "h-serving", text: "Serving sends features in the wrong order.", isRootCause: false, evidenceIds: ["serving"], explanation: "Payload matches training order; the problem is in the data itself." },
    ],
    fixes: [
      { id: "f-sentinel", label: `Treat ${SENTINEL} as missing: impute the median and add a years_missing flag, then retrain`, explanation: "Real tenure gets its full range back and missingness becomes its own signal." },
      { id: "f-clip", label: "Clip values beyond 3 standard deviations, then retrain", explanation: `The 3σ range already includes ${SENTINEL}, so nothing is clipped.` },
      { id: "f-epochs", label: "Train 4× longer", explanation: "Same squeezed feature, same weights." },
      { id: "f-data", label: "Retrain on 3× more loans from the new export", explanation: "More rows with the same placeholder." },
    ],
    verifications: [
      { id: "v-contract", label: `Add a data check that fails on known sentinel values (${SENTINEL}, 9999, empty strings) in numeric columns, then confirm online accuracy and recall are back to v4 levels`, sound: true, explanation: "Catches the placeholder at the source and measures the outcome that regressed." },
      { id: "v-3sd", label: "Re-run the 3-standard-deviation outlier check", sound: false, explanation: "It passed during the incident; the sentinel distorts the very σ it relies on." },
      { id: "v-jobs", label: "Confirm every pipeline job succeeded", sound: false, explanation: "They all succeeded throughout." },
    ],
    rootCause: `The warehouse migration started exporting unknown numeric fields as ${SENTINEL} instead of NULL, so the pipeline's impute step never fired. To the model, ${SENTINEL} is a real tenure: it drags the mean down and inflates the std, so real job tenure (0–25 years) barely changes the standardized input and the model loses its strongest signal.`,
  },
  {
    id: "ml-group-split", title: "Upsell model jumped offline, sank online", system: "ML", difficulty: "Advanced", relatedTopic: "train-validation-and-test-splits",
    symptom: "New model wins offline by a wide margin, loses online", objective: "Recognize evaluation contamination from repeated entities across a random split, and see how it steers model selection, despite a real but harmless drift.",
    incident: "upgrade-v4 beat upgrade-v3 offline by a wide margin and was shipped. A month later, its upsell list for new customers is worse than v3's. The same week, sales launched an enterprise plan and the seats feature shifted noticeably.",
    architecture: ML_ARCH, run: dupRun,
    hypotheses: [
      { id: "h-group", text: "The new event-level export repeats each customer across many rows; a random row split puts the same customers in train and test, so the test score rewards memorization and model selection picked an unlimited-depth tree.", isRootCause: true, evidenceIds: ["online", "selection", "overlap", "export", "grouped", "deploys"], explanation: "Most test rows belong to customers already in training; the deepest tree wins on that split but loses when scored on held-out customers, and the export change preceded the jump." },
      { id: "h-drift", text: "The enterprise plan shifted the seats distribution (data drift).", isRootCause: false, evidenceIds: ["psi", "slices"], explanation: "Seats did shift, but accuracy is just as poor for self-serve customers inside the training range. The drift is real but not the cause." },
      { id: "h-dupes", text: "Exact duplicate rows leaked from training into test.", isRootCause: false, evidenceIds: ["exact", "overlap"], explanation: "Only a handful of rows are exact duplicates (event_hour usually differs), yet most test rows share a customer with training. The leak is by entity, not by identical row." },
      { id: "h-overfit-only", text: "The tree is simply too deep; it's an ordinary overfit.", isRootCause: false, evidenceIds: ["selection", "grouped"], explanation: "Partly true, but the question is why the pipeline chose it: on the contaminated split the deep tree scored best. Capping depth alone still leaves an evaluation that can't be trusted for the next decision." },
      { id: "h-threshold", text: "The decision threshold needs retuning for the new plan mix.", isRootCause: false, evidenceIds: ["slices"], explanation: "Both slices fail similarly; the scores are memorized, not shifted." },
    ],
    fixes: [
      { id: "f-group", label: "Split train/test by customer_id and re-run model selection", explanation: "The test set now contains only unseen customers, so selection favors a tree that generalizes, and the offline number predicts online." },
      { id: "f-retrain", label: "Retrain on the last month of events, including enterprise customers", explanation: "Fixes the harmless drift, but the same random row split picks the same memorizing tree." },
      { id: "f-dedupe", label: "Drop exact duplicate rows before splitting", explanation: "Only the few events that share an hour are removed; the same customers still sit on both sides of the split." },
    ],
    verifications: [
      { id: "v-group", label: "Use grouped (by customer) or time-based holdouts for every model comparison, assert zero customer_id overlap between splits, and confirm offline ≈ online on the next cohort", sound: true, explanation: "Makes the offline score measure the situation the model faces: customers it has never seen." },
      { id: "v-offline", label: "Confirm upgrade-v4 still wins on the existing test split", sound: false, explanation: "That split is contaminated; winning on it was the symptom." },
      { id: "v-psi", label: "Add a PSI alert on seats", sound: false, explanation: "Useful monitoring, but it targets the red herring." },
    ],
    rootCause: "The export moved to one row per session event, so each customer appears several times with identical account features and label. A random row split put most customers on both sides, the test score rewarded memorizing them, and model selection picked an unlimited-depth tree. On new customers that tree is worse than the old one. The enterprise-plan drift happened at the same time but isn't what broke it.",
  },
  {
    id: "ml-unscaled-gd", title: "New trainer ignores the bureau features", system: "ML", difficulty: "Advanced", relatedTopic: "gradient-descent",
    symptom: "Credit model catches almost no defaults", objective: "Diagnose ill-conditioning: why no learning rate works when feature scales differ by orders of magnitude.",
    incident: "credit-v9 was the first model trained with the in-house gradient-descent trainer, and the first to use the new credit-bureau features. It approves almost everyone. The bureau features ended up with near-zero weights, and the team is debating whether the feed is useless or the learning rate is wrong.",
    architecture: ML_ARCH, run: scaleRun,
    hypotheses: [
      { id: "h-scale", text: "Features are fed unscaled: income (tens of thousands) dominates the gradient, so a learning rate small enough to keep income stable is far too small for debt_ratio and late_payments to learn.", isRootCause: true, evidenceIds: ["weights", "scales", "grads", "lrsweep", "deploys"], explanation: "Feature scales differ by ~5 orders of magnitude, the first-step gradient on income dwarfs the others, and the sweep shows every learning rate either stalls with zero bureau weights or makes the loss jump. The new trainer has no scaler step." },
      { id: "h-feed", text: "The bureau features carry no signal.", isRootCause: false, evidenceIds: ["bureau", "library"], explanation: "Default rates differ sharply by late payments, and the library pipeline gives the bureau features large weights on the same data." },
      { id: "h-lr", text: "The learning rate is simply too low; raise it.", isRootCause: false, evidenceIds: ["lrsweep"], explanation: "Raising it makes the income weight overshoot and the loss jump before the bureau weights move. There's no single good learning rate for these scales." },
      { id: "h-short", text: "The model needs more epochs.", isRootCause: false, evidenceIds: ["curve", "grads"], explanation: "The loss is flat; at this learning rate the bureau weights move about 1e-10 per epoch. Ten times more epochs barely changes anything." },
      { id: "h-imbalance", text: "Defaults are rare, so the model predicts the majority class.", isRootCause: false, evidenceIds: ["online", "library"], explanation: "The same data and class balance give a useful model with the library trainer." },
    ],
    fixes: [
      { id: "f-scale", label: "Standardize features (fit the scaler on training data, ship it with the model) and retrain with lr 0.5", explanation: "All weights now see gradients on the same scale and converge together." },
      { id: "f-lr", label: "Raise the learning rate 100×", explanation: "The income weight overshoots and oscillates; the loss jumps around." },
      { id: "f-epochs", label: "Train 10× longer", explanation: "The bureau weights still move by tiny amounts per epoch." },
      { id: "f-drop", label: "Drop the bureau features since their weights are ~0", explanation: "Throws away the signal the trainer couldn't reach." },
    ],
    verifications: [
      { id: "v-parity", label: "Train the old library pipeline and the new trainer on the same data and require matching weights (on standardized inputs) and online recall before promoting", sound: true, explanation: "A parity test catches any trainer that fails to converge, and recall measures the outcome that broke." },
      { id: "v-loss", label: "Confirm the training loss no longer changes between epochs", sound: false, explanation: "A flat loss is exactly what stalled training looks like." },
      { id: "v-feed", label: "Remove the bureau feed and confirm accuracy is unchanged", sound: false, explanation: "Of course it's unchanged; the model never learned to use it." },
    ],
    rootCause: `The in-house trainer runs gradient descent on raw inputs. income_usd is ~5 orders of magnitude larger than debt_ratio and late_payments_12m, so the loss surface is extremely stretched: any learning rate above ~1e-9 makes the income weight overshoot, and at ${LR_UNSCALED} the bureau weights barely move in 400 epochs. Standardizing the inputs makes one learning rate work for every weight.`,
  },
];