import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type StageId = "commit" | "unit" | "data" | "train" | "eval" | "canary" | "promote";
export const STAGES: { id: StageId; label: string }[] = [
  { id: "commit", label: "Commit" }, { id: "unit", label: "Unit tests" }, { id: "data", label: "Data validation" }, { id: "train", label: "Train" },
  { id: "eval", label: "Evaluate + gates" }, { id: "canary", label: "Canary deploy" }, { id: "promote", label: "Promote" },
];

export type Preset = "healthy" | "schema" | "drift" | "slice" | "canary";
export const PRESETS: Record<Preset, { label: string; commit: string; note: string }> = {
  healthy: { label: "Healthy change", commit: "feat: add tenure buckets", note: "Everything is within thresholds; the candidate is promoted." },
  schema: { label: "Upstream schema change", commit: "chore: bump upstream export to v3", note: "The export renamed income → annual_income. Unit tests pass; data validation catches it." },
  drift: { label: "Input drift in income", commit: "chore: retrain on October data", note: "The income distribution shifted (a pricing change). PSI exceeds the threshold." },
  slice: { label: "Regression on one slice", commit: "feat: reweight loss toward young users", note: "Overall accuracy rises, but customers aged 65+ lose 9 points. Only the slice gate notices." },
  canary: { label: "Canary error spike", commit: "perf: switch inference to fp16", note: "Offline metrics are fine, but the new serving path errors on real traffic. The canary rolls back." },
};

export type Thresholds = { minAccDelta: number; maxP95: number; maxSliceDrop: number; maxPsi: number; maxCanaryErr: number; canaryPct: number; sliceGate: boolean };
export const DEFAULT_THRESHOLDS: Thresholds = { minAccDelta: -0.005, maxP95: 120, maxSliceDrop: 0.03, maxPsi: 0.2, maxCanaryErr: 0.02, canaryPct: 10, sliceGate: true };

export type Check = { id: string; label: string; measured: number | null; threshold: number; op: "<=" | ">="; unit: string; digits: number; pass: boolean; scale: [number, number]; disabled?: boolean; detail?: string };
export type Status = "pending" | "active" | "ok" | "warning" | "error" | "skipped";

/** Population stability index between two binned distributions: Σ (a − e)·ln(a / e). */
export function psi(expected: number[], actual: number[]) {
  return expected.reduce((s, e, i) => s + (actual[i] - e) * Math.log(actual[i] / e), 0);
}
export function percentile(xs: number[], p: number) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil(p * s.length) - 1))];
}

export const EXPECTED_COLUMNS = ["age", "income", "tenure", "plan", "churned"];
const BASE_BINS: Record<string, number[]> = { age: [0.2, 0.25, 0.25, 0.2, 0.1], income: [0.15, 0.25, 0.3, 0.2, 0.1], tenure: [0.3, 0.25, 0.2, 0.15, 0.1] };
const NEW_BINS: Record<string, number[]> = { age: [0.19, 0.25, 0.26, 0.2, 0.1], income: [0.14, 0.25, 0.31, 0.2, 0.1], tenure: [0.28, 0.26, 0.2, 0.16, 0.1] };
const DRIFT_INCOME = [0.05, 0.12, 0.25, 0.3, 0.28];

export type Slice = { name: string; n: number; base: number; cand: number }; // correct counts
const BASE_SLICES = [{ name: "age < 30", n: 400, base: 344 }, { name: "age 30–64", n: 1200, base: 1056 }, { name: "age 65+", n: 300, base: 255 }];
const CAND: Record<"good" | "slice", number[]> = { good: [352, 1074, 258], slice: [360, 1092, 228] };

export type DataDetail = { columns: string[]; missing: string[]; extra: string[]; psi: { feature: string; base: number[]; cur: number[] | null; value: number | null }[] };
export type EvalDetail = { slices: Slice[]; baseAcc: number; candAcc: number; latencies: number[]; p95: number };
export type CanaryMinute = { minute: number; requests: number; errors: number; cumRate: number; baseRate: number };
export type CanaryDetail = { minutes: CanaryMinute[]; rollbackAt: number | null; minRequests: number };

export type CIState = {
  active: StageId | "result";
  statuses: Record<StageId, Status>;
  checks: Check[]; // checks of the active stage
  failedAt?: StageId;
  outcome?: "promoted" | "promoted-with-regression" | "blocked" | "rolled-back";
  data: DataDetail; evalD: EvalDetail; canary: CanaryDetail; loss: number[];
};

/** Display a check value in its unit: fractions shown as points or percent. */
export function fmtCheck(c: Pick<Check, "unit" | "digits">, v: number | null) {
  if (v === null) return "n/a";
  if (c.unit === "pts") return `${v > 0 ? "+" : ""}${(v * 100).toFixed(c.digits)} pts`;
  if (c.unit === "%") return `${(v * 100).toFixed(c.digits)}%`;
  return `${v.toFixed(c.digits)}${c.unit ? ` ${c.unit}` : ""}`;
}

const gate = (c: Omit<Check, "pass">): Check => ({ ...c, pass: c.disabled || c.measured === null ? !!c.disabled : c.op === "<=" ? c.measured <= c.threshold + 1e-12 : c.measured >= c.threshold - 1e-12 });

export function runPipeline(preset: Preset, th: Thresholds = DEFAULT_THRESHOLDS): SimulationStep<CIState>[] {
  const t = { ...th, canaryPct: clamp(Math.round(th.canaryPct), 1, 100) };

  // ── data validation ──
  const columns = preset === "schema" ? ["age", "annual_income", "tenure", "plan", "churned"] : EXPECTED_COLUMNS;
  const missing = EXPECTED_COLUMNS.filter((c) => !columns.includes(c));
  const extra = columns.filter((c) => !EXPECTED_COLUMNS.includes(c));
  const psis = Object.keys(BASE_BINS).map((f) => {
    const cur = missing.includes(f) ? null : f === "income" && preset === "drift" ? DRIFT_INCOME : NEW_BINS[f];
    return { feature: f, base: BASE_BINS[f], cur, value: cur ? psi(BASE_BINS[f], cur) : null };
  });
  const maxPsi = Math.max(...psis.map((p) => p.value ?? 0));
  const data: DataDetail = { columns, missing, extra, psi: psis };
  const dataChecks: Check[] = [
    gate({ id: "schema", label: "missing columns", measured: missing.length, threshold: 0, op: "<=", unit: "", digits: 0, scale: [0, 2], detail: missing.length ? `missing: ${missing.join(", ")}` : "all 5 expected columns present" }),
    gate({ id: "psi", label: "max PSI (drift)", measured: missing.length ? null : maxPsi, threshold: t.maxPsi, op: "<=", unit: "", digits: 3, scale: [0, 0.5], detail: missing.length ? "not computed: schema check failed" : `worst feature: ${psis.reduce((a, b) => ((b.value ?? 0) > (a.value ?? 0) ? b : a)).feature}` }),
  ];

  // ── train ──
  const loss = Array.from({ length: 20 }, (_, e) => Math.round((0.25 + 0.45 * Math.exp(-e / 5)) * 1000) / 1000);
  const trainChecks: Check[] = [gate({ id: "loss", label: "final / initial loss", measured: loss[19] / loss[0], threshold: 0.6, op: "<=", unit: "", digits: 3, scale: [0, 1] })];

  // ── evaluate ──
  const cand = preset === "slice" ? CAND.slice : CAND.good;
  const slices: Slice[] = BASE_SLICES.map((s, i) => ({ ...s, cand: cand[i] }));
  const N = slices.reduce((a, s) => a + s.n, 0);
  const baseAcc = slices.reduce((a, s) => a + s.base, 0) / N;
  const candAcc = slices.reduce((a, s) => a + s.cand, 0) / N;
  const r = rng(17);
  const latencies = Array.from({ length: 200 }, () => Math.round(70 * Math.exp(0.25 * r.gaussian()) * 10) / 10);
  const p95 = percentile(latencies, 0.95);
  const worst = slices.reduce((a, s) => (s.base / s.n - s.cand / s.n > a.base / a.n - a.cand / a.n ? s : a));
  const sliceDrop = worst.base / worst.n - worst.cand / worst.n;
  const evalD: EvalDetail = { slices, baseAcc, candAcc, latencies, p95 };
  const evalChecks: Check[] = [
    gate({ id: "acc", label: "accuracy vs baseline", measured: candAcc - baseAcc, threshold: t.minAccDelta, op: ">=", unit: "pts", digits: 1, scale: [-0.05, 0.05], detail: `${candAcc.toFixed(3)} vs ${baseAcc.toFixed(3)}` }),
    gate({ id: "lat", label: "p95 latency", measured: p95, threshold: t.maxP95, op: "<=", unit: "ms", digits: 1, scale: [0, 200] }),
    gate({ id: "slice", label: "worst slice drop", measured: sliceDrop, threshold: t.maxSliceDrop, op: "<=", unit: "pts", digits: 1, scale: [-0.05, 0.15], disabled: !t.sliceGate, detail: `${worst.name}: ${(worst.base / worst.n).toFixed(3)} → ${(worst.cand / worst.n).toFixed(3)}` }),
  ];

  // ── canary: 1000 req/min, a share goes to the candidate ──
  const cr = rng(29);
  const candErrP = preset === "canary" ? 0.04 : 0.005;
  const perMin = Math.round(1000 * (t.canaryPct / 100));
  const minRequests = 200;
  const minutes: CanaryMinute[] = [];
  let reqs = 0, errs = 0, baseReqs = 0, baseErrs = 0, rollbackAt: number | null = null;
  for (let m = 1; m <= 10 && rollbackAt === null; m++) {
    let e = 0;
    for (let i = 0; i < perMin; i++) if (cr.next() < candErrP) e++;
    let be = 0;
    for (let i = 0; i < 1000 - perMin; i++) if (cr.next() < 0.005) be++;
    reqs += perMin; errs += e; baseReqs += 1000 - perMin; baseErrs += be;
    const cumRate = reqs ? errs / reqs : 0;
    minutes.push({ minute: m, requests: perMin, errors: e, cumRate, baseRate: baseReqs ? baseErrs / baseReqs : 0 });
    if (reqs >= minRequests && cumRate > t.maxCanaryErr) rollbackAt = m;
  }
  // At the end of the window the verdict is final, even if fewer than minRequests arrived.
  if (rollbackAt === null && minutes.at(-1)!.cumRate > t.maxCanaryErr) rollbackAt = minutes.length;
  const canary: CanaryDetail = { minutes, rollbackAt, minRequests };
  const last = minutes.at(-1)!;
  const canaryChecks: Check[] = [
    gate({ id: "err", label: "canary error rate", measured: last.cumRate, threshold: t.maxCanaryErr, op: "<=", unit: "%", digits: 2, scale: [0, 0.06], detail: `${errs} errors / ${reqs} req in ${minutes.length} min` }),
  ];

  // ── assemble steps ──
  const statuses: Record<StageId, Status> = { commit: "pending", unit: "pending", data: "pending", train: "pending", eval: "pending", canary: "pending", promote: "pending" };
  const steps: SimulationStep<CIState>[] = [];
  const common = { data, evalD, canary, loss };
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "pipeline", message, severity, tex });
  const checkEvents = (cs: Check[]) => cs.map((c) => ev(c.id, c.disabled ? `${c.label}: gate disabled (measured ${fmtCheck(c, c.measured)})` : `${c.label}: ${fmtCheck(c, c.measured)} ${c.op === "<=" ? "≤" : "≥"} ${fmtCheck(c, c.threshold)} → ${c.pass ? "pass" : "FAIL"}`, c.disabled ? "warning" : c.pass ? undefined : "error"));
  const push = (id: StageId, title: string, description: string, checks: Check[], events: SimulationEvent[], extra: Partial<CIState> = {}) => {
    const failed = checks.some((c) => !c.pass);
    statuses[id] = failed ? "error" : checks.some((c) => c.disabled) ? "warning" : "ok";
    steps.push({ id, title, description, state: { active: id, statuses: { ...statuses }, checks, ...common, ...extra }, events });
    return failed;
  };
  const finish = (failedAt: StageId, outcome: CIState["outcome"], title: string, description: string, events: SimulationEvent[]) => {
    for (const s of STAGES) if (statuses[s.id] === "pending") statuses[s.id] = "skipped";
    steps.push({ id: "result", title, description, state: { active: "result", statuses: { ...statuses }, checks: [], failedAt, outcome, ...common }, events });
    return steps;
  };

  push("commit", "Commit triggers the pipeline", `"${PRESETS[preset].commit}" is pushed. CI checks out the code and starts the stages in order.`, [], [ev("c", `push: ${PRESETS[preset].commit}`)]);
  push("unit", "Unit tests", "48 fast tests on feature code and the training loop run in seconds.",
    [gate({ id: "tests", label: "tests passed", measured: 48, threshold: 48, op: ">=", unit: "", digits: 0, scale: [0, 48] })], [ev("u", "48/48 tests passed")]);
  if (push("data", "Data validation: schema and drift", missing.length ? `Schema check fails: ${missing.join(", ")} is missing (found ${extra.join(", ")}).` : `Schema matches. Largest drift: PSI ${maxPsi.toFixed(3)} (threshold ${t.maxPsi}).`, dataChecks, checkEvents(dataChecks)))
    return finish("data", "blocked", "Pipeline stopped before training", "Bad data never reaches training, so no model is built from it and production is untouched.", [ev("stop", "pipeline halted at data validation", "error")]);
  push("train", "Train the candidate", `20 epochs; loss falls from ${loss[0]} to ${loss[19]}.`, trainChecks, checkEvents(trainChecks));
  if (push("eval", "Evaluate against the baseline", `Candidate accuracy ${candAcc.toFixed(3)} vs baseline ${baseAcc.toFixed(3)}; p95 ${p95} ms; worst slice ${worst.name} ${sliceDrop >= 0 ? "drops" : "gains"} ${(Math.abs(sliceDrop) * 100).toFixed(1)} pts.`, evalChecks, checkEvents(evalChecks)))
    return finish("eval", "blocked", "Gate blocks the candidate", "The failing gate is reported on the commit. The model is registered but never deployed.", [ev("stop", "candidate rejected by evaluation gates", "error")]);
  if (push("canary", rollbackAt ? `Canary at ${t.canaryPct}%: error rate too high` : `Canary at ${t.canaryPct}% for 10 minutes`,
    rollbackAt ? `After ${rollbackAt} min the candidate's error rate is ${(last.cumRate * 100).toFixed(2)}% (> ${(t.maxCanaryErr * 100).toFixed(1)}%). Automatic rollback.` : `${t.canaryPct}% of live traffic for 10 min: error rate ${(last.cumRate * 100).toFixed(2)}%, baseline ${(last.baseRate * 100).toFixed(2)}%.`,
    canaryChecks, [...checkEvents(canaryChecks), ...(rollbackAt ? [ev("rb", `rollback at minute ${rollbackAt}: traffic returns to the baseline`, "error")] : [])]))
    return finish("canary", "rolled-back", "Canary rolled back", `Only ${t.canaryPct}% of traffic for ${rollbackAt} min saw the bad version: about ${errs} failed requests instead of a full outage.`, [ev("stop", "candidate rolled back; baseline serves 100%", "error")]);
  const regression = !t.sliceGate && evalChecks.find((c) => c.id === "slice")!.measured! > t.maxSliceDrop;
  push("promote", regression ? "Promoted, with a hidden slice regression" : "Promote to 100%",
    regression ? `Every enabled gate passed, so the candidate now serves everyone, including a ${(sliceDrop * 100).toFixed(1)}-pt drop for ${worst.name}.` : "All gates passed. The candidate becomes the new production model and the new baseline.",
    [], [ev("p", "candidate promoted to 100% of traffic", regression ? "warning" : undefined), ...(regression ? [ev("reg", `${worst.name} accuracy ${(worst.base / worst.n).toFixed(3)} → ${(worst.cand / worst.n).toFixed(3)} shipped (slice gate disabled)`, "warning")] : [])],
    { outcome: regression ? "promoted-with-regression" : "promoted" });
  if (regression) steps.at(-1)!.state.statuses.promote = "warning";
  return steps;
}
