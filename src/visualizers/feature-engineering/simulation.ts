import { mean, solveLinear } from "../../lib/math";
import { r2 } from "../../lib/metrics";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Hood = "Riverside" | "Hilltop" | "Downtown";
export type House = { id: number; built: string; sqft: number; hood: Hood; price: number; test: boolean };
export type TransformId = "date" | "log" | "onehot" | "scale" | "interact";
export type FeConfig = { enabled: Record<TransformId, boolean>; leakScaler: boolean; lambda: number; seed: number };

export const N_ROWS = 40, N_TEST = 12;
export const REF_DATE = "2025-01-01";
export const HOODS: Hood[] = ["Riverside", "Hilltop", "Downtown"];
export const TRANSFORMS: { id: TransformId; label: string; short: string }[] = [
  { id: "date", label: "Parse built_date → age", short: "date → age" },
  { id: "log", label: "log(sqft)", short: "log sqft" },
  { id: "onehot", label: "One-hot neighbourhood", short: "one-hot" },
  { id: "scale", label: "Standardise numeric columns", short: "standardise" },
  { id: "interact", label: "Interaction: size × Downtown", short: "interaction" },
];

const pad = (n: number) => String(n).padStart(2, "0");
/** Years between an ISO date string and REF_DATE. Returns NaN for an unparseable string. */
export function ageYears(built: string, ref = REF_DATE) {
  const t = Date.parse(`${built}T00:00:00Z`), r = Date.parse(`${ref}T00:00:00Z`);
  return Number.isFinite(t) ? (r - t) / (365.25 * 24 * 3600 * 1000) : Number.NaN;
}

/**
 * Synthetic houses. The true price (k$) depends on log(size), age, neighbourhood and a Downtown × size
 * interaction, plus Gaussian noise, so every transform has something real to recover.
 */
export function generateHouses(seed: number): House[] {
  const r = rng(seed);
  const rows = Array.from({ length: N_ROWS }, (_, id) => {
    const year = 1960 + Math.floor(r.next() * 64), month = 1 + Math.floor(r.next() * 12), day = 1 + Math.floor(r.next() * 28);
    const built = `${year}-${pad(month)}-${pad(day)}`;
    const sqft = Math.round(Math.exp(Math.log(450) + r.next() * (Math.log(9000) - Math.log(450))) / 10) * 10;
    const u = r.next();
    const hood: Hood = u < 0.4 ? "Riverside" : u < 0.7 ? "Hilltop" : "Downtown";
    const ls = Math.log(sqft / 1000), dt = hood === "Downtown" ? 1 : 0;
    const price = 160 + 150 * ls + (hood === "Hilltop" ? 45 : dt * 110) - 2.5 * ageYears(built) + 110 * ls * dt + 25 * r.gaussian();
    return { id, built, sqft, hood, price: Math.max(40, Math.round(price)), test: false };
  });
  // Seeded shuffle picks the held-out rows.
  const idx = rows.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  idx.slice(0, N_TEST).forEach((i) => (rows[i].test = true));
  return rows;
}

export type ColKind = "keep" | "new" | "dropped" | "changed";
export type Col = { key: string; label: string; text?: string[]; values?: number[]; numeric: boolean; kind: ColKind; scaled?: boolean };

/** Ridge on the training rows: (XᵀX + λ n I′) w = Xᵀy, intercept not penalised. */
export function fitRidge(X: number[][], y: number[], lambda: number) {
  const p = X[0]?.length ?? 0;
  const rows = X.map((r) => [1, ...r]);
  const A = Array.from({ length: p + 1 }, () => new Array<number>(p + 1).fill(0));
  const b = new Array<number>(p + 1).fill(0);
  rows.forEach((r, i) => r.forEach((v, a) => { b[a] += v * y[i]; r.forEach((u, c) => (A[a][c] += v * u)); }));
  for (let a = 1; a <= p; a++) A[a][a] += Math.max(lambda, 1e-9) * X.length;
  A[0][0] += 1e-9;
  const w = solveLinear(A, b);
  return { w, predict: (x: number[]) => w[0] + x.reduce((s, v, a) => s + v * w[a + 1], 0) };
}

export type Scaler = { key: string; mu: number; sd: number; muTrain: number; sdTrain: number };
export type StageId = "raw" | TransformId | "summary";
export type FeState = {
  stage: StageId; headline: string; sub: string; on: boolean;
  cols: Col[]; features: string[]; weights: number[]; preds: number[]; r2Test: number; r2Train: number; prevR2: number; scalers: Scaler[];
};

const std = (xs: number[]) => { const m = mean(xs); return Math.sqrt(mean(xs.map((v) => (v - m) ** 2))); };

export function simulate(cfg: FeConfig) {
  const houses = generateHouses(cfg.seed);
  const y = houses.map((h) => h.price);
  const trainIdx = houses.flatMap((h, i) => (h.test ? [] : [i])), testIdx = houses.flatMap((h, i) => (h.test ? [i] : []));
  // Working columns, mutated stage by stage; each step stores a structured clone.
  let cols: Col[] = [
    { key: "built_date", label: "built_date", text: houses.map((h) => h.built), numeric: false, kind: "keep" },
    { key: "sqft", label: "sqft", values: houses.map((h) => h.sqft), numeric: true, kind: "keep" },
    { key: "neighbourhood", label: "neighbourhood", text: houses.map((h) => h.hood), numeric: false, kind: "keep" },
  ];
  const steps: SimulationStep<FeState>[] = [];
  let prevR2 = Number.NaN;
  let scalers: Scaler[] = [];

  const fit = () => {
    const feats = cols.filter((c) => c.numeric && c.kind !== "dropped");
    const X = houses.map((_, i) => feats.map((c) => c.values![i]));
    const model = fitRidge(trainIdx.map((i) => X[i]), trainIdx.map((i) => y[i]), cfg.lambda);
    const preds = X.map(model.predict);
    return {
      features: feats.map((c) => c.label), weights: model.w, preds,
      r2Test: r2(testIdx.map((i) => y[i]), testIdx.map((i) => preds[i])),
      r2Train: r2(trainIdx.map((i) => y[i]), trainIdx.map((i) => preds[i])),
    };
  };
  const push = (stage: StageId, title: string, description: string, headline: string, sub: string, on: boolean, events: SimulationEvent[]) => {
    const f = fit();
    steps.push({ id: stage, title, description, state: { stage, headline, sub, on, cols: structuredClone(cols), ...f, prevR2, scalers: structuredClone(scalers) }, events: [
      ...events,
      { id: `${stage}-r2`, type: "metric", message: `refit on ${trainIdx.length} rows: held-out R² = ${f.r2Test.toFixed(3)}${Number.isFinite(prevR2) ? ` (${f.r2Test - prevR2 >= 0 ? "+" : ""}${(f.r2Test - prevR2).toFixed(3)})` : ""}`, severity: Number.isFinite(prevR2) && f.r2Test < prevR2 - 0.005 ? "warning" : undefined },
    ] });
    prevR2 = f.r2Test;
    // Dropped columns leave; new/changed columns become ordinary for the next step.
    cols = cols.filter((c) => c.kind !== "dropped").map((c) => ({ ...c, kind: "keep" }));
  };
  const col = (key: string) => cols.find((c) => c.key === key);
  const off = (t: TransformId) => !cfg.enabled[t];
  const skipped = (name: string) => `Skipped (toggled off): the table and model are unchanged. Turn "${name}" back on to see its effect.`;

  push("raw", "Raw table: only sqft is usable", `A linear model needs numbers. built_date is a string and neighbourhood is a category, so the baseline model can only use sqft. Fit on ${trainIdx.length} training rows, scored on ${testIdx.length} held-out rows.`,
    "Raw data: the model can only use sqft", "Dates and category names are not numbers, so a linear model cannot use them yet.", true,
    [{ id: "raw-cols", type: "schema", message: "usable numeric columns: sqft; ignored: built_date (string), neighbourhood (category)" }]);

  // 1. date → age
  if (off("date")) push("date", "Parse built_date (skipped)", skipped("Parse built_date → age"), "Skipped: built_date stays an unusable string", "Toggle it on in Parameters to let the model see each house's age.", false, []);
  else {
    col("built_date")!.kind = "dropped";
    cols.splice(1, 0, { key: "age", label: "age_years", values: houses.map((h) => Math.round(ageYears(h.built) * 10) / 10), numeric: true, kind: "new" });
    push("date", "Parse built_date → age_years", `Each date string is parsed and turned into age at ${REF_DATE}: e.g. ${houses[0].built} → ${ageYears(houses[0].built).toFixed(1)} years. Older houses sell for less, and now the model can see it.`,
      "Parse the date string into a number: age in years", `"${houses[0].built}" becomes ${ageYears(houses[0].built).toFixed(1)} years old. The string column is dropped.`, true,
      [{ id: "age", type: "transform", message: `age = (${REF_DATE} − built_date) / 365.25 days` }]);
  }
  // 2. log(sqft)
  if (off("log")) push("log", "log(sqft) (skipped)", skipped("log(sqft)"), "Skipped: sqft stays on its raw scale", "Price grows with the log of size here, so a straight line in raw sqft bends the wrong way.", false, []);
  else {
    col("sqft")!.kind = "dropped";
    const at = cols.findIndex((c) => c.key === "sqft");
    cols.splice(at + 1, 0, { key: "log_sqft", label: "log_sqft", values: houses.map((h) => Math.log(h.sqft)), numeric: true, kind: "new" });
    push("log", "Replace sqft with log(sqft)", `Doubling size adds a roughly constant amount to price here, so price is linear in log(sqft), not sqft. ${houses[0].sqft} sqft → ln = ${Math.log(houses[0].sqft).toFixed(3)}.`,
      "Take the log of size so the relationship becomes a straight line", "Big houses stop dominating: 450 sqft → 6.11, 9,000 sqft → 9.10.", true,
      [{ id: "log", type: "transform", message: `log_sqft = ln(sqft); range ${Math.log(Math.min(...houses.map((h) => h.sqft))).toFixed(2)}–${Math.log(Math.max(...houses.map((h) => h.sqft))).toFixed(2)}` }]);
  }
  // 3. one-hot
  if (off("onehot")) push("onehot", "One-hot neighbourhood (skipped)", skipped("One-hot neighbourhood"), "Skipped: neighbourhood stays text the model cannot read", "Location is a big price driver, so leaving it out costs a lot of R².", false, []);
  else {
    col("neighbourhood")!.kind = "dropped";
    const at = cols.findIndex((c) => c.key === "neighbourhood");
    cols.splice(at + 1, 0,
      { key: "is_Hilltop", label: "is_Hilltop", values: houses.map((h) => (h.hood === "Hilltop" ? 1 : 0)), numeric: true, kind: "new" },
      { key: "is_Downtown", label: "is_Downtown", values: houses.map((h) => (h.hood === "Downtown" ? 1 : 0)), numeric: true, kind: "new" });
    push("onehot", "One-hot encode neighbourhood", "Each category becomes a 0/1 column. Riverside is the baseline (both columns 0), so the weights read as 'price premium over Riverside'. Dropping one level avoids a column that is the sum of the others.",
      "Turn the neighbourhood name into 0/1 columns", "Riverside = baseline; is_Hilltop and is_Downtown carry each area's premium.", true,
      [{ id: "oh", type: "transform", message: `counts: ${HOODS.map((h) => `${h} ${houses.filter((x) => x.hood === h).length}`).join(", ")}` }]);
  }
  // 4. standardise
  if (off("scale")) push("scale", "Standardise (skipped)", skipped("Standardise numeric columns"), "Skipped: columns keep their original units", "With ridge (λ > 0) unscaled columns are penalised unevenly.", false, []);
  else {
    const fitRows = cfg.leakScaler ? houses.map((_, i) => i) : trainIdx;
    scalers = cols.filter((c) => c.numeric && !c.key.startsWith("is_")).map((c) => {
      const v = c.values!;
      const sub = fitRows.map((i) => v[i]), tr = trainIdx.map((i) => v[i]);
      return { key: c.key, mu: mean(sub), sd: std(sub) || 1, muTrain: mean(tr), sdTrain: std(tr) || 1 };
    });
    for (const s of scalers) {
      const c = col(s.key)!;
      c.values = c.values!.map((v) => (v - s.mu) / s.sd);
      c.kind = "changed"; c.scaled = true; c.label = `${c.label} (z)`;
    }
    push("scale", cfg.leakScaler ? "Standardise (scaler fit on ALL rows: leak)" : "Standardise using training rows only",
      cfg.leakScaler
        ? `The scaler's mean and standard deviation were computed on all ${N_ROWS} rows, including the ${testIdx.length} held-out rows. Information from the test set has leaked into the features, so the held-out score is no longer a clean estimate.`
        : `Each numeric column becomes z = (x − μ)/σ, with μ and σ computed on the ${trainIdx.length} training rows only and then applied unchanged to the held-out rows. Every coefficient is now on the same scale, so the ridge penalty treats them fairly.`,
      cfg.leakScaler ? "Leak: the scaler peeked at the held-out rows" : "Standardise: rescale columns using training statistics only",
      cfg.leakScaler ? "μ and σ include held-out rows. The score below is contaminated, even if only slightly." : "z = (x − μ_train) / σ_train. One-hot columns stay 0/1.", true,
      [
        ...scalers.map((s) => ({ id: `sc-${s.key}`, type: "transform", message: `${s.key}: μ = ${s.mu.toFixed(3)}, σ = ${s.sd.toFixed(3)}${cfg.leakScaler ? ` (train-only would be μ = ${s.muTrain.toFixed(3)}, σ = ${s.sdTrain.toFixed(3)})` : ""}`, severity: cfg.leakScaler ? ("warning" as const) : undefined })),
        ...(cfg.lambda === 0 ? [{ id: "ols", type: "note", message: "λ = 0 (plain least squares): rescaling a column only rescales its weight, so predictions and R² do not change." }] : []),
      ]);
  }
  // 5. interaction
  const size = col("log_sqft") ?? col("sqft")!;
  const dt = col("is_Downtown");
  if (off("interact") || !dt) push("interact", "Interaction term (skipped)", dt ? skipped("Interaction: size × Downtown") : "The interaction needs the is_Downtown column, so it is skipped while one-hot encoding is off.",
    dt ? "Skipped: no size × Downtown term" : "Can't build size × Downtown without one-hot columns", "Downtown space is worth more per doubling of size; without this term the model can't express that.", false,
    dt ? [] : [{ id: "dep", type: "dependency", message: "interaction requires is_Downtown (one-hot)", severity: "warning" }]);
  else {
    cols.push({ key: "size_x_dt", label: `${size.label.replace(" (z)", "")}×Downtown`, values: size.values!.map((v, i) => v * dt.values![i]), numeric: true, kind: "new" });
    push("interact", "Add interaction: size × Downtown", "A new column multiplies size by is_Downtown. It is 0 outside Downtown, so its weight is the extra price per unit of size that only applies Downtown.",
      "Multiply two columns: size matters more Downtown", "The new column is 0 everywhere except Downtown rows, where it copies the size value.", true,
      [{ id: "ix", type: "transform", message: `size×Downtown = ${size.label} · is_Downtown` }]);
  }
  const first = steps[0].state.r2Test, last = steps.at(-1)!.state.r2Test;
  const finalState = steps.at(-1)!.state;
  steps.push({ id: "summary", title: "Contribution of each transform", description: `Held-out R² went from ${first.toFixed(3)} (raw sqft only) to ${last.toFixed(3)}. Toggle transforms off to see how much each one is worth on this data.`,
    state: { ...structuredClone(finalState), cols: structuredClone(cols), stage: "summary", on: true, prevR2: first, headline: `R² ${first.toFixed(2)} → ${last.toFixed(2)}: what each transform added`, sub: "Bars show the change in held-out R² at each step. Toggle a transform off and replay to measure it alone." },
    events: [{ id: "sum", type: "metric", message: `held-out R²: raw ${first.toFixed(3)} → engineered ${last.toFixed(3)}`, severity: cfg.leakScaler ? "warning" : undefined }] });

  return { houses, trainIdx, testIdx, steps };
}

export type FeSim = ReturnType<typeof simulate>;
