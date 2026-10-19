import { clamp } from "../../lib/math";
import type { SimulationStep } from "../../types/simulation";

/**
 * Hypothetical LLM workload and prices (not any real provider's). Cost per request =
 * input tokens × input price + output tokens × output price. Quality numbers are made-up per-difficulty scores.
 */
export type ModelId = "large" | "small";
export const PRICES: Record<ModelId, { in: number; out: number }> = { large: { in: 3, out: 15 }, small: { in: 0.25, out: 1.25 } }; // $ per 1M tokens
export const REQUESTS = 2_000_000; // per month
export const IN_TOKENS = 1800;
export const DIFFS = [
  { id: "easy", share: 0.6, out: 250 },
  { id: "medium", share: 0.25, out: 500 },
  { id: "hard", share: 0.15, out: 1100 },
] as const;
export const QUALITY: Record<ModelId, [number, number, number]> = { large: [0.95, 0.92, 0.88], small: [0.93, 0.78, 0.5] };
export const BATCH_DISCOUNT = 0.5;
export const BATCHABLE = 0.3; // share of traffic that is offline and can wait for a batch job

export type Config = { trim: number; hitRate: number; route: number; batching: boolean; cap: number | null };
export const BASELINE: Config = { trim: 0, hitRate: 0, route: 0, batching: false, cap: null };
export const DEFAULT_CONFIG: Config = { trim: 0.3, hitRate: 0.2, route: 1, batching: true, cap: 800 };
export const ROUTES = ["all → large", "easy → small", "easy + medium → small", "all → small"];

export type LeverId = "trim" | "cache" | "route" | "batch" | "cap";
export const LEVERS: { id: LeverId; name: string }[] = [
  { id: "trim", name: "Prompt trimming" }, { id: "cache", name: "Response caching" }, { id: "route", name: "Model routing" },
  { id: "batch", name: "Batching" }, { id: "cap", name: "Output caps" },
];

export function sanitize(c: Config): Config {
  return { trim: clamp(c.trim, 0, 0.6), hitRate: clamp(c.hitRate, 0, 0.9), route: clamp(Math.round(c.route), 0, 3), batching: c.batching, cap: c.cap === null ? null : Math.max(50, c.cap) };
}

export const modelFor = (route: number, d: number): ModelId => (d < route ? "small" : "large");
export const outTokens = (d: number, cap: number | null) => (cap === null ? DIFFS[d].out : Math.min(DIFFS[d].out, cap));
export const requestCost = (m: ModelId, inTok: number, outTok: number) => (inTok * PRICES[m].in + outTok * PRICES[m].out) / 1e6;

export type Breakdown = { d: number; model: ModelId; requests: number; billed: number; inTok: number; outTok: number; perReq: number; cost: number; quality: number };

/** Monthly cost and average quality for one configuration. */
export function evaluate(raw: Config) {
  const c = sanitize(raw);
  const trimF = 1 - 0.25 * Math.max(0, c.trim - 0.3); // trimming past 30% starts cutting useful context
  const batchF = c.batching ? 1 - BATCH_DISCOUNT * BATCHABLE : 1;
  const rows: Breakdown[] = DIFFS.map((df, d) => {
    const model = modelFor(c.route, d);
    const requests = REQUESTS * df.share;
    const billed = requests * (1 - c.hitRate); // cache hits cost no model tokens
    const inTok = IN_TOKENS * (1 - c.trim), outTok = outTokens(d, c.cap);
    const perReq = requestCost(model, inTok, outTok) * batchF;
    const capF = Math.min(1, 0.5 + (0.5 * outTok) / df.out); // truncated answers lose quality
    return { d, model, requests, billed, inTok, outTok, perReq, cost: billed * perReq, quality: QUALITY[model][d] * trimF * capF };
  });
  return { config: c, rows, cost: rows.reduce((s, r) => s + r.cost, 0), quality: rows.reduce((s, r) => s + DIFFS[r.d].share * r.quality, 0) };
}

/** Apply the levers one at a time (fixed order) so each bar of the waterfall is that lever's marginal saving. */
export function waterfall(raw: Config) {
  const c = sanitize(raw);
  const partial: Config[] = [BASELINE];
  const apply: Record<LeverId, (x: Config) => Config> = {
    trim: (x) => ({ ...x, trim: c.trim }), cache: (x) => ({ ...x, hitRate: c.hitRate }), route: (x) => ({ ...x, route: c.route }),
    batch: (x) => ({ ...x, batching: c.batching }), cap: (x) => ({ ...x, cap: c.cap }),
  };
  for (const l of LEVERS) partial.push(apply[l.id](partial[partial.length - 1]));
  const evals = partial.map(evaluate);
  return {
    baseline: evals[0],
    final: evals[evals.length - 1],
    stages: LEVERS.map((l, i) => ({ ...l, before: evals[i].cost, after: evals[i + 1].cost, saving: evals[i].cost - evals[i + 1].cost, quality: evals[i + 1].quality })),
  };
}

/** Every grid config (with the current cache/batching), plus its cost-quality Pareto frontier. */
export function frontier(raw: Config) {
  const c = sanitize(raw);
  const pts: { cost: number; quality: number; cfg: Config }[] = [];
  for (let route = 0; route <= 3; route++) for (const trim of [0, 0.15, 0.3, 0.45, 0.6]) for (const cap of [150, 300, 500, 800, null]) {
    const cfg = { ...c, route, trim, cap };
    const e = evaluate(cfg);
    pts.push({ cost: e.cost, quality: e.quality, cfg });
  }
  const sorted = [...pts].sort((a, b) => a.cost - b.cost || b.quality - a.quality);
  const front: typeof pts = [];
  for (const p of sorted) if (!front.length || p.quality > front[front.length - 1].quality + 1e-12) front.push(p);
  return { pts, front };
}

export type CostState = { phase: "request" | "baseline" | "lever" | "total" | "frontier"; lever?: LeverId; shown: number };

const usd = (v: number) => `$${Math.round(v).toLocaleString("en-US")}`;

export function simulate(raw: Config) {
  const c = sanitize(raw);
  const w = waterfall(c), f = frontier(c), cur = evaluate(c);
  const hardLarge = requestCost("large", IN_TOKENS, DIFFS[2].out);
  const steps: SimulationStep<CostState>[] = [
    { id: "request", title: "Cost of one request", description: `A hard request on the large model: ${IN_TOKENS} input tokens × $${PRICES.large.in}/M + ${DIFFS[2].out} output tokens × $${PRICES.large.out}/M = $${hardLarge.toFixed(4)}. Output tokens cost 5× more each.`,
      state: { phase: "request", shown: 0 }, events: [{ id: "r", type: "cost", message: `per request: $${hardLarge.toFixed(4)}`, tex: `${IN_TOKENS}\\cdot 3 + ${DIFFS[2].out}\\cdot 15 = ${Math.round(hardLarge * 1e6)}\\;\\mu\\$` }] },
    { id: "baseline", title: "Monthly baseline", description: `${REQUESTS.toLocaleString("en-US")} requests/month, all on the large model, full prompts, no cache: ${usd(w.baseline.cost)} per month at quality ${w.baseline.quality.toFixed(3)}.`,
      state: { phase: "baseline", shown: 0 }, events: w.baseline.rows.map((r) => ({ id: `b${r.d}`, type: "cost", message: `${DIFFS[r.d].id}: ${r.requests.toLocaleString("en-US")} req × $${r.perReq.toFixed(5)} = ${usd(r.cost)}` })) },
    ...w.stages.map((s, i): SimulationStep<CostState> => ({
      id: `lever-${s.id}`, title: s.name,
      description: `${s.name} saves ${usd(s.saving)} (${usd(s.before)} → ${usd(s.after)}); quality is now ${s.quality.toFixed(3)}.`,
      state: { phase: "lever", lever: s.id, shown: i + 1 },
      events: [{ id: s.id, type: "cost", severity: s.quality < (i ? w.stages[i - 1].quality : w.baseline.quality) - 1e-9 ? "warning" : undefined, message: `−${usd(s.saving)} · quality ${s.quality.toFixed(3)}` }],
    })),
    { id: "total", title: "Optimised monthly cost", description: `${usd(w.baseline.cost)} → ${usd(w.final.cost)} (−${((1 - w.final.cost / w.baseline.cost) * 100).toFixed(0)}%), quality ${w.baseline.quality.toFixed(3)} → ${w.final.quality.toFixed(3)}.`,
      state: { phase: "total", shown: LEVERS.length }, events: [{ id: "t", type: "cost", message: `final ${usd(w.final.cost)}/month` }] },
    { id: "frontier", title: "Cost vs quality frontier", description: `${f.pts.length} configurations (routing × trimming × caps). The frontier holds the ones no other config beats on both cost and quality.`,
      state: { phase: "frontier", shown: LEVERS.length }, events: [{ id: "f", type: "cost", message: `current: ${usd(cur.cost)}, q ${cur.quality.toFixed(3)}; frontier has ${f.front.length} points` }] },
  ];
  return { config: c, waterfall: w, frontier: f, current: cur, hardLarge, steps };
}
