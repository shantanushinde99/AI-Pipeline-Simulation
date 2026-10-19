import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export const MINUTES = 60;
export const CHUNK = 6;
export const RPM = 1000; // simulated requests per minute

export type ScenarioId = "quality-regression" | "error-outage" | "slow-dependency" | "healthy-spiky";
export const SCENARIOS: { id: ScenarioId; name: string; blurb: string; incident?: { start: number; end: number; metric: RuleId } }[] = [
  { id: "quality-regression", name: "Silent quality regression (prompt v2 at min 24)", blurb: "A prompt change makes answers worse. Latency and errors stay perfectly normal.", incident: { start: 24, end: 59, metric: "quality" } },
  { id: "error-outage", name: "Error outage (min 30–41)", blurb: "A dependency returns 5xx for 12 minutes.", incident: { start: 30, end: 41, metric: "errors" } },
  { id: "slow-dependency", name: "Slow dependency (min 18–31)", blurb: "The vector DB slows down for 14 minutes; p95 latency stays high.", incident: { start: 18, end: 31, metric: "latency" } },
  { id: "healthy-spiky", name: "Healthy, but spiky (no incident)", blurb: "Nothing is wrong. Only one-minute GC/cold-start blips.", incident: undefined },
];

export type Window = { t: number; latency: number; errors: number; requests: number; errorRate: number; quality: number };

/** Every random draw happens on every minute in the same order, so scenarios differ only where their incident applies. */
export function generateMetrics(scenario: ScenarioId, seed: number, spikes: boolean): Window[] {
  const r = rng(seed);
  return Array.from({ length: MINUTES }, (_, t) => {
    const g1 = r.gaussian(), spikeRoll = r.next(), spikeSize = r.next(), g2 = r.gaussian(), g3 = r.gaussian(), u = r.next();
    let latency = 420 + g1 * 35;
    if (spikes && spikeRoll < 0.1) latency = 860 + spikeSize * 260; // one-minute GC pause / cold start
    let errRate = 0.0022 + Math.abs(g2) * 0.0009;
    let quality = 0.86 + g3 * 0.022;
    if (scenario === "error-outage" && t >= 30 && t <= 41) errRate = 0.042 + u * 0.02;
    if (scenario === "slow-dependency" && t >= 18 && t <= 31) latency = 1050 + g1 * 60;
    if (scenario === "quality-regression" && t >= 24) quality = 0.64 + g3 * 0.03;
    const errors = Math.round(errRate * RPM);
    return { t, latency: Math.round(latency), errors, requests: RPM, errorRate: errors / RPM, quality: Math.round(clamp(quality, 0, 1) * 100) / 100 };
  });
}

export type RuleId = "latency" | "errors" | "quality";
export type AlertState = "ok" | "pending" | "firing" | "resolved";
export const ALERT_STATES: AlertState[] = ["ok", "pending", "firing", "resolved"];

export type RuleConfig = {
  latencyMs: number; latencyFor: number;
  errorMode: "static" | "burn"; errorPct: number; burnFactor: number; burnWindow: number;
  qualityOn: boolean; qualityMin: number; qualityFor: number;
  slo: number; // success-rate SLO in %, e.g. 99.5
};

export const PRESETS: Record<"noisy" | "useful", RuleConfig> = {
  noisy: { latencyMs: 800, latencyFor: 1, errorMode: "static", errorPct: 0.3, burnFactor: 6, burnWindow: 5, qualityOn: false, qualityMin: 0.75, qualityFor: 3, slo: 99.5 },
  useful: { latencyMs: 800, latencyFor: 5, errorMode: "burn", errorPct: 1, burnFactor: 6, burnWindow: 5, qualityOn: true, qualityMin: 0.75, qualityFor: 3, slo: 99.5 },
};

export type RuleEval = { value: number; breach: boolean; state: AlertState; prev: AlertState; streak: number; forN: number };
export type WindowEval = {
  w: Window;
  rules: Partial<Record<RuleId, RuleEval>>;
  burn: { errors: number; requests: number; rate: number };
  budgetTotal: number; budgetUsed: number; budgetLeft: number; // fraction, may go below 0
};

/** Burn rate = observed error rate over the lookback / error rate the SLO allows. 1 means "spending budget exactly on schedule". */
export function burnRate(ws: Window[], t: number, lookback: number, slo: number) {
  const slice = ws.slice(Math.max(0, t - lookback + 1), t + 1);
  const errors = slice.reduce((s, w) => s + w.errors, 0);
  const requests = slice.reduce((s, w) => s + w.requests, 0);
  const allowed = 1 - slo / 100;
  return { errors, requests, rate: requests && allowed > 0 ? errors / requests / allowed : 0 };
}

/** Alert state machine for one evaluation: condition must hold for `forN` consecutive windows before it fires. */
export function nextState(prev: AlertState, streak: number, breach: boolean, forN: number): { state: AlertState; streak: number } {
  if (breach) {
    const s = streak + 1;
    return { state: s >= Math.max(1, forN) ? "firing" : "pending", streak: s };
  }
  return { state: prev === "firing" ? "resolved" : "ok", streak: 0 };
}

export function evaluate(ws: Window[], c: RuleConfig): WindowEval[] {
  const rules: RuleId[] = ["latency", "errors", ...(c.qualityOn ? (["quality"] as const) : [])];
  const mem: Record<RuleId, { state: AlertState; streak: number }> = { latency: { state: "ok", streak: 0 }, errors: { state: "ok", streak: 0 }, quality: { state: "ok", streak: 0 } };
  const budgetTotal = (1 - c.slo / 100) * RPM * MINUTES;
  let used = 0;
  return ws.map((w, t) => {
    const burn = burnRate(ws, t, c.burnWindow, c.slo);
    used += w.errors;
    const out: WindowEval["rules"] = {};
    for (const id of rules) {
      const value = id === "latency" ? w.latency : id === "quality" ? w.quality : c.errorMode === "burn" ? burn.rate : w.errorRate * 100;
      const breach = id === "latency" ? value > c.latencyMs : id === "quality" ? value < c.qualityMin : c.errorMode === "burn" ? value > c.burnFactor : value > c.errorPct;
      const forN = id === "latency" ? c.latencyFor : id === "quality" ? c.qualityFor : 1;
      const prev = mem[id].state;
      mem[id] = nextState(prev, mem[id].streak, breach, forN);
      out[id] = { value, breach, state: mem[id].state, prev, streak: mem[id].streak, forN };
    }
    return { w, rules: out, burn, budgetTotal, budgetUsed: used, budgetLeft: budgetTotal > 0 ? 1 - used / budgetTotal : 0 };
  });
}

export const isPage = (e: RuleEval | undefined) => !!e && e.state === "firing" && e.prev !== "firing";

export function summarize(evals: WindowEval[], scenario: ScenarioId) {
  const inc = SCENARIOS.find((s) => s.id === scenario)?.incident;
  let pages = 0, noise = 0;
  let detectedAt: number | undefined, detectedBy: RuleId | undefined;
  evals.forEach((e) => (Object.entries(e.rules) as [RuleId, RuleEval][]).forEach(([id, r]) => {
    if (!isPage(r)) return;
    pages++;
    // A page is "real" only if the rule watching the incident's metric fires while the incident is on.
    const during = inc && id === inc.metric && e.w.t >= inc.start && e.w.t <= inc.end + 2;
    if (!during) noise++;
    else if (detectedAt === undefined) { detectedAt = e.w.t; detectedBy = id; }
  }));
  return { pages, noise, useful: pages - noise, incident: inc, detectedAt, detectedBy, minutesToDetect: inc && detectedAt !== undefined ? detectedAt - inc.start : undefined, budgetLeft: evals.at(-1)?.budgetLeft ?? 1 };
}

export const RULE_NAMES: Record<RuleId, string> = { latency: "p95 latency", errors: "error rate", quality: "quality score" };

export function ruleText(id: RuleId, c: RuleConfig) {
  if (id === "latency") return c.latencyFor <= 1 ? `p95 latency > ${c.latencyMs} ms (static threshold)` : `p95 latency > ${c.latencyMs} ms for ${c.latencyFor} min`;
  if (id === "errors") return c.errorMode === "static" ? `error rate > ${c.errorPct}% (static threshold)` : `SLO burn rate (${c.burnWindow}-min) > ${c.burnFactor}×`;
  return `quality score < ${c.qualityMin} for ${c.qualityFor} min`;
}

export type MonState = {
  phase: "setup" | "chunk" | "summary";
  from: number; // first new window shown in this step
  upTo: number; // windows visible once the step finishes
  headline: string;
  sub: string;
  evals: WindowEval[]; // every window evaluated so far
  pages: number;
};

export type MonInput = { scenario: ScenarioId; seed: number; spikes: boolean; config: RuleConfig };

const pct = (v: number) => `${Math.round(v * 100)}%`;

export function simulate({ scenario, seed, spikes, config }: MonInput): SimulationStep<MonState>[] {
  const ws = generateMetrics(scenario, seed, spikes);
  const evals = evaluate(ws, config);
  const rules = Object.keys(evals[0].rules) as RuleId[];
  const steps: SimulationStep<MonState>[] = [];
  let pages = 0;

  steps.push({
    id: "setup",
    title: "Rules armed",
    description: `${rules.length} alert rules are evaluated once per one-minute window: ${rules.map((id) => ruleText(id, config)).join("; ")}. Error budget for this 60-minute demo: ${Math.round(evals[0].budgetTotal)} failed requests (SLO ${config.slo}%).`,
    state: { phase: "setup", from: 0, upTo: 0, headline: "Three metric streams are about to arrive, one point per minute", sub: "Each rule checks every new window and moves through ok → pending → firing → resolved.", evals: [], pages: 0 },
    events: rules.map((id) => ({ id: `rule-${id}`, type: "rule", message: `Rule armed: ${ruleText(id, config)}` })),
  });

  for (let from = 0; from < MINUTES; from += CHUNK) {
    const upTo = Math.min(MINUTES, from + CHUNK);
    const chunk = evals.slice(from, upTo);
    const events: SimulationEvent[] = [];
    let fired: RuleId | undefined, pending: RuleId | undefined;
    for (const e of chunk) {
      for (const id of rules) {
        const r = e.rules[id]!;
        if (r.state === r.prev) continue;
        if (r.state === "firing") { pages++; fired ??= id; }
        if (r.state === "pending") pending ??= id;
        if (r.state === "ok" && r.prev === "resolved") continue;
        events.push({
          id: `${e.w.t}-${id}`, type: "alert",
          severity: r.state === "firing" ? "error" : r.state === "pending" ? "warning" : "info",
          message: `min ${e.w.t}: ${RULE_NAMES[id]} ${r.prev} → ${r.state}${r.state === "firing" ? " (page sent)" : ""}`,
        });
      }
    }
    const last = chunk.at(-1)!;
    const qualityDrop = chunk.some((e) => e.w.quality < 0.75);
    const headline = fired
      ? `Minutes ${from}–${upTo - 1}: the ${RULE_NAMES[fired]} rule FIRES and pages on-call`
      : pending
        ? `Minutes ${from}–${upTo - 1}: ${RULE_NAMES[pending]} breaches, rule is pending (not long enough yet)`
        : qualityDrop && !config.qualityOn
          ? `Minutes ${from}–${upTo - 1}: answer quality is down, but no rule watches it`
          : `Minutes ${from}–${upTo - 1}: all rules stay OK`;
    const sub = `Error budget left: ${pct(Math.max(0, last.budgetLeft))}${last.budgetLeft <= 0 ? " (exhausted)" : ""} · pages so far: ${pages}`;
    steps.push({
      id: `chunk-${from}`,
      title: `Minutes ${from}–${upTo - 1}`,
      description: `Six new one-minute windows arrive. For each, every rule checks its condition; consecutive breaches move it from pending to firing. ${events.length ? `${events.length} state change(s) this step.` : "No state changes this step."}`,
      state: { phase: "chunk", from, upTo, headline, sub, evals: evals.slice(0, upTo), pages },
      events: events.length ? events : [{ id: `quiet-${from}`, type: "alert", message: "No alert state changes." }],
    });
  }

  const s = summarize(evals, scenario);
  const other = summarize(evaluate(ws, config.qualityOn && config.errorMode === "burn" ? PRESETS.noisy : PRESETS.useful), scenario);
  steps.push({
    id: "summary",
    title: "Verdict",
    description: `${s.pages} page(s) in 60 minutes: ${s.useful} about the real incident, ${s.noise} noise. ${s.incident ? (s.minutesToDetect !== undefined ? `Detected ${s.minutesToDetect} min after the incident began (by the ${RULE_NAMES[s.detectedBy!]} rule).` : "The incident was never detected.") : "There was no incident, so every page was noise."}`,
    state: {
      phase: "summary", from: MINUTES, upTo: MINUTES, evals, pages: s.pages,
      headline: s.incident ? (s.minutesToDetect !== undefined ? `Incident caught after ${s.minutesToDetect} min, with ${s.noise} noisy page(s)` : "The incident was never caught: every dashboard stayed green") : `No incident, yet ${s.pages} page(s) woke someone up`,
      sub: `For comparison, the ${config.qualityOn && config.errorMode === "burn" ? "noisy" : "useful"} preset would send ${other.pages} page(s) here (${other.noise} noise).`,
    },
    events: [{ id: "sum", type: "summary", severity: s.incident && s.minutesToDetect === undefined ? "error" : s.noise > 2 ? "warning" : "info", message: `pages=${s.pages}, noise=${s.noise}, budget left=${pct(Math.max(0, s.budgetLeft))}` }],
  });
  return steps;
}
