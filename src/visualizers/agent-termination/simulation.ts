import { clamp } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** One scripted iteration of the (rule-based, simulated) agent. */
export type Action = { tool: string; args: string; obsTokens: number; toolMs: number; final?: boolean };
export type PresetId = "healthy" | "loop" | "slow" | "expensive";

const llmMs = 1200;
export const SYSTEM_TOKENS = 900;
export const OUTPUT_TOKENS = 150;

/** Each script is cycled if the run outlives it. */
export const PRESETS: Record<PresetId, { label: string; note: string; script: Action[] }> = {
  healthy: {
    label: "Healthy run",
    note: "Search, read, compute, answer. The final-answer condition should fire first.",
    script: [
      { tool: "search_docs", args: "refund policy", obsTokens: 300, toolMs: 400 },
      { tool: "get_document", args: "refund-policy-v3", obsTokens: 900, toolMs: 700 },
      { tool: "calculate", args: "49 * 12 * 0.85", obsTokens: 40, toolMs: 50 },
      { tool: "final_answer", args: "Refund: $499.80", obsTokens: 0, toolMs: 0, final: true },
    ],
  },
  loop: {
    label: "Stuck in a loop",
    note: "The agent keeps issuing the identical search because the result never changes its plan.",
    script: [
      { tool: "search_docs", args: "pricing 2025", obsTokens: 300, toolMs: 400 },
      { tool: "get_document", args: "pricing-2025", obsTokens: 120, toolMs: 300 },
      { tool: "search_docs", args: "pricing 2025", obsTokens: 300, toolMs: 400 },
      { tool: "get_document", args: "pricing-2025", obsTokens: 120, toolMs: 300 },
    ],
  },
  slow: {
    label: "Slow tool",
    note: "A reporting tool takes 9 s per call. Wall time grows much faster than iterations.",
    script: [
      { tool: "search_docs", args: "quarterly revenue", obsTokens: 250, toolMs: 400 },
      { tool: "run_report", args: "revenue Q1", obsTokens: 400, toolMs: 9000 },
      { tool: "run_report", args: "revenue Q2", obsTokens: 400, toolMs: 9000 },
      { tool: "run_report", args: "revenue Q3", obsTokens: 400, toolMs: 9000 },
      { tool: "run_report", args: "revenue Q4", obsTokens: 400, toolMs: 9000 },
      { tool: "final_answer", args: "Revenue by quarter…", obsTokens: 0, toolMs: 0, final: true },
    ],
  },
  expensive: {
    label: "Expensive context",
    note: "Each step reads a whole contract into the context. Every later call re-sends all of it.",
    script: [
      { tool: "get_document", args: "contract-A (full)", obsTokens: 6000, toolMs: 600 },
      { tool: "get_document", args: "contract-B (full)", obsTokens: 6000, toolMs: 600 },
      { tool: "get_document", args: "contract-C (full)", obsTokens: 6000, toolMs: 600 },
      { tool: "get_document", args: "contract-D (full)", obsTokens: 6000, toolMs: 600 },
      { tool: "final_answer", args: "Clause comparison…", obsTokens: 0, toolMs: 0, final: true },
    ],
  },
};

export type Limits = {
  maxIterations: number;
  maxTokens: number;
  maxWallMs: number;
  /** Stop after this many identical consecutive-or-recent actions; 0 disables the detector. */
  noProgressWindow: number;
  /** Simulated user presses Cancel after this iteration (null = never). */
  cancelAfter: number | null;
};
export const DEFAULT_LIMITS: Limits = { maxIterations: 10, maxTokens: 30000, maxWallMs: 30000, noProgressWindow: 3, cancelAfter: null };
export const PRICE_PER_1K = 0.01;

export type StopReason = "final_answer" | "cancelled" | "no_progress" | "max_iterations" | "token_budget" | "wall_time";
export const STOP_TEXT: Record<StopReason, { status: string; message: string; ok: boolean }> = {
  final_answer: { status: "completed", message: "Final answer produced.", ok: true },
  cancelled: { status: "cancelled", message: "Stopped by the user. Partial work is returned.", ok: false },
  no_progress: { status: "stopped: no progress", message: "The same action repeated without new information.", ok: false },
  max_iterations: { status: "stopped: iteration limit", message: "Reached the maximum number of iterations.", ok: false },
  token_budget: { status: "stopped: token budget", message: "The token (cost) budget is used up.", ok: false },
  wall_time: { status: "stopped: wall-time limit", message: "The run took longer than allowed.", ok: false },
};

export type Iteration = { n: number; action: Action; tokens: number; ms: number; cumTokens: number; cumMs: number; repeats: number };
export type TermState = {
  iterations: Iteration[];
  /** Usage after the latest iteration. */
  usage: { iterations: number; tokens: number; ms: number; repeats: number };
  stop?: StopReason;
};
export type TermRun = { iterations: Iteration[]; stop: StopReason; cost: number };

export function sanitize(l: Limits): Limits {
  return {
    maxIterations: Math.round(clamp(l.maxIterations, 1, 30)),
    maxTokens: clamp(l.maxTokens, 1000, 500000),
    maxWallMs: clamp(l.maxWallMs, 1000, 600000),
    noProgressWindow: Math.round(clamp(l.noProgressWindow, 0, 10)),
    cancelAfter: l.cancelAfter === null ? null : Math.round(clamp(l.cancelAfter, 1, 30)),
  };
}

/** Tokens for one LLM call: system prompt + everything observed so far (re-sent each time) + the model's output. */
export const callTokens = (historyTokens: number) => SYSTEM_TOKENS + historyTokens + OUTPUT_TOKENS;

/** Conditions are checked after every iteration in this order; the first one that holds stops the run. */
export function checkStop(it: Iteration, l: Limits): StopReason | null {
  if (it.action.final) return "final_answer";
  if (l.cancelAfter !== null && it.n >= l.cancelAfter) return "cancelled";
  if (l.noProgressWindow > 1 && it.repeats >= l.noProgressWindow) return "no_progress";
  if (it.cumTokens >= l.maxTokens) return "token_budget";
  if (it.cumMs >= l.maxWallMs) return "wall_time";
  if (it.n >= l.maxIterations) return "max_iterations";
  return null;
}

export function runAgent(preset: PresetId, limitsIn: Limits): TermRun {
  const l = sanitize(limitsIn);
  const script = PRESETS[preset].script;
  const iterations: Iteration[] = [];
  let history = 0, cumTokens = 0, cumMs = 0;
  const seen = new Map<string, number>();
  for (let n = 1; ; n++) {
    const action = script[(n - 1) % script.length];
    const tokens = callTokens(history);
    const ms = llmMs + action.toolMs;
    cumTokens += tokens;
    cumMs += ms;
    history += action.obsTokens;
    const key = `${action.tool}(${action.args})`;
    const repeats = (seen.get(key) ?? 0) + 1;
    seen.set(key, repeats);
    const it: Iteration = { n, action, tokens, ms, cumTokens, cumMs, repeats };
    iterations.push(it);
    const stop = checkStop(it, l);
    if (stop) return { iterations, stop, cost: (cumTokens / 1000) * PRICE_PER_1K };
  }
}

export function simulate(preset: PresetId, limits: Limits): { run: TermRun; steps: SimulationStep<TermState>[] } {
  const l = sanitize(limits);
  const run = runAgent(preset, l);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "budget", message, severity, tex });
  const steps: SimulationStep<TermState>[] = [{
    id: "start", title: "Set the budgets",
    description: `Before the first call the runtime fixes every stop condition: ${l.maxIterations} iterations, ${l.maxTokens.toLocaleString("en-US")} tokens, ${l.maxWallMs / 1000} s wall time${l.noProgressWindow > 1 ? `, and a no-progress detector that trips when one action has been issued ${l.noProgressWindow} times` : ", no no-progress detector"}.`,
    state: { iterations: [], usage: { iterations: 0, tokens: 0, ms: 0, repeats: 0 } },
    events: [ev("cfg", `preset: ${PRESETS[preset].label}`)],
  }];
  run.iterations.forEach((it, i) => {
    const last = i === run.iterations.length - 1;
    const usage = { iterations: it.n, tokens: it.cumTokens, ms: it.cumMs, repeats: it.repeats };
    steps.push({
      id: `it${it.n}`, title: `Iteration ${it.n}: ${it.action.tool}`,
      description: `The planner calls ${it.action.tool}("${it.action.args}"). This LLM call costs ${it.tokens.toLocaleString("en-US")} tokens because it re-sends the system prompt and every earlier observation; the iteration takes ${(it.ms / 1000).toFixed(1)} s.${it.repeats > 1 ? ` This exact action has now been issued ${it.repeats} times.` : ""}`,
      state: { iterations: run.iterations.slice(0, i + 1), usage },
      events: [
        ev(`t${it.n}`, `tokens ${it.tokens}`, undefined, `\\text{tokens}_{${it.n}} = ${SYSTEM_TOKENS} + ${it.tokens - SYSTEM_TOKENS - OUTPUT_TOKENS} + ${OUTPUT_TOKENS} = ${it.tokens}`),
        ...(it.repeats > 1 ? [ev(`r${it.n}`, `repeat #${it.repeats} of ${it.action.tool}(${it.action.args})`, "warning")] : []),
        ...(last ? [ev("fire", `${STOP_TEXT[run.stop].status} fires after iteration ${it.n}`, STOP_TEXT[run.stop].ok ? "info" : "warning")] : []),
      ],
    });
  });
  const lastIt = run.iterations.at(-1)!;
  steps.push({
    id: "stop", title: `Stop: ${STOP_TEXT[run.stop].status}`,
    description: `${STOP_TEXT[run.stop].message} ${run.iterations.length} iterations, ${lastIt.cumTokens.toLocaleString("en-US")} tokens ($${run.cost.toFixed(3)}), ${(lastIt.cumMs / 1000).toFixed(1)} s.`,
    state: { iterations: run.iterations, usage: { iterations: lastIt.n, tokens: lastIt.cumTokens, ms: lastIt.cumMs, repeats: lastIt.repeats }, stop: run.stop },
    events: [ev("stop", `${STOP_TEXT[run.stop].status}: ${STOP_TEXT[run.stop].message}`, STOP_TEXT[run.stop].ok ? "info" : "error")],
  });
  return { run, steps };
}
