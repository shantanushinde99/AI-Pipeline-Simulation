import { DEFAULT_AGENT, TOOL_SCHEMAS, runAgent, type AgentConfig, type AgentFaults } from "../visualizers/agents/simulation";
import { agentScenario } from "./agent-scenarios";
import type { Evidence, Scenario } from "./types";

/** Adds observations that the shared agent evidence set doesn't cover; each is computed from a real run. */
function withEvidence(s: Scenario, extra: (cfg: AgentConfig) => Evidence[], cfg: Partial<AgentConfig>, faults: AgentFaults): Scenario {
  return { ...s, run: (fixId) => { const o = s.run(fixId); return { ...o, evidence: [...o.evidence, ...extra({ ...DEFAULT_AGENT, ...cfg, faults })] }; } };
}

const WRONG_CFG: Partial<AgentConfig> = { task: "quote", maxIterations: 5 };
const WRONG_FAULTS: AgentFaults = { wrongTool: true };

const TIMEOUT_CFG: Partial<AgentConfig> = { task: "quote", maxRetries: 1 };
const TIMEOUT_FAULTS: AgentFaults = { slowTool: true };

export const AGENT_EXTRA_SCENARIOS: Scenario[] = [
  withEvidence(agentScenario({
    id: "agent-wrong-tool", title: "Quote agent runs out of steps after a budget cut", difficulty: "Beginner",
    symptom: "“iteration budget of 5 exhausted”", objective: "Read a trace to find a wasted step, and see why a tight budget turns one mistake into a failure.",
    incident: "After this week's cost-saving change, every seat-quote request ends with “Stopped: iteration budget of 5 exhausted without an answer.” Other agent tasks are unaffected.",
    cfg: WRONG_CFG, faults: WRONG_FAULTS,
    actions: ["final", "history", "failing", "budget", "latency", "memory", "retryPolicy", "deploys"],
    relevant: ["history", "failing", "budget"],
    deploys: ["Oct 15 agent-config: max_iterations 8 → 5 (cost reduction; 'happy path needs 5')", "Oct 14 tools: calculator description shortened for token savings", "Oct 12 llm-gateway: upgraded TLS certificates"],
    hypotheses: [
      { id: "h-tool", text: "The planner calls the calculator before it knows the price, wasting an iteration the tightened budget can't absorb.", isRootCause: true, evidenceIds: ["history", "failing", "budget", "toolDesc"], explanation: "Call #1 is calculator('12 * PRO_PRICE * 12'), which fails on an unknown identifier. The remaining correct path needs exactly 5 more steps; with the wasted one, 5 is not enough. The tool description never says it only accepts numbers." },
      { id: "h-calc", text: "The calculator service is broken.", isRootCause: false, evidenceIds: ["history", "failing"], explanation: "Later in the same run the calculator computes the total correctly. It only rejected an expression with a placeholder variable." },
      { id: "h-slow", text: "Tools are slow and the agent times out.", isRootCause: false, evidenceIds: ["latency"], explanation: "Every tool answers well within the timeout. The run ends on iteration count, not time." },
      { id: "h-kb", text: "The knowledge base is missing the discount document.", isRootCause: false, evidenceIds: ["memory"], explanation: "Memory holds both the price and the 15% discount; the agent just ran out of steps before submitting." },
    ],
    fixes: [
      { id: "f-desc", label: "Rewrite the calculator description: “numbers only; fetch prices first”, so the planner doesn't call it early", explanation: "The planner gathers inputs first and finishes in 5 iterations.", faults: { wrongTool: false } },
      { id: "f-retries", label: "Raise max retries to 4", explanation: "The calculator error isn't retryable and retrying wouldn't help; the wasted iteration remains.", cfg: { maxRetries: 4 } },
      { id: "f-timeout", label: "Raise the tool timeout to 5 s", explanation: "Nothing timed out; the budget is spent on steps, not time.", cfg: { timeoutMs: 5000 } },
    ],
    verifications: [
      { id: "v-trace", label: "Replay the quote task: the first tool call is search_docs, the quote equals $5,997.6, and the run uses ≤ 5 iterations; add a test that the budget leaves headroom over the happy path", sound: true, explanation: "Checks the wasted step is gone, the result is right, and the budget isn't set to exactly the minimum again." },
      { id: "v-cost", label: "Confirm cost per request went down", sound: false, explanation: "It went down because requests stop early. Cost says nothing about correctness." },
      { id: "v-calc", label: "Unit-test the calculator with '12 * 49 * 12'", sound: false, explanation: "The calculator already handles numbers; the problem was when the planner called it." },
    ],
    rootCause: "A shortened calculator description no longer said it only accepts numbers, so the planner calls it first with a placeholder (PRO_PRICE) and wastes an iteration. The happy path needs exactly 5 iterations, and the budget was just cut to 5, so the one wasted step means the agent never submits.",
  }), (cfg) => {
    const happy = runAgent({ ...cfg, faults: {}, maxIterations: 20 });
    return [{
      id: "toolDesc", action: "Read the tool descriptions the planner sees", title: "Read the tool descriptions the planner sees", relevant: true,
      lines: [`calculator: “${TOOL_SCHEMAS.calculator.description}” (no mention that only numeric literals are accepted)`, `search_docs: “${TOOL_SCHEMAS.search_docs.description}”`,
        `happy path for this task (simulated with correct tool order): ${happy.iterations} iterations: ${happy.toolCalls.map((c) => c.tool).join(" → ")}`],
    }];
  }, WRONG_CFG, WRONG_FAULTS),

  withEvidence(agentScenario({
    id: "agent-tool-timeout", title: "Quote agent gives up while the document service warms up", difficulty: "Intermediate",
    symptom: "“get_document timed out 2 time(s)”", objective: "Tell a slow dependency from a dead one, and size retries and timeouts to its real latency.",
    incident: "Since the knowledge-base migration this morning, most quote requests fail with “I couldn't retrieve the needed document”. The KB team says the service is up and every request it receives completes.",
    cfg: TIMEOUT_CFG, faults: TIMEOUT_FAULTS,
    actions: ["final", "history", "failing", "latency", "retryPolicy", "budget", "searchResults", "deploys"],
    relevant: ["failing", "latency", "retryPolicy"],
    deploys: ["Oct 19 08:00 kb-service: moved to new cluster (document cache starts cold)", "Oct 18 agent-runtime 1.7: max_retries 2 → 1 to cut tail latency", "Oct 17 planner prompt v15"],
    hypotheses: [
      { id: "h-timeout", text: "get_document is slow (cold cache) and the client's 2 s timeout with a single retry gives up before it warms up.", isRootCause: true, evidenceIds: ["failing", "latency", "retryPolicy", "serverLatency"], explanation: "Both attempts hit the 2,000 ms timeout, yet the server completes the same request in about 3.4 s. With max_retries cut to 1 the client gives up before latency falls under the timeout." },
      { id: "h-down", text: "The get_document service is down.", isRootCause: false, evidenceIds: ["serverLatency", "searchResults"], explanation: "The server completes every request; it is slow, not down. search_docs on the same cluster works." },
      { id: "h-budget", text: "The agent runs out of iterations.", isRootCause: false, evidenceIds: ["budget", "final"], explanation: "It uses 2 of 8 iterations and stops because the tool failed, not because of the budget." },
      { id: "h-id", text: "The search returns a document id that doesn't exist.", isRootCause: false, evidenceIds: ["searchResults", "failing"], explanation: "The id is current; the failure is a timeout, not a 404." },
    ],
    fixes: [
      { id: "f-retry", label: "Retry timeouts up to 3 times with exponential backoff", explanation: "Later attempts land after the cache warms and succeed.", cfg: { maxRetries: 3 } },
      { id: "f-tooltimeout", label: "Give get_document its own 4 s timeout, matching its measured latency", explanation: "The first attempt completes within the timeout.", cfg: { timeoutMs: 4000 } },
      { id: "f-iter", label: "Raise max iterations to 15", explanation: "The agent stops after the tool failure regardless of how many iterations remain.", cfg: { maxIterations: 15 } },
    ],
    verifications: [
      { id: "v-latency", label: "Replay with a cold cache: get_document succeeds, the quote equals $5,997.6, and alert when a tool's server p95 exceeds its client timeout", sound: true, explanation: "Reproduces the condition, checks the outcome, and catches the next mismatch between timeout and real latency." },
      { id: "v-uptime", label: "Confirm kb-service uptime is 100%", sound: false, explanation: "It was 100% throughout. Slow is not down." },
      { id: "v-warm", label: "Wait for the cache to warm and confirm requests succeed", sound: false, explanation: "Hides the problem until the next deploy or cache flush." },
    ],
    rootCause: "The KB migration started get_document with a cold cache, so early requests take ~3.4 s. The agent's client timeout is 2 s, and a recent change cut retries to 1, so both attempts time out before the cache warms and the agent fails the task.",
  }), (cfg) => {
    const probe = runAgent({ ...cfg, timeoutMs: 10_000 });
    const doc = probe.toolCalls.filter((c) => c.tool === "get_document");
    return [{
      id: "serverLatency", action: "Read kb-service server-side request logs", title: "Read kb-service server-side request logs", relevant: true,
      lines: [...doc.map((c) => `GET /documents/${String(c.args.id)} → 200 in ${c.attempts[0].latencyMs} ms (cold cache)`), `client timeout configured in agent-runtime: ${cfg.timeoutMs} ms`, "server error rate: 0%"],
    }];
  }, TIMEOUT_CFG, TIMEOUT_FAULTS),
];
