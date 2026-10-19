import { Bot, Calculator, FileSearch, FileText, ShieldAlert, Send, User, Wrench } from "lucide-react";
import { DEFAULT_AGENT, EXPECTED_TOTAL, TASKS, TOOL_SCHEMAS, runAgent, type AgentConfig, type AgentFaults, type AgentRun } from "../visualizers/agents/simulation";
import type { Evidence, Outcome, Scenario } from "./types";
import type { Difficulty } from "../types/topic";

const AGENT_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "user", label: "User request", Icon: User, x: 0, y: 130, match: (s) => s.id === "agent.run" },
    { id: "planner", label: "Planner (sim)", Icon: Bot, x: 180, y: 130, match: (s) => s.component === "planner" },
    { id: "runtime", label: "Tool runtime", Icon: Wrench, x: 360, y: 130, match: () => false },
    { id: "search", label: "search_docs", Icon: FileSearch, x: 560, y: 0, match: (s) => s.component === "tool:search_docs" },
    { id: "getdoc", label: "get_document", Icon: FileText, x: 560, y: 65, match: (s) => s.component === "tool:get_document" },
    { id: "calc", label: "calculator", Icon: Calculator, x: 560, y: 130, match: (s) => s.component === "tool:calculator" },
    { id: "submit", label: "submit_result", Icon: Send, x: 560, y: 195, match: (s) => s.component === "tool:submit_result" },
    { id: "delete", label: "delete_records", Icon: ShieldAlert, x: 560, y: 260, match: (s) => s.component === "tool:delete_records" },
  ],
  edges: [
    { source: "user", target: "planner" }, { source: "planner", target: "runtime" },
    ...["search", "getdoc", "calc", "submit", "delete"].map((t) => ({ source: "runtime", target: t })),
  ],
};

export type AgentAction = "final" | "history" | "failing" | "retryPolicy" | "budget" | "latency" | "searchResults" | "toolSpec" | "memory" | "guardrail" | "toolOutputs" | "userRequest" | "deploys";

const LABEL: Record<AgentAction, string> = {
  final: "Read the final answer and status", history: "Inspect the tool-call history", failing: "Inspect failing tool calls", retryPolicy: "Check the retry policy configuration",
  budget: "Check iteration count against the budget", latency: "Check tool latency", searchResults: "Inspect what search_docs returned", toolSpec: "Compare the agent's tool spec with the runtime schema",
  memory: "Inspect the agent's working memory", guardrail: "Read the guardrail audit log", toolOutputs: "Read the content returned by tools", userRequest: "Re-read the user's request", deploys: "Read the deploy log",
};

function evidence(a: AgentAction, r: AgentRun, cfg: AgentConfig, deploys: string[]): Omit<Evidence, "relevant"> {
  const failing = r.toolCalls.filter((c) => !c.ok);
  const lines = ((): string[] => {
    switch (a) {
      case "final": return [`status: ${r.final.status}`, `answer: “${r.final.text}”`, ...(cfg.task === "quote" ? [`expected total: $${EXPECTED_TOTAL.toLocaleString("en-US")} (12 × $49 × 12 × 0.85)`] : [])];
      case "history": return r.toolCalls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.ok ? "ok" : c.validation ? "rejected by validation" : c.attempts.at(-1)?.message}`);
      case "failing": return failing.length ? failing.slice(0, 4).map((c) => `${c.tool}(${JSON.stringify(c.args)}): ${c.validation ?? c.attempts.map((x) => `#${x.n} ${x.outcome} ${x.latencyMs}ms ${x.outcome === "ok" ? "" : x.message}`).join(" | ")}`).concat(failing.length > 4 ? [`…and ${failing.length - 4} more identical failures`] : []) : ["No tool call failed."];
      case "retryPolicy": return [`max_retries: ${cfg.maxRetries}`, `retry_on: ${cfg.faults.retryAllErrors ? '["*"]  (every error, including 4xx)' : '["timeout", "5xx"]'}`, `errors surfaced to planner: ${cfg.faults.retryAllErrors ? "only after retries are exhausted, and the planner re-issues the same call" : "yes, immediately for non-retryable errors"}`];
      case "budget": return [`iterations used: ${r.iterations} of ${cfg.maxIterations}`, `tool calls: ${r.toolCalls.length}`, `simulated wall time: ${(r.spans[0].durationMs / 1000).toFixed(1)} s`];
      case "latency": return r.toolCalls.map((c) => `${c.tool}: ${c.attempts.map((x) => `${x.latencyMs}ms`).join(", ") || "not executed"}`).concat([`timeout: ${cfg.timeoutMs} ms`]);
      case "searchResults": {
        const s = r.toolCalls.filter((c) => c.tool === "search_docs" && c.ok);
        return s.length ? s.map((c, i) => `search #${i + 1}: ${JSON.stringify(c.result)}`) : ["search_docs never succeeded."];
      }
      case "toolSpec": return [`agent's spec for calculator: { ${cfg.faults.staleToolSpec ? "expr" : "expression"}: string }  (prompt version 2026-09-02)`, `runtime schema: { ${TOOL_SCHEMAS.calculator.required.join(", ")}: string }  (calculator v2, 2026-10-10)`];
      case "memory": {
        const last = r.steps.at(-1)!.state.memory;
        return Object.entries(last).filter(([, v]) => v !== undefined).map(([k, v]) => `${k} = ${JSON.stringify(v)}`);
      }
      case "guardrail": {
        const denied = r.toolCalls.filter((c) => c.attempts.some((x) => x.outcome === "blocked"));
        const executed = r.toolCalls.filter((c) => c.tool === "delete_records" && c.ok);
        return [`guardrails: ${cfg.guardrails ? "enabled" : "DISABLED"}`, ...denied.map((c) => `DENIED ${c.tool}(${JSON.stringify(c.args)}): ${c.attempts[0].message}`), ...executed.map((c) => `EXECUTED ${c.tool}(${JSON.stringify(c.args)}) → ${JSON.stringify(c.result)}`), ...(denied.length + executed.length === 0 ? ["No privileged calls attempted."] : [])];
      }
      case "toolOutputs": return r.toolCalls.filter((c) => c.ok && c.tool === "get_document").map((c) => `get_document(${JSON.stringify(c.args)}): “${(c.result as { text: string }).text}”`);
      case "userRequest": return [`“${TASKS[cfg.task]}”`];
      case "deploys": return deploys;
    }
  })();
  return { id: a, action: LABEL[a], title: LABEL[a], lines };
}

type Def = {
  id: string; title: string; difficulty: Difficulty; symptom: string; objective: string; incident: string;
  cfg: Partial<AgentConfig>; faults: AgentFaults; actions: AgentAction[]; relevant: AgentAction[]; deploys: string[];
  hypotheses: Scenario["hypotheses"]; fixes: (Scenario["fixes"][number] & { cfg?: Partial<AgentConfig>; faults?: AgentFaults })[]; verifications: Scenario["verifications"]; rootCause: string;
};

export function agentScenario(d: Def): Scenario {
  return {
    id: d.id, title: d.title, system: "Agent", difficulty: d.difficulty, symptom: d.symptom, objective: d.objective, incident: d.incident, relatedTopic: "agent-loop",
    architecture: AGENT_ARCH, hypotheses: d.hypotheses, verifications: d.verifications, rootCause: d.rootCause,
    fixes: d.fixes.map(({ id, label, explanation }) => ({ id, label, explanation })),
    run(fixId?: string): Outcome {
      const fix = d.fixes.find((f) => f.id === fixId);
      const cfg: AgentConfig = { ...DEFAULT_AGENT, ...d.cfg, ...(fix?.cfg ?? {}), faults: { ...d.faults, ...(fix?.faults ?? {}) } };
      const r = runAgent(cfg);
      const errors = r.toolCalls.filter((c) => !c.ok).length;
      return {
        spans: r.spans,
        metrics: [
          { label: "Agent service uptime", value: "100%", status: "ok", kind: "health" },
          { label: "Requests completed", value: r.final.status === "budget_exceeded" ? "0%" : "100%", status: r.final.status === "budget_exceeded" ? "error" : "ok", kind: "health", hint: "returned any response" },
          { label: "Iterations used", value: `${r.iterations}/${cfg.maxIterations}`, status: r.iterations >= cfg.maxIterations ? "error" : r.iterations > 6 ? "warning" : "ok", kind: "health" },
          { label: "Tool error rate", value: `${Math.round((errors / Math.max(1, r.toolCalls.length)) * 100)}%`, status: errors ? "warning" : "ok", kind: "health" },
          { label: "Task correct", value: r.correct ? "yes" : "no", status: r.correct ? "ok" : "error", kind: "quality" },
        ],
        transcript: [{ label: "User", text: TASKS[cfg.task] }, { label: "Agent", text: r.final.text, tone: r.correct ? "ok" : "err" }],
        resolved: r.correct,
        evidence: d.actions.map((a) => ({ ...evidence(a, r, cfg, d.deploys), relevant: d.relevant.includes(a) })),
      };
    },
  };
}

export const AGENT_SCENARIOS: Scenario[] = [
  agentScenario({
    id: "agent-retry-loop", title: "Quote agent burns its budget and never answers", difficulty: "Intermediate",
    symptom: "“Stopped: iteration budget exhausted”", objective: "Distinguish retryable from permanent errors and see why retries can loop.",
    incident: "Sales reps say the quoting agent spins for ~30 seconds and then reports it ran out of steps. Cost per request is 6× normal. The knowledge base was reorganized last week.",
    cfg: { task: "quote" }, faults: { staleDocId: true, retryAllErrors: true },
    actions: ["final", "history", "failing", "searchResults", "retryPolicy", "budget", "latency", "deploys"],
    relevant: ["history", "failing", "retryPolicy"],
    deploys: ["Oct 08 kb: pricing-2025 merged into 'pricing'; old id removed (search cache TTL 30 days)", "Oct 03 agent-runtime 1.6: unified retry middleware (retry_on: all errors)", "Sep 29 planner prompt v14"],
    hypotheses: [
      { id: "h-retry", text: "The retry policy retries a permanent 404, and the planner never gets a chance to re-plan, so it re-issues the same call.", isRootCause: true, evidenceIds: ["history", "failing", "retryPolicy"], explanation: "The same get_document('pricing-2025') call repeats with NotFound (404) every time. retry_on: ['*'] retries errors that can never succeed." },
      { id: "h-budget", text: "The iteration budget is too small for this task.", isRootCause: false, evidenceIds: ["budget", "history"], explanation: "A healthy run needs 5 iterations. Every iteration here repeats the same failing call; more budget means a longer loop (try the fix)." },
      { id: "h-down", text: "The get_document service is down.", isRootCause: false, evidenceIds: ["failing", "latency"], explanation: "It answers quickly (220 ms) with a definitive 404 for one id. That's not an outage." },
      { id: "h-tool", text: "The planner chose the wrong tool.", isRootCause: false, evidenceIds: ["history"], explanation: "get_document is the right tool; the id it was given is stale." },
    ],
    fixes: [
      { id: "f-policy", label: "Retry only transient errors (timeouts, 5xx); surface 4xx to the planner", explanation: "The 404 reaches the planner, which re-searches, gets the current id and finishes.", faults: { retryAllErrors: false } },
      { id: "f-budget", label: "Raise max iterations to 20", explanation: "The loop just runs longer and costs more.", cfg: { maxIterations: 20 } },
      { id: "f-retries", label: "Raise max retries to 4", explanation: "More retries of an error that can never succeed.", cfg: { maxRetries: 4 } },
    ],
    verifications: [
      { id: "v-replay", label: "Replay the trace: the 404 triggers one re-search, the quote equals $5,997.6, and a test asserts 4xx errors are never retried", sound: true, explanation: "Covers the mechanism, the outcome and a regression test." },
      { id: "v-budget", label: "Check that a normal request no longer hits the budget", sound: false, explanation: "Normal requests never hit it; the failure needs the stale id." },
      { id: "v-uptime", label: "Confirm get_document uptime is 100%", sound: false, explanation: "It was 100% throughout." },
    ],
    rootCause: "Search returned a deleted document id, and a new retry middleware retried every error including permanent 404s. The planner never saw the failure, so it re-issued the same call until the iteration budget ran out.",
  }),
  agentScenario({
    id: "agent-invalid-args", title: "Quotes are missing the volume discount", difficulty: "Intermediate",
    symptom: "Agent reports success with a 15% higher total", objective: "Follow a validation failure to a silently wrong final answer.",
    incident: "Finance noticed that quotes for 10+ seats are 15% higher than the price list allows. The agent reports success on every request and no alerts fired.",
    cfg: { task: "quote" }, faults: { staleToolSpec: true },
    actions: ["final", "history", "failing", "memory", "toolSpec", "budget", "deploys"],
    relevant: ["failing", "toolSpec", "history", "deploys"],
    deploys: ["Oct 10 calculator v2: argument renamed expr → expression", "Oct 10 agent prompt: tool specs NOT regenerated (manual step)", "Oct 02 kb: discounts page updated wording"],
    hypotheses: [
      { id: "h-spec", text: "The agent's tool spec is outdated: calculator calls fail validation and the agent falls back to estimating.", isRootCause: true, evidenceIds: ["failing", "toolSpec", "history"], explanation: "calculator({expr}) is rejected (missing 'expression'), then the agent submits an estimate that ignores the discount." },
      { id: "h-doc", text: "The discount document is missing.", isRootCause: false, evidenceIds: ["memory", "history"], explanation: "get_document('discounts') succeeded and memory holds discount = 0.15." },
      { id: "h-math", text: "The calculator has an arithmetic bug.", isRootCause: false, evidenceIds: ["failing"], explanation: "The calculator never executed; validation rejected the call first." },
      { id: "h-skip", text: "The planner skipped the discount step.", isRootCause: false, evidenceIds: ["history"], explanation: "The discount was fetched. It was lost when the calculation was replaced by an estimate." },
    ],
    fixes: [
      { id: "f-spec", label: "Regenerate the agent's tool specs from the runtime schemas", explanation: "The calculator call validates and the discount is applied.", faults: { staleToolSpec: false } },
      { id: "f-prompt", label: "Add “always apply volume discounts” to the prompt", explanation: "The agent already knows the discount; its calculation path is broken.", faults: {} },
      { id: "f-retry", label: "Raise max retries to 4", explanation: "Validation failures aren't executed, so they aren't retried.", cfg: { maxRetries: 4 } },
    ],
    verifications: [
      { id: "v-contract", label: "Re-run: calculator validates and executes, the submitted total equals $5,997.6; add a contract test that tool specs match runtime schemas", sound: true, explanation: "Verifies the path, the result, and prevents the drift recurring." },
      { id: "v-success", label: "Confirm the agent reports success", sound: false, explanation: "It reported success during the incident." },
      { id: "v-errors", label: "Confirm zero ValidationErrors over the last hour", sound: false, explanation: "Necessary but not sufficient; it doesn't check the total." },
    ],
    rootCause: "The calculator's argument was renamed (expr → expression) but the agent's tool spec wasn't regenerated. Every calculator call fails validation, and the agent falls back to an estimate that drops the 15% discount while still reporting success.",
  }),
  agentScenario({
    id: "agent-injection", title: "Agent tried to delete the invoices table", difficulty: "Advanced",
    symptom: "Guardrail blocked delete_records('invoices')", objective: "Identify prompt injection through tool output and choose a fix that keeps defenses in place.",
    incident: "Security alert: the guardrail blocked delete_records(table='invoices') from the support agent. The user had only asked for a document summary, which the agent then failed to deliver.",
    cfg: { task: "summarize" }, faults: { trustToolOutput: true },
    actions: ["userRequest", "final", "history", "guardrail", "toolOutputs", "memory", "deploys"],
    relevant: ["toolOutputs", "history", "guardrail"],
    deploys: ["Oct 12 kb: vendor-notes.md edited by external vendor portal sync", "Oct 05 guardrails 2.0: privileged tools require approval", "Sep 30 planner prompt v14"],
    hypotheses: [
      { id: "h-inject", text: "Prompt injection: the agent followed an instruction embedded in a document it read.", isRootCause: true, evidenceIds: ["toolOutputs", "history", "guardrail"], explanation: "delete_records was chosen immediately after reading vendor-notes, whose text instructs AI assistants to call delete_records with table='invoices'." },
      { id: "h-user", text: "The user asked the agent to delete invoices.", isRootCause: false, evidenceIds: ["userRequest"], explanation: "The request was only to summarize the document." },
      { id: "h-guard", text: "The guardrail is misconfigured.", isRootCause: false, evidenceIds: ["guardrail"], explanation: "It correctly blocked the call. It's the reason this is a near-miss and not data loss." },
      { id: "h-random", text: "A random planner glitch picked the wrong tool.", isRootCause: false, evidenceIds: ["history", "toolOutputs"], explanation: "The call's arguments exactly match the text in the document, so it is caused, not random." },
    ],
    fixes: [
      { id: "f-data", label: "Treat tool output as untrusted data: the planner never takes instructions from it (keep the guardrail)", explanation: "The agent summarizes and flags the embedded instruction.", faults: { trustToolOutput: false } },
      { id: "f-noguard", label: "Disable the guardrail so the agent can complete its task", explanation: "The deletion executes. The task 'succeeds' and 18,442 invoice rows are gone (simulated).", cfg: { guardrails: false } },
      { id: "f-iter", label: "Raise max iterations so the agent can recover after the denial", explanation: "The agent stops anyway after the denial; nothing changes its trust in tool output.", cfg: { maxIterations: 15 } },
    ],
    verifications: [
      { id: "v-redteam", label: "Replay with the injected document plus a red-team set: no privileged call attempted, summary flags the injected text, guardrail denials stay at zero", sound: true, explanation: "Tests the root cause (trust boundary), not just the safety net." },
      { id: "v-guard", label: "Confirm the guardrail still blocks delete_records", sound: false, explanation: "It already did; that doesn't show the agent stopped following injected instructions." },
      { id: "v-summary", label: "Confirm the summary is produced", sound: false, explanation: "Disabling the guardrail also produces a summary, after deleting data." },
    ],
    rootCause: "A document edited through an external sync contained instructions addressed to AI assistants. The agent treated tool output as instructions and tried a destructive call. The guardrail prevented damage, but the trust boundary was wrong.",
  }),
];
