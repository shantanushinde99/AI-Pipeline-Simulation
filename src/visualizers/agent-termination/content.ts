import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch iteration, token, time and no-progress budgets fill and see which stop condition fires first.", minutes: 10, hasMetrics: false };

const t = "agent-termination";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "An agent loop has no natural end: the model decides when it is done. Termination logic is the set of conditions the runtime checks after every iteration: a final answer, a user cancel, a no-progress detector, and hard budgets on iterations, tokens (cost) and wall-clock time. The first condition that holds ends the run." },
    { heading: "Why several budgets", body: "Each budget catches a different failure. Loops repeat cheap actions, so iterations or a repeat detector catch them first. Slow tools burn wall time long before iterations run out. Reading large documents re-sends them on every later call, so token cost grows roughly quadratically and the token budget trips first." },
    { heading: "A clear final status", body: "Stopping is only half the job. The run must return why it stopped (completed, cancelled, budget exceeded, no progress) and any partial results, so callers can retry, escalate or show an honest message instead of a silent half-answer." },
    { heading: "Assumptions in this visualizer", body: "The agent is a scripted, rule-based sequence of actions per preset, not an LLM. Tokens are computed as system prompt (900) + all earlier observations + output (150) per call; every LLM call takes a simulated 1.2 s plus the tool's simulated latency; price is a made-up $0.01 per 1k tokens. Checks run in a fixed order after each iteration." },
  ],
  code: [
    {
      title: "Budgeted agent loop with a no-progress detector (Python)",
      source: `import time
from collections import Counter

def run(agent, task, max_iters=10, max_tokens=30_000, max_seconds=30, repeat_limit=3, cancel=None):
    start, tokens, seen = time.monotonic(), 0, Counter()
    for i in range(1, max_iters + 1):
        step = agent.step(task)                       # one LLM call + optional tool call
        tokens += step.usage.total_tokens
        if step.final_answer:
            return Result("completed", step.final_answer, i, tokens)
        if cancel and cancel.is_set():
            return Result("cancelled", agent.partial(), i, tokens)
        seen[(step.tool, step.args_json)] += 1
        if seen[(step.tool, step.args_json)] >= repeat_limit:
            return Result("no_progress", agent.partial(), i, tokens)
        if tokens >= max_tokens:
            return Result("token_budget", agent.partial(), i, tokens)
        if time.monotonic() - start >= max_seconds:
            return Result("wall_time", agent.partial(), i, tokens)
    return Result("max_iterations", agent.partial(), max_iters, tokens)`,
    },
  ],
  failureModes: [
    { title: "Infinite or long loops", symptom: "Runs hit the iteration cap; the same tool call appears over and over.", detect: "Repeated identical (tool, args) pairs in the trace; no new information in observations.", fix: "Add a no-progress detector; feed errors back to the planner; cap iterations.", scenarioId: "agent-retry-loop" },
    { title: "Runaway cost", symptom: "A few runs consume most of the token budget.", detect: "Per-call prompt tokens growing every iteration; large tool outputs kept verbatim in history.", fix: "Token/cost budgets per run; summarise or truncate observations; retrieve instead of pasting whole documents." },
    { title: "Hung runs", symptom: "Users wait minutes; workers stay busy.", detect: "Wall time far above iterations × typical latency; one slow tool dominates.", fix: "Wall-clock deadline for the run plus per-tool timeouts; allow user cancellation." },
    { title: "Silent stop", symptom: "Users get a partial or empty answer that looks final.", detect: "Final status missing from logs; budget stops counted as successes.", fix: "Always return an explicit status and reason; surface partial work as partial." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Default limits, preset 'Stuck in a loop'. Which condition fires first?", options: ["Iteration limit", "Token budget", "No-progress detector", "Wall time"], answer: 2, explanation: "The search action is issued for the third time at iteration 5, tripping the detector long before 10 iterations, 30k tokens or 30 s." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "You disable the no-progress detector on the loop preset. What happens?", options: ["The run finishes successfully", "It loops until the iteration limit (10) stops it", "The token budget stops it at iteration 3", "It never stops"], answer: 1, explanation: "Without the detector, the next guard is the iteration cap; tokens (about 20k) and time (about 15 s) stay under their limits." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "In 'Expensive context', each document adds 6,000 tokens of history. Why does the token budget trip at iteration 4 even though the agent read only 3 documents by then?", options: ["Output tokens are huge", "Every call re-sends all earlier observations, so call cost grows each iteration (1,050 → 7,050 → 13,050 → 19,050)", "The price per token increases", "The detector counts tokens"], answer: 1, explanation: "Cumulative tokens are the sum of growing per-call prompts: 1,050 + 7,050 + 13,050 + 19,050 = 40,200 ≥ 30,000." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "The 'Slow tool' preset stops with 'wall-time limit' after 4 iterations. What is the best fix if the report is genuinely needed?", options: ["Raise max iterations", "Raise the token budget", "Cache or parallelise the report calls, or raise the wall-time limit deliberately", "Disable the detector"], answer: 2, explanation: "Time, not iterations or tokens, is the binding constraint: each run_report call takes 9 s." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "After adding cancel support, how do you verify it works?", options: ["Check that healthy runs still complete", "Press cancel mid-run and confirm the run stops after that iteration with status 'cancelled' and partial work, not 'completed'", "Lower the token budget", "Count tool calls"], answer: 1, explanation: "Verification needs the cancel path itself: the run must stop promptly and report the correct status." },
  ],
};
