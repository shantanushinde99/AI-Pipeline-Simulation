import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Follow messages between orchestrator and specialist agents, with hand-off loops, conflicts and timeouts.", minutes: 15, hasMetrics: false };

const t = "multi-agent-coordination";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A multi-agent system splits work across specialised agents (here a researcher, a coder and a reviewer) that exchange messages and share state. The coordination pattern decides who talks to whom: an orchestrator dispatching workers, a pipeline of hand-offs, or a proposer and a critic debating while a judge decides." },
    { heading: "Messages vs shared state", body: "Messages carry instructions and artifacts between two agents. A shared blackboard holds facts everyone can read (the rate limit, the current code version). Conflicts surface on the blackboard: two agents writing different values for the same key is a signal that needs arbitration, not a race to the last write." },
    { heading: "Why coordination fails", body: "Agents that each act sensibly can still fail together: ownership is unclear so a task bounces between them (ping-pong), outputs contradict each other, or one hung worker blocks everyone waiting on it. Hop limits, explicit ownership, arbitration rules and per-worker timeouts are the coordination equivalent of budgets in a single-agent loop." },
    { heading: "Cost and latency", body: "More agents mean more messages, more tokens and more latency. Parallel workers can cut wall-clock time (agent work overlaps), while pipelines and debates are sequential: every round adds the full latency of each participant." },
    { heading: "Assumptions in this visualizer", body: "Agents are scripted, not LLMs: each pattern and failure produces a fixed message sequence. Work times and a 50 ms network delay are simulated; a sender works for its stated time after it last became ready. The 'truth' (100/min, cited in the docs) is fixed so the outcome can be graded. Arbitration is a simple rule: the cited value wins." },
  ],
  code: [
    {
      title: "Orchestrator with timeouts, a hop limit and arbitration (Python, asyncio)",
      source: `import asyncio

async def orchestrate(task, research, coder, reviewer, timeout=4.0, max_handoffs=4):
    board = Blackboard()
    try:   # dispatch workers in parallel, each with its own timeout
        limit_job = asyncio.wait_for(research.run("find the rate limit"), timeout)
        draft_job = coder.run("draft the client skeleton")
        limit, draft = await asyncio.gather(limit_job, draft_job)
        board.write("rate_limit", limit.value, by="research", cited=limit.citation)
    except asyncio.TimeoutError:
        board.write("rate_limit", "60/min", by="orchestrator", cited=None, assumed=True)

    for key, values in board.conflicts():              # e.g. 100/min (cited) vs 60/min (assumed)
        board.resolve(key, max(values, key=lambda v: v.cited is not None))

    handoffs = 0
    while (msg := await coder.run(f"apply {board['rate_limit']}")).kind == "handoff":
        handoffs += 1
        if handoffs >= max_handoffs:
            raise RuntimeError("hand-off loop: assign an owner")
    review = await asyncio.wait_for(reviewer.run(msg.artifact), timeout)
    return Result(msg.artifact, review, degraded=board.has_assumptions())`,
    },
  ],
  failureModes: [
    { title: "Ping-pong hand-offs", symptom: "Messages and cost climb while no artifact improves.", detect: "The same pair of agents exchanging hand-off messages; hop counter rising.", fix: "Give every task a single owner; enforce a hop limit; the orchestrator assigns explicitly when it trips." },
    { title: "Conflicting outputs", symptom: "The shipped result contradicts a source another agent found.", detect: "Two blackboard writes with the same key and different values; no arbitration event.", fix: "Detect conflicts on write; arbitrate by evidence (citations), not by recency." },
    { title: "Hung worker", symptom: "The whole run stalls or fails when one agent's tool hangs.", detect: "One agent with no reply for longer than its peers; others idle waiting.", fix: "Per-worker timeouts, fallback plans, and cancellation of abandoned work." },
    { title: "Debate that never converges", symptom: "Critic and proposer flip-flop; the design oscillates.", detect: "Critiques alternating between contradictory demands across rounds.", fix: "Round limit plus a judge with explicit acceptance criteria." },
    { title: "Untrusted inter-agent messages", symptom: "One agent's output makes another take an unsafe action.", detect: "Privileged tool calls right after reading another agent's message.", fix: "Treat messages between agents as data; apply the same permission checks as for user input.", scenarioId: "agent-injection" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Orchestrator–worker with no failure: the researcher works 1.5 s and the coder 2 s after dispatch. Why is the wall clock well below the sum of agent work?", options: ["Messages are free", "The researcher and coder run in parallel after the orchestrator dispatches both", "The reviewer is skipped", "The timeout shortens work"], answer: 1, explanation: "Both tasks are dispatched 100 ms apart and overlap, so their work counts once in wall time but twice in total work." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "Pipeline + ping-pong: the run fails after the hop limit. In orchestrator–worker the same failure succeeds. Why?", options: ["The pipeline has a lower timeout", "The orchestrator owns the task and can break the loop by assigning explicitly; the pipeline has no owner", "The orchestrator ignores hand-offs", "The pipeline has no blackboard"], answer: 1, explanation: "A hop limit only stops the damage. Recovery needs an owner who can change the plan." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "Failure 'A worker hangs': the researcher would answer after 9 s. Which timeout setting makes the orchestrator wait and finish with the correct 100/min?", options: ["4 s", "6 s", "8.5 s", "10 s"], answer: 3, explanation: "Only a timeout above 9 s lets the slow answer arrive. Lower timeouts finish sooner but on an assumed 60/min (degraded)." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "Conflicting outputs with arbitration off ship 60/min. What is the right fix?", options: ["Let the last write win", "Detect the conflicting blackboard values and arbitrate by evidence (the cited value wins)", "Add another coder", "Raise the hop limit"], answer: 1, explanation: "The conflict is visible on the blackboard. Recency is not evidence; the cited 100/min should win." },
    { id: `${t}-5`, topicId: t, skill: "metric", question: "In the summary, 'parallelism' = Σ agent work / wall clock. Which pattern scores closest to 1.0?", options: ["Orchestrator–worker", "Pipeline", "All are equal", "It depends only on the network delay"], answer: 1, explanation: "In a pipeline each stage waits for the previous one, so work never overlaps." },
  ],
};
