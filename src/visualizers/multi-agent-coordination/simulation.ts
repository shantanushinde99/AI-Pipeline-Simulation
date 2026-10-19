import { clamp } from "../../lib/math";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type AgentId = "user" | "orch" | "research" | "coder" | "review";
export const AGENTS: { id: AgentId; name: string }[] = [
  { id: "user", name: "User" }, { id: "orch", name: "Orchestrator" }, { id: "research", name: "Researcher" }, { id: "coder", name: "Coder" }, { id: "review", name: "Reviewer" },
];
export type Pattern = "orchestrator" | "pipeline" | "debate";
export type Failure = "none" | "pingpong" | "conflict" | "timeout";
export const PATTERNS: Record<Pattern, string> = { orchestrator: "Orchestrator–worker", pipeline: "Pipeline (hand-offs in a line)", debate: "Debate / critic" };
export const FAILURES: Record<Failure, string> = { none: "No failure", pingpong: "Ping-pong hand-off loop", conflict: "Conflicting outputs", timeout: "A worker hangs (timeout)" };

export type CoordConfig = { pattern: Pattern; failure: Failure; maxHandoffs: number; workerTimeoutMs: number; arbitration: boolean };
export const DEFAULT_COORD: CoordConfig = { pattern: "orchestrator", failure: "none", maxHandoffs: 4, workerTimeoutMs: 4000, arbitration: true };
export function sanitize(c: CoordConfig): CoordConfig {
  return { ...c, maxHandoffs: Math.round(clamp(c.maxHandoffs, 1, 12)), workerTimeoutMs: clamp(c.workerTimeoutMs, 500, 30000) };
}

/** How long the stuck worker would take. */
export const STUCK_MS = 9000;
export const NET_MS = 50;
export const TRUE_LIMIT = 100;

export type Kind = "task" | "result" | "handoff" | "critique" | "timeout" | "late" | "arbitrate" | "final";
export type BBWrite = { key: string; value: string; cited: boolean };
type Template = { from: AgentId; to: AgentId; kind: Kind; label: string; work: number; bb?: BBWrite; /** timeout: fires `after` ms after message `ref` was sent */ timer?: { ref: number; after: number } };
export type Message = Template & { i: number; sent: number; arrive: number };
export type BBEntry = BBWrite & { by: AgentId; at: number; status: "ok" | "conflict" | "resolved" | "superseded" };
export type Outcome = { status: "success" | "degraded" | "failed"; limitUsed: number | null; reason: string };

function build(c: CoordConfig): { msgs: Template[]; outcome: Outcome } {
  const m: Template[] = [];
  const add = (t: Template) => { m.push(t); return m.length - 1; };
  const handoffLoop = (a: AgentId, b: AgentId, la: string, lb: string) => {
    for (let h = 0; h < c.maxHandoffs; h++) add(h % 2 === 0 ? { from: a, to: b, kind: "handoff", label: la, work: 400 } : { from: b, to: a, kind: "handoff", label: lb, work: 400 });
  };
  const stuck = c.failure === "timeout" && c.workerTimeoutMs < STUCK_MS;
  const slow = c.failure === "timeout" && !stuck;

  if (c.pattern === "orchestrator") {
    add({ from: "user", to: "orch", kind: "task", label: "Build a rate-limited Weather API client", work: 0 });
    const dispatch = add({ from: "orch", to: "research", kind: "task", label: "Find the API's rate limit", work: 300 });
    add({ from: "orch", to: "coder", kind: "task", label: "Draft the client skeleton", work: 100 });
    if (c.failure === "pingpong") {
      handoffLoop("research", "coder", "hand-off: writing the client is your job", "hand-off: I need the limit first");
      add({ from: "orch", to: "research", kind: "arbitrate", label: `loop detected (${c.maxHandoffs} hand-offs): report the limit to me`, work: 200 });
    }
    let limit = TRUE_LIMIT;
    let status: Outcome["status"] = "success";
    let reason = "Every worker delivered and the reviewer approved.";
    if (stuck) {
      add({ from: "research", to: "orch", kind: "timeout", label: `no reply after ${c.workerTimeoutMs / 1000} s`, work: 0, timer: { ref: dispatch, after: c.workerTimeoutMs } });
      add({ from: "orch", to: "coder", kind: "task", label: "Researcher timed out: assume 60/min", work: 150, bb: { key: "rate_limit", value: "60/min", cited: false } });
      add({ from: "research", to: "orch", kind: "late", label: "late result 100/min (discarded)", work: STUCK_MS });
      limit = 60;
      status = "degraded";
      reason = "The orchestrator re-planned around the hung researcher, so the run finished, but on an unverified assumption (60/min).";
    } else {
      add({ from: "research", to: "orch", kind: "result", label: "rate limit = 100/min (docs §4)", work: slow ? STUCK_MS : 1500, bb: { key: "rate_limit", value: "100/min", cited: true } });
      if (slow) reason = `The researcher was slow (${STUCK_MS / 1000} s) but within the ${c.workerTimeoutMs / 1000} s timeout, so the run waited and succeeded.`;
    }
    add({ from: "coder", to: "orch", kind: "result", label: c.failure === "conflict" ? "client.py v1 (assumes 60/min)" : "client.py v1", work: 2000,
      bb: c.failure === "conflict" ? { key: "rate_limit", value: "60/min", cited: false } : { key: "code", value: "client.py v1", cited: false } });
    if (c.failure === "conflict") {
      if (c.arbitration) {
        add({ from: "orch", to: "review", kind: "arbitrate", label: "arbitrate rate_limit: 100 (cited) vs 60 (assumed)", work: 200 });
        add({ from: "review", to: "orch", kind: "result", label: "use 100/min: the cited source wins", work: 600, bb: { key: "rate_limit", value: "100/min", cited: true } });
        reason = "The blackboard showed two values for rate_limit; arbitration picked the cited one.";
      } else {
        limit = 60;
        status = "degraded";
        reason = "Two agents wrote different rate limits and nobody arbitrated: the last write (60/min, uncited) silently won.";
      }
    }
    add({ from: "orch", to: "coder", kind: "task", label: `apply limit ${limit}/min`, work: 200 });
    add({ from: "coder", to: "orch", kind: "result", label: "client.py v2", work: 800, bb: { key: "code", value: `client.py v2 (${limit}/min)`, cited: false } });
    add({ from: "orch", to: "review", kind: "task", label: "review client.py v2", work: 100 });
    add({ from: "review", to: "orch", kind: "result", label: "approved (2 nits)", work: 900, bb: { key: "review", value: "approved", cited: false } });
    add({ from: "orch", to: "user", kind: "final", label: `client.py v2 · ${limit}/min${status === "degraded" ? " · ⚠ unverified" : ""}`, work: 150 });
    return { msgs: m, outcome: { status, limitUsed: limit, reason } };
  }

  if (c.pattern === "pipeline") {
    const first = add({ from: "user", to: "research", kind: "task", label: "Build a rate-limited Weather API client", work: 0 });
    if (c.failure === "pingpong") {
      handoffLoop("research", "coder", "hand-off: spec is vague, you decide", "hand-off: I need the limit first");
      add({ from: "coder", to: "user", kind: "final", label: `failed: hop limit (${c.maxHandoffs}) reached`, work: 100 });
      return { msgs: m, outcome: { status: "failed", limitUsed: null, reason: "Researcher and coder kept handing the task back and forth. With no orchestrator to break the tie, the hop limit killed the run." } };
    }
    if (stuck) {
      add({ from: "research", to: "coder", kind: "timeout", label: `no spec after ${c.workerTimeoutMs / 1000} s`, work: 0, timer: { ref: first, after: c.workerTimeoutMs } });
      add({ from: "coder", to: "user", kind: "final", label: "failed: upstream stage timed out", work: 100 });
      add({ from: "research", to: "coder", kind: "late", label: "late spec (pipeline already failed)", work: STUCK_MS });
      return { msgs: m, outcome: { status: "failed", limitUsed: null, reason: "A pipeline has no one to re-route work: one hung stage fails the whole run." } };
    }
    add({ from: "research", to: "coder", kind: "result", label: "spec: limit 100/min (docs §4)", work: slow ? STUCK_MS : 1500, bb: { key: "rate_limit", value: "100/min", cited: true } });
    let limit = TRUE_LIMIT;
    let status: Outcome["status"] = "success";
    let reason = slow ? "A slow stage delays everything behind it, but the run succeeded." : "Each stage handed a finished artifact to the next.";
    if (c.failure === "conflict") {
      add({ from: "coder", to: "review", kind: "result", label: "client.py v1 (uses 60/min)", work: 2000, bb: { key: "rate_limit", value: "60/min", cited: false } });
      if (c.arbitration) {
        add({ from: "review", to: "coder", kind: "critique", label: "conflict: spec says 100 (cited), code uses 60", work: 700 });
        add({ from: "coder", to: "review", kind: "result", label: "client.py v2 (100/min)", work: 800, bb: { key: "rate_limit", value: "100/min", cited: true } });
        reason = "The reviewer compared the blackboard values and sent the code back.";
      } else {
        limit = 60;
        status = "degraded";
        reason = "Nobody compared the coder's 60/min with the spec's 100/min: the wrong value shipped.";
      }
    } else {
      add({ from: "coder", to: "review", kind: "result", label: "client.py v1", work: 2000, bb: { key: "code", value: "client.py v1", cited: false } });
      add({ from: "review", to: "coder", kind: "critique", label: "changes: add backoff on 429", work: 700 });
      add({ from: "coder", to: "review", kind: "result", label: "client.py v2", work: 800, bb: { key: "code", value: "client.py v2", cited: false } });
    }
    add({ from: "review", to: "user", kind: "final", label: `approved · ${limit}/min${status === "degraded" ? " · ⚠ wrong" : ""}`, work: 400 });
    return { msgs: m, outcome: { status, limitUsed: limit, reason } };
  }

  // debate / critic
  add({ from: "user", to: "orch", kind: "task", label: "Build a rate-limited Weather API client", work: 0 });
  add({ from: "orch", to: "coder", kind: "task", label: "propose a design", work: 200 });
  const proposal = add({ from: "coder", to: "review", kind: "result", label: "proposal v1 (100/min, no backoff)", work: 1500, bb: { key: "design", value: "v1", cited: false } });
  if (stuck) {
    add({ from: "review", to: "orch", kind: "timeout", label: `critic silent for ${c.workerTimeoutMs / 1000} s`, work: 0, timer: { ref: proposal, after: c.workerTimeoutMs } });
    add({ from: "orch", to: "user", kind: "final", label: "v1 accepted unreviewed · ⚠", work: 200 });
    add({ from: "review", to: "orch", kind: "late", label: "late critique (ignored)", work: STUCK_MS });
    return { msgs: m, outcome: { status: "degraded", limitUsed: TRUE_LIMIT, reason: "The judge timed out the critic and shipped the proposal without review." } };
  }
  if (c.failure === "pingpong") {
    for (let h = 0; h < c.maxHandoffs; h++) {
      add(h % 2 === 0
        ? { from: "review", to: "coder", kind: "critique", label: h % 4 === 0 ? "critique: add backoff" : "critique: backoff too complex, remove it", work: 600 }
        : { from: "coder", to: "review", kind: "result", label: `proposal v${(h + 3) / 2} (${h % 4 === 1 ? "with" : "without"} backoff)`, work: 800, bb: { key: "design", value: `v${(h + 3) / 2}`, cited: false } });
    }
    add({ from: "orch", to: "review", kind: "arbitrate", label: `round limit (${c.maxHandoffs}): judge decides`, work: 200 });
    add({ from: "orch", to: "user", kind: "final", label: "latest proposal shipped · ⚠ unresolved critique", work: 200 });
    return { msgs: m, outcome: { status: "degraded", limitUsed: TRUE_LIMIT, reason: "The critic flip-flopped, so the debate never converged. The round limit and the judge ended it." } };
  }
  add({ from: "review", to: "coder", kind: "critique", label: c.failure === "conflict" ? "critique: limit should be 60/min" : "critique: add backoff on 429", work: slow ? STUCK_MS : 700, bb: c.failure === "conflict" ? { key: "rate_limit", value: "60/min", cited: false } : undefined });
  let limit = TRUE_LIMIT;
  let status: Outcome["status"] = "success";
  let reason = slow ? "The critic was slow but inside the timeout; the debate converged." : "One round of critique improved the design; the critic accepted v2.";
  if (c.failure === "conflict") {
    add({ from: "orch", to: "research", kind: "task", label: "fact-check the rate limit", work: 150 });
    add({ from: "research", to: "orch", kind: "result", label: "docs §4: 100/min", work: 1200, bb: { key: "rate_limit", value: "100/min", cited: true } });
    if (c.arbitration) {
      add({ from: "orch", to: "coder", kind: "arbitrate", label: "conflict resolved: cited 100/min wins", work: 200, bb: { key: "rate_limit", value: "100/min", cited: true } });
      reason = "Critic and researcher disagreed; the judge arbitrated in favour of the cited source.";
    } else {
      add({ from: "coder", to: "orch", kind: "result", label: "v2 follows the critic (60/min)", work: 800 });
      limit = 60;
      status = "degraded";
      reason = "The coder followed the louder critic. Without arbitration the uncited 60/min shipped.";
    }
  }
  add({ from: "coder", to: "review", kind: "result", label: `proposal v2 (${limit}/min + backoff)`, work: 800, bb: { key: "design", value: "v2", cited: false } });
  add({ from: "review", to: "orch", kind: "result", label: "accept v2", work: 500 });
  add({ from: "orch", to: "user", kind: "final", label: `design v2 · ${limit}/min${status === "degraded" ? " · ⚠ wrong" : ""}`, work: 200 });
  return { msgs: m, outcome: { status, limitUsed: limit, reason } };
}

/** Assigns simulated times: a sender works `work` ms after it last became ready; messages arrive NET_MS later. */
export function schedule(templates: Template[]): Message[] {
  const ready: Record<AgentId, number> = { user: 0, orch: 0, research: 0, coder: 0, review: 0 };
  const out: Message[] = [];
  templates.forEach((t, i) => {
    let sent: number;
    if (t.timer) sent = out[t.timer.ref].sent + t.timer.after;
    else if (t.kind === "late") sent = Math.max(...out.filter((o) => o.to === t.from).map((o) => o.arrive), 0) + t.work;
    else { sent = ready[t.from] + t.work; ready[t.from] = sent; }
    const arrive = sent + NET_MS;
    // The receiver of a timeout is the waiting agent; a late result changes nothing.
    if (t.kind !== "late") ready[t.to] = Math.max(ready[t.to], arrive);
    out.push({ ...t, i, sent, arrive });
  });
  // Show them in the order they happen (a sequence diagram reads top to bottom in time).
  return out.sort((a, b) => a.sent - b.sent || a.i - b.i).map((x, i) => ({ ...x, i }));
}

export type CoordState = {
  shown: number;
  bb: BBEntry[];
  totals: { messages: number; handoffs: number; clock: number; work: number };
  done: boolean;
};

export function applyWrite(bb: BBEntry[], w: BBWrite, by: AgentId, at: number): BBEntry[] {
  const same = bb.filter((e) => e.key === w.key && e.status !== "superseded");
  if (!same.length) return [...bb, { ...w, by, at, status: "ok" }];
  if (same.some((e) => e.value === w.value)) {
    // Re-asserting a value (e.g. an arbitration verdict) resolves an open conflict; otherwise nothing changes.
    if (!same.some((e) => e.status === "conflict")) return bb;
    return bb.map((e) => (e.key === w.key && e.status !== "superseded" ? { ...e, status: e.value === w.value ? ("resolved" as const) : ("superseded" as const) } : e));
  }
  if (w.key === "code" || w.key === "design" || w.key === "review") return [...bb.map((e) => (e.key === w.key ? { ...e, status: "superseded" as const } : e)), { ...w, by, at, status: "ok" }];
  return [...bb.map((e) => (e.key === w.key && e.status !== "superseded" ? { ...e, status: "conflict" as const } : e)), { ...w, by, at, status: "conflict" }];
}

export function simulate(cfgIn: CoordConfig): { msgs: Message[]; outcome: Outcome; steps: SimulationStep<CoordState>[] } {
  const c = sanitize(cfgIn);
  const { msgs: templates, outcome } = build(c);
  const msgs = schedule(templates);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "message", message, severity });
  const name = (a: AgentId) => AGENTS.find((x) => x.id === a)!.name;
  const steps: SimulationStep<CoordState>[] = [{
    id: "intro", title: `Pattern: ${PATTERNS[c.pattern]}`,
    description: c.pattern === "orchestrator" ? "A central orchestrator splits the task, dispatches workers in parallel, merges their results and decides what happens next."
      : c.pattern === "pipeline" ? "Each agent does its part and hands the artifact to the next. Simple, but there is no one in charge when something goes wrong."
      : "A proposer and a critic iterate; a judge (the orchestrator) decides when the debate is over.",
    state: { shown: 0, bb: [], totals: { messages: 0, handoffs: 0, clock: 0, work: 0 }, done: false },
    events: [ev("cfg", `${PATTERNS[c.pattern]} · ${FAILURES[c.failure]}`)],
  }];
  let bb: BBEntry[] = [];
  let handoffs = 0, work = 0, clock = 0;
  msgs.forEach((msg, i) => {
    if (msg.kind === "handoff") handoffs++;
    if (msg.kind !== "timeout" && msg.kind !== "late") work += msg.work;
    clock = Math.max(clock, msg.arrive);
    if (msg.bb && msg.kind !== "late") bb = applyWrite(bb, msg.bb, msg.from, msg.arrive);
    const conflictNow = bb.some((e) => e.status === "conflict");
    steps.push({
      id: `m${i}`, title: `${name(msg.from)} → ${name(msg.to)}: ${msg.label.length > 38 ? `${msg.label.slice(0, 36)}…` : msg.label}`,
      description: msg.kind === "timeout" ? `${name(msg.to)} stops waiting for ${name(msg.from)} after ${(msg.timer!.after / 1000).toFixed(1)} s and decides what to do without it.`
        : msg.kind === "late" ? `${name(msg.from)} finally answers at ${(msg.arrive / 1000).toFixed(1)} s, after the decision was made. The result is ignored (work wasted).`
        : msg.kind === "handoff" ? `Hand-off #${handoffs}: ${name(msg.from)} passes the task to ${name(msg.to)} instead of doing it. Limit: ${c.maxHandoffs}.`
        : `${name(msg.from)} works ${(msg.work / 1000).toFixed(1)} s, then sends "${msg.label}". It arrives at ${(msg.arrive / 1000).toFixed(2)} s.${msg.bb ? ` Blackboard: ${msg.bb.key} = ${msg.bb.value}${msg.bb.cited ? " (cited)" : ""}.` : ""}${conflictNow ? " The blackboard now holds conflicting values." : ""}`,
      state: { shown: i + 1, bb, totals: { messages: i + 1, handoffs, clock, work }, done: false },
      events: [
        ev(`m${i}`, `${(msg.sent / 1000).toFixed(2)} s ${msg.from} → ${msg.to} [${msg.kind}] ${msg.label}`, msg.kind === "timeout" || msg.kind === "late" ? "warning" : msg.kind === "handoff" ? "warning" : undefined),
        ...(msg.bb && msg.kind !== "late" ? [ev(`bb${i}`, `blackboard ${msg.bb.key} = ${msg.bb.value}`, conflictNow ? "error" : undefined)] : []),
      ],
    });
  });
  const last = steps.at(-1)!.state;
  steps.push({
    id: "summary", title: outcome.status === "success" ? "Done: success" : outcome.status === "degraded" ? "Done, but degraded" : "Run failed",
    description: `${outcome.reason} ${msgs.length} messages, ${last.totals.handoffs} hand-offs, ${(last.totals.clock / 1000).toFixed(1)} s wall clock, ${(last.totals.work / 1000).toFixed(1)} s of agent work.`,
    state: { ...last, done: true },
    events: [ev("out", outcome.reason, outcome.status === "success" ? "info" : "error")],
  });
  return { msgs, outcome, steps };
}
