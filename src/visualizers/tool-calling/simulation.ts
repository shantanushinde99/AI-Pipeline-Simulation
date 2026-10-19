import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Lane = "user" | "model" | "runtime" | "tool";
export const LANES: Lane[] = ["user", "model", "runtime", "tool"];
export type Mode = "sequential" | "parallel" | "malformed";
export type Question = "one" | "two";
export type OnParseError = "return" | "crash";
export type CallingConfig = { question: Question; mode: Mode; onParseError: OnParseError };
export const DEFAULT_CALLING: CallingConfig = { question: "two", mode: "sequential", onParseError: "return" };

export const QUESTIONS: Record<Question, { text: string; cities: City[] }> = {
  one: { text: "What's the weather in Paris right now?", cities: ["Paris"] },
  two: { text: "Is it warmer in Paris or Tokyo right now?", cities: ["Paris", "Tokyo"] },
};

export type City = "Paris" | "Tokyo";
/** Simulated weather service: fixed readings and latencies so every run is reproducible. */
export const WEATHER: Record<City, { temp_c: number; condition: string; latencyMs: number }> = {
  Paris: { temp_c: 18, condition: "cloudy", latencyMs: 120 },
  Tokyo: { temp_c: 24, condition: "sunny", latencyMs: 180 },
};
/** Simulated time for one model turn (reading the transcript and emitting the next message). */
export const MODEL_TURN_MS = 400;

export const TOOL_SPEC = {
  name: "get_weather",
  description: "Current weather for a city.",
  parameters: { type: "object", properties: { city: { type: "string", enum: ["Paris", "Tokyo"] } }, required: ["city"], additionalProperties: false },
} as const;

export type ToolCall = { id: string; name: string; arguments: string };
export type ChatMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string; isError?: boolean };

export type ArrowKind = "user" | "tool_call" | "parse" | "validate" | "execute" | "result" | "tool_msg" | "final" | "crash";
export type Arrow = { id: string; from: Lane; to: Lane; kind: ArrowKind; label: string; ok: boolean; row: number; travel: number };

export type Check = { label: string; ok: boolean };
export type Detail =
  | { kind: "user"; text: string }
  | { kind: "tool_call"; calls: ToolCall[] }
  | { kind: "parse"; items: { callId: string; raw: string; ok: boolean; parsed?: Record<string, unknown>; error?: string }[] }
  | { kind: "validate"; items: { callId: string; checks: Check[] }[] }
  | { kind: "execute"; items: { callId: string; city: City; latencyMs: number }[]; wallMs: number }
  | { kind: "result"; items: { callId: string; city: City; result: { temp_c: number; condition: string }; latencyMs: number }[] }
  | { kind: "tool_msg"; items: { callId: string; serialized: string; isError: boolean }[] }
  | { kind: "final"; text: string }
  | { kind: "crash"; error: string };

export type CallingState = {
  arrows: Arrow[];
  newArrowIds: string[];
  transcript: ChatMessage[];
  newMessages: number;
  detail: Detail;
  clockMs: number;
  modelTurns: number;
  toolExecutions: number;
  status: "running" | "done" | "crashed";
};

export type CallingRun = { steps: SimulationStep<CallingState>[]; final: CallingState; toolLatencies: number[]; wallToolMs: number };

const isCity = (v: unknown): v is City => v === "Paris" || v === "Tokyo";

/** Validates parsed arguments against TOOL_SPEC; one row per check so the UI can tick them off. */
export function validateArgs(name: string, args: Record<string, unknown>): Check[] {
  const checks: Check[] = [{ label: `tool "${name}" exists`, ok: name === TOOL_SPEC.name }];
  checks.push({ label: "arguments is an object", ok: typeof args === "object" && args !== null && !Array.isArray(args) });
  checks.push({ label: "required field city is present", ok: "city" in args });
  checks.push({ label: "city is a string", ok: typeof args.city === "string" });
  checks.push({ label: 'city is one of "Paris", "Tokyo"', ok: isCity(args.city) });
  const extra = Object.keys(args).filter((k) => k !== "city");
  checks.push({ label: extra.length ? `no extra fields (found ${extra.join(", ")})` : "no extra fields", ok: extra.length === 0 });
  return checks;
}

/**
 * Rule-based stand-in for the model: reads the transcript and decides the next assistant message.
 * It asks for every city it has no reading for yet (one per turn, or all at once in parallel mode),
 * and in malformed mode its very first tool call has a truncated JSON argument string.
 */
function modelTurn(cfg: CallingConfig, transcript: ChatMessage[], nextId: () => string): Extract<ChatMessage, { role: "assistant" }> {
  const cities = QUESTIONS[cfg.question].cities;
  const known = new Map<City, { temp_c: number; condition: string }>();
  for (const m of transcript) {
    if (m.role !== "tool" || m.isError) continue;
    const call = transcript.flatMap((x) => (x.role === "assistant" ? x.tool_calls ?? [] : [])).find((c) => c.id === m.tool_call_id);
    const city = call ? (JSON.parse(call.arguments) as { city: City }).city : undefined;
    if (city) known.set(city, JSON.parse(m.content));
  }
  const missing = cities.filter((c) => !known.has(c));
  if (missing.length === 0) {
    const r = cities.map((c) => ({ c, ...known.get(c)! }));
    const text = r.length === 1
      ? `It's ${r[0].temp_c} °C and ${r[0].condition} in ${r[0].c}.`
      : (() => {
        const [a, b] = [...r].sort((x, y) => y.temp_c - x.temp_c);
        return `${a.c} is warmer: ${a.temp_c} °C (${a.condition}) vs ${b.temp_c} °C (${b.condition}) in ${b.c}.`;
      })();
    return { role: "assistant", content: text };
  }
  const anyCallYet = transcript.some((m) => m.role === "assistant" && m.tool_calls);
  const ask = cfg.mode === "parallel" ? missing : [missing[0]];
  const tool_calls = ask.map((city, i) => ({
    id: nextId(),
    name: TOOL_SPEC.name,
    // The malformed first call drops the closing brace: the model emitted invalid JSON.
    arguments: cfg.mode === "malformed" && !anyCallYet && i === 0 ? `{"city": "${city}"` : JSON.stringify({ city }),
  }));
  return { role: "assistant", content: null, tool_calls };
}

export function simulateCalling(cfgIn: Partial<CallingConfig> = {}): CallingRun {
  const cfg: CallingConfig = { ...DEFAULT_CALLING, ...cfgIn };
  const steps: SimulationStep<CallingState>[] = [];
  const transcript: ChatMessage[] = [];
  const arrows: Arrow[] = [];
  let row = 0, clock = 0, modelTurns = 0, toolExecutions = 0, evN = 0, callN = 0;
  const toolLatencies: number[] = [];
  let wallToolMs = 0;
  let status: CallingState["status"] = "running";
  const nextId = () => `call_${++callN}`;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id: `e${evN++}`, type: "log", message, severity, tex });

  const push = (title: string, description: string, newArrows: Omit<Arrow, "id" | "row">[], newMsgs: ChatMessage[], detail: Detail, events: SimulationEvent[]) => {
    const added = newArrows.map((a) => ({ ...a, id: `a${row}`, row: row++ }));
    arrows.push(...added);
    transcript.push(...newMsgs);
    steps.push({
      id: `s${steps.length}`, title, description, events,
      state: { arrows: [...arrows], newArrowIds: added.map((a) => a.id), transcript: [...transcript], newMessages: newMsgs.length, detail, clockMs: clock, modelTurns, toolExecutions, status },
    });
  };

  const q = QUESTIONS[cfg.question].text;
  push("User sends a message", "The conversation starts with one user message.", [{ from: "user", to: "model", kind: "user", label: "user message", ok: true, travel: 1 }], [{ role: "user", content: q }], { kind: "user", text: q }, [ev(`messages[0] = user: "${q}"`)]);

  for (let guard = 0; guard < 12; guard++) {
    modelTurns++;
    clock += MODEL_TURN_MS;
    const msg = modelTurn(cfg, transcript, nextId);
    if (!msg.tool_calls) {
      status = "done";
      push("Model answers the user", "No more tool calls: the model writes the final answer from the tool results in the transcript.", [{ from: "model", to: "user", kind: "final", label: "assistant answer", ok: true, travel: 1 }], [msg], { kind: "final", text: msg.content ?? "" }, [ev(`final answer after ${modelTurns} model turns, ${toolExecutions} tool executions`)]);
      break;
    }
    const calls = msg.tool_calls;
    push(
      calls.length > 1 ? `Model emits ${calls.length} tool calls at once` : `Model emits a tool call (${calls[0].id})`,
      `The model does not run anything. It returns an assistant message whose tool_calls list ${calls.length > 1 ? "has several entries" : "names a tool"} and carries the arguments as a JSON string.`,
      [{ from: "model", to: "runtime", kind: "tool_call", label: calls.length > 1 ? `tool_calls ×${calls.length}` : `tool_call ${calls[0].id}`, ok: true, travel: 1 }],
      [msg], { kind: "tool_call", calls },
      calls.map((c) => ev(`${c.id}: ${c.name}(${c.arguments})`)),
    );

    // 1. parse
    const parsed = calls.map((c) => {
      try {
        return { callId: c.id, raw: c.arguments, ok: true, parsed: JSON.parse(c.arguments) as Record<string, unknown> };
      } catch (e) {
        return { callId: c.id, raw: c.arguments, ok: false, error: (e as Error).message };
      }
    });
    const parseOk = parsed.every((p) => p.ok);
    push(
      parseOk ? "Runtime parses the arguments string" : "Runtime cannot parse the arguments",
      parseOk ? "JSON.parse turns each arguments string into an object." : `JSON.parse threw: ${parsed.find((p) => !p.ok)!.error}`,
      [{ from: "runtime", to: "runtime", kind: "parse", label: parseOk ? `JSON.parse${calls.length > 1 ? ` ×${calls.length}` : ""} ✓` : "JSON.parse ✗", ok: parseOk, travel: 1 }],
      [], { kind: "parse", items: parsed },
      parsed.map((p) => (p.ok ? ev(`${p.callId}: parsed ${JSON.stringify(p.parsed)}`) : ev(`${p.callId}: SyntaxError: ${p.error}`, "error"))),
    );
    if (!parseOk) {
      if (cfg.onParseError === "crash") {
        status = "crashed";
        const err = `Unhandled SyntaxError: ${parsed.find((p) => !p.ok)!.error}`;
        push("Runtime crashes: the user gets an error", "The parse error was not caught, so the whole request fails instead of letting the model fix its call.", [{ from: "runtime", to: "user", kind: "crash", label: "500 Internal Error", ok: false, travel: 1 }], [], { kind: "crash", error: err }, [ev(err, "error")]);
        break;
      }
      const items = parsed.map((p) => ({
        callId: p.callId, isError: !p.ok,
        serialized: JSON.stringify(p.ok ? { error: "not executed: another call in this turn had invalid JSON" } : { error: `Invalid JSON in arguments: ${p.error}. Send arguments as one JSON object, e.g. {"city": "Paris"}.` }),
      }));
      push("Runtime returns the parse error to the model", "The error becomes a tool message with the same tool_call_id, so the model can see what went wrong and try again.",
        [{ from: "runtime", to: "model", kind: "tool_msg", label: "tool: error", ok: false, travel: 1 }],
        items.map((i) => ({ role: "tool" as const, tool_call_id: i.callId, content: i.serialized, isError: true })),
        { kind: "tool_msg", items }, items.map((i) => ev(`messages += tool(${i.callId}): ${i.serialized}`, "warning")));
      continue;
    }

    // 2. validate
    const validations = calls.map((c, i) => ({ callId: c.id, checks: validateArgs(c.name, parsed[i].parsed!) }));
    push("Runtime validates against the tool schema", "Each parsed object is checked against get_weather's JSON Schema before anything runs.",
      [{ from: "runtime", to: "runtime", kind: "validate", label: "schema check ✓", ok: true, travel: 1 }], [],
      { kind: "validate", items: validations }, validations.map((v) => ev(`${v.callId}: ${v.checks.filter((c) => c.ok).length}/${v.checks.length} checks pass`)));

    // 3. execute (concurrently when there are several calls in one turn)
    const exec = calls.map((c, i) => { const city = parsed[i].parsed!.city as City; return { callId: c.id, city, latencyMs: WEATHER[city].latencyMs }; });
    const wall = Math.max(...exec.map((e) => e.latencyMs));
    const maxLat = wall;
    push(exec.length > 1 ? "Runtime runs both calls concurrently" : "Runtime executes the tool",
      exec.length > 1 ? `Both requests leave at once; the turn waits for the slower one (${wall} ms).` : `get_weather(city="${exec[0].city}") is called on the real function.`,
      exec.map((e) => ({ from: "runtime" as const, to: "tool" as const, kind: "execute" as const, label: `get_weather(${e.city})`, ok: true, travel: 1 })), [],
      { kind: "execute", items: exec, wallMs: wall }, exec.map((e) => ev(`execute get_weather({city: "${e.city}"})`)));
    toolExecutions += exec.length;
    toolLatencies.push(...exec.map((e) => e.latencyMs));
    wallToolMs += wall;
    clock += wall;
    const results = exec.map((e) => ({ ...e, result: { temp_c: WEATHER[e.city].temp_c, condition: WEATHER[e.city].condition } }));
    push(results.length > 1 ? "Both results come back" : "The tool returns a result",
      results.length > 1 ? "The faster call lands first; the runtime waits for every call in the turn." : `The function returned after ${results[0].latencyMs} ms (simulated).`,
      results.map((r) => ({ from: "tool" as const, to: "runtime" as const, kind: "result" as const, label: `${r.result.temp_c} °C · ${r.latencyMs} ms`, ok: true, travel: r.latencyMs / maxLat })), [],
      { kind: "result", items: results }, results.map((r) => ev(`${r.callId} returned ${JSON.stringify(r.result)} in ${r.latencyMs} ms`, "info", `T_{\\text{${r.callId.replace("_", "\\_")}}} = ${r.latencyMs}\\,\\text{ms}`)));

    // 4. serialize + append tool messages
    const items = results.map((r) => ({ callId: r.callId, serialized: JSON.stringify(r.result), isError: false }));
    push(items.length > 1 ? "Runtime appends two tool messages" : "Runtime appends a tool message",
      "Each result is serialized to a string and appended with role=tool and the matching tool_call_id. The model is called again with the longer transcript.",
      [{ from: "runtime", to: "model", kind: "tool_msg", label: items.length > 1 ? `tool ×${items.length}` : `tool ${items[0].callId}`, ok: true, travel: 1 }],
      items.map((i) => ({ role: "tool" as const, tool_call_id: i.callId, content: i.serialized })),
      { kind: "tool_msg", items }, items.map((i) => ev(`messages += tool(${i.callId}): ${i.serialized}`)));
  }
  return { steps, final: steps.at(-1)!.state, toolLatencies, wallToolMs };
}
