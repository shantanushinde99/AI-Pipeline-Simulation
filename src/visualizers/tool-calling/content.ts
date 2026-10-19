import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Follow a tool call between user, model, runtime and tool, with parallel calls and malformed JSON.", minutes: 10, hasMetrics: false };

const t = "tool-calling";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Tool calling is a protocol between a model and the code around it. The model never runs a tool: it returns an assistant message with a tool_calls list (a tool name plus arguments encoded as a JSON string). The runtime parses those arguments, validates them against the tool's schema, executes the function, serializes the result and appends it to the transcript as a role=tool message carrying the same tool_call_id. Then it calls the model again." },
    { heading: "Why the transcript matters", body: "The model is stateless between calls. Everything it knows about earlier tool results is in messages[]. If a result is missing, mislabelled with the wrong tool_call_id, or never appended because the runtime threw, the model cannot use it." },
    { heading: "Parallel tool calls", body: "When calls are independent, a model can request several in one turn. The runtime can execute them concurrently, so the turn costs the slowest call instead of the sum, and the run needs fewer model turns. Each result still gets its own tool message and id." },
    { heading: "Malformed arguments", body: "Arguments are text produced token by token, so they can be truncated or invalid JSON. Treat parsing as fallible: catch the error and return it to the model as a tool message, which usually lets it fix the call on the next turn." },
    { heading: "Assumptions in this visualizer", body: "The 'model' is a deterministic rule: it asks for each city it has no reading for (one per turn, or all at once in parallel mode) and answers when it has them all. The weather tool returns fixed values with fixed simulated latencies (Paris 120 ms, Tokyo 180 ms) and every model turn costs a simulated 400 ms. Parsing uses the browser's real JSON.parse, so the error text is genuine. No model or weather API is called." },
  ],
  code: [
    {
      title: "The runtime side of a tool-calling loop (Python, OpenAI-style messages)",
      source: `import json, asyncio

async def run(messages, tools, registry):
    while True:
        reply = await llm(messages=messages, tools=tools)       # model proposes, never executes
        messages.append(reply)
        if not reply.tool_calls:
            return reply.content                                  # final answer

        async def handle(call):
            try:
                args = json.loads(call.function.arguments)        # arguments is a STRING
                registry.validate(call.function.name, args)       # JSON Schema check
                result = await registry[call.function.name](**args)
                content = json.dumps(result)
            except (json.JSONDecodeError, ValidationError) as e:
                content = json.dumps({"error": f"{type(e).__name__}: {e}. Fix the arguments and retry."})
            return {"role": "tool", "tool_call_id": call.id, "content": content}

        # independent calls in one turn run concurrently
        messages += await asyncio.gather(*(handle(c) for c in reply.tool_calls))`,
    },
  ],
  failureModes: [
    { title: "Malformed JSON crashes the request", symptom: "Intermittent 500s on agent endpoints, often on long arguments.", detect: "Stack traces with JSONDecodeError / SyntaxError inside the tool dispatcher.", fix: "Catch parse errors and return them as a tool message with the same tool_call_id; cap retries.", scenarioId: "agent-invalid-args" },
    { title: "Tool result not appended", symptom: "The model calls the same tool again, or answers without the data.", detect: "Transcript has an assistant tool_call with no matching role=tool message.", fix: "Always append exactly one tool message per tool_call_id, including errors." },
    { title: "Mismatched tool_call_id", symptom: "Answer mixes up results (Paris temperature reported for Tokyo).", detect: "Tool message ids that do not match the ids in the preceding assistant message.", fix: "Copy ids from the call you executed; never re-number; test parallel turns specifically." },
    { title: "Sequential calls where parallel would do", symptom: "Slow responses that grow with the number of items asked about.", detect: "Trace shows one tool call per model turn with no data dependency between them.", fix: "Enable parallel tool calls and execute calls in a turn concurrently (e.g. asyncio.gather)." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "In sequential mode with the two-city question, how many times is the model called in total?", options: ["1", "2", "3", "4"], answer: 2, explanation: "Turn 1 asks for Paris, turn 2 (after the Paris result) asks for Tokyo, turn 3 writes the final answer. Parallel mode needs only 2 turns." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "Paris takes 120 ms and Tokyo 180 ms. In parallel mode, how long does the tool part of the run take?", options: ["120 ms", "180 ms", "300 ms", "360 ms"], answer: 1, explanation: "Both calls start together, so the turn waits for the slower one: max(120, 180) = 180 ms. Sequential mode pays 120 + 180 = 300 ms." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "The model emits arguments '{\"city\": \"Paris\"' (no closing brace). Which component should catch this?", options: ["The weather API", "The runtime, when it parses the arguments string", "The user", "Nobody: the model output is always valid JSON"], answer: 1, explanation: "Arguments arrive as text. The runtime's JSON.parse is the first thing that can fail, and it should turn the failure into a tool message the model can act on." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "With 'Throw (no error handling)' the malformed run ends in a 500. What is the smallest fix?", options: ["Retry the whole user request", "Catch the parse error and append it as a role=tool message with the same tool_call_id", "Remove the schema", "Switch to parallel mode"], answer: 1, explanation: "Returning the error keeps the conversation alive; in the visualizer the model fixes its call on the next turn and the run completes." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How can you verify every tool call was answered in a transcript?", options: ["Count assistant messages", "Check that each tool_call id has exactly one role=tool message with that tool_call_id", "Check the final answer is non-empty", "Measure latency"], answer: 1, explanation: "Ids are the contract between call and result. A missing or duplicated id means the model saw an incomplete or ambiguous history." },
  ],
};
