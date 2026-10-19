import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Push model JSON through parse, schema validation, repair and retries; see what each layer catches.", minutes: 12, hasMetrics: false };

const t = "structured-output-validation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "When an application asks a model for JSON, the reply is still just text. Structured output validation turns that text into a trusted, typed object in layers: parse it (is it JSON at all?), validate it against a schema (right fields, right types, allowed values, nothing extra), repair what can be fixed safely, and retry with the error message when it cannot." },
    { heading: "What each layer catches", body: "The parser catches syntax: trailing commas, truncated output. The schema validator catches everything that parses but is wrong: a number sent as a string, a missing required field, an unexpected key, a value outside the enum. Repair fixes mechanical problems (strip a trailing comma, coerce \"3\" to 3, drop unknown keys) but must never invent data. Retries give the model another chance with the validator's message. Constrained decoding removes syntax errors at generation time, but schema errors can remain, so validation stays." },
    { heading: "Why parse-only is dangerous", body: "A pipeline that only checks JSON.parse accepts every output that is well-formed but wrong. Those objects flow downstream and fail later, far from the cause, or worse, never fail and silently corrupt data. The red segment in the batch view counts them." },
    { heading: "Assumptions in this visualizer", body: "The model is simulated: its first output is one of seven fixed variants, and retries draw a new variant from a seeded random generator (valid with the chosen probability, otherwise a random error kind). Constrained decoding is simulated as 'syntax errors cannot occur'. Parsing is the real JSON.parse; the validator, the truncation repair and the type coercion are real code over a toy four-field schema. The error position is located by our own scan because browsers word parse errors differently." },
  ],
  code: [
    {
      title: "Pydantic: validate, then retry with the error",
      source: `from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")           # reject unknown keys
    title: str
    priority: Literal["low", "medium", "high"]
    estimate_hours: float = Field(ge=0, le=100)          # "3" is coerced to 3.0 in lax mode
    tags: list[str]

def get_ticket(llm, prompt: str, max_retries: int = 2) -> Ticket:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries + 1):
        raw = llm(messages)
        try:
            return Ticket.model_validate_json(raw)        # parse + validate in one step
        except ValidationError as err:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Invalid JSON for the schema:\\n{err}\\nReturn only corrected JSON."}]
    raise RuntimeError("model never produced a valid Ticket")`,
    },
    {
      title: "Zod in TypeScript",
      source: `import { z } from "zod";

const Ticket = z.object({
  title: z.string(),
  priority: z.enum(["low", "medium", "high"]),
  estimate_hours: z.coerce.number().min(0).max(100),
  tags: z.array(z.string()),
}).strict();

export function parseTicket(raw: string) {
  let json: unknown;
  try { json = JSON.parse(raw); } catch (e) { return { ok: false as const, error: \`parse: \${e}\` }; }
  const r = Ticket.safeParse(json);
  return r.success ? { ok: true as const, ticket: r.data } : { ok: false as const, error: r.error.issues };
}`,
    },
  ],
  failureModes: [
    { title: "Parse-only acceptance", symptom: "Downstream jobs crash on 'estimate_hours is not a number' days after the model call.", detect: "Count objects that parse but fail schema validation; the batch view's red segment.", fix: "Validate against a strict schema at the boundary; reject unknown keys.", scenarioId: "agent-invalid-args" },
    { title: "Over-eager repair", symptom: "Missing fields are filled with defaults and nobody notices wrong priorities.", detect: "Log every repair; alert when repairs change values instead of syntax.", fix: "Only apply lossless repairs (commas, numeric strings, unknown keys). Retry for anything else." },
    { title: "Truncated output from token limits", symptom: "Objects cut mid-field; 'repair' produces a valid-looking object missing fields.", detect: "Check finish_reason == length; see the 'Truncated JSON' variant lose two fields.", fix: "Raise max tokens, shorten the schema, or stream and validate incrementally." },
    { title: "Retries that never converge", symptom: "Latency and cost spike; the same enum error repeats.", detect: "Track attempts per request. Set 'Retry returns valid JSON' to 0% and watch calls rise with no gain.", fix: "Cap retries, include the exact validator message, and fail loudly with a fallback." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "Which output passes JSON.parse but fails schema validation?", options: ["Trailing comma", "Truncated JSON", "estimate_hours: \"3\"", "None: if it parses it is valid"], answer: 2, explanation: "\"3\" is valid JSON (a string). Only the schema knows estimate_hours must be a number. Parse-only pipelines accept it silently." },
    { id: `${t}-2`, topicId: t, skill: "fix", question: "Which problem should repair NOT try to fix?", options: ["A trailing comma", "A numeric string where a number is expected", "priority: \"urgent\" when the enum is low/medium/high", "An unknown extra key"], answer: 2, explanation: "Mapping 'urgent' to some allowed value means guessing the model's intent. Retry with the validator's message instead." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Each retry passes with probability q = 0.85. What is the chance a failed output is fixed within 2 retries?", options: ["0.85", "0.9775", "0.7225", "1.0"], answer: 1, explanation: "1 − (1 − 0.85)² = 1 − 0.0225 = 0.9775. Retries help quickly but never reach certainty." },
    { id: `${t}-4`, topicId: t, skill: "prediction", question: "You enable constrained decoding (JSON grammar). Which error type can still reach your validator?", options: ["Trailing comma", "Truncated JSON", "Missing required field", "None"], answer: 2, explanation: "A grammar that guarantees JSON syntax says nothing about which keys are present. The validator still catches missing fields, wrong types and enum violations." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "In the batch view, why does '+ schema validation' have a lower acceptance rate than 'Parse only'?", options: ["Validation breaks valid outputs", "It rejects outputs that parse but are wrong, which parse-only accepted silently", "It uses more model calls", "Random variation"], answer: 1, explanation: "Parse-only counts wrong objects as accepted (red segment). Validation converts them into explicit rejections, the first step to fixing them with repair and retries." },
  ],
};
