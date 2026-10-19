import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Walk a JSON Schema node by node over bad tool arguments, then see the error message and repair.", minutes: 12, hasMetrics: false };

const t = "tool-argument-validation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Before a runtime executes a tool call it checks the arguments against the tool's JSON Schema: types, required fields, allowed values (enum), numeric ranges, string patterns, and whether unknown fields are allowed. The validator walks the schema like a tree, so every failure comes with an exact path such as /guest/email." },
    { heading: "Why it matters", body: "Model output is untrusted input. Without validation, a wrong type or out-of-range number reaches a real system: a booking for 30 nights, an email sent to '42'. With validation, the bad call is stopped and turned into feedback the model can act on." },
    { heading: "Strict vs. coercion, all errors vs. first error", body: "Coercion converts lossless scalars (\"3\" → 3, 42 → \"42\") before checking. It rescues harmless formatting slips but can hide real mistakes, and it never makes a bad value good: 42 becomes \"42\", which is still not an email. Reporting all errors lets the model fix everything in one round trip; stopping at the first error (ajv's default) hides the rest until the next attempt." },
    { heading: "Model-friendly errors and repair", body: "Return errors as short, specific instructions per field (\"nights: must be at most 14, you sent 30\"), plus what to do when a value is unknown: ask the user. A repair can fix form (types, casing, unknown fields) but cannot invent missing facts, and a repair that clamps a number passes validation while changing what the user asked for, so re-validate and watch for meaning changes." },
    { heading: "Assumptions in this visualizer", body: "The validator is a small, real JSON Schema subset implemented here (type, required, additionalProperties, enum, minimum/maximum, minLength, pattern) with ajv-like error objects; it is not ajv itself. The candidate arguments are fixed examples, and the 'repair' is a deterministic rule set standing in for the model's next attempt. No model is called." },
  ],
  code: [
    {
      title: "Validate tool arguments with ajv and return model-friendly errors (TypeScript)",
      source: `import Ajv from "ajv";
const ajv = new Ajv({ allErrors: true, coerceTypes: false });   // strict, report every error
const validate = ajv.compile(createBookingSchema);

function checkArgs(args: unknown) {
  if (validate(args)) return { ok: true as const, args };
  const lines = validate.errors!.map((e) => {
    const field = e.instancePath.slice(1).replace(/\\//g, ".") || String(e.params.missingProperty ?? "");
    switch (e.keyword) {
      case "maximum": return \`- \${field}: must be at most \${e.params.limit} (you sent \${JSON.stringify(e.data)})\`;
      case "enum": return \`- \${field}: must be one of \${e.params.allowedValues.join(", ")}\`;
      case "required": return \`- \${e.params.missingProperty}: required field is missing\`;
      case "additionalProperties": return \`- \${e.params.additionalProperty}: not a field of this tool; remove it\`;
      default: return \`- \${field}: \${e.message}\`;
    }
  });
  return { ok: false as const, toolResult: ["create_booking was NOT called.", ...lines,
    "Fix only these fields. If a value is unknown, ask the user instead of guessing."].join("\\n") };
}`,
    },
    {
      title: "The same contract with Pydantic (Python)",
      source: `from typing import Literal
from pydantic import BaseModel, EmailStr, Field, ValidationError, ConfigDict

class Guest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    email: EmailStr

class CreateBooking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)   # strict=False would coerce "3" -> 3
    guest: Guest
    check_in: str = Field(pattern=r"^\\d{4}-\\d{2}-\\d{2}$")
    nights: int = Field(ge=1, le=14)
    room: Literal["single", "double", "suite"]
    breakfast: bool | None = None

try:
    CreateBooking.model_validate(args)
except ValidationError as e:
    tool_result = "\\n".join(f"- {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())`,
    },
  ],
  failureModes: [
    { title: "No validation before execution", symptom: "Odd records in production: 30-night bookings, emails to '42'.", detect: "Tool spans with no validation step; errors thrown deep inside the tool instead of at the boundary.", fix: "Validate every call against the tool's JSON Schema before executing; reject unknown fields.", scenarioId: "agent-invalid-args" },
    { title: "Unhelpful error messages", symptom: "The model retries with the same or a different wrong value.", detect: "Repeated validation failures on the same field across attempts.", fix: "Return per-field messages with the constraint and the received value; ask the user for missing facts." },
    { title: "Repair that changes intent", symptom: "Validation passes but the action differs from the request (14 nights instead of 30).", detect: "Diff between first and repaired arguments on user-provided fields.", fix: "Never auto-clamp user values; confirm with the user when a repair changes meaning." },
    { title: "Stop-at-first-error ping-pong", symptom: "Many round trips to get one call right; latency and cost spike.", detect: "Several consecutive validation failures, each on a different field.", fix: "Enable all-errors reporting so the model can fix every field at once; cap repair attempts." },
    { title: "Over-eager coercion", symptom: "Wrong values accepted silently (\"1e3\" → 1000 nights, true → \"true\").", detect: "Coercion counters in logs; compare raw vs. coerced arguments.", fix: "Use strict mode for user-impacting fields, or coerce only specific, known-safe conversions." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "With 'Wrong type (nights as a string)' and coercion ON, what happens?", options: ["Validation fails on nights", "\"3\" is coerced to 3 and the call is valid", "The repair step clamps nights", "The validator stops at the root"], answer: 1, explanation: "\"3\" converts losslessly to the integer 3, so every check passes. Turn coercion off to see the type error and the repair instead." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "With coercion ON, the nested candidate (guest.email: 42) still fails. Which keyword fails?", options: ["type", "required", "pattern", "additionalProperties"], answer: 2, explanation: "Coercion turns 42 into \"42\", so the type check passes, but \"42\" does not match the email pattern. Coercion fixes form, not content." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "For 'Several mistakes at once', how many errors does the first pass report with 'Report all errors' OFF?", options: ["0", "1", "4", "5"], answer: 1, explanation: "Stop-at-first reports a single error and never checks the remaining nodes. With it ON the same arguments show 5 errors, all fixable in one round trip." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "Repairing 'nights: 30' by clamping it to 14 passes validation. What should the system do?", options: ["Execute: the call is valid", "Confirm with the user, because the repair changed what they asked for", "Disable the maximum", "Switch to coercion mode"], answer: 1, explanation: "A valid call is not necessarily the requested call. Clamping changes intent; the visualizer flags this as 'valid, but not what the user asked for'." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "For a missing check_in, the simulated repair asks the user instead of filling a date. Why is that correct?", options: ["Dates are hard to format", "The value is a fact only the user knows; inventing one would pass validation with wrong data", "The schema has no check_in field", "Coercion is off"], answer: 1, explanation: "Validation can only check form. When a required fact is missing, the right repair is to ask, and to verify that the re-validated call uses user-provided values." },
  ],
};
