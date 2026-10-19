import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Score a request against every tool, then watch vague descriptions and huge catalogues pick wrong.", minutes: 10, hasMetrics: false };

const t = "tool-selection";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Before a model can call a tool it has to choose one. It sees each tool's name, description and parameter schema in its prompt and picks the one that best fits the request. Everything it knows about a tool comes from that text, so tool descriptions are effectively part of your prompt." },
    { heading: "What makes a good description", body: "Say what the tool covers in the words users actually use, list the inputs it needs, and say what it is NOT for. Two tools with copy-pasted or vague descriptions are indistinguishable: the model has to guess, and ordering or chance decides." },
    { heading: "Why big catalogues degrade", body: "Every tool definition costs prompt space. With dozens of tools, more of them overlap, margins between the best and second-best tool shrink, and in practice details get truncated or ignored. Common fixes: fewer, broader tools; a retrieval step that pre-selects the top-k relevant tools; namespacing; and evaluating selection accuracy on a labelled request set." },
    { heading: "Assumptions in this visualizer", body: "The selector is a transparent rule-based scorer, not an LLM: score = 1.0 × keyword overlap + 0.3 × argument fit − 1.0 × 'Not for' penalty, ties broken by catalogue order. Words stand in for tokens, and when the catalogue exceeds the budget each description is cut to an equal share of words. Real models are more robust to wording but fail in the same directions: ambiguous descriptions and crowded catalogues make wrong picks more likely." },
  ],
  code: [
    {
      title: "Tool definitions that disambiguate (Anthropic-style tools list, Python)",
      source: `tools = [
    {
        "name": "search_handbook",
        "description": (
            "Search the internal employee handbook for HR policy: vacation days, PTO, "
            "benefits, expenses and holidays. Not for: public web facts or news."
        ),
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "web_search",
        "description": "Search the public internet for news and facts. Not for: internal company policy.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
]

# With many tools, pre-select a relevant subset per request instead of sending all of them.
def tools_for(request: str, registry, k=8):
    scored = sorted(registry, key=lambda t: similarity(embed(request), t.embedding), reverse=True)
    return [t.definition for t in scored[:k]]`,
    },
    {
      title: "Measure selection accuracy on labelled requests (TypeScript)",
      source: `const cases = [
  { request: "How many vacation days do I get per year?", expected: "search_handbook" },
  { request: "What's the weather in Paris tomorrow?", expected: "get_weather" },
];
let correct = 0;
for (const c of cases) {
  const picked = await chooseTool(c.request, tools);   // your model call
  if (picked === c.expected) correct++;
  else console.warn("wrong tool", c.request, { picked, expected: c.expected });
}
console.log(\`tool-selection accuracy: \${correct}/\${cases.length}\`);`,
    },
  ],
  failureModes: [
    { title: "Copy-pasted or vague descriptions", symptom: "The agent uses the web for internal policy questions (or vice versa).", detect: "Two tools with near-identical descriptions; selection accuracy drops on requests that need one of them.", fix: "Rewrite descriptions with scope, example inputs and a 'Not for' clause; re-run the labelled set." },
    { title: "Too many tools in one prompt", symptom: "Accuracy falls as tools are added, even though each tool works.", detect: "Accuracy vs. catalogue size on a fixed eval set; long tool lists close to the context budget.", fix: "Retrieve the top-k tools per request, merge overlapping tools, or route to specialised sub-agents." },
    { title: "Tool chosen but cannot be called", symptom: "Right tool name, missing or invented arguments.", detect: "Validation errors on required parameters right after selection.", fix: "Describe required inputs in the description; ask the user for missing values instead of guessing.", scenarioId: "agent-invalid-args" },
    { title: "Fragile margins", symptom: "Small rewordings of the same request flip the chosen tool.", detect: "Paraphrase tests: the same intent phrased 3–5 ways should pick the same tool.", fix: "Strengthen distinguishing words in descriptions; add paraphrases to the eval set." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "With the overlapping catalogue, 'How many vacation days do I get per year?' goes to web_search. Why?", options: ["web_search has a higher keyword overlap", "Both tools score exactly the same, and the tie goes to the tool listed first", "The request has no content words", "search_handbook has a missing required argument"], answer: 1, explanation: "Identical descriptions give identical scores (0.30 each). The tie-break is list order, which is an accident, not understanding." },
    { id: `${t}-2`, topicId: t, skill: "fix", question: "Which change fixes the vacation request in the overlapping catalogue?", options: ["Raise the prompt budget", "Clarify both descriptions: scope plus a 'Not for' clause", "Add more tools", "Lower the argument-fit weight"], answer: 1, explanation: "Clarified, search_handbook matches vacation/day/year (overlap 1.0) and web_search's 'Not for: vacation' subtracts 1, so the right tool wins by a wide margin." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "With 24 tools and a 160-word budget, each description keeps 6 words. What is the smallest budget (to the nearest 10) that removes truncation?", options: ["170", "200", "260", "400"], answer: 2, explanation: "The 24 descriptions total 251 words; any budget of at least 251 sends them in full. Use the 'Raise the budget' button to see the vacation request return to search_handbook." },
    { id: `${t}-4`, topicId: t, skill: "prediction", question: "In the 24-tool catalogue at 160 words, what happens to the weather request?", options: ["get_weather wins clearly", "get_weather and weather_alerts tie and alphabetical order decides", "web_search wins", "No tool scores above 0"], answer: 1, explanation: "Both truncated descriptions keep 'weather' and both can fill city=Paris, so they tie at 0.63. The pick is right only by luck of ordering." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "You rewrote some descriptions. How do you know selection improved?", options: ["Try one request in the playground", "Re-run a labelled set of requests and compare accuracy before and after", "Check that descriptions are longer", "Count the tools"], answer: 1, explanation: "One demo query can't show regressions elsewhere. The visualizer shows accuracy on 4 sample requests for this reason." },
  ],
};
