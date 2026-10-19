import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Toggle trimming, caching, routing, batching and caps; watch the cost waterfall and quality move.", minutes: 12, hasMetrics: false };

const t = "cost-optimization";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "LLM cost is token arithmetic: cost per request = input tokens × input price + output tokens × output price, and monthly cost = that × requests. Cost optimisation means shrinking one of those factors: fewer input tokens (trimming), fewer billed requests (caching), cheaper tokens (routing to a smaller model, batch discounts), or fewer output tokens (caps and concise answers)." },
    { heading: "Why output tokens matter", body: "Output tokens are usually several times more expensive than input tokens, and long answers on hard requests dominate the bill even when they are a small share of traffic. Measure cost per difficulty segment, not just the average." },
    { heading: "Cost is a trade-off with quality", body: "Caching and batching mostly save money for free (if the cache is fresh and the work can wait). Routing, aggressive trimming and tight output caps save money by spending quality. Plot each configuration as a point on cost vs quality and choose from the frontier: the configs nothing else beats on both axes." },
    { heading: "Assumptions in this visualizer", body: "All prices are hypothetical ($3/$15 per 1M tokens for a 'large' model, $0.25/$1.25 for a 'small' one) and do not describe any real provider. The workload (2M requests/month, 1,800 input tokens, three difficulty levels) and the quality scores (including the trimming and truncation penalties) are made-up numbers chosen to show the shape of the trade-off. Cache hits are assumed free and as good as a fresh answer; cache infrastructure cost is ignored. No model is called." },
  ],
  code: [
    {
      title: "cost.py — monthly cost from token counts",
      source: `PRICES = {"large": (3.00, 15.00), "small": (0.25, 1.25)}  # $ per 1M tokens (hypothetical)

def request_cost(model, in_tokens, out_tokens):
    p_in, p_out = PRICES[model]
    return (in_tokens * p_in + out_tokens * p_out) / 1_000_000

def monthly_cost(segments, hit_rate=0.0, batch_share=0.0, batch_discount=0.5):
    """segments: [{'requests', 'model', 'in_tokens', 'out_tokens'}]"""
    total = 0.0
    for s in segments:
        billed = s["requests"] * (1 - hit_rate)
        total += billed * request_cost(s["model"], s["in_tokens"], s["out_tokens"])
    return total * (1 - batch_discount * batch_share)`,
    },
    {
      title: "router.ts — send easy requests to the small model",
      source: `type Model = "small" | "large";

export function route(prompt: string, difficulty: (p: string) => number, cutoff = 0.4): Model {
  // difficulty() might be a cheap classifier or heuristics (length, task type, tools needed)
  return difficulty(prompt) < cutoff ? "small" : "large";
}

export async function answer(prompt: string, cache: Map<string, string>, call: (m: Model, p: string) => Promise<string>) {
  const key = prompt.trim().toLowerCase();
  const hit = cache.get(key);
  if (hit) return hit;                       // no tokens billed
  const out = await call(route(prompt, (p) => Math.min(1, p.length / 2000)), prompt);
  cache.set(key, out);
  return out;
}`,
    },
  ],
  failureModes: [
    { title: "Over-aggressive routing", symptom: "The bill drops sharply, then complaints and escalations rise on complex tasks.", detect: "Track quality per difficulty segment and per model, not only overall averages.", fix: "Route only segments where the small model's measured quality is close to the large model's; fall back to the large model on low confidence." },
    { title: "Output caps truncating answers", symptom: "Answers end mid-sentence; JSON outputs fail to parse.", detect: "Monitor the share of responses that hit max_tokens (finish reason 'length').", fix: "Ask for concise answers in the prompt and size caps per task from the output-length distribution instead of one global cap.", scenarioId: "rag-context-truncation" },
    { title: "Stale or wrong cache hits", symptom: "Users get outdated answers, or an answer meant for a different question.", detect: "Sample cache hits and re-evaluate them; track hit age and invalidation events.", fix: "Key the cache on normalised input plus model/prompt version; set TTLs and invalidate on data changes." },
    { title: "Trimming away the context", symptom: "Cost per request fell, but grounded answers became vague or wrong.", detect: "Run the golden set before and after a prompt change and compare quality, not just tokens.", fix: "Trim boilerplate and redundant examples first; keep retrieved evidence; validate each cut with an eval." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "A hard request has 1,800 input tokens and 1,100 output tokens on the large model ($3/M in, $15/M out). What does it cost?", options: ["$0.0219", "$0.0087", "$0.0522", "$0.0054"], answer: 0, explanation: "1,800 × 3 / 1M = $0.0054 and 1,100 × 15 / 1M = $0.0165; the sum is $0.0219. Output is three quarters of the bill.", skill: "metric" },
    { id: `${t}-2`, topicId: t, question: "With every other lever off, you raise the cache hit rate from 0% to 50%. What happens to cost and quality in this visualizer?", options: ["Cost halves; quality is unchanged", "Cost halves; quality halves", "Cost is unchanged", "Cost drops 25%"], answer: 0, explanation: "Hits are not billed, so billed requests fall by half. The model assumes cached answers are as good as fresh ones, so quality stays put (in reality: watch staleness).", skill: "prediction" },
    { id: `${t}-3`, topicId: t, question: "You switch routing to 'all → small'. Cost drops ~90% but quality falls to about 0.83. Which segment is responsible for most of the drop?", options: ["Hard requests (small model scores 0.50 there)", "Easy requests", "Cached requests", "Batched requests"], answer: 0, explanation: "On easy requests the small model is almost as good (0.93 vs 0.95). On hard ones it falls from 0.88 to 0.50, which drags the average.", skill: "diagnosis" },
    { id: `${t}-4`, topicId: t, question: "Which pair of levers in this visualizer saves money without lowering the quality score?", options: ["Caching and batching", "Routing and output caps", "Trimming beyond 30% and caps", "Routing and trimming beyond 30%"], answer: 0, explanation: "Caching removes billed requests and batching discounts deferrable ones; neither changes the answers. The others cut tokens or model capability.", skill: "parameter" },
    { id: `${t}-5`, topicId: t, question: "On the frontier step your dot sits below and to the right of the green line. What should you do?", options: ["Switch to a frontier config that is cheaper at the same or better quality", "Nothing, any point is fine", "Disable caching", "Raise prices"], answer: 0, explanation: "A point inside the frontier is dominated: another configuration is cheaper and at least as good. The step card names one to try.", skill: "verification" },
  ],
};
