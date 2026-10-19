import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Score a golden set question by question on six RAG metrics, then A/B two configs and find what flipped.", minutes: 15, hasMetrics: false };

const t = "rag-evaluation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "RAG evaluation runs a fixed golden set (questions with known answers and known source documents) through the whole pipeline and scores each stage separately: retrieval (recall@k, reciprocal rank), context assembly (did the fact survive?), generation (faithfulness, correctness) and attribution (citation accuracy). Separate metrics tell you where an answer was lost, not just that it was." },
    { heading: "Metrics in this visualizer", body: "recall@k = relevant chunks in the top-k ÷ relevant chunks in the index. RR = 1 / rank of the first relevant chunk (0 if none in top-k); its mean over questions is MRR. Context hit = the expected fact is in the assembled context. Faithfulness = the answer is supported by the context (an abstention counts as faithful). Correctness = the answer contains the expected fact. Citation accuracy = the cited passage contains the answer sentence." },
    { heading: "A/B: deltas and flips", body: "Comparing two configurations on the same golden set gives per-metric deltas, but averages compress: one broken question out of six is only −0.17. The per-question flip matrix shows which questions changed, which is what you debug and what a regression test should gate on. Note that some changes (rerank on/off here) flip nothing on a small set; that is a coverage gap in the golden set, not proof of safety." },
    { heading: "Where metrics disagree", body: "Context budget 60: recall stays 1.0 but context hit and correctness drop, so the fact was retrieved and then cut. Citation off-by-one: correctness stays 1.0 and only citation accuracy falls. Grounding removed: retrieval is perfect yet faithfulness falls because the model answered from memory. Each pattern points to a different stage." },
    { heading: "Assumptions in this visualizer", body: "Everything runs on the simulated RAG pipeline: five short fictional documents, hashed bag-of-words embeddings, a term-overlap reranker and a rule-based extractive generator. Relevance labels are automatic (a chunk is relevant if it contains the expected fact string) and correctness is substring matching, both far simpler than human or LLM-judge labels. Six questions is a toy golden set; real ones have hundreds." },
  ],
  code: [
    {
      title: "Golden-set metrics (Python)",
      source: `def evaluate(pipeline, golden, k=4):
    rows = []
    for q in golden:
        run = pipeline(q.text)
        rel = [c.id for c in pipeline.index.chunks if q.expect.lower() in c.text.lower()]
        top = [h.id for h in run.reranked[:k]]
        first = next((i + 1 for i, cid in enumerate(top) if cid in rel), None)
        rows.append({
            "recall@k": len(set(top) & set(rel)) / len(rel) if rel else 0.0,
            "rr": 1 / first if first else 0.0,
            "context_hit": q.expect.lower() in run.context_text.lower(),
            "faithful": run.abstained or judge_supported(run.answer, run.context_text),
            "correct": q.expect.lower() in run.answer.lower(),
            "citation": cited_span_supports(run),
        })
    means = {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}
    return rows, means   # means["rr"] is MRR`,
    },
    {
      title: "A/B gate on flips, not just averages (TypeScript)",
      source: `const flips = baseline.rows.flatMap((a, i) =>
  METRICS.filter((m) => candidate.rows[i][m] < a[m]).map((m) => ({ q: a.id, m })));
if (flips.length > 0) {
  throw new Error(\`Regression: \${flips.map((f) => \`\${f.q}.\${f.m}\`).join(", ")}\`);
}`,
    },
  ],
  failureModes: [
    { title: "Averages hide a broken question", symptom: "Mean correctness drops from 1.00 to 0.83 and looks 'fine'.", detect: "Per-question flip matrix: one question went ✓ → ✗ on recall, RR, context and correctness together.", fix: "Gate releases on per-question regressions; investigate every flip.", scenarioId: "rag-chunk-split" },
    { title: "Retrieved but lost in context", symptom: "recall@k = 1 but context hit and correctness fall.", detect: "Compare recall@k with context hit per question.", fix: "Raise the context budget, truncate on sentence boundaries, or put the best chunk first.", scenarioId: "rag-context-truncation" },
    { title: "Stale index passes most of the set", symptom: "Only the question about recently changed docs fails.", detect: "Failing questions cluster on documents updated after the index build date.", fix: "Re-index on document change; add golden questions for recently edited pages.", scenarioId: "rag-stale-index" },
    { title: "Golden set too small to see a change", symptom: "An A/B shows zero flips (rerank on vs off) though the change matters in production.", detect: "No golden question exercises the changed component (here every relevant chunk is already rank 1).", fix: "Add harder questions: ambiguous ones, near-duplicate docs, long documents with many chunks." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "With chunk size 20 at k = 2, q-price has two chunks containing '$49' and only one is in the top-2. What are recall@2 and RR?", options: ["recall 1.0, RR 1.0", "recall 0.5, RR = 1/rank of the one that made it", "recall 0.5, RR 0.5 always", "recall 0, RR 0"], answer: 1, explanation: "Recall counts how many of the relevant chunks were retrieved (1 of 2). RR only looks at the first relevant chunk's rank." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "Baseline vs 'Context budget 60 tokens': which pattern tells you the failure is in context assembly?", options: ["recall falls, correctness stays", "recall stays 1.0 while context hit and correctness fall for q-refund", "citation accuracy falls", "faithfulness falls"], answer: 1, explanation: "The refund chunk is retrieved (recall 1.0) but the budget cut it before the sentence with '30 days of purchase', so the fact never reached the generator." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "Which metric moves when you A/B baseline against 'citation off-by-one'?", options: ["Only citation accuracy", "Correctness and citation accuracy", "Recall@k", "All six"], answer: 0, explanation: "The text is still extracted correctly from context, so correctness and faithfulness hold. Only the label → passage mapping is wrong." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "Rerank on vs off flips no questions here. What is the right conclusion?", options: ["Reranking is useless", "The golden set doesn't exercise reranking (every relevant chunk is already rank 1); add harder questions before concluding", "Reranking hurts", "MRR is broken"], answer: 1, explanation: "A null result on six easy questions is a coverage gap. Absence of flips is only evidence of safety if the golden set contains cases the change affects." },
    { id: `${t}-5`, topicId: t, skill: "metric", question: "Query encoder upgraded to embed-v2 (index still v1). What happens to MRR at k = 4?", options: ["Stays 1.0", "Falls to 0.25: only two questions keep a relevant chunk in the top-4, at ranks 1 and 2", "Falls to 0", "Rises"], answer: 1, explanation: "(1 + 1/2 + 0 + 0 + 0 + 0) / 6 = 0.25. Mismatched embedding spaces scramble similarity for almost every query." },
  ],
};
