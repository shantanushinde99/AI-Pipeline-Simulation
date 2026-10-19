import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Rescore the top-N candidates pair by pair, watch rows re-sort, and weigh nDCG gains against latency.", minutes: 12, hasMetrics: false };

const t = "reranking";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Reranking is a second retrieval stage. A fast first stage (vector or hybrid search) returns a few dozen candidates. A slower, more accurate model, usually a cross-encoder, scores each (query, passage) pair and re-orders them. Only the best few go to the LLM." },
    { heading: "Why a cross-encoder is better and slower", body: "A bi-encoder embeds the query and each document separately, so documents can be pre-computed and searched in milliseconds, but the two texts never interact. A cross-encoder reads both together, so it notices that “refunds are not available” contradicts the question, or that a passage contains the number the user asked for. The price is one model call per candidate at query time." },
    { heading: "Choosing N", body: "The reranker can only reorder what it is given. If the answer is at first-stage rank 7 and N = 5, nothing can rescue it. Each extra candidate adds latency (and cost) roughly linearly, so N is a quality/latency trade-off you tune on a golden set." },
    { heading: "Assumptions in this visualizer", body: "The cross-encoder is simulated: a logistic score over four explicit, hand-weighted features (query-term overlap, an exact two-word phrase match, an intent cue such as a number for “how long” questions, and negation words). Real cross-encoders learn far richer signals. First-stage scores and relevance labels are hand-set. Latency is simulated as 4 ms fixed + 6 ms per pair + 0.3 ms per token. nDCG@3 uses graded labels (answer = 2, partial = 1)." },
  ],
  code: [
    {
      title: "rerank.py: sentence-transformers CrossEncoder",
      source: `from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, n=20, keep=4):
    candidates = vector_index.search(query, top_k=n)          # fast, bi-encoder
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs, batch_size=16)          # one forward pass per pair
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:keep]]`,
    },
    {
      title: "Hosted reranker (TypeScript)",
      source: `const hits = await index.query({ vector: qVec, topK: 25 });
const res = await rerankClient.rerank({
  query,
  documents: hits.map((h) => h.text),
  topN: 4,
});
const context = res.results.map((r) => hits[r.index]);`,
    },
  ],
  failureModes: [
    { title: "Answer cut before reranking", symptom: "Reranking is on, but answers still miss obvious documents.", detect: "Log the first-stage rank of the gold passage; it is often above N.", fix: "Raise N, or improve the first stage (hybrid search) so the answer lands inside N.", scenarioId: "rag-context-truncation" },
    { title: "Negation and contradiction", symptom: "The answer cites a passage that says the opposite (“not available”).", detect: "Faithfulness checks fail; top passages share every keyword but flip the meaning.", fix: "Use a real cross-encoder or NLI-aware reranker; evaluate on negated look-alikes.", scenarioId: "rag-unfaithful" },
    { title: "Latency blow-up", symptom: "p95 latency jumps after enabling reranking.", detect: "Trace shows the rerank span growing linearly with N and passage length.", fix: "Lower N, truncate passages, batch pairs, or use a smaller reranker model." },
    { title: "Reranker trained on a different domain", symptom: "Reranked order is worse than the first stage on your data.", detect: "nDCG after < nDCG before on a labelled golden set.", fix: "Evaluate before shipping; fine-tune or pick a reranker trained on similar queries." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "In the annual-refund preset, what is the smallest N for which the answer passage reaches rank 1 after reranking?", options: ["3", "5", "7", "10"], answer: 2, explanation: "The answer is at first-stage rank 7. The reranker only scores the top N, so N must be at least 7; once included it scores highest (z = 3.0)." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "In the deleted-projects preset, what happens when you turn the negation feature off?", options: ["Nothing changes", "The backups passage that says “cannot be restored” jumps to rank 1", "Latency drops to zero", "All scores become 0.5"], answer: 1, explanation: "That passage matches every query term, the phrase and the number cue. Only the −2.5 negation penalty kept it below the true answer." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Why can nDCG@3 after reranking be lower at N = 5 than at N = 4 in the refund preset?", options: ["Latency penalises nDCG", "Candidate R5 is a look-alike that the reranker scores above a partial match", "nDCG ignores order", "The ideal DCG changes with N"], answer: 1, explanation: "R5 (“Upgrading to an annual plan…”) shares terms and the phrase but is not relevant. Adding it to the pool lets it displace a partially relevant passage from the top 3." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "p95 latency doubled after raising N from 10 to 20. What does the trace most likely show?", options: ["Embedding the query got slower", "The rerank span grew roughly linearly with the number of pairs", "The LLM generated longer answers", "The vector index rebuilt"], answer: 1, explanation: "A cross-encoder runs once per (query, passage) pair, so its cost scales with N (and passage length), exactly like the latency meter here." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How should you verify a new reranker before shipping?", options: ["Check that scores are between 0 and 1", "Compare nDCG@k before and after on a labelled golden set, and measure latency at the chosen N", "Ask the LLM if the order looks right", "Confirm the top result changed"], answer: 1, explanation: "Reranking must earn its latency. Measure ranking quality against labels with and without it, at the N you will run in production." },
  ],
};
