import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Grow k one rank at a time and watch precision, recall, MRR, nDCG and context cost move.", minutes: 10, hasMetrics: false };

const t = "top-k-retrieval";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A retriever scores every chunk in the index against the query and returns them sorted. Top-k retrieval keeps only the first k and passes them to the LLM as context. k is a cutoff: rank k is in, rank k + 1 is invisible to the model." },
    { heading: "Measuring a cutoff", body: "precision@k is the share of the k retrieved chunks that are relevant. recall@k is the share of all relevant chunks that made it into the top k. Reciprocal rank is 1 / (rank of the first relevant chunk); averaged over many queries it becomes MRR. nDCG@k rewards relevant chunks more the higher they sit, uses graded labels (answer = 2, partial = 1), and divides by the best possible ordering so 1.0 means a perfect ranking." },
    { heading: "Why k is a trade-off", body: "Raising k can only add relevant chunks, so recall never goes down. But every extra chunk costs prompt tokens (latency and money) and usually lowers precision, which means more distracting text for the model. When the answer sits at rank 7, k = 5 retrieves nothing useful; the fix is a better ranking (hybrid search, reranking), not just a bigger k." },
    { heading: "Assumptions in this visualizer", body: "Retriever scores and relevance labels are hand-set for teaching, not produced by a real embedding model. Token counts are a toy estimate of about 1.3 tokens per word. Metrics are computed exactly from the labels shown. With a single query, MRR equals the reciprocal rank of that query. nDCG always uses the graded labels; the relevance threshold only changes precision, recall and RR." },
  ],
  code: [
    {
      title: "metrics.py: ranking metrics at k",
      source: `import math

def precision_at_k(grades, k, thr=1):
    return sum(g >= thr for g in grades[:k]) / k

def recall_at_k(grades, k, total_relevant, thr=1):
    return sum(g >= thr for g in grades[:k]) / total_relevant

def reciprocal_rank(grades, k, thr=1):
    for i, g in enumerate(grades[:k]):
        if g >= thr:
            return 1 / (i + 1)
    return 0.0

def dcg(grades, k):
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))

def ndcg_at_k(grades, k, pool):
    ideal = dcg(sorted(pool, reverse=True), k)
    return dcg(grades, k) / ideal if ideal else 0.0

grades = [0, 1, 1, 0, 0, 0, 2, 0, 0, 0]   # answer at rank 7
print(recall_at_k(grades, 5, 1, thr=2))     # 0.0
print(recall_at_k(grades, 7, 1, thr=2))     # 1.0`,
    },
    {
      title: "Choosing k against a token budget (TypeScript)",
      source: `const hits = await index.query({ vector: q, topK: 20 });
let used = 0;
const context = [];
for (const h of hits) {
  const cost = countTokens(h.text);
  if (context.length >= k || used + cost > budget) break;
  context.push(h);
  used += cost;
}`,
    },
  ],
  failureModes: [
    { title: "k too small for a buried answer", symptom: "The model says “I don't know” although the document exists.", detect: "Log the rank of the gold chunk per query; recall@k on a golden set is low while recall@20 is high.", fix: "Improve ranking (hybrid search, reranker) or retrieve more candidates and rerank down to k." },
    { title: "k too large", symptom: "Answers get slower, pricier and sometimes cite the wrong chunk.", detect: "precision@k falls steadily while recall@k is flat; prompt tokens per request climb.", fix: "Lower k, add a score threshold, or rerank and keep the best few.", scenarioId: "rag-context-truncation" },
    { title: "Relevant chunk never retrieved", symptom: "recall@k stays below 1 at every k.", detect: "Find the gold chunk's true rank beyond the candidate list; check parsing and chunking.", fix: "Fix chunk boundaries or parsing, add keyword search for identifiers.", scenarioId: "rag-chunk-split" },
    { title: "Metric mismatch", symptom: "precision@k looks fine but users still get poor answers.", detect: "Compare with nDCG: relevant chunks sit low, or only partial matches are counted.", fix: "Use graded labels and a strict relevance threshold that requires the answer itself." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "In the “answer buried at rank 7” preset (strict threshold), what is the smallest k with recall@k = 1?", options: ["3", "5", "7", "10"], answer: 2, explanation: "Only the chunk at rank 7 contains the answer, so recall stays 0 until k reaches 7." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "In the same preset, what happens to precision@k when k goes from 7 to 8?", options: ["It rises", "It falls from 1/7 to 1/8", "It stays at 1/7", "It becomes 0"], answer: 1, explanation: "Rank 8 is not relevant: the numerator stays 1 while k grows, so 1/7 ≈ 0.143 drops to 1/8 = 0.125." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "Switch to the lenient threshold (partial counts). What happens to reciprocal rank in the rank-7 preset?", options: ["Stays 1/7", "Becomes 1/2, because rank 2 is a partial match", "Becomes 1", "Becomes 0"], answer: 1, explanation: "Rank 2 (monthly refunds) is labelled partial. Under the lenient threshold it is the first relevant hit, so RR = 1/2, even though it doesn't answer the question." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "In the SSO preset recall@10 is stuck at 0.5. Why?", options: ["k is too small", "The threshold is too strict", "A required chunk (the DNS step) was never in the candidate list", "nDCG is broken"], answer: 2, explanation: "Recall counts every relevant chunk, including the one the retriever ranked 14th. No k within the list can recover it; fix retrieval, not k." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "The answer often sits around rank 7 and you can only afford ~60 context tokens. What is the best fix?", options: ["Set k = 10", "Retrieve 20 candidates, rerank, and keep the top 2", "Lower the temperature", "Use a stricter relevance threshold"], answer: 1, explanation: "Raising k pays for every distractor above the answer. Reranking a wider candidate set moves the answer up so a small k is enough." },
  ],
};
