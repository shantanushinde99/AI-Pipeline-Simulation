import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Grow a radius ring to find exact nearest neighbours, and see cosine, dot and L2 rankings disagree.", minutes: 10, hasMetrics: false };

const t = "similarity-search";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Similarity search finds the stored vectors closest to a query vector. Exact (brute-force) k-nearest-neighbour search scores the query against every stored vector, sorts, and keeps the top k. It is the ground truth that approximate indexes (HNSW, IVF) are measured against, and it is fast enough for collections up to a few hundred thousand vectors." },
    { heading: "Cosine, dot product and L2", body: "Cosine similarity compares direction only. Dot product is cosine times both lengths, so long vectors win. L2 (Euclidean) distance measures straight-line distance, so short vectors near the origin can win. On raw vectors the three can produce different top-k lists. After normalizing every vector to length 1 they agree: dot equals cosine and squared L2 equals 2 − 2·cosine. Use the metric your embedding model was trained for; most text embedders expect cosine (or dot on normalized vectors)." },
    { heading: "Ties, k and near-duplicates", body: "When two items have the same score at the k-th position, only the tie-break rule decides which one is kept, and different engines break ties differently, so results can differ between systems with identical data. Near-duplicate documents (the same page indexed from mirrors) sit next to each other in vector space and can fill all k slots, crowding out other sources. Deduplicate at indexing time or collapse results per source." },
    { heading: "Assumptions in this visualizer", body: "Vectors are hand-placed 2-D points, not real embeddings (real ones have hundreds of dimensions). The search is exact brute force over all points. Ties are broken by item id alphabetically. “Collapse near-duplicates” keeps the best item per source label, a simple stand-in for real near-duplicate detection. The expanding ring always uses L2 distance on raw vectors; the rank lists show all three metrics." },
  ],
  code: [
    {
      title: "Exact kNN with NumPy",
      source: `import numpy as np

def knn(q, X, k, metric="cosine"):
    if metric == "cosine":
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        scores = Xn @ (q / np.linalg.norm(q))
    elif metric == "dot":
        scores = X @ q
    else:                                   # L2: smaller is better, so negate
        scores = -np.linalg.norm(X - q, axis=1)
    # stable sort on (-score, index) makes ties deterministic
    order = np.lexsort((np.arange(len(X)), -scores))
    return order[:k], scores[order[:k]]`,
    },
    {
      title: "Collapse near-duplicates per source (TypeScript)",
      source: `function topKOnePerSource(ranked: { id: string; source: string; score: number }[], k: number) {
  const seen = new Set<string>();
  const out = [];
  for (const r of ranked) {               // ranked best-first
    if (seen.has(r.source)) continue;     // a copy of something already kept
    seen.add(r.source);
    out.push(r);
    if (out.length === k) break;
  }
  return out;
}`,
    },
  ],
  failureModes: [
    { title: "Metric mismatch with the embedding model", symptom: "Retrieval favours long, generic documents (dot product on unnormalized vectors) or tiny fragments (L2).", detect: "Compare top-k under cosine and under the index metric on a golden set; large disagreement means a mismatch.", fix: "Use the metric the embedder was trained with, or normalize vectors so all three metrics agree." },
    { title: "Near-duplicates crowd out the top-k", symptom: "All k results are copies of one page; the answer needs a second source and never gets retrieved.", detect: "Count distinct sources (or document ids) per result list; alert when it drops below k.", fix: "Deduplicate at index time (hash or MinHash), or collapse to one result per source before the cutoff.", scenarioId: "rag-chunk-split" },
    { title: "Unstable ties at the cutoff", symptom: "The same query returns different documents on different replicas or after a re-index.", detect: "Log whether the k-th and (k+1)-th scores are equal (within epsilon).", fix: "Add a deterministic tie-break (document id), or retrieve k+margin and rerank." },
    { title: "k too small", symptom: "The relevant chunk is ranked k+1 and the model says it does not know.", detect: "Measure recall@k on a golden set for several k values.", fix: "Raise k and rely on reranking or context assembly to trim.", scenarioId: "rag-context-truncation" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "On the “Mixed lengths” preset with raw vectors, which metric leaves the long vector C out of its top 3?", options: ["Cosine", "Dot product", "L2 distance", "None of them"], answer: 2, explanation: "C points almost exactly along q, so cosine ranks it #1 and dot product (which also rewards length) ranks it #1 by a wide margin. L2 only sees that C is far from q, so short nearby vectors like A and B win.", skill: "prediction" },
    { id: `${t}-2`, topicId: t, question: "You switch on “Index stores normalized vectors”. What happens to the three rank lists?", options: ["They become identical", "Only cosine changes", "L2 reverses order", "Dot product becomes undefined"], answer: 0, explanation: "On unit vectors dot = cosine and ‖a − b‖² = 2 − 2·cos, so all three are monotone in cosine and sort the same way.", skill: "parameter" },
    { id: `${t}-3`, topicId: t, question: "On the “Tie at the cutoff” preset with k = 3, why is C kept and D dropped?", options: ["C is closer to q", "C has a higher cosine", "The id tie-break prefers C", "D is a near-duplicate"], answer: 2, explanation: "C and D are mirror images around q, so every metric scores them identically. Only the alphabetical tie-break separates them; another engine might keep D.", skill: "diagnosis" },
    { id: `${t}-4`, topicId: t, question: "On the “Near-duplicate crowding” preset, what fixes the low distinct-source count?", options: ["Increase k to 4", "Switch to dot product", "Collapse near-duplicates (one per source)", "Normalize vectors"], answer: 2, explanation: "The FAQ copies all sit together, so any metric ranks them first. Keeping only the best item per source frees slots for the guide and policy.", skill: "fix" },
    { id: `${t}-5`, topicId: t, question: "Exact kNN over N vectors of dimension d costs how much per query?", options: ["O(log N)", "O(k)", "O(N·d)", "O(d²)"], answer: 2, explanation: "Brute force computes one d-dimensional distance per stored vector: the scan step draws one line per vector.", skill: "metric" },
  ],
};
