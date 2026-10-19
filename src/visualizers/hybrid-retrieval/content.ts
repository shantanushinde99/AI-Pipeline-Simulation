import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Score docs with BM25 and vectors side by side, then fuse the rankings with RRF or an α blend.", minutes: 14, hasMetrics: false };

const t = "hybrid-retrieval";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Hybrid retrieval runs two searches over the same documents: a keyword (lexical) search such as BM25 and a vector (semantic) search over embeddings. It then fuses the two ranked lists into one." },
    { heading: "Why one method is not enough", body: "BM25 matches exact strings, so rare identifiers (error codes, SKUs, function names) score very high. It finds nothing when the user paraphrases. Embeddings match meaning, so paraphrases work, but an identifier the model never learned carries almost no signal. Each method fails on the queries the other handles well." },
    { heading: "BM25 in one line", body: "For each query term: idf × tf·(k1 + 1) / (tf + k1·(1 − b + b·|D|/avgdl)). IDF rewards rare terms. k1 makes repeated terms saturate instead of growing forever. b decides how much long documents are penalised (b = 0 ignores length)." },
    { heading: "Fusing rankings", body: "Reciprocal Rank Fusion (RRF) adds 1/(k + rank) from every list a document appears in. It ignores raw scores, so BM25's unbounded numbers and cosine's [−1, 1] never need calibrating; k = 60 is the common default. A weighted blend α·norm(BM25) + (1 − α)·cosine uses the scores themselves but needs normalisation and a tuned α." },
    { heading: "Assumptions in this visualizer", body: "The corpus is 8 short, fictional help-centre documents. BM25 is computed exactly on them (Lucene-style IDF). The “embedding” is a hand-built 9-concept lexicon, not a neural model: known words add weight to concept axes, unknown words (like NW-4013 or 429) are ignored. That is a deliberate stand-in that keeps the vector maths visible. Relevance labels are hand-assigned. The weighted blend divides BM25 by its maximum for this query." },
  ],
  code: [
    {
      title: "hybrid.py: BM25 + dense + RRF",
      source: `from rank_bm25 import BM25Okapi
import numpy as np

bm25 = BM25Okapi([tokenize(d) for d in docs], k1=1.2, b=0.75)
doc_vecs = embed(docs)                      # (n_docs, dim), L2-normalised

def hybrid(query, k_rrf=60, top_n=5):
    kw = np.argsort(-bm25.get_scores(tokenize(query)))
    dense = np.argsort(-(doc_vecs @ embed([query])[0]))
    fused = {}
    for ranking in (kw, dense):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0) + 1 / (k_rrf + rank)
    return sorted(fused, key=fused.get, reverse=True)[:top_n]`,
    },
    {
      title: "Elasticsearch / OpenSearch: built-in RRF",
      source: `POST /help-center/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        { "standard": { "query": { "match": { "body": "NW-4013 error" } } } },
        { "knn": { "field": "embedding", "query_vector": [/* ... */], "k": 20, "num_candidates": 100 } }
      ],
      "rank_constant": 60,
      "rank_window_size": 50
    }
  }
}`,
    },
  ],
  failureModes: [
    { title: "Vector-only search misses identifiers", symptom: "Searching an exact error code returns generic error pages.", detect: "Queries containing codes, SKUs or names have low recall; the code is out of the embedding's vocabulary.", fix: "Add BM25 (or a keyword filter) and fuse; boost exact-match fields." },
    { title: "Keyword-only search misses paraphrases", symptom: "“Locked out of my account” never finds the password-reset page.", detect: "Zero BM25 hits or only lexical look-alikes for natural-language questions.", fix: "Add dense retrieval, synonyms or query expansion." },
    { title: "Unnormalised score blending", symptom: "One retriever silently dominates the fused list.", detect: "Changing α barely moves results, or one method's raw scores are 10× the other's.", fix: "Normalise scores per query or switch to RRF, which uses ranks only." },
    { title: "Stale or mismatched index", symptom: "BM25 and vector lists disagree on which documents even exist.", detect: "Compare document ids and versions across the two indexes.", fix: "Index both from the same snapshot in one pipeline.", scenarioId: "rag-stale-index" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "For the query “NW-4013 error”, why does vector search rank the error-code reference (D8) above the certificate page (D2)?", options: ["D8 is longer", "NW-4013 is out of the embedding's vocabulary, so only the word “error” counts", "Cosine favours short documents", "BM25 weights leak into the vector score"], answer: 1, explanation: "The concept embedding has no entry for the identifier, so the query vector is pure “error”, which D8 matches best. BM25 sees NW-4013 as a rare term with high IDF and ranks D2 first." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "With RRF (k = 60), a document ranked 1st by BM25 and 3rd by vectors scores:", options: ["1/61 + 1/63 ≈ 0.0323", "1/1 + 1/3 ≈ 1.333", "(1 + 3) / 60", "1/64"], answer: 0, explanation: "RRF adds 1/(k + rank) per list: 1/61 + 1/63 = 0.01639 + 0.01587 ≈ 0.0323." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "Set the BM25 b slider to 0. What changes?", options: ["IDF becomes 0", "Document length no longer affects BM25 scores", "Term frequency is ignored", "Vector scores change"], answer: 1, explanation: "b scales the |D|/avgdl length factor. At b = 0 the denominator becomes tf + k1, identical for long and short documents." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "In the mixed preset, switch to the weighted blend with α = 0.5. D6 (API access keys) slips into the top 2. Why?", options: ["D6 is relevant", "D6's BM25 score is high and normalised BM25 dominates the blend at this α", "RRF is still running", "D6 has the highest cosine"], answer: 1, explanation: "D6 matches “api” and “client”, so its max-normalised BM25 is 0.84. Blended scores depend on α and on normalisation; RRF ranks it third because it is only 5th in the vector list." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Users search both error codes and plain-language questions. What should you ship?", options: ["Only a bigger embedding model", "Only BM25 with stemming", "BM25 and vector search fused with RRF, evaluated on both query types", "Increase top-k to 50"], answer: 2, explanation: "Each method covers the other's blind spot. RRF needs no score calibration, and a golden set with both query types shows whether fusion actually helps." },
  ],
};
