import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch six retrieval faults push the needed chunk out of the top-k, then watch each fix pull it back.", minutes: 14, hasMetrics: false };

const t = "retrieval-failures";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Most wrong RAG answers are retrieval failures: the chunk that holds the answer exists, but it is not among the top-k chunks handed to the generator. The generator then answers from the wrong evidence or says it can't find the answer. Each case here shows the needed chunk's rank before the fault, after it, and after the fix." },
    { heading: "Six common causes", body: "Vocabulary mismatch (the user's words are not the docs' words). Chunk boundary splits (the fact is cut away from the words that match the query). Stale index (the source changed, the index did not). Near-duplicate crowding (copies of one page take every top-k slot). Wrong metadata filter (the right document is removed before scoring). Embedding model mismatch (queries and chunks are embedded by different models, so similarity is noise)." },
    { heading: "Why it matters", body: "Retrieval failures look like generation failures: the answer is fluent and confident. Without a retrieval metric (recall@k of the needed chunk on a golden set) you will tune prompts for a problem that lives in the index. Always ask first: was the right chunk in the context?" },
    { heading: "Assumptions in this visualizer", body: "Embeddings are a toy hashed bag-of-words model from the RAG pipeline visualizer (embed-v1/embed-v2), not a neural encoder, so 'semantic' matches only happen through shared words. The corpus is five short help-center documents. The query rewrite, filters, dedupe threshold (cosine ≥ 0.95) and version dates are hand-written to illustrate each failure. Every rank and score shown is computed from those vectors." },
  ],
  code: [
    {
      title: "recall_at_k.py — find which failure you have",
      source: `def recall_at_k(golden, retrieve, k=3):
    """golden: list of (query, needed_chunk_id). Returns the hit rate and the misses."""
    misses = []
    for query, needed in golden:
        ids = [c.id for c in retrieve(query, k=50)]
        rank = ids.index(needed) + 1 if needed in ids else None
        if rank is None or rank > k:
            misses.append((query, needed, rank))   # None = filtered or not indexed
    return 1 - len(misses) / len(golden), misses

# Triage each miss:
#   rank None + chunk in index  -> metadata filter
#   rank None + chunk missing   -> stale index / ingestion
#   rank > k, low overlap       -> vocabulary mismatch -> query rewrite / hybrid
#   top-k all near-identical    -> dedupe / MMR
#   all scores suddenly low     -> embedding model mismatch`,
    },
    {
      title: "guards.ts — cheap assertions that catch two of these at query time",
      source: `export function search(index: Index, query: string, k: number) {
  if (index.embedModel !== QUERY_MODEL) throw new Error(\`model mismatch: \${index.embedModel} vs \${QUERY_MODEL}\`);
  const candidates = index.filter(query.filters);
  metrics.histogram("retrieval.filter_kept", candidates.length);   // alert when a filter empties a doc set
  const ranked = candidates.rank(embed(query.text, QUERY_MODEL));
  return mmr(ranked, { k, lambda: 0.7 });                          // diversity against near-duplicates
}`,
    },
  ],
  failureModes: [
    { title: "Chunk boundary split", symptom: "The answer cites the right page but misses the key step.", detect: "Look up the needed chunk: does it share terms with the query? Check whether the fact sits at a chunk edge.", fix: "Sentence-aware chunking, overlap, or parent/child retrieval.", scenarioId: "rag-chunk-split" },
    { title: "Stale index", symptom: "Answers quote last quarter's numbers.", detect: "Compare each chunk's doc version/updatedAt with the source of truth; alert on index age.", fix: "Re-index on change; store versions per chunk.", scenarioId: "rag-stale-index" },
    { title: "Filter removes the right document", symptom: "'I couldn't find this' for a page that exists.", detect: "Log how many candidates each filter removes; run golden queries with and without filters.", fix: "Validate and normalize metadata at ingestion." },
    { title: "Near-duplicates fill the context", symptom: "Multi-part questions get half an answer.", detect: "Count distinct source documents in the top-k.", fix: "Deduplicate at ingestion (hash/MinHash) and use MMR at query time." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "In the 'Wrong metadata filter' case, why can't raising top-k to 5 fix the problem?", options: ["The scores are too low", "The needed chunk is removed before ranking, so it has no rank at all", "k is capped at 3", "The generator ignores chunks after #3"], answer: 1, explanation: "Filters run before scoring. A filtered-out chunk is never ranked, so no k is large enough. The fix is to correct the metadata or the filter.", skill: "diagnosis" },
    { id: `${t}-2`, topicId: t, question: "Select 'Vocabulary mismatch' and set top-k to 5. After the fault, is the needed chunk inside the top-k?", options: ["Yes, at #5", "No, it is at #9", "It is filtered out", "It is missing from the index"], answer: 1, explanation: "With the user's wording ('money back', 'yearly') the refund chunk falls to #9 of 14. Even k = 5 misses it; only the rewrite brings it back to #2.", skill: "prediction" },
    { id: `${t}-3`, topicId: t, question: "Which observation best distinguishes an embedding model mismatch from a vocabulary mismatch?", options: ["The needed chunk drops a few ranks", "Similarity scores for every query collapse at once, including the top-1", "Only refund questions fail", "The index is older than the docs"], answer: 1, explanation: "A model mismatch makes all similarities noise: top-1 similarity falls from 0.50 to 0.14 here. Vocabulary mismatch affects only queries phrased differently from the docs.", skill: "diagnosis" },
    { id: `${t}-4`, topicId: t, question: "Near-duplicate crowding: what does the near-duplicate filter do to the ranked list?", options: ["Re-embeds the copies", "Drops results whose cosine to a better-ranked result is ≥ 0.95, freeing slots for distinct evidence", "Deletes the original page", "Lowers top-k"], answer: 1, explanation: "Three mirrored copies are removed, so the pricing chunk with '$49' climbs back into the top-3.", skill: "fix" },
    { id: `${t}-5`, topicId: t, question: "After shipping a fix, how should you verify it?", options: ["Ask one user", "Re-run recall@k for the needed chunks on a golden set and confirm the miss is gone without new misses", "Check that p95 latency is unchanged", "Increase temperature"], answer: 1, explanation: "Retrieval fixes can help one query and hurt others (e.g. a rewrite that over-expands). Measure recall@k on the whole golden set.", skill: "verification" },
  ],
};
