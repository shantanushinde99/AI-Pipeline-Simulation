import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Place a query in a 2-D teaching embedding space and rank documents by similarity.", minutes: 12, hasMetrics: false };

const t = "embeddings";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "An embedding model maps text to a vector so that texts with similar meaning end up close together. Vector search embeds the query the same way and returns the documents whose vectors are most similar." },
    { heading: "Similarity is not relevance", body: "Similarity measures how alike two texts are in the model's learned space; relevance asks whether a document answers the question. 'Cancel my subscription' and 'Start a subscription' share topic and vocabulary, so they are close, but one is the opposite of what the user needs. Rerankers, hybrid keyword search and relevance-labelled evaluation exist to close that gap." },
    { heading: "Choosing a metric", body: "Cosine similarity compares direction only. Dot product also rewards vector length, so long or generic documents can dominate if vectors are not normalized. Euclidean distance mixes both. Many embedding models output unit-length vectors, in which case all three give the same ranking." },
    { heading: "Assumptions in this visualizer", body: "Vectors are two-dimensional and hand-placed so geometry stays visible; real embeddings have hundreds to thousands of dimensions and are learned. Free-text queries are embedded with a tiny keyword lexicon, not a neural model. All scores are computed exactly from the displayed vectors." },
  ],
  code: [
    {
      title: "Cosine top-k with NumPy",
      source: `import numpy as np

def top_k(query_vec, doc_matrix, k=3):
    q = query_vec / np.linalg.norm(query_vec)
    D = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    scores = D @ q                     # cosine similarity for every doc
    idx = np.argsort(-scores)[:k]      # highest first
    return idx, scores[idx]`,
    },
    {
      title: "With an embedding model (sentence-transformers)",
      source: `from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(docs, normalize_embeddings=True)
q_emb = model.encode("How do I cancel my subscription?", normalize_embeddings=True)
hits = util.semantic_search(q_emb, doc_emb, top_k=3)[0]`,
    },
  ],
  failureModes: [
    { title: "Similar but wrong", symptom: "The top result is on-topic but does not answer the question.", detect: "Label a small set of queries with relevant docs and measure precision@k / MRR, not just similarity.", fix: "Add a reranker, hybrid (keyword + vector) retrieval, or better document titles." },
    { title: "Unnormalized dot product", symptom: "The same long, generic document appears for many unrelated queries.", detect: "Compare rankings with cosine; check vector norms.", fix: "Normalize vectors or use cosine similarity." },
    { title: "Query and index use different models", symptom: "Retrieval quality collapses after a deploy while everything else looks healthy.", detect: "Compare embedding model versions on the index and the query path.", fix: "Re-embed the corpus or pin the query encoder to the index's model.", scenarioId: "prod-everything-green" },
    { title: "Out-of-vocabulary queries", symptom: "Queries using unseen jargon land in arbitrary places.", detect: "Inspect nearest neighbours for rare-term queries.", fix: "Domain-adapted embeddings or keyword fallback for exact terms." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "For 'How do I cancel my subscription?' the top cosine result is 'Start a subscription'. Why?", options: ["The index is corrupted", "It shares topic and wording, so its vector points almost the same way, even though the intent is opposite", "Cosine similarity favours shorter documents", "top-k is too small"], answer: 1, explanation: "Embeddings capture topical similarity well and negation/intent poorly. The relevant doc uses different words ('end your membership')." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "Switching from cosine to dot product puts 'Complete account FAQ' first for the refund query. What property causes this?", options: ["Its vector has a much larger norm", "It was indexed most recently", "It contains the word refund", "Dot product ignores direction"], answer: 0, explanation: "q·d = ‖q‖‖d‖cos θ. A long vector with a moderately similar direction can outscore a shorter, better-aligned one." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Which metric best captures 'did the user's answer appear in the first 3 results'?", options: ["Mean cosine similarity of the top 3", "Recall@3 against relevance labels", "Vector norm", "Index size"], answer: 1, explanation: "Similarity scores have no ground truth. Recall@k / precision@k need labelled relevant documents." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "Users searching 'cancel' keep getting sign-up pages. What is the most targeted fix?", options: ["Increase top-k to 50", "Add a reranker or keyword signal that scores intent terms like 'cancel' and 'end'", "Switch to Euclidean distance", "Shorten all documents"], answer: 1, explanation: "A cross-encoder reranker or hybrid retrieval looks at query and document together and can separate opposite intents that sit close in embedding space." },
  ],
};
