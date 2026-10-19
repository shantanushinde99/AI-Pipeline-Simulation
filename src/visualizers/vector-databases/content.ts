import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Compare flat, IVF and HNSW indexes on one query: count distance computations and measure recall@k.", minutes: 14, hasMetrics: false };

const t = "vector-databases";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A vector database stores embeddings and answers 'which stored vectors are nearest to this query?'. Doing that exactly means computing a distance to every vector (a flat index). Approximate nearest-neighbour (ANN) indexes add structure so a query touches only a small part of the data, accepting that a true neighbour is occasionally missed." },
    { heading: "IVF: clusters and probes", body: "An inverted-file index runs k-means to place nlist centroids and files every vector under its nearest centroid (a Voronoi cell). A query is compared with the centroids, then only the nprobe nearest cells are scanned. Cost ≈ nlist + vectors in the probed cells. Neighbours just across a cell border are missed unless nprobe is raised." },
    { heading: "HNSW: a navigable graph", body: "HNSW builds a multi-layer proximity graph. Each vector gets a random level; upper layers hold few nodes with long links, layer 0 holds everything. Search enters at the top, hops greedily toward the query on each layer, then runs a best-first beam search with width ef on layer 0. Larger ef (and M, the links per node) buys recall with more distance computations and memory." },
    { heading: "How to evaluate an index", body: "Measure recall@k against exact (flat) results on a sample of real queries, together with latency or distance computations, and plot the trade-off curve while sweeping nprobe or ef. Also test inserts and deletes: IVF centroids drift as data changes, and HNSW graphs degrade with many deletions." },
    { heading: "Assumptions in this visualizer", body: "Vectors are 200 seeded 2-D points (real embeddings have hundreds to thousands of dimensions, where these effects are stronger). Distance is Euclidean. The HNSW here is simplified: levels are drawn as in HNSW but capped at 2, each node links to its M nearest earlier-inserted nodes without the neighbour-selection heuristic, and the insert step picks neighbours exhaustively. Cost is counted in distance computations, not milliseconds." },
  ],
  code: [
    {
      title: "FAISS: flat vs IVF vs HNSW",
      source: `import faiss, numpy as np

d = 768
xb = np.load("chunks.npy").astype("float32")      # (N, d)
xq = np.load("queries.npy").astype("float32")

flat = faiss.IndexFlatL2(d); flat.add(xb)
_, truth = flat.search(xq, 10)                  # exact answers

quant = faiss.IndexFlatL2(d)
ivf = faiss.IndexIVFFlat(quant, d, 1024)        # nlist = 1024
ivf.train(xb); ivf.add(xb)
ivf.nprobe = 8

hnsw = faiss.IndexHNSWFlat(d, 32)               # M = 32
hnsw.hnsw.efConstruction = 200
hnsw.add(xb)
hnsw.hnsw.efSearch = 64

def recall(I): return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, truth)])
for name, idx in [("ivf", ivf), ("hnsw", hnsw)]:
    _, I = idx.search(xq, 10)
    print(name, recall(I))`,
    },
  ],
  failureModes: [
    { title: "Low recall from aggressive settings", symptom: "Relevant chunks exist in the index but rarely come back; answers say 'not found'.", detect: "Compare ANN results with a flat search on a query sample (recall@k).", fix: "Raise nprobe / efSearch until recall meets the target; re-measure latency." },
    { title: "Stale IVF centroids", symptom: "Recall slowly drops after months of inserts, especially for new topics.", detect: "Track cell size imbalance and recall over time.", fix: "Periodically retrain centroids and rebuild the index." },
    { title: "Embedding model mismatch", symptom: "Similarities collapse to noise right after a model upgrade.", detect: "Index metadata model ≠ query encoder model.", fix: "Re-embed and rebuild the index whenever the embedding model changes; version both.", scenarioId: "rag-stale-index" },
    { title: "Unfiltered ACL search", symptom: "Restricted documents appear in results, or filtered search returns fewer than k.", detect: "Test retrieval as different users; count results after filtering.", fix: "Use the database's pre-filtering (or partitioned indexes) rather than filtering top-k afterwards." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "IVF, query 'Near a cell border', nprobe = 1. What does recall@5 look like, and what fixes it?", options: ["1.0; nothing to fix", "Low (0.2): most true neighbours are in a neighbouring cell; raising nprobe to 2–3 recovers them", "0; IVF can't search borders", "Raising nlist fixes it"], answer: 1, explanation: "Only the single nearest cell is scanned, but the query sits near a border. Probing more cells scans the neighbours across it, at the price of more distance computations." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "With nlist = 8 and nprobe = 8, IVF costs 208 distance computations for 200 vectors. Why more than flat?", options: ["A bug", "It still compares the query with all 8 centroids, then scans every cell: nlist + N", "IVF uses a different metric", "Centroids are counted twice"], answer: 1, explanation: "Probing every cell degenerates to a flat scan plus the centroid comparisons. IVF only pays off when nprobe ≪ nlist." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "HNSW with ef = k = 5 on the 'Inside a cluster' query returns recall 0. Why can a greedy graph search miss everything?", options: ["The graph is empty", "The beam is too narrow: search settles in a local neighbourhood that looks best from its entry point", "HNSW ignores layer 0", "k is too large"], answer: 1, explanation: "Best-first search with a tiny beam stops as soon as no candidate improves its short list. A wider ef keeps more alternatives alive and escapes the local minimum." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "What is the correct way to report an ANN index's quality?", options: ["Average similarity of returned results", "Recall@k against exact flat search, together with latency or distance computations, on real queries", "Number of vectors stored", "Build time"], answer: 1, explanation: "Similarity scores look fine even when the true neighbours are missing. Recall versus exact search measures what the approximation lost; cost tells you what it saved." },
    { id: `${t}-5`, topicId: t, skill: "diagnosis", question: "After a year of inserts into an IVF index, recall drops only for documents about a newly launched product. Likely cause?", options: ["Too many distance computations", "Centroids were trained on old data; the new topic's vectors crowd one or two cells far from their centroids", "HNSW layers collapsed", "k is too small"], answer: 1, explanation: "Inserts are filed under existing centroids without retraining. New regions of the space end up in ill-fitting cells, so probes miss them." },
  ],
};
