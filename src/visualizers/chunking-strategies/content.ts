import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Cut one handbook five ways (fixed, overlap, sentence, heading, semantic) and count the facts each split.", minutes: 12, hasMetrics: false };

const t = "chunking-strategies";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Chunking splits parsed documents into the passages that get embedded, indexed and retrieved. Each chunk is retrieved on its own, so it should hold a complete thought: big enough to carry context, small enough that its embedding is about one thing and that several fit in the prompt." },
    { heading: "The strategies", body: "Fixed size cuts every N tokens, blind to meaning. Overlap repeats the last few tokens of each chunk at the start of the next, so facts near a cut get a second chance (at the cost of duplicated storage and near-duplicate hits). Sentence-based packing only cuts at sentence ends. Heading-aware chunking starts a new chunk at every section and packs sentences inside it. Semantic chunking embeds sentences and cuts where similarity between neighbours drops, i.e. where the topic changes." },
    { heading: "How to choose", body: "Measure, don't guess: take questions with known answer spans and check whether each span lands whole inside one chunk (the 'facts split' metric here), then check retrieval recall end to end. Structure-aware chunkers usually win on manuals, policies and docs with headings; fixed + overlap is a robust default for unstructured text. Watch the size distribution: tiny chunks carry no context, huge ones dilute their embedding." },
    { heading: "Assumptions in this visualizer", body: "Sizes are counted in words, not model tokens. The document is a short fictional handbook. Semantic chunking uses the app's hashed bag-of-words embedding (shared with the RAG pipeline visualizer), so 'similarity' means shared normalized words, not learned meaning; real semantic chunkers use neural sentence embeddings and often compare windows of sentences. Headings are merged into the following sentence before semantic comparison." },
  ],
  code: [
    {
      title: "LangChain: recursive splitter with overlap",
      source: `from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# Heading-aware first, then size-limited with overlap inside each section.
by_heading = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
sections = by_heading.split_text(markdown)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120,            # characters
    separators=["\\n\\n", "\\n", ". ", " "],          # prefer paragraph, then sentence ends
)
chunks = splitter.split_documents(sections)       # metadata keeps the heading path`,
    },
    {
      title: "Semantic split on similarity drops (TypeScript)",
      source: `function semanticChunks(sentences: string[], embed: (s: string) => number[], tau = 0.15, max = 120) {
  const vecs = sentences.map(embed);               // unit-length vectors
  const chunks: string[][] = [[sentences[0]]];
  for (let i = 1; i < sentences.length; i++) {
    const sim = dot(vecs[i - 1], vecs[i]);
    const cur = chunks[chunks.length - 1];
    const words = cur.join(" ").split(/\\s+/).length + sentences[i].split(/\\s+/).length;
    if (sim < tau || words > max) chunks.push([sentences[i]]);
    else cur.push(sentences[i]);
  }
  return chunks.map((c) => c.join(" "));
}`,
    },
  ],
  failureModes: [
    { title: "Facts cut in half", symptom: "The answer cites a chunk with the condition but not the number (or vice versa).", detect: "For golden questions, check the answer span lands whole in one chunk; here, the 'facts split' metric.", fix: "Sentence/heading-aware boundaries, or add overlap.", scenarioId: "rag-chunk-split" },
    { title: "Orphaned context", symptom: "A retrieved chunk says 'This is required for every admin account' without saying what 'this' is.", detect: "Count chunks that start mid-sentence; read a random sample of chunks.", fix: "Cut at sentence ends; prepend the section heading (or a document title) to each chunk." },
    { title: "Chunks too small or too large", symptom: "Tiny chunks retrieve well but answer nothing; giant chunks blow the context budget and blur the embedding.", detect: "Plot the size distribution; compare recall at several sizes.", fix: "Pick size from evaluation, cap maximum size in structure-aware chunkers." },
    { title: "Overlap duplicates crowd top-k", symptom: "Top-k returns two neighbouring chunks that share most of their text.", detect: "Measure pairwise overlap among retrieved chunks.", fix: "Smaller overlap, or merge/deduplicate adjacent hits before building the prompt." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "With fixed size 30 words and no overlap, how many of the 5 facts are split across a boundary?", options: ["0", "1", "4", "5"], answer: 2, explanation: "Cuts fall every 30 words regardless of sentences; four of the five fact spans straddle one of them (80% split)." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "Switch to fixed + overlap (size 30, overlap 10). Why do fewer facts split?", options: ["Chunks get bigger", "Each chunk restarts 10 words before the previous one ended, so a fact near a cut can fit whole in the next chunk", "Overlap removes sentences", "The facts change"], answer: 1, explanation: "The step shrinks to 20 words while each chunk still spans 30, so spans up to ~10 words past a cut are covered twice." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "With semantic chunking, what happens when you raise τ from 0.15 to 0.3?", options: ["Fewer, larger chunks", "More cuts: weaker similarity drops now count as topic changes, and a two-sentence fact can be split", "No change", "All facts become intact"], answer: 1, explanation: "Any adjacent pair with similarity below τ gets a cut. At 0.3 the cut between 'Customers are notified…' and 'A written post-mortem…' (similarity ≈ 0.21) splits the incident fact." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "A retrieved chunk begins 'authentication is required for every admin account.' What went wrong?", options: ["The embedding model", "The chunk boundary fell mid-sentence: the chunker ignored sentence ends", "The reranker", "The prompt"], answer: 1, explanation: "Only fixed-size cuts can land mid-sentence. The card view flags these as 'starts mid-sentence'." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Heading-aware chunking at size 15 still splits facts. Which change fixes it without losing structure?", options: ["Switch to fixed size", "Raise the maximum size so each section's sentences can stay together", "Lower the overlap", "Remove headings"], answer: 1, explanation: "At 15 words even one two-sentence fact doesn't fit; the max size forces a cut inside it. At 40 words every section-internal fact fits in one chunk again." },
  ],
};
