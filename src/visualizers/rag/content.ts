import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Index a small corpus, retrieve, rerank, assemble context and generate a cited answer.", minutes: 20, hasMetrics: true };

const t = "rag-pipeline";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Retrieval-augmented generation answers questions from your documents instead of relying on what a model memorized. An offline indexing flow turns documents into searchable chunks; an online query flow finds the most relevant chunks and gives them to the model as context, asking it to answer from that context and cite it." },
    { heading: "Two flows, one contract", body: "Indexing (parse → chunk → embed → store) and querying (embed → retrieve → rerank → assemble → prompt → generate) run at different times, often owned by different teams. They must agree on parsing rules, chunking, and above all the embedding model. Many production incidents are a silent disagreement between the two." },
    { heading: "Where answers go wrong", body: "Every stage can lose the answer: the parser drops it, the chunker splits it, retrieval ranks it below k, context assembly truncates it, or the generator ignores it. Debug from the answer backwards: is the fact in the context? In the retrieved set? In the index at all? In the source?" },
    { heading: "Assumptions in this visualizer", body: "The corpus is five short fictional documents. Embeddings are a deterministic hashed bag-of-words (256 dims), the reranker is term-overlap scoring and the generator is a rule-based sentence extractor, so behaviour is reproducible and inspectable. None of them is a neural model, and token counts are approximated as word counts." },
  ],
  code: [
    {
      title: "Minimal RAG loop (Python, framework-free)",
      source: `def index(docs, chunk_size=80, overlap=15):
    chunks = []
    for doc in docs:
        words = parse(doc).split()
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            text = " ".join(words[start:start + chunk_size])
            chunks.append({"id": f"{doc.id}#{len(chunks)}", "text": text,
                           "vec": embed(text), "model": EMBED_MODEL})
            if start + chunk_size >= len(words): break
    return chunks

def answer(question, chunks, k=4, n_ctx=3, budget=300):
    q = embed(question)                       # same model as the index!
    hits = sorted(chunks, key=lambda c: -cosine(q, c["vec"]))[:k]
    hits = rerank(question, hits)
    context, used = [], 0
    for c in hits[:n_ctx]:
        words = c["text"].split()[: budget - used]
        context.append(" ".join(words)); used += len(words)
        if used >= budget: break
    prompt = SYSTEM + "\\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
    return llm(prompt + f"\\nQuestion: {question}")`,
    },
  ],
  failureModes: [
    { title: "Parser drops a table", symptom: "Questions about numbers in tables get 'I don't know' or wrong answers.", detect: "Check parser output and the 'tables dropped' count; search the index for a known value.", fix: "Use a parser that extracts or linearizes tables; add a parsing regression test.", scenarioId: "rag-parser-table" },
    { title: "Chunk boundary splits an instruction", symptom: "Answers stop mid-procedure.", detect: "Inspect chunk boundaries around the expected answer.", fix: "Larger chunks, overlap, or structure-aware (sentence/heading) chunking.", scenarioId: "rag-chunk-split" },
    { title: "Stale index", symptom: "Answers cite an old version of a document.", detect: "Compare indexed doc versions with the source of truth.", fix: "Re-index on document change; monitor index freshness.", scenarioId: "rag-stale-index" },
    { title: "Context truncation", symptom: "The right chunk is retrieved but the answer is still wrong.", detect: "Retrieval hit = yes, answer-in-context = no; look for truncation warnings.", fix: "Raise the budget, use smaller chunks, or truncate whole low-ranked chunks instead of the top one.", scenarioId: "rag-context-truncation" },
    { title: "Wrong citation", symptom: "The answer is right but the linked source doesn't say it.", detect: "Check that each cited passage contains the cited sentence.", fix: "Fix label/offset mapping; add a citation-support check.", scenarioId: "rag-citation" },
    { title: "Answer contradicts context", symptom: "Answer conflicts with the retrieved evidence.", detect: "Faithfulness check: is the answer sentence supported by the context?", fix: "Restore grounding instructions; add a faithfulness evaluator.", scenarioId: "rag-unfaithful" },
    { title: "Embedding model mismatch", symptom: "Quality drops everywhere after a deploy; every health check is green.", detect: "Top-1 similarity drops; query and index model versions differ.", fix: "Pin the query encoder to the index model or re-embed the corpus.", scenarioId: "prod-everything-green" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "The refund chunk is ranked #1 by retrieval but the answer says '14 days' instead of '30 days'. Which stage do you inspect first?", options: ["Parser", "Context assembly: is the sentence still there after the budget is applied?", "Embedding model", "Source documents"], answer: 1, explanation: "Retrieval succeeded, so the problem is downstream. A small budget can truncate the top chunk before the annual-plan sentence." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "You lower chunk size from 80 to 20 words with no overlap. What happens to the SSO answer?", options: ["Improves: smaller chunks are more precise", "It can break: the instruction sentence is split and the half with the answer isn't retrieved", "Nothing changes", "Retrieval stops working entirely"], answer: 1, explanation: "Fixed windows ignore sentence boundaries. The chunk that matches 'enable SSO' no longer contains 'metadata XML'." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "An answer is faithful to its context but wrong. What does that tell you?", options: ["The generator hallucinated", "The context itself was wrong or outdated (e.g. stale index)", "The citation mapping is broken", "Nothing; faithfulness implies correctness"], answer: 1, explanation: "Faithfulness checks agreement with the context, not with reality. If the context is stale, a perfectly grounded answer is still wrong." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "After fixing the parser to keep tables, how do you verify the fix?", options: ["Ask the price question once", "Re-index, re-run the golden set, and add a test that the '$49' value is present in the index", "Check that p95 latency is unchanged", "Increase top-k"], answer: 1, explanation: "Verify the mechanism (value now indexed) and the outcome (golden-set accuracy) and protect against regression." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Reranking is disabled and a weakly related chunk takes the only context slot. Which change targets this directly?", options: ["Enable reranking or allow more chunks in context", "Reduce chunk overlap", "Switch the embedding model", "Shorten the question"], answer: 0, explanation: "Reranking reorders candidates by a more careful relevance signal; more context slots also gives the right chunk room." },
  ],
};
