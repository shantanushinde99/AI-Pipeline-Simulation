import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Fit ranked chunks into a token budget with greedy, dedupe or MMR, then order them and check the fact survived.", minutes: 12, hasMetrics: false };

const t = "context-assembly";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Context assembly turns a ranked list of retrieved chunks into the exact block of text placed in the prompt. It decides which chunks to keep under a token budget, what to do with a chunk that doesn't fit, and in what order the survivors appear." },
    { heading: "Selection strategies", body: "Greedy-by-rank keeps adding the next chunk until the budget runs out; simple, but near-duplicates (the same paragraph from two versions of a page) waste budget. Deduplication skips chunks too similar to one already kept. Maximal Marginal Relevance (MMR) picks, each round, the chunk with the best λ·relevance − (1 − λ)·(max similarity to what's already picked), trading relevance for diversity." },
    { heading: "Overflow and ordering", body: "When the next chunk doesn't fit you can truncate it (fills the budget, but may cut the sentence that matters) or drop it and try smaller ones. Long-context models tend to use information at the start and end of the context better than the middle (“lost in the middle”), so some systems put the best chunks at both ends." },
    { heading: "Assumptions in this visualizer", body: "Eight hand-written chunks with hand-set relevance scores; two pairs are deliberate near-duplicates. Token counts use a toy estimate of 1.3 tokens per word. Similarity is Jaccard overlap of normalized word sets, a cheap stand-in for embedding cosine. The U-shaped position curve is an illustration of the published “lost in the middle” finding, not a measurement of any model. “Answer survived” simply checks whether the fact text is present in the assembled context." },
  ],
  code: [
    {
      title: "assemble.py: dedupe + budget + both-ends ordering",
      source: `def assemble(chunks, budget, count_tokens, sim, dup=0.6):
    kept, used = [], 0
    for c in chunks:                                   # already ranked
        if any(sim(c.text, k.text) >= dup for k in kept):
            continue                                   # near-duplicate
        t = count_tokens(c.text)
        if used + t > budget:
            continue                                   # drop; a smaller one may fit
        kept.append(c); used += t
    front, back = kept[0::2], kept[1::2][::-1]
    return front + back                                # best chunks at both ends`,
    },
    {
      title: "MMR selection (NumPy)",
      source: `def mmr(query_vec, doc_vecs, k, lam=0.7):
    rel = doc_vecs @ query_vec
    picked, rest = [], list(range(len(doc_vecs)))
    while rest and len(picked) < k:
        red = [max((doc_vecs[i] @ doc_vecs[j] for j in picked), default=0) for i in rest]
        scores = [lam * rel[i] - (1 - lam) * r for i, r in zip(rest, red)]
        best = rest[int(np.argmax(scores))]
        picked.append(best); rest.remove(best)
    return picked`,
    },
  ],
  failureModes: [
    { title: "Answer truncated away", symptom: "The model says the policy doesn't mention annual refunds although the page does.", detect: "Trace shows the answer chunk with a “truncated” flag; the fact sits after the cut.", fix: "Drop instead of truncate, reserve budget, or chunk so facts aren't at chunk ends.", scenarioId: "rag-context-truncation" },
    { title: "Budget wasted on duplicates", symptom: "Context holds two copies of the same paragraph; relevant chunks are missing.", detect: "Pairwise similarity between kept chunks is high; unique-token ratio is low.", fix: "Deduplicate at index time (version filtering) and at assembly time." },
    { title: "Over-diversified context", symptom: "Context contains loosely related filler instead of the second-best relevant chunk.", detect: "MMR λ is low; kept chunks have low relevance scores.", fix: "Raise λ (0.6–0.8 is common) and evaluate on a golden set." },
    { title: "Lost in the middle", symptom: "The fact is in the prompt but the answer ignores it.", detect: "Answer accuracy depends on where the gold chunk lands; fails more often mid-context.", fix: "Fewer, better chunks; place the strongest at the start and end." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "With greedy-by-rank, budget 130 and truncate mode, why is the answer fact lost?", options: ["C5 was never retrieved", "Duplicates C2 and C4 used up budget, so C5 was truncated before its last sentence", "MMR removed C5", "The ordering put C5 last"], answer: 1, explanation: "C1–C4 consume 126 tokens including two near-copies. C5 (51 tokens) gets 4 tokens of room; the fact in its final sentence is cut." },
    { id: `${t}-2`, topicId: t, skill: "fix", question: "Which single change saves the fact at budget 130?", options: ["Switch ordering to both ends", "Switch to “dedupe, then greedy”", "Lower the budget", "Switch to MMR with λ = 0.3"], answer: 1, explanation: "Skipping C2 and C4 frees about 60 tokens, so C5 fits whole. Ordering can't restore text that was never included, and λ = 0.3 picks filler." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "In MMR mode, what happens as you lower λ from 0.7 to 0.3?", options: ["Only relevance matters", "Diversity dominates: off-topic chunks like invoices get picked and C5 ends up truncated", "Nothing changes", "Duplicates are added"], answer: 1, explanation: "With small λ the redundancy penalty outweighs the small relevance differences, so dissimilar but less relevant chunks win rounds and the budget runs out before C5." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "MMR with λ = 0.5: chunk relevance 0.84, max similarity to kept chunks 0.24. Its MMR score is:", options: ["0.30", "0.54", "0.84", "0.42"], answer: 0, explanation: "0.5 · 0.84 − 0.5 · 0.24 = 0.42 − 0.12 = 0.30." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How do you check that context assembly isn't silently losing answers?", options: ["Watch total prompt tokens", "For a golden set, check that the gold fact is present (untruncated) in the assembled context, and log what was dropped and why", "Check that the context is under budget", "Count retrieved chunks"], answer: 1, explanation: "Under-budget contexts can still be missing the answer. Test the presence of the fact after assembly, and keep a dropped/truncated log like the panel here." },
  ],
};
