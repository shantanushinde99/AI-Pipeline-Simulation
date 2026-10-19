import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch turns fill a token window, get summarised or evicted, and facts return via vector retrieval.", minutes: 12, hasMetrics: false };

const t = "short-term-and-long-term-memory";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "An LLM only sees what is in its context window. Short-term memory is the recent conversation kept verbatim in that window. When it fills up, old turns are evicted, optionally compressed into a running summary. Long-term memory is an external store (often a vector database) where salient facts are written and later retrieved by similarity when a question needs them." },
    { heading: "Why it matters", body: "Agents and assistants run long conversations and multi-step tasks. Without a memory strategy they forget the user's name, constraints and decisions after a few turns, or they pay to re-send the whole history every call. Each layer trades cost, fidelity and recall differently." },
    { heading: "Summaries are lossy, retrieval is fuzzy", body: "A summary saves tokens but drops details such as dates and numbers. A vector store keeps exact facts, but retrieval ranks by similarity, not by truth or recency: a stale fact can outrank its correction, and irrelevant facts can crowd out relevant ones. Write policies (upsert by key, timestamps, deletion) matter as much as retrieval." },
    { heading: "Assumptions in this visualizer", body: "Tokens are counted as words (a toy tokenizer). Fact extraction and summarisation are hand-written rules, not an LLM; the summary deliberately drops numbers to show lossy compression. Vectors are keyword counts over six hand-picked concept dimensions (food, date, place, money, person, lodging), standing in for a learned embedding. The final answer is filled from a template using the first source that has each fact (window, then summary, then retrieved facts). No model is called." },
  ],
  code: [
    {
      title: "Window + summary + vector memory (Python)",
      source: `def build_context(history, summary, store, question, window=4000, k=3, min_score=0.3):
    # 1. long-term: retrieve facts similar to the question
    q = embed(question)
    hits = [h for h in store.search(q, top_k=k) if h.score >= min_score]
    facts = "\\n".join(f"- {h.text} (saved {h.saved_at:%Y-%m-%d})" for h in hits)

    # 2. short-term: newest turns that fit in what is left of the window
    budget = window - count_tokens(summary) - count_tokens(facts) - count_tokens(question)
    recent = []
    for turn in reversed(history):
        if count_tokens(turn) > budget:
            break
        recent.insert(0, turn); budget -= count_tokens(turn)
    return [system_prompt, f"Summary: {summary}", f"Known facts:\\n{facts}", *recent, question]

def remember(turn, store):
    for fact in extract_facts(turn):             # e.g. {"key": "trip.date", "text": "arrive 19 May"}
        store.upsert(key=fact.key, text=fact.text, vector=embed(fact.text), saved_at=now())`,
    },
  ],
  failureModes: [
    { title: "Forgotten constraint", symptom: "The assistant suggests a steak house to a vegetarian ten turns later.", detect: "The fact appears in the transcript but not in the prompt sent for the failing turn.", fix: "Write salient facts to long-term memory or pin them in a profile block that is never evicted." },
    { title: "Lossy summary", symptom: "Specific numbers, dates or names become vague ('a trip is planned').", detect: "Compare entities in evicted turns against the summary; count dropped numbers.", fix: "Use structured summaries (key: value) or keep exact facts in a store alongside the summary." },
    { title: "Stale memory wins", symptom: "The agent uses an old value after the user corrected it.", detect: "Two retrieved facts share a key with different values; the older one ranks higher.", fix: "Upsert by key, store timestamps and prefer the newest, or delete superseded facts.", scenarioId: "rag-stale-index" },
    { title: "Irrelevant retrieval", symptom: "Retrieved memories crowd the context but do not help.", detect: "Low-similarity hits included; answer quality does not change when memory is removed.", fix: "Tune top-k and a similarity threshold; evaluate retrieval with labelled questions." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Long-term memory off, summarisation on, default window. What does the final answer miss?", options: ["Nothing", "The arrival date: the summary kept 'Trip to Lisbon planned' but dropped the number", "The diet", "The city"], answer: 1, explanation: "The trip turn was evicted and summarised; the rule-based summary keeps the city but drops dates, so no source has the date." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "Query vector q = [2,2,1,0,0,1], fact f = [0,3,1,0,0,0]. What is cos(q, f)?", options: ["0.50", "0.63", "0.70", "0.90"], answer: 2, explanation: "q·f = 6 + 1 = 7; ‖q‖ = √10, ‖f‖ = √10; 7 / 10 = 0.70." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "With the correction turn on, the answer says 12 May although the user moved the flight to 19 May. Why?", options: ["The window is too large", "Both facts are in the store and the stale one has higher similarity to the question", "Summarisation is off", "The threshold is too low"], answer: 1, explanation: "The store appends instead of upserting, and similarity favours the older sentence (it also mentions Lisbon). Retrieval does not know which is current." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "With the correction on, which setting makes the answer use 19 May without changing the store?", options: ["Window of 120 tokens, so the correction is still in the short-term window", "Top-k = 1", "Turn summarisation off", "Threshold 0.8"], answer: 0, explanation: "With a big window the correction is still verbatim in context and the newest in-window fact wins. (Top-k = 1 returns only the stale fact.)" },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "What is the most robust fix for the stale-fact problem?", options: ["Raise top-k", "Upsert facts by key with timestamps so the correction replaces the old value", "Use a bigger model", "Summarise more aggressively"], answer: 1, explanation: "Memory needs a write policy: one current value per key (or recency-aware ranking), not just similarity search." },
  ],
};
