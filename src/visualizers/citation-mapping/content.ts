import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Follow each [n] marker through passage → chunk → document → anchor, then break the mapping.", minutes: 12, hasMetrics: false };

const t = "citation-mapping";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A model writes citations as plain numbers like [3]. Citation mapping turns each number into something a user can open: the prompt passage with that label, the chunk id it was built from, the source document, and the page or anchor where the span lives. Every hop is a lookup, and every lookup can be wrong." },
    { heading: "Why it breaks silently", body: "Mapping bugs never throw. An off-by-one index, a source list that gets deduplicated and renumbered after generation, or a passage that was truncated before the model saw it all still produce a clickable link. The answer text is identical; only where the link lands changes. That is why citations need their own verifier and their own metric." },
    { heading: "Resolve by id, verify by content", body: "Two habits remove most of these bugs. Assign labels once, after any dedupe or sorting, and resolve markers through stable chunk ids instead of list positions. Then verify each citation by checking that the cited span actually supports the sentence, not just that the link resolves." },
    { heading: "Assumptions in this visualizer", body: "The answer and its markers are scripted (a stand-in for a model); the passages come from the simulated RAG index (chunk size 80). Hybrid retrieval is simulated by listing sso#0 twice, as if both vector and keyword search returned it. PDF page numbers are simulated as 3 blocks per page. The verifier is the same word-overlap check used across these visualizers (≥ 60% of words, numbers must match in strict mode), standing in for an NLI model or LLM judge." },
  ],
  code: [
    {
      title: "Stable citation mapping (Python)",
      source: `def build_prompt(hits):
    # Dedupe BEFORE numbering, so labels never change after generation.
    seen, passages = set(), []
    for h in hits:
        if h.chunk_id in seen:
            continue
        seen.add(h.chunk_id)
        passages.append(h)
    label_to_chunk = {i + 1: p.chunk_id for i, p in enumerate(passages)}
    text = "\\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    return text, label_to_chunk

def resolve(answer, label_to_chunk, chunks, docs):
    links = []
    for label in re.findall(r"\\[(\\d+)\\]", answer):
        chunk_id = label_to_chunk.get(int(label))     # 1-based, by id, never by position
        if chunk_id is None:
            links.append((label, None)); continue
        c = chunks[chunk_id]
        links.append((label, f"{docs[c.doc_id].url}#{c.anchor}"))
    return links`,
    },
    {
      title: "Verify each citation (TypeScript)",
      source: `for (const { sentence, label } of splitCitedSentences(answer)) {
  const passage = labelToPassage.get(label);
  const ok = passage !== undefined && passage.sentences.some((s) => entails(s, sentence));
  report.push({ label, ok });
}
const citationAccuracy = report.filter((r) => r.ok).length / report.length;`,
    },
  ],
  failureModes: [
    { title: "Off-by-one label resolution", symptom: "Citations open the passage after the right one; the last marker opens nothing.", detect: "Verifier: cited span doesn't support the sentence; dangling labels appear.", fix: "Use 1-based labels end to end and resolve through an explicit label → chunk-id map.", scenarioId: "rag-citation" },
    { title: "Renumbered after dedupe", symptom: "Some citations are right, the ones after a removed duplicate point one entry too far.", detect: "Compare the marker's chunk id at generation time with the chunk id it resolves to now.", fix: "Dedupe before building the prompt, or remap markers by chunk id when the list changes." },
    { title: "Citing a truncated passage", symptom: "The link opens the right document, but the cited text stops mid-sentence; part of the claim isn't there.", detect: "Strict span-level verification (numbers must match) fails; the context-assembly trace shows truncation.", fix: "Truncate on sentence boundaries, or drop passages that don't fit; never cite a span the model didn't fully see.", scenarioId: "rag-context-truncation" },
    { title: "Lenient verifier hides problems", symptom: "Citation accuracy looks perfect, yet users find unsupported details behind links.", detect: "Spot-check passes with a stricter, sentence-level check that requires numbers to match.", fix: "Verify at sentence/span level with an entailment model; treat numbers and names as must-match." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "With 'labels renumbered after dedupe', marker [1] still works but [3] opens pricing.pdf. Why?", options: ["The pricing chunk is ranked higher", "Removing the duplicate at position 2 shifted every later entry up by one, and the markers weren't remapped", "The verifier is too strict", "Chunk ids changed"], answer: 1, explanation: "Entries after the removed duplicate moved up one position. Marker [3] (retention) now names position 3, which holds pricing; [4] has no entry at all." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Off-by-one bug, default settings: why does marker [1] still pass verification?", options: ["Off-by-one bugs never affect the first marker", "Passage 2 is a duplicate of passage 1, so the wrong passage happens to contain the claim", "The verifier skips the first citation", "The marker is cached"], answer: 1, explanation: "The resolver opens passage 2 instead of 1, but both are sso#0. Lucky passes like this are why spot checks of a few citations can miss a systematic bug." },
    { id: `${t}-3`, topicId: t, skill: "fix", question: "Which single change makes the dedupe bug impossible?", options: ["Turn off the verifier", "Dedupe before building the prompt so labels are assigned once", "Increase top-k", "Use a lenient verifier"], answer: 1, explanation: "If duplicates are removed before numbering, the labels the model sees are the labels users see. Turn on 'Dedupe before building the prompt' and accuracy returns to 100%." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "With 'citing a truncated passage', switching the verifier from strict to lenient changes citation accuracy from 67% to 100%. What does that tell you?", options: ["The lenient verifier is more accurate", "The lenient check passes because 70% of the words appear somewhere in the passage, even though '24/7 phone' was cut off", "Truncation fixed itself", "Both verifiers agree"], answer: 1, explanation: "Bag-of-words overlap over a whole passage ignores numbers and sentence boundaries. The strict check notices 24 and 7 are missing from the cited text." },
    { id: `${t}-5`, topicId: t, skill: "metric", question: "The answer text is identical in every configuration. Which metric moves when you inject a mapping bug?", options: ["Answer correctness", "Citation accuracy", "Retrieval recall", "Latency"], answer: 1, explanation: "Mapping happens after generation, so the text (and its correctness) doesn't change. Only whether each marker leads to supporting evidence does." },
  ],
};
