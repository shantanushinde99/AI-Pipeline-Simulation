import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch documents get fetched, typed, extracted, cleaned, hashed, deduplicated and queued for indexing.", minutes: 10, hasMetrics: false };

const t = "document-ingestion";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Ingestion is everything that happens before a document can be chunked and embedded: connectors fetch raw bytes, the pipeline detects the format, an extractor pulls out text, cleaning strips boilerplate, a content hash removes duplicates and detects changes, and metadata (source, version, updated_at, access control) is attached. The output is a queue of clean, uniquely identified documents." },
    { heading: "Why it matters", body: "A retrieval system can only find what ingestion let through. Duplicates crowd the top-k with the same passage; stale versions answer with last quarter's numbers; documents dropped without a log line create holes nobody notices; missing ACL metadata leaks restricted text to users who should never see it." },
    { heading: "Incremental re-ingestion", body: "Re-embedding an entire corpus every night is slow and expensive. Instead, store each document's content hash. On the next run, recompute it: equal hash means unchanged (skip), a new hash means re-chunk and re-embed. Hashing the cleaned, normalized text (not the raw bytes) makes the check robust to whitespace, line-ending or re-export changes." },
    { heading: "Accounting", body: "A healthy run satisfies received = queued + duplicates + unchanged + rejected. Every reject is logged and parked in a dead-letter queue so it can be fixed and replayed. If the identity does not hold, something was dropped silently." },
    { heading: "Assumptions in this visualizer", body: "The seven sources are tiny, hand-written stand-ins for real files; fetching is instant and the byte counts are the lengths of those strings. Extractors are simple regex rules, not real PDF/HTML parsers. The content hash is 32-bit FNV-1a (production systems use SHA-256). The 'previous run' is a fixed snapshot in which only the rate-limit page differs." },
  ],
  code: [
    {
      title: "ingest.py: hash-based dedupe and incremental updates",
      source: `import hashlib, logging, re

BOILERPLATE = [re.compile(r"^page \\d+ of \\d+", re.I), re.compile(r"^©")]

def clean(text: str) -> str:
    lines = [re.sub(r"\\s+", " ", l).strip() for l in text.replace("\\r", "").split("\\n")]
    return "\\n".join(l for l in lines if l and not any(p.match(l) for p in BOILERPLATE))

def ingest(sources, store, queue, dead_letter):
    seen = set()
    for src in sources:
        extractor = EXTRACTORS.get(detect_type(src.raw, src.name))
        if extractor is None:
            logging.warning("unsupported type: %s", src.uri)   # never skip silently
            dead_letter.put(src)
            continue
        text = clean(extractor(src.raw))
        h = hashlib.sha256(text.lower().encode()).hexdigest()
        if store.get_hash(src.uri) == h:
            continue                                         # unchanged since last run
        if h in seen or store.has_hash(h):
            logging.info("duplicate: %s", src.uri)
            continue
        seen.add(h)
        queue.put({"text": text, "source": src.uri, "version": src.version,
                   "updated_at": src.updated_at, "acl": src.acl, "content_hash": h})`,
    },
  ],
  failureModes: [
    { title: "Silent skips", symptom: "Questions about some documents never get answers, yet every ingestion run is green.", detect: "Check received = queued + duplicates + unchanged + rejected; switch 'Unsupported file type' to skip silently and watch the identity break.", fix: "Log every reject, keep a dead-letter queue, and alert on the accounting gap." },
    { title: "Duplicates crowd retrieval", symptom: "Top-k contains the same passage twice from two file copies.", detect: "Count identical content hashes in the index; look for chunks with equal text but different source URIs.", fix: "Dedupe on a hash of cleaned, normalized text, not raw bytes or file names." },
    { title: "Stale versions in the index", symptom: "Answers quote last quarter's numbers after the page was updated.", detect: "Compare each source's updated_at/hash with what the index stores.", fix: "Incremental re-ingestion keyed by content hash; delete or supersede old versions' chunks.", scenarioId: "rag-stale-index" },
    { title: "Missing ACL metadata", symptom: "Users see snippets from documents they are not allowed to open.", detect: "Audit index entries without an acl field; test retrieval as a low-privilege user.", fix: "Make acl a required field at ingestion and filter on it at query time." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "Switch the dedupe hash to raw bytes. Why does pricing (copy).pdf now reach the index queue?", options: ["Its file name is different", "Its bytes differ (extra spaces, CRLF line endings), so its raw hash differs even though the cleaned text is identical", "PDFs cannot be hashed", "Raw hashing is slower"], answer: 1, explanation: "Dedupe compares hashes. Formatting-only differences change raw bytes, so the hashes differ. Hashing the cleaned, lower-cased text makes the two copies collide." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "In incremental mode with the default settings, how many documents reach the index queue?", options: ["0", "1", "5", "7"], answer: 1, explanation: "Four documents hash the same as last run (skipped), the copy duplicates pricing.pdf, the diagram is rejected, and only api-rate-limits.md changed (300 → 600)." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "With 'Skip silently', the final tally reads received 7, queued 5, duplicates 1, rejected 0. What should an operator conclude?", options: ["The run is healthy", "One document is unaccounted for: 7 ≠ 5 + 1 + 0", "Dedupe is too aggressive", "The queue is full"], answer: 1, explanation: "The accounting identity received = queued + duplicates + unchanged + rejected fails by one. That gap is the silently skipped .vsdx file." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "Which metadata field lets retrieval stop support-only FAQ rows from appearing in a public chatbot's answers?", options: ["content_hash", "updated_at", "acl", "type"], answer: 2, explanation: "The access control list travels with every chunk; the retriever filters on it before anything reaches the prompt." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Why does cleaning run before hashing?", options: ["Hashing needs short input", "So that boilerplate and whitespace differences don't make identical content look different", "Cleaning needs the hash", "It doesn't matter"], answer: 1, explanation: "The hash should identify content, not formatting. Footers, nav bars and spacing vary between exports of the same document." },
  ],
};
