import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Train a tiny BPE tokenizer merge by merge, then watch your own text split into token ids.", minutes: 10, hasMetrics: false };

const t = "tokenization";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Language models do not read characters or words; they read integer token ids. A tokenizer maps text to ids and back. Byte-pair encoding (BPE), used by GPT-style models, learns its vocabulary from data: start with single characters (or bytes), then repeatedly merge the most frequent adjacent pair into a new token." },
    { heading: "Why it matters", body: "Token count, not word count, sets how much fits in the context window and what an API call costs. Common words become one token; rare words, typos, other languages, code and numbers split into several. That is why the same sentence can cost very different amounts in different languages, and why models are bad at counting letters or doing digit-by-digit arithmetic." },
    { heading: "Encoding new text", body: "To tokenize new text, it is pre-tokenized (words keep their leading space, so ' the' and 'the' are different tokens), split into characters, and the learned merges are replayed in the order they were learned. A word never seen in training still encodes, as a sequence of known sub-word pieces: no word is ever out of vocabulary." },
    { heading: "Assumptions in this visualizer", body: "A toy tokenizer, trained live in your browser on a ~60-word built-in corpus. The base vocabulary is the 95 printable ASCII characters plus <unk>; real byte-level BPE (GPT-2, tiktoken) starts from 256 bytes and so never needs <unk>. Real vocabularies have 32k–200k tokens learned from billions of words, and real pre-tokenizers handle digits, contractions and Unicode more carefully. Ties between equally frequent pairs are broken alphabetically. The merge algorithm itself is the real one." },
  ],
  code: [
    {
      title: "bpe_train.py — the core loop",
      source: `from collections import Counter

def train_bpe(words: Counter, num_merges: int):
    # words: {" cat": 8, " sat": 2, ...}; each word starts as characters
    seg = {w: list(w) for w in words}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w, freq in words.items():
            s = seg[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += freq
        if not pairs or pairs.most_common(1)[0][1] < 2:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))
        for w in seg:                       # merge everywhere
            s, out, i = seg[w], [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(s[i]); i += 1
            seg[w] = out
    return merges`,
    },
    {
      title: "Counting real tokens with tiktoken",
      source: `import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
for text in ["the cat sat", "The CAT sat", "antidisestablishmentarianism", "2024-10-19"]:
    ids = enc.encode(text)
    print(len(ids), [enc.decode([i]) for i in ids])`,
    },
  ],
  failureModes: [
    { title: "Budgeting by words instead of tokens", symptom: "Prompts that 'should fit' get truncated, or bills are higher than estimated.", detect: "Compare word and token counts for real traffic (the compare step); non-English text and code often need 2–4× more tokens.", fix: "Count tokens with the model's own tokenizer before sending; budget in tokens.", scenarioId: "rag-context-truncation" },
    { title: "Tokenizer mismatch", symptom: "A fine-tuned or self-hosted model outputs garbage or ignores special tokens.", detect: "Ids decode to unexpected text; vocabulary sizes differ between tokenizer and model embedding table.", fix: "Always ship the tokenizer files with the model and version them together." },
    { title: "Case, whitespace and numbers split unexpectedly", symptom: "'Cat', ' cat' and 'CAT' behave differently; numbers are split into odd chunks; arithmetic and string matching are unreliable.", detect: "Tokenize the variants (try the Case, Spaces and Numbers presets) and compare ids.", fix: "Normalize input where meaning allows, avoid relying on exact surface forms, use tools for arithmetic." },
    { title: "Characters outside the vocabulary", symptom: "Text with accents, emoji or other scripts turns into <unk> and the model loses information.", detect: "Look for <unk> ids (try the 'Not ASCII' preset).", fix: "Use byte-level BPE, which can encode any byte sequence, or a tokenizer trained on the right languages." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "Set merges to 0. What happens to the BPE token count for 'the fastest rats were sitting'?", options: ["It drops to 5, one per word", "It equals the number of characters", "It becomes 0", "It stays the same as with 16 merges"], answer: 1, explanation: "With no merges, every character (including each leading space) is its own token: a character-level tokenizer. Each merge you add can only reduce the count." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "With 16 merges, how is the unseen word ' fastest' (from the 'Unseen word' preset) tokenized?", options: ["As <unk>", "As one token ' fastest'", "As known pieces: '␣', 'f', 'a', 'st', 'est'", "It is dropped from the input"], answer: 2, explanation: "' fastest' never appears in the corpus, but the merges 's'+'t' → 'st' and 'e'+'st' → 'est' were learned from words like ' fattest' and ' slowest'. BPE encodes any new word as known sub-word pieces; it never needs an unknown-word token for characters it knows." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "'the cat' and 'The CAT' produce very different token counts. Why?", options: ["Upper-case letters are illegal", "Merges were learned from a lower-case corpus, so 'T', 'C', 'A' never merged with neighbours", "Spaces are ignored", "BPE sorts tokens alphabetically"], answer: 1, explanation: "BPE only knows the pairs it saw. Upper-case pairs never appeared in training, so those characters stay separate tokens. Real tokenizers see more case variety, but the effect still exists." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "Your text has 29 characters and tokenizes to 18 tokens. What is the characters-per-token ratio, and what happens as you add merges?", options: ["0.62; it falls", "1.61; it rises", "18; it stays", "29; it falls"], answer: 1, explanation: "29 / 18 ≈ 1.61 characters per token. Each extra merge can only fuse tokens, so tokens get longer and the ratio rises (English with real tokenizers is about 4 characters per token)." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Why does training stop after a fixed number of merges here, even if you ask for more?", options: ["The vocabulary is full at 128", "No adjacent pair occurs at least twice any more", "The corpus has been deleted", "Merges become negative"], answer: 1, explanation: "Merging a pair that occurs only once would just memorise a single word. This implementation stops when the best pair count drops below 2 (a minimum-frequency rule)." },
  ],
};
