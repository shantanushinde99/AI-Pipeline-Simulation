import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Generate text one word at a time from n-gram counts: probabilities, greedy picks, loops, unseen contexts.", minutes: 12, hasMetrics: false };

const t = "next-token-prediction";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A language model assigns a probability to every possible next token given the tokens so far: P(next | context). Generation is a loop: compute the distribution, pick a token, append it, repeat. That is autoregressive generation, and it is exactly how GPT-style models produce text, one token per step." },
    { heading: "From counts to probabilities", body: "An n-gram model estimates P(next | previous n−1 tokens) by counting: P(cat | the) = C(the cat) / C(the). A bigram uses one word of context, a trigram two. Longer contexts give sharper predictions but are seen less often, so more contexts end up with no data." },
    { heading: "Unseen contexts and repetition", body: "If a context never occurred, the counts are 0/0. Backoff uses a shorter context that was seen; add-one smoothing gives every word a small probability so nothing is ever impossible. Greedy decoding (always pick the top token) is deterministic, so whenever a context repeats, the output repeats forever. Neural LMs share both problems in milder forms, which is why sampling and repetition penalties exist." },
    { heading: "How neural LLMs differ", body: "A Transformer computes the same kind of distribution, but from a learned network that reads thousands of tokens of context and generalises to contexts it never saw verbatim. The final step (softmax over the vocabulary, then pick a token) is the same as here." },
    { heading: "Assumptions in this visualizer", body: "This is a counting n-gram model over a 46-token built-in corpus, not a neural network, and no real LLM is called. Tokens are lower-case words and punctuation. Backoff is 'stupid backoff' without discounting (it does not redistribute probability mass like Katz backoff). Ties are broken by first appearance in the corpus. All probabilities are computed live from the counts shown." },
  ],
  code: [
    {
      title: "bigram_lm.py",
      source: `from collections import Counter, defaultdict

corpus = "the cat sat on the mat . the cat ate the fish .".split()
counts = defaultdict(Counter)
for prev, nxt in zip(corpus, corpus[1:]):
    counts[prev][nxt] += 1

def next_dist(prev):
    c = counts[prev]
    total = sum(c.values())
    return {w: n / total for w, n in c.items()}   # KeyError-free, but {} if unseen

seq = ["the"]
for _ in range(8):                              # greedy, autoregressive
    dist = next_dist(seq[-1])
    if not dist:
        break                                   # unseen context: needs smoothing/backoff
    seq.append(max(dist, key=dist.get))
print(" ".join(seq))   # the cat sat on the cat sat on the`,
    },
    {
      title: "Same loop with a neural LM (Hugging Face)",
      source: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
ids = tok("the cat", return_tensors="pt").input_ids
for _ in range(8):
    logits = model(ids).logits[0, -1]          # scores for every vocab token
    probs = torch.softmax(logits, dim=-1)      # P(next | context)
    nxt = probs.argmax().view(1, 1)            # greedy pick
    ids = torch.cat([ids, nxt], dim=1)
print(tok.decode(ids[0]))`,
    },
  ],
  failureModes: [
    { title: "Repetition loops", symptom: "Output repeats the same phrase over and over.", detect: "Look for repeated n-grams in generations; with greedy decoding a repeated context guarantees a cycle (the 'Done' step flags it).", fix: "Sample with temperature/top-p, add repetition or frequency penalties, use a stronger model." },
    { title: "Unseen context (sparsity)", symptom: "The model has no prediction, or falls back to generic high-frequency words.", detect: "Context counts are 0 (try 'the zebra' with 'None').", fix: "Backoff or smoothing for n-grams; neural models generalise across similar contexts; more training data." },
    { title: "Over-smoothing", symptom: "Fluent text turns into high-frequency junk like 'the the the'.", detect: "Add-one smoothing on an unseen trigram gives a near-uniform distribution (try trigram + add-one + 'a bird').", fix: "Use better smoothing (Kneser–Ney), interpolate with lower orders, or back off instead." },
    { title: "Context too short", symptom: "Locally fluent but globally incoherent text; the model forgets the subject.", detect: "Compare bigram and trigram output for the same prompt.", fix: "Longer context (higher order or a Transformer); for LLMs, keep the needed facts inside the context window.", scenarioId: "rag-context-truncation" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "metric", question: "In the corpus, 'the' is followed by 'cat' 4 times out of 11 occurrences of 'the' (that are followed by a word). What is P(cat | the) for the bigram model?", options: ["4/46", "4/11 ≈ 0.364", "1/16", "11/4"], answer: 1, explanation: "Bigram probability = C(the cat) / C(the) = 4/11. The denominator is how often the context occurs, not the corpus size." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Prompt 'the cat', bigram, greedy, 8 tokens. What does the output look like?", options: ["A different sentence every run", "the cat sat on the cat sat on the cat — a loop", "It stops after one token", "the cat ate the fish ."], answer: 1, explanation: "cat → sat → on → the → cat … each step takes the most frequent follower. 'the' is followed by 'cat' most often, so the context repeats and greedy decoding cycles." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "With 'None' selected, the prompt 'the zebra' produces nothing. Why?", options: ["'zebra' is too long", "The context 'zebra' never occurs in the corpus, so all counts are 0 and the distribution is undefined", "Greedy decoding is broken", "The vocabulary is full"], answer: 1, explanation: "C(zebra) = 0, so C(zebra w)/C(zebra) is 0/0. Count models need backoff or smoothing for unseen contexts." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "Trigram + add-one smoothing turns 'a bird' into 'a bird the the the …'. What is the better fix?", options: ["Increase tokens to generate", "Switch the unseen-context rule to backoff, so 'bird' (seen) is used as context", "Remove the full stops from the corpus", "Use a 4-gram"], answer: 1, explanation: "'a bird' was never seen, so add-one makes all 16 words equally likely and the tie goes to 'the'. Backoff uses the bigram context 'bird', which predicts 'sang'." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How can you confirm the bars are real probabilities?", options: ["They are always 0.5", "Each is count ÷ (total count of the context), and together (including hidden words) they sum to 1", "They come from a neural network", "They are random"], answer: 1, explanation: "The card shows count ÷ total for each candidate. Summing over every possible next word gives exactly 1 (or with add-one, (C+V)/(C+V) = 1)." },
  ],
};
