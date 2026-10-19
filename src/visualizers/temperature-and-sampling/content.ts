import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Reshape next-token probabilities with temperature, top-k and top-p, then sample with a seed.", minutes: 12, hasMetrics: false };

const t = "temperature-and-sampling";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A language model outputs one score (logit) per vocabulary token. Decoding turns those scores into a choice. Softmax with temperature converts logits to probabilities, optional truncation (top-k, top-p) removes the unlikely tail, and a random draw picks the token. Greedy decoding skips the randomness and always takes the most likely token." },
    { heading: "Temperature", body: "p_i = exp(z_i/T) / Σ_j exp(z_j/T). T < 1 sharpens the distribution (more predictable, more repetitive), T > 1 flattens it (more varied, more mistakes). As T → 0 the top token gets all the probability, which is exactly greedy decoding. T = 1 uses the model's own distribution." },
    { heading: "Top-k and top-p", body: "Top-k keeps the k most likely tokens regardless of how the probability is spread. Top-p (nucleus sampling) keeps the smallest set of top tokens whose probabilities sum to at least p, so it keeps few tokens when the model is confident and many when it is not. Both renormalize the survivors to sum to 1 before sampling." },
    { heading: "When to use what", body: "Extraction, classification, code and tool arguments: low temperature or greedy for consistency. Creative writing or brainstorming: T around 0.7–1.0 with top-p 0.9–0.95. Evaluations should fix the seed or use T = 0 so results are reproducible; even then, some providers do not guarantee bit-identical outputs." },
    { heading: "Assumptions in this visualizer", body: "The eight logits are fixed, hand-picked numbers standing in for a model's output layer after the prompt 'The weather today is'; no model is called. Real vocabularies have 50k–200k tokens. Top-p is applied after top-k on the renormalized distribution (as Hugging Face does). Randomness comes from a seeded generator (mulberry32), so every draw is reproducible. All arithmetic shown is computed live." },
  ],
  code: [
    {
      title: "sample.py — NumPy",
      source: `import numpy as np

def sample(logits, T=1.0, top_k=0, top_p=1.0, rng=np.random.default_rng(7)):
    z = np.asarray(logits, dtype=float)
    if T == 0:
        return int(z.argmax())                      # greedy
    p = np.exp((z - z.max()) / T)
    p /= p.sum()                                    # softmax with temperature
    if top_k:
        p[np.argsort(p)[:-top_k]] = 0               # keep the k largest
        p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cut = order[np.searchsorted(cum, top_p) + 1:]   # beyond the nucleus
        p[cut] = 0
        p /= p.sum()
    return int(rng.choice(len(p), p=p))

print(sample([3.2, 2.5, 2.1, 1.6, 1.1, 0.8, 0.2, -1.5], T=0.7, top_p=0.9))`,
    },
    {
      title: "Hugging Face generate()",
      source: `out = model.generate(
    **inputs,
    do_sample=True,        # False = greedy
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    max_new_tokens=50,
)
# torch.manual_seed(7) beforehand makes the draws reproducible`,
    },
  ],
  failureModes: [
    { title: "Temperature too high", symptom: "Rambling, off-topic or invented content; malformed JSON.", detect: "Entropy of the final distribution is high; unlikely tokens (here 'purple') get real probability (try 'Break it: T = 2').", fix: "Lower T, add top-p 0.9, or constrain the output format.", scenarioId: "agent-invalid-args" },
    { title: "Greedy repetition", symptom: "Output loops or is bland and identical across runs.", detect: "T = 0 puts all mass on one token: the histogram shows a single bar.", fix: "Use a moderate temperature with top-p, or a repetition penalty." },
    { title: "Top-k cuts good options (or keeps bad ones)", symptom: "With a flat distribution, top-k removes plausible tokens; with a peaked one it still admits junk.", detect: "Compare how many tokens top-k and top-p keep as you change T.", fix: "Prefer top-p, which adapts to the distribution's shape." },
    { title: "Non-reproducible evaluations", symptom: "Scores change between identical eval runs.", detect: "Re-run with the same prompt; different samples appear unless the seed is fixed.", fix: "Use T = 0 or fixed seeds for evals; report variance over several seeds." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "What happens to P(sunny) as you drag temperature from 1.0 down to 0.3?", options: ["It falls toward 1/8", "It rises toward 1", "It stays the same", "It becomes negative"], answer: 1, explanation: "Dividing logits by a small T stretches the gaps, and exponentiation turns bigger gaps into bigger ratios, so the top token's probability approaches 1 (at T = 0 it is exactly 1)." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "With T = 0, what does the 'Repeat' histogram show?", options: ["Eight equal bars", "A single bar on 'sunny' with every draw", "A random mix", "Nothing is sampled"], answer: 1, explanation: "T → 0 makes softmax one-hot on the argmax, so every draw is 'sunny': sampling becomes greedy decoding." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "At T = 1, top-p = 0.9 keeps sunny, cloudy, rainy, warm and cold. Why not 'nice'?", options: ["'nice' has a negative logit", "The cumulative probability already reaches 0.9 at 'cold', so the nucleus stops there", "Top-p always keeps 5 tokens", "'nice' was removed by top-k"], answer: 1, explanation: "Sorted cumulative probabilities: sunny 0.43, + cloudy 0.65, + rainy 0.79, + warm 0.88, + cold 0.94 ≥ 0.9. The nucleus is the smallest prefix reaching p, so 'nice' and the rest are cut." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "An agent's tool arguments are sometimes malformed. Its decoding uses T = 1.6 and no truncation. What is the likeliest fix?", options: ["Raise T to 2", "Lower T (or use greedy) and add top-p", "Increase max tokens", "Change the random seed"], answer: 1, explanation: "High temperature gives low-probability (often invalid) tokens a real chance. Structured outputs want low T or greedy; a seed change only hides the problem." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How can you confirm the sampler is correct, using this visualizer?", options: ["Check that u is always 0.5", "With many draws, the observed frequencies converge to the final probabilities (TV distance shrinks)", "Check that 'sunny' is always picked", "Count the bars"], answer: 1, explanation: "Sampling is correct when the empirical frequencies match the distribution; the total-variation distance shrinks roughly like 1/√n as you increase the number of draws." },
  ],
};
