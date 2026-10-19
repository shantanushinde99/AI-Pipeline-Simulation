import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Score, scale, softmax and mix: watch \"it\" find \"animal\" in a real 10-token attention pass.", minutes: 14, hasMetrics: false };

const t = "self-attention";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Self-attention lets every token in a sequence build a new representation of itself by looking at every other token. Each token's query is compared with every key (dot product), the scores are scaled by 1/√d and turned into weights with softmax, and the token's output is the weighted sum of all value vectors." },
    { heading: "Why it matters", body: "It is how a transformer resolves context: in \"the animal didn't cross the street because it was tired\", the representation of \"it\" needs information from \"animal\". Attention gives a direct, one-step path between any two positions, no matter how far apart, and all positions are computed in parallel." },
    { heading: "Why scale by √d", body: "If query and key entries have unit variance, their dot product has variance d. Large scores push softmax into a near one-hot regime where gradients vanish. Dividing by √d keeps the spread roughly constant as d grows. Turn scaling off here and watch the entropy fall." },
    { heading: "Masks", body: "Decoder models (GPT-style) add a causal mask: scores for future positions are set to −∞ before softmax, so each token only sees itself and the past. Encoders (BERT-style) attend in both directions." },
    { heading: "Assumptions in this visualizer", body: "Embeddings are 4-dimensional and hand-placed with named features (animate, place, pronoun, verb-ish); W_Q and W_K are hand-designed so pronoun queries look for animate keys; W_V is the identity so outputs keep readable feature names. One head, one layer, no training. Every number on screen is computed from these matrices. The \"Random\" preset uses seeded random matrices to show what an untrained head does." },
  ],
  code: [
    {
      title: "attention.py — NumPy",
      source: `import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v, causal=False):
    Q, K, V = X @ W_q.T, X @ W_k.T, X @ W_v.T      # (n, d) each
    scores = Q @ K.T / np.sqrt(Q.shape[-1])        # (n, n)
    if causal:
        scores = np.where(np.tri(len(X), dtype=bool), scores, -np.inf)
    A = softmax(scores)                            # rows sum to 1
    return A @ V, A`,
    },
    {
      title: "PyTorch built-in",
      source: `import torch.nn.functional as F

# q, k, v: (batch, heads, seq, d_head)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`,
    },
  ],
  failureModes: [
    { title: "Forgetting the 1/√d scale", symptom: "Training is unstable; attention collapses onto one token early and gradients vanish.", detect: "Attention entropy per head is near 0 at initialization; softmax inputs have very large magnitude.", fix: "Divide scores by √d_head (or use a fused kernel that does it)." },
    { title: "Missing or wrong causal mask", symptom: "Training loss is suspiciously low, then generation is garbage.", detect: "Attention weights on future positions are non-zero (toggle the causal mask here and compare the last columns).", fix: "Apply an upper-triangular −∞ mask before softmax; test that position i never reads j > i." },
    { title: "Padding tokens get attention", symptom: "Outputs change when you add padding to a batch.", detect: "Non-zero attention weight on pad positions.", fix: "Pass an attention/key-padding mask so pad keys get −∞." },
    { title: "Over-reading attention maps as explanations", symptom: "A heatmap 'shows' the model used a word, but ablating it changes nothing.", detect: "Compare attention with gradient- or ablation-based attributions.", fix: "Treat attention weights as one signal, not proof; values and later layers also matter." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "With the default settings, which token gets the largest attention weight when \"it\" is the query?", options: ["it", "street", "animal", "tired"], answer: 2, explanation: "The pronoun query has a large component on the \"animate\" feature, and \"animal\"'s key advertises it strongly, so q·k is largest there (≈ 53% after softmax)." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "You switch off \"Scale by 1/√d\". What happens to \"it\"'s attention distribution?", options: ["It becomes uniform", "It becomes sharper (lower entropy); \"animal\" gets even more weight", "Nothing: scaling does not change softmax", "All weights become negative"], answer: 1, explanation: "Without dividing by 2, the scores are twice as far apart, so softmax concentrates more mass on the top score. The Key values panel shows entropy dropping." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "With the causal mask on and \"it\" as the query, which weights are exactly 0?", options: ["\"was\" and \"tired\"", "\"the\" and \"animal\"", "Only \"it\" itself", "None"], answer: 0, explanation: "Causal masking sets scores for positions after the query to −∞; e^−∞ = 0. \"was\" and \"tired\" come after \"it\"." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "How can you check that the output vector of \"it\" really is the weighted sum of values?", options: ["Check that it equals the query", "Multiply each token's weight by its value and add per dimension; compare with the stacked bars", "Check the heatmap diagonal", "It cannot be checked"], answer: 1, explanation: "In the mix step, each segment of a bar is w_j × v_j[d]; their total is out[d]. The events panel lists the result for comparison." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "With \"Random (untrained)\" weights, \"it\" no longer attends to \"animal\". What fixes this in a real model?", options: ["Removing softmax", "Training W_Q and W_K on data so useful query/key alignments are learned", "Adding more tokens", "Setting W_V to zero"], answer: 1, explanation: "The pronoun→antecedent link lives in the projection matrices. Random matrices give arbitrary scores; training shapes them so matching queries and keys align." },
  ],
};
