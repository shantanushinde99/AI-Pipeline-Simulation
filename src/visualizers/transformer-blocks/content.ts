import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Push 3 tokens through LayerNorm, attention, residual adds and a GELU MLP, one stage at a time.", minutes: 15, hasMetrics: false };

const t = "transformer-blocks";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A transformer block is the unit that gets stacked dozens of times in an LLM. In the common pre-norm form it does two things to every token's vector (the residual stream): x ← x + Attention(LayerNorm(x)), then x ← x + MLP(LayerNorm(x)). Attention moves information between tokens; the MLP transforms each token on its own." },
    { heading: "The pieces", body: "LayerNorm rescales each token to mean 0 and variance 1 (then applies learned γ, β). Self-attention mixes tokens with softmax(QKᵀ/√d)V and an output projection W_O. The MLP expands to a wider hidden size (usually 4× d), applies a non-linearity such as GELU, and projects back. Residual adds put each sub-layer's output on top of its input instead of replacing it." },
    { heading: "Why residuals matter", body: "With residuals, each sub-layer only has to learn an update, the original signal always survives, and gradients have a direct path back through the stack. Without them, repeated attention averaging plus normalization drives token vectors toward each other (rank collapse). Toggle residuals off here and the three tokens end up almost identical after two blocks." },
    { heading: "Where you meet it", body: "GPT, Llama, Mistral and BERT are all stacks of these blocks with variations: RMSNorm instead of LayerNorm, SwiGLU instead of GELU, grouped-query attention, rotary positions. Debugging training instabilities, quantization errors or activation outliers usually means looking at what happens inside one block." },
    { heading: "Assumptions in this visualizer", body: "3 tokens, d = 4, MLP hidden size 8, one attention head. All weights are hand-set (not trained); when 2 blocks are stacked they share the same weights to keep the page small (real models do not). LayerNorm uses γ = 1, β = 0 and ε = 1e−5. GELU uses the tanh approximation. Every number shown is computed from these weights." },
  ],
  code: [
    {
      title: "Pre-norm block — PyTorch",
      source: `import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads, hidden=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, hidden or 4 * d), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden or 4 * d, d))

    def forward(self, x, mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]   # residual 1
        x = x + self.mlp(self.ln2(x))                                       # residual 2
        return x`,
    },
    {
      title: "LayerNorm by hand — NumPy",
      source: `import numpy as np

def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

layer_norm(np.array([1.0, -0.5, 0.3, 0.8]))`,
    },
  ],
  failureModes: [
    { title: "Missing or broken residual path", symptom: "Deep model trains slowly or not at all; representations of different tokens become nearly identical.", detect: "Measure cosine similarity between token vectors per layer; it approaches 1. Toggle residuals off here to see it.", fix: "Use x + f(x) around every sub-layer; check that refactors did not replace + with =." },
    { title: "Post-norm instability at depth", symptom: "Loss spikes or diverges early in training of deep stacks.", detect: "Gradient norms explode in the first layers; needs long warm-up to train.", fix: "Use pre-norm (LayerNorm inside the residual branch, as here) or careful initialization/warm-up." },
    { title: "Activation outliers", symptom: "INT8 quantization badly hurts accuracy.", detect: "A few residual-stream dimensions have values 10–100× larger than the rest.", fix: "Per-channel or mixed-precision quantization (e.g. LLM.int8, SmoothQuant)." },
    { title: "Shape mismatch in the MLP", symptom: "Runtime error or silent broadcasting bug after changing hidden size.", detect: "Check that W₁ is (hidden × d) and W₂ is (d × hidden) so the output can be added to the stream.", fix: "Assert the sub-layer output shape equals the residual stream shape." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "verification", question: "After LayerNorm, what are the mean and variance of every token's vector (γ = 1, β = 0)?", options: ["Mean 1, variance 0", "Mean 0, variance ≈ 1", "Unchanged", "Mean 0, variance 4"], answer: 1, explanation: "LayerNorm subtracts the token's own mean and divides by its standard deviation (with a tiny ε), giving mean 0 and variance just under 1." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "With 2 blocks and residuals off, what happens to the similarity between the three tokens' outputs?", options: ["It drops toward −1", "It stays near the input similarity", "It rises to about 1: the tokens collapse to the same vector", "It becomes undefined"], answer: 2, explanation: "Near-uniform attention gives each token almost the same blend; LayerNorm and the MLP then process near-identical inputs. Without the skip path nothing keeps the tokens apart." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Which number on the page best shows that a block preserved each token's identity?", options: ["The attention weight on itself", "cos(output, input) per token", "The size of the MLP hidden layer", "The LayerNorm ε"], answer: 1, explanation: "Cosine between a token's block output and its input measures how much of the original direction survived. It stays above 0.9 with residuals and falls below 0.5 without them." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "In the GELU step, which of the 8 hidden values change the most?", options: ["Large positive ones", "Negative ones, which are squashed toward 0", "None: GELU is the identity", "Only the first"], answer: 1, explanation: "GELU(z) ≈ z for large positive z, and ≈ 0 for negative z. The dots for negative values drop from the y = z line onto the curve near 0." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Why must the MLP project back from 8 to 4 dimensions?", options: ["To save memory", "So its output has the same size as the residual stream and can be added to it", "Because GELU needs 4 inputs", "It doesn't need to"], answer: 1, explanation: "Residual addition is element-wise: f(x) must have exactly the stream's shape (d = 4 here)." },
  ],
};
