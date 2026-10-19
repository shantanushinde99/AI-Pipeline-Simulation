import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Prove attention is order-blind, then fix it with sinusoidal encodings and RoPE rotations.", minutes: 14, hasMetrics: false };

const t = "positional-information";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Self-attention treats its input as a set: scores q·k and the weighted sum of values do not depend on where a token sits. Permute the tokens and the outputs are permuted the same way (permutation equivariance). Positional information is whatever we add so that \"dog bites man\" and \"man bites dog\" produce different representations." },
    { heading: "Sinusoidal (absolute) encodings", body: "The original Transformer adds a fixed vector PE(pos) to each embedding: sines and cosines whose frequency falls geometrically across dimensions. Low dimensions change quickly with position, high dimensions slowly, so each position gets a unique fingerprint and nearby positions get similar ones. Learned absolute position embeddings (GPT-2, BERT) work the same way but are trained." },
    { heading: "Rotary embeddings (RoPE)", body: "Most modern LLMs (Llama, Mistral, Qwen) rotate each 2-D pair of query and key dimensions by an angle proportional to position, instead of adding a vector. Because R(mθ)q · R(nθ)k = qᵀR((n−m)θ)k, the attention score depends only on the relative distance n − m. Values are not rotated." },
    { heading: "Why it matters in practice", body: "Position schemes decide how well a model handles long inputs. Absolute learned embeddings cannot go past the trained length. RoPE extrapolates somewhat and can be stretched (position interpolation, NTK/YaRN scaling) to extend context windows." },
    { heading: "Assumptions in this visualizer", body: "Word embeddings are 4-D and hand-placed; W_Q, W_K, W_V are fixed hand-set matrices (not trained); one bidirectional head with no causal mask (a causal mask alone would already leak order). The sinusoidal formula uses the real base 10000, which with d = 4 means the last two dimensions barely move over a few positions. The RoPE demo uses a single hand-picked 2-D query and key. A PE strength multiplier is added as a teaching knob; real models use strength 1." },
  ],
  code: [
    {
      title: "Sinusoidal PE — NumPy",
      source: `import numpy as np

def sinusoidal_pe(n_pos, d, base=10000):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / base ** (i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe

x = embeddings + sinusoidal_pe(len(embeddings), embeddings.shape[1])`,
    },
    {
      title: "RoPE — PyTorch (rotate-half form)",
      source: `import torch

def rope(x, pos, base=10000):            # x: (seq, d), d even
    d = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2) / d)
    ang = pos[:, None] * inv_freq[None, :]   # (seq, d/2)
    cos, sin = ang.cos().repeat(1, 2), ang.sin().repeat(1, 2)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin       # apply to q and k, not v`,
    },
  ],
  failureModes: [
    { title: "No positional signal at all", symptom: "The model confuses word order (\"dog bites man\" = \"man bites dog\"), bag-of-words behaviour.", detect: "Permute the input and compare outputs: a max difference of 0 means order-blind. Set PE strength to 0 here to see it.", fix: "Add absolute embeddings or apply RoPE to queries and keys." },
    { title: "Sequence longer than the trained positions", symptom: "Quality collapses beyond the training context length.", detect: "Evaluate perplexity as a function of position; it jumps past the trained limit.", fix: "Use RoPE scaling (position interpolation, YaRN) and fine-tune briefly on longer sequences." },
    { title: "Position ids off by one or reset wrongly", symptom: "Generation with a KV cache diverges from the non-cached result.", detect: "Compare cached vs. uncached logits; check that each new token gets position = cache length.", fix: "Pass explicit position ids; test padding (left vs right) and packed sequences." },
    { title: "Rotating the values", symptom: "Subtle quality loss after a custom RoPE implementation.", detect: "Check which tensors the rotation touches.", fix: "Rotate only q and k; scores need relative position, values do not." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "verification", question: "With PE strength 0, you reorder \"dog bites man\" to \"man bites dog\". What is the max difference between each word's outputs?", options: ["About 0.5", "Exactly 0", "It depends on the random seed", "Infinite"], answer: 1, explanation: "Without positions, attention is permutation-equivariant: each word's output depends only on the set of words, so the difference is 0 (up to floating point)." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "In the d = 4 sinusoidal PE matrix, which columns change most between positions 0 and 7?", options: ["sin(p) and cos(p)", "sin(p/100) and cos(p/100)", "All equally", "None: PE is constant"], answer: 0, explanation: "Dimension pair i uses frequency 1/10000^(2i/d). For i = 0 that is 1 (fast); for i = 1 it is 1/100, so those columns barely move over 8 positions." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "With RoPE, q at position 1 and k at position 3 give score s. Which pair also gives exactly s?", options: ["m = 3, n = 1", "m = 5, n = 7", "m = 1, n = 4", "m = 0, n = 0"], answer: 1, explanation: "RoPE scores depend only on n − m. (5, 7) has the same distance 2 as (1, 3). (3, 1) has distance −2, which rotates the other way." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "In order B with PE on, the middle word keeps its position but its output still changes. Why?", options: ["A bug", "Its own PE changed", "Its neighbours' inputs changed (they moved), so its scores and the values it mixes changed", "Softmax is random"], answer: 2, explanation: "The middle word's own x + PE is the same, but the keys and values of the other tokens now carry different positions, so its attention output changes." },
  ],
};
