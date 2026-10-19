import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch W_Q, W_K and W_V turn embeddings into queries, keys and values, then edit a weight or swap keys.", minutes: 14, hasMetrics: false };

const t = "query-key-and-value";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Attention does not compare raw embeddings. Each token's embedding x is multiplied by three learned matrices: W_Q gives a query (what this token is looking for), W_K gives a key (what this token can be found by) and W_V gives a value (what this token hands over when attended to). Scores come from q·k; outputs are weighted sums of v." },
    { heading: "Why three separate projections", body: "Separating 'how I am found' (key) from 'what I contribute' (value) lets a model match on one property and copy another. In the example, \"drank\" finds \"tea\" because tea's key says drinkable, but what flows back is tea's value vector. Swap two keys and attention moves while values stay where they were: the model retrieves the wrong content." },
    { heading: "The library analogy", body: "Query = the request you type at the library desk. Key = the catalogue card on each shelf. Value = the book itself. The difference from a real library: attention never returns one book. It returns a blend of every book, weighted by how well each card matched (softmax)." },
    { heading: "Where it shows up", body: "Every attention layer in every transformer, every head, every token. Multi-head attention simply runs several smaller W_Q/W_K/W_V sets in parallel. KV caching in LLM serving stores the K and V rows of past tokens so they are not recomputed during generation." },
    { heading: "Assumptions in this visualizer", body: "Embeddings are 4-D and hand-placed with named features (person, action, temperature, drinkable). W_Q, W_K, W_V are hand-designed so query/key dimensions have readable meanings (seek agent, seek action, seek modifier, seek drinkable); nothing is trained. Single head, no biases, no output projection. All numbers are computed from the matrices on screen and change when you edit them." },
  ],
  code: [
    {
      title: "qkv.py — NumPy",
      source: `import numpy as np

X = np.array([[1, 0, 0, 0],      # Kim
              [0, 1, 0, .2],     # drank
              [0, 0, 1, .3],     # hot
              [0, 0, .2, 1]])    # tea
W_q = np.array([[0, 1, 0, 0], [1.5, 0, 0, 1], [0, 0, 0, 1], [0, 2, 1.5, 0]])
W_k = np.diag([2, 2, 1.5, 2])
W_v = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .5, 1]])

Q, K, V = X @ W_q.T, X @ W_k.T, X @ W_v.T
scores = Q[1] @ K.T / np.sqrt(4)          # query = "drank"
w = np.exp(scores - scores.max()); w /= w.sum()
print(w.round(3), w @ V)                  # tea gets the most weight`,
    },
    {
      title: "PyTorch: one module, three projections",
      source: `import torch.nn as nn

class Head(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x):                 # x: (batch, seq, d_model)
        q, k, v = self.q(x), self.k(x), self.v(x)
        att = (q @ k.transpose(-2, -1)) / k.shape[-1] ** 0.5
        return att.softmax(-1) @ v`,
    },
  ],
  failureModes: [
    { title: "Keys and values misaligned", symptom: "The model attends to a sensible position but copies the wrong content.", detect: "Swap-keys here: attention weights move, values stay put, and the output changes. In real systems this shows up as KV-cache bugs (wrong position or wrong layer's cache).", fix: "Test the KV cache: outputs with and without caching must match token for token." },
    { title: "Transposed weight matrix", symptom: "Loading weights from another framework gives garbage attention.", detect: "Compare one hand-computed q = W_Q·x against the library output; shapes may still match for square matrices.", fix: "Check the convention (x·Wᵀ vs W·x) when porting weights; add a known-answer test." },
    { title: "Query/key scale blow-up", symptom: "Attention becomes one-hot early in training; loss spikes.", detect: "Track the magnitude of q·k; set W_Q entries large here and watch one weight approach 100%.", fix: "Keep the 1/√d scale, use QK-norm or careful initialization." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "With default settings and \"drank\" as the query, which token gets the most attention?", options: ["Kim", "drank", "hot", "tea"], answer: 3, explanation: "q_drank has a large 'seek drinkable' entry (2.0) and tea's key has a large 'is drinkable' entry (2.0), so q·k = 4.06 is the largest score." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "Set W_Q[3][1] (how much an action seeks drinkables) to 0. What happens for the query \"drank\"?", options: ["Attention to tea rises", "Attention to tea falls and the weights spread out", "Nothing changes", "The values change"], answer: 1, explanation: "q[3] = W_Q[3][1]·1 + … drops to 0, so tea's score loses its 4.0 term. Scores become similar and softmax spreads the weight." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "After swapping the keys of \"hot\" and \"tea\", \"drank\" now attends mostly to \"hot\". Why does the output change?", options: ["Because the query changed", "Because attention weights follow the keys, but the content returned is each position's own value", "Because softmax is no longer normalized", "Because the embeddings were swapped"], answer: 1, explanation: "Keys decide where to look. Values are untouched, so the highest weight now multiplies hot's value instead of tea's, and the output shifts." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "In the mix step, the weights are [w₀..w₃] and the output dimension 'drinkable' is out[3]. How is out[3] computed?", options: ["max_j v_j[3]", "Σ_j w_j · v_j[3]", "q[3] · k[3]", "v_top[3]"], answer: 1, explanation: "Each output dimension is a weighted sum of that dimension across all value vectors. The table in the card lists each w_j·v_j[3] term." },
  ],
};
