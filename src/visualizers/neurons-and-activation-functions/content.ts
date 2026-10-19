import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Push inputs through one neuron and five activations; see where gradients vanish or die.", minutes: 10, hasMetrics: false };

const t = "neurons-and-activation-functions";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A neuron computes a weighted sum of its inputs plus a bias, z = Σ wᵢxᵢ + b, then passes z through an activation function, a = f(z). The weighted sum is linear; the activation adds the bend that lets stacked layers represent curved decision boundaries." },
    { heading: "The common activations", body: "ReLU = max(0, z) is cheap and the default for many networks. Leaky ReLU keeps a small slope for negative z. Sigmoid squashes to (0, 1) and tanh to (−1, 1); both are flat for large |z|. GELU is a smooth ReLU-like curve used in Transformers such as GPT and BERT." },
    { heading: "Why the derivative matters", body: "Backpropagation multiplies the error signal by f′(z) at every neuron it passes through. Where the curve is flat (sigmoid/tanh at large |z|, ReLU for z ≤ 0) f′(z) is near or exactly 0, so the weights feeding that neuron barely change. Across many layers these small factors multiply: the vanishing-gradient problem." },
    { heading: "When to pick which", body: "Hidden layers: ReLU, leaky ReLU or GELU. Output layers: sigmoid for a binary probability, softmax for classes, linear for regression. tanh still appears inside recurrent cells. Good initialization and normalization keep z in the region where the slope is useful." },
    { heading: "Assumptions in this visualizer", body: "One neuron with three inputs and hand-set weights; nothing is trained. GELU uses the common tanh approximation, and leaky ReLU uses α = 0.1 (frameworks default to 0.01) so the slope is visible. The 'flat' threshold |f′(z)| < 0.05 is a teaching choice. The 'stacked layers' bars assume every layer sits at the same z and ignore weights: a simplification of real backprop." },
  ],
  code: [
    {
      title: "neuron.py — NumPy",
      source: `import numpy as np
from math import erf, sqrt

x = np.array([1.0, 0.5, -1.0])
w = np.array([0.8, -0.4, 0.3])
b = 0.5

z = w @ x + b                      # 0.8

relu    = lambda z: np.maximum(0, z)
leaky   = lambda z, a=0.01: np.where(z > 0, z, a * z)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
gelu    = lambda z: 0.5 * z * (1 + erf(z / sqrt(2)))

for f in (relu, leaky, sigmoid, np.tanh, gelu):
    print(f.__name__, f(z))

# slope of sigmoid: tiny when |z| is large -> vanishing gradient
s = sigmoid(8.0); print(s * (1 - s))  # 0.000335`,
    },
    {
      title: "PyTorch: activations and their gradients",
      source: `import torch, torch.nn.functional as F

z = torch.tensor([-4.0, 0.8, 8.0], requires_grad=True)
for name, f in [("relu", F.relu), ("leaky", F.leaky_relu), ("sigmoid", torch.sigmoid),
                ("tanh", torch.tanh), ("gelu", F.gelu)]:
    z.grad = None
    f(z).sum().backward()
    print(name, z.grad)   # relu gives 0 at z=-4: a dead unit`,
    },
  ],
  failureModes: [
    { title: "Dead ReLU", symptom: "A unit outputs 0 for every example and its incoming weights stop changing.", detect: "Track the fraction of zero activations per layer; here the derivative plot shows slope 0 (try the 'Dead ReLU' preset).", fix: "Smaller learning rate, better init, less negative bias, or leaky ReLU / GELU." },
    { title: "Saturated sigmoid or tanh", symptom: "Training loss plateaus early; early layers barely update.", detect: "Large |z| values; f′(z) near 0 in the derivative plot (try 'Saturated sigmoid').", fix: "Normalize inputs, scale initial weights (Xavier/He), use ReLU-family activations in hidden layers, add normalization layers." },
    { title: "Vanishing gradients in deep stacks", symptom: "Deep network trains no better than a shallow one.", detect: "Per-layer gradient norms shrink toward the input layers; the f′(z)ⁿ bars show how fast.", fix: "Residual connections, normalization, ReLU/GELU activations, careful initialization." },
    { title: "Wrong output activation", symptom: "Regression predictions never exceed 1, or class probabilities don't sum to 1.", detect: "Check the output range against the target range.", fix: "Linear output for regression, sigmoid for one probability, softmax for mutually exclusive classes." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "With the 'Healthy' preset (z = 0.8) and ReLU, which single change makes the neuron dead for this input?", options: ["Set w₁ to 3.0", "Set the bias to −4", "Switch to leaky ReLU", "Set x₃ to 2"], answer: 1, explanation: "z = 0.3 − 4 = −3.7 ≤ 0, so ReLU outputs 0 and its slope is 0. A larger w₁ or x₃ = 2 keeps z positive, and leaky ReLU never has slope 0." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "In the compare step with z = 7.5 (the 'Saturated sigmoid' preset), which activation has the smallest slope?", options: ["ReLU", "Leaky ReLU", "GELU", "Sigmoid or tanh (both near 0)"], answer: 3, explanation: "Sigmoid′(7.5) ≈ 0.00055 and tanh′(7.5) ≈ 0.0000012. ReLU, leaky ReLU and GELU all have slope ≈ 1 for large positive z." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "A hidden layer's weights stop changing during training and its activations are all exactly 0. Most likely cause?", options: ["Sigmoid saturation", "Dead ReLUs (z ≤ 0 for every example)", "Learning rate too small", "Too few output classes"], answer: 1, explanation: "Exactly 0 outputs plus zero updates is the ReLU signature: for z ≤ 0 both the output and the slope are 0. Saturated sigmoids output values near 0 or 1, not exactly 0." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "Sigmoid's slope is at most 0.25. Ignoring weights, what fraction of the gradient survives 5 sigmoid layers at best?", options: ["0.25", "About 0.001", "About 0.5", "It stays 1"], answer: 1, explanation: "0.25⁵ ≈ 0.00098. This is why deep sigmoid networks were hard to train, and why the stacked-layer bars turn red." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Which activation keeps a non-zero gradient for every negative z?", options: ["ReLU", "Leaky ReLU", "Step function", "Clipped ReLU"], answer: 1, explanation: "Leaky ReLU has slope α (0.1 here) for z ≤ 0, so a neuron can recover. ReLU's slope there is exactly 0." },
  ],
};
