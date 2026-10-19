import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Follow real numbers through weights, biases and activations in a 2-3-1 network.", minutes: 12, hasMetrics: true };

const t = "forward-propagation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Forward propagation is how a neural network turns inputs into an output. Each neuron computes a weighted sum of its inputs plus a bias (z = Σ wᵢxᵢ + b) and passes it through an activation function (a = f(z)). Layer by layer, those activations become the next layer's inputs." },
    { heading: "Why activations matter", body: "Without a non-linear activation, stacking layers collapses into a single linear function, so the network could never separate XOR. Switch the hidden activation to 'linear' and the hand-tuned XOR weights stop working: no choice of weights in a linear model can classify all four points." },
    { heading: "Inputs and outputs", body: "Inputs are numeric features. The output here is a sigmoid, so ŷ lies in (0, 1) and is read as the probability of class 1. A threshold (0.5) turns it into a class label." },
    { heading: "Assumptions in this visualizer", body: "The network is tiny (2-3-1) and the weights are set by hand or from a fixed seed, not learned. The arithmetic is real: every number shown is computed from the weights on screen. Training (backpropagation + gradient descent) has its own visualizer; the ∂ŷ/∂param readout hints at what it uses." },
  ],
  code: [
    {
      title: "forward.py — NumPy",
      source: `import numpy as np

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

W1 = np.array([[1, -1], [-1, 1], [0.5, 0.5]])  # (hidden, inputs)
b1 = np.zeros(3)
W2 = np.array([6, 6, 0])
b2 = -3

def forward(x):
    z1 = W1 @ x + b1        # weighted sums of hidden layer
    h = relu(z1)            # hidden activations
    z2 = W2 @ h + b2        # output weighted sum
    return sigmoid(z2)      # probability of class 1

print(forward(np.array([1, 0])))  # ≈ 0.9526`,
    },
    {
      title: "Same network in PyTorch",
      source: `import torch, torch.nn as nn

net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid())
with torch.no_grad():
    net[0].weight.copy_(torch.tensor([[1., -1.], [-1., 1.], [.5, .5]]))
    net[0].bias.zero_()
    net[2].weight.copy_(torch.tensor([[6., 6., 0.]]))
    net[2].bias.fill_(-3.)
print(net(torch.tensor([[1., 0.]])))`,
    },
  ],
  failureModes: [
    { title: "Dead ReLU neurons", symptom: "A hidden neuron outputs 0 for every input; changing its outgoing weight does nothing.", detect: "Inspect hidden activations across a batch; the sensitivity readout drops to 0.", fix: "Adjust initialization or bias, lower the learning rate, or use a leaky activation." },
    { title: "Saturated sigmoid", symptom: "Output stuck near 0 or 1 regardless of small input changes.", detect: "|z| is large (> ~5); ∂ŷ/∂w is near zero, so learning stalls.", fix: "Normalize inputs, scale weights down, use better initialization." },
    { title: "Unscaled features", symptom: "One feature dominates the weighted sum (try the exam dataset's hour values with random weights).", detect: "Compare the magnitude of each wᵢxᵢ term in the calculation panel.", fix: "Standardize inputs before the first layer; apply the same scaling at inference." },
    { title: "Feature order mismatch at inference", symptom: "Predictions are confidently wrong in production.", detect: "Swap x1 and x2 here and watch the output change for the same 'record'.", fix: "Pin the feature schema and validate column order at the serving boundary.", scenarioId: "ml-feature-order" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "With the XOR preset and input (1, 0), h1 = relu(x1 − x2) = 1. What happens to ŷ if you set the weight h1 → y from 6 to 0?", options: ["ŷ increases toward 1", "ŷ drops to σ(−3) ≈ 0.047", "ŷ is unchanged because h1 is a hidden neuron", "The network errors out"], answer: 1, explanation: "z_y = 0·1 + 6·0 + 0·h3 − 3 = −3, and σ(−3) ≈ 0.047. The only active path to the output was h1 → y." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Why can't the XOR preset work if you switch the hidden activation to 'linear'?", options: ["Linear activations are slower", "A stack of linear layers is still a linear function, and XOR is not linearly separable", "The sigmoid output requires ReLU inputs", "Bias terms are ignored by linear activations"], answer: 1, explanation: "Composing linear maps gives a linear map. No single line can separate XOR's classes, so a non-linear activation is required." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "For input (0, 0) every hidden neuron outputs 0 under ReLU. Which parameter can still change ŷ?", options: ["Any input → hidden weight", "Any hidden → output weight", "Only the output bias", "None"], answer: 2, explanation: "All weighted sums are 0 (no bias in the preset), ReLU gives 0, so every downstream product is 0. Only b_y reaches the output; the sensitivity readout confirms ∂ŷ/∂w = 0 for the weights." },
    { id: `${t}-4`, topicId: t, skill: "metric", question: "Two weight settings both get 100% accuracy on XOR. One has log loss 0.05, the other 0.6. What does that tell you?", options: ["They are equivalent", "The first is more confident in the correct answers", "The second generalizes better", "Accuracy is wrong"], answer: 1, explanation: "Log loss rewards probabilities that are confident and correct. Accuracy only checks which side of 0.5 each prediction falls on." },
  ],
};
