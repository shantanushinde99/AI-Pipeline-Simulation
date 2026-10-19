import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch the error signal flow backward through a tiny network, edge by edge, then update every weight.", minutes: 12, hasMetrics: false };

const t = "backpropagation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Backpropagation computes ∂L/∂w for every weight in one backward sweep. After a forward pass and a loss, it starts at the output with the error signal δ_y = ∂L/∂z_y and walks the network in reverse, applying the chain rule at each step: multiply by the weight the signal came through, then by the activation's slope f′(z). Each weight's gradient is the δ of the neuron it feeds times the value that travelled along it in the forward pass." },
    { heading: "Why it matters", body: "Gradient descent needs a gradient for every parameter. Nudging each weight and re-running the network (finite differences) costs one forward pass per weight; backprop gets all of them for about the cost of one extra pass, by reusing the forward values. That is what makes training networks with billions of weights possible." },
    { heading: "Where it goes wrong", body: "The backward signal is a product of weights and slopes. Slopes near 0 (saturated sigmoid/tanh, dead ReLU) shrink it: vanishing gradients, and those weights stop learning. Large weights grow it: exploding gradients. Even with a correct gradient, a learning rate that is too big overshoots and the loss goes up, which you can trigger with the η slider." },
    { heading: "Assumptions in this visualizer", body: "The 2-3-1 network, its forward pass and its activations are reused from the neural-network visualizer. Loss is binary cross-entropy on one XOR example (batch size 1), so ∂L/∂z_y = ŷ − t exactly. All gradients are computed analytically in simulation.ts and checked against central finite differences in the tests. Starting weights are seeded pseudo-random or hand-picked, not trained. Real frameworks do the same arithmetic over batches with automatic differentiation." },
  ],
  code: [
    {
      title: "numpy — one backward pass by hand",
      source: `import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))

def step(x, t, W1, b1, w2, b2, lr=0.5):
    z1 = W1 @ x + b1; h = np.tanh(z1)          # forward
    y = sig(w2 @ h + b2)
    loss = -(t * np.log(y) + (1 - t) * np.log(1 - y))
    dz2 = y - t                                 # backward: sigmoid + BCE
    gw2, gb2 = dz2 * h, dz2
    dz1 = dz2 * w2 * (1 - h**2)                 # chain rule through tanh
    gW1, gb1 = np.outer(dz1, x), dz1
    return (W1 - lr * gW1, b1 - lr * gb1, w2 - lr * gw2, b2 - lr * gb2), loss`,
    },
    {
      title: "PyTorch — autograd does the same sweep",
      source: `import torch
net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
opt = torch.optim.SGD(net.parameters(), lr=0.5)
x, t = torch.tensor([[1., 0.]]), torch.tensor([[1.]])

loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), t)
opt.zero_grad()
loss.backward()          # fills p.grad for every parameter
opt.step()               # p ← p − lr · p.grad`,
    },
  ],
  failureModes: [
    { title: "Vanishing gradients", symptom: "Early-layer weights barely change; loss plateaus.", detect: "Log per-layer gradient norms; they shrink by orders of magnitude toward the input. Watch f′(z) under each hidden neuron here.", fix: "ReLU-family activations, careful initialization, normalization layers, residual connections." },
    { title: "Dead ReLU", symptom: "A neuron outputs 0 for every input and never recovers.", detect: "Fraction of zero activations per unit; zero gradient on its incoming weights (red f′(z)=0 here).", fix: "Lower learning rate, Leaky ReLU/GELU, better initialization." },
    { title: "Learning rate too high", symptom: "Loss jumps up or becomes NaN after an update.", detect: "Loss after the step is higher than before; gradient norm spikes.", fix: "Reduce η, use warmup, clip gradients, or an adaptive optimizer." },
    { title: "Wrong gradient implementation", symptom: "Training is unstable or stalls even with sensible settings.", detect: "Gradient check: compare analytic gradients to finite differences on a tiny network (as this visualizer's tests do).", fix: "Fix the derivative; prefer framework autograd over hand-written backward passes." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "At the output (sigmoid + cross-entropy), what is the error signal δ_y?", options: ["ŷ − t", "(ŷ − t)²", "log ŷ", "t / ŷ"], answer: 0, explanation: "The sigmoid's derivative cancels the cross-entropy's 1/ŷ terms, leaving ∂L/∂z_y = ŷ − t. Check the 'Error signal' step.", skill: "metric" },
    { id: `${t}-2`, topicId: t, question: "With ReLU, a hidden neuron shows f′(z) = 0. What happens to the gradients of its incoming weights?", options: ["They are exactly 0, so those weights don't change", "They double", "They equal δ_y", "They become negative"], answer: 0, explanation: "δ_h = ∂L/∂h · f′(z). A zero slope blocks the signal, so x_i · δ_h = 0 for both incoming weights.", skill: "diagnosis" },
    { id: `${t}-3`, topicId: t, question: "Set η to 8 and play to the last step. What usually happens?", options: ["The loss goes up because the step overshoots", "The loss drops to exactly 0", "Gradients become zero", "Nothing changes"], answer: 0, explanation: "The gradient is correct only locally. A huge step jumps past the minimum along that direction, so the new loss can be higher.", skill: "parameter" },
    { id: `${t}-4`, topicId: t, question: "Why is ∂L/∂w_{h1→y} equal to δ_y · h1?", options: ["z_y depends on that weight with slope h1, and the chain rule multiplies slopes", "Because h1 is always 1", "It's a convention", "Because the loss is linear"], answer: 0, explanation: "z_y = Σ w_j h_j + b, so ∂z_y/∂w_1 = h_1. Chain rule: ∂L/∂w_1 = ∂L/∂z_y · ∂z_y/∂w_1 = δ_y h_1.", skill: "prediction" },
    { id: `${t}-5`, topicId: t, question: "How can you verify a hand-written backward pass?", options: ["Compare each gradient with a finite difference (L(w+ε) − L(w−ε)) / 2ε", "Check that the loss is positive", "Run more epochs", "Check the weights are small"], answer: 0, explanation: "A gradient check compares analytic and numeric gradients; they should agree to many decimals.", skill: "verification" },
  ],
};
