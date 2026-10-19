import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Roll a ball down a loss surface: see each gradient, each update, and how η, momentum and noise change the path.", minutes: 12, hasMetrics: false };

const t = "gradient-descent";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Gradient descent minimizes a loss by repeating one move: measure the gradient ∇f (the direction of steepest increase) at the current parameters, then step a little the opposite way, p ← p − η·∇f. The learning rate η sets the step size. Every neural network is trained by some variant of this loop." },
    { heading: "The learning rate decides everything", body: "Too small and progress crawls. Just right and the loss drops fast. Too large in a steep direction and each step overshoots to the opposite wall: the path zig-zags (oscillates) and, past a threshold, every bounce is bigger than the last (divergence). For a quadratic, plain GD is stable only if η < 2 / (largest curvature); the elongated bowl here has curvatures 1 and 10, so η must stay below 0.2 even though the shallow x-direction would love a bigger step." },
    { heading: "Momentum and stochastic gradients", body: "Momentum keeps a running velocity v ← β·v + ∇f and moves by −η·v. Consistent directions accumulate speed (it fixes the slow preset), while alternating ones partly cancel, though too much momentum overshoots. Stochastic gradient descent (SGD) estimates the gradient from a mini-batch, so each step is noisy: it is cheap and can shake loose from flat regions, but it never settles exactly; it hovers near the minimum unless η decays." },
    { heading: "Saddles and valleys", body: "In many dimensions most flat points are saddles, not minima: the surface curves up in some directions and down in others. Near the saddle the gradient is tiny, so plain GD stalls for many steps before rolling into a valley. Try the saddle surface and watch the loss curve plateau then drop." },
    { heading: "Assumptions in this visualizer", body: "The loss surfaces are hand-picked 2-D functions with exact analytic gradients, not a real model's loss. 'Stochastic' adds seeded Gaussian noise (σ adjustable) to the exact gradient to mimic mini-batch noise. The outcome badge (slow / good / oscillating / diverging / noisy) is a simple heuristic on the simulated path: whether loss rose overall, how often the y-step reversed direction, and how far loss fell. Positions beyond ±60 are treated as diverged." },
  ],
  code: [
    {
      title: "gd.py — plain, momentum and noisy updates",
      source: `import numpy as np

def grad(p):                      # f(x, y) = 0.5 * (x**2 + 10 * y**2)
    return np.array([p[0], 10 * p[1]])

def descend(p, lr=0.12, beta=0.0, noise=0.0, steps=25, seed=4):
    rng = np.random.default_rng(seed)
    v = np.zeros(2)
    for _ in range(steps):
        g = grad(p) + noise * rng.standard_normal(2)   # SGD-style noisy gradient
        v = beta * v + g                               # beta = 0 → plain GD
        p = p - lr * v
    return p

print(descend(np.array([-4.5, 2.5])))`,
    },
    {
      title: "PyTorch",
      source: `import torch

p = torch.tensor([-4.5, 2.5], requires_grad=True)
opt = torch.optim.SGD([p], lr=0.12, momentum=0.8)
for _ in range(25):
    opt.zero_grad()
    loss = 0.5 * (p[0] ** 2 + 10 * p[1] ** 2)
    loss.backward()            # fills p.grad with ∇f
    opt.step()                 # p ← p − lr · v
print(p, loss.item())`,
    },
  ],
  failureModes: [
    { title: "Learning rate too high (divergence)", symptom: "Loss becomes NaN or inf after a few steps.", detect: "Loss rises step over step; the ball flies off the chart. Try the Diverging preset.", fix: "Lower η (often by 3-10×), add warmup, clip gradients, normalize inputs." },
    { title: "Oscillation in steep directions", symptom: "Loss curve zig-zags; training is noisy even without SGD.", detect: "The update arrow keeps flipping across the narrow valley (Oscillating preset).", fix: "Reduce η, use momentum with a lower η, or an adaptive optimizer (Adam/RMSProp) that scales per-direction steps." },
    { title: "Learning rate too small", symptom: "Loss falls steadily but painfully slowly; the run ends far from the minimum.", detect: "Each step's move is tiny compared with the distance to the minimum (Slow preset).", fix: "Increase η, use momentum, or a learning-rate schedule (warmup then decay)." },
    { title: "Plateau at a saddle", symptom: "Loss stalls for many steps, then suddenly drops.", detect: "Gradient norm is near zero but loss is not at its minimum (Saddle surface).", fix: "Momentum or gradient noise helps escape; check initialization; be patient before declaring convergence." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "On the elongated bowl f = ½(x² + 10y²), which learning rate is the largest that does not diverge with plain GD?", options: ["Just under 2", "Just under 1", "Just under 0.2", "Any value works because the bowl is convex"], answer: 2, explanation: "Stability needs η < 2 / (largest curvature) = 2/10 = 0.2. The Diverging preset (0.21) is just past it." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Which way does the red gradient arrow point relative to the ball's actual move?", options: ["Same direction", "Opposite direction (gradient points uphill, the move goes downhill)", "Perpendicular", "It depends on the learning rate"], answer: 1, explanation: "The move is −η·∇f. Without momentum or noise it is always exactly opposite the gradient, scaled by η." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "With η = 0.18 on the bowl, the loss still falls but the path zig-zags in y. Why?", options: ["The gradient is wrong", "In y the step factor is 1 − 0.18·10 = −0.8: each step overshoots to the other side, shrinking by 20%", "Momentum is on", "The surface has two minima"], answer: 1, explanation: "Along y, y_new = (1 − η·10)·y. A negative factor flips the sign each step; |−0.8| < 1 so it still converges, slowly and noisily." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "The Slow preset (η = 0.02) ends far from the minimum. Which change helps most without risking divergence?", options: ["Turn on momentum", "Turn on stochastic noise", "Reduce the number of steps", "Start closer to the saddle"], answer: 0, explanation: "Momentum accumulates the consistent downhill direction, so effective steps grow along the shallow x-direction; the final loss drops by well over half." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How can you tell a saddle plateau from true convergence using only the loss curve and gradient?", options: ["You can't", "At a saddle the gradient is tiny but the loss later drops; at a minimum the loss stays flat and nearby moves in every direction increase it", "The loss is negative at a saddle", "Saddles always produce NaN"], answer: 1, explanation: "Both have a small gradient. A minimum curves up in every direction; a saddle curves down in some, so a small push (noise or momentum) leads to further decrease." },
  ],
};
