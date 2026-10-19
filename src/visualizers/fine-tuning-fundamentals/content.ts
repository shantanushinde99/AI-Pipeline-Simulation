import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = {
  summary: "Adapt a pretrained model to a new domain with 8 examples: watch full fine-tuning forget, LoRA train a fraction of the weights, and replay protect the old task.",
  minutes: 15,
  hasMetrics: true,
};

const t = "fine-tuning-fundamentals";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Fine-tuning takes a model that was already trained (pretrained) on lots of data and continues training it on a small labelled dataset for a new task or domain. It reuses everything the model already knows, so a handful of examples can be enough, where training from scratch would need thousands." },
    { heading: "Why the base model fails", body: "Here the base model learned sentiment from movie reviews, where \"long\" is bad (\"too long\") and \"unpredictable\" is good (\"an unpredictable plot\"). On phone reviews those two words mean the opposite: a long battery is good, an unpredictable screen is bad. The model is confidently wrong exactly where the domains disagree." },
    { heading: "Full fine-tuning and catastrophic forgetting", body: "Full fine-tuning updates every weight with gradient descent on the new examples. Phone loss falls fast, but nothing in that loss mentions movies, so the weights for \"long\" and \"unpredictable\" are simply overwritten: movie accuracy drops. This is catastrophic forgetting. It is not a bug in the optimizer; the training objective never asked the model to remember." },
    { heading: "Learning rate and epochs", body: "Too high a learning rate makes each step overshoot and the loss explodes (diverges). Too many epochs on a tiny dataset lets the model memorise it, including the one mislabeled example: training loss keeps falling while validation loss turns back up (overfitting). Watch validation loss and stop at its minimum (early stopping)." },
    { heading: "LoRA: low-rank adapters", body: "LoRA freezes the pretrained W and learns a small correction ΔW = B·A, where B is 8×r and A is r×12. Only r·(8 + 12) numbers train instead of all of W. B starts at zero, so training starts from the exact base model, and because W is never edited, removing the adapter restores the original model perfectly. With the same learning rate and epochs, LoRA moves the model less: it learns the new task a little less and forgets the old one much less. Higher rank learns more and forgets more." },
    { heading: "Freezing and replay", body: "Freezing the base and training only the output head changes very few numbers, but here it cannot fix the flipped words: the frozen first layer still encodes their movie meaning. Replay mixes some original-task data into fine-tuning, so the model is forced to keep both behaviours. It has to learn to use the topic word (plot vs battery) instead of overwriting \"long\"." },
    { heading: "Assumptions in this visualizer", body: "This is a deliberately tiny simulation computed in your browser: a 12-word bag-of-words input, one tanh hidden layer of 8 units and a sigmoid output (113 numbers). The base model is pretrained here on 25 synthetic movie reviews (600 full-batch steps, fixed seed); forgetting is measured on 11 held-out movie reviews and the new task on 28 phone reviews the model never trains on. Fine-tuning is plain mini-batch SGD (batch 4, seeded shuffles) with binary cross-entropy. Exact numbers depend on these choices; the qualitative patterns (forgetting, divergence, overfitting, LoRA's smaller footprint) are what transfer to real models." },
  ],
  code: [
    {
      title: "Full fine-tuning vs LoRA — PyTorch",
      source: `import torch, torch.nn as nn

base = torch.load("sentiment_base.pt")          # pretrained on movie reviews

# Full fine-tuning: every parameter trains
opt = torch.optim.SGD(base.parameters(), lr=0.3)

# LoRA: freeze W, learn a low-rank correction B @ A
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=1, alpha=1.0):
        super().__init__()
        self.linear = linear.requires_grad_(False)          # frozen W, b
        self.A = nn.Parameter(torch.randn(r, linear.in_features) / linear.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(linear.out_features, r))   # B = 0 → starts as the base
        self.scale = alpha / r
    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.A.T @ self.B.T)

base.hidden = LoRALinear(base.hidden, r=1)
base.head.requires_grad_(False)
opt = torch.optim.SGD([p for p in base.parameters() if p.requires_grad], lr=0.3)`,
    },
    {
      title: "Hugging Face PEFT + replay + early stopping",
      source: `from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import concatenate_datasets

model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
model.print_trainable_parameters()   # e.g. "trainable: 0.39% of all params"

# Replay: mix 40% of the original data back in to limit forgetting
train = concatenate_datasets([phone_train, movie_train.shuffle(seed=0).select(range(int(0.4 * len(movie_train))))])

args = TrainingArguments("out", learning_rate=2e-4, num_train_epochs=20,
                         eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True)
trainer = Trainer(model=model, args=args, train_dataset=train,
                  eval_dataset={"new_task": phone_val, "original_task": movie_val},   # track forgetting too
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
trainer.train()`,
    },
  ],
  failureModes: [
    { title: "Catastrophic forgetting", symptom: "The fine-tuned model is great on the new task and noticeably worse at what it used to do.", detect: "Keep an evaluation set from the original task and track it every epoch (the red curve). Here movie accuracy falls from 100% to about 64% with full fine-tuning.", fix: "Use a parameter-efficient method (LoRA) or freeze layers, mix in replay data, lower the learning rate or epochs, or keep separate adapters per task." },
    { title: "Learning rate too high", symptom: "Loss spikes or becomes huge within the first epochs; predictions collapse to one class.", detect: "Training loss rises instead of falling (try the 'Too high' preset).", fix: "Lower the learning rate (fine-tuning usually needs a smaller one than pretraining), add warmup, clip gradients." },
    { title: "Overfitting a tiny dataset", symptom: "Training loss near 0, validation loss climbing; the model reproduces label mistakes.", detect: "Validation loss has a minimum and then rises while training loss keeps falling. Try LoRA rank 3 for 100 epochs.", fix: "Early stopping at the best validation epoch, fewer epochs, more or cleaner data, lower rank." },
    { title: "Frozen features that don't fit the new domain", symptom: "Training only the head plateaus well below the accuracy you need.", detect: "Training loss stays high even with many epochs (frozen base + head here).", fix: "Unfreeze the top layers or add LoRA adapters so the representation itself can change." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "After full fine-tuning on 8 phone reviews, phone accuracy rose but movie accuracy fell from 100% to about 64%. What happened?", options: ["The learning rate was too low", "Catastrophic forgetting: the shared weights for \"long\" and \"unpredictable\" were overwritten, and the loss never asked about movies", "The movie evaluation set leaked into training", "LoRA was accidentally enabled"], answer: 1, explanation: "Fine-tuning minimises loss on the new data only. Where the two domains disagree, the cheapest fix is to overwrite the shared weights, which breaks the old behaviour." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "A LoRA adapter of rank r is added to the 8 × 12 matrix W. How many numbers does it train?", options: ["96", "r × 96", "r × (8 + 12)", "8 + 12"], answer: 2, explanation: "B is 8 × r and A is r × 12, so r·8 + r·12 = r·(8 + 12). For r = 1 that is 20 numbers instead of 96 (and at real scale, 0.39% of a 4096 × 4096 layer for r = 8)." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "Why does a LoRA model produce exactly the base model's outputs before the first update?", options: ["A is initialised to zero", "B is initialised to zero, so B·A = 0 and W + B·A = W", "The learning rate starts at zero", "LoRA skips the first batch"], answer: 1, explanation: "Standard LoRA initialises B to 0 and A randomly. The product is 0, so training starts from the pretrained model and only moves away as B learns." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "You must use full fine-tuning but want to keep movie performance. Which change targets the forgetting directly?", options: ["Raise the learning rate", "Train for more epochs", "Replay: mix part of the original movie data into the fine-tuning set", "Remove the validation set"], answer: 2, explanation: "Replay puts the old task back into the loss, so the model has to find weights that work for both (here by using the topic word). With 40% replay, movie accuracy stays above 90%." },
    { id: `${t}-5`, topicId: t, skill: "diagnosis", question: "Training loss keeps falling to ~0 while validation loss bottoms out at epoch 40 and then rises. What should you do?", options: ["Train longer: the training loss is still improving", "Stop at the epoch with the lowest validation loss (early stopping) or reduce epochs", "Increase the learning rate", "Delete the validation set"], answer: 1, explanation: "The model is memorising the 8 examples, including the mislabeled one. The validation curve tells you when generalisation stopped improving." },
    { id: `${t}-6`, topicId: t, skill: "verification", question: "How can you verify that LoRA left the original model intact?", options: ["Check that training loss went down", "Detach the adapter and confirm original-task metrics match the base model exactly, because W was never modified", "Count the parameters", "Look at the learning-rate schedule"], answer: 1, explanation: "W is frozen, so removing B·A gives back the base weights bit for bit. The result step shows the detached movie metrics equal the base." },
  ],
};
