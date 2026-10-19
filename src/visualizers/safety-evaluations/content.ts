import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Score a red-team set, slide a block threshold and watch over- and under-refusal trade.", minutes: 12, hasMetrics: false };

const t = "safety-evaluations";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A safety evaluation runs a labelled red-team set through the system's safety layer and counts its mistakes. Each prompt is labelled 'should be blocked' (or redirected to a supportive safe completion) or 'should be answered'. A classifier gives every prompt a risk score; a threshold turns the score into a decision; a confusion matrix counts the four outcomes." },
    { heading: "Two kinds of failure", body: "Under-refusal (FN / risky prompts) lets harmful requests through. Over-refusal (FP / benign prompts) fails real users: someone asking about a normal heart rate, how to support a friend, or how to kill a stuck Python process. Lowering the threshold trades one for the other, so a safety eval must report both, per category, not one blended accuracy." },
    { heading: "Why benign lookalikes matter", body: "Prompts that sound risky but are harmless ('shoot portraits', 'attack the Queen's Gambit') are the only way to measure over-refusal from keyword-style rules. An eval set made only of obvious cases makes a naive keyword filter look perfect." },
    { heading: "Per-category rates", body: "Averages hide who pays. One threshold can be fine for personal-data requests while over-refusing support-seeking messages in the self-harm category, where a cold refusal is itself harmful. Report and set targets per category." },
    { heading: "Assumptions in this visualizer", body: "The classifier is a hand-written keyword rule list with fixed weights passed through a sigmoid, not a learned model; no real model or moderation API is called. The 24 prompts and labels are hand-made; risky prompts are written as abstract [bracketed descriptions] so the page contains no harmful content. Real red-team sets are far larger and need careful, reviewed labels." },
  ],
  code: [
    {
      title: "safety_eval.py — rates per category",
      source: `from collections import defaultdict

def evaluate(dataset, classifier, threshold):
    """dataset: [{'text', 'category', 'should_block'}]"""
    cells = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for ex in dataset:
        blocked = classifier.risk(ex["text"]) >= threshold
        key = ("tp" if blocked else "fn") if ex["should_block"] else ("fp" if blocked else "tn")
        cells[ex["category"]][key] += 1
    report = {}
    for cat, c in cells.items():
        benign, risky = c["fp"] + c["tn"], c["tp"] + c["fn"]
        report[cat] = {
            "over_refusal": c["fp"] / benign if benign else None,
            "under_refusal": c["fn"] / risky if risky else None,
        }
    return report`,
    },
    {
      title: "Pick the threshold from a policy, not from accuracy",
      source: `import numpy as np

def choose_threshold(scores, should_block, max_under=0.05):
    # highest threshold whose under-refusal stays within the policy limit
    for t in np.linspace(1, 0, 101):
        blocked = scores >= t
        under = (~blocked & should_block).sum() / should_block.sum()
        if under <= max_under:
            over = (blocked & ~should_block).sum() / (~should_block).sum()
            return t, under, over
    return 0.0, 0.0, 1.0`,
    },
  ],
  failureModes: [
    { title: "Over-refusal of harmless lookalikes", symptom: "Users asking how to 'kill a process' or 'shoot portraits' get refused; support tickets mention the assistant being preachy.", detect: "Include benign lookalike prompts in the eval set and track over-refusal (FP / benign) per category.", fix: "Add context-aware features or a learned classifier; tune the threshold per category; add lookalikes to the regression set." },
    { title: "Eval set with only obvious cases", symptom: "Safety dashboard shows 0% over-refusal, yet users complain.", detect: "Check the eval set composition: is there any benign prompt that shares vocabulary with risky ones?", fix: "Curate hard negatives (benign lookalikes) and hard positives (paraphrased or indirect risky requests)." },
    { title: "Single blended metric hides a category", symptom: "Overall accuracy is 92% but nearly every support-seeking message in one category is refused.", detect: "Break the confusion matrix down by category and report over- and under-refusal separately.", fix: "Set per-category targets and thresholds; review the worst category before every release." },
    { title: "Threshold tuned for one error only", symptom: "Under-refusal reached 0% after lowering the threshold, but over-refusal jumped.", detect: "Plot both rates against the threshold and look at the chosen point on both curves.", fix: "Write the policy as two limits (e.g. under ≤ 5%, over ≤ 10%) and pick the threshold that satisfies both, or improve the classifier if none does." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, question: "You lower the block threshold from 0.50 to 0.20. What happens to the two error rates?", options: ["Over-refusal rises or stays the same; under-refusal falls or stays the same", "Both fall", "Both rise", "Over-refusal falls; under-refusal rises"], answer: 0, explanation: "A lower threshold blocks more prompts, so more benign prompts are blocked (over-refusal up) and fewer risky ones slip through (under-refusal down). The trade-off chart shows the two curves moving in opposite directions.", skill: "parameter" },
    { id: `${t}-2`, topicId: t, question: "With context-aware rules off, which prompts are most likely to become false positives?", options: ["Benign lookalikes such as 'kill a Python process'", "Risky personal-data requests", "Support-seeking messages that use no risk words", "Prompts with no keywords at all"], answer: 0, explanation: "Without context words like 'python' or 'gambit' pulling the score down, words like 'kill', 'shoot' and 'attack' push harmless lookalikes over the threshold.", skill: "prediction" },
    { id: `${t}-3`, topicId: t, question: "Context rules are off and lookalikes are excluded from the eval set. Measured over-refusal is 0%. What is the right conclusion?", options: ["The eval set cannot see keyword over-refusal; the 0% is an artefact of the set", "The classifier never over-refuses", "The threshold is optimal", "Under-refusal must also be 0%"], answer: 0, explanation: "The last step's grid shows the same keyword filter over-refusing lookalikes once they are in the set. What the eval contains decides what it can detect.", skill: "diagnosis" },
    { id: `${t}-4`, topicId: t, question: "An eval has FP = 1, TN = 14, FN = 1, TP = 8. What is the over-refusal rate?", options: ["1/15 ≈ 6.7%", "1/9 ≈ 11.1%", "1/24 ≈ 4.2%", "2/24 ≈ 8.3%"], answer: 0, explanation: "Over-refusal = FP / (FP + TN) = benign prompts blocked over all benign prompts = 1/15. Under-refusal would be FN / (FN + TP) = 1/9.", skill: "metric" },
    { id: `${t}-5`, topicId: t, question: "Support-seeking messages in the self-harm category are being refused. What is the best fix?", options: ["Add context features (or a better classifier) and track that category's over-refusal separately, answering support requests with supportive resources", "Raise the global threshold to 0.95", "Remove the self-harm category from the eval set", "Report only overall accuracy"], answer: 0, explanation: "Raising the global threshold would let risky prompts through elsewhere, and removing the category hides the problem. Fix the scorer and measure the category directly.", skill: "fix" },
  ],
};
