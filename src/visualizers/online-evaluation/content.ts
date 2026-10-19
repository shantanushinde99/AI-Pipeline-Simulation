import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Run an A/B test day by day, watch the CI narrow, and see how peeking and guardrails change the call.", minutes: 15, hasMetrics: false };

const t = "online-evaluation";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Online evaluation measures a change on real users. In an A/B test, each user is randomly assigned to the current system (A, control) or the new one (B). Randomization makes the arms comparable, so a difference in the outcome metric is caused by the change, not by who happened to see it." },
    { heading: "Confidence intervals and p-values", body: "Each day adds users, and the estimate of B − A gets more precise: the 95% confidence interval narrows roughly like 1/√n. A two-proportion z-test turns the observed difference into a p-value: how surprising this difference would be if A and B were truly identical. Decide the sample size up front with a power calculation for the smallest effect you care about (the MDE)." },
    { heading: "Peeking", body: "A p-value is only valid at the sample size you fixed in advance. Checking every day and stopping at the first p < 0.05 gives noise many chances to cross the line, so the false-positive rate of 'winners' climbs well above 5%. Use a fixed horizon, or a sequential method designed for repeated looks (alpha spending, always-valid p-values)." },
    { heading: "Guardrail metrics", body: "A launch must not damage things you are not optimizing: latency, error rate, complaints, cost. Guardrails are tested as non-inferiority: the worst plausible degradation (upper confidence bound) must stay inside a pre-agreed margin. A guardrail failure blocks the launch even when the primary metric wins." },
    { heading: "Assumptions in this visualizer", body: "Users, conversions and latencies are simulated with a seeded random generator: each user is assigned A or B with probability 0.5 and converts with their arm's true rate; daily p95 latency is 820 ms plus Gaussian noise (σ = 15 ms) plus B's added latency. The 'true lift' is known only to the simulation, which is why we can count false positives. The peeking panel runs 200 A/A experiments with exactly equal arms. The z-test, CIs, power formula and guardrail bound are real computations." },
  ],
  code: [
    {
      title: "ab_test.py — statsmodels",
      source: `from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# 1) Plan: users per arm to detect +1.5 pts on a 10% base, alpha=0.05, power=0.8
es = proportion_effectsize(0.115, 0.10)
n_per_arm = NormalIndPower().solve_power(es, alpha=0.05, power=0.8)

# 2) Analyse ONCE, at the planned horizon
z, p = proportions_ztest([conv_b, conv_a], [n_b, n_a])
lo, hi = confint_proportions_2indep(conv_b, n_b, conv_a, n_a, compare="diff")

guardrail_ok = p95_increase_upper_bound <= 100  # ms, non-inferiority margin
ship = p < 0.05 and (conv_b / n_b > conv_a / n_a) and guardrail_ok`,
    },
    {
      title: "Deterministic assignment (TypeScript)",
      source: `import { createHash } from "node:crypto";

// Same user → same arm on every request, independent across experiments.
export function assign(userId: string, experiment: string): "A" | "B" {
  const h = createHash("sha256").update(\`\${experiment}:\${userId}\`).digest();
  return h.readUInt32BE(0) / 2 ** 32 < 0.5 ? "A" : "B";
}`,
    },
  ],
  failureModes: [
    { title: "Peeking and stopping early", symptom: "Many launched 'wins' fail to replicate; the effect shrinks after launch.", detect: "Compare the stop day to the planned horizon; run A/A tests and count how many 'win'.", fix: "Pre-register the horizon, or use a sequential test with alpha spending / always-valid p-values." },
    { title: "Underpowered test", symptom: "Results are 'not significant' even when the change is good; CIs are wide.", detect: "Power calculation: planned users per arm below the required n for the MDE.", fix: "Run longer, increase traffic share, use variance reduction (e.g. CUPED), or accept a larger MDE." },
    { title: "Guardrail ignored", symptom: "Conversion is up, but latency or complaints quietly rise and hurt retention later.", detect: "Track guardrails with their own CIs; test the upper bound against a margin.", fix: "Make guardrail failures launch-blocking in the decision rule." },
    { title: "Sample ratio mismatch", symptom: "One arm gets noticeably more users than 50%, and results look odd.", detect: "Chi-square test on arm counts; a tiny p-value means assignment or logging is broken.", fix: "Fix the bucketing or logging bug before trusting any metric." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Load the 'A/A test + peeking' preset. Why does the test stop on an early day with p < 0.05 even though A and B are identical?", options: ["The simulation has a bug", "Random noise crossed 0.05 at one of the daily looks, and peeking stops at the first crossing", "B is actually better", "The guardrail forced a stop"], answer: 1, explanation: "With no real effect, p-values wander. Each look is another chance to dip below 0.05. The ghost points in the decision step show the full-horizon p-value is not significant." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "In the last step, what does the fixed-horizon false-positive rate stay near, and what happens with daily peeking?", options: ["0% and 5%", "About 5%, while peeking multiplies it several times", "50% for both", "Both equal the true lift"], answer: 1, explanation: "α = 0.05 controls false positives for one pre-planned look. Peeking every day over two weeks pushes the rate to roughly 20–30% in these simulations." },
    { id: `${t}-3`, topicId: t, skill: "parameter", question: "With the default 'Real +1.5 pt lift', raise B's added latency to 180 ms. What is the decision, and why?", options: ["Ship: conversion is still significantly up", "Blocked: the upper bound of the p95 increase exceeds the 100 ms guardrail", "No ship: p-value rises", "Ship after one more day"], answer: 1, explanation: "Latency doesn't change conversions in this simulation, so the primary metric still wins, but the guardrail rule blocks the launch." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "You halve users per day (1000 → 500) without changing the planned 14 days. What does the setup step warn about?", options: ["Nothing changes", "The test is underpowered: planned users per arm fall below the required n", "The p-value becomes invalid", "The guardrail is skipped"], answer: 1, explanation: "Required n per arm depends on the base rate and MDE, not on traffic. Fewer users per day means the plan no longer reaches it, so a real lift will often look insignificant." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "How does the CI width on day 14 compare to day 1 with constant daily traffic?", options: ["The same", "About 1/√14 as wide (≈ 27%)", "1/14 as wide", "Wider"], answer: 1, explanation: "Standard error scales as 1/√n. Fourteen times the users gives an interval about 1/√14 ≈ 0.27 as wide; the day table in the step card shows it." },
  ],
};
