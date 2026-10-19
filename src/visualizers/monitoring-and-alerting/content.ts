import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Stream an hour of metrics through alert rules and watch them go ok → pending → firing → resolved.", minutes: 12, hasMetrics: false };

const t = "monitoring-and-alerting";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Monitoring turns a service's behaviour into time series (latency percentiles, error rate, throughput, and for AI systems a sampled quality score). Alerting evaluates rules against those series on a fixed interval. Each rule is a small state machine: ok → pending (condition true, but not for long enough) → firing (a page is sent) → resolved (condition cleared) → ok." },
    { heading: "Three kinds of rule", body: "A static threshold fires the first window a value crosses a line, so one-minute blips page people. A threshold-for-duration ('for 5m') waits until the condition holds for N consecutive windows, trading a few minutes of detection time for far fewer false pages. An SLO burn-rate rule compares the recent error rate with what the SLO allows: burn = (errors / requests) / (1 − SLO). A burn of 1 spends the error budget exactly on schedule; 6× empties it six times too fast." },
    { heading: "Why AI systems need a quality signal", body: "A prompt or model change can make answers worse while every infrastructure metric stays green: no errors, normal latency. Only a quality metric (an LLM judge or heuristic on a sample of traffic, user feedback, task success) catches that. Pick the 'Silent quality regression' scenario and run it with and without the quality rule." },
    { heading: "When it matters", body: "Alert fatigue is a reliability problem: when most pages are noise, people stop trusting and start muting them, and the real incident gets missed. Good alerts are few, symptom-based (what users feel), tied to an SLO, and each one has an action." },
    { heading: "Assumptions in this visualizer", body: "All metrics are simulated from a seeded random generator (seed 7): 1,000 requests per minute for 60 minutes, p95 latency around 420 ms with optional one-minute spikes, a 0.2–0.4% background error rate, and a 'quality score' standing in for a sampled LLM-judge score. Incidents are injected at fixed minutes. The error budget is computed over this 60-minute demo instead of the usual 28–30 days, and the burn-rate rule uses a single 5-minute window instead of the usual pair of long and short windows. Rule evaluation, burn rate and budget arithmetic are real and computed on the numbers shown." },
  ],
  code: [
    {
      title: "alerts.yaml — Prometheus alerting rules",
      source: `groups:
- name: llm-api
  rules:
  - alert: HighLatencyP95
    expr: histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[5m]))) > 0.8
    for: 5m                      # threshold-for-duration: ignore one-minute blips
    labels: { severity: page }
  - alert: ErrorBudgetBurn
    # burn rate = error ratio / (1 - SLO); 99.5% SLO allows 0.005
    expr: |
      (sum(rate(http_requests_total{code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))) / 0.005 > 6
      and
      (sum(rate(http_requests_total{code=~"5.."}[1h])) / sum(rate(http_requests_total[1h]))) / 0.005 > 6
    labels: { severity: page }
  - alert: AnswerQualityDrop
    expr: avg_over_time(llm_judge_score[10m]) < 0.75   # sampled online eval
    for: 3m
    labels: { severity: page }`,
    },
    {
      title: "alert_state.py — the state machine each rule runs",
      source: `def next_state(prev, streak, breach, for_windows):
    if breach:
        streak += 1
        return ("firing" if streak >= max(1, for_windows) else "pending"), streak
    return ("resolved" if prev == "firing" else "ok"), 0

def burn_rate(errors, requests, slo=0.995):
    return (errors / requests) / (1 - slo) if requests else 0.0`,
    },
  ],
  failureModes: [
    { title: "Alert fatigue from static thresholds", symptom: "On-call gets paged several times an hour for blips that recover by themselves.", detect: "Count pages per week and the share that led to an action; most resolve within one window.", fix: "Add a for-duration, alert on SLO burn rate instead of raw thresholds, route non-urgent signals to tickets." },
    { title: "Silent quality regression", symptom: "Users complain about worse answers while latency and error dashboards are green.", detect: "A sampled quality score (LLM judge, user feedback, task success) drops right after a prompt or model deploy.", fix: "Add an online-eval quality metric with its own alert, and annotate dashboards with deploys.", scenarioId: "prod-everything-green" },
    { title: "Threshold set inside the noise band", symptom: "The error-rate alert flaps between firing and resolved all day.", detect: "Plot the baseline distribution; the threshold sits within normal variation.", fix: "Derive thresholds from the SLO (burn rate) or from the baseline's high percentile plus margin." },
    { title: "Alert too slow to matter", symptom: "A long for-duration or long averaging window fires after most of the error budget is gone.", detect: "Compare time-to-detect with how fast the budget drained during past incidents.", fix: "Use multi-window burn-rate alerts: a fast window for big burns, a slow one for steady leaks." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Run 'Healthy, but spiky' with the Noisy preset. Why do pages fire although nothing is wrong?", options: ["The SLO is too strict", "The static latency threshold fires on single-minute spikes that recover immediately", "The quality score is broken", "The error budget is exhausted"], answer: 1, explanation: "With 'must hold for' = 1 minute, a lone GC-pause spike above 800 ms goes straight to firing and resolves the next minute. Setting it to 5 minutes makes the rule ignore one-window blips." },
    { id: `${t}-2`, topicId: t, skill: "diagnosis", question: "In the 'Silent quality regression' scenario with the Noisy preset, what does the on-call engineer see?", options: ["A latency page at minute 24", "An error-rate page at minute 24", "Nothing related to the regression: latency and errors stay normal", "The error budget drains to zero"], answer: 2, explanation: "The prompt change only lowers the quality score. Without a rule on that metric, no infrastructure signal moves, so the incident is invisible." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "In a 5-minute window there are 150 errors out of 5,000 requests. With a 99.5% SLO, what is the burn rate?", options: ["0.03×", "3×", "6×", "30×"], answer: 2, explanation: "Error ratio = 150 / 5000 = 0.03. Allowed = 1 − 0.995 = 0.005. Burn = 0.03 / 0.005 = 6×, exactly at the default threshold." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "In 'Slow dependency', you raise 'Latency: must hold for' from 1 to 5 minutes. What happens to time-to-detect?", options: ["It drops by 4 minutes", "It grows by 4 minutes, but spike-driven pages disappear", "Nothing changes", "The rule never fires"], answer: 1, explanation: "The rule now needs 5 consecutive breaching windows, so it fires 4 minutes later, a deliberate trade for not paging on one-minute spikes." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Which change best reduces noisy error-rate pages without missing the 12-minute outage?", options: ["Raise the static threshold to 10%", "Delete the error rule", "Switch to an SLO burn-rate rule (5-min, 6×)", "Evaluate rules once per hour"], answer: 2, explanation: "Burn rate ties the alert to the error budget: background errors burn about 0.5×, the outage burns 8–12×. 10% would miss the outage, and hourly evaluation delays detection." },
  ],
};
