import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch attempts, timeouts, jittered backoff and a deadline on a timeline, plus duplicate charges.", minutes: 12, hasMetrics: true };

const t = "retries-and-timeouts";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A tool or API call made by an agent can fail transiently (timeouts, 429 rate limits, 5xx errors) or permanently (400 bad request, 404 not found). A retry policy decides how many attempts to make, how long each may take (timeout), how long to wait between them (exponential backoff with jitter), and when to give up altogether (an overall deadline)." },
    { heading: "Why timeouts do not cancel work", body: "A timeout only means the client stopped waiting. The server may still finish the request after the client gave up. If the call has a side effect (charge a card, send an email, create a ticket), retrying it can do the side effect twice. Idempotency keys let the server recognise a repeat and replay the stored result instead." },
    { heading: "Backoff, jitter and deadlines", body: "Exponential backoff (base × 2^(n−1), capped) gives an overloaded service time to recover. Jitter randomises the wait so thousands of clients that failed together do not retry in lock-step. A deadline bounds the total time a caller can be kept waiting, regardless of how many attempts remain." },
    { heading: "Assumptions in this visualizer", body: "The server is a deterministic script per scenario (latency and status per attempt), not a real service; all times are simulated milliseconds. Jitter uses a seeded random generator so runs are reproducible. The Metrics tab uses a simple independent-failure model where a failure costs the full timeout. No network calls are made." },
  ],
  code: [
    {
      title: "Retry with backoff, jitter, deadline and an idempotency key (Python)",
      source: `import random, time, uuid, requests

RETRYABLE = {429, 500, 502, 503, 504}

def charge(amount, max_attempts=4, timeout=2.0, base=0.2, cap=4.0, deadline=8.0):
    key = str(uuid.uuid4())                      # same key on every attempt
    start = time.monotonic()
    for n in range(1, max_attempts + 1):
        remaining = deadline - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("deadline exceeded")
        try:
            r = requests.post("https://pay.example/charges", json={"amount": amount},
                              headers={"Idempotency-Key": key}, timeout=min(timeout, remaining))
            if r.status_code < 400:
                return r.json()
            if r.status_code not in RETRYABLE:
                r.raise_for_status()             # 400/404: permanent, do not retry
        except requests.Timeout:
            pass                                 # transient: the server may still be working!
        if n == max_attempts:
            break
        ceiling = min(cap, base * 2 ** (n - 1))
        time.sleep(random.uniform(0, ceiling))   # full jitter
    raise RuntimeError("all attempts failed")`,
    },
  ],
  failureModes: [
    { title: "Duplicate side effects", symptom: "Customers charged twice; duplicate emails or tickets after a slow period.", detect: "Two server-side writes for one client request; a timeout followed by a successful retry in the trace.", fix: "Send an idempotency key with every mutating call and dedupe server-side; make handlers idempotent." },
    { title: "Retrying permanent errors", symptom: "Agent loops on the same 404/400 until its budget runs out.", detect: "Identical failing calls with the same status repeated; no transient errors in the mix.", fix: "Retry only timeouts, 429 and 5xx; surface permanent errors to the planner.", scenarioId: "agent-retry-loop" },
    { title: "Retry storm (no jitter)", symptom: "Recovering service is knocked over again by synchronized retries.", detect: "Request rate spikes at exact multiples of the backoff interval.", fix: "Add full or equal jitter; cap concurrency; honour Retry-After." },
    { title: "Unbounded latency", symptom: "Users wait tens of seconds for a call that will fail anyway.", detect: "p99 latency ≈ attempts × timeout + Σ backoff.", fix: "Set an overall deadline and propagate the remaining budget to every attempt's timeout." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Scenario 'Charge a card', no idempotency key. Attempt 1 times out at 2000 ms but the server finishes at 2600 ms; attempt 2 succeeds. How many charges end up on the card?", options: ["0", "1", "2", "It depends on jitter"], answer: 2, explanation: "The timeout did not cancel the server's work, so attempt 1 charged the card; attempt 2 is a new request without a key and charges again." },
    { id: `${t}-2`, topicId: t, skill: "metric", question: "Base backoff 200 ms, no jitter, cap 4000 ms. How long is the wait before attempt 4?", options: ["200 ms", "600 ms", "800 ms", "1400 ms"], answer: 2, explanation: "The wait before attempt n+1 is base × 2^(n−1): before attempt 4, n = 3, so 200 × 2² = 800 ms." },
    { id: `${t}-3`, topicId: t, skill: "diagnosis", question: "With 'retry every error' on, the Bad request scenario makes 4 attempts. What is wrong?", options: ["The timeout is too short", "A 400 is permanent; retrying it only wastes time and load", "Jitter is missing", "The deadline is too long"], answer: 1, explanation: "The same invalid request fails the same way every time. Only transient errors (timeouts, 429, 5xx) are worth retrying." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "Cold-start scenario: latencies 2600, 2300, 900 ms. With a 2000 ms timeout and enough attempts it succeeds on attempt 3. What happens if the overall deadline is 3000 ms?", options: ["It succeeds on attempt 2", "It gives up: the deadline runs out before attempt 3 can finish", "It retries forever", "The timeout grows automatically"], answer: 1, explanation: "Attempt 1 uses 2000 ms; after the backoff, attempt 2 is cut off by the deadline. Deadlines trade success probability for bounded latency." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Many clients failed at the same moment and now retry at exactly 200, 400, 800 ms. The service falls over at each retry wave. Which change helps most?", options: ["More attempts", "Shorter timeout", "Add jitter to the backoff", "Remove the deadline"], answer: 2, explanation: "Jitter spreads the retries randomly below each ceiling, turning synchronized waves into a smoother load." },
  ],
};
