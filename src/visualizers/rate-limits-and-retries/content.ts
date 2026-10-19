import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Drain a token bucket with many clients and see retry waves vanish with backoff, jitter and Retry-After.", minutes: 12, hasMetrics: true };

const t = "rate-limits-and-retries";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A token-bucket rate limiter holds up to b tokens (the burst) and refills r tokens per second. Each request takes one token; with no token left the server answers HTTP 429 Too Many Requests, often with a Retry-After header saying when to try again. Over time a client can get at most r requests per second through, plus a one-off burst of b." },
    { heading: "Why retries need backoff and jitter", body: "When many clients fail at the same time (after an outage, a deploy, or a burst), a fixed retry delay makes them all come back at the same moment: a thundering herd. The bucket admits b of them and rejects the rest, which retry together again. Exponential backoff alone keeps clients synchronized (they double their delays in lock-step, leaving the bucket idle between waves); full jitter (a random delay between 0 and the exponential cap) breaks the synchronization so retries trickle in as tokens refill." },
    { heading: "Retry-After", body: "The server knows when its next token will be available. A client that waits at least that long never wastes an attempt that is certain to fail. Honouring Retry-After and adding jitter on top is the standard recommendation of most API providers, including LLM APIs." },
    { heading: "What to watch", body: "The 429 rate (wasted work), the shape of attempts over time (spikes = synchronized clients), the worst retry spike, how many clients gave up after max attempts, and when the last client succeeded. The best policy is not always the fastest; it is the one that gets everyone through with the fewest wasted calls." },
    { heading: "Assumptions in this visualizer", body: "A seeded simulation: all clients send their first request within 100 ms, each needs exactly one success, server processing time is ignored, and Retry-After is computed as the time until the next token (rounded up to 10 ms; real APIs often send whole seconds). Delays are capped at 8 s. No real API is called." },
  ],
  code: [
    {
      title: "retry.py — exponential backoff with full jitter, honouring Retry-After",
      source: `import random, time, requests

def call_with_retries(url, max_attempts=6, base=0.5, cap=8.0):
    for attempt in range(1, max_attempts + 1):
        r = requests.post(url, json={"q": "hi"}, timeout=10)
        if r.status_code != 429 and r.status_code < 500:
            return r
        if attempt == max_attempts:
            r.raise_for_status()
        backoff = random.uniform(0, min(cap, base * 2 ** (attempt - 1)))   # full jitter
        retry_after = float(r.headers.get("retry-after", 0))
        time.sleep(max(backoff, retry_after))`,
    },
    {
      title: "tokenBucket.ts — the limiter on the server side",
      source: `export class TokenBucket {
  private tokens: number;
  private last = Date.now();
  constructor(private rate: number, private burst: number) { this.tokens = burst; }

  take(): { ok: true } | { ok: false; retryAfterMs: number } {
    const now = Date.now();
    this.tokens = Math.min(this.burst, this.tokens + (this.rate * (now - this.last)) / 1000);
    this.last = now;
    if (this.tokens >= 1) { this.tokens -= 1; return { ok: true }; }
    return { ok: false, retryAfterMs: Math.ceil(((1 - this.tokens) / this.rate) * 1000) };
  }
}`,
    },
  ],
  failureModes: [
    { title: "Thundering herd from fixed retry delays", symptom: "Load arrives in periodic spikes; most requests in each spike get 429.", detect: "Attempts-per-interval chart shows tall spikes separated by silence; 429 rate above 50%.", fix: "Exponential backoff with full jitter; stagger reconnects after deploys and outages." },
    { title: "Ignoring Retry-After", symptom: "Clients burn their retry budget before the server can accept them, then give up.", detect: "Attempts arrive before the advertised Retry-After; 'gave up' count rises while tokens sit unused later.", fix: "Wait max(backoff, Retry-After); log the header value to verify clients use it." },
    { title: "Retry amplification across layers", symptom: "A brief overload turns into a long outage as each layer retries the layer below.", detect: "Requests per user action multiply (3 retries × 3 layers = up to 27 attempts).", fix: "Retry at one layer only, use retry budgets, and fail fast on 429 at inner layers.", scenarioId: "agent-retry-loop" },
    { title: "Bucket sized for average, not burst", symptom: "Legitimate bursts (page load fan-out) get 429 even though the average rate is low.", detect: "429s cluster at the start of user sessions while the per-minute rate is far below the limit.", fix: "Raise burst b (bucket size) separately from the sustained rate r." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "Bucket r = 5/s, b = 5, 24 clients send at once. You set a fixed 1 s retry delay. About how many get through in the first wave and in each later wave?", options: ["24, then 0", "5, then about 5 per second", "1 per wave", "All 24 after 1 s"], answer: 1, explanation: "The full bucket admits b = 5 at once. One second later it has refilled r × 1 s = 5 tokens, so each synchronized wave admits about 5." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Default setup (24 clients, fixed 500 ms delay, 6 attempts). You switch to 'exponential + full jitter'. What changes?", options: ["Same spikes, just later", "Retries spread into small bars over time, the worst retry spike shrinks, and every client gets through", "Nobody gets a 429 any more", "Everyone gives up"], answer: 1, explanation: "Random delays de-synchronize clients. The bucket admits about r per second either way, but spread-out retries no longer burn all their attempts inside a few rejected waves." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "A rejected attempt found 0.2 tokens with r = 5 tokens/s. What Retry-After does the server send?", options: ["200 ms", "160 ms", "1000 ms", "40 ms"], answer: 1, explanation: "It needs 1 − 0.2 = 0.8 more tokens at 5 per second: 0.8 / 5 = 0.16 s = 160 ms." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "With the '40 clients, fast fixed retries' preset several clients give up even though the server is idle later. Why?", options: ["The bucket is too large", "Their fast synchronized retries all landed while the bucket was empty and used up max attempts", "The server crashed", "Jitter was too large"], answer: 1, explanation: "300 ms fixed retries refill only 1.5 tokens per wave, so most attempts in each wave fail; clients exhaust their attempts in about 1.2 s and stop, leaving capacity unused afterwards." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "With jitter and a 100 ms base delay, many retries arrive before the bucket could possibly have a token. Which change removes exactly those attempts?", options: ["Retry immediately", "Respect the Retry-After header (wait max(backoff, Retry-After))", "Increase max attempts to 100", "Switch to a fixed 100 ms delay"], answer: 1, explanation: "Retry-After tells each client when the next token exists. Waiting at least that long drops 'retries before Retry-After' to 0; jitter on top keeps clients from all returning at that same instant." },
  ],
};
