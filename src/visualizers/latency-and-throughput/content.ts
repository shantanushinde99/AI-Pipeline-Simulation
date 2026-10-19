import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch a queue fill as load nears capacity, then check Little's law and p99 with real numbers.", minutes: 12, hasMetrics: true };

const t = "latency-and-throughput";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Latency is how long one request takes (waiting + being served). Throughput is how many requests finish per second. A system with c workers that each take E[S] seconds per request has a capacity of c / E[S] requests per second; throughput can never exceed it, no matter how much traffic arrives." },
    { heading: "Why utilization drives tail latency", body: "Utilization ρ = λ·E[S]/c is the fraction of time workers are busy. Because arrivals are random, requests sometimes bunch up and wait. The waiting grows roughly like ρ/(1 − ρ): at 50% busy the queue is short, at 95% it is long, and at ρ ≥ 1 it grows for as long as the overload lasts. The mean hides this; p95 and p99 show it first." },
    { heading: "Little's law", body: "L = λW: the average number of requests in the system equals the arrival rate times the average time each spends inside. It holds for any arrival pattern, service distribution or queue order, which makes it a sanity check for dashboards (e.g. 200 concurrent requests at 50 req/s means each takes 4 s on average)." },
    { heading: "Bursts and heavy tails", body: "Two systems with the same average load can behave very differently. A burst briefly pushes λ above capacity and leaves a backlog that takes seconds to drain. A heavy-tailed service time (a few very slow requests) blocks workers and makes everyone behind them wait (head-of-line blocking)." },
    { heading: "Assumptions in this visualizer", body: "This is a seeded discrete-event simulation, not a real server: Poisson arrivals over 8 simulated seconds, one FIFO queue, c identical workers (M/G/c), no timeouts or retries, and the queue drains after arrivals stop. The service time distributions are stand-ins. Every number on screen is computed from the simulated requests you see." },
  ],
  code: [
    {
      title: "queue_sim.py — the same M/G/c FIFO simulation",
      source: `import heapq, random

def simulate(lam, mean_service, workers, horizon=8.0, seed=7):
    rnd = random.Random(seed)
    free = [0.0] * workers            # time each worker becomes free
    t, latencies = 0.0, []
    while True:
        t += rnd.expovariate(lam)      # Poisson arrivals
        if t >= horizon: break
        s = rnd.expovariate(1 / mean_service)
        w = min(range(workers), key=free.__getitem__)
        start = max(t, free[w])        # wait if every worker is busy
        free[w] = start + s
        latencies.append(free[w] - t)  # wait + service
    latencies.sort()
    p = lambda q: latencies[int(q * (len(latencies) - 1))]
    return p(0.5), p(0.95), p(0.99)

print(simulate(lam=11.5, mean_service=0.25, workers=3))  # ρ ≈ 0.96`,
    },
    {
      title: "Little's law as a dashboard sanity check (TypeScript)",
      source: `// in-flight requests should equal arrival rate × mean latency
function littleCheck(inFlight: number, reqPerSec: number, meanLatencySec: number) {
  const expected = reqPerSec * meanLatencySec;
  const drift = Math.abs(inFlight - expected) / Math.max(expected, 1e-9);
  return { expected, ok: drift < 0.2 };   // large drift: a metric is wrong or requests are leaking
}`,
    },
  ],
  failureModes: [
    { title: "Running hot to save money", symptom: "Average latency looks fine but p99 is many times the service time; small traffic spikes cause timeouts.", detect: "Utilization above ~80% together with a long right tail in the latency histogram.", fix: "Keep headroom (target ρ ≈ 0.6–0.7 for latency-sensitive paths), autoscale on queue length, not CPU." },
    { title: "Burst backlog", symptom: "Latency stays high for seconds after a traffic spike has passed.", detect: "Queue length keeps falling slowly after the burst window; throughput pinned at capacity meanwhile.", fix: "Admission control or load shedding, rate limits per client, pre-scaling for known peaks." },
    { title: "Head-of-line blocking from slow requests", symptom: "Fast requests are slow too, only sometimes.", detect: "Switch to the heavy-tail distribution: p99 jumps while the mean service time is unchanged.", fix: "Separate queues or worker pools for slow request classes, timeouts, or shortest-job-first scheduling." },
    { title: "Monitoring only the mean", symptom: "Users complain while the average-latency dashboard is green.", detect: "Compare p50 with p99; the gap is the experience of your unluckiest users.", fix: "Alert on p95/p99 and queue length; use histograms rather than averages.", scenarioId: "prod-everything-green" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "With 3 workers and E[S] = 250 ms, what is the capacity, and what happens to throughput at λ = 15 req/s?", options: ["12 req/s; throughput stays near 12 while the queue grows", "15 req/s; throughput reaches 15", "4 req/s; throughput drops to 4", "36 req/s; nothing queues"], answer: 0, explanation: "Capacity = c / E[S] = 3 / 0.25 = 12 req/s. Above it, completions per second are capped at about 12 and the excess waits in the queue (see the Metrics tab's flat throughput line)." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "You move λ from 6 to 11.5 req/s (ρ ≈ 0.5 → 0.96). Which changes most?", options: ["The service time", "p99 latency", "The capacity", "Throughput per worker beyond 4 req/s"], answer: 1, explanation: "Utilization near 1 lets random bunching build long queues. Service time and capacity are unchanged; tail latency explodes." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "A run shows λ = 10 req/s and mean latency W = 0.4 s. What is the average number of requests in the system?", options: ["0.04", "4", "25", "10.4"], answer: 1, explanation: "Little's law: L = λW = 10 × 0.4 = 4. The summary step computes L independently from N(t) and matches λW." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "With the burst preset, latency is still high at 4–5 s, after the burst ended at 3.5 s. Why?", options: ["The workers slowed down", "The backlog built during the burst still has to drain at capacity", "Little's law stopped holding", "The random seed changed"], answer: 1, explanation: "During the burst arrivals exceeded capacity, so a queue built up. After it ends, workers clear it at most at capacity minus the normal arrival rate, which takes time." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "p99 is too high at ρ ≈ 0.95. Which change helps most without reducing traffic?", options: ["Add a worker (lower ρ)", "Report the mean instead of p99", "Increase the queue length limit", "Use a constant service time distribution label"], answer: 0, explanation: "Adding a worker raises capacity and lowers utilization, which shortens queues dramatically. Changing what you report doesn't change user latency." },
  ],
};
