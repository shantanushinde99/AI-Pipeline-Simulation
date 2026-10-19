import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Route requests to GPU replicas, form dynamic batches, and watch throughput, p95 and cold starts.", minutes: 14, hasMetrics: true };

const t = "model-serving";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Model serving is the system that turns a trained model into an API: a load balancer spreads requests over replicas (each holding the model on a GPU), each replica queues requests, groups them into batches, runs one forward pass per batch and returns every response." },
    { heading: "Why dynamic batching", body: "A GPU forward pass has a large fixed cost (kernel launches, reading weights from memory) and a small per-request cost. Running 8 requests together costs barely more than running 1, so throughput rises sharply with batch size. The batcher closes a batch when it is full or when the oldest request has waited max-wait milliseconds, which caps the extra latency batching can add." },
    { heading: "Cold starts and readiness", body: "A new replica must download and load the weights before it can serve, which takes seconds to minutes for real models. Autoscaling therefore reacts late, and the load balancer must not route to a replica until its readiness check passes, or requests sit in a queue that nobody is processing." },
    { heading: "What to measure", body: "Throughput (responses per second), latency percentiles (p50, p95: users feel the tail), GPU utilization (fraction of time a batch is running), and average batch size. A high p95 with low GPU utilization usually means a configuration problem (too long a max wait, or traffic going to the wrong place), not a lack of hardware." },
    { heading: "Assumptions in this visualizer", body: "Everything is a seeded simulation: Poisson arrivals over 1.2 simulated seconds, round-robin load balancing, and a made-up GPU cost model of 20 ms + 2.5 ms per request in the batch. Cold start is a fixed 450 ms (real ones are much longer). No real model or GPU is used; requests are not real inferences." },
  ],
  code: [
    {
      title: "dynamic_batcher.py — asyncio batcher with max size and max wait",
      source: `import asyncio, time

class DynamicBatcher:
    def __init__(self, model, max_batch=8, max_wait_ms=10):
        self.model, self.max_batch, self.max_wait = model, max_batch, max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()

    async def infer(self, x):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((x, fut))
        return await fut

    async def run(self):
        while True:
            x, fut = await self.queue.get()            # block for the first request
            batch, deadline = [(x, fut)], time.monotonic() + self.max_wait
            while len(batch) < self.max_batch:          # close on size ...
                timeout = deadline - time.monotonic()
                if timeout <= 0: break                  # ... or on max wait
                try: batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError: break
            outputs = self.model([b[0] for b in batch])  # one forward pass for the whole batch
            for (_, f), y in zip(batch, outputs): f.set_result(y)`,
    },
    {
      title: "Kubernetes readiness probe: keep cold replicas out of rotation",
      source: `readinessProbe:
  httpGet: { path: /ready, port: 8080 }   # returns 200 only after weights are loaded and a warm-up batch ran
  periodSeconds: 2
  failureThreshold: 1
startupProbe:
  httpGet: { path: /ready, port: 8080 }
  failureThreshold: 90                     # allow up to 3 minutes to load a large model`,
    },
  ],
  failureModes: [
    { title: "No batching under load", symptom: "Queues grow and p95 explodes while each GPU is 100% busy at low throughput.", detect: "Average batch size ≈ 1; per-request GPU time equals the full launch cost.", fix: "Enable dynamic batching with a max batch that fits GPU memory and a max wait of a few ms." },
    { title: "Max wait too long for light traffic", symptom: "Latency at low traffic is worse than at high traffic.", detect: "Batches mostly close because of 'max wait', not 'full'; p50 ≈ max wait + inference time.", fix: "Lower max wait or make it adaptive to arrival rate." },
    { title: "Routing to a replica that is still loading", symptom: "A latency spike right after scale-out, on exactly 1/N of requests.", detect: "Requests on the new replica wait for the whole cold start; readiness never checked.", fix: "Readiness probes and warm-up requests before joining the load balancer pool." },
    { title: "Autoscaling too late", symptom: "During a traffic spike latency climbs for the whole cold-start duration before recovering.", detect: "Queue length rises immediately; new capacity only appears after weights load.", fix: "Keep headroom, scale on queue length, pre-warm replicas, or cache weights on local disk." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "parameter", question: "With the default 300 req/s on 2 replicas, what happens if you set max batch size to 1?", options: ["Throughput rises because nothing waits", "Each replica can only do ~44 req/s, so queues grow and p95 explodes", "GPU utilization drops to 0%", "Nothing changes"], answer: 1, explanation: "A batch of 1 costs 22.5 ms, so a replica serves 1000 / 22.5 ≈ 44 req/s. Each replica receives 150 req/s, so it falls far behind (try the 'no batching' preset)." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Traffic is 40 req/s, max batch 16, max wait 50 ms. Why do most batches close for 'max wait'?", options: ["The GPU is too slow", "At low traffic the batch rarely fills before the timer fires", "The load balancer drops requests", "Max batch is too small"], answer: 1, explanation: "Each replica gets ~20 req/s, i.e. one request every 50 ms on average, so 16 requests never accumulate before the 50 ms timer closes the batch." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Using t(b) = 20 + 2.5b ms, what is the per-request GPU cost at b = 8?", options: ["22.5 ms", "5 ms", "40 ms", "2.5 ms"], answer: 1, explanation: "t(8) = 20 + 20 = 40 ms for 8 requests, i.e. 5 ms each, versus 22.5 ms each for b = 1." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "After scale-out, some requests take ~450 ms longer than others. The new replica shows a queue during its cold start. What's wrong?", options: ["Batching is disabled", "The load balancer routes to the replica before it is ready", "The GPU is overloaded", "Max wait is too long"], answer: 1, explanation: "Round-robin sends every Nth request to the loading replica. They can't run until weights are loaded. A readiness check would keep it out of rotation." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "p95 is high and the batch log shows mostly 'GPU was busy' closes with full-ish batches. What helps most?", options: ["Lower max wait", "Add a replica (or a bigger GPU)", "Set max batch to 1", "Disable the load balancer"], answer: 1, explanation: "'GPU was busy' means requests waited for a GPU that was already working, a capacity problem. More replicas add capacity; changing max wait doesn't." },
  ],
};
