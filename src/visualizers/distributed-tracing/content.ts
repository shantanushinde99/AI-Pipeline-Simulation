import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Follow one RAG request across six services as its trace waterfall grows span by span.", minutes: 13, hasMetrics: false };

const t = "distributed-tracing";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A trace records one request's journey through every service it touches. Each unit of work is a span: a name, a start time, a duration, a status, attributes and events, plus the id of its parent span. All spans of one request share a trace id, so a backend can reassemble them into a tree and draw the waterfall." },
    { heading: "Context propagation", body: "Services are separate processes, so the trace id must travel with the request. The W3C Trace Context standard puts it in a traceparent header: version-traceid-parentid-flags, e.g. 00-4bf92f…-00f067aa0ba902b7-01. On every hop the caller writes its own span id as the parent id; the callee reads the header and starts a child span. Drop the header anywhere and the trace splits in two." },
    { heading: "Reading a waterfall", body: "Parallel calls (a fan-out to vector-DB shards) overlap; the parent waits for the slowest child. The critical path is the chain of spans that sets end-to-end latency; only shortening spans on it makes the request faster. Retries and timeouts show up as extra spans, which is often the first time anyone notices them." },
    { heading: "Sampling", body: "Recording every span of every request is expensive, so systems sample. Head sampling decides at the first service (here: a hash of the trace id below the rate) and propagates the decision in the flags byte, so every service agrees. It is cheap but blind: it cannot know which requests will turn out slow or failed. Tail sampling records everything, buffers complete traces in a collector, and keeps the interesting ones (errors, slow traces) plus a random share." },
    { heading: "Assumptions in this visualizer", body: "The services, their timings and the request are simulated: span durations are hand-set baselines with seeded jitter, the slow shard is a fixed 820 ms ± 40 ms, and the tool timeout is 300 ms. Trace and span ids are random hex from a seeded generator. The hash used for head sampling is the last 8 hex digits of the trace id, scaled to [0, 1). No real tracing SDK or network call is involved; the span objects follow OpenTelemetry's shape so the detail view works like a real trace viewer." },
  ],
  code: [
    {
      title: "tracing.py — OpenTelemetry spans + header propagation",
      source: `from opentelemetry import trace
from opentelemetry.propagate import inject, extract
import httpx, asyncio

tracer = trace.get_tracer("rag-gateway")

async def ask(question: str):
    with tracer.start_as_current_span("POST /ask") as root:
        headers = {}
        inject(headers)            # adds traceparent: 00-<trace>-<root span>-01
        chunks = (await httpx.AsyncClient().post(RETRIEVER, json={"q": question}, headers=headers)).json()
        with tracer.start_as_current_span("rerank"):
            top = rerank(question, chunks)
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("gen_ai.usage.input_tokens", count_tokens(top))
            return await llm(question, top)

# in the retriever service
async def retrieve(request):
    ctx = extract(request.headers)   # continue the caller's trace
    with tracer.start_as_current_span("retrieve", context=ctx):
        qv = embed(request.json["q"])
        # fan-out: three child spans that overlap in time
        return await asyncio.gather(*(search_shard(i, qv) for i in range(3)))`,
    },
    {
      title: "collector.yaml — tail sampling in the OpenTelemetry Collector",
      source: `processors:
  tail_sampling:
    decision_wait: 10s
    policies:
      - { name: errors, type: status_code, status_code: { status_codes: [ERROR] } }
      - { name: slow,   type: latency,     latency: { threshold_ms: 1800 } }
      - { name: base,   type: probabilistic, probabilistic: { sampling_percentage: 20 } }`,
    },
  ],
  failureModes: [
    { title: "Broken context propagation", symptom: "Each service shows its own one-span trace; you cannot see the request end to end.", detect: "Spans from downstream services have no parent or a different trace id; the traceparent header is missing on some hop (often a queue, a thread pool or a custom HTTP client).", fix: "Use instrumented clients, inject/extract the context explicitly across queues and async boundaries." },
    { title: "Head sampling drops the traces you need", symptom: "The p99 latency alert fires but no slow trace is available to inspect.", detect: "Compare the share of slow or failed traces in storage with their share in metrics.", fix: "Use tail sampling (keep errors and slow traces) or raise the rate for specific routes." },
    { title: "Optimizing off the critical path", symptom: "A team makes a service 3× faster and end-to-end latency does not move.", detect: "The span was overlapped by a slower sibling (parallel fan-out) and was never on the critical path.", fix: "Compute the critical path from the waterfall first, then optimize the biggest self time on it." },
    { title: "Hidden retries", symptom: "Occasional requests are 300–500 ms slower with no error returned to users.", detect: "A failed span followed by a retry span of the same operation in the waterfall.", fix: "Lower the timeout, add hedged requests, or fix the flaky dependency; alert on retry rate." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "With the 'Slow vector-DB shard' preset, which span is the bottleneck?", options: ["llm.generate #1", "ann_search shard-2", "rerank", "POST /ask"], answer: 1, explanation: "Shard-2 takes about 820 ms and the retriever must wait for the slowest shard, so it has the largest self time on the critical path." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "In the healthy preset, shards take about 48 ms each. What happens to the retrieve span when you turn off parallel fan-out?", options: ["Nothing", "It grows by roughly the sum of the two other shard times", "It shrinks", "The trace id changes"], answer: 1, explanation: "Sequential calls add up: max(a, b, c) becomes a + b + c, so retrieve grows by about two shard durations plus gaps." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "In traceparent 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01, what does 00f067aa0ba902b7 identify?", options: ["The trace", "The span of the caller that sent the request (the parent)", "The service name", "The sampling rate"], answer: 1, explanation: "The third field is the parent-id: the caller's span id. The callee creates a new span whose parent is this id. The trace id (second field) stays the same on every hop." },
    { id: `${t}-4`, topicId: t, skill: "parameter", question: "With head sampling at 20% and tail sampling off, roughly what share of the slow traces end up stored?", options: ["All of them", "About 20%", "None", "About 80%"], answer: 1, explanation: "Head sampling decides before the request runs, independent of how slow it will be, so it keeps about the same 20% of slow traces as of fast ones. Turn on tail sampling and every slow or failed trace is kept." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Downstream services' spans appear as separate traces. What is the most likely fix?", options: ["Increase the sampling rate", "Propagate the traceparent header on every outgoing call (inject) and read it on every incoming one (extract)", "Use a faster tracing backend", "Add more span attributes"], answer: 1, explanation: "Separate traces mean the context did not cross the process boundary. Instrumented clients or explicit inject/extract restore the parent-child links." },
  ],
};
