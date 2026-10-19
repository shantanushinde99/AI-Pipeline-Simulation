# AI Systems Visualizer

Interactive, step-by-step simulations of AI systems — and a Failure Lab for debugging them when they break.
Everything runs in the browser: no backend, no API keys, no model calls. All simulations are deterministic.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
```

| Command | What it does |
| --- | --- |
| `npm run build` | Type-check and build to `dist/` |
| `npm run preview` | Serve the production build |
| `npm test` | Unit + component tests (Vitest, jsdom) |
| `npm run lint` | ESLint (incl. React hooks rules) |
| `npm run typecheck` | `tsc -b` |
| `npm run e2e` | Playwright journeys (run `npx playwright install chromium` once) |

## What's implemented

**Visualizers** (`/visualizer/:topicId`) — each has Visualize, Concept, Code, Failure Modes and Challenges tabs (plus Metrics where there is something to measure):

- `forward-propagation` — 2-3-1 network, real forward pass, editable weights/biases/activation, finite-difference sensitivity.
- `overfitting` — ridge polynomial regression solved exactly; train/test MSE, R², complexity sweep.
- `embeddings` — 2-D teaching vectors, cosine / dot / Euclidean ranking, precision@k against relevance labels.
- `rag-pipeline` — indexing flow (parse → chunk → embed → index) and query flow (embed → retrieve → rerank → context → prompt → generate → cite), with fault injection and golden-set evaluation.
- `agent-loop` — rule-based planner over mock tools with validation, retries, timeouts, iteration budget and guardrails.

**Failure Lab** (`/failure-lab`) — 13 scenarios across RAG, production, agents and ML. Each runs the real simulation with a fault, hides the cause, exposes evidence computed from that run, and re-runs the system with whichever fix you choose.

**Also:** topic registry with planned topics clearly marked, learning paths, interview practice, local progress tracking, light/dark theme.

## Honest simplifications

- The RAG "embedding" is a hashed bag-of-words, the reranker is term overlap, and the generator is a rule-based sentence extractor. None is a neural model; the UI labels them as simulated.
- The agent planner is a deterministic policy over working memory, not an LLM.
- Traces and latencies are simulated but internally consistent; `TraceSpan` mirrors OpenTelemetry spans so real traces can be mapped in later.
- Progress lives in `localStorage` behind `src/lib/storage.ts` — swap its `backend` for an API to persist server-side.

## Layout

```
src/
  app/            routes
  components/     ui primitives, PlaybackControls, TraceInspector, PipelineCanvas (React Flow), LineChart
  data/           topic registry, learning paths
  features/       explore, visualizer page, failure-lab, learning-paths, interview-practice, progress
  hooks/          usePlayback (shared step engine)
  lib/            math, metrics, seeded RNG, storage
  scenarios/      scenario types, RAG/agent/ML scenario definitions, scoring, ML model
  visualizers/    one folder per visualizer: simulation.ts (pure), Visualizer.tsx, content.ts, tests
e2e/              Playwright journeys
```

Adding a visualizer: create `src/visualizers/<name>/` with a pure `simulation.ts`, a default-exported `Visualizer.tsx` and `content.ts`, register it in `src/visualizers/registry.ts`, and mark the topic `implemented` in `src/data/topics.ts` (a test enforces the two stay in sync).
