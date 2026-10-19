import type { CategoryId, Difficulty, Topic } from "../types/topic";
import { VISUALIZERS } from "../visualizers/registry";

export const CATEGORIES: { id: CategoryId; title: string; short: string }[] = [
  { id: "ml", title: "Machine Learning Fundamentals", short: "ML" },
  { id: "dl", title: "Deep Learning & Transformers", short: "Deep Learning" },
  { id: "rag", title: "RAG Systems", short: "RAG" },
  { id: "agents", title: "AI Agents", short: "Agents" },
  { id: "prod", title: "AI Production Engineering", short: "Production" },
  { id: "eval", title: "AI Evaluation & Reliability", short: "Evaluation" },
];

const IMPLEMENTED: Topic[] = [
  {
    id: "forward-propagation",
    title: "Neural network forward propagation",
    category: "ml",
    difficulty: "Beginner",
    minutes: 12,
    status: "implemented",
    summary: "Follow real numbers through weights, biases and activations in a 2-3-1 network.",
  },
  {
    id: "overfitting",
    title: "Overfitting & model evaluation",
    category: "ml",
    difficulty: "Intermediate",
    minutes: 15,
    status: "implemented",
    summary: "Fit polynomials of growing complexity and watch training and test error diverge.",
  },
  {
    id: "embeddings",
    title: "Embeddings & vector search",
    category: "dl",
    difficulty: "Beginner",
    minutes: 12,
    status: "implemented",
    summary: "Place a query in a 2-D teaching embedding space and rank documents by similarity.",
  },
  {
    id: "rag-pipeline",
    title: "Complete RAG pipeline",
    category: "rag",
    difficulty: "Intermediate",
    minutes: 20,
    status: "implemented",
    summary: "Index a small corpus, retrieve, rerank, assemble context and generate a cited answer.",
  },
  {
    id: "agent-loop",
    title: "AI agent execution loop",
    category: "agents",
    difficulty: "Intermediate",
    minutes: 15,
    status: "implemented",
    summary: "Step through decide → call tool → observe, including bad arguments, timeouts and budgets.",
  },
];

type P = [title: string, difficulty: Difficulty, coveredBy?: string];
const PLANNED: Record<CategoryId, P[]> = {
  ml: [
    ["Supervised vs. unsupervised learning", "Beginner"],
    ["Linear regression", "Beginner", "overfitting"],
    ["Logistic regression", "Beginner"],
    ["Decision trees", "Beginner"],
    ["Backpropagation", "Intermediate"],
    ["Gradient descent", "Intermediate"],
    ["Loss functions", "Beginner", "overfitting"],
    ["Bias and variance", "Intermediate", "overfitting"],
    ["Feature engineering", "Intermediate"],
    ["Train, validation, and test splits", "Beginner", "overfitting"],
    ["Data leakage", "Intermediate"],
    ["Classification metrics", "Beginner"],
    ["Regression metrics", "Beginner", "overfitting"],
    ["Confusion matrices", "Beginner"],
    ["Model drift", "Advanced"],
  ],
  dl: [
    ["Neurons and activation functions", "Beginner", "forward-propagation"],
    ["Tokenization", "Beginner"],
    ["Self-attention", "Advanced"],
    ["Query, key, and value", "Advanced"],
    ["Positional information", "Advanced"],
    ["Transformer blocks", "Advanced"],
    ["Next-token prediction", "Intermediate"],
    ["Temperature and sampling", "Intermediate"],
    ["Context windows", "Intermediate", "rag-pipeline"],
    ["Fine-tuning fundamentals", "Advanced"],
  ],
  rag: [
    ["Document ingestion", "Beginner", "rag-pipeline"],
    ["Parsing and extraction", "Intermediate", "rag-pipeline"],
    ["Chunking strategies", "Intermediate", "rag-pipeline"],
    ["Vector databases", "Intermediate", "embeddings"],
    ["Similarity search", "Beginner", "embeddings"],
    ["Top-k retrieval", "Beginner", "rag-pipeline"],
    ["Hybrid retrieval", "Advanced"],
    ["Reranking", "Intermediate", "rag-pipeline"],
    ["Context assembly", "Intermediate", "rag-pipeline"],
    ["Prompt construction", "Beginner", "rag-pipeline"],
    ["Grounded generation", "Intermediate", "rag-pipeline"],
    ["Citation mapping", "Intermediate", "rag-pipeline"],
    ["RAG evaluation", "Advanced"],
    ["Retrieval failures", "Advanced"],
  ],
  agents: [
    ["Planning and execution", "Intermediate", "agent-loop"],
    ["Tool selection", "Beginner", "agent-loop"],
    ["Tool calling", "Beginner", "agent-loop"],
    ["Tool argument validation", "Intermediate", "agent-loop"],
    ["Short-term and long-term memory", "Intermediate"],
    ["Multi-step workflows", "Intermediate", "agent-loop"],
    ["Retries and timeouts", "Intermediate", "agent-loop"],
    ["Agent termination", "Intermediate", "agent-loop"],
    ["Multi-agent coordination", "Advanced"],
    ["Guardrails and permissions", "Advanced", "agent-loop"],
  ],
  prod: [
    ["Model serving", "Intermediate"],
    ["Training vs. inference", "Beginner"],
    ["Feature and training-serving skew", "Advanced"],
    ["Model versioning", "Intermediate"],
    ["Experiment tracking", "Intermediate"],
    ["CI/CD for ML", "Advanced"],
    ["Monitoring and alerting", "Intermediate"],
    ["Data drift and concept drift", "Advanced"],
    ["Distributed tracing", "Advanced"],
    ["Latency and throughput", "Intermediate"],
    ["Caching", "Intermediate"],
    ["Rate limits and retries", "Intermediate"],
    ["Rollbacks", "Intermediate"],
    ["Fallback strategies", "Advanced"],
    ["Cost optimization", "Advanced"],
  ],
  eval: [
    ["Offline evaluation", "Intermediate"],
    ["Online evaluation", "Advanced"],
    ["Golden datasets", "Intermediate"],
    ["Regression testing", "Intermediate"],
    ["Hallucination analysis", "Advanced"],
    ["Faithfulness and relevance", "Advanced", "rag-pipeline"],
    ["Prompt injection", "Advanced"],
    ["Structured output validation", "Intermediate", "agent-loop"],
    ["Safety evaluations", "Advanced"],
    ["Silent quality degradation", "Advanced"],
    ["Root-cause analysis", "Advanced"],
  ],
};

const slug = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");

/** A topic is implemented exactly when a visualizer folder exists for it; its card text then comes from that visualizer's meta. */
const withStatus = (t: Topic): Topic => {
  const v = VISUALIZERS[t.id];
  return v ? { ...t, status: "implemented", summary: v.meta.summary, minutes: v.meta.minutes } : { ...t, status: "planned" };
};

export const TOPICS: Topic[] = [
  ...IMPLEMENTED,
  ...Object.entries(PLANNED).flatMap(([category, list]) =>
    list.map(([title, difficulty, coveredBy]) => ({
      id: slug(title),
      title,
      category: category as CategoryId,
      difficulty,
      minutes: 10,
      status: "planned" as const,
      summary: coveredBy ? "Not a standalone visualizer yet; its core idea appears in a working one." : "Planned visualizer.",
      coveredBy,
    })),
  ),
].map(withStatus);

export const getTopic = (id: string) => TOPICS.find((t) => t.id === id);
export const categoryOf = (id: CategoryId) => CATEGORIES.find((c) => c.id === id)!;

export function searchTopics(query: string, category: CategoryId | "all" = "all") {
  const q = query.trim().toLowerCase();
  return TOPICS.filter(
    (t) =>
      (category === "all" || t.category === category) &&
      (!q || t.title.toLowerCase().includes(q) || t.summary.toLowerCase().includes(q)),
  );
}
