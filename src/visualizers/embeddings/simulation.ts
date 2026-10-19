import { cosine, dot, euclidean, norm } from "../../lib/math";
import type { SimulationStep } from "../../types/simulation";

export type Vec2 = [number, number];
export type Doc = { id: string; title: string; text: string; v: Vec2 };
export type Metric = "cosine" | "dot" | "euclidean";

/**
 * Hand-placed 2-D teaching vectors for a support knowledge base. Direction encodes topic
 * (billing ↗, account security ↑, playback ←, family/content ↓); length loosely encodes how much text the doc has.
 */
export const DOCS: Doc[] = [
  { id: "d1", title: "Start a subscription", text: "Choose a plan and start your subscription in minutes. Monthly and annual billing available.", v: [3.0, 2.2] },
  { id: "d2", title: "Ending your membership", text: "You can end your membership any time from Account → Plan. Access continues until the period ends.", v: [2.0, 2.4] },
  { id: "d3", title: "Update payment method", text: "Change the card or PayPal account used for billing.", v: [3.2, 1.0] },
  { id: "d4", title: "Refund policy", text: "Refunds are available within 14 days of a charge if nothing was watched.", v: [2.4, 0.9] },
  { id: "d5", title: "Reset your password", text: "Use 'Forgot password' on the sign-in page to receive a reset link.", v: [-0.4, 3.0] },
  { id: "d6", title: "Enable two-factor authentication", text: "Protect your account with a one-time code from an authenticator app.", v: [-1.2, 2.8] },
  { id: "d7", title: "Install the desktop app", text: "Download the app for Windows or macOS from the downloads page.", v: [-2.8, 1.0] },
  { id: "d8", title: "Fix playback buffering", text: "If video pauses or stutters, lower the quality setting or restart your router.", v: [-3.0, -0.5] },
  { id: "d9", title: "Supported devices", text: "Smart TVs, consoles and browsers that can stream in HD and 4K.", v: [-2.2, -1.4] },
  { id: "d10", title: "Download for offline viewing", text: "Save episodes to your phone and watch without a connection.", v: [-0.8, -2.8] },
  { id: "d11", title: "Create a kids profile", text: "Kids profiles only show age-appropriate titles.", v: [1.0, -2.6] },
  { id: "d12", title: "Parental controls", text: "Set maturity ratings and lock profiles with a PIN.", v: [1.8, -2.0] },
  { id: "d13", title: "Complete account FAQ", text: "A long page answering dozens of general account, plan and billing questions.", v: [4.8, 1.6] },
];

export type PresetQuery = { id: string; text: string; v: Vec2; relevant: string[]; note: string };
export const PRESET_QUERIES: PresetQuery[] = [
  { id: "cancel", text: "How do I cancel my subscription?", v: [2.9, 2.3], relevant: ["d2"], note: "The nearest vector is about starting a subscription: same topic, opposite intent. The answer ('Ending your membership') uses different words." },
  { id: "buffer", text: "My video keeps stopping", v: [-2.6, -0.6], relevant: ["d8", "d9"], note: "A well-behaved query: the closest vectors are also the most useful documents." },
  { id: "kids", text: "Keep my children safe while watching", v: [1.4, -2.4], relevant: ["d11", "d12"], note: "Two relevant documents sit close together; top-k = 2 retrieves both." },
  { id: "refund", text: "Can I get my money back?", v: [2.6, 1.1], relevant: ["d4"], note: "Try dot product: the long, generic FAQ jumps to the top because dot product rewards vector length." },
];

/** Tiny lexicon so free-text queries land somewhere sensible. Query vector = mean of matched word vectors × 3. */
const LEXICON: Record<string, Vec2> = {
  cancel: [0.7, 0.75], end: [0.6, 0.8], stop: [-0.4, 0.2], subscription: [0.8, 0.6], membership: [0.65, 0.75], plan: [0.85, 0.5],
  pay: [0.95, 0.3], payment: [0.95, 0.3], card: [0.95, 0.25], billing: [0.9, 0.4], charge: [0.9, 0.35], refund: [0.93, 0.36], money: [0.92, 0.38], price: [0.9, 0.45],
  password: [-0.12, 0.99], login: [-0.2, 0.97], sign: [-0.2, 0.95], account: [0.4, 0.9], security: [-0.4, 0.9], hacked: [-0.35, 0.93], code: [-0.4, 0.9],
  install: [-0.94, 0.33], app: [-0.9, 0.35], download: [-0.6, -0.6], desktop: [-0.95, 0.3], windows: [-0.95, 0.3], mac: [-0.95, 0.3],
  video: [-0.95, -0.2], playback: [-0.98, -0.15], buffering: [-0.98, -0.16], slow: [-0.9, -0.3], quality: [-0.9, -0.25], tv: [-0.85, -0.5], device: [-0.85, -0.55], stream: [-0.9, -0.4],
  offline: [-0.3, -0.95], episodes: [-0.3, -0.95], phone: [-0.4, -0.9],
  kids: [0.36, -0.93], children: [0.36, -0.93], child: [0.36, -0.93], parental: [0.67, -0.74], safe: [0.6, -0.8], pin: [0.6, -0.8], profile: [0.4, -0.9],
};

export function embedText(text: string): { v: Vec2 | null; matched: string[] } {
  const words = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const matched = words.map((w) => (LEXICON[w] ? w : LEXICON[w.replace(/s$/, "")] ? w.replace(/s$/, "") : null)).filter((w): w is string => !!w);
  if (!matched.length) return { v: null, matched };
  const sum = matched.reduce<Vec2>((s, w) => [s[0] + LEXICON[w][0], s[1] + LEXICON[w][1]], [0, 0]);
  const n = Math.hypot(...sum);
  return { v: n === 0 ? null : [(sum[0] / n) * 3, (sum[1] / n) * 3], matched };
}

/** Higher is more similar for every metric (Euclidean distance is negated). */
export function score(metric: Metric, q: Vec2, d: Vec2) {
  return metric === "cosine" ? cosine(q, d) : metric === "dot" ? dot(q, d) : -euclidean(q, d);
}

/** `tex` is LaTeX. */
export const METRIC_INFO: Record<Metric, { name: string; tex: string }> = {
  cosine: { name: "Cosine similarity", tex: "\\cos(q, d) = \\frac{q \\cdot d}{\\lVert q \\rVert\\, \\lVert d \\rVert}" },
  dot: { name: "Dot product", tex: "q \\cdot d = q_1 d_1 + q_2 d_2" },
  euclidean: { name: "Negative Euclidean distance", tex: "-\\lVert q - d \\rVert = -\\sqrt{(q_1 - d_1)^2 + (q_2 - d_2)^2}" },
};

export type Scored = { doc: Doc; score: number; rank: number };
export type SearchState = { phase: "embed" | "score" | "rank" | "topk" | "judge"; scoredIds: string[]; current?: string };

export function rank(q: Vec2, metric: Metric, docs = DOCS): Scored[] {
  return docs
    .map((doc) => ({ doc, score: score(metric, q, doc.v), rank: 0 }))
    .sort((a, b) => b.score - a.score || a.doc.id.localeCompare(b.doc.id, undefined, { numeric: true }))
    .map((s, i) => ({ ...s, rank: i + 1 }));
}

export function precisionAtK(ranked: Scored[], k: number, relevant: string[]) {
  const top = ranked.slice(0, k);
  return top.filter((s) => relevant.includes(s.doc.id)).length / Math.max(1, top.length);
}

export function simulate(q: Vec2, metric: Metric, k: number, relevant?: string[]) {
  const ranked = rank(q, metric);
  const steps: SimulationStep<SearchState>[] = [];
  const scored: string[] = [];
  steps.push({ id: "embed", title: "Embed the query", description: `The query becomes the vector q = [${q.map((v) => v.toFixed(2)).join(", ")}] with length ‖q‖ = ${norm(q).toFixed(2)}.`, state: { phase: "embed", scoredIds: [] }, events: [] });
  for (const d of DOCS) {
    scored.push(d.id);
    const s = score(metric, q, d.v);
    steps.push({
      id: `score-${d.id}`, title: `Score "${d.title}"`, description: `${METRIC_INFO[metric].name} between q and ${d.id} = ${s.toFixed(3)}.`,
      state: { phase: "score", scoredIds: [...scored], current: d.id },
      events: [{
        id: d.id, type: "score", message: `${metric}(q, ${d.id}) = ${s.toFixed(4)}`,
        tex: `${{ cosine: `\\cos(q, d_{${d.id.slice(1)}})`, dot: `q \\cdot d_{${d.id.slice(1)}}`, euclidean: `-\\lVert q - d_{${d.id.slice(1)}} \\rVert` }[metric]} = ${s.toFixed(4)}`,
      }],
    });
  }
  steps.push({ id: "rank", title: "Sort by score", description: "Documents are ordered from most to least similar. This ordering is all a vector index returns.", state: { phase: "rank", scoredIds: [...scored] }, events: [] });
  steps.push({ id: "topk", title: `Keep top-${k}`, description: `Only the ${k} highest-scoring documents are passed on: ${ranked.slice(0, k).map((r) => r.doc.title).join(", ")}.`, state: { phase: "topk", scoredIds: [...scored] }, events: [] });
  if (relevant) {
    const p = precisionAtK(ranked, k, relevant);
    const firstRel = ranked.find((r) => relevant.includes(r.doc.id));
    steps.push({
      id: "judge", title: "Compare with relevance labels", description: `Precision@${k} = ${p.toFixed(2)}. The first truly relevant document is at rank ${firstRel?.rank}.`,
      state: { phase: "judge", scoredIds: [...scored] },
      events: firstRel && firstRel.rank > 1 ? [{ id: "miss", type: "relevance", message: `Rank 1 ("${ranked[0].doc.title}") is the most similar vector but is not relevant to the question.`, severity: "warning" }] : [],
    });
  }
  return { steps, ranked };
}
