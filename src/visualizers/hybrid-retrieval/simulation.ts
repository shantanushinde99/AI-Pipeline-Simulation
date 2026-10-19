import type { SimulationEvent, SimulationStep } from "../../types/simulation";

// ---------- corpus ----------
export type Doc = { id: string; title: string; text: string };
export const DOCS: Doc[] = [
  { id: "D1", title: "Error NW-4012: TLS handshake failed", text: "NW-4012 means the TLS handshake with the edge proxy failed. Check that your HTTP library supports TLS 1.2 or newer." },
  { id: "D2", title: "Error NW-4013: certificate expired", text: "NW-4013 means the certificate for your custom domain has expired. Renew the certificate under Domains, then retry." },
  { id: "D3", title: "API rate limits", text: "Each API key may send 600 requests per minute. Requests above the limit get HTTP 429 Too Many Requests." },
  { id: "D4", title: "Handling throttling", text: "When the service throttles your client, pause before sending again. Use exponential backoff with jitter and honor the Retry-After header." },
  { id: "D5", title: "Resetting your password", text: "Forgot your password? Click the reset link on the sign-in page and we will email you a link to choose a new password." },
  { id: "D6", title: "API access keys", text: "Create API access keys per project. Each client needs a key for programmatic access. Keys can be revoked from the access settings page." },
  { id: "D7", title: "Single sign-on", text: "Enterprise organizations can require SSO so members sign in through the company identity provider." },
  { id: "D8", title: "Error code reference", text: "Every error response includes a code such as NW-4012 or NW-5001 and a request id. Include both when you contact support." },
];

// ---------- tokenizer (keeps identifiers like nw-4013 and retry-after whole) ----------
const STOP = new Set("a an the is are was were be of to in on for and or with by at as it its this that how do does i my me can can't cannot what when which who your you we will so such then both get keep keeps getting should".split(" "));
export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+(?:[-'][a-z0-9]+)*/g) ?? [])
    .filter((w) => !STOP.has(w))
    .map((w) => (w.length > 3 && /^[a-z]+$/.test(w) && w.endsWith("s") && !w.endsWith("ss") ? w.slice(0, -1) : w));
}

// ---------- BM25 ----------
export type Bm25Params = { k1: number; b: number };
export const DEFAULT_BM25: Bm25Params = { k1: 1.2, b: 0.75 };
export type TermScore = { term: string; tf: number; df: number; idf: number; part: number };
export type Bm25Doc = { id: string; score: number; dl: number; terms: TermScore[] };

export const idf = (N: number, df: number) => Math.log(1 + (N - df + 0.5) / (df + 0.5));
/** One term's BM25 contribution: idf · tf(k1+1) / (tf + k1(1 − b + b·dl/avgdl)). */
export const bm25Term = (idfV: number, tf: number, dl: number, avgdl: number, { k1, b }: Bm25Params) =>
  tf === 0 ? 0 : (idfV * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avgdl));

export function bm25(query: string, docs: Doc[], p: Bm25Params) {
  const toks = docs.map((d) => tokenize(`${d.title} ${d.text}`));
  const N = docs.length;
  const avgdl = toks.reduce((s, t) => s + t.length, 0) / N;
  const qTerms = [...new Set(tokenize(query))];
  const dfs = new Map(qTerms.map((t) => [t, toks.filter((d) => d.includes(t)).length]));
  const scored: Bm25Doc[] = docs.map((d, i) => {
    const dl = toks[i].length;
    const terms = qTerms.map((term) => {
      const tf = toks[i].filter((w) => w === term).length;
      const df = dfs.get(term)!;
      const idfV = idf(N, df);
      return { term, tf, df, idf: idfV, part: bm25Term(idfV, tf, dl, avgdl, p) };
    });
    return { id: d.id, score: terms.reduce((s, t) => s + t.part, 0), dl, terms };
  });
  return { N, avgdl, qTerms, dfs, scored };
}

// ---------- toy "semantic" embedding: words → hand-built concept axes ----------
export const CONCEPTS = ["account access", "recovery", "error", "network/TLS", "certificate", "rate limiting", "retry", "API keys", "SSO"] as const;
type C = (typeof CONCEPTS)[number];
const L = (o: Partial<Record<C, number>>) => o;
/** Each word activates one or more concepts. Identifiers like NW-4013 and numbers like 429 are out of vocabulary (no vector). */
export const LEXICON: Record<string, Partial<Record<C, number>>> = {
  password: L({ "account access": 1, recovery: 0.6 }), forgot: L({ recovery: 1 }), reset: L({ recovery: 1 }), "sign-in": L({ "account access": 1 }),
  sign: L({ "account access": 0.8 }), login: L({ "account access": 1 }), locked: L({ "account access": 0.7, recovery: 0.5 }), recover: L({ recovery: 1 }),
  remember: L({ recovery: 0.8 }), secret: L({ "account access": 0.8 }), credential: L({ "account access": 1 }), account: L({ "account access": 0.8 }),
  email: L({ recovery: 0.3 }), access: L({ "account access": 0.5, "API keys": 0.5 }),
  error: L({ error: 1 }), failed: L({ error: 0.8 }), code: L({ error: 0.6 }), response: L({ error: 0.3 }), support: L({ error: 0.2 }), broken: L({ error: 0.8 }), fail: L({ error: 0.8 }),
  tls: L({ "network/TLS": 1 }), handshake: L({ "network/TLS": 1 }), proxy: L({ "network/TLS": 0.8 }), client: L({ "network/TLS": 0.4, retry: 0.2 }), connection: L({ "network/TLS": 1 }),
  certificate: L({ certificate: 1 }), expired: L({ certificate: 0.6, error: 0.3 }), domain: L({ certificate: 0.5 }), renew: L({ certificate: 0.7 }),
  limit: L({ "rate limiting": 1 }), minute: L({ "rate limiting": 0.5 }), request: L({ "rate limiting": 0.3 }), many: L({ "rate limiting": 0.4 }),
  throttle: L({ "rate limiting": 1, retry: 0.4 }), slow: L({ "rate limiting": 0.8, retry: 0.4 }), down: L({ "rate limiting": 0.3 }), pause: L({ retry: 0.8 }),
  backoff: L({ retry: 1 }), jitter: L({ retry: 0.8 }), "retry-after": L({ retry: 1, "rate limiting": 0.5 }), retry: L({ retry: 1 }), wait: L({ retry: 0.8 }), back: L({ retry: 0.5, "rate limiting": 0.25 }), off: L({ retry: 0.5, "rate limiting": 0.25 }),
  api: L({ "API keys": 0.6 }), key: L({ "API keys": 1 }), project: L({ "API keys": 0.3 }), revoked: L({ "API keys": 0.5 }), programmatic: L({ "API keys": 0.6 }),
  sso: L({ SSO: 1 }), identity: L({ SSO: 0.8 }), provider: L({ SSO: 0.6 }), enterprise: L({ SSO: 0.4 }),
};
export function embed(text: string): number[] {
  const v = CONCEPTS.map(() => 0);
  for (const t of tokenize(text)) {
    const e = LEXICON[t];
    if (e) CONCEPTS.forEach((c, i) => (v[i] += e[c] ?? 0));
  }
  const n = Math.hypot(...v);
  return n ? v.map((x) => x / n) : v;
}
export const cosine = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

// ---------- fusion ----------
export type FusionMode = "rrf" | "weighted";
export type Ranked = { id: string; score: number; rank: number };
const rankBy = (rows: { id: string; score: number }[]): Ranked[] =>
  rows.filter((r) => r.score > 1e-12).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).map((r, i) => ({ ...r, rank: i + 1 }));

export type FusedRow = Ranked & { bm25Rank?: number; vecRank?: number; fromBm25: number; fromVec: number };
export function fuse(bmList: Ranked[], vecList: Ranked[], mode: FusionMode, kRrf: number, alpha: number): FusedRow[] {
  const ids = [...new Set([...bmList, ...vecList].map((r) => r.id))];
  const maxBm = Math.max(1e-12, ...bmList.map((r) => r.score));
  return rankBy(ids.map((id) => {
    const b = bmList.find((r) => r.id === id), v = vecList.find((r) => r.id === id);
    const fromBm25 = mode === "rrf" ? (b ? 1 / (kRrf + b.rank) : 0) : alpha * ((b?.score ?? 0) / maxBm);
    const fromVec = mode === "rrf" ? (v ? 1 / (kRrf + v.rank) : 0) : (1 - alpha) * (v?.score ?? 0);
    return { id, score: fromBm25 + fromVec, bm25Rank: b?.rank, vecRank: v?.rank, fromBm25, fromVec };
  })) as FusedRow[];
}

// ---------- presets ----------
export type QueryPreset = { id: string; label: string; query: string; relevant: string[]; expect: string };
export const QUERIES: QueryPreset[] = [
  { id: "code", label: "Exact error code", query: "NW-4013 error", relevant: ["D2"], expect: "BM25 wins: the identifier is a rare exact token; the concept embedding has never seen it." },
  { id: "paraphrase", label: "Paraphrased question", query: "Locked out: how do I recover access when I can't remember my login secret?", relevant: ["D5"], expect: "Vectors win: no important word overlaps with the password-reset page, but the meaning does." },
  { id: "mixed", label: "Mixed: code + intent", query: "API calls fail with 429. How should my client back off?", relevant: ["D3", "D4"], expect: "Fusion wins: BM25 finds the 429 page, vectors find the back-off advice; neither finds both." },
];

export type HybridInput = { queryId: string; customQuery?: string; k1: number; b: number; mode: FusionMode; kRrf: number; alpha: number; topN: number };
export const DEFAULT_INPUT: HybridInput = { queryId: "mixed", k1: 1.2, b: 0.75, mode: "rrf", kRrf: 60, alpha: 0.5, topN: 2 };

export type Run = ReturnType<typeof runHybrid>;
export function runHybrid(input: HybridInput) {
  const preset = QUERIES.find((q) => q.id === input.queryId) ?? QUERIES[0];
  const query = input.customQuery?.trim() || preset.query;
  const p = { k1: Math.max(0, input.k1), b: Math.min(1, Math.max(0, input.b)) };
  const bm = bm25(query, DOCS, p);
  const qVec = embed(query);
  const vec = DOCS.map((d) => ({ id: d.id, score: cosine(qVec, embed(`${d.title} ${d.text}`)) }));
  const bmList = rankBy(bm.scored);
  const vecList = rankBy(vec);
  const kRrf = Math.max(0, input.kRrf), alpha = Math.min(1, Math.max(0, input.alpha));
  const fused = fuse(bmList, vecList, input.mode, kRrf, alpha);
  const topN = Math.max(1, Math.min(DOCS.length, Math.round(input.topN)));
  const relevant = input.customQuery?.trim() ? [] : preset.relevant;
  const recall = (list: Ranked[]) => (relevant.length ? list.slice(0, topN).filter((r) => relevant.includes(r.id)).length / relevant.length : NaN);
  return {
    preset, query, params: p, bm, qVec, vec, bmList, vecList, fused, topN, relevant, mode: input.mode, kRrf, alpha,
    qTokens: tokenize(query), oov: tokenize(query).filter((t) => !LEXICON[t]),
    recall: { bm25: recall(bmList), vector: recall(vecList), fused: recall(fused) },
  };
}

export type Phase = "tokenize" | "idf" | "bm25" | "vector" | "lists" | "fuse" | "compare";
export type HybridState = { phase: Phase };

export function simulate(input: HybridInput): { run: Run; steps: SimulationStep<HybridState>[] } {
  const run = runHybrid(input);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "retrieval", message, severity, tex });
  const top = (l: Ranked[]) => l.slice(0, run.topN).map((r) => r.id).join(", ") || "none";
  const pct = (v: number) => (Number.isNaN(v) ? "n/a" : `${Math.round(v * 100)}%`);
  const bestIdf = run.bm.qTerms.filter((t) => run.bm.dfs.get(t)! > 0).sort((a, b) => idf(run.bm.N, run.bm.dfs.get(b)!) - idf(run.bm.N, run.bm.dfs.get(a)!))[0];
  const steps: SimulationStep<HybridState>[] = [
    {
      id: "tokenize", title: "Split the query two ways",
      description: `BM25 sees exact terms (${run.bm.qTerms.join(", ") || "none"}). The embedding sees concepts; ${run.oov.length ? `${run.oov.join(", ")} ${run.oov.length === 1 ? "is" : "are"} out of its vocabulary` : "every term is in its vocabulary"}.`,
      state: { phase: "tokenize" },
      events: [ev("t", `terms: ${run.bm.qTerms.join(" · ")}`), ...(run.oov.length ? [ev("oov", `no embedding for: ${run.oov.join(", ")}`, "warning")] : [])],
    },
    {
      id: "idf", title: "BM25: how rare is each term? (IDF)",
      description: `IDF rewards rare terms. With N = ${run.bm.N} documents, a term found in 1 document gets idf ${idf(run.bm.N, 1).toFixed(2)}; one found in 4 gets ${idf(run.bm.N, 4).toFixed(2)}.`,
      state: { phase: "idf" },
      events: run.bm.qTerms.map((t) => { const df = run.bm.dfs.get(t)!; return ev(`idf-${t}`, `idf(${t}) = ln(1 + (${run.bm.N} − ${df} + 0.5)/(${df} + 0.5)) = ${idf(run.bm.N, df).toFixed(3)}`, df === 0 ? "warning" : undefined, `\\text{idf}(\\text{${t}}) = \\ln\\!\\left(1 + \\tfrac{${run.bm.N} - ${df} + 0.5}{${df} + 0.5}\\right) = ${idf(run.bm.N, df).toFixed(3)}`); }),
    },
    {
      id: "bm25", title: "BM25: score every document",
      description: `Each matching term adds idf × a saturating term-frequency factor, scaled down for long documents (b = ${run.params.b}). Top ${run.topN}: ${top(run.bmList)}.`,
      state: { phase: "bm25" },
      events: [ev("bm", `BM25 ranking: ${run.bmList.map((r) => `${r.id} ${r.score.toFixed(2)}`).join(", ") || "no document shares a term"}`), ...(bestIdf ? [ev("rare", `rarest matching term: ${bestIdf} (in ${run.bm.dfs.get(bestIdf)} doc(s))`)] : [])],
    },
    {
      id: "vector", title: "Vectors: cosine similarity of concepts",
      description: `The query becomes a ${CONCEPTS.length}-dimensional concept vector and is compared with each document's vector. Top ${run.topN}: ${top(run.vecList)}.`,
      state: { phase: "vector" },
      events: [ev("vec", `vector ranking: ${run.vecList.map((r) => `${r.id} ${r.score.toFixed(2)}`).join(", ") || "query vector is empty"}`, run.vecList.length ? undefined : "warning")],
    },
    {
      id: "lists", title: "Two ranked lists, side by side",
      description: `BM25 top ${run.topN}: ${top(run.bmList)}. Vector top ${run.topN}: ${top(run.vecList)}. Documents that score 0 are not in a list.`,
      state: { phase: "lists" },
      events: [ev("cmp", `recall@${run.topN}: BM25 ${pct(run.recall.bm25)}, vector ${pct(run.recall.vector)}`)],
    },
    {
      id: "fuse", title: run.mode === "rrf" ? `Reciprocal Rank Fusion (k = ${run.kRrf})` : `Weighted blend (α = ${run.alpha})`,
      description: run.mode === "rrf"
        ? `Each list contributes 1/(k + rank). Scores are ignored, only positions count, so a document ranked well by both lists rises to the top.`
        : `fused = α · BM25/max(BM25) + (1 − α) · cosine. Scores must be normalized first because BM25 is unbounded and cosine lies in [−1, 1].`,
      state: { phase: "fuse" },
      events: run.fused.slice(0, 4).map((f) => ev(`f-${f.id}`, `${f.id}: ${f.fromBm25.toFixed(4)} + ${f.fromVec.toFixed(4)} = ${f.score.toFixed(4)}`, undefined,
        run.mode === "rrf" ? `${f.id}: ${f.bm25Rank ? `\\tfrac{1}{${run.kRrf}+${f.bm25Rank}}` : "0"} + ${f.vecRank ? `\\tfrac{1}{${run.kRrf}+${f.vecRank}}` : "0"} = ${f.score.toFixed(4)}` : `${f.id}: ${f.fromBm25.toFixed(3)} + ${f.fromVec.toFixed(3)} = ${f.score.toFixed(3)}`)),
    },
    {
      id: "compare", title: "Who found the relevant documents?",
      description: run.relevant.length
        ? `recall@${run.topN}: BM25 ${pct(run.recall.bm25)}, vectors ${pct(run.recall.vector)}, fused ${pct(run.recall.fused)}. ${run.preset.expect}`
        : "Custom query: no relevance labels, so compare the three lists by eye.",
      state: { phase: "compare" },
      events: [ev("res", `fused top ${run.topN}: ${top(run.fused)}`, run.relevant.length && run.recall.fused < 1 ? "warning" : undefined)],
    },
  ];
  return { run, steps };
}
