import { dot } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { attend, identity, matVec, type Mat, type Vec } from "./attention";

export const TOKENS = ["the", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"];
/** Teaching-scale feature names for the 4 hand-designed embedding dimensions. */
export const DIMS = ["animate", "place", "pronoun", "verb-ish"];
export const D = 4;

/** Hand-placed embeddings (one row per token), all ≥ 0 so value contributions stack cleanly. */
export const EMBEDDINGS: Mat = [
  [0.1, 0.1, 0, 0.4], // the
  [1, 0.1, 0, 0], // animal
  [0, 0, 0.1, 1], // didn't
  [0.1, 0.3, 0, 0.9], // cross
  [0.1, 0.1, 0, 0.4], // the
  [0, 1, 0, 0.1], // street
  [0, 0, 0.2, 0.6], // because
  [0.2, 0.2, 1, 0], // it
  [0, 0, 0.1, 0.9], // was
  [0.6, 0, 0, 0.5], // tired
];

/** Designed so a pronoun's query asks "where is an animate thing (or a place)?" */
export const W_Q_DESIGNED: Mat = [
  [0.5, 0, 3, 0.6],
  [0, 0.5, 1.5, 0.6],
  [0, 0, 0, 0],
  [0.3, 0.3, 0, 0.5],
];
/** Keys advertise "animate" and "place" strongly. */
export const W_K_DESIGNED: Mat = [
  [2, 0, 0, 0],
  [0, 2, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

export type WeightsPreset = "designed" | "random";
export type SAParams = { query: number; scaled: boolean; causal: boolean; weights: WeightsPreset; seed: number };
export const DEFAULT_PARAMS: SAParams = { query: 7, scaled: true, causal: false, weights: "designed", seed: 7 };

export function randomMatrix(seed: number, n = D, s = 1): Mat {
  const r = rng(seed);
  return Array.from({ length: n }, () => Array.from({ length: n }, () => Math.round(r.gaussian() * s * 10) / 10));
}

export function projections(p: Pick<SAParams, "weights" | "seed">) {
  const WQ = p.weights === "designed" ? W_Q_DESIGNED : randomMatrix(p.seed);
  const WK = p.weights === "designed" ? W_K_DESIGNED : randomMatrix(p.seed + 101);
  const WV = identity(D); // value = embedding, so the output is readable in the same 4 named features
  const X = EMBEDDINGS;
  return { WQ, WK, WV, X, Q: X.map((x) => matVec(WQ, x)), K: X.map((x) => matVec(WK, x)), V: X.map((x) => matVec(WV, x)) };
}

export function compute(params: SAParams) {
  const p = { ...params, query: Math.max(0, Math.min(TOKENS.length - 1, Math.round(params.query))) };
  const proj = projections(p);
  const att = attend(proj.Q, proj.K, proj.V, { scaled: p.scaled, causal: p.causal });
  const i = p.query;
  const weights = att.weights[i];
  const top = weights.reduce((b, w, j) => (w > weights[b] ? j : b), 0);
  const entropy = -weights.reduce((s, w) => s + (w > 0 ? w * Math.log(w) : 0), 0);
  return { ...proj, ...att, p, i, q: proj.Q[i], row: { scores: att.scores[i], scaled: att.scaled[i], weights, out: att.out[i] }, top, entropy };
}

export type Phase = "tokens" | "query" | "scores" | "scale" | "softmax" | "lines" | "mix" | "heatmap";
export type SAState = {
  phase: Phase;
  query: number;
  scores: Vec;
  scaled: Vec;
  weights: Vec;
  out: Vec;
  matrix: Mat;
  top: number;
};

const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "compute", message, tex, severity });
const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "−∞");

export function simulate(params: SAParams): SimulationStep<SAState>[] {
  const c = compute(params);
  const { i, row, top, p } = c;
  const tok = TOKENS[i];
  const base = { query: i, scores: [...row.scores], scaled: [...row.scaled], weights: [...row.weights], out: [...row.out], matrix: c.weights.map((r) => [...r]), top };
  const s = (phase: Phase) => ({ ...base, phase });
  const sqrtD = Math.sqrt(D);
  const masked = row.scaled.filter((v) => !Number.isFinite(v)).length;

  return [
    {
      id: "tokens", title: "Tokens become vectors", state: s("tokens"),
      description: `Each of the ${TOKENS.length} tokens is a 4-number embedding (teaching-scale features: ${DIMS.join(", ")}). From it, every token gets a query, a key and a value.`,
      events: [ev("emb", `x_${tok} = [${c.X[i].join(", ")}]`, `x_{\\text{${tok}}} = [${c.X[i].join(", ")}]`)],
    },
    {
      id: "query", title: `"${tok}" forms its query`, state: s("query"),
      description: `The query q = W_Q·x asks what "${tok}" is looking for. Every token's key k = W_K·x advertises what it offers.`,
      events: [ev("q", `q_${tok} = [${c.q.map(f2).join(", ")}]`, `q_{\\text{${tok}}} = [${c.q.map(f2).join(", ")}]`)],
    },
    {
      id: "scores", title: "Score against every token", state: s("scores"),
      description: `Dot product q·k_j for each of the ${TOKENS.length} tokens: a big number means the key matches what the query asks for.`,
      events: [
        ev("s-top", `Highest raw score: "${TOKENS[top]}" = ${f2(row.scores[top])}`, `q \\cdot k_{\\text{${TOKENS[top]}}} = ${f2(row.scores[top])}`),
        ...(masked ? [ev("mask", `${masked} future token(s) will be masked by the causal mask.`, undefined, "warning")] : []),
      ],
    },
    {
      id: "scale", title: p.scaled ? "Scale by 1/√d" : "Scaling skipped", state: s("scale"),
      description: p.scaled
        ? `Divide every score by √d = √${D} = ${sqrtD}. This keeps scores from growing with dimension and saturating the softmax.`
        : "Scaling is switched off, so the raw scores go straight into softmax and the distribution becomes sharper.",
      events: [
        p.scaled
          ? ev("sc", `${f2(row.scores[top])} / ${sqrtD} = ${f2(row.scaled[top])}`, `\\frac{${f2(row.scores[top])}}{\\sqrt{${D}}} = ${f2(row.scaled[top])}`)
          : ev("sc", "No scaling applied (break-it mode).", undefined, "warning"),
        ...(masked ? [ev("mask2", `Future positions set to −∞ (causal mask).`, undefined, "warning")] : []),
      ],
    },
    {
      id: "softmax", title: "Softmax turns scores into weights", state: s("softmax"),
      description: `e^score for each token, divided by the total, gives weights that are positive and sum to 1. "${TOKENS[top]}" gets ${(row.weights[top] * 100).toFixed(1)}%.`,
      events: [ev("w", `w_${TOKENS[top]} = ${row.weights[top].toFixed(3)}; Σw = ${row.weights.reduce((a, b) => a + b, 0).toFixed(3)}`, `w_{\\text{${TOKENS[top]}}} = ${row.weights[top].toFixed(3)},\\ \\sum_j w_j = ${row.weights.reduce((a, b) => a + b, 0).toFixed(3)}`)],
    },
    {
      id: "lines", title: "Attention lines", state: s("lines"),
      description: `Line thickness ∝ weight. "${tok}" attends most to "${TOKENS[top]}"; entropy ${c.entropy.toFixed(2)} nats (uniform over 10 would be ${Math.log(TOKENS.length).toFixed(2)}).`,
      events: [ev("ent", `entropy = ${c.entropy.toFixed(3)} nats`, `H = -\\sum_j w_j \\ln w_j = ${c.entropy.toFixed(3)}`)],
    },
    {
      id: "mix", title: "Weighted sum of values", state: s("mix"),
      description: `Each token sends weight × value. The contributions stack into the new vector for "${tok}", which now carries features borrowed from "${TOKENS[top]}".`,
      events: [ev("out", `out_${tok} = [${row.out.map((v) => v.toFixed(3)).join(", ")}]`, `\\text{out}_{\\text{${tok}}} = \\sum_j w_j v_j = [${row.out.map((v) => v.toFixed(3)).join(", ")}]`)],
    },
    {
      id: "heatmap", title: "Every token does this at once", state: s("heatmap"),
      description: "Stacking every token's weight row gives the full attention matrix. Row = who is looking, column = who is looked at. Click a row to re-run for that token.",
      events: [ev("rows", `All ${TOKENS.length} rows computed in parallel; each row sums to 1.`)],
    },
  ];
}

/** Raw score q·k used by the card. */
export const score = (q: Vec, k: Vec) => dot(q, k);
