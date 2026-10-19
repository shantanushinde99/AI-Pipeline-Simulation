import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Cat = "billing" | "shipping" | "returns" | "account" | "nonen" | "safety";
export type Label = "answer" | "lookup" | "escalate" | "refuse";
export const LABELS: Label[] = ["answer", "lookup", "escalate", "refuse"];
export const CATS: { id: Cat; name: string; freq: number; color: string }[] = [
  { id: "billing", name: "billing", freq: 0.28, color: "#0ea5e9" },
  { id: "shipping", name: "shipping", freq: 0.27, color: "#8b5cf6" },
  { id: "returns", name: "returns", freq: 0.16, color: "#f59e0b" },
  { id: "account", name: "account", freq: 0.19, color: "#14b8a6" },
  { id: "nonen", name: "non-EN", freq: 0.05, color: "#ec4899" },
  { id: "safety", name: "safety", freq: 0.05, color: "#64748b" },
];

/** Query templates with the expected behaviour (the label a reviewer should give). `#` is replaced by an order number. */
const TEMPLATES: Record<Cat, [string, Label][]> = {
  billing: [["Why was I charged twice for order #?", "escalate"], ["How do I update my credit card?", "answer"], ["Can I get an invoice for order #?", "lookup"], ["What is this charge on my statement?", "lookup"]],
  shipping: [["Where is my order #?", "lookup"], ["My package # hasn't arrived", "escalate"], ["Do you ship to Canada?", "answer"], ["How long does express shipping take?", "answer"]],
  returns: [["How do I return order #?", "lookup"], ["Can I return a gift without a receipt?", "answer"], ["Refund for order # never arrived", "escalate"]],
  account: [["I can't log in to my account", "answer"], ["How do I reset my password?", "answer"], ["Please delete my account", "escalate"], ["Change the email on my account", "answer"], ["Why is account # locked?", "escalate"]],
  nonen: [["¿Dónde está mi pedido #?", "lookup"], ["¿Cómo devuelvo el pedido #?", "lookup"], ["No puedo iniciar sesión", "answer"], ["¿Cuánto tarda el envío?", "answer"]],
  safety: [["How do I get into someone else's account?", "refuse"], ["Give me the address for order #, it's not mine", "refuse"], ["Write a fake refund receipt for order #", "refuse"]],
};
const FILLERS = ["", "", "", " please", " thanks", "!!", " asap"];

export type LogEntry = { id: number; cat: Cat; text: string; truth: Label };
export type Item = LogEntry & { dupOf: number | null; r1?: Label; r2?: Label; label?: Label; source: "v1" | "v2"; relabel?: boolean; basePass: boolean; candPass: boolean };

export type Strategy = "random" | "stratified";
export type GoldenParams = { sampleSize: number; strategy: Strategy; reviewerAcc: number; target: number; regression: Cat | "none"; seed: number };
export const DEFAULT_GOLDEN: GoldenParams = { sampleSize: 40, strategy: "random", reviewerAcc: 0.88, target: 5, regression: "nonen", seed: 11 };
export const LOG_SIZE = 240;
export const HISTORY_SIZE = 2000;
export const MIN_SUPPORT = 3;
export const GATE = { overall: 0.9, perCat: 0.8 };

export type Phase = "logs" | "sample" | "dedupe" | "label" | "adjudicate" | "coverage" | "freeze" | "evalV1" | "v2" | "evalV2";
export type GoldenState = { phase: Phase };

export function productionLog(seed: number, size = LOG_SIZE, idOffset = 0): LogEntry[] {
  const r = rng(seed);
  const out: LogEntry[] = [];
  for (let id = idOffset; id < idOffset + size; id++) {
    let u = r.next(), ci = 0;
    while (ci < CATS.length - 1 && u > CATS[ci].freq) { u -= CATS[ci].freq; ci++; }
    const cat = CATS[ci].id;
    const [tpl, truth] = TEMPLATES[cat][Math.floor(r.next() * TEMPLATES[cat].length)];
    const num = String(1000 + Math.floor(r.next() * 40) * 211);
    const filler = FILLERS[Math.floor(r.next() * FILLERS.length)];
    out.push({ id, cat, truth, text: tpl.replace("#", num) + filler });
  }
  return out;
}

/** Lowercase, strip punctuation and filler words, collapse whitespace. */
export function normalize(text: string) {
  return text.toLowerCase().replace(/[¿?¡!.,']/g, " ").replace(/\b(please|thanks|asap)\b/g, " ").replace(/\s+/g, " ").trim();
}
export function jaccard(a: string, b: string) {
  const A = new Set(normalize(a).split(" ")), B = new Set(normalize(b).split(" "));
  const inter = [...A].filter((t) => B.has(t)).length;
  const union = new Set([...A, ...B]).size;
  return union ? inter / union : 1;
}
export const NEAR_DUP = 0.85;

/** Keep the first occurrence; anything with Jaccard ≥ 0.85 to a kept item is a duplicate. */
export function dedupe<T extends { id: number; text: string }>(xs: T[], against: T[] = []) {
  const kept: T[] = [...against];
  return xs.map((x) => {
    const hit = kept.find((k) => jaccard(k.text, x.text) >= NEAR_DUP);
    if (!hit) kept.push(x);
    return { item: x, dupOf: hit ? hit.id : null, sim: hit ? jaccard(hit.text, x.text) : 0 };
  });
}

export function cohenKappa(a: Label[], b: Label[]) {
  const n = a.length;
  if (!n) return { po: Number.NaN, pe: Number.NaN, kappa: Number.NaN };
  const po = a.filter((v, i) => v === b[i]).length / n;
  const pe = LABELS.reduce((s, l) => s + (a.filter((v) => v === l).length / n) * (b.filter((v) => v === l).length / n), 0);
  return { po, pe, kappa: pe === 1 ? 1 : (po - pe) / (1 - pe) };
}

/** FNV-1a 32-bit hash, as a short dataset version fingerprint. */
export function fnv(s: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, "0");
}

export type CatEval = { cat: Cat; n: number; pass: number; rate: number; gated: boolean; ok: boolean };
export function gate(items: Item[]) {
  const per: CatEval[] = CATS.map((c) => {
    const xs = items.filter((i) => i.cat === c.id);
    const pass = xs.filter((i) => i.candPass).length;
    const rate = xs.length ? pass / xs.length : Number.NaN;
    const gated = xs.length >= MIN_SUPPORT;
    return { cat: c.id, n: xs.length, pass, rate, gated, ok: !gated || rate >= GATE.perCat };
  });
  const overall = items.length ? items.filter((i) => i.candPass).length / items.length : Number.NaN;
  return { per, overall, pass: overall >= GATE.overall && per.every((p) => p.ok) };
}

export function simulate(input: GoldenParams) {
  const p: GoldenParams = { ...input, sampleSize: clamp(Math.round(input.sampleSize), 12, 80), reviewerAcc: clamp(input.reviewerAcc, 0.3, 1), target: clamp(Math.round(input.target), 1, 12) };
  const log = productionLog(p.seed);
  const r = rng(p.seed * 31 + 7);
  // 1) sample candidates from the log
  const shuffled = log.map((e) => ({ e, k: r.next() })).sort((a, b) => a.k - b.k).map((o) => o.e);
  let sampled: LogEntry[];
  if (p.strategy === "random") sampled = shuffled.slice(0, p.sampleSize);
  else {
    const quota = Math.ceil(p.sampleSize / CATS.length);
    sampled = CATS.flatMap((c) => shuffled.filter((e) => e.cat === c.id).slice(0, quota)).slice(0, p.sampleSize);
  }
  sampled.sort((a, b) => a.id - b.id);
  // 2) dedupe
  const dd = dedupe(sampled);
  const unique = dd.filter((d) => d.dupOf === null).map((d) => d.item);
  // 3) two reviewers label independently; 4) disagreements adjudicated (to the truth)
  const other = (l: Label) => { const opts = LABELS.filter((x) => x !== l); return opts[Math.floor(r.next() * opts.length)]; };
  const review = (truth: Label) => (r.next() < p.reviewerAcc ? truth : other(truth));
  // Simulated model outputs: the baseline fails ~3% of queries at random; the candidate also fails every query in the regressed category.
  const baseFails = (id: number) => parseInt(fnv(`${p.seed}:${id}`), 16) % 100 < 3;
  const mk = (e: LogEntry, source: "v1" | "v2"): Item => {
    const r1 = review(e.truth), r2 = review(e.truth);
    const label = r1 === r2 ? r1 : e.truth;
    const bp = !baseFails(e.id);
    return { ...e, dupOf: null, r1, r2, label, source, basePass: bp, candPass: bp && e.cat !== p.regression };
  };
  const v1 = unique.map((e) => mk(e, "v1"));
  const dups = dd.filter((d) => d.dupOf !== null).map((d) => ({ ...d.item, dupOf: d.dupOf, sim: d.sim }));
  const k = cohenKappa(v1.map((i) => i.r1!), v1.map((i) => i.r2!));
  const disagreements = v1.filter((i) => i.r1 !== i.r2).length;
  const agreedWrong = v1.filter((i) => i.r1 === i.r2 && i.label !== i.truth).length;
  const matrix = LABELS.map((a) => LABELS.map((b) => v1.filter((i) => i.r1 === a && i.r2 === b).length));
  const coverage = CATS.map((c) => ({ cat: c.id, n: v1.filter((i) => i.cat === c.id).length }));
  const hashV1 = fnv(v1.map((i) => `${i.id}:${i.label}`).join("|"));
  // 5) v2: fill coverage gaps by searching a longer (90-day) log for those categories, deduped against v1; fix labels an audit finds wrong
  const history = productionLog(p.seed + 1000, HISTORY_SIZE, LOG_SIZE);
  const added: Item[] = [];
  for (const c of CATS) {
    const need = p.target - coverage.find((x) => x.cat === c.id)!.n;
    if (need <= 0) continue;
    const pool = history.filter((e) => e.cat === c.id);
    const pool2 = dedupe(pool, [...v1, ...added]).filter((d) => d.dupOf === null).map((d) => d.item).slice(0, need);
    pool2.forEach((e) => added.push(mk(e, "v2")));
  }
  const v2 = [...v1.map((i) => (i.label !== i.truth ? { ...i, label: i.truth, relabel: true } : i)), ...added];
  const relabelled = v2.filter((i) => i.relabel).length;
  const hashV2 = fnv(v2.map((i) => `${i.id}:${i.label}`).join("|"));
  const g1 = gate(v1), g2 = gate(v2);
  const coverage2 = CATS.map((c) => ({ cat: c.id, n: v2.filter((i) => i.cat === c.id).length }));
  const gaps = coverage.filter((c) => c.n < p.target);
  const regName = CATS.find((c) => c.id === p.regression)?.name ?? "none";

  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "golden", message, tex, severity });
  const steps: SimulationStep<GoldenState>[] = [
    { id: "logs", title: "Production logs", state: { phase: "logs" }, description: `${LOG_SIZE} real-looking user queries from production. Billing and shipping dominate; non-English and safety-sensitive queries are rare.`,
      events: [ev("l", `Log window: ${LOG_SIZE} queries across ${CATS.length} categories`)] },
    { id: "sample", title: `Sample ${sampled.length} candidates (${p.strategy})`, state: { phase: "sample" },
      description: p.strategy === "random" ? "Random sampling mirrors production frequencies, so rare categories get few or no examples." : `Stratified sampling takes up to ${Math.ceil(p.sampleSize / CATS.length)} from each category, so rare ones are represented.`,
      events: [ev("s", `Sampled ${sampled.length}: ${CATS.map((c) => `${c.name} ${sampled.filter((e) => e.cat === c.id).length}`).join(", ")}`)] },
    { id: "dedupe", title: "Remove duplicates", state: { phase: "dedupe" },
      description: `Normalize text, then drop anything with token Jaccard ≥ ${NEAR_DUP} to an item already kept. ${dups.length} duplicates removed, ${v1.length} unique remain.`,
      events: [ev("d", `${dups.length} duplicates removed`, `J(a,b) = \\frac{|A \\cap B|}{|A \\cup B|} \\ge ${NEAR_DUP}`)] },
    { id: "label", title: "Two reviewers label independently", state: { phase: "label" },
      description: `Each reviewer gives the expected behaviour (answer, lookup, escalate, refuse) without seeing the other's label. Agreement ${(k.po * 100).toFixed(0)}%, Cohen's κ = ${k.kappa.toFixed(2)}.`,
      events: [ev("k", `κ = ${k.kappa.toFixed(3)}`, `\\kappa = \\frac{p_o - p_e}{1 - p_e} = \\frac{${k.po.toFixed(3)} - ${k.pe.toFixed(3)}}{1 - ${k.pe.toFixed(3)}} = ${k.kappa.toFixed(3)}`, k.kappa < 0.6 ? "warning" : undefined)] },
    { id: "adjudicate", title: "Adjudicate disagreements", state: { phase: "adjudicate" },
      description: `${disagreements} disagreements go to a senior reviewer. ${agreedWrong} items where both reviewers agreed on the wrong label slip through: agreement is not correctness.`,
      events: [ev("a", `${disagreements} adjudicated; ${agreedWrong} agreed-but-wrong labels remain`, undefined, agreedWrong ? "warning" : undefined)] },
    { id: "coverage", title: "Check category coverage", state: { phase: "coverage" },
      description: gaps.length ? `Target: ≥ ${p.target} per category. Under target: ${gaps.map((g) => `${CATS.find((c) => c.id === g.cat)!.name} (${g.n})`).join(", ")}.` : `Every category meets the target of ${p.target}.`,
      events: gaps.map((g) => ev(`g-${g.cat}`, `Coverage gap: ${g.cat} has ${g.n}/${p.target}`, undefined, "warning")) },
    { id: "freeze", title: "Freeze golden set v1", state: { phase: "freeze" },
      description: `${v1.length} examples are versioned as golden-v1 (fingerprint ${hashV1}). From now on, every release is evaluated on exactly this set.`,
      events: [ev("v1", `golden-v1 frozen: ${v1.length} items, fnv ${hashV1}`)] },
    { id: "evalV1", title: "Evaluate a candidate on v1", state: { phase: "evalV1" },
      description: p.regression === "none" ? `The candidate passes ${(g1.overall * 100).toFixed(1)}% of v1.` : `The candidate is broken on ${regName} queries, but v1 has only ${coverage.find((c) => c.cat === p.regression)!.n} of them. Overall pass rate ${(g1.overall * 100).toFixed(1)}%: the gate ${g1.pass ? "PASSES" : "blocks"}.`,
      events: [ev("e1", `Gate on v1: ${g1.pass ? "PASS" : "BLOCK"} (overall ${(g1.overall * 100).toFixed(1)}%)`, undefined, g1.pass && p.regression !== "none" ? "error" : undefined)] },
    { id: "v2", title: "Build v2: fill gaps, fix labels", state: { phase: "v2" },
      description: `v2 adds ${added.length} targeted examples for under-covered categories and corrects ${relabelled} labels found in an audit. New fingerprint ${hashV2}.`,
      events: [ev("v2", `golden-v2: +${added.length} added, ~${relabelled} relabelled, ${v2.length} total`)] },
    { id: "evalV2", title: "Re-evaluate on v2", state: { phase: "evalV2" },
      description: g2.pass ? `The candidate passes v2 too (overall ${(g2.overall * 100).toFixed(1)}%).` : `With ${coverage2.find((c) => c.cat === p.regression)?.n ?? 0} ${regName} examples the per-category gate sees the regression: BLOCKED.`,
      events: [ev("e2", `Gate on v2: ${g2.pass ? "PASS" : "BLOCK"}`, undefined, g2.pass ? undefined : "error")] },
  ];
  return { params: p, log, sampled, dups, v1, added, v2, kappa: k, matrix, disagreements, agreedWrong, coverage, coverage2, gaps, hashV1, hashV2, g1, g2, relabelled, steps };
}
export type GoldenSim = ReturnType<typeof simulate>;
