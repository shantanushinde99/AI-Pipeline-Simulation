import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { CORPUS, type SourceDoc } from "../rag/corpus";
import { DEFAULT_RAG, buildIndex, wordCount } from "../rag/simulation";
import { bestSupport, checkSupport, sentenceSpans, type SupportCheck } from "../grounded-generation/claims";

export type CMBug = "none" | "offByOne" | "dedupe" | "truncated";
/** strict = sentence-level check with numbers; lenient = bag of words over the whole passage (a weak verifier). */
export type CMParams = { bug: CMBug; dedupeFirst: boolean; strict: boolean };
export const DEFAULT_CM: CMParams = { bug: "none", dedupeFirst: false, strict: true };

export const BUGS: Record<CMBug, { label: string; note: string }> = {
  none: { label: "No bug (healthy mapping)", note: "Markers are resolved by stable chunk id, so dedupe and numbering can't break them." },
  offByOne: { label: "Off-by-one labels", note: "The resolver treats [n] as a 0-based index, so every marker lands on the next passage." },
  dedupe: { label: "Labels renumbered after dedupe", note: "The source list is deduplicated and renumbered after generation, but the answer's markers are not remapped." },
  truncated: { label: "Citing a truncated passage", note: "The context budget cut passage [4] mid-sentence; the model completed the sentence from memory and still cited [4]." },
};

export const QUESTION = "Which plan includes SSO, how long are its request logs kept, and what support does it get?";
/** The (simulated) generator's answer: three claims. Each cites the first prompt passage it was read from. */
export const CLAIMS = [
  "SSO is available on the Enterprise plan only.",
  "Request logs are retained for 30 days on Starter and Pro plans and for 365 days on the Enterprise plan.",
  "Enterprise plan: Price custom pricing; Seats unlimited; Support 24/7 phone.",
];
/** Hybrid retrieval: vector and keyword search both return sso#0, so the merged list holds a duplicate. */
const RETRIEVED: { chunkId: string; via: "vector" | "keyword" }[] = [
  { chunkId: "sso#0", via: "vector" }, { chunkId: "sso#0", via: "keyword" }, { chunkId: "retention#0", via: "vector" }, { chunkId: "pricing#0", via: "keyword" },
];
const TRUNCATE_AFTER = "Enterprise plan: Price custom pricing;";

export type PromptPassage = { label: number; chunkId: string; docId: string; via: "vector" | "keyword"; text: string; truncated: boolean; words: number };
export type Anchor = { page: number | null; anchor: string };
export type Citation = {
  claim: string; marker: number;
  intended: PromptPassage; // what the generator actually read
  resolved: PromptPassage | null; // what the mapping returns
  doc: SourceDoc | null; anchor: Anchor | null;
  check: SupportCheck | null; pass: boolean; brokenAt: "passage" | null; wrongSource: boolean;
};
export type CMResult = { params: CMParams; prompt: PromptPassage[]; display: PromptPassage[]; removedDup: number | null; citations: Citation[]; accuracy: number };

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
const firstWords = (s: string, n: number) => s.split(/\s+/).slice(0, n).join(" ");

/** Where in its source document a sentence lives: PDFs get a page (3 blocks per page, simulated), Markdown gets a heading anchor + paragraph. */
export function anchorFor(doc: SourceDoc, sentence: string): Anchor {
  const heading = doc.blocks.find((b) => b.type === "heading");
  const headingText = heading && heading.type === "heading" ? heading.text : doc.title;
  let para = 0;
  for (let i = 0; i < doc.blocks.length; i++) {
    const b = doc.blocks[i];
    const page = doc.file.endsWith(".pdf") ? 1 + Math.floor(i / 3) : null;
    if (b.type === "table") {
      const row = b.rows.findIndex((r) => sentence.startsWith(`${r[0]} plan:`));
      if (row >= 0) return { page, anchor: `table “${b.caption}”, row ${row + 1}` };
    } else if (b.type === "paragraph") {
      para++;
      if (b.text.includes(sentence) || sentence.includes(b.text.slice(0, 30))) return { page, anchor: page ? `¶${para}` : `#${slug(headingText)} ¶${para}` };
    }
  }
  return { page: doc.file.endsWith(".pdf") ? 1 : null, anchor: `#${slug(headingText)}` };
}

export function mapCitations(params: CMParams): CMResult {
  const index = buildIndex(DEFAULT_RAG);
  const chunkText = (id: string) => index.chunks.find((c) => c.id === id)!.text;
  const retrieved = params.dedupeFirst ? RETRIEVED.filter((r, i) => RETRIEVED.findIndex((x) => x.chunkId === r.chunkId) === i) : RETRIEVED;
  const prompt: PromptPassage[] = retrieved.map((r, i) => {
    let text = chunkText(r.chunkId);
    const cut = params.bug === "truncated" && r.chunkId === "pricing#0";
    if (cut) text = text.slice(0, text.indexOf(TRUNCATE_AFTER) + TRUNCATE_AFTER.length);
    return { label: i + 1, chunkId: r.chunkId, docId: r.chunkId.split("#")[0], via: r.via, text, truncated: cut, words: wordCount(text) };
  });
  // Post-generation dedupe of the source list shown to the user.
  const removedDup = prompt.findIndex((p, i) => prompt.findIndex((x) => x.chunkId === p.chunkId) !== i);
  const display = prompt.filter((p, i) => prompt.findIndex((x) => x.chunkId === p.chunkId) === i).map((p, i) => ({ ...p, label: i + 1 }));

  const citations: Citation[] = CLAIMS.map((claim) => {
    const intended = prompt.find((p) => p.text.includes(firstWords(claim, 5)))!;
    const marker = intended.label;
    let resolved: PromptPassage | null;
    if (params.bug === "offByOne") resolved = prompt[marker] ?? null; // [n] read as 0-based index
    else if (params.bug === "dedupe") resolved = display[marker - 1] ?? null; // positions shifted, markers not remapped
    else resolved = prompt[marker - 1]; // stable: label → prompt passage → chunk id
    if (!resolved) return { claim, marker, intended, resolved: null, doc: null, anchor: null, check: null, pass: false, brokenAt: "passage", wrongSource: true };
    const doc = CORPUS.find((d) => d.id === resolved.docId)!;
    const spans = sentenceSpans([{ label: resolved.label, text: resolved.text }]);
    const best = bestSupport(claim, spans, true);
    const check: SupportCheck = params.strict ? best : { ...checkSupport(claim, resolved.text, false), spanId: null };
    const spanText = best.spanId ? spans.find((s) => s.id === best.spanId)!.text : resolved.text;
    return { claim, marker, intended, resolved, doc, anchor: anchorFor(doc, spanText), check, pass: check.supported, brokenAt: null, wrongSource: resolved.chunkId !== intended.chunkId };
  });
  return { params, prompt, display, removedDup: removedDup >= 0 ? removedDup + 1 : null, citations, accuracy: citations.filter((c) => c.pass).length / citations.length };
}

export type CMState = { phase: "answer" | "sources" | "resolve" | "verify" | "score"; focus: number | null; resolved: number };

export function simulate(params: CMParams): { result: CMResult; steps: SimulationStep<CMState>[] } {
  const result = mapCitations(params);
  const { prompt, display, citations } = result;
  let n = 0;
  const ev = (message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id: `e${n++}`, type: "citation", message, severity, tex });
  const steps: SimulationStep<CMState>[] = [];
  steps.push({
    id: "answer", title: "Answer with [n] markers",
    description: `The generator answered in ${citations.length} sentences, each ending with a marker that names the prompt passage it came from: ${citations.map((c) => `[${c.marker}]`).join(" ")}.`,
    state: { phase: "answer", focus: null, resolved: 0 },
    events: citations.map((c, i) => ev(`sentence ${i + 1} cites [${c.marker}] (read from ${c.intended.chunkId})`)),
  });
  const dedupeNote = params.dedupeFirst
    ? "Duplicates were removed before the prompt was built, so labels never change after generation."
    : `Hybrid retrieval returned ${prompt[result.removedDup! - 1]?.chunkId} twice. The source list shown to users is deduplicated after generation: ${prompt.length} → ${display.length} entries.`;
  steps.push({
    id: "sources", title: params.dedupeFirst ? "Prompt passages (deduped first)" : "Dedupe the source list",
    description: `${dedupeNote} ${params.bug === "dedupe" ? "Bug: the list is renumbered by position, but the markers in the answer are left as they were." : "Markers are kept tied to chunk ids, so renumbering the display list is safe."}`,
    state: { phase: "sources", focus: null, resolved: 0 },
    events: [
      ev(`prompt passages: ${prompt.map((p) => `[${p.label}] ${p.chunkId} (${p.via})`).join(", ")}`),
      ...(result.removedDup ? [ev(`dedupe removed [${result.removedDup}] (duplicate of ${prompt[result.removedDup - 1].chunkId})`, params.bug === "dedupe" ? "warning" : undefined)] : []),
      ...prompt.filter((p) => p.truncated).map((p) => ev(`[${p.label}] ${p.chunkId} truncated by the context budget at ${p.words} words`, "warning")),
    ],
  });
  citations.forEach((c, i) => {
    const chain = c.resolved ? `[${c.marker}] → passage ${c.resolved.label} → ${c.resolved.chunkId} → ${c.doc!.file} → ${c.anchor!.page ? `p.${c.anchor!.page} ` : ""}${c.anchor!.anchor}` : `[${c.marker}] → no passage with that label`;
    steps.push({
      id: `resolve-${i}`, title: `Resolve [${c.marker}] for sentence ${i + 1}`,
      description: c.resolved
        ? `${chain}. ${c.wrongSource ? `Wrong source: the sentence was read from ${c.intended.chunkId}.` : c.resolved.truncated ? "Right passage, but it was truncated: the stored text stops mid-sentence." : "Same chunk the generator read from."}`
        : `${chain}. The marker is dangling: the user's click goes nowhere.`,
      state: { phase: "resolve", focus: i, resolved: i + 1 },
      events: [ev(chain, c.resolved && !c.wrongSource ? undefined : "error")],
    });
  });
  steps.push({
    id: "verify", title: "Verify each citation",
    description: params.strict
      ? "For every marker, the verifier asks: does one sentence of the cited passage support this sentence (≥ 60% of its words, all numbers matching)?"
      : "Lenient verifier: it only checks that ≥ 60% of the sentence's words appear anywhere in the cited passage. Numbers and sentence boundaries are ignored.",
    state: { phase: "verify", focus: null, resolved: citations.length },
    events: citations.map((c, i) => ev(c.check ? `[${c.marker}] sentence ${i + 1}: ${c.pass ? "pass" : "fail"} — best span covers ${(c.check.coverage * 100).toFixed(0)}%${c.check.missingNumbers.length ? `, missing ${c.check.missingNumbers.join(", ")}` : ""}` : `[${c.marker}] sentence ${i + 1}: fail — no passage`, c.pass ? undefined : "error")),
  });
  const passes = citations.filter((c) => c.pass).length;
  steps.push({
    id: "score", title: "Citation accuracy",
    description: `${passes} of ${citations.length} citations point to a passage that supports their sentence: citation accuracy = ${(result.accuracy * 100).toFixed(0)}%.`,
    state: { phase: "score", focus: null, resolved: citations.length },
    events: [ev(`citation accuracy = ${passes}/${citations.length}`, passes < citations.length ? "warning" : undefined, `\\text{citation accuracy} = \\frac{${passes}}{${citations.length}} = ${result.accuracy.toFixed(2)}`)],
  });
  return { result, steps };
}
