import { splitSentences, tokens, wordCount } from "../rag/simulation";

/**
 * Shared, rule-based claim checker used by grounded-generation, citation-mapping and faithfulness-and-relevance.
 * It stands in for an NLI model / LLM judge: a claim is "supported" by a span when most of its content words
 * appear in the span and (in strict mode) every number in the claim appears there too.
 */
export const SUPPORT_TAU = 0.6;

export type Span = { id: string; passage: number; text: string };
export type SupportCheck = {
  spanId: string | null;
  claimTokens: string[];
  matched: string[];
  coverage: number;
  numbers: string[];
  missingNumbers: string[];
  supported: boolean;
};

const isNum = (t: string) => /\d/.test(t);

export function checkSupport(claim: string, span: string, strict = true): Omit<SupportCheck, "spanId"> {
  const claimTokens = [...new Set(tokens(claim))];
  const have = new Set(tokens(span));
  const matched = claimTokens.filter((t) => have.has(t));
  const coverage = claimTokens.length ? matched.length / claimTokens.length : 0;
  const numbers = claimTokens.filter(isNum);
  const missingNumbers = numbers.filter((t) => !have.has(t));
  const supported = coverage >= SUPPORT_TAU && (!strict || missingNumbers.length === 0);
  return { claimTokens, matched, coverage, numbers, missingNumbers, supported };
}

/** The best span for a claim: supported spans first, then highest coverage, then earliest. */
export function bestSupport(claim: string, spans: Span[], strict = true): SupportCheck {
  let best: SupportCheck = { ...checkSupport(claim, "", strict), spanId: null };
  for (const s of spans) {
    const c = { ...checkSupport(claim, s.text, strict), spanId: s.id };
    if ((c.supported && !best.supported) || (c.supported === best.supported && c.coverage > best.coverage)) best = c;
  }
  return best.coverage > 0 ? best : { ...best, spanId: null };
}

/** Sentence-level spans of numbered passages. Fragments under 4 words (headings) are skipped. */
export function sentenceSpans(passages: { label: number; text: string }[]): Span[] {
  return passages.flatMap((p) => splitSentences(p.text).filter((s) => wordCount(s) >= 4).map((text, i) => ({ id: `${p.label}.${i + 1}`, passage: p.label, text })));
}

/** Greedy word wrap for SVG text (SVG has no automatic line breaking). */
export function wrap(text: string, maxChars: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const w of text.split(/\s+/).filter(Boolean)) {
    if (line && line.length + 1 + w.length > maxChars) {
      lines.push(line);
      line = w;
    } else line = line ? `${line} ${w}` : w;
  }
  if (line) lines.push(line);
  return lines.length ? lines : [""];
}

export const pct = (v: number) => `${Math.round(v * 100)}%`;
