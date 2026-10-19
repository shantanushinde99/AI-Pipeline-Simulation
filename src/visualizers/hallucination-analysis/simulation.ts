import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/**
 * Claim-level hallucination checking on a fictional observatory. The answers and their atomic-claim
 * decomposition are hand-written (a stand-in for an LLM decomposer); the checker is a real rule-based
 * comparison of each structured claim against structured source facts.
 */
export type SourceId = "S1" | "S2" | "S3";
export type Fact = { id: string; source: SourceId; attr: string; value: string | number; text: string; exact?: boolean; quantifier?: "some" | "all" };
export type Source = { id: SourceId; title: string; facts: Fact[] };

export const SOURCES: Source[] = [
  { id: "S1", title: "S1 · Observatory history page", facts: [
    { id: "opened", source: "S1", attr: "opened", value: 1962, exact: true, text: "Opened to researchers in 1962." },
    { id: "founder", source: "S1", attr: "founder", value: "Arlo Science Trust", text: "Founded by the Arlo Science Trust." },
  ] },
  { id: "S2", title: "S2 · Instrument fact sheet", facts: [
    { id: "mirror", source: "S2", attr: "mirror", value: 2.4, text: "Main mirror: 2.4 m in diameter." },
    { id: "spectro", source: "S2", attr: "spectrograph", value: 1998, exact: true, text: "Spectrograph installed in 1998." },
  ] },
  { id: "S3", title: "S3 · Site survey 2023", facts: [
    { id: "alt", source: "S3", attr: "altitude", value: 2150, text: "Altitude: 2,150 m above sea level." },
    { id: "nights", source: "S3", attr: "clearNights", value: 210, text: "About 210 clear nights per year." },
    { id: "winter", source: "S3", attr: "winterClosure", value: "closed", quantifier: "some", text: "Some winter nights close due to snow." },
  ] },
];
export const SOURCE_IDS = SOURCES.map((s) => s.id) as string[];

export type Claim = { id: string; text: string; span: string; attr: string; value: string | number; cite?: string; quantifier?: "some" | "all" };
export type Answer = { id: string; name: string; parts: (string | Claim)[] };

const c = (id: string, span: string, text: string, attr: string, value: string | number, cite?: string, quantifier?: "some" | "all"): Claim => ({ id, span, text, attr, value, cite, quantifier });

export const ANSWERS: Answer[] = [
  { id: "grounded", name: "Grounded answer", parts: [
    "Mount Arlo Observatory ", c("g1", "opened in 1962 [S1]", "The observatory opened in 1962", "opened", 1962, "S1"),
    " and ", c("g2", "was founded by the Arlo Science Trust [S1]", "It was founded by the Arlo Science Trust", "founder", "Arlo Science Trust", "S1"),
    ". ", c("g3", "Its main mirror is 2.4 m across [S2]", "Its main mirror is 2.4 m across", "mirror", 2.4, "S2"),
    ", and ", c("g4", "it sits at roughly 2,100 m [S3]", "It sits at about 2,100 m altitude", "altitude", 2100, "S3"),
    ". ", c("g5", "It gets about 210 clear nights a year [S3]", "It gets about 210 clear nights a year", "clearNights", 210, "S3"), ".",
  ] },
  { id: "confident", name: "Confident but wrong", parts: [
    "Mount Arlo Observatory ", c("w1", "opened in 1962 [S1]", "The observatory opened in 1962", "opened", 1962, "S1"),
    " and ", c("w2", "was founded by the Lumen Institute [S1]", "It was founded by the Lumen Institute", "founder", "Lumen Institute", "S1"),
    ". ", c("w3", "Its 3.6 m main mirror [S2]", "Its main mirror is 3.6 m across", "mirror", 3.6, "S2"),
    " ", c("w4", "sits at 2,150 m [S3]", "It sits at 2,150 m altitude", "altitude", 2150, "S3"),
    " and ", c("w5", "enjoys some 260 clear nights a year", "It gets about 260 clear nights a year", "clearNights", 260),
    ". ", c("w6", "Its spectrograph dates from 1989 [S2]", "Its spectrograph was installed in 1989", "spectrograph", 1989, "S2"), ".",
  ] },
  { id: "citations", name: "Citation-happy", parts: [
    "Mount Arlo Observatory ", c("h1", "opened in 1962 [S1]", "The observatory opened in 1962", "opened", 1962, "S1"),
    ". ", c("h2", "A 2019 upgrade doubled its resolution [S5]", "A 2019 upgrade doubled its resolution", "upgrade", "2019 upgrade", "S5"),
    ". ", c("h3", "It is always closed in winter [S3]", "It is closed on all winter nights", "winterClosure", "closed", "S3", "all"),
    ", but ", c("h4", "its spectrograph (1998) [S2]", "Its spectrograph was installed in 1998", "spectrograph", 1998, "S2"),
    " ", c("h5", "makes it the most visited observatory in the region [S4]", "It is the most visited observatory in the region", "visitors", "most visited", "S4"), ".",
  ] },
  { id: "subtle", name: "Subtle drift", parts: [
    c("s1", "Founded by the Arlo Science Trust", "It was founded by the Arlo Science Trust", "founder", "Arlo Science Trust", "S1"),
    ", the observatory ", c("s2", "opened in 1962 [S1]", "The observatory opened in 1962", "opened", 1962, "S1"),
    ". ", c("s3", "Its mirror measures 2.5 m [S2]", "Its main mirror is 2.5 m across", "mirror", 2.5, "S2"),
    " and ", c("s4", "it stands at 1,250 m [S3]", "It sits at 1,250 m altitude", "altitude", 1250, "S3"),
    ". ", c("s5", "Some winter nights are lost to snow [S3]", "Some winter nights close due to snow", "winterClosure", "closed", "S3", "some"), ".",
  ] },
];

export const claimsOf = (a: Answer) => a.parts.filter((p): p is Claim => typeof p !== "string");

export type Verdict = "supported" | "contradicted" | "unverifiable";
export type HType = "fabricated citation" | "entity swap" | "number drift" | "unsupported generalization";
export type Check = { claim: Claim; verdict: Verdict; type: HType | null; fact?: Fact; relDiff?: number; reason: string };
export type CheckConfig = { tolerance: number; missing: SourceId | "none"; countUnverifiable: boolean };
export const DEFAULT_CONFIG: CheckConfig = { tolerance: 0.05, missing: "none", countUnverifiable: true };

/** The rule-based checker: citation exists? → matching fact? → compare value (exact, tolerance, quantifier, entity). */
export function checkClaim(claim: Claim, cfg: CheckConfig): Check {
  const tol = Math.max(0, Math.min(1, cfg.tolerance));
  const sources = SOURCES.filter((s) => s.id !== cfg.missing);
  if (claim.cite && !SOURCE_IDS.includes(claim.cite))
    return { claim, verdict: "unverifiable", type: "fabricated citation", reason: `Cites [${claim.cite}], which does not exist in the retrieved set.` };
  const fact = sources.flatMap((s) => s.facts).find((f) => f.attr === claim.attr);
  if (!fact) return { claim, verdict: "unverifiable", type: null, reason: claim.cite === cfg.missing ? `Its source [${claim.cite}] was not retrieved, so nothing can confirm it.` : "No retrieved source says anything about this." };
  if (typeof claim.value === "number" && typeof fact.value === "number") {
    const relDiff = Math.abs(claim.value - fact.value) / Math.abs(fact.value);
    const ok = fact.exact ? claim.value === fact.value : relDiff <= tol + 1e-12;
    return ok
      ? { claim, verdict: "supported", type: null, fact, relDiff, reason: relDiff === 0 ? "Value matches the source exactly." : `Off by ${(relDiff * 100).toFixed(1)}%, within the ${(tol * 100).toFixed(0)}% tolerance.` }
      : { claim, verdict: "contradicted", type: "number drift", fact, relDiff, reason: fact.exact ? `A year must match exactly: ${claim.value} ≠ ${fact.value}.` : `Off by ${(relDiff * 100).toFixed(1)}%, more than the ${(tol * 100).toFixed(0)}% tolerance.` };
  }
  if (claim.quantifier === "all" && fact.quantifier === "some")
    return { claim, verdict: "unverifiable", type: "unsupported generalization", fact, reason: "The source says some, the claim says all: the source cannot back the stronger statement." };
  return claim.value === fact.value
    ? { claim, verdict: "supported", type: null, fact, reason: "The source states the same thing." }
    : { claim, verdict: "contradicted", type: "entity swap", fact, reason: `The source names "${fact.value}", not "${claim.value}".` };
}

export function summarize(checks: Check[], countUnverifiable: boolean) {
  const n = checks.length;
  const count = (v: Verdict) => checks.filter((c) => c.verdict === v).length;
  const supported = count("supported"), contradicted = count("contradicted"), unverifiable = count("unverifiable");
  const hallucinated = contradicted + (countUnverifiable ? unverifiable : 0);
  const byType: Partial<Record<HType, number>> = {};
  for (const ch of checks) if (ch.type) byType[ch.type] = (byType[ch.type] ?? 0) + 1;
  return { n, supported, contradicted, unverifiable, hallucinated, rate: n ? hallucinated / n : 0, byType };
}

export type HState = { phase: "answer" | "decompose" | "check" | "rate"; current: number | null; checked: number };

export function simulate(answer: Answer, cfg: CheckConfig) {
  const claims = claimsOf(answer);
  const checks = claims.map((cl) => checkClaim(cl, cfg));
  const summary = summarize(checks, cfg.countUnverifiable);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "check", message, severity, tex });
  const steps: SimulationStep<HState>[] = [
    { id: "answer", title: "The answer arrives", description: "A fluent answer with inline citations. Fluency says nothing about whether each statement is true.",
      state: { phase: "answer", current: null, checked: 0 }, events: [ev("a", `${claims.length} factual statements, ${claims.filter((x) => x.cite).length} with citations`)] },
    { id: "decompose", title: "Split into atomic claims", description: "Each sentence is broken into claims that are individually true or false, each with one attribute and one value.",
      state: { phase: "decompose", current: null, checked: 0 }, events: claims.map((cl, i) => ev(`d${i}`, `C${i + 1}: ${cl.attr} = ${cl.value}${cl.cite ? ` [${cl.cite}]` : ""}`)) },
    ...checks.map((ch, i): SimulationStep<HState> => ({
      id: `check-${i}`, title: `Check claim C${i + 1}`, description: `"${ch.claim.text}" → ${ch.verdict}${ch.type ? ` (${ch.type})` : ""}. ${ch.reason}`,
      state: { phase: "check", current: i, checked: i + 1 },
      events: [ev(`c${i}`, `C${i + 1} ${ch.verdict}${ch.type ? `: ${ch.type}` : ""}. ${ch.reason}`, ch.verdict === "contradicted" ? "error" : ch.verdict === "unverifiable" ? "warning" : undefined,
        ch.relDiff !== undefined ? `\\frac{|${ch.claim.value} - ${ch.fact!.value}|}{${ch.fact!.value}} = ${ch.relDiff.toFixed(3)}` : undefined)],
    })),
    { id: "rate", title: "Hallucination rate", description: `${summary.hallucinated} of ${summary.n} claims count as hallucinated → ${(summary.rate * 100).toFixed(0)}%.`,
      state: { phase: "rate", current: null, checked: checks.length },
      events: [ev("r", `rate = ${summary.hallucinated}/${summary.n} = ${summary.rate.toFixed(3)}`, summary.rate > 0 ? "warning" : undefined,
        `\\text{rate} = \\frac{${summary.contradicted}${cfg.countUnverifiable ? ` + ${summary.unverifiable}` : ""}}{${summary.n}} = ${summary.rate.toFixed(3)}`)] },
  ];
  return { claims, checks, summary, steps };
}
