import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Strategy = "naive" | "layout" | "table";
export type UnitKind = "header" | "footer" | "h1" | "intro" | "h2" | "line" | "caption" | "cell";
export type Unit = { id: string; kind: UnitKind; text: string; x: number; y: number; w: number; h: number; col: "full" | "left" | "right"; row?: number; c?: number };

export const PAGE = { w: 400, h: 392 };
export const MIN_GUTTER = 12;
export const STRATEGIES: Record<Strategy, { name: string; short: string }> = {
  naive: { name: "Naive text extraction", short: "Reads every text line left to right, top to bottom, across the whole page width." },
  layout: { name: "Layout-aware", short: "Finds columns from the whitespace gutter, reads each column top to bottom, drops repeated page furniture." },
  table: { name: "Layout + table-aware", short: "Layout-aware, plus detects the table grid and writes each row as a sentence with its column headers." },
};

const LEFT = ["SSO is available on the", "Enterprise plan only. To enable", "it, an admin uploads the", "identity provider metadata XML", "file under Settings, Security."];
const RIGHT = ["Request logs are retained for", "30 days on Starter and Pro", "and 365 days on Enterprise.", "Deleted projects can be", "recovered for 7 days."];
const TABLE = { caption: "Table 4.1 API rate limits", headers: ["Plan", "Requests/min", "Seats"], rows: [["Starter", "60", "1"], ["Pro", "600", "10"], ["Enterprise", "custom", "unlimited"]] };

/** A fictional admin-guide page: header, title, intro, two columns with sub-headings, a full-width table, footer. */
export function buildPage(gutterIn = 24): Unit[] {
  const gutter = Math.max(2, Math.min(40, Math.round(gutterIn)));
  const m = 24, colW = (PAGE.w - 2 * m - gutter) / 2, rx = m + colW + gutter;
  const units: Unit[] = [
    { id: "header", kind: "header", text: "Northwind Cloud Admin Guide", x: m, y: 14, w: PAGE.w - 2 * m, h: 14, col: "full" },
    { id: "h1", kind: "h1", text: "4. Security and limits", x: m, y: 44, w: PAGE.w - 2 * m, h: 22, col: "full" },
    { id: "intro", kind: "intro", text: "This chapter applies to all workspaces created after 2025.", x: m, y: 74, w: PAGE.w - 2 * m, h: 14, col: "full" },
    { id: "h2-l", kind: "h2", text: "4.1 Single sign-on", x: m, y: 104, w: colW, h: 16, col: "left" },
    { id: "h2-r", kind: "h2", text: "4.2 Data retention", x: rx, y: 104, w: colW, h: 16, col: "right" },
  ];
  LEFT.forEach((t, i) => units.push({ id: `l${i}`, kind: "line", text: t, x: m, y: 128 + i * 18, w: colW, h: 14, col: "left" }));
  RIGHT.forEach((t, i) => units.push({ id: `r${i}`, kind: "line", text: t, x: rx, y: 128 + i * 18, w: colW, h: 14, col: "right" }));
  const ty = 240, cw = [140, 110, 102];
  units.push({ id: "caption", kind: "caption", text: TABLE.caption, x: m, y: ty, w: PAGE.w - 2 * m, h: 14, col: "full" });
  [TABLE.headers, ...TABLE.rows].forEach((r, ri) =>
    r.forEach((t, ci) => units.push({ id: `t${ri}-${ci}`, kind: "cell", text: t, x: m + cw.slice(0, ci).reduce((s, v) => s + v, 0), y: ty + 20 + ri * 22, w: cw[ci], h: 22, col: "full", row: ri, c: ci })),
  );
  units.push({ id: "footer", kind: "footer", text: "Confidential · Page 12 of 40", x: m, y: PAGE.h - 28, w: PAGE.w - 2 * m, h: 14, col: "full" });
  return units;
}

export const gutterOf = (units: Unit[]) => {
  const l = units.find((u) => u.id === "l0")!, r = units.find((u) => u.id === "r0")!;
  return r.x - (l.x + l.w);
};

export type Segment = { units: string[]; text: string; kind: UnitKind | "row" };
export type ParseResult = { segments: Segment[]; text: string; dropped: string[]; furniture: string[]; columnsDetected: boolean };

const rowSentence = (cells: string[]) => `${cells[0]} plan: ${TABLE.headers.slice(1).map((h, i) => `${h} ${cells[i + 1]}`).join("; ")}.`;

export function parse(units: Unit[], strategy: Strategy): ParseResult {
  const cells = units.filter((u) => u.kind === "cell");
  const textUnits = units.filter((u) => u.kind !== "cell");
  const seg = (u: Unit): Segment => ({ units: [u.id], text: u.kind === "h1" || u.kind === "h2" || u.kind === "caption" ? `${u.text}.` : u.text, kind: u.kind });
  if (strategy === "naive") {
    // Text-layer order: sort every line by its baseline, then left to right. Table cells are drawn as
    // positioned glyphs in a grid this extractor doesn't read, so they are lost.
    const ordered = [...textUnits].sort((a, b) => a.y - b.y || a.x - b.x);
    const segments = ordered.map(seg);
    return { segments, text: segments.map((s) => s.text).join(" "), dropped: cells.map((c) => c.id), furniture: [], columnsDetected: false };
  }
  const furniture = units.filter((u) => u.kind === "header" || u.kind === "footer").map((u) => u.id);
  const body = textUnits.filter((u) => !furniture.includes(u.id));
  const columnsDetected = gutterOf(units) >= MIN_GUTTER;
  const colRank = (u: Unit) => (u.col === "right" ? 2 : u.col === "left" ? 1 : 0);
  const columnBand = (u: Unit) => u.col !== "full";
  const full = (u: Unit) => !columnBand(u);
  // Full-width blocks above the columns, then left column, then right column, then blocks below.
  const colTop = Math.min(...body.filter(columnBand).map((u) => u.y));
  const above = body.filter((u) => full(u) && u.y < colTop).sort((a, b) => a.y - b.y);
  const below = body.filter((u) => full(u) && u.y > colTop).sort((a, b) => a.y - b.y);
  const inCols = body.filter(columnBand).sort(columnsDetected ? (a, b) => colRank(a) - colRank(b) || a.y - b.y : (a, b) => a.y - b.y || a.x - b.x);
  const segments = [...above, ...inCols, ...below].map(seg);
  if (strategy === "layout") {
    for (const c of [...cells].sort((a, b) => a.y - b.y || a.x - b.x)) segments.push({ units: [c.id], text: c.text, kind: "cell" });
  } else {
    const rows = [...new Set(cells.map((c) => c.row!))].filter((r) => r > 0);
    for (const r of rows) {
      const rc = cells.filter((c) => c.row === r).sort((a, b) => a.c! - b.c!);
      segments.push({ units: [...cells.filter((c) => c.row === 0).map((c) => c.id), ...rc.map((c) => c.id)], text: rowSentence(rc.map((c) => c.text)), kind: "row" });
    }
  }
  return { segments, text: segments.map((s) => s.text).join(" "), dropped: [], furniture, columnsDetected };
}

/** What a careful human would type: correct reading order, no page furniture, table rows as sentences. */
export function referenceText(): string {
  return parse(buildPage(24), "table").text;
}

export const FACTS = [
  { id: "f1", label: "Scope: all workspaces created after 2025", phrase: "applies to all workspaces created after 2025" },
  { id: "f2", label: "SSO is Enterprise-only", phrase: "sso is available on the enterprise plan only" },
  { id: "f3", label: "SSO needs the IdP metadata XML file", phrase: "identity provider metadata xml file" },
  { id: "f4", label: "Logs kept 30 days on Starter and Pro", phrase: "request logs are retained for 30 days on starter and pro" },
  { id: "f5", label: "Deleted projects recoverable for 7 days", phrase: "deleted projects can be recovered for 7 days" },
  { id: "f6", label: "Pro plan: 600 requests/min", phrase: "pro plan requests min 600" },
];

export const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
export const factFound = (text: string, phrase: string) => ` ${normalize(text)} `.includes(` ${phrase} `);

export type DiffOp = { op: "same" | "extra" | "missing"; word: string };
/** Word-level LCS diff of extracted text against the reference. */
export function diffWords(extracted: string, reference: string): DiffOp[] {
  const a = extracted.split(/\s+/).filter(Boolean), b = reference.split(/\s+/).filter(Boolean);
  const na = a.map(normalize), nb = b.map(normalize);
  const L = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--) L[i][j] = na[i] === nb[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: DiffOp[] = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && na[i] === nb[j]) { out.push({ op: "same", word: a[i] }); i++; j++; }
    else if (j < b.length && (i >= a.length || L[i][j + 1] >= L[i + 1][j])) { out.push({ op: "missing", word: b[j] }); j++; }
    else { out.push({ op: "extra", word: a[i] }); i++; }
  }
  return out;
}

export type Score = { facts: { id: string; label: string; found: boolean }[]; recovered: number; wordRecall: number; extraWords: number; refWords: number; same: number };
export function score(text: string): Score {
  const ref = referenceText();
  const d = diffWords(text, ref);
  const same = d.filter((x) => x.op === "same").length;
  const refWords = ref.split(/\s+/).filter(Boolean).length;
  const facts = FACTS.map((f) => ({ id: f.id, label: f.label, found: factFound(text, f.phrase) }));
  return { facts, recovered: facts.filter((f) => f.found).length, wordRecall: same / refWords, extraWords: d.filter((x) => x.op === "extra").length, refWords, same };
}

export type ParseState = { phase: "page" | "read" | "diff" | "facts"; /** segments revealed after this step */ upto: number; /** first segment index animated in this step */ from: number };
export const READ_STEPS = 5;

export function simulate(strategy: Strategy, gutter = 24) {
  const units = buildPage(gutter);
  const result = parse(units, strategy);
  const sc = score(result.text);
  const n = result.segments.length;
  const per = Math.ceil(n / READ_STEPS);
  let e = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id: `e${e++}`, type: "log", message, severity, tex });
  const steps: SimulationStep<ParseState>[] = [
    {
      id: "page", title: "The page as the parser sees it", state: { phase: "page", upto: 0, from: 0 },
      description: "A PDF page is a set of positioned text runs, not a document tree. Header, title, two columns, a table and a footer are just boxes at x/y coordinates.",
      events: [ev(`${units.length} text boxes on the page; column gutter ${gutterOf(units).toFixed(0)} px`)],
    },
  ];
  for (let s = 0; s < READ_STEPS; s++) {
    const from = s * per, upto = Math.min(n, (s + 1) * per);
    if (from >= n) break;
    const segs = result.segments.slice(from, upto);
    const events = segs.map((g) => ev(`emit ${g.kind}: “${g.text.length > 48 ? `${g.text.slice(0, 48)}…` : g.text}”`));
    if (s === 0 && strategy !== "naive") events.unshift(ev(`header and footer detected as page furniture and skipped`));
    if (s === 0 && strategy !== "naive" && !result.columnsDetected) events.unshift(ev(`gutter ${gutterOf(units).toFixed(0)} px < ${MIN_GUTTER} px: columns not detected, falling back to line order`, "warning"));
    if (s === 0 && strategy === "naive") events.unshift(ev(`table grid not in text layer: ${result.dropped.length} cells skipped`, "warning"));
    steps.push({
      id: `read-${s}`, title: `Read segments ${from + 1}–${upto}`, state: { phase: "read", upto, from }, events,
      description: strategy === "naive"
        ? "The naive extractor walks lines by vertical position. Two lines at the same height, one in each column, are glued together."
        : strategy === "layout"
          ? "Columns are read one after the other. Table cells come out one by one in grid order, without their row/column meaning."
          : "Columns are read one after the other; each table row becomes one sentence that repeats the column headers.",
    });
  }
  steps.push({
    id: "diff", title: "Diff against the reference text", state: { phase: "diff", upto: n, from: n },
    description: "The extracted text is compared word by word (longest common subsequence) with a hand-made reference: correct reading order, no furniture, table rows as sentences.",
    events: [ev(`${sc.same}/${sc.refWords} reference words matched in order; ${sc.extraWords} extra words`, sc.wordRecall < 0.9 ? "warning" : "info", `\\text{word recall} = ${sc.same}/${sc.refWords} = ${sc.wordRecall.toFixed(2)}`)],
  });
  steps.push({
    id: "facts", title: "Which facts survive?", state: { phase: "facts", upto: n, from: n },
    description: "A fact is recoverable only if its words appear together, in order, in the extracted text. That is what a chunk needs for retrieval to find it and for the model to quote it.",
    events: sc.facts.map((f) => ev(`${f.found ? "found" : "MISSING"}: ${f.label}`, f.found ? "info" : "warning")),
  });
  return { steps, units, result, score: sc };
}
