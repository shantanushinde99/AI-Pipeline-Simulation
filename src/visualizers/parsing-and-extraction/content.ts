import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Watch three PDF parsers read a two-column page with a table, then diff their text and count surviving facts.", minutes: 12, hasMetrics: false };

const t = "parsing-and-extraction";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Parsing turns a file into text a chunker can use. For HTML or Markdown the structure is explicit. For PDFs it isn't: a page is a list of glyph runs drawn at x/y coordinates. The parser has to infer reading order (columns), recognise page furniture (headers, footers, page numbers) and rebuild tables from positioned cells." },
    { heading: "Three strategies", body: "Naive text extraction sorts lines by height and reads straight across the page, so two-column layouts come out interleaved and tables drawn as graphics are lost. Layout-aware parsers find columns from whitespace gutters and drop repeated furniture, but still emit table cells as loose words. Table-aware parsers detect the grid and write each row as a sentence that repeats its column headers, so values stay attached to their meaning." },
    { heading: "How to measure it", body: "Keep a small set of reference pages with hand-checked text. Diff the parser output word by word (longest common subsequence) for order and noise, and check a list of facts that must appear contiguously. Facts are what retrieval needs: a fact split by an interleaved line can't be matched or quoted." },
    { heading: "When it matters", body: "Any corpus of PDFs, slides or scanned forms: contracts, manuals, research papers, financial statements. Parser regressions are silent: retrieval scores look plausible while specific numbers (prices, limits, dates) simply aren't in the index." },
    { heading: "Assumptions in this visualizer", body: "The page is synthetic: text boxes with hand-set coordinates rather than a real PDF. The three parsers are simplified rule-based models of real behaviour (sort-by-position, gutter-based column detection with a 12 px threshold, header-repeating table linearisation). Real tools (pdfminer, PyMuPDF, Unstructured, Docling, cloud layout models) use richer heuristics or learned layout models, and OCR for scanned pages." },
  ],
  code: [
    {
      title: "layout_parse.py: columns, furniture and tables (PyMuPDF)",
      source: `import fitz  # PyMuPDF

doc = fitz.open("admin-guide.pdf")
page = doc[11]

# 1. Text blocks with coordinates; sort=True gives a column-aware reading order.
blocks = page.get_text("blocks", sort=True)
body = [b for b in blocks if 40 < b[1] < page.rect.height - 40]   # drop header/footer bands

# 2. Tables: detect the grid, then linearize each row with its headers.
for table in page.find_tables():
    header, *rows = table.extract()
    for row in rows:
        print(f"{row[0]} plan: " + "; ".join(f"{h} {v}" for h, v in zip(header[1:], row[1:])) + ".")`,
    },
    {
      title: "Regression check: facts must survive parsing",
      source: `FACTS = ["sso is available on the enterprise plan only",
         "pro plan requests min 600"]

def norm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())

def test_parser_keeps_facts():
    text = norm(parse("fixtures/admin-guide-p12.pdf"))
    missing = [f for f in FACTS if f" {f} " not in f" {text} "]
    assert not missing, f"parser lost: {missing}"`,
    },
  ],
  failureModes: [
    { title: "Tables dropped or flattened", symptom: "Questions about prices or limits get 'I don't know' or a wrong number.", detect: "Grep the parsed text for a known table value; check that it sits next to its row and column labels.", fix: "Use a table-aware parser and linearize rows with headers (or keep tables as Markdown).", scenarioId: "rag-parser-table" },
    { title: "Interleaved columns", symptom: "Chunks read like nonsense: halves of sentences from two different topics.", detect: "Diff against a reference page; low word recall with many out-of-order words.", fix: "Layout-aware parsing; verify column detection on narrow-gutter templates." },
    { title: "Page furniture in every chunk", symptom: "'Confidential · Page 12 of 40' appears in retrieved passages and wastes context tokens.", detect: "Count lines repeated across most pages of a document.", fix: "Strip repeated header/footer bands before chunking." },
    { title: "Silent parser regression", symptom: "After a library upgrade, answer accuracy drops for one document family only.", detect: "Run a golden set of facts against parsed text in CI.", fix: "Pin parser versions; gate upgrades on the fact-survival test." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "With the naive parser, the text reads 'SSO is available on the Request logs are retained for Enterprise plan only'. What happened?", options: ["The embedding model reordered words", "Lines at the same height in the two columns were read straight across and glued together", "The PDF is corrupt", "The chunker split the sentence"], answer: 1, explanation: "Naive extraction sorts text runs by vertical position, then left to right. In a two-column layout that alternates between columns line by line." },
    { id: `${t}-2`, topicId: t, skill: "prediction", question: "Switch to layout-aware, then shrink the column gutter to 8 px. What happens to facts recoverable?", options: ["Stays 5/6", "Drops: the gutter is below the detection threshold, so it falls back to line order", "Rises to 6/6", "The table disappears"], answer: 1, explanation: "Column detection relies on a vertical whitespace band. Below 12 px the parser sees one wide column and interleaves lines like the naive parser (though it still removes the header and footer)." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "Layout-aware output contains 'Pro 600 10'. Why does the fact 'Pro plan: 600 requests/min' still count as missing?", options: ["600 is not in the text", "The value is no longer attached to its column header, so 'Pro plan requests min 600' never appears together", "Numbers are ignored", "The fact list is wrong"], answer: 1, explanation: "Flattened cells lose their meaning: a retriever or model can't tell whether 600 is seats, requests or dollars." },
    { id: `${t}-4`, topicId: t, skill: "fix", question: "Which change takes the page from 5/6 to 6/6 facts?", options: ["Bigger chunks", "A table-aware parser that writes each row with its headers", "A better embedding model", "Removing the footer"], answer: 1, explanation: "Only the table fact was missing; linearizing rows as 'Pro plan: Requests/min 600; Seats 10.' restores it." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "What is the most reliable way to catch a parser regression before it reaches users?", options: ["Watch average retrieval similarity", "A CI test that parses reference pages and asserts known facts appear contiguously", "Manual spot checks once a quarter", "Increase top-k"], answer: 1, explanation: "Similarity scores stay plausible even when facts are lost. A fact-survival test fails exactly when the text needed for an answer is gone." },
  ],
};
