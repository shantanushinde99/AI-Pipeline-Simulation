import { useCallback, useMemo, useState } from "react";
import { motion } from "motion/react";
import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../document-ingestion/kit";
import { MIN_GUTTER, PAGE, STRATEGIES, buildPage, diffWords, parse, referenceText, score, simulate, type ParseResult, type ParseState, type Score, type Strategy, type Unit } from "./simulation";

// One schedule: segment i of the current step lights up, the cursor arrives and its text is appended at segAt(i).
const SEG = 0.42;
const segAt = (i: number) => 0.3 + i * SEG;

const KIND_STYLE: Record<string, { fill: string; stroke: string; label: string }> = {
  header: { fill: "var(--surface-2)", stroke: "var(--muted)", label: "header" },
  footer: { fill: "var(--surface-2)", stroke: "var(--muted)", label: "footer" },
  h1: { fill: "var(--violet)", stroke: "var(--violet)", label: "title" },
  h2: { fill: "var(--violet)", stroke: "var(--violet)", label: "sub-heading" },
  intro: { fill: "var(--fg)", stroke: "var(--line)", label: "paragraph" },
  line: { fill: "var(--fg)", stroke: "var(--line)", label: "text line" },
  caption: { fill: "var(--warn)", stroke: "var(--warn)", label: "table caption" },
  cell: { fill: "var(--warn)", stroke: "var(--warn)", label: "table cell" },
};

function narrate(s: ParseState, strategy: Strategy, r: ParseResult, sc: Score) {
  if (s.phase === "page") return { headline: "A PDF page is just boxes of text at x/y positions", sub: "There is no 'paragraph' or 'table' in the file. The parser has to infer reading order and structure." };
  if (s.phase === "diff") return { headline: `${sc.same} of ${sc.refWords} reference words come out in the right order`, sub: sc.extraWords ? `${sc.extraWords} extra or misplaced words: furniture, interleaved lines or flattened cells.` : "The extracted text matches the reference exactly." };
  if (s.phase === "facts") return { headline: `${sc.recovered} of ${sc.facts.length} facts are recoverable from this text`, sub: sc.recovered < sc.facts.length ? "A fact split by interleaved lines or a lost table can't be retrieved or quoted, however good the embeddings are." : "Every fact sits together, in order, ready to be chunked." };
  if (strategy === "naive") return { headline: "Naive extraction reads straight across both columns", sub: "Each line on the left is glued to the line at the same height on the right. Watch the cursor zig-zag." };
  if (!r.columnsDetected) return { headline: "The gutter is too narrow: columns are not detected", sub: `Below ${MIN_GUTTER} px of whitespace the layout parser sees one wide column and falls back to line order.` };
  return strategy === "layout"
    ? { headline: "Layout-aware: down the left column, then down the right", sub: "Header and footer are recognized as page furniture and skipped. Table cells still come out as loose words." }
    : { headline: "Table-aware: columns in order, table rows as sentences", sub: "Each row is written with its column headers, so '600' stays attached to 'Pro' and 'Requests/min'." };
}

export default function ParsingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [strategy, setStrategy] = useState<Strategy>("naive");
  const [gutter, setGutter] = useState(24);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(strategy, gutter), [strategy, gutter]);
  const duration = useCallback((i: number) => {
    const s = sim.steps[i]?.state;
    if (!s) return 3000;
    if (s.phase === "read") return (segAt(s.upto - s.from) + 1.2) * 1000;
    return s.phase === "page" ? 2600 : 4200;
  }, [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${strategy}-${gutter}-${step.id}-${replay}`;
  const story = narrate(step.state, strategy, sim.result, sim.score);
  const comparison = useMemo(() => (["naive", "layout", "table"] as Strategy[]).map((s) => ({ s, sc: score(parse(buildPage(gutter), s).text) })), [gutter]);

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title={`Page 12 · ${STRATEGIES[strategy].name}`} actions={<SimulatedLabel>Synthetic page</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={`${strategy}-${step.id}`} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 md:grid-cols-[minmax(0,400px)_minmax(0,1fr)]">
              <PageSvg units={sim.units} result={sim.result} state={step.state} k={k} animKey={animKey} />
              <ExtractedText result={sim.result} state={step.state} k={k} animKey={animKey} />
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
              <StageDetail state={step.state} result={sim.result} sc={sim.score} comparison={comparison} k={k} strategy={strategy} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parser settings" actions={<Button size="sm" variant="ghost" onClick={() => { setStrategy("naive"); setGutter(24); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Parser strategy" value={strategy} onChange={setStrategy} options={(Object.keys(STRATEGIES) as Strategy[]).map((s) => ({ value: s, label: STRATEGIES[s].name }))} />
              <Slider label="Column gutter (whitespace between columns)" value={gutter} min={4} max={40} onChange={setGutter} format={(v) => `${v} px`} hint={`Layout detection needs a gutter of at least ${MIN_GUTTER} px. Shrink it to break the layout-aware parsers.`} />
            </div>
            <p className="mt-2 text-xs text-muted">{STRATEGIES[strategy].short}</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why="Chunking and embedding only ever see this text. Order and structure lost here can't be recovered downstream." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`\\text{facts} = \\frac{|\\{f \\in F : f \\subseteq \\text{text}\\}|}{|F|} = \\frac{${sim.score.recovered}}{${sim.score.facts.length}}`}</TexBlock>
              <TexBlock>{`\\text{word recall} = \\frac{|\\mathrm{LCS}(\\text{text}, \\text{ref})|}{|\\text{ref}|} = \\frac{${sim.score.same}}{${sim.score.refWords}} = ${sim.score.wordRecall.toFixed(2)}`}</TexBlock>
            </div>
            <p className="mt-2 text-[11px] text-muted">f ⊆ text means the fact's words appear contiguously, in order, after lower-casing and stripping punctuation.</p>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["segments emitted", sim.result.segments.length],
              ["columns detected", sim.result.columnsDetected ? "yes" : "no"],
              ["furniture removed", sim.result.furniture.length],
              ["cells dropped", sim.result.dropped.length],
              ["extra words", sim.score.extraWords],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const anchor = (u: Unit): [number, number] => [u.x + 9, u.y + u.h / 2];

function PageSvg({ units, result, state, k, animKey }: { units: Unit[]; result: ParseResult; state: ParseState; k: number; animKey: string }) {
  const segOf = new Map<string, number>();
  result.segments.forEach((s, i) => s.units.forEach((u) => !segOf.has(u) && segOf.set(u, i)));
  const byId = new Map(units.map((u) => [u.id, u]));
  const anchors = result.segments.map((s) => anchor(byId.get(s.units[s.units.length > 3 ? 3 : 0])!));
  const reading = state.phase !== "page";
  const prevPath = anchors.slice(0, Math.max(1, state.from)).map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
  const cur = state.phase === "read" ? anchors.slice(Math.max(0, state.from - 1), state.upto) : [];
  return (
    <svg viewBox={`0 0 ${PAGE.w} ${PAGE.h}`} className="h-auto w-full rounded-md border border-line bg-surface" role="img" aria-label="Page layout with header, two text columns, a table and a footer">
      {units.map((u) => {
        const st = KIND_STYLE[u.kind];
        const si = segOf.get(u.id);
        const done = si !== undefined && si < state.upto;
        const now = si !== undefined && si >= state.from && si < state.upto && state.phase === "read";
        const dropped = reading && result.dropped.includes(u.id);
        const furniture = reading && result.furniture.includes(u.id);
        const fs = u.kind === "h1" ? 13 : u.kind === "h2" ? 10.5 : 9;
        return (
          <motion.g key={`${animKey}-${u.id}`} initial={state.phase === "page" ? { opacity: 0, y: 6 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: state.phase === "page" ? (u.y / PAGE.h) * 1.4 * k : 0, duration: 0.3 * k }}>
            <motion.rect x={u.x} y={u.y} width={u.w} height={u.h} rx={2} stroke={st.stroke} strokeOpacity={0.5} strokeDasharray={u.kind === "header" || u.kind === "footer" ? "3 2" : undefined}
              initial={{ fill: now ? "var(--surface)" : done ? "var(--accent-soft)" : "var(--surface)" }}
              animate={{ fill: done ? "var(--accent-soft)" : "var(--surface)" }}
              transition={{ delay: now ? segAt(si! - state.from) * k : 0, duration: 0.25 * k }} />
            <text x={u.x + (u.kind === "cell" ? 5 : 18)} y={u.y + u.h / 2 + fs / 2.8} fontSize={fs} fontWeight={u.kind === "h1" || u.kind === "h2" || (u.kind === "cell" && u.row === 0) ? 700 : 400}
              fill={u.kind === "h1" || u.kind === "h2" ? "var(--violet)" : u.kind === "header" || u.kind === "footer" ? "var(--muted)" : "var(--fg)"}>{u.text}</text>
            {dropped && <rect x={u.x} y={u.y} width={u.w} height={u.h} fill="var(--err)" fillOpacity={0.12} stroke="var(--err)" strokeDasharray="4 2" />}
            {furniture && <rect x={u.x} y={u.y} width={u.w} height={u.h} fill="var(--muted)" fillOpacity={0.15} />}
          </motion.g>
        );
      })}
      {reading && result.dropped.length > 0 && (
        <g>
          <rect x={PAGE.w - 186} y={238} width={162} height={16} rx={8} fill="var(--err)" />
          <text x={PAGE.w - 105} y={249.5} textAnchor="middle" fontSize={9.5} fill="var(--surface)">✗ table not in text layer: skipped</text>
        </g>
      )}
      {reading && result.furniture.length > 0 && (
        <text x={PAGE.w - 24} y={24} textAnchor="end" fontSize={9} fill="var(--muted)">⊘ furniture: removed</text>
      )}
      {/* Order badges for segments already emitted. */}
      {result.segments.slice(0, state.upto).map((_s, i) => {
        const [x, y] = anchors[i];
        const now = state.phase === "read" && i >= state.from;
        return (
          <motion.g key={`${animKey}-b${i}`} initial={now ? { scale: 0, opacity: 0 } : false} animate={{ scale: 1, opacity: 1 }} style={{ transformBox: "fill-box", transformOrigin: "center" }}
            transition={{ delay: now ? segAt(i - state.from) * k : 0, type: "spring", stiffness: 320, damping: 18 }}>
            <circle cx={x} cy={y} r={7} fill={now ? "var(--accent)" : "var(--accent-soft)"} stroke="var(--accent)" />
            <text x={x} y={y + 3} textAnchor="middle" fontSize={8} fontWeight={700} fill={now ? "var(--surface)" : "var(--accent)"}>{i + 1}</text>
          </motion.g>
        );
      })}
      {reading && state.from > 1 && <path d={prevPath} fill="none" stroke="var(--accent)" strokeOpacity={0.3} strokeWidth={1.5} />}
      {cur.length > 1 && cur.slice(1).map((p, i) => {
        const q = cur[i];
        const segIndex = state.from - (state.from > 0 ? 1 : 0) + i + 1;
        return (
          <motion.line key={`${animKey}-p${i}`} x1={q[0]} y1={q[1]} x2={p[0]} y2={p[1]} stroke="var(--accent)" strokeWidth={2}
            initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 0.8 }} transition={{ delay: (segAt(segIndex - state.from) - SEG) * k, duration: SEG * k, ease }} />
        );
      })}
      {cur.length > 0 && (
        <motion.circle key={`${animKey}-cursor`} r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5}
          initial={{ cx: cur[0][0], cy: cur[0][1] }}
          animate={{ cx: cur.map((p) => p[0]), cy: cur.map((p) => p[1]) }}
          transition={{ delay: (segAt(0) - (state.from > 0 ? SEG : 0)) * k, duration: (cur.length - 1) * SEG * k || 0.01, ease: "linear" }} />
      )}
    </svg>
  );
}

function ExtractedText({ result, state, k, animKey }: { result: ParseResult; state: ParseState; k: number; animKey: string }) {
  const shown = result.segments.slice(0, state.upto);
  return (
    <div className="min-w-0 rounded-md border border-line bg-surface-2/50 p-2.5">
      <div className="mb-1.5 flex items-center justify-between text-[11px] font-semibold uppercase text-muted">
        <span>Extracted text</span><span className="font-mono normal-case">{shown.length}/{result.segments.length} segments</span>
      </div>
      {state.phase === "page" ? <p className="text-xs text-muted">Nothing extracted yet. Press → to start reading.</p> : (
        <p className="text-[12.5px] leading-relaxed">
          {shown.map((s, i) => {
            const now = state.phase === "read" && i >= state.from;
            return (
              <motion.span key={`${animKey}-${i}`} initial={now ? { opacity: 0 } : false} animate={{ opacity: 1 }}
                transition={{ delay: now ? segAt(i - state.from) * k : 0, duration: 0.3 * k }}
                className={cx("rounded px-0.5", now && "bg-accent-soft", s.kind === "header" || s.kind === "footer" ? "text-muted italic" : s.kind === "h1" || s.kind === "h2" ? "font-semibold" : s.kind === "cell" ? "font-mono text-warn" : s.kind === "row" ? "text-ok" : "")}>
                <sup className="mr-0.5 text-[9px] text-accent">{i + 1}</sup>{s.text}{" "}
              </motion.span>
            );
          })}
        </p>
      )}
    </div>
  );
}

function StageDetail({ state, result, sc, comparison, k, strategy }: { state: ParseState; result: ParseResult; sc: Score; comparison: { s: Strategy; sc: Score }[]; k: number; strategy: Strategy }) {
  if (state.phase === "page") {
    const counts = Object.entries(KIND_STYLE).map(([kind, st]) => ({ kind, st, n: buildPage().filter((u) => u.kind === kind).length })).filter((c) => c.n);
    return (
      <div className="flex flex-wrap gap-1.5">
        {counts.map((c, i) => (
          <motion.span key={c.kind} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.15) * k }}><Badge>{c.n} × {c.st.label}</Badge></motion.span>
        ))}
        <p className="mt-1 w-full text-xs text-muted">A human sees two columns and a table. The file only stores where each run of glyphs is drawn. The parser's job is to rebuild the reading order: {STRATEGIES[strategy].short.toLowerCase()}</p>
      </div>
    );
  }
  if (state.phase === "read") {
    const segs = result.segments.slice(state.from, state.upto);
    return (
      <div className="space-y-2">
        <div className="h-2 overflow-hidden rounded bg-surface-2">
          <motion.div className="h-full bg-accent" initial={{ width: `${(state.from / result.segments.length) * 100}%` }} animate={{ width: `${(state.upto / result.segments.length) * 100}%` }} transition={{ delay: segAt(0) * k, duration: (segAt(segs.length) - segAt(0)) * k, ease: "linear" }} />
        </div>
        <ol className="space-y-1">
          {segs.map((s, i) => (
            <motion.li key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: segAt(i) * k, duration: 0.25 * k }} className="flex items-start gap-2 text-xs">
              <span className="font-mono text-accent">#{state.from + i + 1}</span>
              <Badge tone={s.kind === "header" || s.kind === "footer" ? "neutral" : s.kind === "cell" ? "warn" : s.kind === "row" ? "ok" : "accent"}>{s.kind === "row" ? "table row" : KIND_STYLE[s.kind].label}</Badge>
              <span className="min-w-0">{s.text}</span>
            </motion.li>
          ))}
        </ol>
      </div>
    );
  }
  if (state.phase === "diff") return <DiffView text={result.segments.map((s) => s.text).join(" ")} sc={sc} k={k} />;
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="space-y-1">
        {sc.facts.map((f, i) => (
          <motion.div key={f.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.4) * k }} className={cx("flex items-center gap-2 rounded px-2 py-1 text-xs", f.found ? "bg-ok-soft" : "bg-err-soft")}>
            {f.found ? <CheckCircle2 size={13} className="text-ok" /> : <XCircle size={13} className="text-err" />}
            <span>{f.label}</span><span className="ml-auto text-[10px] text-muted">{f.found ? "found" : "missing"}</span>
          </motion.div>
        ))}
        <p className="pt-1 font-mono text-sm font-semibold">facts recoverable: <AnimatedNumber value={sc.recovered} from={0} digits={0} delay={0.3 * k} duration={2.4 * k} /> / {sc.facts.length}</p>
      </div>
      <div>
        <h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">All three parsers on this page</h4>
        {comparison.map(({ s, sc: c }, i) => (
          <div key={s} className="mb-1.5 grid grid-cols-[120px_minmax(0,1fr)_36px] items-center gap-2 text-xs">
            <span className={cx(s === strategy && "font-semibold")}>{STRATEGIES[s].name.replace(" extraction", "")}</span>
            <div className="h-3.5 rounded bg-surface-2"><motion.div className={cx("h-full rounded", c.recovered === c.facts.length ? "bg-ok/70" : c.recovered >= c.facts.length / 2 ? "bg-warn/70" : "bg-err/70")} initial={{ width: 0 }} animate={{ width: `${(c.recovered / c.facts.length) * 100}%` }} transition={{ delay: (2.8 + i * 0.25) * k, duration: 0.5 * k, ease }} /></div>
            <span className="text-right font-mono">{c.recovered}/{c.facts.length}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DiffView({ text, sc, k }: { text: string; sc: Score; k: number }) {
  const d = useMemo(() => diffWords(text, referenceText()), [text]);
  const step = Math.min(0.02, 2 / d.length);
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2 text-xs">
        <Badge tone="ok">same <AnimatedNumber value={sc.same} from={0} digits={0} duration={2 * k} /></Badge>
        <Badge tone="err">missing <AnimatedNumber value={d.filter((x) => x.op === "missing").length} from={0} digits={0} duration={2 * k} /></Badge>
        <Badge tone="warn">extra / out of order <AnimatedNumber value={sc.extraWords} from={0} digits={0} duration={2 * k} /></Badge>
      </div>
      <p className="text-[12px] leading-relaxed">
        {d.map((w, i) => (
          <motion.span key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * step * k, duration: 0.15 }}
            className={cx("rounded px-px", w.op === "missing" && "bg-err-soft text-err line-through", w.op === "extra" && "bg-warn-soft text-warn underline decoration-dotted")}>
            {w.word}{" "}
          </motion.span>
        ))}
      </p>
      <p className="text-[11px] text-muted"><span className="text-err line-through">struck red</span> = in the reference but missing here (or out of order); <span className="text-warn underline decoration-dotted">dotted amber</span> = extra or out of place.</p>
    </div>
  );
}
