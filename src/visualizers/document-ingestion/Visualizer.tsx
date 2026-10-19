import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { CheckCircle2, Copy, HelpCircle, MinusCircle, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "./kit";
import { DEFAULT_INGEST, STAGES, STAGE_LABEL, simulate, type DocResult, type DocView, type IngestConfig, type StageId, type Tally } from "./simulation";

// ---------- one timing schedule (seconds at 1×) for cards, detail card and counters ----------
const MOVE = 0.55;
const moveAt = (i: number) => 0.15 + i * 0.12;
const arriveAt = (i: number) => moveAt(i) + MOVE;
const CHAR = 0.05;
const verdictAt = (i: number) => arriveAt(i) + 8 * CHAR + 0.15;
const DURATIONS: Record<StageId, number> = { inbox: 2600, fetch: 2900, detect: 3300, extract: 3000, clean: 3300, hash: 3900, metadata: 3200, queue: 3800 };
const stepDuration = (i: number) => DURATIONS[STAGES[i]] ?? 3000;

const COLW = 94, ROWH = 46, CARDW = 88, TOP = 30;
const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

const NARRATION: Record<StageId, (t: Tally, cfg: IngestConfig) => { headline: string; sub: string }> = {
  inbox: () => ({ headline: "Seven sources arrive in the ingestion inbox", sub: "Different formats, one stray copy and one binary file. Watch which ones make it to the index queue." }),
  fetch: (t) => ({ headline: `Connectors download ${t.bytes} raw bytes`, sub: "Fetching only moves bytes. Nothing knows yet what is inside each file." }),
  detect: (_, cfg) => ({ headline: "Each file's first bytes decide which extractor runs", sub: cfg.unsupported === "log" ? "The .vsdx diagram has no extractor: it is logged and parked in a dead-letter queue." : "The .vsdx diagram has no extractor and is skipped silently: no log line, no trace." }),
  extract: () => ({ headline: "Extractors turn every format into plain text", sub: "Tags, PDF headers and Markdown symbols disappear; database rows become Q/A lines." }),
  clean: () => ({ headline: "Cleaning strips boilerplate and normalizes whitespace", sub: "Page footers, nav bars and copyright lines would otherwise be repeated in every chunk." }),
  hash: (t, cfg) => ({
    headline: t.duplicate ? `Hashing catches ${t.duplicate} duplicate${t.duplicate > 1 ? "s" : ""}${t.unchanged ? ` and skips ${t.unchanged} unchanged` : ""}` : cfg.hashOn === "raw" ? "Raw-bytes hashing misses the re-formatted copy" : "Every hash is new: nothing to drop",
    sub: cfg.hashOn === "raw" ? "pricing (copy).pdf differs only in spaces and line endings, so its raw bytes hash differently." : "Identical cleaned text gives an identical hash, however the file was named or formatted.",
  }),
  metadata: (t) => ({ headline: `${t.queued} survivors get source, version, timestamp and ACL`, sub: "Retrieval will use these to filter by permission, prefer fresh versions and cite sources." }),
  queue: (t) => ({
    headline: t.lost ? `${t.queued} queued, but ${t.lost} document is unaccounted for` : `${t.queued} documents queued for chunking and embedding`,
    sub: t.lost ? "Received ≠ queued + dropped + skipped + rejected. A silent skip leaves a hole nobody can see." : "Every received document is either queued, dropped as a duplicate, skipped as unchanged, or rejected with a log line.",
  }),
};

export default function DocumentIngestionVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<IngestConfig>(DEFAULT_INGEST);
  const [focus, setFocus] = useState("sso");
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length, stepDuration);
  const step = sim.steps[pb.index];
  const k = 1 / pb.speed;
  const stage = step.state.stage;
  const animKey = `${step.id}-${replay}`;
  const focusDoc = sim.docs.find((d) => d.source.id === focus)!;
  const story = NARRATION[stage](sim.tally, cfg);
  const set = (p: Partial<IngestConfig>) => setCfg((c) => ({ ...c, ...p }));

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Ingestion pipeline" actions={<span className="text-[11px] text-muted">Click a card to inspect it</span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <Board docs={sim.docs} views={step.state.docs} prevViews={sim.steps[Math.max(0, pb.index - 1)].state.docs} stageIndex={step.state.stageIndex} focus={focus} onFocus={setFocus} k={k} replay={replay} />
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><CheckCircle2 size={12} className="text-ok" /> queued</span>
              <span className="flex items-center gap-1"><Copy size={12} className="text-warn" /> duplicate, dropped</span>
              <span className="flex items-center gap-1"><MinusCircle size={12} /> unchanged, skipped</span>
              <span className="flex items-center gap-1"><XCircle size={12} className="text-err" /> rejected, logged</span>
              <span className="flex items-center gap-1"><HelpCircle size={12} className="text-err" /> vanished silently</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
              <StageDetail stage={stage} docs={sim.docs} tally={sim.tally} focus={focusDoc} k={k} cfg={cfg} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Pipeline settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_INGEST); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Run mode" value={cfg.mode} onChange={(v) => set({ mode: v })} options={[{ value: "full", label: "Full re-ingest" }, { value: "incremental", label: "Incremental (changed only)" }]} />
              <Select label="Unsupported file type" value={cfg.unsupported} onChange={(v) => set({ unsupported: v })} options={[{ value: "log", label: "Log + dead-letter queue" }, { value: "silent", label: "Skip silently (break it)" }]} />
              <Select label="Dedupe hash computed on" value={cfg.hashOn} onChange={(v) => set({ hashOn: v })} options={[{ value: "normalized", label: "Cleaned, normalized text" }, { value: "raw", label: "Raw bytes (break it)" }]} />
            </div>
            <p className="mt-2 text-[11px] text-muted">Incremental mode compares each hash with what the previous run stored. Only api-rate-limits.md changed since then (300 → 600 requests per minute).</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why="Everything retrieval can find, filter or cite is decided here. A document lost, duplicated or stale at ingestion is wrong in every answer that touches it." />
          <Panel title="Calculation" actions={<SimulatedLabel>FNV-1a stands in for SHA-256</SimulatedLabel>}>
            <Calculation stage={stage} docs={sim.docs} tally={sim.tally} focus={focusDoc} cfg={cfg} />
          </Panel>
          <Panel title={`Focused: ${focusDoc.source.name}`}>
            <KV rows={[
              ["uri", focusDoc.source.uri], ["detected", focusDoc.kind], ["bytes", focusDoc.bytes], ["clean words", words(focusDoc.cleaned)],
              ["content hash", focusDoc.hash], ["outcome", focusDoc.outcome], ["why", focusDoc.reason],
            ]} />
          </Panel>
          <Panel title="Tally">
            <KV rows={[["received", sim.tally.received], ["queued", sim.tally.queued], ["duplicates dropped", sim.tally.duplicate], ["unchanged skipped", sim.tally.unchanged], ["rejected (logged)", sim.tally.rejected], ["unaccounted", sim.tally.lost]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function outcomeStyle(o: DocView["outcome"]) {
  switch (o) {
    case "queued": return { cls: "border-ok/60 bg-ok-soft", Icon: CheckCircle2, icon: "text-ok", text: "queued" };
    case "duplicate": return { cls: "border-warn/60 bg-warn-soft", Icon: Copy, icon: "text-warn", text: "duplicate" };
    case "unchanged": return { cls: "border-line bg-surface-2", Icon: MinusCircle, icon: "text-muted", text: "unchanged" };
    case "rejected": return { cls: "border-err/60 bg-err-soft", Icon: XCircle, icon: "text-err", text: "dead-letter" };
    case "lost": return { cls: "border-dashed border-err/40 bg-surface", Icon: HelpCircle, icon: "text-err", text: "no log" };
    default: return { cls: "border-accent/50 bg-surface", Icon: null, icon: "", text: "" };
  }
}

function cardLabel(d: DocResult, col: number): string {
  switch (STAGES[col]) {
    case "inbox": return d.source.connector;
    case "fetch": return `${d.bytes} B`;
    case "detect": return d.kind;
    case "extract": return `${words(d.extracted)} words`;
    case "clean": return `−${d.removed.length} lines`;
    case "hash": return `#${d.hash}`;
    case "metadata": return `${d.source.version} · ${d.source.acl}`;
    default: return "ready";
  }
}

function Board({ docs, views, prevViews, stageIndex, focus, onFocus, k, replay }: {
  docs: DocResult[]; views: DocView[]; prevViews: DocView[]; stageIndex: number; focus: string; onFocus: (id: string) => void; k: number; replay: number;
}) {
  const W = STAGES.length * COLW;
  const H = TOP + docs.length * ROWH;
  return (
    <div className="relative min-w-[752px]" style={{ height: H }}>
      {STAGES.map((s, c) => (
        <div key={s} className={cx("absolute top-0 rounded-md transition-colors duration-500", c === stageIndex ? "bg-accent-soft" : c < stageIndex ? "bg-surface-2/60" : "")} style={{ left: c * COLW, width: COLW - 4, height: H }}>
          <div className={cx("pt-1.5 text-center text-[10px] font-semibold whitespace-nowrap", c === stageIndex ? "text-accent" : "text-muted")}>{c ? `${c}. ` : ""}{STAGE_LABEL[s]}</div>
        </div>
      ))}
      <svg className="pointer-events-none absolute inset-0" width={W} height={H} aria-hidden>
        {STAGES.slice(1).map((s, c) => <line key={s} x1={(c + 1) * COLW - 4} x2={(c + 1) * COLW - 4} y1={TOP} y2={H} stroke="var(--line)" strokeDasharray="2 4" />)}
      </svg>
      {docs.map((d, i) => {
        const v = views[i];
        const prev = prevViews[i];
        const st = outcomeStyle(v.outcome);
        const moving = prev.col !== v.col;
        const hashing = STAGES[v.col] === "hash" && v.col === stageIndex;
        const gone = v.outcome === "lost";
        return (
          <motion.button
            key={`${d.source.id}-${replay}`}
            type="button"
            onClick={() => onFocus(d.source.id)}
            aria-label={`${d.source.name}: ${v.outcome === "moving" ? `at ${STAGE_LABEL[STAGES[v.col]]}` : st.text}`}
            className={cx("absolute left-0 flex flex-col justify-center rounded-md border px-1.5 text-left shadow-sm", st.cls, focus === d.source.id && "ring-2 ring-accent")}
            style={{ top: TOP + i * ROWH + 2, width: CARDW, height: ROWH - 8 }}
            initial={{ x: prev.col * COLW, y: 0, opacity: 1 }}
            animate={{ x: v.col * COLW, y: v.stopped ? 3 : 0, opacity: gone ? 0.35 : v.stopped ? 0.8 : 1 }}
            transition={{ x: { delay: moving ? moveAt(i) * k : 0, duration: MOVE * k, ease }, opacity: { delay: (moving ? arriveAt(i) : 0.2) * k, duration: 0.4 * k }, y: { delay: arriveAt(i) * k } }}
          >
            <span className="flex items-center gap-1 truncate font-mono text-[10px] font-medium leading-tight">
              {st.Icon && <st.Icon size={11} className={cx("shrink-0", st.icon)} aria-hidden />}
              <span className="truncate">{d.source.name}</span>
            </span>
            <span className="truncate font-mono text-[10px] leading-tight text-muted">
              {hashing ? <HashReveal hash={d.hash} delay={arriveAt(i) * k} k={k} /> : v.stopped ? st.text : cardLabel(d, v.col)}
            </span>
          </motion.button>
        );
      })}
    </div>
  );
}

function HashReveal({ hash, delay, k }: { hash: string; delay: number; k: number }) {
  return (
    <span>#{hash.split("").map((ch, j) => (
      <motion.span key={j} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: delay + j * CHAR * k, duration: 0.1 }}>{ch}</motion.span>
    ))}</span>
  );
}

// ---------- the explanatory card under the board ----------
function StageDetail({ stage, docs, tally, focus, k, cfg }: { stage: StageId; docs: DocResult[]; tally: Tally; focus: DocResult; k: number; cfg: IngestConfig }) {
  const fade = (i: number) => ({ initial: { opacity: 0, x: -10 }, animate: { opacity: 1, x: 0 }, transition: { delay: arriveAt(i) * k, duration: 0.3 * k } });
  switch (stage) {
    case "inbox":
      return (
        <ul className="grid gap-1.5 sm:grid-cols-2">
          {docs.map((d, i) => (
            <motion.li key={d.source.id} {...fade(i)} className="flex items-center justify-between gap-2 rounded-md border border-line px-2 py-1 text-xs">
              <span className="truncate font-mono">{d.source.uri}</span><Badge>{d.source.connector}</Badge>
            </motion.li>
          ))}
        </ul>
      );
    case "fetch": {
      const max = Math.max(...docs.map((d) => d.bytes));
      return (
        <div className="space-y-1">
          {docs.map((d, i) => (
            <div key={d.source.id} className="grid grid-cols-[130px_minmax(0,1fr)_56px] items-center gap-2 text-xs">
              <span className="truncate font-mono">{d.source.name}</span>
              <div className="h-3.5 rounded bg-surface-2"><motion.div className="h-full rounded bg-accent/70" initial={{ width: 0 }} animate={{ width: `${(d.bytes / max) * 100}%` }} transition={{ delay: moveAt(i) * k, duration: MOVE * k, ease }} /></div>
              <span className="text-right font-mono"><AnimatedNumber value={d.bytes} from={0} digits={0} delay={moveAt(i) * k} duration={MOVE * k} /> B</span>
            </div>
          ))}
          <p className="pt-1 text-right font-mono text-xs font-semibold">total <AnimatedNumber value={tally.bytes} from={0} digits={0} delay={0.15 * k} duration={1.4 * k} /> bytes</p>
        </div>
      );
    }
    case "detect":
      return (
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">file</th><th>first bytes</th><th>evidence</th><th>route</th></tr></thead>
          <tbody>
            {docs.map((d, i) => (
              <motion.tr key={d.source.id} {...fade(i)} className="border-t border-line">
                <td className="py-1 font-mono">{d.source.name}</td>
                <td className="font-mono text-muted">{JSON.stringify(d.source.raw.slice(0, 9)).replace(/\\u0003|\\u0004/g, "·")}</td>
                <td>{d.evidence}</td>
                <td>{d.kind !== "unsupported" ? <Badge tone="accent">{d.kind} extractor</Badge> : cfg.unsupported === "log" ? <Badge tone="err"><XCircle size={11} /> dead-letter + log</Badge> : <Badge tone="neutral"><HelpCircle size={11} /> dropped, no log</Badge>}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      );
    case "extract":
      return <BeforeAfter k={k} leftTitle={`raw ${focus.source.name}`} left={focus.source.raw} rightTitle="extracted text" right={focus.extracted || "(nothing: no extractor for this type)"} note="Click any card on the board to see a different document." />;
    case "clean":
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">extracted lines</h4>
            <ul className="space-y-0.5 font-mono text-[11px]">
              {focus.extracted.split("\n").filter((l) => l.trim()).map((l, j) => {
                const drop = focus.removed.includes(l.replace(/\s+/g, " ").trim());
                return (
                  <motion.li key={j} className={cx("rounded px-1", drop && "bg-err-soft")} initial={false} animate={{ opacity: drop ? 0.55 : 1 }} transition={{ delay: (0.4 + j * 0.25) * k }}>
                    {drop ? <s>{l}</s> : l} {drop && <span className="text-err">← boilerplate</span>}
                  </motion.li>
                );
              })}
              {!focus.extracted && <li className="text-muted">(no text)</li>}
            </ul>
          </div>
          <div>
            <h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">cleaned</h4>
            <motion.pre className="whitespace-pre-wrap rounded bg-surface-2 p-2 font-mono text-[11px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.6 * k }}>{focus.cleaned || "(empty)"}</motion.pre>
            <p className="mt-1 text-[11px] text-muted">Across the batch: {docs.reduce((s, d) => s + d.removed.length, 0)} boilerplate lines removed.</p>
          </div>
        </div>
      );
    case "hash": {
      return (
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">document</th><th>hash of {cfg.hashOn === "raw" ? "raw bytes" : "cleaned text"}</th><th>lookup</th></tr></thead>
          <tbody>
            {docs.map((d, i) => {
              if (d.kind === "unsupported") return null;
              const verdict = d.outcome === "duplicate" ? <Badge tone="warn"><Copy size={11} /> = {d.dupOf}: drop</Badge>
                : d.outcome === "unchanged" ? <Badge><MinusCircle size={11} /> same as last run: skip</Badge>
                : <Badge tone="ok"><CheckCircle2 size={11} /> new hash: keep</Badge>;
              return (
                <tr key={d.source.id} className={cx("border-t border-line", d.outcome === "duplicate" && "bg-warn-soft/50")}>
                  <td className="py-1 font-mono">{d.source.name}</td>
                  <td className="font-mono"><HashReveal hash={d.hash} delay={arriveAt(i) * k} k={k} /></td>
                  <td><motion.span initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: verdictAt(i) * k, type: "spring", stiffness: 300, damping: 20 }} className="inline-block">{verdict}</motion.span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      );
    }
    case "metadata": {
      const shown = focus.metadata ? focus : docs.find((d) => d.metadata)!;
      const rows = Object.entries(shown.metadata ?? {});
      return (
        <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <pre className="rounded bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">
            {"{\n"}
            {rows.map(([key, val], j) => (
              <motion.span key={key} className="block" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + j * 0.22) * k }}>{`  "${key}": "${val}",`}</motion.span>
            ))}
            {"}"}
          </pre>
          <div className="space-y-1.5 text-xs leading-relaxed text-muted">
            {shown !== focus && <p className="text-fg">{focus.source.name} did not reach this stage, so {shown.source.name} is shown.</p>}
            <p><strong className="text-fg">acl</strong> lets retrieval filter out chunks the asking user may not see (the FAQ rows are support-team only).</p>
            <p><strong className="text-fg">version / updated_at</strong> let you prefer the newest copy and spot a stale index.</p>
            <p><strong className="text-fg">content_hash</strong> is what the next incremental run compares against.</p>
          </div>
        </div>
      );
    }
    case "queue": {
      const parts: [string, number, string][] = [["queued", tally.queued, "bg-ok/70"], ["duplicate", tally.duplicate, "bg-warn/70"], ["unchanged", tally.unchanged, "bg-muted/40"], ["rejected", tally.rejected, "bg-err/70"], ["unaccounted", tally.lost, "bg-err/20"]];
      let acc = 0;
      return (
        <div className="space-y-2">
          <div className="relative h-7 overflow-hidden rounded-md bg-surface-2">
            {parts.filter(([, v]) => v > 0).map(([label, v, cls], j) => {
              const left = (acc / tally.received) * 100;
              acc += v;
              return (
                <motion.div key={label} className={cx("absolute inset-y-0 flex items-center justify-center border-r border-surface text-[10px] font-medium", cls, label === "unaccounted" && "border-2 border-dashed border-err")} style={{ left: `${left}%` }}
                  initial={{ width: 0 }} animate={{ width: `${(v / tally.received) * 100}%` }} transition={{ delay: (0.3 + j * 0.35) * k, duration: 0.4 * k, ease }}>
                  {label} {v}
                </motion.div>
              );
            })}
          </div>
          <p className="font-mono text-sm">
            received <AnimatedNumber value={tally.received} from={0} digits={0} duration={0.5 * k} /> = queued <AnimatedNumber value={tally.queued} from={0} digits={0} delay={0.3 * k} duration={0.4 * k} /> + dropped {tally.duplicate} + skipped {tally.unchanged} + rejected {tally.rejected}
            {tally.lost > 0 && <span className="text-err"> + <strong>{tally.lost} unexplained</strong></span>}
          </p>
          {tally.lost > 0
            ? <Callout tone="err" title="A silent skip">architecture.vsdx never produced a log line. Dashboards show a green run, yet questions about the architecture will never find an answer.</Callout>
            : <Callout tone="ok" title="Every document accounted for">Log rejects, keep a dead-letter queue, and alert when received ≠ the sum of outcomes.</Callout>}
        </div>
      );
    }
  }
}

function BeforeAfter({ left, right, leftTitle, rightTitle, note, k }: { left: string; right: string; leftTitle: string; rightTitle: string; note: string; k: number }) {
  return (
    <div>
      <div className="grid items-start gap-3 md:grid-cols-[minmax(0,1fr)_24px_minmax(0,1fr)]">
        <div><h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">{leftTitle}</h4><pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded bg-surface-2 p-2 font-mono text-[11px]">{left}</pre></div>
        <motion.div className="hidden self-center text-center text-accent md:block" initial={{ x: -8, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.5 * k, duration: 0.4 * k }}>→</motion.div>
        <div>
          <h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">{rightTitle}</h4>
          <pre className="whitespace-pre-wrap rounded border border-accent/40 bg-accent-soft/40 p-2 font-mono text-[11px]">
            {right.split("\n").map((l, j) => <motion.span key={j} className="block" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.7 + j * 0.3) * k }}>{l}</motion.span>)}
          </pre>
        </div>
      </div>
      <p className="mt-1 text-[11px] text-muted">{note}</p>
    </div>
  );
}

function Calculation({ stage, docs, tally, focus, cfg }: { stage: StageId; docs: DocResult[]; tally: Tally; focus: DocResult; cfg: IngestConfig }) {
  switch (stage) {
    case "fetch":
      return <TexBlock>{`\\text{bytes} = ${docs.map((d) => d.bytes).join(" + ")} = ${tally.bytes}`}</TexBlock>;
    case "hash":
      return (
        <div className="space-y-2">
          <TexBlock>{`\\begin{aligned} h_0 &= 2166136261 \\\\ h_i &= (h_{i-1} \\oplus b_i) \\times 16777619 \\bmod 2^{32} \\\\ h(\\text{${focus.source.id}}) &= \\texttt{${focus.hash}} \\end{aligned}`}</TexBlock>
          <TexBlock>{`\\text{drop } d \\iff h(d) \\in S_{\\text{seen}}${cfg.mode === "incremental" ? " \\cup S_{\\text{last run}}" : ""}`}</TexBlock>
          <p className="text-[11px] text-muted">b_i are the bytes of the {cfg.hashOn === "raw" ? "raw file" : "cleaned, lower-cased text"}.</p>
        </div>
      );
    case "queue":
      return <TexBlock>{`\\begin{aligned} N_{\\text{received}} &= N_{\\text{queued}} + N_{\\text{dup}} + N_{\\text{unchanged}} + N_{\\text{rejected}} \\\\ ${tally.received} &${tally.lost ? "\\ne" : "="} ${tally.queued} + ${tally.duplicate} + ${tally.unchanged} + ${tally.rejected}${tally.lost ? ` = ${tally.accounted}` : ""} \\end{aligned}`}</TexBlock>;
    case "clean":
      return <KV rows={[["chars before", focus.extracted.length], ["chars after", focus.cleaned.length], ["lines removed", focus.removed.length]]} />;
    case "extract":
      return <KV rows={[["raw chars", focus.source.raw.length], ["extracted chars", focus.extracted.length], ["kept", focus.source.raw.length ? `${((focus.extracted.length / focus.source.raw.length) * 100).toFixed(0)}%` : "—"]]} />;
    case "detect":
      return <KV rows={docs.map((d) => [d.source.name, d.kind] as [string, string])} />;
    case "metadata":
      return <KV rows={[["documents tagged", tally.queued], ["fields per document", 7]]} />;
    default:
      return <KV rows={[["sources", docs.length], ["connectors", [...new Set(docs.map((d) => d.source.connector))].join(", ")]]} />;
  }
}
