import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { pct, wrap } from "../grounded-generation/claims";
import { EASE, StepCaption } from "../grounded-generation/StepCaption";
import { BUGS, DEFAULT_CM, QUESTION, simulate, type CMBug, type CMParams, type CMResult, type CMState, type Citation } from "./simulation";

// ---------- timing (seconds at 1×) ----------
const HOP = 0.55, HOP0 = 0.3;
const hopAt = (j: number) => HOP0 + j * HOP;
const VERIFY0 = 0.4, VERIFY_GAP = 0.9;
const DUR: Record<CMState["phase"], number> = { answer: 2800, sources: 3800, resolve: (hopAt(4) + 1.4) * 1000, verify: (VERIFY0 + 3 * VERIFY_GAP + 1.3) * 1000, score: 2800 };

const W = 800;
const COLS = [
  { x: 12, w: 84, label: "marker" }, { x: 122, w: 112, label: "prompt passage" }, { x: 260, w: 112, label: "chunk id" },
  { x: 398, w: 136, label: "source document" }, { x: 560, w: 140, label: "page / anchor" }, { x: 724, w: 66, label: "verifier" },
];
const ROW_H = 58, BOX_H = 40;

function chainCells(c: Citation, i: number): { top: string; sub: string; bad: boolean }[] {
  const r = c.resolved;
  return [
    { top: `[${c.marker}]`, sub: `sentence ${i + 1}`, bad: false },
    r ? { top: `passage ${r.label}`, sub: r.truncated ? `✂ cut at ${r.words} words` : `via ${r.via}`, bad: c.wrongSource || r.truncated } : { top: "∅ no passage", sub: `label [${c.marker}] missing`, bad: true },
    r ? { top: r.chunkId, sub: c.wrongSource ? `read from ${c.intended.chunkId}` : "same as read", bad: c.wrongSource } : { top: "—", sub: "", bad: true },
    r ? { top: c.doc!.file, sub: `${c.doc!.version} · ${c.doc!.updatedAt}`, bad: c.wrongSource } : { top: "—", sub: "", bad: true },
    r ? { top: c.anchor!.page ? `p.${c.anchor!.page} ${c.anchor!.anchor}` : c.anchor!.anchor, sub: "where the click lands", bad: c.wrongSource } : { top: "—", sub: "", bad: true },
  ];
}

export default function CitationMappingVisualizer() {
  const [params, setParams] = useState<CMParams>(DEFAULT_CM);
  const [replay, setReplay] = useState(0);
  const { result, steps } = useMemo(() => simulate(params), [params]);
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "answer"]);
  const step = steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(params)}`;
  const set = (p: Partial<CMParams>) => setParams((v) => ({ ...v, ...p }));
  const story = narrate(step.state, result);

  return (
    <Workspace
      main={
        <>
          <Panel title="Citation mapping" actions={<SimulatedLabel>Simulated answer & resolver</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={animKey} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[720px]">
                <Canvas r={result} state={step.state} k={k} animKey={animKey} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>the marker travelling through each lookup</span>
              <span>red box = lookup returned the wrong thing · ✂ = passage text was truncated</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard r={result} state={step.state} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Bugs & fixes" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_CM); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1">
                <Select label="Inject a mapping bug" value={params.bug} onChange={(v: CMBug) => set({ bug: v })} options={(Object.keys(BUGS) as CMBug[]).map((b) => ({ value: b, label: BUGS[b].label }))} />
                <p className="text-[11px] text-muted">{BUGS[params.bug].note}</p>
              </div>
              <div className="space-y-3">
                <Toggle label="Dedupe before building the prompt (fix)" checked={params.dedupeFirst} onChange={(v) => set({ dedupeFirst: v })} hint="Labels are assigned once, after dedupe, so nothing can renumber them later." />
                <Toggle label="Strict verifier" checked={params.strict} onChange={(v) => set({ strict: v })} hint="On: one sentence must support the claim, numbers included. Off: word overlap with the whole passage (weak)." />
              </div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[step.state.phase]} />
          <Panel title="Calculation"><Calculation r={result} state={step.state} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["prompt passages", result.prompt.length],
              ["source list after dedupe", result.display.length],
              ["mapping bug", BUGS[params.bug].label],
              ["verifier", params.strict ? "sentence + numbers" : "passage word overlap"],
              ["citation accuracy", step.state.phase === "score" ? pct(result.accuracy) : "—"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const WHY: Record<CMState["phase"], string> = {
  answer: "A citation is a promise: 'click here and you'll see where this came from'. The marker is only a number; everything depends on how it is mapped back.",
  sources: "Any step that changes the list after generation (dedupe, sorting, filtering) can silently change what a number means.",
  resolve: "Each hop is a lookup that can go wrong independently. Resolving by stable chunk id instead of position removes a whole class of bugs.",
  verify: "Checking that the cited span actually contains the claim catches mapping bugs and unfaithful sentences alike.",
  score: "Citation accuracy is separate from answer correctness: a perfect answer with broken citations still fails users who check sources.",
};

function narrate(s: CMState, r: CMResult) {
  const c = s.focus !== null ? r.citations[s.focus] : null;
  switch (s.phase) {
    case "answer": return { headline: `The answer arrives with markers ${r.citations.map((x) => `[${x.marker}]`).join(" ")}`, sub: "Each number refers to a passage in the prompt. Now it must be turned into a link a user can click." };
    case "sources": return r.params.dedupeFirst
      ? { headline: "Duplicates were removed before prompting", sub: "The labels the model saw are the labels users will see." }
      : { headline: `The source list is deduplicated: ${r.prompt.length} → ${r.display.length} entries`, sub: r.params.bug === "dedupe" ? "Bug: entries are renumbered by position, the answer's markers are not." : "Markers stay tied to chunk ids, so the renumbering is harmless." };
    case "resolve":
      if (!c!.resolved) return { headline: `[${c!.marker}] resolves to nothing`, sub: "The label doesn't exist in the list the resolver used: a dangling citation." };
      if (c!.wrongSource) return { headline: `[${c!.marker}] lands on ${c!.resolved.chunkId}, not ${c!.intended.chunkId}`, sub: "The chain completes, so nothing errors, but the user is sent to the wrong document." };
      if (c!.resolved.truncated) return { headline: `[${c!.marker}] reaches the right chunk, but its text was cut`, sub: "The stored passage stops mid-sentence; the rest of the claim came from model memory." };
      return { headline: `[${c!.marker}] → ${c!.resolved.chunkId} → ${c!.doc!.file}`, sub: "Marker → prompt passage → chunk id → source document → page/anchor." };
    case "verify": return { headline: "The verifier checks every cited span against its sentence", sub: r.params.strict ? "Pass only if one sentence of the cited passage supports the claim, numbers included." : "Lenient mode: word overlap with the whole passage. Watch what slips through." };
    case "score": return { headline: `Citation accuracy: ${r.citations.filter((x) => x.pass).length} of ${r.citations.length} (${pct(r.accuracy)})`, sub: r.accuracy === 1 ? "Every marker leads to a span that supports its sentence." : "The answer text never changed; only the mapping did." };
  }
}

function Canvas({ r, state, k, animKey }: { r: CMResult; state: CMState; k: number; animKey: string }) {
  const answerLines = r.citations.map((c) => wrap(c.claim, 108));
  const answerY = answerLines.map((_, i) => 46 + answerLines.slice(0, i).reduce((s, ls) => s + ls.length * 16 + 6, 0));
  const answerH = answerY.at(-1)! + answerLines.at(-1)!.length * 16 + 10;
  const gridTop = answerH + 34;
  const gridBottom = gridTop + r.citations.length * ROW_H;
  const stripTop = gridBottom + 16;
  const H = stripTop + 150;
  const cx0 = (j: number) => COLS[j].x + COLS[j].w / 2;
  const rowY = (i: number) => gridTop + i * ROW_H;
  const ph = state.phase;
  const highlightRow = ph === "resolve" ? state.focus : null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Answer with citation markers, a mapping chain per marker, and the passage list before and after dedupe">
      <text x={8} y={16} fontSize={12} fontWeight={600} fill="var(--fg)">Q: {QUESTION}</text>
      <rect x={8} y={26} width={W - 16} height={answerH - 26} rx={8} fill="var(--surface-2)" stroke="var(--line)" />
      <text x={18} y={40} fontSize={10} letterSpacing={0.5} fill="var(--muted)">ANSWER</text>
      {r.citations.map((c, i) => (
        <motion.g key={`a${i}-${ph === "answer" ? animKey : "s"}`} initial={ph === "answer" ? { opacity: 0, x: -12 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.5) * k, duration: 0.4 * k }}>
          {answerLines[i].map((l, li) => (
            <text key={li} x={18} y={answerY[i] + 10 + li * 16} fontSize={11.5} fill="var(--fg)">
              {l}{li === answerLines[i].length - 1 && <tspan fontWeight={700} fill={highlightRow === i ? "var(--accent)" : "var(--violet)"}> [{c.marker}]</tspan>}
            </text>
          ))}
        </motion.g>
      ))}

      {COLS.map((c) => <text key={c.label} x={c.x + c.w / 2} y={gridTop - 10} textAnchor="middle" fontSize={10} letterSpacing={0.4} fill="var(--muted)">{c.label.toUpperCase()}</text>)}
      {r.citations.map((c, i) => {
        const cells = chainCells(c, i);
        const shown = i < state.resolved;
        const animating = ph === "resolve" && state.focus === i;
        const brokenIdx = c.resolved ? 5 : 2; // cells after a dangling passage stay empty
        const verifying = ph === "verify";
        const vDelay = VERIFY0 + i * VERIFY_GAP;
        const showVerdict = ph === "verify" || ph === "score";
        const y0 = rowY(i);
        return (
          <g key={`row${i}`}>
            {highlightRow === i && <rect x={4} y={y0 - 4} width={W - 8} height={BOX_H + 8} rx={8} fill="var(--accent-soft)" opacity={0.6} />}
            {[0, 1, 2, 3].map((j) => (
              <line key={j} x1={COLS[j].x + COLS[j].w} y1={y0 + BOX_H / 2} x2={COLS[j + 1].x} y2={y0 + BOX_H / 2} stroke={shown && j + 1 < brokenIdx ? (cells[j + 1].bad ? "var(--err)" : "var(--ok)") : "var(--line)"} strokeWidth={1.5} strokeDasharray={shown ? undefined : "3 3"} />
            ))}
            {cells.map((cell, j) => {
              const visible = j === 0 || (shown && j < brokenIdx);
              const d = animating ? hopAt(j) : 0;
              return (
                <g key={`${j}-${animating ? animKey : "s"}`}>
                  <rect x={COLS[j].x} y={y0} width={COLS[j].w} height={BOX_H} rx={6} fill="var(--surface)" stroke="var(--line)" strokeDasharray={visible ? undefined : "4 3"} />
                  {visible && (
                    <motion.g initial={animating && j > 0 ? { opacity: 0, scale: 0.85 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: d * k, duration: 0.3 * k }} style={{ transformOrigin: `${cx0(j)}px ${y0 + BOX_H / 2}px` }}>
                      <rect x={COLS[j].x} y={y0} width={COLS[j].w} height={BOX_H} rx={6} fill={cell.bad ? "var(--err-soft)" : j === 0 ? "var(--accent-soft)" : "var(--ok-soft)"} stroke={cell.bad ? "var(--err)" : j === 0 ? "var(--accent)" : "var(--ok)"} />
                      <text x={cx0(j)} y={y0 + 17} textAnchor="middle" fontSize={j === 4 ? 9.5 : 11} fontWeight={600} fontFamily={j === 2 || j === 0 ? "var(--font-mono)" : undefined} fill="var(--fg)">{fit(cell.top, j === 4 ? 26 : 19)}</text>
                      <text x={cx0(j)} y={y0 + 31} textAnchor="middle" fontSize={9} fill={cell.bad ? "var(--err)" : "var(--muted)"}>{fit(cell.sub, 24)}</text>
                    </motion.g>
                  )}
                </g>
              );
            })}
            {/* verifier cell */}
            <rect x={COLS[5].x} y={y0} width={COLS[5].w} height={BOX_H} rx={6} fill="var(--surface)" stroke="var(--line)" strokeDasharray={showVerdict ? undefined : "4 3"} />
            {showVerdict && (
              <motion.g key={`v-${verifying ? animKey : "s"}`} initial={verifying ? { opacity: 0, scale: 0.4 } : false} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: (verifying ? vDelay + 0.45 : 0) * k, type: "spring", stiffness: 320, damping: 18 }} style={{ transformOrigin: `${cx0(5)}px ${y0 + BOX_H / 2}px` }}>
                <rect x={COLS[5].x} y={y0} width={COLS[5].w} height={BOX_H} rx={6} fill={c.pass ? "var(--ok)" : "var(--err)"} />
                <text x={cx0(5)} y={y0 + 18} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--surface)">{c.pass ? "✓ pass" : "✗ fail"}</text>
                <text x={cx0(5)} y={y0 + 32} textAnchor="middle" fontSize={9} fill="var(--surface)">{c.check ? `${pct(c.check.coverage)} words` : "no passage"}</text>
              </motion.g>
            )}
            {verifying && (
              <motion.rect key={`scan-${animKey}-${i}`} x={COLS[1].x} y={y0 - 3} height={BOX_H + 6} rx={6} fill="var(--accent)" initial={{ width: 0, opacity: 0 }}
                animate={{ width: [0, COLS[5].x - COLS[1].x, COLS[5].x - COLS[1].x], opacity: [0.18, 0.18, 0] }} transition={{ delay: vDelay * k, duration: 0.6 * k, times: [0, 0.75, 1] }} />
            )}
          </g>
        );
      })}
      {/* The marker travelling hop by hop along the chain. */}
      {ph === "resolve" && state.focus !== null && (() => {
        const c = r.citations[state.focus];
        const last = c.resolved ? 4 : 1;
        const xs = Array.from({ length: last + 1 }, (_, j) => cx0(j));
        const times = xs.map((_, j) => (j === 0 ? 0 : j / last));
        return (
          <motion.g key={`tok-${animKey}`} initial={{ x: xs[0], opacity: 0 }} animate={{ x: xs, opacity: 1 }} transition={{ delay: HOP0 * k, duration: last * HOP * k, ease: "easeInOut", times, opacity: { duration: 0.2 * k, delay: HOP0 * k } }}>
            <circle cx={0} cy={rowY(state.focus) - 2} r={7} fill={c.resolved && !c.wrongSource ? "var(--accent)" : "var(--err)"} stroke="var(--surface)" strokeWidth={2} />
            <text x={0} y={rowY(state.focus) + 1} textAnchor="middle" fontSize={8} fontWeight={700} fill="var(--surface)">{c.marker}</text>
          </motion.g>
        );
      })()}

      <SourceStrip r={r} top={stripTop} phase={ph} k={k} animKey={animKey} />
    </svg>
  );
}

const fit = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function SourceStrip({ r, top, phase, k, animKey }: { r: CMResult; top: number; phase: CMState["phase"]; k: number; animKey: string }) {
  const chipW = 172, gap = 12, x0 = 12;
  const anim = phase === "sources";
  const row2 = top + 92;
  const dupIdx = r.removedDup ? r.removedDup - 1 : -1;
  return (
    <g>
      <text x={8} y={top + 12} fontSize={10} letterSpacing={0.4} fill="var(--muted)">PROMPT PASSAGES (WHAT THE GENERATOR READ, LABELS IT CITED)</text>
      {r.prompt.map((p, i) => (
        <motion.g key={`p${i}-${anim ? animKey : "s"}`} initial={anim ? { opacity: 0, y: 8 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.18 * k, duration: 0.3 * k }}>
          <rect x={x0 + i * (chipW + gap)} y={top + 20} width={chipW} height={36} rx={6} fill="var(--surface-2)" stroke={i === dupIdx ? "var(--warn)" : "var(--line)"} />
          <text x={x0 + i * (chipW + gap) + 8} y={top + 35} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">[{p.label}] {p.chunkId}</text>
          <text x={x0 + i * (chipW + gap) + 8} y={top + 49} fontSize={9} fill={p.truncated ? "var(--err)" : i === dupIdx ? "var(--warn)" : "var(--muted)"}>
            {p.truncated ? `✂ truncated at ${p.words} words` : i === dupIdx ? `duplicate (via ${p.via})` : `via ${p.via} search`}
          </text>
        </motion.g>
      ))}
      {phase !== "answer" && <>
      <text x={8} y={row2 - 12} fontSize={10} letterSpacing={0.4} fill="var(--muted)">
        {r.params.dedupeFirst ? "SOURCE LIST SHOWN TO THE USER (SAME LABELS)" : `SOURCE LIST SHOWN TO THE USER (AFTER DEDUPE${r.params.bug === "dedupe" ? ", RENUMBERED, MARKERS NOT REMAPPED" : ", MARKERS REMAPPED BY CHUNK ID"})`}
      </text>
      {r.display.map((p) => {
        const from = r.prompt.findIndex((x) => x.chunkId === p.chunkId);
        const to = p.label - 1;
        const remapped = r.params.bug !== "dedupe" && !r.params.dedupeFirst && from !== to;
        return (
          <motion.g key={`d${p.label}-${anim ? animKey : "s"}`} initial={anim ? { x: (from - to) * (chipW + gap), y: -72, opacity: 0.3 } : false} animate={{ x: 0, y: 0, opacity: 1 }}
            transition={{ delay: (1.4 + to * 0.25) * k, duration: 0.7 * k, ease: EASE }}>
            <rect x={x0 + to * (chipW + gap)} y={row2} width={chipW} height={36} rx={6} fill="var(--surface)" stroke={r.params.bug === "dedupe" && from !== to ? "var(--err)" : "var(--accent)"} />
            <text x={x0 + to * (chipW + gap) + 8} y={row2 + 15} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">[{p.label}] {p.chunkId}</text>
            <text x={x0 + to * (chipW + gap) + 8} y={row2 + 29} fontSize={9} fill={r.params.bug === "dedupe" && from !== to ? "var(--err)" : "var(--muted)"}>
              {from === to ? "label unchanged" : remapped ? `marker [${from + 1}] remapped → [${p.label}]` : `was [${from + 1}]; marker [${from + 1}] not updated`}
            </text>
          </motion.g>
        );
      })}
      </>}
      {dupIdx >= 0 && anim && (
        <motion.g key={`dup-${animKey}`} initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 1, 0] }} transition={{ delay: 0.9 * k, duration: 1.4 * k, times: [0, 0.2, 0.7, 1] }}>
          <line x1={x0 + dupIdx * (chipW + gap) + 6} y1={top + 38} x2={x0 + dupIdx * (chipW + gap) + chipW - 6} y2={top + 38} stroke="var(--err)" strokeWidth={2} />
          <text x={x0 + dupIdx * (chipW + gap) + chipW / 2} y={top + 72} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--err)">removed: same chunk id</text>
        </motion.g>
      )}
    </g>
  );
}

function StepCard({ r, state, k }: { r: CMResult; state: CMState; k: number }) {
  if (state.phase === "answer") {
    return (
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">#</th><th>sentence</th><th>marker</th><th>actually read from</th></tr></thead>
        <tbody>{r.citations.map((c, i) => (
          <motion.tr key={i} className="border-t border-line align-top" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + i * 0.5) * k }}>
            <td className="py-1 font-mono">{i + 1}</td><td className="pr-2">{c.claim}</td><td className="font-mono">[{c.marker}]</td><td className="font-mono">{c.intended.chunkId}</td>
          </motion.tr>
        ))}</tbody>
      </table>
    );
  }
  if (state.phase === "sources") {
    return (
      <div className="space-y-2 text-xs">
        <p>Each answer marker must keep pointing at the chunk the generator read. Here is what every marker means after this step:</p>
        <table className="w-full">
          <thead className="text-left text-muted"><tr><th className="py-1">marker in answer</th><th>should open</th><th>source-list entry with that number</th><th /></tr></thead>
          <tbody>{r.citations.map((c, i) => {
            const byPos = r.display[c.marker - 1];
            const safe = r.params.bug !== "dedupe";
            const ok = safe || byPos?.chunkId === c.intended.chunkId;
            return (
              <motion.tr key={i} className="border-t border-line" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (1.4 + i * 0.3) * k }}>
                <td className="py-1 font-mono">[{c.marker}]</td><td className="font-mono">{c.intended.chunkId}</td>
                <td className="font-mono">{safe ? `remapped by chunk id → ${c.intended.chunkId}` : byPos ? `[${c.marker}] ${byPos.chunkId}` : "none"}</td>
                <td><Badge tone={ok ? "ok" : "err"}>{ok ? "✓ consistent" : "✗ shifted"}</Badge></td>
              </motion.tr>
            );
          })}</tbody>
        </table>
      </div>
    );
  }
  if (state.phase === "resolve") {
    const c = r.citations[state.focus!];
    const rows: [string, string, boolean][] = c.resolved
      ? [
          [`resolve label [${c.marker}]`, `passage ${c.resolved.label}${r.params.bug === "offByOne" ? " (index [n] used as 0-based)" : r.params.bug === "dedupe" ? " (position in deduped list)" : ""}`, c.wrongSource],
          ["passage.chunk_id", c.resolved.chunkId, c.wrongSource],
          ["chunks[id].source", `${c.doc!.file} (${c.doc!.version})`, c.wrongSource],
          ["locate span in source", c.anchor!.page ? `page ${c.anchor!.page}, ${c.anchor!.anchor}` : c.anchor!.anchor, c.wrongSource],
        ]
      : [[`resolve label [${c.marker}]`, "∅ no passage has that label", true]];
    return (
      <div className="space-y-1.5 text-xs">
        {rows.map(([op, out, bad], j) => (
          <motion.div key={op} className="grid grid-cols-[160px_1fr_auto] items-center gap-2 rounded-md border border-line px-2 py-1" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: hopAt(j + 1) * k, duration: 0.3 * k }}>
            <span className="font-mono text-muted">{op}</span><span className="font-mono">{out}</span><Badge tone={bad ? "err" : "ok"}>{bad ? "✗" : "✓"}</Badge>
          </motion.div>
        ))}
        {c.resolved?.truncated && <p className="text-err">The cited passage text ends at “…{c.resolved.text.split(" ").slice(-5).join(" ")}”. The sentence's last part (“Seats unlimited; Support 24/7 phone”) is not in it.</p>}
      </div>
    );
  }
  if (state.phase === "verify") {
    return (
      <div className="space-y-2 text-xs">
        {r.citations.map((c, i) => (
          <motion.div key={i} className="grid gap-2 sm:grid-cols-[1fr_140px_70px] sm:items-center" initial={{ opacity: 0.3 }} animate={{ opacity: 1 }} transition={{ delay: (VERIFY0 + i * VERIFY_GAP) * k }}>
            <span><span className="font-mono text-violet">[{c.marker}]</span> {c.claim}</span>
            <div className="relative h-3 rounded-sm bg-surface-2">
              <motion.div className={cx("absolute inset-y-0 left-0 rounded-sm", c.pass ? "bg-ok" : "bg-err")} initial={{ width: 0 }} animate={{ width: `${(c.check?.coverage ?? 0) * 100}%` }} transition={{ delay: (VERIFY0 + i * VERIFY_GAP) * k, duration: 0.5 * k, ease: EASE }} />
              <div className="absolute inset-y-[-3px] w-0.5 bg-fg" style={{ left: "60%" }} />
            </div>
            <Badge tone={c.pass ? "ok" : "err"}>{c.pass ? "✓ pass" : "✗ fail"}</Badge>
          </motion.div>
        ))}
        <p className="text-muted">Bar = share of the sentence's words found {r.params.strict ? "in the best single sentence" : "anywhere"} of the cited passage; tick = 60%.{r.params.strict && " Strict mode also fails a claim whose numbers are missing."}</p>
      </div>
    );
  }
  const passes = r.citations.filter((c) => c.pass).length;
  return (
    <div className="space-y-2 text-xs">
      <div className="flex justify-between"><span>citation accuracy = {passes}/{r.citations.length}</span><span className="font-mono"><AnimatedNumber value={r.accuracy * 100} from={0} digits={0} delay={0.3 * k} duration={0.9 * k} />%</span></div>
      <div className="h-3 overflow-hidden rounded-sm bg-surface-2"><motion.div className={cx("h-full", r.accuracy === 1 ? "bg-ok" : "bg-err")} initial={{ width: 0 }} animate={{ width: `${r.accuracy * 100}%` }} transition={{ delay: 0.3 * k, duration: 0.9 * k, ease: EASE }} /></div>
      <p className="text-muted">The answer text was identical in every configuration. {r.params.bug === "truncated" && !r.params.strict ? "The lenient verifier passed the truncated citation: its words overlap the passage even though the numbers aren't there." : ""}</p>
    </div>
  );
}

function Calculation({ r, state }: { r: CMResult; state: CMState }) {
  if (state.phase === "resolve") {
    const c = r.citations[state.focus!];
    const idx = r.params.bug === "offByOne" ? `P[${c.marker}]\\ (\\text{0-based})` : r.params.bug === "dedupe" ? `D[${c.marker}]\\ (\\text{deduped list})` : `P[${c.marker}]`;
    return <TexBlock>{`[${c.marker}] \\mapsto ${idx} \\mapsto ${c.resolved ? `\\texttt{${c.resolved.chunkId.replace("#", "\\#")}}` : "\\varnothing"}`}</TexBlock>;
  }
  if (state.phase === "verify") {
    return (
      <div className="space-y-2">
        <TexBlock>{"\\text{pass}_i = \\Big[\\tfrac{|C_i \\cap S_i|}{|C_i|} \\ge 0.6\\Big]" + (r.params.strict ? "\\wedge \\text{numbers}(C_i) \\subseteq S_i" : "")}</TexBlock>
        <p className="text-[11px] text-muted">C = the sentence's words, S = {r.params.strict ? "the best sentence of the cited passage" : "the whole cited passage"}.</p>
      </div>
    );
  }
  const passes = r.citations.filter((c) => c.pass).length;
  return <TexBlock>{state.phase === "score" ? `\\text{citation accuracy} = \\frac{\\#\\text{pass}}{\\#\\text{citations}} = \\frac{${passes}}{${r.citations.length}} = ${r.accuracy.toFixed(2)}` : "\\text{citation accuracy} = \\frac{\\#\\text{pass}}{\\#\\text{citations}}"}</TexBlock>;
}
