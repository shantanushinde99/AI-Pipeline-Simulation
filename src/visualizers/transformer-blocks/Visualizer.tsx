import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { BarStrip, HBarRaw, StepHeader, ease } from "../self-attention/shared";
import type { Vec } from "../self-attention/attention";
import { DEFAULT_PARAMS, TOKENS, W, gelu, run, simulate, type BlockParams, type BlockState, type Phase } from "./simulation";

const BOXES: { phase: Phase; label: string }[] = [
  { phase: "input", label: "x" }, { phase: "ln1", label: "LN" }, { phase: "attn", label: "Attn" }, { phase: "add1", label: "⊕" }, { phase: "ln2", label: "LN" },
  { phase: "up", label: "4→8" }, { phase: "gelu", label: "GELU" }, { phase: "down", label: "8→4" }, { phase: "add2", label: "⊕" }, { phase: "output", label: "out" },
];
const PX = (i: number) => 50 + i * 80;
const DURATIONS: Record<Phase, number> = { input: 2400, ln1: 3200, attn: 3000, add1: 3200, ln2: 3200, up: 2800, gelu: 3000, down: 2800, add2: 3200, output: 3400 };
const narrate = (s: BlockState, residual: boolean): { headline: string; sub: string } => ({
  input: { headline: "Three tokens enter the block as 4-number vectors", sub: "This running vector per token is called the residual stream." },
  ln1: { headline: "LayerNorm: centre and rescale each token", sub: "Subtract the token's own mean, divide by its standard deviation. Watch the dashed mean line." },
  attn: { headline: "Self-attention mixes information across tokens", sub: "Each token takes a weighted blend of all three, then W_O projects it back." },
  add1: residual ? { headline: "Residual add: stream + attention output", sub: "The update is stacked on top of the untouched input, not swapped in." } : { headline: "No residual: attention output replaces the stream", sub: "Break-it mode: the original token vector is thrown away here." },
  ln2: { headline: "LayerNorm again before the MLP", sub: "Same recipe: mean 0, variance 1 for every token." },
  up: { headline: "MLP expands each token from 4 to 8 numbers", sub: "A wider hidden layer gives room for feature detectors." },
  gelu: { headline: "GELU bends the 8 numbers", sub: "Positives pass, negatives are squashed toward 0 (smoothly)." },
  down: { headline: "MLP projects back down to 4", sub: "Now it has the same size as the stream and can be added." },
  add2: residual ? { headline: "Residual add: stream + MLP output", sub: "Second update stacked onto the stream." } : { headline: "No residual: MLP output replaces the stream", sub: "Each sub-layer must now carry everything forward by itself." },
  output: residual ? { headline: "Block output: the stream, updated twice", sub: "Tokens keep their identity and have absorbed context." } : { headline: "Without residuals the tokens blur together", sub: "Compare the health numbers below with residuals on." },
})[s.phase];

export default function TransformerBlockVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<BlockParams>(DEFAULT_PARAMS);
  const [tok, setTok] = useState(0);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(params), [params]);
  const pb = usePlayback(steps.length, (i) => DURATIONS[steps[i]?.state.phase ?? "input"]);
  const step = steps[pb.index];
  const st = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const healthOn = useMemo(() => run({ ...params, residual: true }).health, [params]);
  const healthOff = useMemo(() => run({ ...params, residual: false }).health, [params]);
  const story = narrate(st, params.residual);
  if (tab === "metrics") return null;
  const cur = params.residual ? healthOn : healthOff;

  return (
    <Workspace
      main={
        <>
          <Panel title="One transformer block, end to end" actions={<SimulatedLabel>hand-set weights, 3 tokens × d = 4</SimulatedLabel>}>
            <StepHeader index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[660px]">
                <BlockSvg st={st} residual={params.residual} blocks={params.blocks} k={k} animKey={animKey} prevPhase={steps[Math.max(0, pb.index - 1)].state.phase} />
              </div>
            </div>
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>each strip = one token's vector (bar up = positive, <span className="text-err">red</span> = negative)</span>
              <span className="text-violet">violet = sub-layer output being added</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <Card st={st} tok={tok} k={k} residual={params.residual} healthOn={healthOn} healthOff={healthOff} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_PARAMS); setTok(0); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Blocks stacked" value={String(params.blocks)} options={[{ value: "1", label: "1 block (10 steps)" }, { value: "2", label: "2 blocks (18 steps)" }]} onChange={(v) => setParams({ ...params, blocks: v === "2" ? 2 : 1 })} />
              <Select label="Inspect token" value={String(tok)} options={TOKENS.map((t, i) => ({ value: String(i), label: t }))} onChange={(v) => setTok(Number(v))} />
              <Toggle label="Residual connections" checked={params.residual} onChange={(v) => setParams({ ...params, residual: v })} hint="Turn off (break it) and compare the output step." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title={`Calculation · "${TOKENS[tok]}"`}><Calculation st={st} tok={tok} residual={params.residual} /></Panel>
          <Panel title="Signal health (final output)">
            <KV rows={[
              ["cos(output, input), mean", <span className={cur.meanKeep < 0.6 ? "text-err" : "text-ok"}>{cur.meanKeep.toFixed(3)}</span>],
              ["similarity between tokens", <span className={cur.tokenSimilarity > 0.8 ? "text-err" : "text-ok"}>{cur.tokenSimilarity.toFixed(3)}</span>],
              ["similarity at input", healthOn.inputSimilarity.toFixed(3)],
              ["residuals", params.residual ? "on" : "off (break it)"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const ROW_Y = (t: number) => 128 + t * 112;
const STRIP_H = 72;

function BlockSvg({ st, residual, blocks, k, animKey, prevPhase }: { st: BlockState; residual: boolean; blocks: number; k: number; animKey: string; prevPhase: Phase }) {
  const bi = BOXES.findIndex((b) => b.phase === st.phase);
  const pi = BOXES.findIndex((b) => b.phase === prevPhase);
  const max = Math.max(1, ...st.before.flat().map(Math.abs), ...st.after.flat().map(Math.abs), ...(st.branch?.flat().map(Math.abs) ?? []));
  const sameDim = st.before[0].length === st.after[0].length;
  const isAdd = st.phase === "add1" || st.phase === "add2";
  const isLN = st.phase === "ln1" || st.phase === "ln2";
  const w = (n: number) => (n === 8 ? 12 : 22);
  const g = (n: number) => (n === 8 ? 3 : 6);
  const width = (n: number) => n * (w(n) + g(n));
  return (
    <svg viewBox="0 0 820 470" className="h-auto w-full" role="img" aria-label={`Transformer block stage: ${st.phase}`}>
      {/* pipeline diagram */}
      {blocks === 2 && st.block >= 0 && <text x={8} y={14} fontSize={11} fontWeight={700} fill="var(--accent)">block {st.block + 1} of 2</text>}
      {[[0, 3], [3, 8]].map(([a, b]) => {
        const x1 = PX(a) + 24, x2 = PX(b);
        return (
          <g key={a}>
            <path d={`M ${x1} 40 C ${x1} 14, ${x2} 14, ${x2} 40`} fill="none" stroke={residual ? "var(--ok)" : "var(--err)"} strokeWidth={2} strokeDasharray={residual ? undefined : "4 4"} />
            <text x={(x1 + x2) / 2} y={18} textAnchor="middle" fontSize={10} fill={residual ? "var(--ok)" : "var(--err)"}>{residual ? "residual (skip)" : "✗ residual removed"}</text>
          </g>
        );
      })}
      {BOXES.map((b, i) => (
        <g key={b.phase}>
          {i < BOXES.length - 1 && <line x1={PX(i) + 30} x2={PX(i + 1) - 30} y1={56} y2={56} stroke="var(--line)" strokeWidth={1.5} />}
          <rect x={PX(i) - 30} y={41} width={60} height={30} rx={7} fill={i === bi ? "var(--accent)" : "var(--surface)"} stroke={i === bi ? "var(--accent)" : "var(--line)"} />
          <text x={PX(i)} y={61} textAnchor="middle" fontSize={12} fontWeight={600} fill={i === bi ? "var(--surface)" : "var(--fg)"}>{b.label}</text>
        </g>
      ))}
      <motion.circle key={`${animKey}-dot`} r={6} cy={56} fill="var(--violet)" initial={{ cx: PX(pi >= 0 && pi !== bi ? pi : bi), opacity: 1 }} animate={{ cx: PX(bi), opacity: [1, 1, 0] }} transition={{ duration: 0.6 * k, ease }} />

      {/* per-token strips */}
      <text x={110 + 50} y={108} textAnchor="middle" fontSize={11} fill="var(--muted)">{st.phase === "output" ? "block input" : isAdd ? "stream (before)" : "in"}</text>
      <text x={isAdd ? 360 : 480} y={108} textAnchor="middle" fontSize={11} fill={isAdd ? "var(--violet)" : "var(--muted)"}>{isAdd ? "sub-layer output" : st.phase === "output" ? "block output" : "out"}</text>
      {isAdd && <text x={600} y={108} textAnchor="middle" fontSize={11} fill="var(--muted)">{residual ? "stream + output" : "output only"}</text>}
      {TOKENS.map((t, i) => {
        const y = ROW_Y(i);
        const nb = st.before[i].length, na = st.after[i].length;
        const s = st.stats?.[i];
        const mid = y + STRIP_H / 2;
        const sc = STRIP_H / 2 / max;
        return (
          <g key={`${animKey}-${i}`}>
            <text x={20} y={mid + 4} fontSize={13} fontWeight={600} fill="var(--fg)">{t}</text>
            <BarStrip x={110 + 50 - width(nb) / 2} y={y} values={st.before[i]} max={max} w={w(nb)} gap={g(nb)} h={STRIP_H} k={k} delay={0} stagger={0} />
            {isLN && s && (
              <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 * k }}>
                <line x1={160 - width(nb) / 2 - 4} x2={160 + width(nb) / 2} y1={mid - s.mu * sc} y2={mid - s.mu * sc} stroke="var(--warn)" strokeWidth={2} strokeDasharray="4 3" />
                <text x={160 + width(nb) / 2 + 6} y={mid - s.mu * sc + 4} fontSize={10} fill="var(--warn)">μ {s.mu.toFixed(2)}</text>
                <text x={160} y={y + STRIP_H + 13} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">σ² = {s.variance.toFixed(2)}</text>
              </motion.g>
            )}
            {isAdd ? (
              <>
                <BarStrip x={360 - width(4) / 2} y={y} values={st.branch![i]} max={max} w={22} gap={6} h={STRIP_H} k={k} delay={0.3} color="var(--violet)" />
                <text x={260} y={mid + 6} textAnchor="middle" fontSize={20} fill="var(--muted)">+</text>
                <text x={480} y={mid + 6} textAnchor="middle" fontSize={20} fill="var(--muted)">=</text>
                {/* the branch slides over and lands on the stream */}
                <motion.g initial={{ x: 0, opacity: 0 }} animate={{ x: [0, 240], opacity: [0, 0.8, 0] }} transition={{ delay: 0.9 * k, duration: 0.9 * k, ease }}>
                  <BarStrip x={360 - width(4) / 2} y={y} values={st.branch![i]} max={max} w={22} gap={6} h={STRIP_H} k={0} color="var(--violet)" />
                </motion.g>
                <BarStrip x={600 - width(4) / 2} y={y} values={st.after[i]} from={residual ? st.before[i] : undefined} max={max} w={22} gap={6} h={STRIP_H} k={k} delay={1.6} />
              </>
            ) : (
              <>
                <motion.path d={`M 290 ${mid} L 370 ${mid}`} stroke="var(--muted)" strokeWidth={2} fill="none" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3 * k, duration: 0.5 * k }} />
                <path d={`M 364 ${mid - 5} L 372 ${mid} L 364 ${mid + 5}`} fill="none" stroke="var(--muted)" strokeWidth={2} />
                <text x={330} y={mid - 8} textAnchor="middle" fontSize={10} fill="var(--muted)">{st.phase === "up" ? "W₁x+b₁" : st.phase === "down" ? "W₂h+b₂" : st.phase === "gelu" ? "GELU" : isLN ? "(x−μ)/σ" : st.phase === "attn" ? "attend" : ""}</text>
                <BarStrip x={480 - width(na) / 2} y={y} values={st.after[i]} from={sameDim ? st.before[i] : undefined} max={max} w={w(na)} gap={g(na)} h={STRIP_H} k={k} delay={0.8} />
                {st.phase === "output" && <text x={600} y={mid + 4} fontSize={11} fontFamily="var(--font-mono)" fill="var(--muted)">cos = {cosOf(st.before[i], st.after[i]).toFixed(2)}</text>}
              </>
            )}
          </g>
        );
      })}
      {st.phase === "attn" && st.attn && (
        <g>
          <text x={700} y={120} textAnchor="middle" fontSize={11} fill="var(--muted)">attention weights</text>
          {st.attn.map((row, i) => row.map((v, j) => (
            <motion.g key={`${animKey}-a${i}${j}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.3 + i * 0.25 + j * 0.06) * k }}>
              <rect x={640 + j * 40} y={130 + i * 40} width={38} height={38} rx={3} fill="var(--accent)" fillOpacity={0.1 + v * 0.8} />
              <text x={659 + j * 40} y={153 + i * 40} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={v > 0.5 ? "var(--surface)" : "var(--fg)"}>{v.toFixed(2)}</text>
            </motion.g>
          )))}
          {TOKENS.map((t, j) => <text key={t} x={659 + j * 40} y={262} textAnchor="middle" fontSize={10} fill="var(--muted)">{t}</text>)}
        </g>
      )}
    </svg>
  );
}

const cosOf = (a: Vec, b: Vec) => {
  const d = a.reduce((s, v, i) => s + v * b[i], 0);
  return d / (Math.hypot(...a) * Math.hypot(...b) || 1);
};

function Card({ st, tok, k, residual, healthOn, healthOff }: { st: BlockState; tok: number; k: number; residual: boolean; healthOn: ReturnType<typeof run>["health"]; healthOff: ReturnType<typeof run>["health"] }) {
  const x = st.before[tok], y = st.after[tok];
  const row = (cells: ReactNode[], key: string | number, delay: number) => (
    <motion.tr key={key} className="border-t border-line" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: delay * k, duration: 0.3 * k }}>
      {cells.map((c, i) => <td key={i} className={cx("py-0.5", i > 0 && "font-mono")}>{c}</td>)}
    </motion.tr>
  );
  const table = (head: string[], body: ReactNode) => (
    <div className="overflow-x-auto"><table className="w-full text-xs"><thead className="text-left text-muted"><tr>{head.map((h, i) => <th key={i} className="py-1 font-medium">{h}</th>)}</tr></thead><tbody>{body}</tbody></table></div>
  );
  const f = (v: Vec) => v.map((n) => n.toFixed(2));
  switch (st.phase) {
    case "input":
      return table(["token", "d0", "d1", "d2", "d3"], st.after.map((v, i) => row([TOKENS[i], ...f(v)], i, 0.2 + i * 0.3)));
    case "ln1":
    case "ln2": {
      const s = st.stats![tok];
      return (
        <div className="space-y-2">
          {table([`"${TOKENS[tok]}"`, "d0", "d1", "d2", "d3", ""], [
            row(["x", ...f(x), ""], "x", 0.1),
            row(["x − μ", ...f(x.map((v) => v - s.mu)), `μ = ${s.mu.toFixed(3)}`], "c", 0.6),
            row(["(x − μ)/σ", ...f(y), `σ = ${Math.sqrt(s.variance).toFixed(3)}`], "n", 1.1),
          ])}
          <p className="text-xs text-muted">Every token now has mean 0 and variance 1, whatever its original scale. γ = 1 and β = 0 here (real models learn them).</p>
        </div>
      );
    }
    case "attn":
      return (
        <div className="space-y-2">
          {table(["query ↓ / key →", ...TOKENS], st.attn!.map((r, i) => row([TOKENS[i], ...r.map((v) => `${(v * 100).toFixed(0)}%`)], i, 0.3 + i * 0.25)))}
          <p className="text-xs text-muted">Weights are fairly even here (small, untrained W_Q and W_K), so every token gets a similar blend. That is exactly why the residual matters next.</p>
        </div>
      );
    case "add1":
    case "add2": {
      const n = (v: Vec) => Math.hypot(...v);
      return (
        <div className="space-y-1.5">
          {TOKENS.map((t, i) => (
            <div key={t} className="grid grid-cols-[40px_1fr_1fr_1fr] items-center gap-2 text-xs">
              <span>{t}</span>
              <div><div className="text-[10px] text-muted">|stream| {n(st.before[i]).toFixed(2)}</div><HBarRaw v={n(st.before[i])} max={4} at={0.1} k={k} tone="bg-line" /></div>
              <div><div className="text-[10px] text-violet">|update| {n(st.branch![i]).toFixed(2)}</div><HBarRaw v={n(st.branch![i])} max={4} at={0.4} k={k} tone="bg-violet/70" /></div>
              <div><div className="text-[10px] text-muted">|result| {n(st.after[i]).toFixed(2)}</div><HBarRaw v={n(st.after[i])} max={4} at={1.6} k={k} /></div>
            </div>
          ))}
          <p className="text-xs text-muted">{residual ? "The update is added, so the original direction of each token survives (cosine with the stream stays high)." : "With no skip path the result is the update alone; the stream's information has to pass through the sub-layer or is lost."}</p>
        </div>
      );
    }
    case "up":
    case "down":
      return (
        <div className="space-y-1">
          <p className="text-xs text-muted">"{TOKENS[tok]}": {st.phase === "up" ? "4 inputs → 8 hidden pre-activations" : "8 activations → 4 outputs"}</p>
          {y.map((v, j) => (
            <div key={j} className="grid grid-cols-[50px_1fr_56px] items-center gap-2 text-xs"><span className="text-muted">{st.phase === "up" ? "h" : "o"}{j}</span><HBarRaw v={v} max={Math.max(...y.map(Math.abs), 1e-9)} at={0.8 + j * 0.06} k={k} /><span className="text-right font-mono">{v.toFixed(3)}</span></div>
          ))}
        </div>
      );
    case "gelu":
      return <GeluPlot z={x} k={k} tok={TOKENS[tok]} />;
    case "output":
      return (
        <div className="space-y-2">
          {table(["", "residuals on", "residuals off"], [
            row(["cos(output, input), mean", healthOn.meanKeep.toFixed(3), healthOff.meanKeep.toFixed(3)], "k", 0.2),
            row(["similarity between tokens", healthOn.tokenSimilarity.toFixed(3), healthOff.tokenSimilarity.toFixed(3)], "s", 0.5),
            row(["similarity at input", healthOn.inputSimilarity.toFixed(3), healthOn.inputSimilarity.toFixed(3)], "i", 0.8),
          ])}
          <p className="text-xs text-muted">{residual ? <>Now toggle residuals off and replay: <Badge tone="warn">token similarity → {healthOff.tokenSimilarity.toFixed(2)}</Badge> the three tokens collapse toward the same vector.</> : <>Attention averaging plus LayerNorm, with no skip path, drives the tokens toward one vector (similarity {healthOff.tokenSimilarity.toFixed(2)}). Deep stacks without residuals also lose the gradient highway.</>}</p>
        </div>
      );
  }
}

function GeluPlot({ z, k, tok }: { z: Vec; k: number; tok: string }) {
  const Wd = 320, H = 160, P = 22, R = Math.max(2.5, ...z.map((v) => Math.abs(v) + 0.3));
  const sx = (v: number) => P + ((v + R) / (2 * R)) * (Wd - 2 * P);
  const sy = (v: number) => H - P - ((v + 0.5) / (R + 0.5)) * (H - 2 * P);
  const xs = Array.from({ length: 81 }, (_, i) => -R + (2 * R * i) / 80);
  const d = xs.map((v, i) => `${i ? "L" : "M"}${sx(v).toFixed(1)} ${sy(gelu(v)).toFixed(1)}`).join(" ");
  return (
    <div className="grid items-center gap-3 sm:grid-cols-[320px_1fr]">
      <svg viewBox={`0 0 ${Wd} ${H}`} className="h-auto w-full max-w-[320px]" role="img" aria-label="GELU curve with this token's 8 values">
        <line x1={P} x2={Wd - P} y1={sy(0)} y2={sy(0)} stroke="var(--line)" />
        <line x1={sx(0)} x2={sx(0)} y1={P / 2} y2={H - P} stroke="var(--line)" />
        <line x1={sx(-0.5)} y1={sy(-0.5)} x2={sx(R)} y2={sy(R)} stroke="var(--muted)" strokeDasharray="3 3" strokeOpacity={0.5} />
        <motion.path d={d} fill="none" stroke="var(--violet)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.7 * k }} />
        {z.map((v, j) => (
          <motion.circle key={j} cx={sx(v)} r={4.5} fill={v < 0 ? "var(--err)" : "var(--accent)"} initial={{ cy: sy(Math.max(-0.5, v)) }} animate={{ cy: sy(gelu(v)) }} transition={{ delay: (0.9 + j * 0.1) * k, duration: 0.6 * k, ease }} />
        ))}
        <text x={Wd - P} y={sy(0) + 14} textAnchor="end" fontSize={10} fill="var(--muted)">z</text>
        <text x={sx(0) + 4} y={P / 2 + 6} fontSize={10} fill="var(--muted)">GELU(z)</text>
      </svg>
      <p className="text-sm leading-relaxed text-muted">"{tok}"'s 8 hidden values start on the dashed line y = z and drop onto the GELU curve. <span className="text-err">Negative</span> values shrink toward 0; positive values barely change. {z.filter((v) => v < 0).length} of 8 are negative here. <AnimatedNumber value={z.reduce((s, v) => s + gelu(v), 0)} from={z.reduce((s, v) => s + v, 0)} delay={0.9 * k} duration={1.4 * k} /> = sum after GELU.</p>
    </div>
  );
}

function Calculation({ st, tok, residual }: { st: BlockState; tok: number; residual: boolean }) {
  const x = st.before[tok], y = st.after[tok];
  const v = (a: Vec, d = 2) => `[${a.map((n) => n.toFixed(d)).join(",\\ ")}]`;
  switch (st.phase) {
    case "input": return <TexBlock>{`x_{\\text{${TOKENS[tok]}}} = ${v(y)}`}</TexBlock>;
    case "ln1": case "ln2": {
      const s = st.stats![tok];
      return <TexBlock>{`\\begin{aligned} \\mu &= \\tfrac{1}{4}\\textstyle\\sum_i x_i = ${s.mu.toFixed(3)} \\\\ \\sigma^2 &= \\tfrac{1}{4}\\textstyle\\sum_i (x_i - \\mu)^2 = ${s.variance.toFixed(3)} \\\\ \\hat{x} &= \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} = ${v(y)} \\end{aligned}`}</TexBlock>;
    }
    case "attn": return <TexBlock>{`\\begin{aligned} a &= W_O \\operatorname{softmax}\\!\\left(\\tfrac{QK^\\top}{\\sqrt{4}}\\right) V \\\\ a_{\\text{${TOKENS[tok]}}} &= ${v(y)} \\end{aligned}`}</TexBlock>;
    case "add1": case "add2": return <TexBlock>{residual ? `x + f(x) = ${v(x)} + ${v(st.branch![tok])} = ${v(y)}` : `\\text{no residual: } f(x) = ${v(y)}`}</TexBlock>;
    case "up": return <TexBlock>{`h = W_1 \\hat{x} + b_1,\\quad W_1 \\in \\mathbb{R}^{${W.W1.length} \\times 4}`}</TexBlock>;
    case "gelu": return <TexBlock>{`\\operatorname{GELU}(z) \\approx \\tfrac{z}{2}\\left(1 + \\tanh\\!\\left[\\sqrt{2/\\pi}\\,(z + 0.044715 z^3)\\right]\\right)`}</TexBlock>;
    case "down": return <TexBlock>{`m = W_2\\, \\operatorname{GELU}(h) + b_2 = ${v(y)}`}</TexBlock>;
    case "output": return <TexBlock>{`\\cos(\\text{out}, x) = ${cosOf(x, y).toFixed(3)}`}</TexBlock>;
  }
}
