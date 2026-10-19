import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw, Scissors } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, KV, NumberField, Panel, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption, useFrames } from "../neurons-and-activation-functions/kit";
import { CHECKPOINTS, DEFAULTS, LOGITS, PHASES, PROMPT, VOCAB, entropyBits, simulate, type Phase, type Pipeline, type SampleConfig } from "./simulation";

/* one schedule (seconds at 1×) */
const T = { morph: 0.2, morphDur: 0.8, fade: 0.5, fadeStep: 0.12, strip: 0.2, drop: 0.8, hit: 1.6, frames: 0.4, frameMs: 130 };
const DUR: Record<Phase, number> = { logits: 2200, scale: 2500, exp: 2500, normalize: 2700, topk: 2500, topp: 2900, renorm: 2500, sample: 3200, repeat: 4000 };
const H = 150;

function chartValues(phase: Phase, pl: Pipeline): number[] {
  switch (phase) {
    case "logits": return LOGITS;
    case "scale": return pl.scaled ?? pl.expd;
    case "exp": return pl.expd;
    case "normalize": case "topk": return pl.probs;
    case "topp": return pl.afterK;
    default: return pl.final;
  }
}

function narrate(phase: Phase, pl: Pipeline) {
  const { cfg } = pl;
  switch (phase) {
    case "logits": return { headline: `The model scores every possible next word after "${PROMPT}…"`, sub: "These raw scores (logits) are not probabilities yet: they can be any size and even negative." };
    case "scale": return cfg.T === 0 ? { headline: "T = 0: the top score takes everything", sub: "Dividing by a temperature that shrinks to 0 blows the gaps up to infinity. This is greedy decoding." }
      : { headline: `Divide every score by T = ${cfg.T}: gaps ${cfg.T < 1 ? "stretch" : cfg.T > 1 ? "shrink" : "stay the same"}`, sub: cfg.T < 1 ? "Low temperature makes the favourite even more dominant." : cfg.T > 1 ? "High temperature flattens the field: unlikely words get a real chance." : "T = 1 leaves the model's scores unchanged." };
    case "exp": return { headline: "Exponentiate: every score becomes positive", sub: "Bigger score → exponentially bigger number. The top word is 1 after subtracting the max." };
    case "normalize": return { headline: `Divide by the sum: probabilities that add to 1 ("${VOCAB[0]}" = ${(pl.probs[0] * 100).toFixed(1)}%)`, sub: `That's softmax. Spread (entropy): ${entropyBits(pl.probs).toFixed(2)} bits.` };
    case "topk": return cfg.topK === 0 || cfg.topK >= VOCAB.length ? { headline: "Top-k is off: every word stays in the running", sub: "Set top-k below to cut the long tail." } : { headline: `Top-k = ${cfg.topK}: everything outside the top ${cfg.topK} is cut`, sub: "A fixed number of candidates, no matter how confident the model is." };
    case "topp": return cfg.topP >= 1 ? { headline: "Top-p is off (p = 1)", sub: "Set top-p below 1 to keep only the most likely words that cover that much probability." } : { headline: `Top-p = ${cfg.topP}: keep the fewest words that cover ${(cfg.topP * 100).toFixed(0)}% of the probability`, sub: "The nucleus adapts: few words when the model is sure, more when it isn't." };
    case "renorm": return { headline: `The survivors are rescaled to sum to 1 again`, sub: `They held ${(pl.kept * 100).toFixed(1)}% of the probability; each is divided by ${pl.kept.toFixed(3)}.` };
    case "sample": return { headline: `A random number u = ${pl.u.toFixed(3)} lands on "${VOCAB[pl.pick]}"`, sub: pl.pick === pl.greedy ? "Here sampling agreed with greedy. Change the seed to see other draws." : `Greedy would always say "${VOCAB[pl.greedy]}"; sampling can pick other words.` };
    case "repeat": return { headline: `${pl.cfg.n} draws: the histogram converges to the distribution`, sub: "Each bar is how often that word was drawn; the dashed outline is its probability." };
  }
}

export default function SamplingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<SampleConfig>(DEFAULTS);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(steps.length, (i) => DUR[PHASES[i]] ?? 2500);
  if (tab === "metrics") return null;

  const step = steps[pb.index];
  const { phase, pl } = step.state;
  const pi = PHASES.indexOf(phase);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(phase, pl);
  const set = (p: Partial<SampleConfig>) => setCfg((c) => ({ ...c, ...p }));
  const faded = pi >= PHASES.indexOf("topp") ? pl.mask.map((m) => !m) : phase === "topk" ? pl.keepK.map((m) => !m) : VOCAB.map(() => false);
  const domain: [number, number] = pi <= 1 ? [Math.min(0, ...chartValues(phase, pl)), Math.max(0.5, ...chartValues(phase, pl))] : [0, 1];

  return (
    <Workspace
      main={
        <>
          <Panel title="Next-token distribution" actions={<SimulatedLabel>fixed logits, not a live model</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((v) => v + 1)} />
            <p className="mb-2 font-mono text-xs text-muted">“{PROMPT} ▢”</p>
            {phase === "repeat" ? <Histogram key={animKey} pl={pl} k={k} /> : (
              <BarChart key={animKey} values={chartValues(phase, pl)} from={pi === 0 ? VOCAB.map(() => 0) : chartValues(PHASES[pi - 1], pl)} domain={domain} faded={faded}
                fadeAnimate={phase === "topk" || phase === "topp"} highlight={phase === "sample" ? pl.pick : null} k={k} digits={pi <= 1 ? 2 : 3} />
            )}
            {phase === "sample" && <Strip key={`${animKey}-strip`} pl={pl} k={k} />}
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard phase={phase} pl={pl} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Decoding settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULTS); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Slider label="Temperature T" value={cfg.T} min={0} max={2} step={0.05} format={(v) => (v === 0 ? "0 (greedy)" : v.toFixed(2))} onChange={(T) => set({ T })} hint="Drag it on any step: the bars reshape live." />
              <Slider label="Top-k" value={cfg.topK} min={0} max={VOCAB.length} onChange={(topK) => set({ topK })} format={(v) => (v === 0 || v >= VOCAB.length ? "off" : String(v))} />
              <Slider label="Top-p (nucleus)" value={cfg.topP} min={0.05} max={1} step={0.05} onChange={(topP) => set({ topP })} format={(v) => (v >= 1 ? "off (1.00)" : v.toFixed(2))} />
              <NumberField label="Random seed" value={cfg.seed} step={1} min={0} max={99999} onChange={(seed) => set({ seed: Math.round(seed) })} />
              <Slider label="Draws in the histogram" value={cfg.n} min={10} max={1000} step={10} onChange={(n) => set({ n })} />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Presets:</span>
              <Button size="sm" onClick={() => setCfg({ ...DEFAULTS, T: 0 })}>Greedy (T = 0)</Button>
              <Button size="sm" onClick={() => setCfg({ ...DEFAULTS, T: 0.5, topP: 0.9 })}>Focused (T 0.5, p 0.9)</Button>
              <Button size="sm" onClick={() => setCfg({ ...DEFAULTS, T: 2 })}>Break it: T = 2</Button>
              <Button size="sm" onClick={() => setCfg({ ...DEFAULTS, topK: 3 })}>Top-k = 3</Button>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calculation phase={phase} pl={pl} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["temperature", pl.cfg.T === 0 ? "0 (greedy)" : pl.cfg.T],
              ["P(sunny) after softmax", pl.probs[0].toFixed(3)],
              ["entropy of final distribution", `${entropyBits(pl.final).toFixed(2)} bits`],
              ["tokens still in play", pl.mask.filter(Boolean).length],
              ["sampled / greedy", `${VOCAB[pl.pick]} / ${VOCAB[pl.greedy]}`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- morphing bar chart (HTML so labels stay readable on phones) ---------- */
function geom(v: number, [lo, hi]: [number, number]) {
  const span = hi - lo || 1;
  const zero = (hi / span) * H;
  const h = (Math.abs(v) / span) * H;
  return { top: v >= 0 ? zero - h : zero, height: Math.max(h, v === 0 ? 0 : 1) };
}

function BarChart({ values, from, domain, faded, fadeAnimate, highlight, k, digits }: {
  values: number[]; from: number[]; domain: [number, number]; faded: boolean[]; fadeAnimate: boolean; highlight: number | null; k: number; digits: number;
}) {
  // the previous step used its own scale; approximate its geometry on this step's scale for a continuous morph
  const fromDomain: [number, number] = [Math.min(domain[0], ...from), Math.max(domain[1], ...from)];
  const zero = geom(0, domain).top;
  let rank = 0;
  return (
    <div className="grid grid-cols-8 gap-1 sm:gap-2" role="img" aria-label={`Bars: ${VOCAB.map((v, i) => `${v} ${values[i].toFixed(3)}`).join(", ")}`}>
      {VOCAB.map((w, i) => {
        const g = geom(values[i], domain);
        const g0 = geom(from[i], fromDomain);
        const cut = faded[i];
        const at = (T.fade + (cut ? rank++ : 0) * T.fadeStep) * k;
        return (
          <div key={w} className="min-w-0">
            <div className="relative rounded-sm bg-surface-2" style={{ height: H }}>
              {domain[0] < 0 && <div className="absolute inset-x-0 h-px bg-line" style={{ top: zero }} />}
              <motion.div className={cx("absolute inset-x-1 rounded-sm", highlight === i ? "bg-ok" : values[i] < 0 ? "bg-err/70" : "bg-accent")}
                initial={{ top: g0.top, height: g0.height, opacity: 1 }}
                animate={{ top: g.top, height: g.height, opacity: cut ? 0.15 : 1 }}
                transition={{ top: { delay: T.morph * k, duration: T.morphDur * k, ease: EASE }, height: { delay: T.morph * k, duration: T.morphDur * k, ease: EASE }, opacity: { delay: fadeAnimate ? at : 0, duration: 0.4 * k } }} />
              {cut && <motion.div className="absolute inset-x-0 bottom-1 flex justify-center text-err" initial={{ opacity: fadeAnimate ? 0 : 1 }} animate={{ opacity: 1 }} transition={{ delay: fadeAnimate ? at : 0 }}><Scissors size={12} aria-label="cut" /></motion.div>}
            </div>
            <div className={cx("mt-1 truncate text-center text-[10px] sm:text-xs", cut && "text-muted line-through")}>{w}</div>
            <div className="text-center font-mono text-[9px] text-muted sm:text-[11px]"><AnimatedNumber value={values[i]} from={from[i]} digits={digits} delay={T.morph * k} duration={T.morphDur * k} /></div>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- cumulative strip with a falling marker ---------- */
function Strip({ pl, k }: { pl: Pipeline; k: number }) {
  const segs = pl.final.map((p, i) => ({ i, p })).filter((s) => s.p > 0);
  return (
    <div className="mt-4">
      <div className="mb-1 flex justify-between text-[11px] text-muted"><span>Cumulative probability 0 → 1 (final distribution)</span><span className="font-mono">u = {pl.u.toFixed(4)}</span></div>
      <div className="relative pt-5">
        <motion.div className="absolute top-0 -translate-x-1/2 text-accent" initial={{ left: `${pl.u * 100}%`, y: -24, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: T.drop * k, type: "spring", stiffness: 260, damping: 14 }}>
          <svg width="14" height="16" viewBox="0 0 14 16" aria-hidden><path d="M0 0 H14 L7 16 Z" fill="currentColor" /></svg>
        </motion.div>
        <motion.div className="flex h-9 overflow-hidden rounded-md border border-line" initial={{ opacity: 0, scaleX: 0.9 }} animate={{ opacity: 1, scaleX: 1 }} transition={{ delay: T.strip * k, duration: 0.4 * k }}>
          {segs.map((s) => (
            <motion.div key={s.i} className={cx("flex items-center justify-center overflow-hidden border-r border-surface text-[10px] font-medium", s.i === pl.pick ? "text-white dark:text-black" : "text-fg")}
              style={{ width: `${s.p * 100}%`, background: `hsl(${(s.i * 41 + 210) % 360} 60% 55% / ${s.i === pl.pick ? 1 : 0.28})` }}
              initial={{ filter: "brightness(1)" }} animate={s.i === pl.pick ? { filter: ["brightness(1)", "brightness(1.4)", "brightness(1)"] } : {}} transition={{ delay: T.hit * k, duration: 0.6 * k }}>
              {s.p > 0.07 ? VOCAB[s.i] : ""}
            </motion.div>
          ))}
        </motion.div>
        <div className="mt-0.5 flex justify-between font-mono text-[10px] text-muted"><span>0</span><span>1</span></div>
      </div>
    </div>
  );
}

/* ---------- histogram of repeated draws ---------- */
function Histogram({ pl, k }: { pl: Pipeline; k: number }) {
  const f = useFrames(CHECKPOINTS, T.frameMs * k, T.frames * 1000 * k);
  const cp = pl.checkpoints[f];
  const max = Math.max(0.05, ...pl.final, ...cp.counts.map((c) => (cp.upto ? c / cp.upto : 0)));
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px] text-muted"><span>draws so far: <span className="font-mono text-fg">{cp.upto}</span> / {pl.cfg.n}</span><span>distance to target (TV): <span className="font-mono text-fg">{cp.tv.toFixed(3)}</span></span></div>
      <div className="grid grid-cols-8 gap-1 sm:gap-2">
        {VOCAB.map((w, i) => {
          const emp = cp.upto ? cp.counts[i] / cp.upto : 0;
          return (
            <div key={w} className="min-w-0">
              <div className="relative rounded-sm bg-surface-2" style={{ height: H }}>
                <div className="absolute inset-x-0.5 rounded-sm border-2 border-dashed border-fg/50" style={{ bottom: 0, height: (pl.final[i] / max) * H }} />
                <motion.div className="absolute inset-x-1.5 bottom-0 rounded-sm bg-violet" initial={{ height: 0 }} animate={{ height: (emp / max) * H }} transition={{ duration: 0.12 * k }} />
              </div>
              <div className="mt-1 truncate text-center text-[10px] sm:text-xs">{w}</div>
              <div className="text-center font-mono text-[9px] text-muted sm:text-[11px]">{cp.counts[i]}</div>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-3 rounded-sm bg-violet" />observed frequency</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-3 rounded-sm border-2 border-dashed border-fg/50" />final probability</span>
      </div>
    </div>
  );
}

/* ---------- card ---------- */
function StepCard({ phase, pl, k }: { phase: Phase; pl: Pipeline; k: number }) {
  const { cfg } = pl;
  const row = (cells: ReactNode[], i: number, dim = false) => (
    <motion.tr key={i} className={cx("border-t border-line", dim && "text-muted line-through")} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.08) * k }}>
      {cells.map((c, j) => <td key={j} className={cx("py-0.5", j > 0 && "text-right")}>{c}</td>)}
    </motion.tr>
  );
  const table = (head: string[], rows: ReactNode) => (
    <div className="overflow-x-auto"><table className="w-full min-w-[300px] font-mono text-xs"><thead className="text-left font-sans text-muted"><tr>{head.map((h, j) => <th key={h} className={cx("py-1", j > 0 && "text-right")}>{h}</th>)}</tr></thead><tbody>{rows}</tbody></table></div>
  );
  if (phase === "logits" || phase === "scale" || phase === "exp" || phase === "normalize") {
    const sum = pl.expd.reduce((a, b) => a + b, 0);
    return (
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        {table(["token", "logit z", "z / T", "e^(z/T − max)", "p"], VOCAB.slice(0, 5).map((w, i) => row([
          w, LOGITS[i].toFixed(1),
          phase === "logits" ? "·" : pl.scaled ? pl.scaled[i].toFixed(2) : "→ ∞ gap",
          ["exp", "normalize"].includes(phase) ? pl.expd[i].toFixed(3) : "·",
          phase === "normalize" ? <strong>{pl.probs[i].toFixed(3)}</strong> : "·",
        ], i)))}
        <div className="space-y-2 text-xs">
          {phase === "normalize" ? <p>Sum of the exponentials = <span className="font-mono">{sum.toFixed(3)}</span>. Each one divided by it gives p. Entropy <span className="font-mono">{entropyBits(pl.probs).toFixed(2)}</span> bits ≈ {Math.pow(2, entropyBits(pl.probs)).toFixed(1)} equally likely choices.</p>
            : phase === "logits" ? <p>The table fills in as the steps run (top 5 of 8 tokens shown). Logits come from the model's last layer; here they are fixed numbers.</p>
            : phase === "scale" ? <p>{cfg.T === 0 ? "With T = 0 we skip the division and put all probability on the top logit." : `Gap between "sunny" and "cloudy": ${(LOGITS[0] - LOGITS[1]).toFixed(1)} → ${((LOGITS[0] - LOGITS[1]) / cfg.T).toFixed(2)}.`} Drag the temperature slider and watch the bars.</p>
            : <p>e^x grows fast, so a gap of {((LOGITS[0] - LOGITS[1]) / Math.max(cfg.T, 1e-9)).toFixed(2)} in z/T becomes a ratio of {cfg.T === 0 ? "∞" : Math.exp((LOGITS[0] - LOGITS[1]) / cfg.T).toFixed(2)}× between sunny and cloudy.</p>}
        </div>
      </div>
    );
  }
  if (phase === "topk" || phase === "topp") {
    const rows = phase === "topk"
      ? pl.probs.map((p, i) => [i, p, 0, !pl.keepK[i]] as const)
      : pl.topP.cumulative.filter((c) => pl.keepK[c.i]).map((c) => [c.i, c.p, c.cum, !pl.topP.mask[c.i]] as const);
    return (
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        {table(phase === "topk" ? ["rank", "token", "p", "kept?"] : ["token", "p", "cumulative", "kept?"], rows.map(([i, p, cum, cut], r) =>
          row(phase === "topk" ? [r + 1, VOCAB[i], p.toFixed(3), cut ? "cut" : "keep"] : [VOCAB[i], p.toFixed(3), cum.toFixed(3), cut ? "cut" : "keep"], r, cut)))}
        <div className="text-xs">
          {phase === "topk" ? <p>{cfg.topK === 0 || cfg.topK >= VOCAB.length ? "Top-k is off." : `Keeps exactly ${cfg.topK} tokens even if the rest are likely, or if those ${cfg.topK} include junk.`}</p>
            : <p>{cfg.topP >= 1 ? "Top-p is off." : `Walk down the sorted list adding probabilities; stop once the running total reaches ${cfg.topP}. Everything after that is cut.`} Top-p is applied after top-k, on the renormalized top-k distribution.</p>}
        </div>
      </div>
    );
  }
  if (phase === "renorm") {
    return table(["token", "before", "÷ kept mass", "after"], VOCAB.map((w, i) => row([w, pl.probs[i].toFixed(3), pl.mask[i] ? `÷ ${pl.kept.toFixed(3)}` : "cut", pl.mask[i] ? <strong>{pl.final[i].toFixed(3)}</strong> : "0"], i, !pl.mask[i])));
  }
  if (phase === "sample") {
    const start = (i: number) => pl.final.slice(0, i).reduce((x, y) => x + y, 0);
    return (
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {table(["token", "interval", "u inside?"], VOCAB.map((w, i) => pl.final[i] > 0 ? row([w, `[${start(i).toFixed(3)}, ${start(i + 1).toFixed(3)})`, i === pl.pick ? "✓ picked" : ""], i) : null))}
        <div className="space-y-1 text-xs">
          <p>u = <span className="font-mono">{pl.u.toFixed(4)}</span> comes from a seeded generator (seed {cfg.seed}), so the same seed always gives the same draw.</p>
          <p className="text-muted">Bigger probability → wider interval → more likely to contain u. That is all “sampling from a distribution” means.</p>
        </div>
      </div>
    );
  }
  const lastCp = pl.checkpoints[CHECKPOINTS];
  return (
    <div className="space-y-2 text-xs">
      <p>Total-variation distance between observed frequencies and the final distribution after {cfg.n} draws: <strong className="font-mono"><AnimatedNumber value={lastCp.tv} from={1} digits={3} delay={T.frames * k} duration={(CHECKPOINTS * T.frameMs) / 1000 * k} /></strong>. It shrinks roughly like 1/√n.</p>
      <p className="text-muted">Try 10 draws, then 1000, with the “Draws” slider: few draws look noisy, many draws match the dashed outline.</p>
      <p className="text-muted">With T = 0 (or top-k = 1) every draw is the same word: sampling becomes greedy decoding.</p>
    </div>
  );
}

function Calculation({ phase, pl }: { phase: Phase; pl: Pipeline }) {
  const { cfg } = pl;
  const sum = pl.expd.reduce((a, b) => a + b, 0);
  const tex = (() => {
    if (phase === "logits") return `z = [${LOGITS.join(",\\ ")}]`;
    if (phase === "scale") return cfg.T === 0 ? "\\lim_{T \\to 0} \\operatorname{softmax}(z/T) = \\operatorname{onehot}(\\arg\\max z)" : `\\frac{z_{\\text{sunny}}}{T} = \\frac{${LOGITS[0]}}{${cfg.T}} = ${pl.scaled![0].toFixed(3)}`;
    if (phase === "exp") return cfg.T === 0 ? "\\text{greedy: no exponentials needed}" : `e^{(z_i - z_{\\max})/T}:\\ e^{(${LOGITS[1]} - ${LOGITS[0]})/${cfg.T}} = ${pl.expd[1].toFixed(4)}`;
    if (phase === "normalize") return `p_i = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}} \\qquad p_{\\text{sunny}} = \\frac{${pl.expd[0].toFixed(3)}}{${sum.toFixed(3)}} = ${pl.probs[0].toFixed(3)}`;
    if (phase === "topk") return cfg.topK === 0 || cfg.topK >= VOCAB.length ? "\\text{top-}k\\ \\text{off}" : `V_k = \\text{the } ${cfg.topK} \\text{ largest } p_i`;
    if (phase === "topp") return cfg.topP >= 1 ? "\\text{top-}p\\ \\text{off}" : `V_p = \\min \\{ S : \\textstyle\\sum_{i \\in S} p_i \\ge ${cfg.topP} \\}`;
    if (phase === "renorm") return `q_i = \\frac{p_i}{\\sum_{j \\in V} p_j} = \\frac{p_i}{${pl.kept.toFixed(3)}}`;
    if (phase === "sample") return `u = ${pl.u.toFixed(4)},\\quad \\text{pick the first } i \\text{ with } \\textstyle\\sum_{j \\le i} q_j > u`;
    return `\\operatorname{TV} = \\tfrac12 \\textstyle\\sum_i \\left| \\hat q_i - q_i \\right| = ${pl.checkpoints[CHECKPOINTS].tv.toFixed(3)}`;
  })();
  return <TexBlock>{tex}</TexBlock>;
}
