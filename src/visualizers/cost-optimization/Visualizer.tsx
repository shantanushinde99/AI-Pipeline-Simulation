import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../hallucination-analysis/kit";
import { BATCHABLE, BATCH_DISCOUNT, DEFAULT_CONFIG, DIFFS, IN_TOKENS, LEVERS, PRICES, QUALITY, REQUESTS, ROUTES, simulate, type Config, type CostState } from "./simulation";

/** One schedule (seconds at 1×) shared by canvas, counters and the step card. */
const T = { block: (i: number) => 0.15 + i * 0.07, fly: 0.6, bar: 0.3, grow: 1.0, card: (i: number) => 0.3 + i * 0.2 };
const DUR: Record<CostState["phase"], number> = { request: 3600, baseline: 3000, lever: 3000, total: 2800, frontier: 3200 };
const usd = (v: number) => `$${Math.round(v).toLocaleString("en-US")}`;
type Run = ReturnType<typeof simulate>;

function narrate(s: CostState, run: Run) {
  const w = run.waterfall;
  if (s.phase === "request") return { headline: `One hard request: ${IN_TOKENS} tokens in, ${DIFFS[2].out} out = $${run.hardLarge.toFixed(4)}`, sub: "Every token is billed; output tokens cost 5× more than input tokens (hypothetical prices)." };
  if (s.phase === "baseline") return { headline: `× ${REQUESTS.toLocaleString("en-US")} requests a month = ${usd(w.baseline.cost)}`, sub: "Baseline: every request goes to the large model with the full prompt and no cache." };
  if (s.phase === "lever") {
    const st = w.stages.find((x) => x.id === s.lever)!;
    return { headline: `${st.name}: −${usd(st.saving)} a month`, sub: st.saving < 1 ? "This lever is off in the settings below, so nothing changes." : `Cost ${usd(st.before)} → ${usd(st.after)}, quality now ${st.quality.toFixed(3)}.` };
  }
  if (s.phase === "total") return { headline: `${usd(w.baseline.cost)} → ${usd(w.final.cost)} a month (−${((1 - w.final.cost / w.baseline.cost) * 100).toFixed(0)}%)`, sub: `Quality ${w.baseline.quality.toFixed(3)} → ${w.final.quality.toFixed(3)}. Savings that cost quality are a trade, not a free lunch.` };
  return { headline: "Every config is a point: cost vs quality", sub: "The frontier is the best quality you can buy at each price. Change a setting and watch your dot move." };
}

export default function CostVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(run.steps.length, (i) => DUR[run.steps[i].state.phase]);
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(s, run);
  const set = (p: Partial<Config>) => setCfg((c) => ({ ...c, ...p }));
  const w = run.waterfall, c = run.config;
  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Tokens → dollars → monthly waterfall" actions={<SimulatedLabel>Hypothetical prices & workload</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[640px]"><Canvas run={run} s={s} k={k} animKey={animKey} /></div></div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard run={run} s={s} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Cost levers" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_CONFIG); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Slider label="Prompt trimming" value={cfg.trim} min={0} max={0.6} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ trim: v })} hint="Cut input tokens. Past 30% you start cutting useful context." />
              <Slider label="Cache hit rate" value={cfg.hitRate} min={0} max={0.9} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ hitRate: v })} hint="Share of requests answered from the response cache." />
              <Select label="Model routing" value={String(cfg.route)} options={ROUTES.map((r, i) => ({ value: String(i), label: r }))} onChange={(v) => set({ route: Number(v) })} />
              <Toggle label={`Batching (${BATCHABLE * 100}% of traffic is offline, ${BATCH_DISCOUNT * 100}% off)`} checked={cfg.batching} onChange={(v) => set({ batching: v })} />
              <Select label="Output cap (max tokens)" value={cfg.cap === null ? "none" : String(cfg.cap)} options={["none", "1200", "800", "500", "300", "150"].map((v) => ({ value: v, label: v }))} onChange={(v) => set({ cap: v === "none" ? null : Number(v) })} />
              <p className="text-[11px] text-muted">Break it: route all → small or cap at 150 and watch quality fall while the bill looks great.</p>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="LLM spend scales with tokens × requests. Each lever cuts one factor; the honest question is how much quality each dollar saved costs." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`\\begin{aligned} C &= \\textstyle\\sum_d N\\,s_d\\,(1-h)\\,c_d\\,(1 - ${BATCH_DISCOUNT}\\,b) \\\\ c_d &= t_{in}\\,p_{in} + t_{out,d}\\,p_{out} \\end{aligned}`}</TexBlock>
              <TexBlock>{`\\begin{aligned} h &= ${c.hitRate.toFixed(2)},\\; t_{in} = ${Math.round(IN_TOKENS * (1 - c.trim))},\\; b = ${c.batching ? BATCHABLE : 0} \\\\ C &= \\$${Math.round(w.final.cost)} \\;\\text{vs}\\; \\$${Math.round(w.baseline.cost)} \\end{aligned}`}</TexBlock>
              <p className="text-[11px] text-muted">Prices per 1M tokens: large ${PRICES.large.in} in / ${PRICES.large.out} out, small ${PRICES.small.in} / ${PRICES.small.out} (hypothetical).</p>
            </div>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["baseline / month", usd(w.baseline.cost)],
              ["current / month", usd(w.final.cost)],
              ["saved", `${((1 - w.final.cost / w.baseline.cost) * 100).toFixed(0)}%`],
              ["quality", `${w.baseline.quality.toFixed(3)} → ${w.final.quality.toFixed(3)}`],
              ["routing", ROUTES[c.route]],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const Y0 = 260, H = 200;
function Canvas({ run, s, k, animKey }: { run: Run; s: CostState; k: number; animKey: string }) {
  const w = run.waterfall;
  if (s.phase === "request") {
    const nIn = IN_TOKENS / 100, nOut = DIFFS[2].out / 100;
    const inCost = (IN_TOKENS * PRICES.large.in) / 1e6, outCost = (DIFFS[2].out * PRICES.large.out) / 1e6;
    return (
      <svg viewBox="0 0 720 260" className="h-auto w-full" role="img" aria-label="Input and output tokens flowing into a price meter">
        <text x={20} y={56} fontSize={11} fontWeight={600} fill="var(--fg)">input tokens (1 block = 100)</text>
        <text x={20} y={156} fontSize={11} fontWeight={600} fill="var(--fg)">output tokens</text>
        {[...Array(nIn + nOut)].map((_, i) => {
          const out = i >= nIn, j = out ? i - nIn : i;
          const x0 = 20 + (j % 9) * 30, y0 = (out ? 168 : 68) + Math.floor(j / 9) * 22;
          return <motion.rect key={`${i}-${animKey}`} width={24} height={16} rx={3} fill={out ? "var(--violet)" : "var(--accent)"}
            initial={{ x: x0, y: y0, opacity: 1 }} animate={{ x: [x0, x0, 470], y: [y0, y0, out ? 160 : 90], opacity: [1, 1, 0] }}
            transition={{ delay: T.block(i) * k, duration: T.fly * 2 * k, times: [0, 0.4, 1], ease }} />;
        })}
        <rect x={460} y={60} width={240} height={150} rx={10} fill="var(--surface-2)" stroke="var(--line)" />
        <text x={476} y={84} fontSize={11} fill="var(--muted)">large model meter</text>
        <text x={476} y={112} fontSize={12} fontFamily="var(--font-mono)" fill="var(--accent)">in  × $3/M  = $<AnimatedNumber key={`a${animKey}`} svg value={inCost} from={0} digits={4} delay={T.block(0) * k} duration={T.block(nIn) * k} /></text>
        <text x={476} y={140} fontSize={12} fontFamily="var(--font-mono)" fill="var(--violet)">out × $15/M = $<AnimatedNumber key={`b${animKey}`} svg value={outCost} from={0} digits={4} delay={T.block(nIn) * k} duration={(T.block(nIn + nOut) - T.block(nIn) + T.fly) * k} /></text>
        <line x1={476} x2={684} y1={156} y2={156} stroke="var(--line)" />
        <text x={476} y={184} fontSize={15} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">total  $<AnimatedNumber key={`c${animKey}`} svg value={run.hardLarge} from={0} digits={4} delay={T.block(0) * k} duration={(T.block(nIn + nOut) + T.fly) * k} /></text>
      </svg>
    );
  }
  if (s.phase === "frontier") {
    const { pts, front } = run.frontier;
    const maxC = Math.max(...pts.map((p) => p.cost), w.baseline.cost);
    const qs = pts.map((p) => p.quality), qMin = Math.min(...qs, run.current.quality) - 0.02, qMax = Math.max(...qs, w.baseline.quality) + 0.01;
    const X = (v: number) => 70 + (v / maxC) * 620, Y = (q: number) => 240 - ((q - qMin) / (qMax - qMin)) * 210;
    const d = front.map((p, i) => `${i ? "L" : "M"}${X(p.cost).toFixed(1)} ${Y(p.quality).toFixed(1)}`).join(" ");
    return (
      <svg viewBox="0 0 720 280" className="h-auto w-full" role="img" aria-label="Scatter of cost versus quality with the Pareto frontier and the current configuration">
        <line x1={70} x2={700} y1={240} y2={240} stroke="var(--line)" /><line x1={70} x2={70} y1={20} y2={240} stroke="var(--line)" />
        <text x={385} y={272} textAnchor="middle" fontSize={10} fill="var(--muted)">monthly cost → (0 … {usd(maxC)})</text>
        <text x={14} y={130} fontSize={10} fill="var(--muted)" transform="rotate(-90 14 130)" textAnchor="middle">quality →</text>
        {[qMin, (qMin + qMax) / 2, qMax].map((q) => <text key={q} x={64} y={Y(q) + 3} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{q.toFixed(2)}</text>)}
        {pts.map((p, i) => <motion.circle key={`${i}-${animKey}`} cx={X(p.cost)} cy={Y(p.quality)} r={3} fill="var(--muted)" initial={{ opacity: 0 }} animate={{ opacity: 0.35 }} transition={{ delay: (i / pts.length) * 0.8 * k }} />)}
        <motion.path key={`f-${animKey}`} d={d} fill="none" stroke="var(--ok)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.8 * k, duration: 1.0 * k }} />
        <circle cx={X(w.baseline.cost)} cy={Y(w.baseline.quality)} r={6} fill="none" stroke="var(--fg)" strokeWidth={1.5} />
        <text x={X(w.baseline.cost) - 8} y={Y(w.baseline.quality) - 10} textAnchor="end" fontSize={10} fill="var(--fg)">baseline</text>
        <motion.g initial={false} animate={{ x: X(run.current.cost), y: Y(run.current.quality) }} transition={{ duration: 0.6, ease }}>
          <circle r={7} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
          <text x={10} y={20} fontSize={10.5} fontWeight={700} fill="var(--accent)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">you ({usd(run.current.cost)}, q {run.current.quality.toFixed(3)})</text>
        </motion.g>
        <text x={690} y={226} textAnchor="end" fontSize={10} fill="var(--ok)">━ frontier (not beaten on both axes)</text>
      </svg>
    );
  }
  const scale = H / w.baseline.cost, Yv = (v: number) => Y0 - v * scale;
  const cols = ["Baseline", ...LEVERS.map((l) => l.name), "Final"];
  const BX = (i: number) => 40 + i * 96;
  return (
    <svg viewBox="0 0 720 300" className="h-auto w-full" role="img" aria-label="Monthly cost waterfall from baseline through each lever to the final cost">
      <line x1={30} x2={710} y1={Y0} y2={Y0} stroke="var(--line)" />
      {cols.map((name, i) => {
        const isLever = i > 0 && i <= LEVERS.length, st = isLever ? w.stages[i - 1] : null;
        const visible = i === 0 || (isLever ? s.shown >= i : s.phase === "total");
        const current = (isLever && s.lever === st!.id) || (i === cols.length - 1 && s.phase === "total") || (i === 0 && s.phase === "baseline");
        const top = st ? Yv(st.before) : Yv(i === 0 ? w.baseline.cost : w.final.cost);
        const h = st ? st.saving * scale : (i === 0 ? w.baseline.cost : w.final.cost) * scale;
        const val = st ? st.saving : i === 0 ? w.baseline.cost : w.final.cost;
        const q = st ? st.quality : i === 0 ? w.baseline.quality : w.final.quality;
        const qDrop = st && q < (i > 1 ? w.stages[i - 2].quality : w.baseline.quality) - 1e-9;
        return (
          <g key={name} opacity={visible ? 1 : 0.25}>
            <text x={BX(i) + 30} y={Y0 + 16} textAnchor="middle" fontSize={9.5} fontWeight={current ? 700 : 500} fill={current ? "var(--accent)" : "var(--fg)"}>{name.replace("Response ", "").replace("Prompt ", "")}</text>
            {visible && <text x={BX(i) + 30} y={Y0 + 30} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill={qDrop ? "var(--warn)" : "var(--muted)"}>q {q.toFixed(3)}{qDrop ? " ▼" : ""}</text>}
            {isLever && i < cols.length - 1 && <line x1={BX(i) - 36} x2={BX(i) + 60} y1={Yv(st!.before)} y2={Yv(st!.before)} stroke="var(--line)" strokeDasharray="3 3" />}
            {visible && (
              <motion.rect key={current ? animKey : "bar"} x={BX(i)} width={60} rx={3} fill={st ? "var(--ok)" : i === 0 ? "var(--err)" : "var(--accent)"} fillOpacity={st ? 0.75 : 0.85}
                initial={current ? (st ? { y: top, height: 0 } : { y: Y0, height: 0 }) : false} animate={{ y: top, height: Math.max(h, 0.5) }}
                transition={{ delay: (current ? T.bar : 0) * k, duration: (current ? T.grow : 0.5) * k, ease }} />
            )}
            {visible && (
              <motion.text key={`t-${current ? animKey : "t"}`} x={BX(i) + 30} textAnchor="middle" fontSize={10.5} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)"
                initial={false} animate={{ y: top - 6 }} transition={{ duration: 0.5 * k }}>
                {st ? "−" : ""}{current ? <AnimatedNumber key={animKey} svg value={val / 1000} from={0} digits={1} delay={T.bar * k} duration={T.grow * k} /> : (val / 1000).toFixed(1)}k
              </motion.text>
            )}
          </g>
        );
      })}
      <text x={30} y={18} fontSize={10} fill="var(--muted)">$ per month (k = thousand); green = saved by the lever; q = quality after it</text>
    </svg>
  );
}

function Bar({ label, v, max, cls, sub, i, k }: { label: string; v: number; max: number; cls: string; sub: string; i: number; k: number }) {
  return (
    <div className="grid grid-cols-[96px_minmax(0,1fr)_92px] items-center gap-2 text-xs">
      <span className="font-medium">{label}</span>
      <div className="h-3.5 rounded bg-surface-2"><motion.div className={cx("h-3.5 rounded", cls)} initial={{ width: 0 }} animate={{ width: `${Math.min(1, v / max) * 100}%` }} transition={{ delay: T.card(i) * k, duration: 0.6 * k, ease }} /></div>
      <span className="text-right font-mono text-[11px]">{sub}</span>
    </div>
  );
}

function StepCard({ run, s, k }: { run: Run; s: CostState; k: number }) {
  const w = run.waterfall, c = run.config, rows = run.current.rows;
  if (s.phase === "request") {
    const a = IN_TOKENS * PRICES.large.in, b = DIFFS[2].out * PRICES.large.out;
    return <div className="space-y-1.5"><Bar k={k} i={1} label="input cost" v={a} max={a + b} cls="bg-accent" sub={`$${(a / 1e6).toFixed(4)}`} /><Bar k={k} i={3} label="output cost" v={b} max={a + b} cls="bg-violet" sub={`$${(b / 1e6).toFixed(4)}`} />
      <p className="text-[11px] text-muted">{IN_TOKENS} input tokens are {Math.round((IN_TOKENS / (IN_TOKENS + DIFFS[2].out)) * 100)}% of the tokens but only {Math.round((a / (a + b)) * 100)}% of the bill.</p></div>;
  }
  if (s.phase === "baseline") {
    return <div className="space-y-1.5">{w.baseline.rows.map((r, i) => <Bar k={k} key={r.d} i={i} label={`${DIFFS[r.d].id} (${DIFFS[r.d].share * 100}%)`} v={r.cost} max={w.baseline.cost} cls="bg-err" sub={usd(r.cost)} />)}
      <p className="text-[11px] text-muted">Hard requests are 15% of traffic but write the longest answers.</p></div>;
  }
  if (s.phase === "lever") {
    if (s.lever === "trim") return <div className="space-y-1.5"><Bar k={k} i={0} label="before" v={IN_TOKENS} max={IN_TOKENS} cls="bg-accent" sub={`${IN_TOKENS} tok`} /><Bar k={k} i={1} label="after trim" v={IN_TOKENS * (1 - c.trim)} max={IN_TOKENS} cls="bg-ok" sub={`${Math.round(IN_TOKENS * (1 - c.trim))} tok`} />
      <p className="text-[11px] text-muted">Shorter system prompt, fewer few-shot examples, tighter retrieved context. {c.trim > 0.3 ? "Above 30% this toy model charges a quality penalty." : "Up to 30% is free in this model."}</p></div>;
    if (s.lever === "cache") return <div><div className="flex flex-wrap gap-1">{[...Array(20)].map((_, i) => { const hit = i < Math.round(c.hitRate * 20); return (
      <motion.span key={i} className={cx("rounded px-1.5 py-0.5 font-mono text-[10px]", hit ? "bg-ok-soft text-ok" : "bg-surface-2 text-muted")} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (0.1 + i * 0.06) * k }}>{hit ? "hit $0" : "miss"}</motion.span>); })}</div>
      <p className="mt-2 text-[11px] text-muted">{Math.round(c.hitRate * 100)}% of requests are served from the cache and never reach the model.</p></div>;
    if (s.lever === "route") return <div className="space-y-1.5">{rows.map((r, i) => <Bar k={k} key={r.d} i={i} label={DIFFS[r.d].id} v={QUALITY[r.model][r.d]} max={1} cls={r.model === "small" ? "bg-ok" : "bg-accent"} sub={`${r.model} q${QUALITY[r.model][r.d].toFixed(2)}`} />)}
      <p className="text-[11px] text-muted">The small model is ~12× cheaper and nearly as good on easy requests, but much worse on hard ones.</p></div>;
    if (s.lever === "batch") return <div className="space-y-1.5"><Bar k={k} i={0} label="realtime" v={c.batching ? 1 - BATCHABLE : 1} max={1} cls="bg-accent" sub="full price" /><Bar k={k} i={1} label="batched" v={c.batching ? BATCHABLE : 0} max={1} cls="bg-ok" sub={c.batching ? `${BATCH_DISCOUNT * 100}% off` : "off"} />
      <p className="text-[11px] text-muted">Offline work (nightly summaries, evals) can wait for a discounted batch job; interactive traffic cannot.</p></div>;
    return <div className="space-y-1.5">{rows.map((r, i) => <div key={r.d}><Bar k={k} i={i} label={DIFFS[r.d].id} v={r.outTok} max={DIFFS[2].out} cls={r.outTok < DIFFS[r.d].out ? "bg-warn" : "bg-violet"} sub={`${r.outTok}/${DIFFS[r.d].out} tok`} /></div>)}
      <p className="text-[11px] text-muted">{c.cap === null ? "No cap set." : `Cap ${c.cap}: answers longer than that are cut off, which costs quality (orange = truncated).`}</p></div>;
  }
  if (s.phase === "total") {
    return <div className="grid grid-cols-3 gap-2 text-center">{([["baseline", w.baseline.cost, "text-err"], ["now", w.final.cost, "text-accent"], ["saved", w.baseline.cost - w.final.cost, "text-ok"]] as const).map(([l, v, cls], i) => (
      <div key={l} className="rounded-md border border-line bg-surface-2 p-2"><div className={cx("font-mono text-xl", cls)}>$<AnimatedNumber value={v} from={0} digits={0} delay={T.card(i) * k} duration={0.8 * k} /></div><div className="text-[11px] text-muted">{l} / month</div></div>))}
      <div className="col-span-3 text-xs">quality <Badge tone={w.final.quality < w.baseline.quality - 0.02 ? "warn" : "ok"}>{w.baseline.quality.toFixed(3)} → {w.final.quality.toFixed(3)}</Badge></div></div>;
  }
  const better = run.frontier.pts.find((p) => p.cost < run.current.cost - 1 && p.quality >= run.current.quality - 1e-9);
  return <p className="text-xs">{better ? <>Not on the frontier: <b>{ROUTES[better.cfg.route]}</b>, trim {Math.round(better.cfg.trim * 100)}%, cap {better.cfg.cap ?? "none"} costs {usd(better.cost)} at quality {better.quality.toFixed(3)} (≥ yours). Try it.</> : <>Your config is on (or at) the frontier: nothing in the grid is cheaper at the same or better quality. Caching and batching shift the whole cloud left without costing quality.</>}</p>;
}
