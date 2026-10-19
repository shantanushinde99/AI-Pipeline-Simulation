import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { LineChart } from "../../components/visualizer/LineChart";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../linear-regression/kit";
import { CURVES, LossChart, WeightGrid, stepMs } from "./canvas";
import { StepStage } from "./StepStage";
import { D, DEFAULT_FT, H, LR_PRESETS, METHOD_LABEL, TARGET_TRAIN, paramCount, run, simulate, type FTConfig, type Method, type Sim } from "./simulation";

const LR_LABEL: Record<keyof typeof LR_PRESETS, string> = { low: "Too low", good: "Good", high: "Too high" };
const roundLr = (v: number) => Number(v.toPrecision(2));

export default function FineTuningVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<FTConfig>(DEFAULT_FT);
  const set = (p: Partial<FTConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useMemo(() => (i: number) => stepMs(sim.steps[i]?.state.kind ?? "epoch"), [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const [replay, setReplay] = useState(0);
  const step = sim.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(cfg)}`;
  if (tab === "metrics") return <Metrics sim={sim} />;

  const prev = sim.history[s.kind === "epoch" ? s.prevEpoch : s.kind === "result" ? sim.history.length - 1 : 0];
  const cur = sim.history[s.kind === "base" || s.kind === "data" ? 0 : s.epoch];
  const batch = [...TARGET_TRAIN, ...sim.train.slice(TARGET_TRAIN.length, TARGET_TRAIN.length + 4)];
  const chartTo = s.kind === "base" || s.kind === "data" ? 0 : s.epoch;
  const chartFrom = s.kind === "epoch" ? s.prevEpoch : chartTo;

  return (
    <Workspace
      main={
        <>
          <Panel title="Fine-tuning a tiny pretrained model" actions={<span className="flex gap-1.5"><Badge tone={sim.regime === "good" ? "ok" : sim.regime === "diverged" ? "err" : "warn"}>outcome: {sim.regime}</Badge><SimulatedLabel>Simulation · real gradients</SimulatedLabel></span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={animKey} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
              <div>
                <WeightGrid prev={prev} cur={cur} method={cfg.method} batch={batch} animate={s.kind === "epoch"} k={k} animKey={animKey} />
              </div>
              <div>
                <LossChart history={sim.history} from={chartFrom} to={chartTo} k={k} animKey={animKey} animate={s.kind === "epoch"} best={s.kind === "result" ? sim.bestEpoch : undefined} />
                <div className="mt-1 space-y-0.5 text-[11px] text-muted">
                  {CURVES.map((c) => (
                    <div key={c.key} className="flex items-center gap-1.5">
                      <svg width="20" height="6" aria-hidden><line x1="0" y1="3" x2="20" y2="3" stroke={c.color} strokeWidth="2" strokeDasharray={c.dashed ? "4 3" : undefined} /></svg>{c.name}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-accent" />positive weight</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-err" />negative weight</span>
              <span>darker = larger</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-full bg-accent" />a phone review flowing in</span>
              {sim.nReplay > 0 && <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-full bg-violet" />a replayed movie review</span>}
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-warn/40" />gradient flowing back</span>
              <span>outlined cell = changed a lot</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, ease }}>
              <StepStage sim={sim} state={s} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          {pb.index === 0 && !pb.playing && <p className="-mt-1 text-xs text-muted">Press <strong>Play</strong> to watch the whole fine-tuning run, or <strong>→</strong> for one step at a time. Then change the method, learning rate or epochs and play again.</p>}
          <Panel title="Fine-tuning settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_FT); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Method" value={cfg.method} options={(Object.keys(METHOD_LABEL) as Method[]).map((m) => ({ value: m, label: METHOD_LABEL[m] }))} onChange={(m) => { set({ method: m }); pb.restart(); }} />
              <div className="space-y-1.5">
                <Slider label="Learning rate η" value={Math.log10(cfg.lr)} min={-2} max={1.3} step={0.05} format={() => String(cfg.lr)} onChange={(v) => set({ lr: roundLr(10 ** v) })} hint="Size of each gradient step (log scale)." />
                <div className="flex flex-wrap gap-1">
                  {(Object.keys(LR_PRESETS) as (keyof typeof LR_PRESETS)[]).map((p) => (
                    <Button key={p} size="sm" variant={cfg.lr === LR_PRESETS[p] ? "primary" : "secondary"} onClick={() => set({ lr: LR_PRESETS[p] })}>{LR_LABEL[p]} {LR_PRESETS[p]}</Button>
                  ))}
                </div>
              </div>
              <Slider label="Epochs" value={cfg.epochs} min={1} max={120} onChange={(v) => set({ epochs: v })} hint="Passes over the 8 examples. Try 100+ with LoRA rank 3 to see overfitting." />
              <Slider label={<>LoRA rank r {cfg.method !== "lora" && <span className="font-normal text-muted">(LoRA only)</span>}</>} value={cfg.rank} min={1} max={4} onChange={(v) => set({ method: "lora", rank: v })}
                hint={`Trainable: r × (${H} + ${D}) = ${paramCount("lora", cfg.rank).trainable} numbers.`} />
              <Slider label="Replay of original data" value={cfg.replay} min={0} max={1} step={0.1} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ replay: v })}
                hint={`Mixes ${sim.nReplay} movie reviews into every epoch.`} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Trainable parameters"><ParamBars method={cfg.method} rank={cfg.rank} /></Panel>
          <Panel title="Update rule"><Calc sim={sim} /></Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function ParamBars({ method, rank }: { method: Method; rank: number }) {
  const rows = (["full", "head", "lora"] as Method[]).map((m) => ({ m, ...paramCount(m, rank) }));
  return (
    <div className="space-y-2 text-xs">
      {rows.map((r) => (
        <div key={r.m} className={cx("space-y-0.5", r.m === method ? "" : "opacity-60")}>
          <div className="flex justify-between"><span className={r.m === method ? "font-semibold" : undefined}>{METHOD_LABEL[r.m]}{r.m === "lora" ? ` (r = ${rank})` : ""}</span><span className="font-mono">{r.trainable} / {r.total}</span></div>
          <div className="h-2.5 rounded bg-surface-2">
            <motion.div className={cx("h-full rounded", r.m === method ? "bg-accent" : "bg-muted/60")} initial={false} animate={{ width: `${(100 * r.trainable) / r.total}%` }} transition={{ duration: 0.5, ease }} />
          </div>
        </div>
      ))}
      <p className="leading-relaxed text-muted">
        At this toy size LoRA saves little (and r = 4 would train {paramCount("lora", 4).trainable} of {rows[0].total}). The saving grows with size: one 4096 × 4096 layer has 16,777,216 weights; LoRA with r = 8 trains 8 × (4096 + 4096) = 65,536 of them (0.39%).
      </p>
    </div>
  );
}

function Calc({ sim }: { sim: Sim }) {
  const { cfg } = sim;
  if (cfg.method === "lora")
    return (
      <div className="space-y-2">
        <TexBlock>{`W_{\\text{eff}} = \\underbrace{W}_{\\text{frozen}\\ ${H}\\times${D}} + \\underbrace{B}_{${H}\\times ${cfg.rank}}\\,\\underbrace{A}_{${cfg.rank}\\times ${D}}`}</TexBlock>
        <TexBlock>{`G = \\frac{\\partial \\mathcal{L}}{\\partial W_{\\text{eff}}},\\quad B \\leftarrow B - ${cfg.lr}\\, G A^{\\top},\\quad A \\leftarrow A - ${cfg.lr}\\, B^{\\top} G`}</TexBlock>
        <p className="text-xs text-muted">B starts at 0, so B·A = 0 and the model starts exactly as the base. A starts random (scale 1/√{D}).</p>
      </div>
    );
  return (
    <div className="space-y-2">
      <TexBlock>{`\\theta \\leftarrow \\theta - ${cfg.lr}\\; \\nabla_{\\theta}\\, \\mathcal{L}_{\\text{batch of }4}, \\qquad \\theta = ${cfg.method === "full" ? "\\{W, b_1, w_2, b_2\\}" : "\\{w_2, b_2\\}"}`}</TexBlock>
      <TexBlock>{`\\mathcal{L} = -\\tfrac{1}{n}\\textstyle\\sum_i \\big[y_i \\log p_i + (1 - y_i) \\log(1 - p_i)\\big],\\quad p = \\sigma\\!\\big(w_2^{\\top} \\tanh(Wx + b_1) + b_2\\big)`}</TexBlock>
      <KV rows={[["updates per epoch", Math.ceil(sim.train.length / 4)], ["total updates", sim.history.at(-1)!.updates]]} />
    </div>
  );
}

function Metrics({ sim }: { sim: Sim }) {
  const { cfg } = sim;
  const compare = useMemo(() => {
    const variants: { label: string; c: FTConfig }[] = [
      { label: METHOD_LABEL.full, c: { ...cfg, method: "full" } },
      { label: METHOD_LABEL.head, c: { ...cfg, method: "head" } },
      ...[1, 2, 4].map((r) => ({ label: `LoRA r = ${r}`, c: { ...cfg, method: "lora" as const, rank: r } })),
    ];
    return variants.map((v) => ({ ...v, r: run(v.c) }));
  }, [cfg]);
  const h0 = sim.history[0].m, last = sim.history.at(-1)!.m;
  return (
    <div className="space-y-3">
      <Callout title="Computed by the simulation in your browser">
        Every number below comes from real gradient descent on the toy model with your current settings (lr {cfg.lr}, {cfg.epochs} epochs, replay {Math.round(cfg.replay * 100)}%). Change them on the Visualize tab.
      </Callout>
      <Panel title={`Loss curves · ${METHOD_LABEL[cfg.method]}`}>
        <LineChart xLabel="epoch" yLabel="loss" markerX={sim.bestEpoch}
          series={CURVES.map((c) => ({ name: c.name, color: c.color, dashed: c.dashed, points: sim.history.map((h) => [h.epoch, Math.min(5, h.m[c.key])] as [number, number]) }))} />
        <p className="mt-1 text-xs text-muted">Dashed vertical line: the epoch with the lowest phone validation loss ({sim.bestEpoch}). Losses above 5 are clipped.</p>
      </Panel>
      <div className="grid gap-3 sm:grid-cols-3">
        <Panel title="Phone accuracy (new)"><div className="font-mono text-2xl">{Math.round(h0.valAcc * 100)}% → {Math.round(last.valAcc * 100)}%</div></Panel>
        <Panel title="Movie accuracy (original)"><div className={cx("font-mono text-2xl", last.origAcc < h0.origAcc - 0.15 && "text-err")}>{Math.round(h0.origAcc * 100)}% → {Math.round(last.origAcc * 100)}%</div></Panel>
        <Panel title="Trainable parameters"><div className="font-mono text-2xl">{sim.params.trainable} / {sim.params.total}</div></Panel>
      </div>
      <Panel title="Same settings, every method">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">method</th><th>trainable</th><th>phone acc</th><th>movie acc</th><th>movie loss change</th><th>outcome</th></tr></thead>
            <tbody className="font-mono">
              {compare.map(({ label, r }) => {
                const a = r.history[0].m, b = r.history.at(-1)!.m;
                return (
                  <tr key={label} className="border-t border-line">
                    <td className="py-1 font-sans">{label}</td>
                    <td>{r.params.trainable} ({Math.round((100 * r.params.trainable) / r.params.total)}%)</td>
                    <td>{Math.round(a.valAcc * 100)} → {Math.round(b.valAcc * 100)}%</td>
                    <td>{Math.round(a.origAcc * 100)} → {Math.round(b.origAcc * 100)}%</td>
                    <td className={r.forgetting > 0.3 ? "text-err" : undefined}>{r.forgetting >= 0 ? "+" : ""}{Number.isFinite(r.forgetting) ? r.forgetting.toFixed(2) : "∞"}</td>
                    <td><Badge tone={r.regime === "good" ? "ok" : r.regime === "diverged" ? "err" : "warn"}>{r.regime}</Badge></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">"Movie loss change" is how much worse the model got at the original task: the size of the forgetting.</p>
      </Panel>
    </div>
  );
}
