import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw, RotateCw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { DEFAULTS, simulate, type SplitInput, type SplitState } from "./simulation";

const ease = [0.22, 1, 0.36, 1] as const;
const DUR: Record<SplitState["phase"], number> = { data: 2000, shuffle: 2200, split: 2800, train: 2400, fold: 2600, validate: 2600, select: 2200, test: 3200 };
const CELL = 13;
const BOX = { train: [10, 360], val: [380, 490], test: [510, 630] } as const;
const DEV: [number, number] = [10, 490];
type Sim = ReturnType<typeof simulate>;

/** Where a row sits on the canvas in a given phase. */
function layout(sim: Sim, st: SplitState) {
  const pos: Record<number, [number, number]> = {};
  if (st.phase === "data" || st.phase === "shuffle") {
    (st.phase === "data" ? sim.data : sim.order).forEach((r, i) => { pos[r.id] = [40 + (i % 40) * 14.2, 80 + Math.floor(i / 40) * 16]; });
    return pos;
  }
  const place = (rows: Sim["data"], [x0, x1]: readonly [number, number]) => {
    const cols = Math.max(1, Math.floor((x1 - x0 - 8) / CELL));
    rows.forEach((r, i) => { pos[r.id] = [x0 + 10 + (i % cols) * CELL, 70 + Math.floor(i / cols) * CELL]; });
  };
  if (sim.p.kfold) {
    const w = (DEV[1] - DEV[0]) / sim.p.folds;
    for (let f = 0; f < sim.p.folds; f++) place(sim.dev.filter((r) => sim.fold[r.id] === f), [DEV[0] + f * w, DEV[0] + (f + 1) * w - 4]);
  } else {
    place(sim.train, BOX.train);
    place(sim.val, BOX.val);
  }
  place(sim.test, BOX.test);
  return pos;
}

export default function SplitsVisualizer({ tab }: { tab: VisualizerTab }) {
  void tab;
  const [p, setP] = useState<SplitInput>(DEFAULTS);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(p), [p]);
  const pb = usePlayback(sim.steps.length, (i) => DUR[sim.steps[i]?.state.phase ?? "data"]);
  const step = sim.steps[Math.min(pb.index, sim.steps.length - 1)];
  const st = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (patch: Partial<SplitInput>) => { setP((v) => ({ ...v, ...patch })); pb.restart(); };
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const evalOn = p.tuneOnTest ? "test" : p.kfold ? "CV" : "validation";

  return (
    <Workspace
      main={
        <>
          <Panel title="Dataset and buckets" actions={<SimulatedLabel>synthetic data</SimulatedLabel>}>
            <div className="mb-2 flex items-start gap-3">
              <span className="mt-0.5 shrink-0 rounded-md bg-accent px-2 py-0.5 font-mono text-xs text-white dark:text-black">{pb.index + 1}/{sim.steps.length}</span>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={step.id} className="min-w-0 flex-1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
                  <p className="text-base font-semibold leading-snug">{step.title}</p>
                  <p className="text-sm text-muted">{step.description}</p>
                </motion.div>
              </AnimatePresence>
              <Button size="sm" variant="ghost" onClick={() => setReplay((r) => r + 1)}><RotateCw size={13} /> Replay step</Button>
            </div>
            <div className="overflow-x-auto"><div className="min-w-[520px]"><Canvas sim={sim} st={st} k={k} animKey={animKey} /></div></div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4.5" fill="var(--accent)" /></svg>label 1</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><rect x="1" y="1" width="8" height="8" fill="var(--violet)" /></svg>label 0</span>
              <span>one mark = one row</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                <Scores sim={sim} st={st} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => set(DEFAULTS)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Select label="Random seed" value={String(p.seed)} options={[1, 2, 3, 4, 5, 6, 7].map((s) => ({ value: String(s), label: `seed ${s}` }))} onChange={(v) => set({ seed: Number(v) })} />
              <Slider label="Rows" value={p.n} min={30} max={200} step={10} onChange={(v) => set({ n: v })} />
              <Slider label="Label noise" value={p.noise} min={0} max={0.4} step={0.05} format={pct} onChange={(v) => set({ noise: v })} />
              <Slider label="Test share" value={p.testFrac} min={0.1} max={0.4} step={0.05} format={pct} onChange={(v) => set({ testFrac: v })} />
              {!p.kfold && <Slider label="Validation share" value={p.valFrac} min={0.1} max={0.4} step={0.05} format={pct} onChange={(v) => set({ valFrac: v })} />}
              {p.kfold && <Slider label="Folds K" value={p.folds} min={3} max={8} onChange={(v) => set({ folds: v })} />}
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Toggle label="k-fold cross-validation" checked={p.kfold} onChange={(v) => set({ kfold: v })} hint="Rotate the validation fold instead of one fixed validation set." />
              <Toggle label="Break it: tune on the test set" checked={p.tuneOnTest} onChange={(v) => set({ tuneOnTest: v })} hint="Pick k by test accuracy, then report that same test accuracy." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Formula">
            <TexBlock>{st.phase === "fold" || (p.kfold && st.phase === "validate")
              ? `\\text{CV}(k) = \\frac{1}{K}\\sum_{f=1}^{K} \\text{acc}_f(k) \\quad K = ${p.folds}`
              : st.phase === "select" || st.phase === "test"
                ? `k^* = \\arg\\max_k \\text{acc}_{\\text{${evalOn}}}(k) = ${sim.winner}`
                : `n = ${p.n} = \\underbrace{${p.kfold ? sim.dev.length : sim.train.length}}_{\\text{${p.kfold ? "dev" : "train"}}}${p.kfold ? "" : ` + \\underbrace{${sim.val.length}}_{\\text{val}}`} + \\underbrace{${sim.test.length}}_{\\text{test}}`}</TexBlock>
          </Panel>
          <Panel title="Key values">
            <KV rows={[["selected on", evalOn], ["winner", `k = ${sim.winner}`], ["reported test acc", pct(sim.reported)], ["fresh-data acc (truth)", pct(sim.freshAcc)], ["gap", `${((sim.reported - sim.freshAcc) * 100).toFixed(1)} pts`]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Canvas({ sim, st, k, animKey }: { sim: Sim; st: SplitState; k: number; animKey: string }) {
  const pos = layout(sim, st);
  const pooled = st.phase === "data" || st.phase === "shuffle";
  const testOpen = st.phase === "test" || st.revealed.includes("test");
  const foldW = (DEV[1] - DEV[0]) / sim.p.folds;
  const n = sim.data.length;
  const box = (x0: number, x1: number, label: string, tone: string, hi: boolean, note?: string) => (
    <g>
      <rect x={x0} y={30} width={x1 - x0} height={250} rx={8} fill={tone} fillOpacity={hi ? 0.16 : 0.06} stroke={tone} strokeOpacity={hi ? 0.9 : 0.35} strokeWidth={hi ? 2 : 1} />
      <text x={x0 + 8} y={48} fontSize={11} fontWeight={600} fill="var(--fg)">{label}</text>
      {note && <text x={x1 - 8} y={48} fontSize={10} textAnchor="end" fill={tone}>{note}</text>}
    </g>
  );
  return (
    <svg viewBox="0 0 640 290" className="h-auto w-full" role="img" aria-label="Rows of data being shuffled and dealt into train, validation and test buckets">
      {pooled ? box(10, 630, `all ${n} rows`, "var(--muted)", false) : (
        <>
          {sim.p.kfold
            ? Array.from({ length: sim.p.folds }, (_, f) => <g key={f}>{box(DEV[0] + f * foldW, DEV[0] + (f + 1) * foldW - 4, `fold ${f + 1}`, "var(--accent)", false)}</g>)
            : <>{box(...BOX.train, `train (${sim.train.length})`, "var(--accent)", st.phase === "train")}{box(...BOX.val, `val (${sim.val.length})`, "var(--ok)", st.phase === "validate" && !sim.p.tuneOnTest)}</>}
          {box(...BOX.test, `test (${sim.test.length})`, sim.p.tuneOnTest && st.revealed.includes("test") ? "var(--err)" : "var(--warn)", st.phase === "test" || (sim.p.tuneOnTest && st.phase === "validate"),
            testOpen ? (sim.p.tuneOnTest && st.phase !== "test" ? "peeked!" : "open") : "locked")}
          {st.phase === "fold" && (
            <motion.rect y={26} width={foldW - 4 + 8} height={258} rx={10} fill="none" stroke="var(--ok)" strokeWidth={3} strokeDasharray="6 4"
              initial={{ x: DEV[0] - 4 + Math.max(0, st.fold! - 1) * foldW }} animate={{ x: DEV[0] - 4 + st.fold! * foldW }} transition={{ duration: 0.7 * k, ease }} />
          )}
          {st.phase === "fold" && <text x={DEV[0] + (st.fold! + 0.5) * foldW} y={22} fontSize={10.5} textAnchor="middle" fill="var(--ok)" fontWeight={600}>validation fold</text>}
        </>
      )}
      {sim.data.map((r) => {
        const [x, y] = pos[r.id];
        const order = sim.order.findIndex((o) => o.id === r.id);
        const delay = st.phase === "split" ? 0.2 + (order / n) * 1.4 : st.phase === "shuffle" ? 0.2 + (order / n) * 0.8 : 0;
        const dim = st.phase === "fold" && sim.bucket[r.id] !== "test" && sim.fold[r.id] !== st.fold ? 0.55 : 1;
        return (
          <motion.g key={r.id} initial={false} animate={{ x, y, opacity: dim }} transition={{ delay: delay * k, duration: 0.7 * k, ease }}>
            {r.y ? <circle r={4.5} fill="var(--accent)" /> : <rect x={-4} y={-4} width={8} height={8} fill="var(--violet)" />}
          </motion.g>
        );
      })}
      <AnimatePresence>
        {st.phase === "select" && (
          <motion.text key={animKey} x={320} y={278} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--fg)" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            final model: k = {sim.winner}, refit on all {sim.dev.length} non-test rows
          </motion.text>
        )}
      </AnimatePresence>
    </svg>
  );
}

function Scores({ sim, st, k }: { sim: Sim; st: SplitState; k: number }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  if (st.phase === "data" || st.phase === "shuffle" || st.phase === "split") {
    const parts = sim.p.kfold ? [["dev (folds)", sim.dev.length, "bg-accent/60"], ["test", sim.test.length, "bg-warn/60"]] as const
      : [["train", sim.train.length, "bg-accent/60"], ["val", sim.val.length, "bg-ok/60"], ["test", sim.test.length, "bg-warn/60"]] as const;
    return (
      <div className="space-y-2">
        <div className="flex h-7 overflow-hidden rounded-md bg-surface-2">
          {parts.map(([name, c, cls], i) => (
            <motion.div key={name} className={cx("flex items-center justify-center text-[11px] font-medium", cls)} initial={{ width: 0 }} animate={{ width: st.phase === "split" ? `${(c / sim.data.length) * 100}%` : i === 0 ? "100%" : "0%" }} transition={{ delay: 0.3 * k, duration: 1.2 * k, ease }}>
              {st.phase === "split" && `${name} ${c}`}
            </motion.div>
          ))}
        </div>
        <p className="text-sm text-muted">{st.phase === "split" ? "Train fits the model, validation picks between models, test is opened once at the very end." : "No split yet: every row is in one pool."}</p>
      </div>
    );
  }
  const showVal = st.phase === "validate" || st.phase === "select" || st.phase === "test";
  const showTest = st.phase === "test" || sim.p.tuneOnTest && showVal;
  const sel = sim.p.tuneOnTest ? "test" : "val";
  const cols: { key: "train" | "val" | "test" | "fold"; name: string; show: boolean; cls: string; at: number }[] = [
    { key: "train", name: "train", show: true, cls: "bg-accent/60", at: 0.2 },
    { key: "fold", name: `fold ${(st.fold ?? 0) + 1}`, show: st.phase === "fold", cls: "bg-ok/60", at: 0.6 },
    { key: "val", name: sim.p.kfold ? "CV mean" : "val", show: showVal && !sim.p.tuneOnTest || (sim.p.kfold && showVal), cls: "bg-ok/60", at: 0.4 },
    { key: "test", name: "test", show: showTest, cls: sim.p.tuneOnTest ? "bg-err/60" : "bg-warn/60", at: 0.4 },
  ];
  const vis = cols.filter((c) => c.show);
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[340px] text-xs">
          <thead className="text-left text-muted"><tr><th className="w-14 py-1">k-NN</th>{vis.map((c) => <th key={c.key}>{c.name} accuracy</th>)}</tr></thead>
          <tbody>
            {sim.scores.map((s) => {
              const win = (st.phase === "select" || st.phase === "test") && s.k === sim.winner;
              return (
                <tr key={s.k} className={cx("border-t border-line", win && "bg-accent-soft")}>
                  <td className="py-1 font-mono">k={s.k}{win && " ★"}</td>
                  {vis.map((c) => {
                    const v = c.key === "fold" ? s.folds[st.fold ?? 0] : s[c.key];
                    const hide = c.key === "test" && st.phase === "test" && !sim.p.tuneOnTest && s.k !== sim.winner;
                    return (
                      <td key={c.key} className="pr-2">
                        {hide ? <span className="text-muted">not evaluated</span> : (
                          <div className="flex items-center gap-1.5">
                            <div className="h-3 flex-1 rounded bg-surface-2"><motion.div className={cx("h-full rounded", c.cls)} initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: c.at * k, duration: 0.8 * k, ease }} /></div>
                            <span className="w-12 text-right font-mono"><AnimatedNumber value={v * 100} from={0} digits={1} delay={c.at * k} duration={0.8 * k} />%</span>
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {st.phase === "test" && (
        <motion.div className="grid gap-2 sm:grid-cols-2" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.2 * k }}>
          <div className="rounded-md border border-line p-2 text-xs">reported test accuracy<div className="font-mono text-xl">{pct(sim.reported)}</div></div>
          <div className="rounded-md border border-line p-2 text-xs">same model on 600 fresh rows<div className="font-mono text-xl">{pct(sim.freshAcc)}</div></div>
          <p className={cx("text-xs sm:col-span-2", sim.p.tuneOnTest ? "text-err" : "text-muted")}>
            {sim.p.tuneOnTest ? `✗ Selected on test: the reported score is the best of ${sim.scores.length} lucky draws on ${sim.test.length} rows, so it overstates real performance by ${((sim.reported - sim.freshAcc) * 100).toFixed(1)} pts.`
              : `✓ Test was used once. A small test set is still noisy (±${(100 * Math.sqrt(0.25 / sim.test.length)).toFixed(0)} pts), but it is not biased by selection.`}
          </p>
        </motion.div>
      )}
      {st.phase !== "test" && <p className="text-xs text-muted">{st.phase === "train" ? "Training accuracy is optimistic: the model is graded on rows it memorised." : `Selection uses ${sel === "test" ? "the TEST set (leak)" : sim.p.kfold ? "the mean over folds" : "the validation set"}.`}</p>}
    </div>
  );
}
