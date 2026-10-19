import { useMemo, useState } from "react";
import { AnimatePresence, motion, type MotionValue } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { LineChart } from "../../components/visualizer/LineChart";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, Stats, TimeBars, f0, pct, useSimClock, useTime } from "../latency-and-throughput/kit";
import {
  BUCKET_MS, DEFAULT_RL, MAX_DELAY_MS, POLICY_LABEL, backoffTex, perBucket, simulate, summarize, tokensAt,
  type Attempt, type Policy, type RLConfig, type RLState,
} from "./simulation";

/** Every window step plays over the same real time, whatever its simulated length. */
const PLAY_S = 3.2;
const stepMs = (s: RLState) => (s.kind === "window" ? PLAY_S * 1000 + 600 : 3500);
const FLY = 150; // simulated ms a request spends travelling on screen

const PRESETS: { label: string; patch: Partial<RLConfig> }[] = [
  { label: "Thundering herd (fixed, no jitter)", patch: { policy: "fixed", baseMs: 500, retryAfter: false, clients: 24, rate: 5, burst: 5, maxAttempts: 6 } },
  { label: "Exponential, no jitter", patch: { policy: "exponential", baseMs: 500, retryAfter: false, clients: 24, rate: 5, burst: 5, maxAttempts: 6 } },
  { label: "Exponential + jitter", patch: { policy: "exp-jitter", baseMs: 500, retryAfter: false, clients: 24, rate: 5, burst: 5, maxAttempts: 6 } },
  { label: "Jitter + Retry-After", patch: { policy: "exp-jitter", baseMs: 500, retryAfter: true, clients: 24, rate: 5, burst: 5, maxAttempts: 6 } },
  { label: "Break it: 40 clients, fast fixed retries", patch: { policy: "fixed", baseMs: 300, retryAfter: false, clients: 40, rate: 5, burst: 5, maxAttempts: 5 } },
];

export default function RateLimitVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<RLConfig>(DEFAULT_RL);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(cfg), [cfg]);
  const durations = useMemo(() => run.steps.map((s) => stepMs(s.state)), [run]);
  const pb = usePlayback(run.steps.length, (i) => durations[i] ?? 3000);
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const clock = useSimClock(s.t0, s.t1, s.kind === "window" ? PLAY_S * k : 0, animKey);
  const set = (p: Partial<RLConfig>) => setCfg((c) => ({ ...c, ...p }));

  if (tab === "metrics") return <Metrics cfg={cfg} />;

  const story = narrate(s, run.cfg, run.sum);
  const example = run.attempts.find((a) => !a.ok && a.nextAt !== undefined && a.t >= s.t0 && a.t < s.t1) ?? run.attempts.find((a) => !a.ok && a.nextAt !== undefined);

  return (
    <Workspace
      main={
        <>
          <Panel title="Clients → token bucket → API" actions={<SimulatedLabel>Simulated clients and server</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} id={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[580px]">
                <BucketSvg attempts={run.attempts} cfg={run.cfg} clock={clock} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>request in flight</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--ok)" /></svg>accepted (took a token)</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--err)" /></svg>429 Too Many Requests</span>
              <span>ring around a client = time left before its retry</span>
              <span>{run.windowMs} simulated ms play over {PLAY_S} s at 1×</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <LiveCard attempts={run.attempts} cfg={run.cfg} clock={clock} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Limiter & retry policy" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_RL); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select<Policy> label="Client retry policy" value={cfg.policy} onChange={(v) => set({ policy: v })} options={(Object.keys(POLICY_LABEL) as Policy[]).map((p) => ({ value: p, label: POLICY_LABEL[p] }))} />
              <Slider label="Base retry delay" value={cfg.baseMs} min={100} max={2000} step={50} format={(v) => `${v} ms`} onChange={(v) => set({ baseMs: v })} />
              <Slider label="Refill rate r" value={cfg.rate} min={1} max={20} format={(v) => `${v} tokens/s`} onChange={(v) => set({ rate: v })} />
              <Slider label="Burst b (bucket size)" value={cfg.burst} min={1} max={20} onChange={(v) => set({ burst: v })} />
              <Slider label="Clients" value={cfg.clients} min={4} max={40} onChange={(v) => set({ clients: v })} />
              <Slider label="Max attempts per client" value={cfg.maxAttempts} min={1} max={10} onChange={(v) => set({ maxAttempts: v })} />
              <Toggle label="Respect the Retry-After header" checked={cfg.retryAfter} onChange={(v) => set({ retryAfter: v })} hint="Wait at least as long as the server says a token will take." />
              <div className="flex items-end"><Button size="sm" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> New random seed ({cfg.seed})</Button></div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Presets:</span>
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => { set(p.patch); pb.restart(); }}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Retries that fire at the same moment arrive as one spike: the bucket can only admit b of them, and the rest come back together again." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`\\text{tokens}(t) = \\min\\big(${run.cfg.burst},\\ \\text{tokens}_0 + ${run.cfg.rate}\\,\\Delta t\\big)`}</TexBlock>
              {example ? (
                <>
                  <p className="text-[11px] text-muted">Client {example.client + 1}, attempt {example.n} at {example.t.toFixed(0)} ms found {example.tokensBefore.toFixed(2)} tokens → 429.</p>
                  <TexBlock>{`\\text{Retry-After} = \\frac{1 - ${example.tokensBefore.toFixed(2)}}{${run.cfg.rate}} \\approx ${example.retryAfterMs}\\,\\text{ms}`}</TexBlock>
                  <TexBlock>{`\\text{delay}_{${example.n}} = ${backoffTex(run.cfg, example)}`}</TexBlock>
                </>
              ) : <p className="text-xs text-muted">No request was rejected: the bucket never ran dry.</p>}
            </div>
          </Panel>
          <Panel title="Whole run (simulated)">
            <KV rows={[
              ["attempts", run.sum.attempts],
              ["429 responses", `${run.sum.rejected} (${pct(run.sum.rate429)})`],
              ["clients succeeded", `${run.sum.succeeded} / ${run.cfg.clients}`],
              ["gave up", run.sum.gaveUp],
              ["all done at", Number.isFinite(run.sum.doneAt) ? `${(run.sum.doneAt / 1000).toFixed(2)} s` : "not all succeeded"],
              ["worst retry spike", `${run.sum.retrySpike} in 100 ms`],
              ["retries before Retry-After", run.sum.early],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(s: RLState, cfg: RLConfig, sum: ReturnType<typeof summarize>) {
  if (s.kind === "summary")
    return {
      headline: `${sum.succeeded}/${cfg.clients} succeeded with ${sum.attempts} attempts (${pct(sum.rate429)} were 429s)`,
      sub: sum.gaveUp ? `${sum.gaveUp} clients burned all ${cfg.maxAttempts} attempts inside retry waves and gave up. Try exponential + jitter.`
        : cfg.policy === "exp-jitter" ? `Jitter spread the retries (worst spike ${sum.retrySpike} per 100 ms), so the bucket's refill was used continuously.`
        : cfg.policy === "exponential" ? "Everyone succeeded, but the synchronized waves left long idle gaps: the bucket sat full while all clients were waiting." : "Synchronized retries arrived in waves and were mostly rejected. Try exponential + jitter.",
    };
  if (s.t0 === 0) return { headline: `${cfg.clients} clients reconnect at once; the bucket holds only ${cfg.burst} tokens`, sub: `The first ${cfg.burst} take a token and go through; the rest get HTTP 429 and schedule a retry.` };
  if (!s.attempts) return { headline: "Quiet: no client is sending right now", sub: `Meanwhile the bucket refills at ${cfg.rate} tokens/s, wasting capacity nobody uses.` };
  if (s.rejected > s.attempts / 2) return { headline: `A retry wave: ${s.attempts} attempts, ${s.rejected} rejected again`, sub: cfg.policy === "exp-jitter" ? "Even with jitter the bucket is empty. Fewer, later retries would help." : "Everyone waited the same delay, so they all came back at the same moment." };
  return { headline: `${s.attempts} attempts, ${s.attempts - s.rejected} admitted`, sub: "Retries arrive spread out, so each finds a token that refilled since the last one." };
}

const BX = 330, BW = 92, BBOT = 262, GATE: [number, number] = [BX - 4, 165], SERVER: [number, number] = [660, 165];

function clientPos(i: number, n: number): [number, number] {
  const cols = n <= 24 ? 4 : 5;
  const rows = Math.ceil(n / cols);
  const rowH = Math.min(30, 240 / rows);
  return [30 + (i % cols) * (cols === 4 ? 44 : 36), 52 + Math.floor(i / cols) * rowH];
}
const lerp = (a: [number, number], b: [number, number], f: number): [number, number] => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];

function BucketSvg({ attempts, cfg, clock }: { attempts: Attempt[]; cfg: RLConfig; clock: MotionValue<number> }) {
  const t = useTime(clock);
  const tokens = tokensAt(attempts, cfg, t);
  const full = Math.floor(tokens + 1e-9);
  const accepted = attempts.filter((a) => a.ok && a.t <= t).length;
  const byClient = (c: number) => attempts.filter((a) => a.client === c && a.t <= t).at(-1);
  const cols = cfg.burst > 10 ? 2 : 1;
  const tokenPos = (i: number): [number, number] => [BX + BW / 2 + (cols === 2 ? (i % 2 ? 11 : -11) : 0), BBOT - 14 - Math.floor(i / cols) * 17];
  const flying = attempts.filter((a) => t >= a.t - FLY && t <= a.t + FLY);
  const BTOP = Math.min(BBOT - 60, tokenPos(cfg.burst - 1)[1] - 26);
  return (
    <svg viewBox="0 0 760 290" className="h-auto w-full" role="img" aria-label={`At ${(t / 1000).toFixed(2)} s: ${tokens.toFixed(1)} tokens in the bucket, ${accepted} requests accepted`}>
      <text x={752} y={18} textAnchor="end" fontSize={12} fontFamily="var(--font-mono)" fill="var(--muted)">t = {(t / 1000).toFixed(2)} s</text>
      <text x={20} y={22} fontSize={11} fontWeight={600} fill="var(--fg)">clients ({cfg.clients})</text>
      {Array.from({ length: cfg.clients }, (_, i) => {
        const [x, y] = clientPos(i, cfg.clients);
        const last = byClient(i);
        const state = !last ? "start" : last.ok ? "done" : last.gaveUp ? "gave up" : "backoff";
        const prog = last && last.nextAt !== undefined ? (t - last.t) / (last.nextAt - last.t) : 0;
        const C = 2 * Math.PI * 11;
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={9} fill={state === "done" ? "var(--ok-soft)" : state === "gave up" ? "var(--err-soft)" : "var(--surface-2)"}
              stroke={state === "done" ? "var(--ok)" : state === "gave up" ? "var(--err)" : "var(--line)"} />
            {state === "backoff" && (
              <circle cx={x} cy={y} r={11} fill="none" stroke="var(--warn)" strokeWidth={2.5} strokeDasharray={`${C * Math.min(1, prog)} ${C}`} transform={`rotate(-90 ${x} ${y})`} />
            )}
            <text x={x} y={y + 3.5} textAnchor="middle" fontSize={9} fontWeight={600} fill={state === "done" ? "var(--ok)" : state === "gave up" ? "var(--err)" : "var(--muted)"}>
              {state === "done" ? "✓" : state === "gave up" ? "✕" : i + 1}
            </text>
          </g>
        );
      })}
      {/* bucket */}
      <text x={BX + BW / 2} y={BTOP - 30} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--fg)">token bucket</text>
      <text x={BX + BW / 2} y={BTOP - 16} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={tokens < 1 ? "var(--err)" : "var(--muted)"}>{tokens.toFixed(1)} / {cfg.burst} tokens</text>
      <text x={BX + BW + 10} y={BTOP + 8} fontSize={10} fontFamily="var(--font-mono)" fill="var(--accent)">+{cfg.rate}/s ↓</text>
      <path d={`M${BX} ${BTOP} L${BX + 6} ${BBOT} L${BX + BW - 6} ${BBOT} L${BX + BW} ${BTOP}`} fill="var(--surface-2)" stroke="var(--fg)" strokeOpacity={0.5} strokeWidth={2} />
      <line x1={BX + 4} x2={BX + BW - 4} y1={tokenPos(cfg.burst - 1)[1] - 9} y2={tokenPos(cfg.burst - 1)[1] - 9} stroke="var(--muted)" strokeDasharray="3 3" />
      <text x={BX - 4} y={tokenPos(cfg.burst - 1)[1] - 6} textAnchor="end" fontSize={9} fill="var(--muted)">b = {cfg.burst}</text>
      {Array.from({ length: cfg.burst }, (_, i) => {
        const [x, y] = tokenPos(i);
        const fill = i < full ? 1 : i === full ? tokens - full : 0;
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={7} fill="none" stroke="var(--line)" />
            <motion.circle cx={x} cy={y} r={7} fill="var(--warn)" initial={false} animate={{ opacity: i < full ? 1 : fill * 0.6, scale: i < full ? 1 : 0.4 + fill * 0.5 }} transition={{ duration: 0.15 }} />
          </g>
        );
      })}
      {/* server */}
      <line x1={BX + BW} y1={GATE[1]} x2={SERVER[0] - 50} y2={SERVER[1]} stroke="var(--line)" strokeDasharray="4 4" />
      <rect x={SERVER[0] - 50} y={SERVER[1] - 34} width={110} height={68} rx={10} fill="var(--surface)" stroke="var(--accent)" />
      <text x={SERVER[0] + 5} y={SERVER[1] - 10} textAnchor="middle" fontSize={11} fontWeight={600}>API server</text>
      <text x={SERVER[0] + 5} y={SERVER[1] + 8} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--ok)">{accepted} served</text>
      <text x={SERVER[0] + 5} y={SERVER[1] + 22} textAnchor="middle" fontSize={9} fill="var(--muted)">max {cfg.rate}/s sustained</text>
      {flying.map((a) => {
        const home = clientPos(a.client, cfg.clients);
        const out = t < a.t;
        const f = out ? 1 - (a.t - t) / FLY : (t - a.t) / FLY;
        const [x, y] = out ? lerp(home, GATE, f) : a.ok ? lerp(GATE, SERVER, f) : lerp(GATE, home, f);
        const color = out ? "var(--accent)" : a.ok ? "var(--ok)" : "var(--err)";
        return (
          <g key={a.id} transform={`translate(${x} ${y})`} opacity={!out && f > 0.85 ? (1 - f) / 0.15 : 1}>
            <circle r={5} fill={color} stroke="var(--surface)" strokeWidth={1.2} />
            {!out && !a.ok && (
              <text y={-8} textAnchor="middle" fontSize={9} fontWeight={700} fill="var(--err)">429{cfg.retryAfter && f < 0.5 ? ` · wait ${a.retryAfterMs}ms` : ""}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function LiveCard({ attempts, cfg, clock }: { attempts: Attempt[]; cfg: RLConfig; clock: MotionValue<number> }) {
  const t = useTime(clock);
  const full = useMemo(() => perBucket(attempts), [attempts]);
  const yMax = Math.max(4, ...full.map((b) => b.ok + b.rejected));
  const shown = full.map((b) => {
    const inB = attempts.filter((a) => a.t >= b.t0 && a.t < b.t1 && a.t <= t);
    const ok = inB.filter((a) => a.ok).length;
    return { t0: b.t0, t1: b.t1, values: [ok, inB.length - ok] };
  });
  const sum = summarize(attempts, cfg, t);
  return (
    <div className="space-y-3">
      <Stats items={[
        { label: "attempts", value: String(sum.attempts) },
        { label: "429 rate", value: pct(sum.rate429), tone: sum.rate429 > 0.5 ? "err" : sum.rate429 > 0.25 ? "warn" : "ok" },
        { label: "clients done", value: `${sum.succeeded}/${cfg.clients}`, tone: "ok" },
        { label: "gave up", value: String(sum.gaveUp), tone: sum.gaveUp ? "err" : "neutral" },
      ]} />
      <TimeBars buckets={shown} series={[{ name: "accepted", color: "var(--ok)" }, { name: "429", color: "var(--err)" }]} yMax={yMax} t={t} label={`Attempts per ${BUCKET_MS} ms`} />
      <p className="text-xs text-muted">
        Tall red-and-green spikes with gaps between them are <strong>retry waves</strong>: every client waited the same delay. The limiter only admits about {cfg.rate * (BUCKET_MS / 1000)} per {BUCKET_MS} ms plus whatever burst is saved up, so the rest of each spike is wasted work.
      </p>
    </div>
  );
}

function Metrics({ cfg }: { cfg: RLConfig }) {
  const rows = useMemo(() => (Object.keys(POLICY_LABEL) as Policy[]).flatMap((policy) => [false, true].map((retryAfter) => {
    const r = simulate({ ...cfg, policy, retryAfter });
    return { policy, retryAfter, sum: r.sum, buckets: perBucket(r.attempts) };
  })), [cfg]);
  const colors: Record<Policy, string> = { fixed: "var(--err)", exponential: "var(--warn)", "exp-jitter": "var(--ok)" };
  return (
    <div className="space-y-3">
      <Callout title="Same clients and limiter, every retry policy (simulated)">
        {cfg.clients} clients, bucket r = {cfg.rate}/s, b = {cfg.burst}, base delay {cfg.baseMs} ms, max {cfg.maxAttempts} attempts. Delays are capped at {MAX_DELAY_MS / 1000} s.
      </Callout>
      <Panel title="Attempts per 250 ms (Retry-After ignored)">
        <LineChart xLabel="time (s)" yLabel="attempts" height={240} series={rows.filter((r) => !r.retryAfter).map((r) => ({
          name: POLICY_LABEL[r.policy], color: colors[r.policy], dashed: r.policy === "exponential", points: r.buckets.map((b) => [b.t0 / 1000, b.ok + b.rejected] as [number, number]),
        }))} />
      </Panel>
      <Panel title="Totals">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">policy</th><th>Retry-After</th><th>attempts</th><th>429 rate</th><th>retry spike / 100 ms</th><th>early retries</th><th>succeeded</th><th>gave up</th><th>all done at</th></tr></thead>
            <tbody className="font-mono">{rows.map((r) => (
              <tr key={`${r.policy}-${r.retryAfter}`} className="border-t border-line">
                <td className="py-1 font-sans">{POLICY_LABEL[r.policy]}</td><td>{r.retryAfter ? "respected" : "ignored"}</td><td>{r.sum.attempts}</td>
                <td>{pct(r.sum.rate429)}</td><td>{r.sum.retrySpike}</td><td>{r.sum.early}</td><td>{r.sum.succeeded}</td>
                <td>{r.sum.gaveUp ? <StatusBadge status="error" label={String(r.sum.gaveUp)} /> : 0}</td>
                <td>{Number.isFinite(r.sum.doneAt) ? `${f0(r.sum.doneAt)} ms` : "—"}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
