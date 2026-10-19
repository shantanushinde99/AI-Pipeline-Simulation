import { useMemo, useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, CheckCircle2, Eye, RotateCcw, Siren, XCircle } from "lucide-react";
import { getScenario } from "../../scenarios/registry";
import { scoreAttempt } from "../../scenarios/scoring";
import type { Outcome, Scenario } from "../../scenarios/types";
import { recordScenario } from "../../lib/storage";
import { PipelineCanvas, type PipeNode, type PipeStatus } from "../../components/pipeline/PipelineCanvas";
import { TraceInspector } from "../../components/visualizer/TraceInspector";
import { Badge, Button, Callout, Panel, SimulatedLabel, StatusBadge, Tabs, cx } from "../../components/ui";
import { DifficultyBadge } from "../explore/ExplorePage";
import { NotFound } from "../NotFound";
import { getTopic } from "../../data/topics";

type Tab = "investigate" | "trace" | "hypotheses" | "diagnose" | "results";
type Mark = "likely" | "unlikely" | "unsure";

function archNodes(s: Scenario, o: Outcome): PipeNode[] {
  return s.architecture.nodes.map((n) => {
    const spans = o.spans.filter(n.match);
    const status: PipeStatus = spans.some((x) => x.status === "error") ? "error" : spans.some((x) => x.status === "warning") ? "warning" : spans.length ? "ok" : "untraced";
    return { id: n.id, label: n.label, Icon: n.Icon, x: n.x, y: n.y, status, sub: spans.length ? `${spans.length} span${spans.length > 1 ? "s" : ""}` : undefined };
  });
}

function Metrics({ o, compare }: { o: Outcome; compare?: Outcome }) {
  const group = (kind: "health" | "quality", title: string, hint: string) => (
    <div>
      <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted">{title} <span className="font-normal normal-case">· {hint}</span></h4>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {o.metrics.filter((m) => m.kind === kind).map((m) => {
          const prev = compare?.metrics.find((x) => x.label === m.label);
          return (
            <div key={m.label} className="rounded-md border border-line p-2">
              <div className="text-[11px] text-muted">{m.label}</div>
              <div className="mt-0.5 flex items-center gap-1.5">
                <span className="font-mono text-sm">{prev && prev.value !== m.value ? <><span className="text-muted line-through">{prev.value}</span> → </> : null}{m.value}</span>
                <StatusBadge status={m.status} label={m.status === "ok" ? "ok" : m.status} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
  return <div className="space-y-3">{group("health", "Health metrics", "infrastructure")}{group("quality", "Quality metrics", "evaluation")}</div>;
}

function Transcript({ o }: { o: Outcome }) {
  return (
    <div className="space-y-1.5">
      {o.transcript.map((t, i) => (
        <div key={i} className={cx("rounded-md border px-2.5 py-1.5 text-sm", t.tone === "err" ? "border-err/40 bg-err-soft" : t.tone === "ok" ? "border-ok/40 bg-ok-soft" : "border-line")}>
          <span className="mr-1 text-[11px] font-semibold uppercase text-muted">{t.label}</span>{t.text}
        </div>
      ))}
    </div>
  );
}

export function ScenarioPage() {
  const { scenarioId = "" } = useParams();
  const s = getScenario(scenarioId);
  if (!s) return <NotFound />;
  return <ScenarioView key={s.id} s={s} />;
}

function ScenarioView({ s }: { s: Scenario }) {
  const before = useMemo(() => s.run(), [s]);
  const [tab, setTab] = useState<Tab>("investigate");
  const [revealed, setRevealed] = useState<string[]>([]);
  const [supporting, setSupporting] = useState<string[]>([]);
  const [marks, setMarks] = useState<Record<string, Mark>>({});
  const [rootCause, setRootCause] = useState("");
  const [fix, setFix] = useState("");
  const [verification, setVerification] = useState("");
  const [explanation, setExplanation] = useState("");
  const [result, setResult] = useState<ReturnType<typeof scoreAttempt> | null>(null);
  const [tryFix, setTryFix] = useState("");
  const [traceFocus, setTraceFocus] = useState<string | undefined>();

  const nodes = useMemo(() => archNodes(s, before), [s, before]);
  const tried = useMemo(() => (tryFix ? s.run(tryFix) : null), [s, tryFix]);
  const reveal = (id: string) => setRevealed((r) => (r.includes(id) ? r : [...r, id]));
  const toggleSupport = (id: string) => setSupporting((x) => (x.includes(id) ? x.filter((v) => v !== id) : [...x, id]));
  const submit = () => {
    const r = scoreAttempt(s, { rootCause, evidence: supporting, fix, verification, explanation }, before);
    setResult(r);
    setTryFix(fix);
    recordScenario(s.id, r.total, r.max);
    setTab("results");
  };
  const reset = () => {
    setRevealed([]); setSupporting([]); setMarks({}); setRootCause(""); setFix(""); setVerification(""); setExplanation(""); setResult(null); setTryFix(""); setTab("investigate");
  };
  const onNode = (id: string) => {
    const n = s.architecture.nodes.find((x) => x.id === id)!;
    setTraceFocus(before.spans.find(n.match)?.id);
    setTab("trace");
  };

  const tabs: { id: Tab; label: string }[] = [
    { id: "investigate", label: `1 · Investigate (${revealed.length}/${before.evidence.length})` },
    { id: "trace", label: "2 · Trace" },
    { id: "hypotheses", label: "3 · Hypotheses" },
    { id: "diagnose", label: "4 · Diagnose & fix" },
    ...(result ? [{ id: "results" as const, label: "Results" }] : []),
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Link to="/failure-lab" className="flex items-center gap-1 text-xs text-muted hover:text-fg"><ArrowLeft size={12} /> Failure Lab</Link>
        <Badge tone="accent">{s.system}</Badge><DifficultyBadge d={s.difficulty} /><SimulatedLabel>Simulated incident</SimulatedLabel>
        {result && <Badge tone={result.total === result.max ? "ok" : "warn"} className="ml-auto">score {result.total}/{result.max}</Badge>}
      </div>
      <h1 className="text-xl font-semibold tracking-tight">{s.title}</h1>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <section className="rounded-lg border border-err/40 bg-err-soft/60 p-3" aria-label="Incident">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-err"><Siren size={14} /> Incident report</div>
            <p className="mt-1.5 text-sm leading-relaxed">{s.incident}</p>
            <p className="mt-2 text-xs text-muted">Root cause: <span className="italic">hidden until you submit a diagnosis</span></p>
          </section>
          <Panel title="What users see"><Transcript o={before} /></Panel>
          <Panel title="Dashboards"><Metrics o={before} /></Panel>
        </div>
        <Panel title="System architecture" actions={<span className="text-[11px] text-muted">Status = what the spans report. Click a component to open its trace.</span>}>
          <PipelineCanvas nodes={nodes} edges={s.architecture.edges} selected={null} onSelect={onNode} height={360} ariaLabel={`${s.system} architecture`} />
        </Panel>
      </div>

      <Tabs label="Investigation steps" tabs={tabs} value={tab} onChange={setTab} />

      {tab === "investigate" && (
        <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
          <Panel title="Diagnostic actions">
            <ul className="space-y-1">
              {before.evidence.map((e) => (
                <li key={e.id}>
                  <button onClick={() => reveal(e.id)} className={cx("flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-surface-2", revealed.includes(e.id) && "text-muted")}>
                    {revealed.includes(e.id) ? <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-ok" aria-label="done" /> : <Eye size={14} className="mt-0.5 shrink-0 text-accent" aria-hidden />}
                    {e.action}
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
          <div className="space-y-3">
            {revealed.length === 0 && <Callout title="Start investigating">Pick a diagnostic action. Each one returns observations from the simulated run. Mark the ones that support your diagnosis; you'll be scored on whether they bear on the actual cause.</Callout>}
            {revealed.map((id) => {
              const e = before.evidence.find((x) => x.id === id)!;
              return (
                <article key={id} className="rounded-lg border border-line bg-surface p-3" aria-label={e.title}>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="ok">Observation</Badge>
                    <h3 className="text-sm font-semibold">{e.title}</h3>
                    <label className="ml-auto flex items-center gap-1.5 text-xs">
                      <input type="checkbox" checked={supporting.includes(id)} onChange={() => toggleSupport(id)} className="accent-[var(--accent)]" />
                      Supports my diagnosis
                    </label>
                  </div>
                  <ul className="mt-2 space-y-0.5 font-mono text-xs leading-relaxed">
                    {e.lines.map((l, i) => <li key={i} className="break-words">{l}</li>)}
                  </ul>
                  {e.spanId && <button onClick={() => { setTraceFocus(e.spanId); setTab("trace"); }} className="mt-2 text-xs text-accent hover:underline">Open related span in trace →</button>}
                </article>
              );
            })}
          </div>
        </div>
      )}

      {tab === "trace" && <Panel title="Trace inspector"><TraceInspector key={traceFocus} spans={before.spans} initial={traceFocus} /></Panel>}

      {tab === "hypotheses" && (
        <Panel title="Hypotheses (possible explanations, not facts)">
          <p className="mb-3 text-xs text-muted">Weigh each explanation against the observations you've collected. This board is for your reasoning; it isn't scored.</p>
          <ul className="space-y-2">
            {s.hypotheses.map((h) => (
              <li key={h.id} className="flex flex-col gap-2 rounded-md border border-line p-2.5 sm:flex-row sm:items-center">
                <span className="flex-1 text-sm">{h.text}</span>
                <div className="flex gap-1" role="group" aria-label={`Rate: ${h.text}`}>
                  {(["likely", "unsure", "unlikely"] as Mark[]).map((m) => (
                    <button key={m} onClick={() => setMarks((x) => ({ ...x, [h.id]: m }))} aria-pressed={marks[h.id] === m}
                      className={cx("rounded border px-2 py-0.5 text-xs", marks[h.id] === m ? (m === "likely" ? "border-ok bg-ok-soft" : m === "unlikely" ? "border-err bg-err-soft" : "border-accent bg-accent-soft") : "border-line text-muted")}>{m}</button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      {tab === "diagnose" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Root cause">
            <fieldset className="space-y-1.5"><legend className="sr-only">Root cause</legend>
              {s.hypotheses.map((h) => (
                <label key={h.id} className={cx("flex cursor-pointer gap-2 rounded-md border px-2.5 py-2 text-sm", rootCause === h.id ? "border-accent bg-accent-soft" : "border-line")}>
                  <input type="radio" name="rc" checked={rootCause === h.id} onChange={() => setRootCause(h.id)} className="mt-1 accent-[var(--accent)]" />{h.text}
                  {marks[h.id] && <Badge className="ml-auto self-start">{marks[h.id]}</Badge>}
                </label>
              ))}
            </fieldset>
            <p className="mt-3 text-xs text-muted">Supporting evidence selected: {supporting.length ? supporting.map((id) => before.evidence.find((e) => e.id === id)?.action).join("; ") : "none yet. Mark observations on the Investigate tab."}</p>
          </Panel>
          <div className="space-y-4">
            <Panel title="Corrective action">
              <fieldset className="space-y-1.5"><legend className="sr-only">Fix</legend>
                {s.fixes.map((f) => (
                  <label key={f.id} className={cx("flex cursor-pointer gap-2 rounded-md border px-2.5 py-2 text-sm", fix === f.id ? "border-accent bg-accent-soft" : "border-line")}>
                    <input type="radio" name="fix" checked={fix === f.id} onChange={() => setFix(f.id)} className="mt-1 accent-[var(--accent)]" />{f.label}
                  </label>
                ))}
              </fieldset>
            </Panel>
            <Panel title="Verification plan">
              <fieldset className="space-y-1.5"><legend className="sr-only">Verification</legend>
                {s.verifications.map((v) => (
                  <label key={v.id} className={cx("flex cursor-pointer gap-2 rounded-md border px-2.5 py-2 text-sm", verification === v.id ? "border-accent bg-accent-soft" : "border-line")}>
                    <input type="radio" name="ver" checked={verification === v.id} onChange={() => setVerification(v.id)} className="mt-1 accent-[var(--accent)]" />{v.label}
                  </label>
                ))}
              </fieldset>
            </Panel>
            <Panel title="Your explanation (optional)">
              <label htmlFor="expl" className="sr-only">Explain your diagnosis</label>
              <textarea id="expl" value={explanation} onChange={(e) => setExplanation(e.target.value)} rows={3} placeholder="What happened, which evidence shows it, and why your fix addresses it…" className="w-full rounded-md border border-line bg-surface p-2 text-sm" />
              <p className="mt-1 text-[11px] text-muted">Not auto-graded; shown next to the reference explanation for self-review.</p>
            </Panel>
            <Button variant="primary" disabled={!rootCause || !fix || !verification} onClick={submit} className="w-full">Submit diagnosis and apply fix</Button>
            {(!rootCause || !fix || !verification) && <p className="text-center text-xs text-muted">Choose a root cause, a fix and a verification plan.</p>}
          </div>
        </div>
      )}

      {tab === "results" && result && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {result.criteria.map((c) => (
              <Panel key={c.label} title={c.label}>
                <div className="flex items-center gap-2">{c.score === 1 ? <CheckCircle2 className="text-ok" size={18} /> : <XCircle className={c.score > 0 ? "text-warn" : "text-err"} size={18} />}<span className="font-mono text-lg">{c.score}/1</span></div>
                <p className="mt-1 text-xs leading-relaxed">{c.note}</p>
              </Panel>
            ))}
          </div>
          <Panel title="Root cause (revealed)">
            <p className="text-sm leading-relaxed">{s.rootCause}</p>
            {explanation && <div className="mt-3 rounded-md bg-surface-2 p-2 text-sm"><span className="text-xs font-semibold text-muted">Your explanation: </span>{explanation}</div>}
          </Panel>
          <Panel title="How the evidence confirms or rejects each hypothesis">
            <ul className="space-y-2">
              {s.hypotheses.map((h) => (
                <li key={h.id} className="rounded-md border border-line p-2.5 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    {h.isRootCause ? <Badge tone="ok">confirmed</Badge> : <Badge tone="err">rejected</Badge>}
                    <span className="font-medium">{h.text}</span>
                    {h.id === rootCause && <Badge tone="accent">your answer</Badge>}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed">{h.explanation}</p>
                  <p className="mt-1 text-[11px] text-muted">Evidence: {h.evidenceIds.map((id) => before.evidence.find((e) => e.id === id)?.action).join("; ")}</p>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Before and after the fix" actions={
            <label className="flex items-center gap-1 text-xs">Try another fix:
              <select value={tryFix} onChange={(e) => setTryFix(e.target.value)} className="h-7 max-w-[260px] rounded border border-line bg-surface px-1 text-xs">
                {s.fixes.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
              </select>
            </label>
          }>
            {tried && (
              <div className="space-y-3">
                <Callout tone={tried.resolved ? "ok" : "err"} title={tried.resolved ? "Resolved: the re-run behaves correctly" : "Not resolved: the re-run still fails"}>{s.fixes.find((f) => f.id === tryFix)?.explanation}</Callout>
                <div className="grid gap-4 lg:grid-cols-2">
                  <div><h4 className="mb-1.5 text-xs font-semibold">Before</h4><Transcript o={before} /></div>
                  <div><h4 className="mb-1.5 text-xs font-semibold">After</h4><Transcript o={tried} /></div>
                </div>
                <Metrics o={tried} compare={before} />
              </div>
            )}
          </Panel>
          <div className="flex flex-wrap gap-2">
            <Button onClick={reset}><RotateCcw size={14} /> Try the scenario again</Button>
            {s.relatedTopic && <Link to={`/visualizer/${s.relatedTopic}`} className="inline-flex h-8 items-center rounded-md border border-line px-3 text-sm hover:bg-surface-2">Open the {getTopic(s.relatedTopic)?.title} visualizer</Link>}
            <Link to="/failure-lab" className="inline-flex h-8 items-center rounded-md border border-line px-3 text-sm hover:bg-surface-2">More scenarios</Link>
          </div>
        </div>
      )}
    </div>
  );
}
