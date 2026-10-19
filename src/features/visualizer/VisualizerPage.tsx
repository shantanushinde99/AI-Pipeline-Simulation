import { Suspense, useEffect, useState } from "react";
import { Link, NavLink, useParams, useSearchParams } from "react-router";
import { CheckCircle2, Circle, Clock, FlaskConical, ListTree, PanelLeftClose, PanelLeftOpen, X } from "lucide-react";
import { CATEGORIES, TOPICS, categoryOf, getTopic } from "../../data/topics";
import { VISUALIZERS } from "../../visualizers/registry";
import { markVisited, setCompleted, useProgress } from "../../lib/storage";
import { Badge, Button, CodeBlock, Panel, Tabs, cx } from "../../components/ui";
import { DifficultyBadge } from "../explore/ExplorePage";
import { ChallengeList } from "../challenges/ChallengeList";
import { getScenario } from "../../scenarios/registry";
import { NotFound } from "../NotFound";
import { ConceptFlow } from "./ConceptFlow";

type TabId = "visualize" | "concept" | "code" | "metrics" | "failures" | "challenges";

function TopicNav({ currentId, onNavigate }: { currentId: string; onNavigate?: () => void }) {
  const [filter, setFilter] = useState("");
  const progress = useProgress();
  const current = getTopic(currentId);
  const f = filter.toLowerCase();
  return (
    <nav aria-label="Topics" className="space-y-3">
      <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter topics" aria-label="Filter topics"
        className="h-8 w-full rounded-md border border-line bg-surface px-2.5 text-sm" />
      {CATEGORIES.map((c) => {
        const items = TOPICS.filter((t) => t.category === c.id && (!f || t.title.toLowerCase().includes(f)));
        if (!items.length) return null;
        const isCurrent = c.id === current?.category;
        return (
          <details key={c.id} open={isCurrent || !!f} className="group">
            <summary className={cx("cursor-pointer list-none text-xs font-semibold uppercase tracking-wide", isCurrent ? "text-fg" : "text-muted")}>
              {c.title} <span className="font-normal">({items.filter((t) => t.status === "implemented").length})</span>
            </summary>
            <ul className="mt-1.5 space-y-0.5">
              {items.map((t) =>
                t.status === "implemented" ? (
                  <li key={t.id}>
                    <NavLink to={`/visualizer/${t.id}`} onClick={onNavigate}
                      className={({ isActive }) => cx("flex items-center gap-2 rounded-md px-2 py-1.5 text-sm", isActive ? "bg-accent-soft font-medium text-accent" : "hover:bg-surface-2")}>
                      {progress.completed[t.id] ? <CheckCircle2 size={13} className="shrink-0 text-ok" aria-label="completed" /> : <Circle size={13} className="shrink-0 text-muted" aria-hidden />}
                      <span className="flex-1">{t.title}</span>
                      <span className="text-[10px] text-muted">{t.difficulty[0]}</span>
                    </NavLink>
                  </li>
                ) : (
                  <li key={t.id} className="flex items-center gap-2 px-2 py-1 text-xs text-muted">
                    <span className="flex-1 truncate">{t.title}</span>
                    <span className="text-[10px]">planned</span>
                  </li>
                ),
              )}
            </ul>
          </details>
        );
      })}
    </nav>
  );
}

export function VisualizerPage() {
  const { topicId = "" } = useParams();
  const topic = getTopic(topicId);
  const mod = VISUALIZERS[topicId];
  const [params, setParams] = useSearchParams();
  const [drawer, setDrawer] = useState(false);
  const [navOpen, setNavOpen] = useState(() => { try { return localStorage.getItem("nav-collapsed") !== "1"; } catch { return true; } });
  const toggleNav = () => setNavOpen((o) => { try { localStorage.setItem("nav-collapsed", o ? "1" : "0"); } catch { /* storage blocked */ } return !o; });
  const progress = useProgress();

  useEffect(() => {
    if (mod) markVisited(topicId);
  }, [topicId, mod]);

  if (!topic) return <NotFound />;
  if (!mod) {
    return (
      <div className="mx-auto max-w-xl space-y-3 py-10">
        <Badge>planned</Badge>
        <h1 className="text-xl font-semibold">{topic.title}</h1>
        <p className="text-sm text-muted">This topic does not have an interactive visualizer yet.</p>
        {topic.coveredBy && <p className="text-sm">Its core idea is exercised in <Link className="text-accent hover:underline" to={`/visualizer/${topic.coveredBy}`}>{getTopic(topic.coveredBy)?.title}</Link>.</p>}
        <Link to="/explore" className="text-sm text-accent hover:underline">← Back to Explore</Link>
      </div>
    );
  }

  const tabs: { id: TabId; label: string }[] = [
    { id: "visualize", label: "Visualize" },
    { id: "concept", label: "Concept" },
    { id: "code", label: "Code" },
    ...(mod.meta.hasMetrics ? [{ id: "metrics" as const, label: "Metrics" }] : []),
    { id: "failures", label: "Failure Modes" },
    { id: "challenges", label: `Challenges (${mod.content.challenges.length})` },
  ];
  const requested = params.get("tab") as TabId | null;
  const tab: TabId = tabs.some((t) => t.id === requested) ? requested! : "visualize";
  const setTab = (t: TabId) => setParams(t === "visualize" ? {} : { tab: t }, { replace: true });
  const live = tab === "visualize" || tab === "metrics";
  const done = !!progress.completed[topicId];
  const { Component, content } = mod;

  return (
    <div className={cx("grid gap-5", navOpen && "lg:grid-cols-[230px_minmax(0,1fr)]")}>
      <aside className={cx("hidden", navOpen && "lg:block")}><div className="sticky top-18 max-h-[calc(100vh-5rem)] overflow-y-auto pr-1">
        <Button size="sm" variant="secondary" className="mb-2 w-full" onClick={toggleNav} aria-label="Hide topic sidebar"><PanelLeftClose size={13} /> Hide topics</Button><TopicNav currentId={topicId} /></div></aside>
      {drawer && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Topics">
          <div className="absolute inset-0 bg-black/40" onClick={() => setDrawer(false)} />
          <div className="absolute inset-y-0 left-0 w-80 max-w-[85vw] overflow-y-auto bg-surface p-4">
            <div className="mb-3 flex items-center justify-between"><span className="font-semibold">Topics</span><button aria-label="Close topics" onClick={() => setDrawer(false)}><X size={18} /></button></div>
            <TopicNav currentId={topicId} onNavigate={() => setDrawer(false)} />
          </div>
        </div>
      )}

      <div className="min-w-0 space-y-4">
        <header className="flex flex-wrap items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
              <Button size="sm" variant="secondary" className="lg:hidden" onClick={() => setDrawer(true)}><ListTree size={13} /> Topics</Button>
              {!navOpen && <Button size="sm" variant="secondary" className="hidden lg:inline-flex" onClick={toggleNav} aria-label="Show topic sidebar"><PanelLeftOpen size={13} /> Topics</Button>}
              <span>{categoryOf(topic.category).title}</span>
              <DifficultyBadge d={topic.difficulty} />
              <span className="flex items-center gap-1"><Clock size={11} aria-hidden />{topic.minutes} min</span>
            </div>
            <h1 className="mt-1 text-xl font-semibold tracking-tight">{topic.title}</h1>
            <p className="mt-0.5 text-sm text-muted">{topic.summary}</p>
          </div>
          <Button variant={done ? "secondary" : "primary"} size="sm" onClick={() => setCompleted(topicId, !done)} aria-pressed={done}>
            {done ? <><CheckCircle2 size={14} className="text-ok" /> Completed</> : "Mark as complete"}
          </Button>
        </header>

        <Tabs label="Visualizer sections" tabs={tabs} value={tab} onChange={setTab} />

        <div hidden={!live}>
          <Suspense fallback={<p className="text-sm text-muted">Loading simulation…</p>}>
            <Component tab={tab === "metrics" ? "metrics" : "visualize"} />
          </Suspense>
        </div>

        {tab === "concept" && <ConceptFlow key={topicId} sections={content.concept} />}
        {tab === "code" && (
          <div className="max-w-4xl space-y-3">
            <p className="text-xs text-muted">Reference implementations of the same idea. They are not executed in the browser.</p>
            {content.code.map((c) => <CodeBlock key={c.title} title={c.title} code={c.source} />)}
          </div>
        )}
        {tab === "failures" && (
          <div className="grid gap-3 md:grid-cols-2">
            {content.failureModes.map((f) => {
              const sc = f.scenarioId ? getScenario(f.scenarioId) : undefined;
              return (
                <Panel key={f.title} title={f.title}>
                  <dl className="space-y-1.5 text-sm leading-relaxed">
                    <div><dt className="inline font-semibold">Symptom: </dt><dd className="inline">{f.symptom}</dd></div>
                    <div><dt className="inline font-semibold">How to detect: </dt><dd className="inline">{f.detect}</dd></div>
                    <div><dt className="inline font-semibold">Fix: </dt><dd className="inline">{f.fix}</dd></div>
                  </dl>
                  {sc && <Link to={`/failure-lab/${sc.id}`} className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"><FlaskConical size={12} /> Debug this in the Failure Lab</Link>}
                </Panel>
              );
            })}
          </div>
        )}
        {tab === "challenges" && <ChallengeList challenges={content.challenges} />}
      </div>
    </div>
  );
}
