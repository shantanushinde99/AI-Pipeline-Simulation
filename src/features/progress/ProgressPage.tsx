import { Link } from "react-router";
import { CheckCircle2, Circle } from "lucide-react";
import { TOPICS } from "../../data/topics";
import { ALL_CHALLENGES } from "../../visualizers/registry";
import { SCENARIOS } from "../../scenarios/registry";
import { progressStore, setCompleted, useProgress } from "../../lib/storage";
import { Badge, Button, Panel } from "../../components/ui";

export function ProgressPage() {
  const p = useProgress();
  const implemented = TOPICS.filter((t) => t.status === "implemented");
  const correct = ALL_CHALLENGES.filter((c) => p.challenges[c.id]?.correct).length;
  const answered = ALL_CHALLENGES.filter((c) => p.challenges[c.id]).length;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Progress</h1>
          <p className="mt-1 text-sm text-muted">Stored only in this browser (local storage). Nothing is sent anywhere.</p>
        </div>
        <Button size="sm" onClick={() => window.confirm("Reset all learning progress in this browser?") && progressStore.reset()}>Reset progress</Button>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <Panel title="Visualizers completed"><div className="font-mono text-2xl" data-testid="topics-done">{implemented.filter((t) => p.completed[t.id]).length}/{implemented.length}</div></Panel>
        <Panel title="Concept questions"><div className="font-mono text-2xl">{correct}/{ALL_CHALLENGES.length}</div><p className="text-xs text-muted">{answered} answered</p></Panel>
        <Panel title="Incidents solved (4/4)"><div className="font-mono text-2xl">{SCENARIOS.filter((s) => p.scenarios[s.id]?.score === p.scenarios[s.id]?.max).length}/{SCENARIOS.length}</div></Panel>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Visualizers">
          <ul className="divide-y divide-line">
            {implemented.map((t) => (
              <li key={t.id} className="flex items-center gap-2 py-2 text-sm">
                {p.completed[t.id] ? <CheckCircle2 size={15} className="text-ok" aria-label="completed" /> : <Circle size={15} className="text-muted" aria-hidden />}
                <Link to={`/visualizer/${t.id}`} className="flex-1 hover:text-accent">{t.title}</Link>
                {p.visited[t.id] && !p.completed[t.id] && <Badge>started</Badge>}
                <Button size="sm" variant="ghost" onClick={() => setCompleted(t.id, !p.completed[t.id])}>{p.completed[t.id] ? "Mark incomplete" : "Mark complete"}</Button>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel title="Failure Lab">
          <ul className="divide-y divide-line">
            {SCENARIOS.map((s) => {
              const r = p.scenarios[s.id];
              return (
                <li key={s.id} className="flex items-center gap-2 py-2 text-sm">
                  <Link to={`/failure-lab/${s.id}`} className="flex-1 hover:text-accent">{s.title}</Link>
                  {r ? <Badge tone={r.score === r.max ? "ok" : "warn"}>best {r.score}/{r.max}</Badge> : <span className="text-xs text-muted">not attempted</span>}
                </li>
              );
            })}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
