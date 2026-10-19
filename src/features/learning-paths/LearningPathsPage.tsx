import { Link } from "react-router";
import { CheckCircle2, Circle, FlaskConical, PlayCircle } from "lucide-react";
import { PATHS, type PathStep } from "../../data/paths";
import { getTopic } from "../../data/topics";
import { getScenario } from "../../scenarios/registry";
import { useProgress, type Progress } from "../../lib/storage";

export const stepDone = (p: Progress, st: PathStep) =>
  st.kind === "topic" ? !!p.completed[st.id] : !!p.scenarios[st.id] && p.scenarios[st.id].score === p.scenarios[st.id].max;

export function LearningPathsPage() {
  const progress = useProgress();
  return (
    <div className="space-y-5">
      <section className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Learning paths</h1>
        <p className="mt-2 text-sm text-muted">Each path pairs visualizers (learn how it works) with Failure Lab scenarios (debug it when it breaks). A visualizer step counts as done when you mark it complete; a scenario step when you score 4/4.</p>
      </section>
      <div className="grid gap-4 lg:grid-cols-2">
        {PATHS.map((path) => {
          const done = path.steps.filter((st) => stepDone(progress, st)).length;
          return (
            <section key={path.id} className="rounded-lg border border-line bg-surface p-4" aria-labelledby={`p-${path.id}`}>
              <div className="flex items-baseline justify-between gap-2">
                <h2 id={`p-${path.id}`} className="font-semibold">{path.title}</h2>
                <span className="font-mono text-xs text-muted">{done}/{path.steps.length}</span>
              </div>
              <p className="mt-1 text-sm text-muted">{path.summary}</p>
              <div className="mt-2 h-1.5 overflow-hidden rounded bg-surface-2" role="progressbar" aria-valuenow={done} aria-valuemin={0} aria-valuemax={path.steps.length} aria-label={`${path.title} progress`}>
                <div className="h-full bg-ok" style={{ width: `${(done / path.steps.length) * 100}%` }} />
              </div>
              <ol className="mt-3 space-y-1">
                {path.steps.map((st, i) => {
                  const isDone = stepDone(progress, st);
                  const title = st.kind === "topic" ? getTopic(st.id)?.title : getScenario(st.id)?.title;
                  return (
                    <li key={st.id}>
                      <Link to={st.kind === "topic" ? `/visualizer/${st.id}` : `/failure-lab/${st.id}`} className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-surface-2">
                        <span className="w-4 font-mono text-xs text-muted">{i + 1}</span>
                        {isDone ? <CheckCircle2 size={15} className="text-ok" aria-label="done" /> : <Circle size={15} className="text-muted" aria-hidden />}
                        {st.kind === "topic" ? <PlayCircle size={14} className="text-accent" aria-label="visualizer" /> : <FlaskConical size={14} className="text-warn" aria-label="scenario" />}
                        <span>{title}</span>
                      </Link>
                    </li>
                  );
                })}
              </ol>
            </section>
          );
        })}
      </div>
    </div>
  );
}
