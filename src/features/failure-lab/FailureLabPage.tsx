import { useState } from "react";
import { Link } from "react-router";
import { ArrowRight, FlaskConical } from "lucide-react";
import { SCENARIOS } from "../../scenarios/registry";
import type { SystemType } from "../../scenarios/types";
import { useProgress } from "../../lib/storage";
import { Badge, cx } from "../../components/ui";
import { DifficultyBadge } from "../explore/ExplorePage";

export function FailureLabPage() {
  const [system, setSystem] = useState<SystemType | "all">("all");
  const progress = useProgress();
  const list = SCENARIOS.filter((s) => system === "all" || s.system === system);
  return (
    <div className="space-y-5">
      <section className="max-w-3xl">
        <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight"><FlaskConical size={22} className="text-accent" /> Failure Lab</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Each scenario runs a real simulation with a fault injected. You start with the symptom only. Run diagnostic actions to collect evidence,
          read the trace, weigh hypotheses, then name the root cause, pick a fix and a verification plan. Your fix is applied and the system is re-run, so you see whether it actually works.
        </p>
      </section>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by system">
        {(["all", "Production", "RAG", "Agent", "ML"] as const).map((s) => (
          <button key={s} onClick={() => setSystem(s)} aria-pressed={system === s} className={cx("rounded-full border px-3 py-1 text-xs font-medium", system === s ? "border-accent bg-accent-soft text-accent" : "border-line bg-surface text-muted hover:text-fg")}>
            {s === "all" ? "All systems" : s}
          </button>
        ))}
      </div>
      <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {list.map((s) => {
          const res = progress.scenarios[s.id];
          return (
            <li key={s.id}>
              <Link to={`/failure-lab/${s.id}`} className={cx("group flex h-full flex-col rounded-lg border bg-surface p-4 transition-colors hover:border-accent", s.id === "prod-everything-green" ? "border-accent/50" : "border-line")}>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone="accent">{s.system}</Badge><DifficultyBadge d={s.difficulty} />
                  {res && <Badge tone={res.score === res.max ? "ok" : "warn"} className="ml-auto">best {res.score}/{res.max}</Badge>}
                </div>
                <h2 className="mt-2 text-sm font-semibold group-hover:text-accent">{s.title}</h2>
                <dl className="mt-2 flex-1 space-y-1 text-xs">
                  <div><dt className="inline text-muted">Symptom: </dt><dd className="inline">{s.symptom}</dd></div>
                  <div><dt className="inline text-muted">You'll learn: </dt><dd className="inline">{s.objective}</dd></div>
                </dl>
                <span className="mt-3 flex items-center gap-1 text-xs font-medium text-accent">Investigate <ArrowRight size={12} /></span>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
