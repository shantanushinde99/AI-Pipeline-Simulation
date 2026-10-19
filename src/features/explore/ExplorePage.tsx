import { Link, useSearchParams } from "react-router";
import { ArrowRight, CheckCircle2, Clock, FlaskConical, Search } from "lucide-react";
import { CATEGORIES, TOPICS, categoryOf, getTopic, searchTopics } from "../../data/topics";
import type { CategoryId, Topic } from "../../types/topic";
import { useProgress } from "../../lib/storage";
import { Badge, cx } from "../../components/ui";
import { TopicPreview } from "./TopicPreview";
import { SCENARIOS } from "../../scenarios/registry";

export function DifficultyBadge({ d }: { d: Topic["difficulty"] }) {
  return <Badge tone={d === "Beginner" ? "ok" : d === "Intermediate" ? "accent" : "warn"}>{d}</Badge>;
}

export function ExplorePage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const cat = (params.get("category") ?? "all") as CategoryId | "all";
  const progress = useProgress();
  const set = (k: string, v: string) => {
    const n = new URLSearchParams(params);
    if (v && v !== "all") n.set(k, v);
    else n.delete(k);
    setParams(n, { replace: true });
  };

  const results = searchTopics(q, cat);
  const implemented = results.filter((t) => t.status === "implemented");
  const planned = results.filter((t) => t.status === "planned");
  const scenarios = q
    ? SCENARIOS.filter((s) => `${s.title} ${s.symptom} ${s.system}`.toLowerCase().includes(q.toLowerCase()))
    : [];
  const doneCount = TOPICS.filter((t) => t.status === "implemented" && progress.completed[t.id]).length;

  return (
    <div className="space-y-6">
      <section className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Explore AI systems by running them</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Each visualizer is a deterministic simulation you can step through, change and break. Numbers on screen are computed from the
          inputs on screen. Pick a topic, press play, then change a parameter and watch what moves.
        </p>
      </section>

      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <div className="relative md:w-96">
          <Search size={15} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" aria-hidden />
          <input value={q} onChange={(e) => set("q", e.target.value)} placeholder="Search e.g. chunking, ReLU, retries…" aria-label="Search the topic library"
            className="h-10 w-full rounded-lg border border-line bg-surface pr-3 pl-9 text-sm" />
        </div>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by category">
          {[{ id: "all", short: "All" }, ...CATEGORIES].map((c) => (
            <button key={c.id} onClick={() => set("category", c.id)} aria-pressed={cat === c.id}
              className={cx("rounded-full border px-3 py-1 text-xs font-medium", cat === c.id ? "border-accent bg-accent-soft text-accent" : "border-line bg-surface text-muted hover:text-fg")}>
              {c.short}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted md:ml-auto">{doneCount} of {TOPICS.filter((t) => t.status === "implemented").length} visualizers completed</span>
      </div>

      <section aria-labelledby="impl-h">
        <h2 id="impl-h" className="mb-3 text-sm font-semibold">Interactive visualizers <span className="font-normal text-muted">({implemented.length})</span></h2>
        {implemented.length === 0 ? (
          <p className="text-sm text-muted">No working visualizer matches this search.</p>
        ) : (
          <div className="space-y-6">
          {CATEGORIES.filter((c) => implemented.some((t) => t.category === c.id)).map((c) => (
          <div key={c.id}>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">{c.title} <span className="font-normal">({implemented.filter((t) => t.category === c.id).length})</span></h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {implemented.filter((t) => t.category === c.id).map((t) => (
              <Link key={t.id} to={`/visualizer/${t.id}`} className="group flex flex-col rounded-lg border border-line bg-surface p-3 transition-all duration-200 hover:-translate-y-1 hover:border-accent hover:shadow-lg">
                <div className="rounded-md bg-surface-2 p-2"><TopicPreview id={t.id} category={t.category} /></div>
                <div className="mt-3 flex items-center gap-1.5 text-[11px] text-muted">
                  <span>{categoryOf(t.category).short}</span>·<Clock size={11} aria-hidden />{t.minutes} min
                  {progress.completed[t.id] ? <Badge tone="ok" className="ml-auto"><CheckCircle2 size={11} />Done</Badge> : progress.visited[t.id] ? <Badge className="ml-auto">Started</Badge> : null}
                </div>
                <h3 className="mt-1 text-sm font-semibold group-hover:text-accent">{t.title}</h3>
                <p className="mt-1 flex-1 text-xs leading-relaxed text-muted">{t.summary}</p>
                <div className="mt-3 flex items-center justify-between"><DifficultyBadge d={t.difficulty} /><span className="flex items-center gap-1 text-xs font-medium text-accent">Open <ArrowRight size={12} /></span></div>
              </Link>
            ))}
          </div>
          </div>
          ))}
          </div>
        )}
      </section>

      {scenarios.length > 0 && (
        <section aria-labelledby="sc-h">
          <h2 id="sc-h" className="mb-3 text-sm font-semibold">Failure Lab scenarios</h2>
          <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {scenarios.map((s) => (
              <li key={s.id}><Link to={`/failure-lab/${s.id}`} className="flex items-start gap-2 rounded-lg border border-line bg-surface p-3 text-sm hover:border-accent">
                <FlaskConical size={15} className="mt-0.5 text-accent" aria-hidden /><span><span className="font-medium">{s.title}</span><span className="block text-xs text-muted">{s.symptom}</span></span>
              </Link></li>
            ))}
          </ul>
        </section>
      )}

      {planned.length > 0 && <section aria-labelledby="lib-h">
        <h2 id="lib-h" className="mb-1 text-sm font-semibold">Full library <span className="font-normal text-muted">({planned.length} planned)</span></h2>
        <p className="mb-3 text-xs text-muted">Planned topics do not have their own simulation yet. Where one is already exercised inside a working visualizer, it links there.</p>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {CATEGORIES.filter((c) => planned.some((t) => t.category === c.id)).map((c) => (
            <div key={c.id} className="rounded-lg border border-line bg-surface">
              <h3 className="border-b border-line px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted">{c.title}</h3>
              <ul className="divide-y divide-line">
                {planned.filter((t) => t.category === c.id).map((t) => (
                  <li key={t.id} className="flex items-center gap-2 px-3 py-1.5 text-sm">
                    <span className="flex-1 truncate">{t.title}</span>
                    {t.coveredBy ? (
                      <Link to={`/visualizer/${t.coveredBy}`} className="shrink-0 text-[11px] text-accent hover:underline" title={`Covered in ${getTopic(t.coveredBy)?.title}`}>covered in visualizer →</Link>
                    ) : (
                      <Badge>planned</Badge>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>}
    </div>
  );
}
