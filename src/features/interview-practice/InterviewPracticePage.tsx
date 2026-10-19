import { useState } from "react";
import { Link } from "react-router";
import { FlaskConical } from "lucide-react";
import { ALL_CHALLENGES } from "../../visualizers/registry";
import { ChallengeList } from "../challenges/ChallengeList";
import { SCENARIOS } from "../../scenarios/registry";
import { getTopic } from "../../data/topics";
import { useProgress } from "../../lib/storage";
import { Badge, Select, Tabs } from "../../components/ui";

export function InterviewPracticePage() {
  const [tab, setTab] = useState<"concepts" | "diagnosis">("concepts");
  const [topic, setTopic] = useState("all");
  const [onlyOpen, setOnlyOpen] = useState(false);
  const progress = useProgress();
  const topics = [...new Set(ALL_CHALLENGES.map((c) => c.topicId))];
  const list = ALL_CHALLENGES.filter((c) => (topic === "all" || c.topicId === topic) && (!onlyOpen || !progress.challenges[c.id]?.correct));
  const answered = ALL_CHALLENGES.filter((c) => progress.challenges[c.id]);
  return (
    <div className="space-y-5">
      <section className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Interview practice</h1>
        <p className="mt-2 text-sm text-muted">Concept questions test what happens when a parameter changes, which stage causes a symptom, and which metric fits. Diagnosis challenges are full Failure Lab incidents: root cause, evidence, fix and verification are scored separately.</p>
        <p className="mt-2 font-mono text-xs text-muted">{answered.filter((c) => progress.challenges[c.id].correct).length}/{ALL_CHALLENGES.length} concept questions correct · {Object.keys(progress.scenarios).length}/{SCENARIOS.length} incidents attempted</p>
      </section>
      <Tabs label="Practice type" value={tab} onChange={setTab} tabs={[{ id: "concepts", label: `Concept questions (${ALL_CHALLENGES.length})` }, { id: "diagnosis", label: `Diagnosis challenges (${SCENARIOS.length})` }]} />
      {tab === "concepts" ? (
        <>
          <div className="flex flex-wrap items-end gap-4">
            <div className="w-72"><Select label="Topic" value={topic} onChange={setTopic} options={[{ value: "all", label: "All topics" }, ...topics.map((t) => ({ value: t, label: getTopic(t)?.title ?? t }))]} /></div>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={onlyOpen} onChange={(e) => setOnlyOpen(e.target.checked)} className="accent-[var(--accent)]" /> Hide questions I've answered correctly</label>
          </div>
          {list.length ? <ChallengeList key={`${topic}-${onlyOpen}`} challenges={list} showTopic /> : <p className="text-sm text-muted">Nothing left here. Every question in this filter is answered correctly.</p>}
        </>
      ) : (
        <ul className="grid gap-3 md:grid-cols-2">
          {SCENARIOS.map((s) => {
            const r = progress.scenarios[s.id];
            return (
              <li key={s.id} className="rounded-lg border border-line bg-surface p-4">
                <div className="flex items-center gap-2"><Badge tone="accent">{s.system}</Badge>{r && <Badge tone={r.score === r.max ? "ok" : "warn"}>best {r.score}/{r.max}</Badge>}</div>
                <h2 className="mt-2 text-sm font-semibold">“{s.symptom}”</h2>
                <p className="mt-1 text-xs text-muted">Prompt: which stage is responsible, what evidence would you inspect first, what would you change, and how would you verify it?</p>
                <Link to={`/failure-lab/${s.id}`} className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"><FlaskConical size={12} /> Start the incident</Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
