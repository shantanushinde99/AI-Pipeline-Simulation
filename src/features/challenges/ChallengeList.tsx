import { useState } from "react";
import { Link } from "react-router";
import { CheckCircle2, XCircle } from "lucide-react";
import type { Challenge } from "../../visualizers/types";
import { recordChallenge, useProgress } from "../../lib/storage";
import { Badge, Button, cx } from "../../components/ui";
import { getTopic } from "../../data/topics";

function ChallengeCard({ c, showTopic }: { c: Challenge; showTopic?: boolean }) {
  const [choice, setChoice] = useState<number | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const prior = useProgress().challenges[c.id];
  const correct = choice === c.answer;
  return (
    <article className="rounded-lg border border-line bg-surface p-4" aria-label={c.question}>
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <Badge>{c.skill}</Badge>
        {showTopic && <Link to={`/visualizer/${c.topicId}`} className="text-xs text-accent hover:underline">{getTopic(c.topicId)?.title}</Link>}
        {prior && <Badge tone={prior.correct ? "ok" : "warn"}>{prior.correct ? "previously correct" : "previously missed"}</Badge>}
      </div>
      <h3 className="text-sm font-medium leading-relaxed">{c.question}</h3>
      <fieldset className="mt-3 space-y-1.5" disabled={submitted}>
        <legend className="sr-only">Options</legend>
        {c.options.map((o, i) => (
          <label key={i} className={cx(
            "flex cursor-pointer items-start gap-2 rounded-md border px-3 py-2 text-sm",
            submitted && i === c.answer ? "border-ok/50 bg-ok-soft" : submitted && i === choice ? "border-err/50 bg-err-soft" : "border-line hover:bg-surface-2",
          )}>
            <input type="radio" name={c.id} checked={choice === i} onChange={() => setChoice(i)} className="mt-1 accent-[var(--accent)]" />
            <span>{o}</span>
          </label>
        ))}
      </fieldset>
      <div className="mt-3 flex items-center gap-2">
        {!submitted ? (
          <Button variant="primary" size="sm" disabled={choice === null} onClick={() => { setSubmitted(true); recordChallenge(c.id, correct); }}>Submit answer</Button>
        ) : (
          <Button size="sm" onClick={() => { setSubmitted(false); setChoice(null); }}>Try again</Button>
        )}
      </div>
      {submitted && (
        <div role="status" className={cx("mt-3 flex gap-2 rounded-md p-3 text-sm leading-relaxed", correct ? "bg-ok-soft" : "bg-err-soft")}>
          {correct ? <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-ok" /> : <XCircle size={16} className="mt-0.5 shrink-0 text-err" />}
          <div><strong>{correct ? "Correct." : "Not quite."}</strong> {c.explanation}</div>
        </div>
      )}
    </article>
  );
}

export function ChallengeList({ challenges, showTopic }: { challenges: Challenge[]; showTopic?: boolean }) {
  return <div className="grid gap-3 lg:grid-cols-2">{challenges.map((c) => <ChallengeCard key={c.id} c={c} showTopic={showTopic} />)}</div>;
}
