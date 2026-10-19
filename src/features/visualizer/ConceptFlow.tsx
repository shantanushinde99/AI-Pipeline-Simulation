import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowLeft, ArrowRight, Check, Lightbulb, ShieldCheck } from "lucide-react";
import { Button, cx } from "../../components/ui";

const ease = [0.22, 1, 0.36, 1] as const;
/** First sentence is the key point; the rest become bullets. */
const split = (body: string) => body.split(/(?<=[.!?])\s+(?=[A-Z(‘'"])/);

/** Concept sections as a step-by-step card flow: one idea per card. */
export function ConceptFlow({ sections }: { sections: { heading: string; body: string }[] }) {
  const [i, setI] = useState(0);
  const [seen, setSeen] = useState(() => new Set([0]));
  const go = (n: number) => {
    const k = Math.max(0, Math.min(sections.length - 1, n));
    setI(k);
    setSeen((s) => new Set(s).add(k));
  };
  const s = sections[i];
  const [key, ...rest] = split(s.body);
  const isAssumption = /assumption/i.test(s.heading);

  return (
    <div className="max-w-4xl space-y-4" onKeyDown={(e) => { if (e.key === "ArrowRight") go(i + 1); if (e.key === "ArrowLeft") go(i - 1); }}>
      <ol className="flex flex-wrap items-center gap-y-2" aria-label="Concept steps">
        {sections.map((x, n) => (
          <li key={x.heading} className="flex items-center">
            <button onClick={() => go(n)} aria-current={n === i ? "step" : undefined}
              className={cx("flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
                n === i ? "border-accent bg-accent text-white" : seen.has(n) ? "border-accent/40 bg-accent-soft text-accent" : "border-line bg-surface text-muted hover:text-fg")}>
              <span className={cx("flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold", n === i ? "bg-white/25" : "bg-surface-2")}>
                {seen.has(n) && n !== i ? <Check size={11} /> : n + 1}
              </span>
              {x.heading}
            </button>
            {n < sections.length - 1 && <span className={cx("mx-1 h-0.5 w-5", n < i ? "bg-accent" : "bg-line")} aria-hidden />}
          </li>
        ))}
      </ol>

      <div className="h-1 overflow-hidden rounded-full bg-surface-2">
        <motion.div className="h-full bg-accent" animate={{ width: `${((i + 1) / sections.length) * 100}%` }} transition={{ duration: 0.4, ease }} />
      </div>

      <AnimatePresence mode="wait">
        <motion.article key={i} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -24 }} transition={{ duration: 0.3, ease }}
          className="rounded-xl border border-line bg-surface p-5 shadow-sm">
          <p className="text-xs font-semibold tracking-wide text-muted uppercase">Idea {i + 1} of {sections.length}</p>
          <h2 className="mt-1 text-xl font-semibold">{s.heading}</h2>

          <div className={cx("mt-4 flex gap-3 rounded-lg p-4", isAssumption ? "bg-surface-2" : "bg-accent-soft")}>
            {isAssumption ? <ShieldCheck size={22} className="shrink-0 text-muted" aria-hidden /> : <Lightbulb size={22} className="shrink-0 text-accent" aria-hidden />}
            <p className="text-base leading-relaxed font-medium">{key}</p>
          </div>

          {rest.length > 0 && (
            <ul className="mt-4 space-y-2">
              {rest.map((t, n) => (
                <motion.li key={t} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + n * 0.15, ease }}
                  className="flex gap-3 rounded-lg border border-line p-3 text-sm leading-relaxed">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-surface-2 text-[10px] font-bold text-muted">{n + 1}</span>
                  {t}
                </motion.li>
              ))}
            </ul>
          )}
        </motion.article>
      </AnimatePresence>

      <div className="flex items-center justify-between">
        <Button variant="secondary" onClick={() => go(i - 1)} disabled={i === 0}><ArrowLeft size={14} /> Previous</Button>
        <span className="hidden text-xs text-muted sm:inline">Tip: use ← → keys</span>
        {i < sections.length - 1
          ? <Button variant="primary" onClick={() => go(i + 1)}>Next: {sections[i + 1].heading} <ArrowRight size={14} /></Button>
          : <span className="flex items-center gap-1 text-sm font-medium text-ok"><Check size={15} /> All {sections.length} ideas covered</span>}
      </div>
    </div>
  );
}
