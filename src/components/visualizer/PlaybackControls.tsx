import { useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { SPEEDS, type Playback } from "../../hooks/usePlayback";
import { Button, cx } from "../ui";

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

/** Shared transport: play/pause, step, restart, scrub, speed. Keyboard: Space, ←, →, Home. */
export function PlaybackControls({ pb, stepTitles }: { pb: Playback; stepTitles: string[] }) {
  const { index, total, playing } = pb;
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore keys while typing, and while this visualizer sits in a hidden tab.
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey || !ref.current || ref.current.closest("[hidden]")) return;
      if (e.key === "ArrowRight") pb.next();
      else if (e.key === "ArrowLeft") pb.prev();
      else if (e.key === "Home") pb.restart();
      else if (e.key === " " && !(e.target instanceof HTMLButtonElement)) {
        e.preventDefault();
        if (playing) pb.pause();
        else pb.play();
      } else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pb, playing]);

  return (
    <div ref={ref} className="space-y-2 rounded-lg border border-line bg-surface p-2.5" aria-label="Playback controls" role="group">
      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="sm" onClick={pb.restart} aria-label="Restart" title="Restart (Home)"><RotateCcw size={14} /></Button>
        <Button size="sm" onClick={pb.prev} disabled={index === 0} aria-label="Previous step" title="Previous (←)"><ChevronLeft size={14} /></Button>
        <Button size="sm" variant="primary" onClick={playing ? pb.pause : pb.play} aria-label={playing ? "Pause" : "Play"} title="Play/Pause (Space)" className="w-20">
          {playing ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Play</>}
        </Button>
        <Button size="sm" onClick={pb.next} disabled={pb.atEnd} aria-label="Next step" title="Next (→)"><ChevronRight size={14} /></Button>
        <span className="ml-1 font-mono text-xs text-muted" aria-live="polite" data-testid="step-counter">
          Step {index + 1} / {total}
        </span>
        <label className="ml-auto flex items-center gap-1 text-xs text-muted">
          Speed
          <select value={pb.speed} onChange={(e) => pb.setSpeed(Number(e.target.value))} className="h-7 rounded border border-line bg-surface px-1 font-mono text-xs text-fg">
            {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </label>
      </div>
      <div className="flex gap-0.5" aria-label="Step timeline">
        {stepTitles.map((t, i) => (
          <button
            key={i}
            onClick={() => pb.seek(i)}
            title={`${i + 1}. ${t}`}
            aria-label={`Jump to step ${i + 1}: ${t}`}
            aria-current={i === index ? "step" : undefined}
            className={cx("h-2 flex-1 rounded-sm transition-all duration-300", i === index ? "scale-y-150 bg-accent" : i < index ? "bg-accent/40" : "bg-surface-2 hover:bg-line")}
          />
        ))}
      </div>
      <div className="truncate text-xs"><span className="text-muted">Current: </span><span className="font-medium">{stepTitles[index]}</span></div>
    </div>
  );
}
