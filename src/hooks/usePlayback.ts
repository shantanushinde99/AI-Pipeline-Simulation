import { useCallback, useEffect, useState } from "react";

export const SPEEDS = [0.5, 1, 2, 4] as const;

/**
 * Playback over a precomputed list of steps. It only moves an index, so the same inputs always
 * render the same frames. The index is clamped (not reset) when the step count changes, which lets
 * a learner tweak an input and see its effect on the step they are looking at.
 */
export function usePlayback(total: number, interval: number | ((index: number) => number) = 1400) {
  const [rawIndex, setIndex] = useState(0);
  const [wantPlay, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const last = Math.max(0, total - 1);
  const index = Math.min(rawIndex, last);
  const atEnd = index >= last;
  const playing = wantPlay && !atEnd;

  useEffect(() => {
    if (!playing) return;
    // A function interval lets each step stay on screen as long as its own animation takes.
    const ms = typeof interval === "function" ? interval(index) : interval;
    const t = window.setTimeout(() => {
      setIndex(index + 1);
      if (index + 1 >= last) setPlaying(false);
    }, ms / speed);
    return () => window.clearTimeout(t);
  }, [playing, index, last, speed, interval]);

  const play = useCallback(() => {
    if (atEnd) setIndex(0);
    setPlaying(true);
  }, [atEnd]);
  const pause = useCallback(() => setPlaying(false), []);
  const next = useCallback(() => {
    setPlaying(false);
    setIndex((i) => Math.min(Math.min(i, last) + 1, last));
  }, [last]);
  const prev = useCallback(() => {
    setPlaying(false);
    setIndex((i) => Math.max(Math.min(i, last) - 1, 0));
  }, [last]);
  const restart = useCallback(() => {
    setPlaying(false);
    setIndex(0);
  }, []);
  const seek = useCallback(
    (i: number) => {
      setPlaying(false);
      setIndex(Math.max(0, Math.min(i, last)));
    },
    [last],
  );

  return { index, total, playing, speed, atEnd, play, pause, next, prev, restart, seek, setSpeed };
}

export type Playback = ReturnType<typeof usePlayback>;
