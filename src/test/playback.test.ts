import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePlayback } from "../hooks/usePlayback";

describe("usePlayback", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("steps, clamps, seeks and restarts", () => {
    const { result } = renderHook(() => usePlayback(3));
    act(() => result.current.prev());
    expect(result.current.index).toBe(0);
    act(() => result.current.next());
    act(() => result.current.next());
    act(() => result.current.next());
    expect(result.current.index).toBe(2);
    expect(result.current.atEnd).toBe(true);
    act(() => result.current.seek(1));
    expect(result.current.index).toBe(1);
    act(() => result.current.restart());
    expect(result.current.index).toBe(0);
  });

  it("plays to the end and stops; speed shortens the interval", () => {
    const { result } = renderHook(() => usePlayback(3, 1000));
    act(() => result.current.setSpeed(2));
    act(() => result.current.play());
    expect(result.current.playing).toBe(true);
    act(() => void vi.advanceTimersByTime(500));
    expect(result.current.index).toBe(1);
    act(() => void vi.advanceTimersByTime(500));
    expect(result.current.index).toBe(2);
    expect(result.current.playing).toBe(false);
    act(() => void vi.advanceTimersByTime(5000));
    expect(result.current.index).toBe(2);
  });

  it("pause stops advancing; play at the end restarts from 0", () => {
    const { result } = renderHook(() => usePlayback(3, 100));
    act(() => result.current.play());
    act(() => result.current.pause());
    act(() => void vi.advanceTimersByTime(1000));
    expect(result.current.index).toBe(0);
    act(() => result.current.seek(2));
    act(() => result.current.play());
    expect(result.current.index).toBe(0);
  });

  it("clamps the index when the number of steps shrinks", () => {
    const { result, rerender } = renderHook(({ n }) => usePlayback(n), { initialProps: { n: 10 } });
    act(() => result.current.seek(8));
    rerender({ n: 4 });
    expect(result.current.index).toBe(3);
  });
});
