import { useSyncExternalStore } from "react";

export type Progress = {
  visited: Record<string, number>;
  completed: Record<string, boolean>;
  challenges: Record<string, { correct: boolean; at: number }>;
  scenarios: Record<string, { score: number; max: number; at: number }>;
};

const KEY = "asv-progress-v1";
const empty: Progress = { visited: {}, completed: {}, challenges: {}, scenarios: {} };

/** The only place that touches persistence. Swap load/save for an API client to move progress to a backend. */
const backend = {
  load(): Progress {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? { ...empty, ...JSON.parse(raw) } : empty;
    } catch {
      return empty;
    }
  },
  save(p: Progress) {
    try {
      localStorage.setItem(KEY, JSON.stringify(p));
    } catch {
      /* storage unavailable (private mode); progress stays in memory */
    }
  },
};

let state = backend.load();
const listeners = new Set<() => void>();

export const progressStore = {
  get: () => state,
  subscribe(fn: () => void) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
  update(fn: (p: Progress) => Progress) {
    state = fn(state);
    backend.save(state);
    listeners.forEach((l) => l());
  },
  reset() {
    progressStore.update(() => empty);
  },
};

export const useProgress = () => useSyncExternalStore(progressStore.subscribe, progressStore.get);

export const markVisited = (topicId: string) =>
  progressStore.update((p) => (p.visited[topicId] ? p : { ...p, visited: { ...p.visited, [topicId]: Date.now() } }));
export const setCompleted = (topicId: string, done: boolean) =>
  progressStore.update((p) => ({ ...p, completed: { ...p.completed, [topicId]: done } }));
export const recordChallenge = (id: string, correct: boolean) =>
  progressStore.update((p) => ({ ...p, challenges: { ...p.challenges, [id]: { correct, at: Date.now() } } }));
/** Keeps the best score per scenario. */
export const recordScenario = (id: string, score: number, max: number) =>
  progressStore.update((p) => (p.scenarios[id] && p.scenarios[id].score >= score ? p : { ...p, scenarios: { ...p.scenarios, [id]: { score, max, at: Date.now() } } }));
