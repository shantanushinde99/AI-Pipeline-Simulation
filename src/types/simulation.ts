export type Severity = "info" | "warning" | "error";

export type SimulationEvent = {
  id: string;
  type: string;
  message: string;
  /** Optional LaTeX rendering of `message` for formula-like events; `message` stays the plain-text log line. */
  tex?: string;
  severity?: Severity;
  metadata?: Record<string, unknown>;
};

/** One immutable snapshot of a simulation. Visualizers precompute every step, playback only moves an index. */
export type SimulationStep<TState> = {
  id: string;
  title: string;
  description: string;
  state: TState;
  events: SimulationEvent[];
};

export type SpanStatus = "ok" | "warning" | "error";

/** Shaped like an OpenTelemetry span so real traces can be mapped in later. Times are simulated milliseconds. */
export type TraceSpan = {
  id: string;
  parentId?: string;
  name: string;
  component: string;
  startMs: number;
  durationMs: number;
  status: SpanStatus;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  events: SimulationEvent[];
  metrics?: Record<string, number | string>;
};
