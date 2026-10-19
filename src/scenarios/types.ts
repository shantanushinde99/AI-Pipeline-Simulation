import type { LucideIcon } from "lucide-react";
import type { TraceSpan } from "../types/simulation";
import type { Difficulty } from "../types/topic";

export type SystemType = "RAG" | "Agent" | "ML" | "Production";

export type HealthMetric = { label: string; value: string; status: "ok" | "warning" | "error"; kind: "health" | "quality"; hint?: string };

/** A confirmed observation produced by running the system. Revealed by a diagnostic action. */
export type Evidence = {
  id: string;
  action: string;
  title: string;
  lines: string[];
  /** True when this observation helps establish the actual root cause. */
  relevant: boolean;
  spanId?: string;
};

export type Outcome = {
  spans: TraceSpan[];
  metrics: HealthMetric[];
  transcript: { label: string; text: string; tone?: "ok" | "err" }[];
  /** Whether the user-facing behaviour is correct after this run. */
  resolved: boolean;
  evidence: Evidence[];
};

export type Hypothesis = {
  id: string;
  text: string;
  isRootCause: boolean;
  /** Which evidence confirms or rules this out, revealed after the attempt. */
  evidenceIds: string[];
  explanation: string;
};

/** A box in the architecture diagram; its status is the worst status of the spans it matches. */
export type ArchNode = { id: string; label: string; Icon: LucideIcon; x: number; y: number; match: (span: TraceSpan) => boolean };

export type Scenario = {
  id: string;
  title: string;
  system: SystemType;
  difficulty: Difficulty;
  symptom: string;
  objective: string;
  incident: string;
  relatedTopic?: string;
  architecture: { nodes: ArchNode[]; edges: { source: string; target: string; sourceHandle?: "r" | "b"; targetHandle?: "l" | "t" }[] };
  /** Runs the real simulation with the scenario's fault; with a fix id, runs it with that change applied. */
  run: (fixId?: string) => Outcome;
  hypotheses: Hypothesis[];
  fixes: { id: string; label: string; explanation: string }[];
  verifications: { id: string; label: string; sound: boolean; explanation: string }[];
  rootCause: string;
};
