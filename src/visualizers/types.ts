import type { ComponentType } from "react";

export type Challenge = {
  id: string;
  topicId: string;
  question: string;
  options: string[];
  answer: number;
  explanation: string;
  skill: "parameter" | "prediction" | "diagnosis" | "metric" | "fix" | "verification";
};

export type VisualizerContent = {
  concept: { heading: string; body: string }[];
  code: { title: string; source: string }[];
  failureModes: { title: string; symptom: string; detect: string; fix: string; scenarioId?: string }[];
  challenges: Challenge[];
};

/** Shown on topic cards; `hasMetrics` adds a Metrics tab (the component then renders it when tab === "metrics"). */
export type VisualizerMeta = { summary: string; minutes: number; hasMetrics: boolean };

export type VisualizerTab = "visualize" | "metrics";
export type VisualizerComponent = ComponentType<{ tab: VisualizerTab }>;
