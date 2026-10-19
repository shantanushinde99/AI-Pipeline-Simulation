import type { Scenario } from "./types";
import { RAG_SCENARIOS } from "./rag-scenarios";
import { RAG_EXTRA_SCENARIOS } from "./rag-extra-scenarios";
import { AGENT_SCENARIOS } from "./agent-scenarios";
import { AGENT_EXTRA_SCENARIOS } from "./agent-extra-scenarios";
import { ML_SCENARIOS } from "./ml-scenarios";
import { ML_EXTRA_SCENARIOS } from "./ml-extra-scenarios";
import { PROD_SCENARIOS } from "./prod-scenarios";

export const SCENARIOS: Scenario[] = [
  ...RAG_SCENARIOS.filter((s) => s.system === "Production"), ...PROD_SCENARIOS,
  ...RAG_SCENARIOS.filter((s) => s.system !== "Production"), ...RAG_EXTRA_SCENARIOS,
  ...AGENT_SCENARIOS, ...AGENT_EXTRA_SCENARIOS,
  ...ML_SCENARIOS, ...ML_EXTRA_SCENARIOS,
];
export const getScenario = (id: string) => SCENARIOS.find((s) => s.id === id);
