import type { Scenario } from "./types";
import { RAG_SCENARIOS } from "./rag-scenarios";
import { RAG_EXTRA_SCENARIOS } from "./rag-extra-scenarios";
import { AGENT_SCENARIOS } from "./agent-scenarios";
import { AGENT_EXTRA_SCENARIOS } from "./agent-extra-scenarios";
import { ML_SCENARIOS } from "./ml-scenarios";
import { ML_EXTRA_SCENARIOS } from "./ml-extra-scenarios";
import { PROD_SCENARIOS } from "./prod-scenarios";
import { BEGINNER_SCENARIOS } from "./beginner-scenarios";
import { RAG_MORE_SCENARIOS } from "./rag-more-scenarios";
import { AGENT_MORE_SCENARIOS } from "./agent-more-scenarios";
import { ML_MORE_SCENARIOS } from "./ml-more-scenarios";
import { PROD_MORE_SCENARIOS } from "./prod-more-scenarios";

const beginner = (system: Scenario["system"]) => BEGINNER_SCENARIOS.filter((s) => s.system === system);

export const SCENARIOS: Scenario[] = [
  ...RAG_SCENARIOS.filter((s) => s.system === "Production"), ...PROD_SCENARIOS, ...beginner("Production"), ...PROD_MORE_SCENARIOS,
  ...RAG_SCENARIOS.filter((s) => s.system !== "Production"), ...RAG_EXTRA_SCENARIOS, ...beginner("RAG"), ...RAG_MORE_SCENARIOS,
  ...AGENT_SCENARIOS, ...AGENT_EXTRA_SCENARIOS, ...beginner("Agent"), ...AGENT_MORE_SCENARIOS,
  ...ML_SCENARIOS, ...ML_EXTRA_SCENARIOS, ...beginner("ML"), ...ML_MORE_SCENARIOS,
];
export const getScenario = (id: string) => SCENARIOS.find((s) => s.id === id);
