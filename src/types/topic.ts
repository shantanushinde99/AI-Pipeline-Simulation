export type CategoryId = "ml" | "dl" | "rag" | "agents" | "prod" | "eval";
export type Difficulty = "Beginner" | "Intermediate" | "Advanced";

export type Topic = {
  id: string;
  title: string;
  category: CategoryId;
  difficulty: Difficulty;
  minutes: number;
  summary: string;
  /** Only implemented topics have a working visualizer route. */
  status: "implemented" | "planned";
  /** A planned topic whose ideas are already exercised inside an implemented visualizer. */
  coveredBy?: string;
};
