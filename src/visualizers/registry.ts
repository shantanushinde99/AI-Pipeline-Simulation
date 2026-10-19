import { lazy, type LazyExoticComponent } from "react";
import type { VisualizerComponent, VisualizerContent, VisualizerMeta } from "./types";

type ContentModule = { content: VisualizerContent; meta: VisualizerMeta };
export type VisualizerModule = { Component: LazyExoticComponent<VisualizerComponent>; content: VisualizerContent; meta: VisualizerMeta };

// Every folder with a content.ts and a Visualizer.tsx is a visualizer. Components load lazily; content is small and eager.
const contents = import.meta.glob<ContentModule>("./*/content.ts", { eager: true });
const components = import.meta.glob<{ default: VisualizerComponent }>("./*/Visualizer.tsx");

/** The folder name is the topic id, except for the first visualizers, which predate that convention. */
const FOLDER_TO_TOPIC: Record<string, string> = { "neural-network": "forward-propagation", "model-evaluation": "overfitting", rag: "rag-pipeline", agents: "agent-loop" };

export const VISUALIZERS: Record<string, VisualizerModule> = Object.fromEntries(
  Object.entries(contents).flatMap(([path, mod]) => {
    const folder = path.split("/")[1];
    const load = components[`./${folder}/Visualizer.tsx`] as (typeof components)[string] | undefined;
    return load && mod.content && mod.meta ? [[FOLDER_TO_TOPIC[folder] ?? folder, { Component: lazy(load), content: mod.content, meta: mod.meta }]] : [];
  }),
);

export const ALL_CHALLENGES = Object.values(VISUALIZERS).flatMap((v) => v.content.challenges);
