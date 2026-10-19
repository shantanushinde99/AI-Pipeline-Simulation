import { describe, expect, it } from "vitest";
import { TOPICS, getTopic, searchTopics } from "../data/topics";
import { VISUALIZERS } from "../visualizers/registry";
import { getScenario } from "../scenarios/registry";
import { PATHS } from "../data/paths";

describe("registries", () => {
  it("a topic is implemented exactly when it has a visualizer", () => {
    expect(TOPICS.filter((t) => t.status === "implemented").map((t) => t.id).sort()).toEqual(Object.keys(VISUALIZERS).sort());
  });
  it("topic ids are unique and coveredBy points at implemented topics", () => {
    expect(new Set(TOPICS.map((t) => t.id)).size).toBe(TOPICS.length);
    TOPICS.filter((t) => t.coveredBy).forEach((t) => expect(getTopic(t.coveredBy!)?.status, t.id).toBe("implemented"));
  });
  it("failure modes and learning paths only link to things that exist", () => {
    Object.values(VISUALIZERS).flatMap((v) => v.content.failureModes).filter((f) => f.scenarioId).forEach((f) => expect(getScenario(f.scenarioId!), f.scenarioId).toBeDefined());
    PATHS.flatMap((p) => p.steps).forEach((s) => expect(s.kind === "topic" ? getTopic(s.id) : getScenario(s.id), s.id).toBeDefined());
  });
  it("every challenge has a valid answer index and unique id", () => {
    const all = Object.values(VISUALIZERS).flatMap((v) => v.content.challenges);
    expect(new Set(all.map((c) => c.id)).size).toBe(all.length);
    all.forEach((c) => expect(c.answer).toBeLessThan(c.options.length));
  });
  it("every visualizer meets the content bar", () => {
    for (const [id, v] of Object.entries(VISUALIZERS)) {
      expect(getTopic(id), `${id} has no topic entry`).toBeDefined();
      expect(v.meta.summary.length, `${id} summary`).toBeGreaterThan(20);
      expect(v.meta.minutes, `${id} minutes`).toBeGreaterThan(0);
      expect(v.content.concept.length, `${id} concept`).toBeGreaterThanOrEqual(3);
      expect(v.content.concept.some((c) => /assumption/i.test(c.heading)), `${id} needs an "Assumptions" concept section`).toBe(true);
      expect(v.content.code.length, `${id} code`).toBeGreaterThanOrEqual(1);
      expect(v.content.failureModes.length, `${id} failure modes`).toBeGreaterThanOrEqual(3);
      expect(v.content.challenges.length, `${id} challenges`).toBeGreaterThanOrEqual(4);
      v.content.challenges.forEach((c) => expect(c.topicId, c.id).toBe(id));
    }
  });
  it("search filters by text and category", () => {
    expect(searchTopics("chunk").map((t) => t.id)).toContain("chunking-strategies");
    expect(searchTopics("", "agents").every((t) => t.category === "agents")).toBe(true);
  });
});
