import type { NNState } from "./simulation";

/**
 * One schedule (in seconds at 1× speed) drives the travelling values in the diagram, the contribution
 * bars and the counters, so everything on screen moves in sync. Divide by playback speed before use.
 */
export const TRAVEL = 0.8;
export const pulseStart = (i: number) => 0.2 + i * 0.5;
export const arrive = (i: number) => pulseStart(i) + TRAVEL;
export const biasAt = (inputs: number) => arrive(inputs - 1) + 0.25;
export const sumAt = (inputs: number) => biasAt(inputs) + 0.45;
export const COUNT = 0.6;
/** Activation step: curve draws, then z → curve → a. */
export const ACT = { curve: 0, zLine: 0.6, dot: 1.1, aLine: 1.35, morph: 1.4 };

/** How long each of the 10 steps stays on screen during auto-play (ms at 1×): long enough to finish its animation. */
const DURATIONS = [2000, 3600, 2900, 3600, 2900, 3600, 2900, 4100, 2900, 2600];
export const stepDuration = (index: number) => DURATIONS[index] ?? 2500;

export function narrate(s: NNState, activation: string, y?: number): { headline: string; sub: string } {
  if (s.phase === "input") return { headline: "The two input values enter the network", sub: "The input layer only holds the numbers. Nothing is calculated yet." };
  if (s.phase === "predict") return { headline: `Final answer: ŷ = ${y?.toFixed(2)}, so the network predicts class ${y !== undefined && y >= 0.5 ? 1 : 0}`, sub: "Every number you saw came from the inputs, weights and biases on screen. Change one and replay." };
  const who = s.active === "y" ? "The output neuron" : `Hidden neuron ${s.active}`;
  if (s.phase === "sum")
    return {
      headline: `${who} adds up what flows into it`,
      sub: s.active === "y"
        ? "Each hidden neuron's output is multiplied by the weight on its line, then the bias is added."
        : "Each input is multiplied by the weight on its line, then the bias is added. Watch the values travel.",
    };
  return {
    headline: s.active === "y" ? "Sigmoid turns that sum into a probability" : `${s.active} passes its sum through ${activation}`,
    sub: s.active === "y" ? "Any number goes in; a value between 0 and 1 comes out." : "This bend is what lets a network learn more than straight-line rules.",
  };
}
