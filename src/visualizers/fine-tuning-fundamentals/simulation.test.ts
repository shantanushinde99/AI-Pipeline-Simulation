import { describe, expect, it } from "vitest";
import {
  D, DEFAULT_FT, H, LR_PRESETS, ORIGINAL_EVAL, ORIGINAL_TRAIN, TARGET_TRAIN, TARGET_VAL, bce, checkpoints, effectiveW, evaluate,
  forward, gradients, matmul, paramCount, pretrained, run, simulate, type Params,
} from "./simulation";

const final = (cfg: Partial<typeof DEFAULT_FT>) => {
  const r = run({ ...DEFAULT_FT, ...cfg });
  return { ...r, first: r.history[0].m, last: r.history.at(-1)!.m };
};

describe("fine-tuning simulation", () => {
  it("backprop gradients match finite differences", () => {
    const p = pretrained();
    const batch = TARGET_TRAIN.slice(0, 4);
    const g = gradients(p, batch);
    const loss = (q: Params) => batch.reduce((s, e) => s + bce(forward(q, e.x).z, e.y), 0) / batch.length;
    const h = 1e-6;
    for (const [j, i] of [[0, 0], [3, 4], [7, 9]]) {
      const up = structuredClone(p), dn = structuredClone(p);
      up.W[j][i] += h; dn.W[j][i] -= h;
      expect(g.W[j][i]).toBeCloseTo((loss(up) - loss(dn)) / (2 * h), 6);
    }
    const up = structuredClone(p), dn = structuredClone(p);
    up.w2[2] += h; dn.w2[2] -= h;
    expect(g.w2[2]).toBeCloseTo((loss(up) - loss(dn)) / (2 * h), 6);
  });

  it("bce is the stable form of −[y log p + (1 − y) log(1 − p)]", () => {
    for (const z of [-3, 0.4, 2]) for (const y of [0, 1]) {
      const p = 1 / (1 + Math.exp(-z));
      expect(bce(z, y)).toBeCloseTo(-(y * Math.log(p) + (1 - y) * Math.log(1 - p)), 12);
    }
    expect(Number.isFinite(bce(800, 0))).toBe(true);
  });

  it("the pretrained base model knows movies but fails on phones", () => {
    const b = pretrained();
    expect(evaluate(b, ORIGINAL_TRAIN).acc).toBe(1);
    expect(evaluate(b, ORIGINAL_EVAL).acc).toBe(1);
    expect(evaluate(b, TARGET_VAL).acc).toBeLessThan(0.75);
    expect(TARGET_TRAIN).toHaveLength(8);
    expect(TARGET_VAL.length).toBeGreaterThan(20);
  });

  it("full fine-tuning learns the new task but forgets the original one (defaults)", () => {
    const { first, last, regime } = final({});
    expect(last.valAcc).toBeGreaterThan(first.valAcc + 0.15);
    expect(last.origAcc).toBeLessThan(first.origAcc - 0.2);
    expect(last.origLoss).toBeGreaterThan(first.origLoss + 0.3);
    expect(regime).toBe("forgot");
  });

  it("LoRA trains far fewer numbers, never edits W, and forgets less than full fine-tuning", () => {
    const full = final({});
    const lora = final({ method: "lora", rank: 1 });
    expect(lora.params.trainable).toBe(H + D);
    expect(lora.params.trainable).toBeLessThan(full.params.trainable / 4);
    expect(lora.history.at(-1)!.params.W).toEqual(pretrained().W);
    expect(lora.last.valAcc).toBeGreaterThan(lora.first.valAcc + 0.1);
    expect(lora.last.origAcc).toBeGreaterThan(full.last.origAcc + 0.15);
    expect(lora.forgetting).toBeLessThan(full.forgetting);
    // Detaching the adapter restores the base model exactly.
    expect(lora.detached!.loss).toBeCloseTo(lora.first.origLoss, 12);
  });

  it("ΔW = B·A has rank ≤ r", () => {
    const r = run({ ...DEFAULT_FT, method: "lora", rank: 2 });
    const s = r.history.at(-1)!;
    const dW = effectiveW(s.params, s.lora).map((row, j) => row.map((v, i) => v - s.params.W[j][i]));
    const BA = matmul(s.lora!.B, s.lora!.A);
    dW.forEach((row, j) => row.forEach((v, i) => expect(v).toBeCloseTo(BA[j][i], 12)));
    // Every 3×3 minor of a rank-2 matrix is 0.
    const m = [0, 1, 2].map((j) => [0, 4, 5].map((i) => dW[j][i]));
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    expect(Math.abs(det)).toBeLessThan(1e-9);
  });

  it("replaying original data reduces forgetting", () => {
    const none = final({});
    const replay = final({ replay: 0.4 });
    expect(replay.nReplay).toBe(Math.round(ORIGINAL_TRAIN.length * 0.4));
    expect(replay.last.origAcc).toBeGreaterThan(none.last.origAcc + 0.15);
    expect(replay.last.valAcc).toBeGreaterThan(replay.first.valAcc + 0.15);
  });

  it("a frozen base with only the head trainable cannot learn the flipped words", () => {
    const head = final({ method: "head" });
    expect(head.params.trainable).toBe(H + 1);
    expect(head.history.at(-1)!.params.W).toEqual(pretrained().W);
    expect(head.last.valAcc).toBeLessThan(final({}).last.valAcc);
    expect(head.regime).toBe("underfit");
  });

  it("too high a learning rate diverges; too many epochs overfits", () => {
    expect(final({ lr: LR_PRESETS.high, epochs: 20 }).regime).toBe("diverged");
    const long = final({ method: "lora", rank: 3, lr: 0.5, epochs: 100 });
    expect(long.regime).toBe("overfit");
    expect(long.last.valLoss).toBeGreaterThan(1.35 * long.bestVal);
    expect(long.last.trainLoss).toBeLessThan(0.05);
  });

  it("is deterministic and produces base → data → epochs → result steps", () => {
    const a = simulate(DEFAULT_FT), b = simulate(DEFAULT_FT);
    expect(a.steps.map((s) => s.state.headline)).toEqual(b.steps.map((s) => s.state.headline));
    expect(a.steps[0].state.kind).toBe("base");
    expect(a.steps[1].state.kind).toBe("data");
    expect(a.steps.at(-1)!.state.kind).toBe("result");
    expect(a.steps.filter((s) => s.state.kind === "epoch").at(-1)!.state.epoch).toBe(DEFAULT_FT.epochs);
    expect(checkpoints(5)).toEqual([1, 2, 3, 4, 5]);
    expect(checkpoints(40).at(-1)).toBe(40);
    expect(paramCount("full", 1)).toEqual({ total: H * D + 2 * H + 1, trainable: H * D + 2 * H + 1 });
  });
});
