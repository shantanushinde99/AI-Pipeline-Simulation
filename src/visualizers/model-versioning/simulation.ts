import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** FNV-1a 32-bit string hash as 8 hex chars. A deterministic stand-in for a content hash (not cryptographic). */
export function fnv(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

export type Snapshot = { id: string; date: string; rows: number; hash: string };
const snap = (date: string, rows: number): Snapshot => ({ id: `snap-${date}`, date, rows, hash: fnv(`rows=${rows}|cutoff=${date}`) });

/** The feature table as it existed at three dates. `latest` is a mutable pointer that moves forward over time. */
export const SNAPSHOTS = { s1: snap("2026-09-01", 12000), s2: snap("2026-10-01", 15000), s3: snap("2026-10-15", 16500) };

export type HParams = { lr: number; depth: number };
export type Lineage = {
  dataRef: string; // what the run config pointed at
  dataHash: string; // the data the run actually read
  pinned: boolean;
  commit: string;
  codeHash: string; // the code that actually ran (differs from commit when the tree was dirty)
  hparams: HParams;
  seed: number | null; // null = not recorded
};
export type Run = { id: string; lineage: Lineage; actualSeed: number; rows: number };
export type ModelVersion = { version: number; runId: string; hash: string; accuracy: number; lineage: Lineage };

export type Gap = "none" | "unpinned-data" | "unlogged-seed" | "dirty-code";
export const GAPS: Record<Gap, { label: string; note: string }> = {
  none: { label: "No gap: full lineage recorded", note: "Data pinned by hash, clean commit, seed logged." },
  "unpinned-data": { label: "Data read from a moving 'latest' pointer", note: "Run 2 reads s3://features/latest. By the time you retrain, 'latest' points at newer rows." },
  "unlogged-seed": { label: "Random seed not recorded", note: "Run 2 seeds from the clock and never logs it, so a retrain draws a different seed." },
  "dirty-code": { label: "Uncommitted code change", note: "Run 2 ran with a local edit on top of commit b7e41c9 that was never committed." },
};

export const LRS = [0.001, 0.003, 0.01, 0.03, 0.1];
export type MVConfig = { v2: HParams; minGain: number; gap: Gap };
export const DEFAULT_MV: MVConfig = { v2: { lr: 0.01, depth: 6 }, minGain: 0.005, gap: "none" };
const V1_HP: HParams = { lr: 0.03, depth: 4 };
const COMMITS = { c1: "4d2a9f0", c2: "b7e41c9" };

/**
 * Simulated training: the "weights" hash is a pure function of exactly what went in, and accuracy is a smooth
 * made-up response surface plus a small seed-dependent jitter. Same inputs → same hash, bit for bit.
 */
export function train(dataHash: string, rows: number, codeHash: string, hp: HParams, seed: number) {
  const hash = fnv(`${dataHash}|${codeHash}|lr=${hp.lr}|depth=${hp.depth}|seed=${seed}`);
  const jitter = (rng(seed).next() - 0.5) * 0.01;
  const acc = 0.84 + 0.03 * Math.log2(rows / 12000) - 0.08 * (Math.log10(hp.lr) + 2) ** 2 - 0.004 * (hp.depth - 6) ** 2 + jitter;
  return { hash, accuracy: Math.round(clamp(acc, 0.5, 0.99) * 10000) / 10000 };
}

export type Repro = { recorded: Lineage; replay: Lineage & { seedUsed: number; rows: number }; hash: string; accuracy: number; match: boolean; diffs: (keyof Lineage)[] };
export type GateResult = { v1: number; v2: number; minGain: number; pass: boolean };

export type MVPhase = "snapshot" | "train" | "alias" | "promote" | "repro" | "audit";
export type MVState = {
  phase: MVPhase;
  snapshots: Snapshot[];
  runs: Run[];
  models: ModelVersion[];
  aliases: { staging: number | null; production: number | null };
  /** the version an alias moved away from at this step, for the slide animation */
  moved?: { alias: "staging" | "production"; from: number | null; to: number | null };
  focusRow: 0 | 1;
  gate?: GateResult;
  repro?: Repro;
};

export function simulateRegistry(cfg: MVConfig): SimulationStep<MVState>[] {
  const v2hp: HParams = { lr: LRS.includes(cfg.v2.lr) ? cfg.v2.lr : 0.01, depth: clamp(Math.round(cfg.v2.depth), 1, 12) };
  const minGain = clamp(cfg.minGain, 0, 0.2);
  const { s1, s2, s3 } = SNAPSHOTS;

  // Run 1: fully recorded.
  const l1: Lineage = { dataRef: s1.id, dataHash: s1.hash, pinned: true, commit: COMMITS.c1, codeHash: COMMITS.c1, hparams: V1_HP, seed: 42 };
  const t1 = train(s1.hash, s1.rows, l1.codeHash, V1_HP, 42);
  const run1: Run = { id: "run-1", lineage: l1, actualSeed: 42, rows: s1.rows };
  const m1: ModelVersion = { version: 1, runId: run1.id, hash: t1.hash, accuracy: t1.accuracy, lineage: l1 };

  // Run 2: what really happened vs what got recorded depends on the lineage gap.
  const gap = cfg.gap;
  const actualSeed = gap === "unlogged-seed" ? 1729 : 7;
  const codeHash = gap === "dirty-code" ? `${COMMITS.c2}+dirty` : COMMITS.c2;
  const l2: Lineage = {
    dataRef: gap === "unpinned-data" ? "s3://features/latest" : s2.id, dataHash: s2.hash, pinned: gap !== "unpinned-data",
    commit: COMMITS.c2, codeHash, hparams: v2hp, seed: gap === "unlogged-seed" ? null : actualSeed,
  };
  const t2 = train(s2.hash, s2.rows, codeHash, v2hp, actualSeed);
  const run2: Run = { id: "run-2", lineage: l2, actualSeed, rows: s2.rows };
  const m2: ModelVersion = { version: 2, runId: run2.id, hash: t2.hash, accuracy: t2.accuracy, lineage: l2 };

  const gate: GateResult = { v1: m1.accuracy, v2: m2.accuracy, minGain, pass: m2.accuracy >= m1.accuracy + minGain - 1e-12 };

  // Reproduce v2 using only what the registry recorded.
  const rData = l2.pinned ? s2 : s3; // 'latest' has moved on
  const rSeed = l2.seed ?? 9001; // nothing recorded → a fresh seed
  const rCode = l2.commit; // checking out the commit drops any uncommitted edit
  const t3 = train(rData.hash, rData.rows, rCode, v2hp, rSeed);
  const replay = { ...l2, dataHash: rData.hash, codeHash: rCode, seed: rSeed, seedUsed: rSeed, rows: rData.rows };
  const diffs: (keyof Lineage)[] = [];
  if (replay.dataHash !== l2.dataHash) diffs.push("dataHash");
  if (rSeed !== actualSeed) diffs.push("seed");
  if (rCode !== codeHash) diffs.push("codeHash");
  const repro: Repro = { recorded: l2, replay, hash: t3.hash, accuracy: t3.accuracy, match: t3.hash === m2.hash, diffs };

  const steps: SimulationStep<MVState>[] = [];
  let st: MVState = { phase: "snapshot", snapshots: [], runs: [], models: [], aliases: { staging: null, production: null }, focusRow: 0 };
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "registry", message, severity, tex });
  const push = (id: string, title: string, description: string, patch: Partial<MVState>, events: SimulationEvent[]) => {
    st = { ...st, moved: undefined, gate: undefined, repro: undefined, ...patch };
    steps.push({ id, title, description, state: structuredClone(st), events });
  };

  push("snap-1", "Snapshot the training data", `The feature table is frozen at ${s1.date} (${s1.rows.toLocaleString("en-US")} rows) and identified by its content hash ${s1.hash}.`,
    { phase: "snapshot", snapshots: [s1], focusRow: 0 }, [ev("s1", `snapshot ${s1.id}: ${s1.rows} rows → hash ${s1.hash}`)]);
  push("train-1", "Run 1 trains model v1", `Run 1 reads snapshot ${s1.hash} with code ${l1.commit}, lr=${V1_HP.lr}, depth=${V1_HP.depth}, seed=42. The registry stores the weights hash ${m1.hash} and the full lineage.`,
    { phase: "train", runs: [run1], models: [m1], focusRow: 0 },
    [ev("r1", `run-1 → model v1, weights ${m1.hash}, accuracy ${m1.accuracy.toFixed(4)}`, undefined, `h_{v1} = H(${s1.hash} \\Vert ${l1.commit} \\Vert \\theta \\Vert 42) = \\texttt{${m1.hash}}`)]);
  push("stage-1", "Alias 'staging' → v1", "An alias is a named pointer to a version. Moving it copies nothing; the model files stay where they are.",
    { phase: "alias", aliases: { staging: 1, production: null }, moved: { alias: "staging", from: null, to: 1 }, focusRow: 0 }, [ev("a1", "set alias staging = v1")]);
  push("prod-1", "Alias 'production' → v1", "The serving endpoint loads whatever 'production' points to, so it now serves v1.",
    { phase: "alias", aliases: { staging: 1, production: 1 }, moved: { alias: "production", from: null, to: 1 }, focusRow: 0 }, [ev("a2", "set alias production = v1; endpoint reloads v1")]);
  push("snap-2", gap === "unpinned-data" ? "Run 2 points at 'latest'" : "A newer snapshot arrives",
    gap === "unpinned-data" ? `Run 2's config says s3://features/latest. Today that resolves to the ${s2.date} rows (${s2.hash}), but the config does not record that.` : `A month later the table is frozen again at ${s2.date} (${s2.rows.toLocaleString("en-US")} rows), hash ${s2.hash}.`,
    { phase: "snapshot", snapshots: [s1, s2], focusRow: 1 },
    [gap === "unpinned-data" ? ev("s2", "data reference is a moving pointer: 'latest' (not pinned by hash)", "warning") : ev("s2", `snapshot ${s2.id}: ${s2.rows} rows → hash ${s2.hash}`)]);
  push("train-2", "Run 2 trains model v2", `Run 2 uses lr=${v2hp.lr}, depth=${v2hp.depth}. It reaches accuracy ${m2.accuracy.toFixed(4)} (v1: ${m1.accuracy.toFixed(4)}).`,
    { phase: "train", runs: [run1, run2], models: [m1, m2], focusRow: 1 },
    [ev("r2", `run-2 → model v2, weights ${m2.hash}, accuracy ${m2.accuracy.toFixed(4)}`),
      ...(gap === "unlogged-seed" ? [ev("r2s", "seed not logged (taken from the clock)", "warning")] : []),
      ...(gap === "dirty-code" ? [ev("r2c", "working tree dirty: code that ran ≠ commit b7e41c9", "warning")] : [])]);
  push("stage-2", "Alias 'staging' moves to v2", "Staging now points at the candidate. Production still points at v1, so users are unaffected.",
    { phase: "alias", aliases: { staging: 2, production: 1 }, moved: { alias: "staging", from: 1, to: 2 }, focusRow: 1 }, [ev("a3", "move alias staging: v1 → v2")]);
  push("promote", gate.pass ? "Gate passes: 'production' moves to v2" : "Gate blocks the promotion",
    gate.pass ? `v2 (${m2.accuracy.toFixed(4)}) beats v1 (${m1.accuracy.toFixed(4)}) by at least ${minGain}. The production alias moves and the endpoint reloads.`
      : `v2 (${m2.accuracy.toFixed(4)}) does not beat v1 (${m1.accuracy.toFixed(4)}) + ${minGain}. Production stays on v1.`,
    { phase: "promote", aliases: { staging: 2, production: gate.pass ? 2 : 1 }, moved: gate.pass ? { alias: "production", from: 1, to: 2 } : undefined, gate, focusRow: 1 },
    [ev("g", `gate: ${m2.accuracy.toFixed(4)} ≥ ${m1.accuracy.toFixed(4)} + ${minGain} → ${gate.pass ? "pass" : "fail"}`, gate.pass ? undefined : "warning",
      `${m2.accuracy.toFixed(4)} \\ge ${m1.accuracy.toFixed(4)} + ${minGain} \\Rightarrow \\text{${gate.pass ? "pass" : "fail"}}`)]);
  push("repro", repro.match ? "Reproducibility check: same hash" : "Reproducibility check FAILS",
    repro.match ? `Retraining from the recorded lineage gives ${repro.hash} again, identical to v2.`
      : `Retraining from the recorded lineage gives ${repro.hash}, not ${m2.hash}. Mismatch in: ${diffs.join(", ")}.`,
    { phase: "repro", repro, focusRow: 1 },
    [ev("rp", `retrain from lineage → ${repro.hash} vs registered ${m2.hash}`, repro.match ? undefined : "error"),
      ...diffs.map((d) => ev(`d-${d}`, `${d}: recorded ${String(l2[d])} → used ${String(d === "seed" ? rSeed : replay[d])}`, "error"))]);
  const prodV = gate.pass ? 2 : 1;
  const prodModel = prodV === 2 ? m2 : m1;
  push("audit", "Audit: what exactly is in production?", `Walk the lineage backwards: endpoint → production → v${prodV} → ${prodModel.runId} → data ${prodModel.lineage.dataHash} + code ${prodModel.lineage.commit}.`,
    { phase: "audit", focusRow: prodV === 2 ? 1 : 0 },
    [ev("au", `production = v${prodV} (${prodModel.hash}) ← ${prodModel.runId} ← data ${prodModel.lineage.dataRef}`, prodModel.lineage.pinned && prodModel.lineage.seed !== null && prodModel.lineage.codeHash === prodModel.lineage.commit ? undefined : "warning")]);
  return steps;
}
