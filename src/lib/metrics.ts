export type Confusion = { tp: number; fp: number; tn: number; fn: number };

export function confusion(labels: readonly number[], preds: readonly number[]): Confusion {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  labels.forEach((y, i) => {
    const p = preds[i];
    if (y === 1 && p === 1) c.tp++;
    else if (y === 0 && p === 1) c.fp++;
    else if (y === 0 && p === 0) c.tn++;
    else c.fn++;
  });
  return c;
}

const safe = (n: number, d: number) => (d === 0 ? 0 : n / d);

export function classificationMetrics(labels: readonly number[], preds: readonly number[]) {
  const c = confusion(labels, preds);
  const precision = safe(c.tp, c.tp + c.fp);
  const recall = safe(c.tp, c.tp + c.fn);
  return {
    ...c,
    accuracy: safe(c.tp + c.tn, labels.length),
    precision,
    recall,
    f1: safe(2 * precision * recall, precision + recall),
  };
}

export const mse = (y: readonly number[], yhat: readonly number[]) =>
  y.length ? y.reduce((s, v, i) => s + (v - yhat[i]) ** 2, 0) / y.length : Number.NaN;

export function r2(y: readonly number[], yhat: readonly number[]) {
  const m = y.reduce((s, v) => s + v, 0) / y.length;
  const ssTot = y.reduce((s, v) => s + (v - m) ** 2, 0);
  const ssRes = y.reduce((s, v, i) => s + (v - yhat[i]) ** 2, 0);
  return ssTot === 0 ? Number.NaN : 1 - ssRes / ssTot;
}
