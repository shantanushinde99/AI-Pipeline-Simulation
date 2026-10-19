import { describe, expect, it } from "vitest";
import { churnData, dropFeature, evaluateModel, fraudData, reorder, trainLogistic } from "./ml-model";

describe("ML scenario models", () => {
  const train = churnData(1, 800), test = churnData(2, 400), prod = churnData(3, 400, true);

  it("leaky feature inflates offline accuracy and collapses in production", () => {
    const leaky = trainLogistic(train);
    const offline = evaluateModel(leaky, test.rows).accuracy;
    const online = evaluateModel(leaky, prod.rows);
    expect(offline).toBeGreaterThan(0.95);
    expect(online.recall).toBeLessThan(0.2);
    const clean = trainLogistic(dropFeature(train, "cancel_ticket_logged"));
    const cleanOff = evaluateModel(clean, dropFeature(test, "cancel_ticket_logged").rows).accuracy;
    const cleanOn = evaluateModel(clean, dropFeature(prod, "cancel_ticket_logged").rows).accuracy;
    expect(cleanOff).toBeLessThan(0.9);
    expect(Math.abs(cleanOff - cleanOn)).toBeLessThan(0.08);
  });

  it("class imbalance: high accuracy, low recall; class weights fix recall", () => {
    const tr = fraudData(4, 3000), te = fraudData(5, 2000);
    const plain = evaluateModel(trainLogistic(tr), te.rows);
    expect(plain.accuracy).toBeGreaterThan(0.95);
    expect(plain.recall).toBeLessThan(0.5);
    const weighted = evaluateModel(trainLogistic(tr, { positiveWeight: 30 }), te.rows);
    expect(weighted.recall).toBeGreaterThan(0.7);
  });

  it("feature order mismatch at serving degrades predictions without errors", () => {
    const ds = dropFeature(train, "cancel_ticket_logged"), te = dropFeature(test, "cancel_ticket_logged");
    const m = trainLogistic(ds);
    const ok = evaluateModel(m, te.rows).accuracy;
    const swapped = evaluateModel(m, reorder(te.rows, [1, 0, 2])).accuracy;
    expect(ok - swapped).toBeGreaterThan(0.1);
  });
});
