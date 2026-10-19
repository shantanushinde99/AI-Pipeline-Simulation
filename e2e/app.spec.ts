import { expect, test as base, type Page } from "@playwright/test";

// Every test fails if the page logs an unexpected console error or throws.
const test = base.extend<{ page: Page }>({
  page: async ({ page }, use) => {
    const errors: string[] = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(String(e)));
    await use(page);
    expect(errors, "browser console errors").toEqual([]);
  },
});

test("Explore → open a visualizer", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/explore$/);
  await page.getByRole("link", { name: /Complete RAG pipeline/ }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Complete RAG pipeline" })).toBeVisible();
});

test("changing an input changes the computed output", async ({ page }) => {
  await page.goto("/visualizer/forward-propagation");
  const calc = page.getByTestId("nn-output");
  const before = await calc.textContent();
  await page.getByLabel("x1 (x1)").fill("0");
  await page.getByLabel("x2 (x2)").fill("0");
  await expect(calc).not.toHaveText(before!);
});

test("play, pause, step forward, back and restart", async ({ page }) => {
  await page.goto("/visualizer/agent-loop");
  const counter = page.getByTestId("step-counter");
  await expect(counter).toHaveText("Step 1 / 15");
  await page.getByRole("button", { name: "Next step" }).click();
  await page.getByRole("button", { name: "Next step" }).click();
  await expect(counter).toHaveText("Step 3 / 15");
  await page.getByRole("button", { name: "Previous step" }).click();
  await expect(counter).toHaveText("Step 2 / 15");
  await page.getByLabel("Speed").selectOption("4");
  await page.getByRole("button", { name: "Play" }).click();
  await expect(counter).not.toHaveText("Step 2 / 15");
  await page.getByRole("button", { name: "Pause" }).click();
  const paused = await counter.textContent();
  await page.waitForTimeout(800);
  await expect(counter).toHaveText(paused!);
  await page.getByRole("button", { name: "Restart" }).click();
  await expect(counter).toHaveText("Step 1 / 15");
});

test("RAG: inspect a node and change retrieval parameters", async ({ page }) => {
  await page.goto("/visualizer/rag-pipeline");
  await page.getByRole("button", { name: /^Retrieve top-k:/ }).click();
  await expect(page.getByText("Inspector: Retrieve top-k")).toBeVisible();
  const inspector = page.getByRole("complementary", { name: "Inspector" });
  await expect(inspector.getByText(/^#\d+ /)).toHaveCount(4);
  await page.getByLabel("top-k retrieved").fill("2");
  await page.getByRole("button", { name: /^Retrieve top-k:/ }).click();
  await expect(inspector.getByText(/^#\d+ /)).toHaveCount(2);
});

test("RAG: injected fault propagates to the answer", async ({ page }) => {
  await page.goto("/visualizer/rag-pipeline");
  await page.getByLabel("Question").selectOption("q-rate");
  await page.getByRole("button", { name: /Jump to step 13/ }).click();
  await expect(page.getByTestId("rag-answer")).toContainText("600 requests per minute");
  await page.getByLabel("Fault").selectOption("stale");
  await expect(page.getByTestId("rag-answer")).toContainText("300 requests per minute");
  await expect(page.getByText("✗ contains “600 requests per minute”")).toBeVisible();
  await page.getByRole("button", { name: /^Vector index:/ }).click();
  await expect(page.getByText(/^v2 \(2026-04-14\)$/)).toBeVisible();
});

test("Failure Lab: diagnose a scenario and get feedback", async ({ page }) => {
  await page.goto("/failure-lab");
  await page.getByRole("link", { name: /Everything is green, but the answers are wrong/ }).click();
  await page.getByRole("button", { name: "Compare query and index embedding models" }).click();
  const obs = page.getByRole("article", { name: "Compare query and index embedding models" });
  await expect(obs.getByText("query encoder: embed-v2")).toBeVisible();
  await obs.getByRole("checkbox", { name: "Supports my diagnosis" }).check();
  await page.getByRole("tab", { name: /Trace/ }).click();
  await page.getByRole("button", { name: /vector search top-4/ }).click();
  await expect(page.getByText(/top-1 similarity 0\.\d+/)).toBeVisible();
  await page.getByRole("tab", { name: /Diagnose & fix/ }).click();
  await page.getByRole("radio", { name: /query encoder was upgraded/ }).check();
  await page.getByRole("radio", { name: /Re-embed the whole corpus/ }).check();
  await page.getByRole("radio", { name: /fail startup if query and index model ids differ/ }).check();
  await page.getByLabel("Explain your diagnosis").fill("Query and index embeddings come from different models.");
  await page.getByRole("button", { name: "Submit diagnosis and apply fix" }).click();
  await expect(page.getByText("score 4/4")).toBeVisible();
  await expect(page.getByText("Resolved: the re-run behaves correctly")).toBeVisible();
});

test("refreshing a deep link loads the same visualizer and tab", async ({ page }) => {
  await page.goto("/visualizer/embeddings?tab=concept");
  await expect(page.getByRole("heading", { level: 1, name: "Embeddings & vector search" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: "Embeddings & vector search" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Concept" })).toHaveAttribute("aria-selected", "true");
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("navigate with the menu and topic drawers; no horizontal overflow", async ({ page }) => {
    await page.goto("/explore");
    await page.getByRole("button", { name: "Open menu" }).click();
    await page.getByRole("dialog", { name: "Menu" }).getByRole("link", { name: "Failure Lab" }).click();
    await expect(page.getByRole("heading", { name: "Failure Lab" })).toBeVisible();
    await page.goto("/visualizer/overfitting");
    await page.getByRole("button", { name: "Topics" }).click();
    const drawer = page.getByRole("dialog", { name: "Topics" });
    await drawer.getByText(/^AI Agents/).click();
    await drawer.getByRole("link", { name: /AI agent execution loop/ }).click();
    await expect(page.getByRole("heading", { level: 1, name: "AI agent execution loop" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Next step" })).toBeVisible();
    for (const path of ["/explore", "/visualizer/rag-pipeline", "/failure-lab/rag-chunk-split", "/practice"]) {
      await page.goto(path);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `horizontal overflow on ${path}`).toBeLessThanOrEqual(1);
    }
  });
});
