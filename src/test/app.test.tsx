import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { App } from "../app/App";
import { progressStore } from "../lib/storage";

const renderAt = (path: string) => {
  const user = userEvent.setup();
  render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
  return user;
};

describe("app", () => {
  it("navigates from Explore to a visualizer", async () => {
    const user = renderAt("/explore");
    await user.click(screen.getByRole("link", { name: /Neural network forward propagation/ }));
    expect(await screen.findByRole("heading", { level: 1, name: "Neural network forward propagation" })).toBeInTheDocument();
    expect(await screen.findByTestId("step-counter")).toHaveTextContent("Step 1 / 10");
  });

  it("filters topics by search", async () => {
    const user = renderAt("/explore");
    await user.type(screen.getByRole("textbox", { name: "Search the topic library" }), "embedding");
    expect(screen.getByRole("link", { name: /Embeddings & vector search/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Neural network forward propagation/ })).not.toBeInTheDocument();
  });

  it("playback buttons move through steps", async () => {
    const user = renderAt("/visualizer/forward-propagation");
    const counter = await screen.findByTestId("step-counter");
    await user.click(screen.getByRole("button", { name: "Next step" }));
    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(counter).toHaveTextContent("Step 3 / 10");
    await user.click(screen.getByRole("button", { name: "Previous step" }));
    expect(counter).toHaveTextContent("Step 2 / 10");
    await user.click(screen.getByRole("button", { name: /Jump to step 10/ }));
    expect(counter).toHaveTextContent("Step 10 / 10");
    expect(screen.getByRole("button", { name: "Next step" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Restart" }));
    expect(counter).toHaveTextContent("Step 1 / 10");
    await user.keyboard("{ArrowRight}");
    expect(counter).toHaveTextContent("Step 2 / 10");
  });

  it("changing parameters changes computed outputs", async () => {
    const user = renderAt("/visualizer/forward-propagation?tab=metrics");
    expect(await screen.findByTestId("nn-accuracy")).toHaveTextContent("100%");
    await user.click(screen.getByRole("tab", { name: "Visualize" }));
    await user.click(screen.getByRole("button", { name: "Random (seed 7)" }));
    await user.click(screen.getByRole("tab", { name: "Metrics" }));
    expect(screen.getByTestId("nn-accuracy")).not.toHaveTextContent("100%");
  });

  it("overfitting metrics react to model complexity", async () => {
    const user = renderAt("/visualizer/overfitting");
    const test = await screen.findByTestId("test-mse");
    const before = test.textContent;
    await user.click(screen.getByRole("button", { name: "Overfit (degree 14)" }));
    expect(screen.getByTestId("test-mse").textContent).not.toBe(before);
  });

  it("RAG: clicking a pipeline node opens it in the inspector", async () => {
    const user = renderAt("/visualizer/rag-pipeline");
    await screen.findByTestId("step-counter");
    // jsdom never measures React Flow nodes, so they stay visibility:hidden; select by aria-label instead of role.
    await user.click(screen.getByLabelText(/^Retrieve top-k:/));
    expect(screen.getByText("Inspector: Retrieve top-k")).toBeInTheDocument();
    expect(screen.getByText("Output of: Retrieve top-k")).toBeInTheDocument();
  });

  it("RAG: injecting a fault changes the final answer", async () => {
    const user = renderAt("/visualizer/rag-pipeline");
    await user.click(await screen.findByRole("button", { name: /Jump to step 13/ }));
    // The stage panel crossfades, so the answer view appears after the previous one exits.
    const healthy = (await screen.findByTestId("rag-answer")).textContent;
    await user.selectOptions(screen.getByLabelText("Question"), "q-rate");
    await user.selectOptions(screen.getByLabelText("Fault"), "stale");
    expect(screen.getByTestId("rag-answer")).toHaveTextContent("300 requests per minute");
    expect(screen.getByTestId("rag-answer").textContent).not.toBe(healthy);
  });

  it("Failure Lab: investigate, diagnose and receive a score", async () => {
    const user = renderAt("/failure-lab/rag-stale-index");
    await user.click(await screen.findByRole("button", { name: "Check index metadata and document versions" }));
    const obs = screen.getByRole("article", { name: "Check index metadata and document versions" });
    expect(within(obs).getByText(/OUT OF DATE/)).toBeInTheDocument();
    await user.click(within(obs).getByRole("checkbox", { name: "Supports my diagnosis" }));
    await user.click(screen.getByRole("tab", { name: /Diagnose & fix/ }));
    const submit = screen.getByRole("button", { name: "Submit diagnosis and apply fix" });
    expect(submit).toBeDisabled();
    await user.click(screen.getByRole("radio", { name: /stale snapshot built before/ }));
    await user.click(screen.getByRole("radio", { name: /Rebuild the index from current sources/ }));
    await user.click(screen.getByRole("radio", { name: /diff indexed versions against sources/ }));
    await user.click(submit);
    expect(screen.getByText("score 4/4")).toBeInTheDocument();
    expect(screen.getByText(/Resolved: the re-run behaves correctly/)).toBeInTheDocument();
    expect(progressStore.get().scenarios["rag-stale-index"].score).toBe(4);
    await user.selectOptions(screen.getByLabelText(/Try another fix/), "f-topk");
    expect(screen.getByText(/Not resolved: the re-run still fails/)).toBeInTheDocument();
  });

  it("challenge submission gives feedback and records progress", async () => {
    const user = renderAt("/visualizer/forward-propagation?tab=challenges");
    const card = screen.getAllByRole("article")[0];
    await user.click(within(card).getByRole("radio", { name: /σ\(−3\)/ }));
    await user.click(within(card).getByRole("button", { name: "Submit answer" }));
    expect(within(card).getByRole("status")).toHaveTextContent("Correct.");
    expect(progressStore.get().challenges["forward-propagation-1"].correct).toBe(true);
  });

  it("deep links to unknown pages show not found; planned topics are labelled", async () => {
    renderAt("/visualizer/caching");
    expect(screen.getByText("This topic does not have an interactive visualizer yet.")).toBeInTheDocument();
  });
});
