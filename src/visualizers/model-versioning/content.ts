import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Register model versions with lineage, move staging/production aliases and test reproducibility.", minutes: 12, hasMetrics: false };

const t = "model-versioning";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "A model registry stores every trained model as an immutable, numbered version together with its lineage: the exact data snapshot (by content hash), the code commit, the hyperparameters, the random seed and the evaluation metrics. Aliases such as 'staging' and 'production' are movable pointers to versions; serving loads whatever 'production' points to." },
    { heading: "Why it matters", body: "Promotion and rollback become a single, logged pointer move instead of copying files around. When a prediction is questioned, you can walk from the endpoint to the alias, the version, the run, and finally the data and code that produced it. Without that chain, debugging, audits and rollbacks rely on memory." },
    { heading: "Reproducibility needs every input pinned", body: "A version is reproducible only if everything that influenced it was recorded exactly: data by hash (not 'latest'), code by a clean commit (not a dirty working tree), and the seed. Any gap means a retrain from the recorded lineage produces a different model, and you cannot tell whether a change in behaviour came from the data, the code or chance." },
    { heading: "Assumptions in this visualizer", body: "Training is simulated: the 'weights hash' is an FNV-1a hash of the lineage string and accuracy comes from a hand-made response surface plus a seeded jitter of at most ±0.005. Real training on GPUs can also be non-deterministic even with every input pinned (non-deterministic kernels), which this simulation does not model. No real registry or storage is called." },
  ],
  code: [
    {
      title: "MLflow: log lineage, register, move aliases",
      source: `import mlflow
from mlflow import MlflowClient

with mlflow.start_run() as run:
    mlflow.log_params({"lr": 0.01, "depth": 6, "seed": 7})
    mlflow.set_tags({"data_snapshot": "sha256:9f1c…", "git_commit": "b7e41c9"})
    model = train(data_snapshot="snap-2026-10-01", lr=0.01, depth=6, seed=7)
    mlflow.log_metric("val_accuracy", evaluate(model))
    mlflow.sklearn.log_model(model, "model", registered_model_name="churn")

client = MlflowClient()
client.set_registered_model_alias("churn", "staging", version=2)
# after the promotion gate passes:
client.set_registered_model_alias("churn", "production", version=2)
# serving resolves the alias at load time
model = mlflow.pyfunc.load_model("models:/churn@production")`,
    },
    {
      title: "Pin data and refuse dirty trees before training",
      source: `import hashlib, subprocess

def snapshot_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

if subprocess.run(["git", "diff", "--quiet"]).returncode != 0:
    raise SystemExit("uncommitted changes: commit before training so lineage is exact")

lineage = {
    "data": snapshot_hash("features/2026-10-01.parquet"),   # never 'latest'
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "seed": 7,
}`,
    },
  ],
  failureModes: [
    { title: "Unpinned data ('latest')", symptom: "Retraining an old version gives different metrics and a different model hash.", detect: "Reproducibility check in CI: retrain from lineage and compare hashes or metrics within tolerance. Pick the 'latest' gap here.", fix: "Record data by content hash or immutable snapshot id; resolve 'latest' to a hash at run start and log that." },
    { title: "Unlogged random seed", symptom: "Two runs with 'identical' settings disagree; nobody can say which one shipped.", detect: "Lineage record has no seed field; repeated runs scatter.", fix: "Set and log the seed for every source of randomness (split, init, shuffling)." },
    { title: "Training from a dirty working tree", symptom: "The registered commit does not contain the code that produced the model.", detect: "Compare `git diff` at run start; flag runs with uncommitted changes.", fix: "Refuse to train with uncommitted changes, or log the diff as an artifact." },
    { title: "Overwriting a version in place", symptom: "The model behind 'v3' changes but nothing in the registry says so; rollback restores the wrong weights.", detect: "Weights hash for a version changes over time.", fix: "Treat versions as immutable; any change is a new version, and only aliases move." },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "diagnosis", question: "Select the 'Data read from a moving latest pointer' gap. At the reproducibility step, which input differs between the original run and the retrain?", options: ["The commit", "The seed", "The data hash, because 'latest' now resolves to newer rows", "The learning rate"], answer: 2, explanation: "The config stored only the name 'latest'. By retrain time it points at the 2026-10-15 rows (16,500 rows), so the data hash differs and so does the model hash." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "Raise the promotion gate's required gain to +8.0 pts with the default v2 settings. What happens at the promotion step?", options: ["Production moves to v2 anyway", "Production stays on v1 while staging points at v2", "Both aliases are removed", "v2 is deleted from the registry"], answer: 1, explanation: "v2 beats v1 by roughly 4 points, short of 8. The gate blocks the move, so production keeps pointing at v1. Staging still points at v2 for further testing." },
    { id: `${t}-3`, topicId: t, skill: "prediction", question: "What does moving the 'production' alias from v2 back to v1 after an incident (a rollback) require?", options: ["Retraining v1", "Re-uploading v1's weights", "Only changing the pointer; v1's version is immutable and still stored", "Deleting v2"], answer: 2, explanation: "Versions are never overwritten. The endpoint resolves the alias, so pointing it back at v1 is instant and fully logged." },
    { id: `${t}-4`, topicId: t, skill: "verification", question: "Which check proves a registered version is reproducible?", options: ["The version has a higher accuracy than the last one", "Retraining from the recorded lineage yields the same weights hash", "The model file exists in storage", "The production alias points at it"], answer: 1, explanation: "Reproducibility means the recorded inputs fully determine the output. Comparing hashes (or metrics within a tolerance on non-deterministic hardware) is the direct test." },
    { id: `${t}-5`, topicId: t, skill: "fix", question: "Run 2 logged its seed as missing. What is the most direct fix for future runs?", options: ["Train twice and keep the better model", "Set an explicit seed and log it in the lineage record", "Use a larger dataset", "Pin the commit"], answer: 1, explanation: "With the 'Random seed not recorded' gap, the only difference in the retrain is the seed. Setting and logging it closes the gap." },
  ],
};
