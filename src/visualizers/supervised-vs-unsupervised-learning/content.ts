import type { VisualizerContent, VisualizerMeta } from "../types";

export const meta: VisualizerMeta = { summary: "Classify the same points with labels, then cluster them without, and see when the groups disagree.", minutes: 10, hasMetrics: false };

const t = "supervised-vs-unsupervised-learning";

export const content: VisualizerContent = {
  concept: [
    { heading: "What it is", body: "Supervised learning trains on examples that come with the answer (a label or target) and learns to predict that answer for new inputs: classification and regression. Unsupervised learning gets inputs only and looks for structure in them: clusters, low-dimensional directions, anomalies. Same data, different question." },
    { heading: "Why the difference matters", body: "A supervised model is graded against the labels, so it optimises exactly what you asked for (and inherits every mistake in the labels). An unsupervised method optimises its own internal objective (for k-means, the total squared distance to the nearest centroid). The groups it finds are real structure in the features, but nothing forces them to be the groups you care about. The 'Clusters ≠ classes' preset shows this: the points form a top and a bottom blob, but the label is left vs. right." },
    { heading: "When to use which", body: "Use supervised learning when you have (or can afford) labels for the outcome you want to predict. Use unsupervised learning to explore unlabeled data, segment customers, compress features or flag outliers, and treat the result as a hypothesis to check, not ground truth. Many real systems mix both: cluster or embed first, label a sample, then train a classifier." },
    { heading: "k-means in one paragraph", body: "Pick k. Place k centroids (here: on k random points). Repeat: assign every point to its nearest centroid, then move each centroid to the mean of its points. Each round can only lower the inertia, so it always stops, but possibly in a poor local minimum that depends on the starting points. That is why libraries run several restarts (n_init) and use smarter starts (k-means++)." },
    { heading: "Assumptions in this visualizer", body: "The 60 points are synthetic Gaussian blobs (seeded). The supervised model is a nearest-centroid classifier, chosen because its training step (average each class) is easy to watch; accuracy is measured on the training points themselves, not a held-out set. k-means is Lloyd's algorithm with a single random (Forgy) start, no k-means++ and no restarts, capped at 12 iterations. Purity is computed after the fact using the hidden labels; k-means never sees them." },
  ],
  code: [
    {
      title: "scikit-learn: the same data, both ways",
      source: `from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

# Supervised: uses y
clf = NearestCentroid().fit(X, y)
print("accuracy", accuracy_score(y, clf.predict(X)))

# Unsupervised: never sees y
km = KMeans(n_clusters=3, init="random", n_init=1, random_state=2).fit(X)
print("inertia", km.inertia_)
print("agreement with labels (ARI)", adjusted_rand_score(y, km.labels_))`,
    },
    {
      title: "kmeans.py: Lloyd's algorithm by hand",
      source: `import numpy as np

def kmeans(X, k, seed=0, max_iter=12):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    assign = None
    for _ in range(max_iter):
        d = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        new = d.argmin(1)                          # assign step
        if assign is not None and (new == assign).all():
            break                                  # converged
        assign = new
        centers = np.array([X[assign == j].mean(0) if (assign == j).any() else centers[j]
                            for j in range(k)])    # update step
    return centers, assign`,
    },
  ],
  failureModes: [
    { title: "Treating clusters as classes", symptom: "A dashboard labels clusters 'churners' and 'loyal', but outcomes do not differ between them.", detect: "Compare clusters against a labelled sample (purity, adjusted Rand index) before naming them.", fix: "Validate clusters against the outcome you care about, or collect labels and train a supervised model." },
    { title: "Wrong k", symptom: "Real groups are split in two or merged together (try k = 5 on three blobs).", detect: "Plot inertia vs. k (elbow) or use silhouette scores; inspect cluster sizes.", fix: "Choose k from the use case and diagnostics, not a default." },
    { title: "Bad initialisation", symptom: "Two runs on the same data give different clusters; one centroid sits between two blobs.", detect: "Re-run with different seeds and compare inertia; the stuck run has higher J.", fix: "Use k-means++ and several restarts (n_init), keep the lowest inertia." },
    { title: "Unscaled features dominate distance", symptom: "Clusters only differ along the feature with the largest numeric range.", detect: "Check per-feature variance; clusters line up with one axis.", fix: "Standardise features before distance-based methods." },
    { title: "Leaky labels in supervised training", symptom: "The classifier looks perfect offline and fails live.", detect: "A feature that is only known after the outcome is among the most predictive.", fix: "Audit features for post-outcome information.", scenarioId: "ml-leakage" },
  ],
  challenges: [
    { id: `${t}-1`, topicId: t, skill: "prediction", question: "Choose the 'Clusters ≠ classes' preset (k = 2). What does k-means find?", options: ["The left and right classes exactly", "A top and a bottom group, each containing both classes", "Two empty clusters", "It refuses to run without labels"], answer: 1, explanation: "The points form two horizontal blobs, so distance favours a top/bottom split. The labels encode left vs. right, which k-means never sees. Purity drops to about 50–70%." },
    { id: `${t}-2`, topicId: t, skill: "parameter", question: "With three well-separated blobs, you set k = 5. What must happen?", options: ["Two centroids are deleted automatically", "Some blobs are split across several clusters", "Purity falls to 0%", "k-means picks k = 3 instead"], answer: 1, explanation: "k-means always produces exactly k clusters. With more centroids than natural groups, some groups are divided. Purity can stay high because each piece is still one class." },
    { id: `${t}-3`, topicId: t, skill: "metric", question: "In each k-means iteration, what happens to the inertia J (total squared distance to the assigned centroid)?", options: ["It can go up or down", "It never increases", "It always halves", "It equals the number of switched points"], answer: 1, explanation: "Assigning to the nearest centroid cannot increase any distance, and moving a centroid to its points' mean minimises their squared distances. So J is non-increasing, which is why k-means always stops." },
    { id: `${t}-4`, topicId: t, skill: "diagnosis", question: "Two runs with different init seeds give inertia 61 and 140 on the same data and k. Which statement is right?", options: ["The data changed between runs", "The 140 run is stuck in a worse local minimum; keep the 61 run", "Higher inertia means better clusters", "Inertia cannot depend on initialisation"], answer: 1, explanation: "k-means only finds a local minimum that depends on the start. Libraries run several starts and keep the lowest inertia." },
    { id: `${t}-5`, topicId: t, skill: "verification", question: "Which extra information does the supervised classifier use that k-means does not?", options: ["The x₁ coordinate", "The number of points", "Each point's class label", "The random seed"], answer: 2, explanation: "Both see the same coordinates. Only the supervised side gets the label column, and that is the whole difference." },
  ],
};
