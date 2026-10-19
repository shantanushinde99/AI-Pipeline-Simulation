import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router";
import { Analytics } from "@vercel/analytics/react";
import { AppShell } from "../components/layout/AppShell";
import { ExplorePage } from "../features/explore/ExplorePage";
import { VisualizerPage } from "../features/visualizer/VisualizerPage";
import { NotFound } from "../features/NotFound";
import { FailureLabPage } from "../features/failure-lab/FailureLabPage";
// Lazy: keeps React Flow out of the entry bundle.
const ScenarioPage = lazy(() => import("../features/failure-lab/ScenarioPage").then((m) => ({ default: m.ScenarioPage })));
import { LearningPathsPage } from "../features/learning-paths/LearningPathsPage";
import { InterviewPracticePage } from "../features/interview-practice/InterviewPracticePage";
import { ProgressPage } from "../features/progress/ProgressPage";

export function App() {
  return (
    <>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/explore" replace />} />
          <Route path="explore" element={<ExplorePage />} />
          <Route path="visualizer" element={<Navigate to="/explore" replace />} />
          <Route path="visualizer/:topicId" element={<VisualizerPage />} />
          <Route path="failure-lab" element={<FailureLabPage />} />
          <Route path="failure-lab/:scenarioId" element={<Suspense fallback={<p className="text-sm text-muted">Loading scenario…</p>}><ScenarioPage /></Suspense>} />
          <Route path="paths" element={<LearningPathsPage />} />
          <Route path="practice" element={<InterviewPracticePage />} />
          <Route path="progress" element={<ProgressPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
      <Analytics />
    </>
  );
}
