import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import { MotionConfig } from "motion/react";
import { Analytics } from "@vercel/analytics/react";
import "@xyflow/react/dist/style.css";
import "./index.css";
import { App } from "./app/App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* Honour the OS "reduce motion" setting for every Motion animation. */}
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <Analytics />
    </MotionConfig>
  </StrictMode>,
);
