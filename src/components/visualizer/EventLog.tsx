import type { SimulationEvent } from "../../types/simulation";
import { Badge } from "../ui";
import { Tex } from "../Tex";

export function EventLog({ events, empty = "No events at this step." }: { events: SimulationEvent[]; empty?: string }) {
  if (!events.length) return <p className="text-xs text-muted">{empty}</p>;
  return (
    <ul className="space-y-1.5">
      {events.map((e) => (
        <li key={e.id} className="flex items-start gap-2 text-xs">
          <Badge tone={e.severity === "error" ? "err" : e.severity === "warning" ? "warn" : "neutral"} className="mt-px font-mono">
            {e.severity ?? "info"}
          </Badge>
          <span className="min-w-0 overflow-x-auto leading-relaxed">{e.tex ? <Tex>{e.tex}</Tex> : e.message}</span>
        </li>
      ))}
    </ul>
  );
}
