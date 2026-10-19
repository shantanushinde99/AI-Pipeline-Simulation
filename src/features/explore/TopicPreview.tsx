import { Bot, BrainCircuit, ClipboardCheck, Layers, Library, Server, type LucideIcon } from "lucide-react";
import type { CategoryId } from "../../types/topic";

const CATEGORY_ICON: Record<CategoryId, LucideIcon> = { ml: BrainCircuit, dl: Layers, rag: Library, agents: Bot, prod: Server, eval: ClipboardCheck };

/** Small thumbnails that hint at what each visualizer shows; other topics get their category icon. Decorative. */
export function TopicPreview({ id, category }: { id: string; category: CategoryId }) {
  const s = { stroke: "var(--muted)", strokeOpacity: 0.6 };
  const a = "var(--accent)";
  let body: React.ReactNode = null;
  if (id === "forward-propagation") {
    const L = [[20, 25], [20, 55]], M = [[70, 15], [70, 40], [70, 65]], R = [[120, 40]];
    body = (<>
      {L.flatMap(([x1, y1]) => M.map(([x2, y2]) => <line key={`${x1}${y1}${x2}${y2}`} x1={x1} y1={y1} x2={x2} y2={y2} {...s} />))}
      {M.map(([x1, y1]) => <line key={`${y1}`} x1={x1} y1={y1} x2={120} y2={40} stroke={a} strokeWidth={1.5} />)}
      {[...L, ...M, ...R].map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r={6} fill="var(--surface)" stroke={a} />)}
    </>);
  } else if (id === "overfitting") {
    body = (<>
      {[[15, 55], [30, 40], [45, 30], [60, 35], [75, 45], [90, 52], [105, 40], [120, 25]].map(([x, y]) => <circle key={x} cx={x} cy={y} r={2.5} fill={a} />)}
      <path d="M10 60 C 40 15, 70 60, 90 50 S 115 20, 130 22" fill="none" stroke={a} strokeWidth={1.5} />
      <path d="M10 58 L 130 28" fill="none" {...s} strokeDasharray="3 3" />
    </>);
  } else if (id === "embeddings") {
    body = (<>
      {[[30, 20], [40, 30], [25, 35], [100, 55], [110, 45], [95, 60], [80, 20]].map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r={3} fill="var(--muted)" />)}
      <line x1={70} y1={70} x2={40} y2={30} stroke={a} strokeDasharray="3 2" /><line x1={70} y1={70} x2={30} y2={20} stroke={a} strokeDasharray="3 2" />
      <circle cx={70} cy={70} r={0} /><path d="M66 66 l8 8 M74 66 l-8 8" stroke={a} strokeWidth={2} />
    </>);
  } else if (id === "rag-pipeline") {
    body = (<>
      {[10, 42, 74, 106].map((x) => <rect key={x} x={x} y={14} width={24} height={14} rx={3} fill="var(--surface)" stroke="var(--muted)" />)}
      {[10, 42, 74, 106].map((x) => <rect key={x} x={x} y={52} width={24} height={14} rx={3} fill="var(--surface)" stroke={a} />)}
      <path d="M34 21h8M66 21h8M98 21h8M34 59h8M66 59h8M98 59h8M118 28v24" {...s} />
    </>);
  } else if (id === "agent-loop") {
    body = (<>
      <circle cx={70} cy={40} r={24} fill="none" stroke={a} strokeDasharray="5 3" />
      {[[70, 16], [94, 40], [70, 64], [46, 40]].map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r={5} fill="var(--surface)" stroke={a} />)}
      <rect x={108} y={30} width={24} height={20} rx={3} fill="var(--surface)" stroke="var(--muted)" />
      <line x1={99} y1={40} x2={108} y2={40} {...s} />
    </>);
  }
  if (!body) {
    const Icon = CATEGORY_ICON[category];
    return <div className="flex h-20 items-center justify-center text-accent/70" aria-hidden><Icon size={36} strokeWidth={1.4} /></div>;
  }
  return <svg viewBox="0 0 140 80" className="h-20 w-full" aria-hidden>{body}</svg>;
}
