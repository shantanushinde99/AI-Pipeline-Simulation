import { Bot, BrainCircuit, ClipboardCheck, Layers, Library, Server, type LucideIcon } from "lucide-react";
import type { CategoryId } from "../../types/topic";
import { THUMBNAILS } from "./topicThumbnails";

const CATEGORY_ICON: Record<CategoryId, LucideIcon> = { ml: BrainCircuit, dl: Layers, rag: Library, agents: Bot, prod: Server, eval: ClipboardCheck };

/** Small thumbnail hinting at what each visualizer shows; planned topics get their category icon. Decorative. */
export function TopicPreview({ id, category }: { id: string; category: CategoryId }) {
  const body = THUMBNAILS[id];
  if (!body) {
    const Icon = CATEGORY_ICON[category];
    return <div className="flex h-20 items-center justify-center text-accent/70" aria-hidden><Icon size={36} strokeWidth={1.4} /></div>;
  }
  return <svg viewBox="0 0 140 80" className="h-20 w-full" aria-hidden>{body()}</svg>;
}
