import { memo, useMemo } from "react";
import { Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import type { LucideIcon } from "lucide-react";
import { AlertTriangle, CheckCircle2, Circle, CircleDashed, CircleDot, XCircle } from "lucide-react";
import { cx } from "../ui";

export type PipeStatus = "pending" | "active" | "ok" | "warning" | "error" | "skipped" | "untraced";
export type PipeNode = { id: string; label: string; sub?: string; Icon: LucideIcon; x: number; y: number; status: PipeStatus };
export type PipeEdge = { source: string; target: string; sourceHandle?: "r" | "b"; targetHandle?: "l" | "t"; active?: boolean };
export type PipeLabel = { id: string; text: string; x: number; y: number };

type Data = { label: string; sub?: string; Icon: LucideIcon; status: PipeStatus; selected: boolean; onSelect: (id: string) => void };
type LabelData = { text: string };

const STATUS: Record<PipeStatus, { Icon: LucideIcon; text: string; cls: string }> = {
  pending: { Icon: Circle, text: "pending", cls: "border-line text-muted" },
  active: { Icon: CircleDot, text: "running", cls: "border-accent text-accent ring-2 ring-accent/30" },
  ok: { Icon: CheckCircle2, text: "done", cls: "border-ok/50 text-ok" },
  warning: { Icon: AlertTriangle, text: "warning", cls: "border-warn text-warn" },
  error: { Icon: XCircle, text: "error", cls: "border-err text-err" },
  skipped: { Icon: CircleDashed, text: "skipped", cls: "border-dashed border-line text-muted opacity-70" },
  untraced: { Icon: CircleDashed, text: "not traced", cls: "border-dashed border-line text-muted" },
};

const hidden = { opacity: 0, width: 1, height: 1, minWidth: 0, minHeight: 0, border: 0 };

const PipelineNodeView = memo(function PipelineNodeView({ id, data }: NodeProps<Node<Data>>) {
  const s = STATUS[data.status];
  return (
    <>
      <Handle type="target" position={Position.Left} id="l" style={hidden} />
      <Handle type="target" position={Position.Top} id="t" style={hidden} />
      <button
        onClick={() => data.onSelect(id)}
        style={{ pointerEvents: "all" }} // React Flow sets pointer-events:none on non-selectable nodes
        aria-pressed={data.selected}
        aria-label={`${data.label}: ${s.text}${data.sub ? `, ${data.sub}` : ""}`}
        className={cx(
          "nodrag flex w-[150px] items-start gap-2 rounded-lg border bg-surface px-2.5 py-2 text-left shadow-sm transition-all duration-500 hover:-translate-y-0.5 hover:shadow-md",
          s.cls,
          data.selected && "outline-2 outline-offset-2 outline-accent",
        )}
      >
        <data.Icon size={16} className="mt-0.5 shrink-0" aria-hidden />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[12px] font-semibold text-fg">{data.label}</span>
          <span className="flex items-center gap-1 text-[10px]"><s.Icon size={10} aria-hidden />{s.text}</span>
          {data.sub && <span className="block truncate font-mono text-[10px] text-muted">{data.sub}</span>}
        </span>
      </button>
      <Handle type="source" position={Position.Right} id="r" style={hidden} />
      <Handle type="source" position={Position.Bottom} id="b" style={hidden} />
    </>
  );
});

const RowLabel = ({ data }: NodeProps<Node<LabelData>>) => <div className="text-[10px] font-semibold uppercase tracking-wide text-muted">{data.text}</div>;

const nodeTypes = { pipe: PipelineNodeView, label: RowLabel };

/** Read-only node graph: nodes are buttons (keyboard accessible); pan is allowed, editing is not. */
export function PipelineCanvas({ nodes, edges, labels = [], selected, onSelect, height = 340, ariaLabel }: {
  nodes: PipeNode[]; edges: PipeEdge[]; labels?: PipeLabel[]; selected: string | null; onSelect: (id: string) => void; height?: number; ariaLabel: string;
}) {
  const rfNodes = useMemo<Node[]>(
    () => [
      ...nodes.map((n) => ({ id: n.id, type: "pipe", position: { x: n.x, y: n.y }, data: { label: n.label, sub: n.sub, Icon: n.Icon, status: n.status, selected: selected === n.id, onSelect } satisfies Data, selectable: false })),
      ...labels.map((l) => ({ id: l.id, type: "label", position: { x: l.x, y: l.y }, data: { text: l.text } satisfies LabelData, selectable: false, focusable: false })),
    ],
    [nodes, labels, selected, onSelect],
  );
  const rfEdges = useMemo<Edge[]>(
    () => edges.map((e) => ({
      id: `${e.source}-${e.target}`, source: e.source, target: e.target, sourceHandle: e.sourceHandle ?? "r", targetHandle: e.targetHandle ?? "l",
      type: "smoothstep", animated: !!e.active, focusable: false,
      markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: e.active ? "var(--accent)" : "var(--muted)" },
      style: { stroke: e.active ? "var(--accent)" : "var(--muted)", strokeWidth: e.active ? 2 : 1.25, opacity: e.active ? 1 : 0.6 },
    })),
    [edges],
  );
  return (
    // Below ~600px the graph would shrink to unreadable size, so it scrolls sideways instead.
    <div className="overflow-x-auto">
    <div style={{ height }} className="w-full min-w-[600px]" role="group" aria-label={ariaLabel}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.08 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
        minZoom={0.3}
      />
    </div>
    </div>
  );
}
