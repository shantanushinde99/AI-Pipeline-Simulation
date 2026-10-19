import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/** Renders a LaTeX string with KaTeX (HTML + MathML for screen readers). Inputs are app-authored, never user text. */
export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: block, throwOnError: false }), [children, block]);
  return <span className={block ? "block overflow-x-auto" : undefined} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** A display-mode formula on the same tinted background the old monospace formula boxes used. */
export function TexBlock({ children }: { children: string }) {
  return <div className="overflow-x-auto rounded-md bg-surface-2 px-2.5 py-2 text-[13px]"><Tex block>{children}</Tex></div>;
}
