import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  localStorage.clear();
});

// React Flow measures nodes with ResizeObserver, which jsdom lacks.
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= RO as unknown as typeof ResizeObserver;
window.matchMedia ??= ((q: string) => ({ matches: false, media: q, onchange: null, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false })) as typeof window.matchMedia;

// React Flow needs element sizes and DOMMatrixReadOnly to place nodes (per its testing guide).
Object.defineProperties(HTMLElement.prototype, {
  offsetHeight: { configurable: true, get() { return parseFloat((this as HTMLElement).style.height) || 100; } },
  offsetWidth: { configurable: true, get() { return parseFloat((this as HTMLElement).style.width) || 100; } },
});
class DOMMatrixRO {
  m22: number;
  constructor(t?: string) {
    const s = t?.match(/scale\(([0-9.]+)\)/)?.[1];
    this.m22 = s !== undefined ? Number(s) : 1;
  }
}
(globalThis as unknown as { DOMMatrixReadOnly: unknown }).DOMMatrixReadOnly ??= DOMMatrixRO;
