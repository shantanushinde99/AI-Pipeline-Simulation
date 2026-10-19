import { useEffect } from "react";
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "motion/react";

/**
 * A number that counts to `value` instead of jumping. Remount it (via `key`) with `from` to replay a count;
 * otherwise it glides from its previous value whenever `value` changes.
 */
export function AnimatedNumber({ value, from, digits = 2, delay = 0, duration = 0.6, svg = false }: {
  value: number; from?: number; digits?: number; delay?: number; duration?: number; svg?: boolean;
}) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(reduce ? value : (from ?? value));
  const text = useTransform(mv, (v) => v.toFixed(digits));
  useEffect(() => {
    const c = animate(mv, value, reduce ? { duration: 0 } : { duration, delay, ease: "easeOut" });
    return () => c.stop();
  }, [mv, value, delay, duration, reduce]);
  return svg ? <motion.tspan>{text}</motion.tspan> : <motion.span>{text}</motion.span>;
}
