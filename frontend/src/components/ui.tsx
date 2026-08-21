/**
 * Shared primitives for the Sentinel Noir system.
 *
 * Kept in one file so the glass treatment, label casing and severity mapping
 * are defined exactly once; the original Stitch screens repeated them per page
 * and had already drifted between screens.
 */

import type { ReactNode } from "react";
import type { VerdictLabel } from "@/lib/types";

export function GlassPanel({
  title,
  action,
  children,
  className = "",
}: {
  title?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`glass rounded-lg ${className}`}>
      {title && (
        <header className="flex items-center justify-between border-b border-[var(--color-glass-border)] px-6 py-4">
          <h2 className="label-tech-lg text-on-surface">{title}</h2>
          {action}
        </header>
      )}
      <div className="p-6">{children}</div>
    </section>
  );
}

const SEVERITY: Record<VerdictLabel, { label: string; className: string }> = {
  // Red is the only hue in the system and appears nowhere but here.
  harmful: {
    label: "Harmful",
    className: "border-[var(--color-harm)] text-[var(--color-harm)]",
  },
  review: { label: "Needs review", className: "border-outline text-on-surface" },
  benign: { label: "Benign", className: "border-outline-variant text-outline" },
};

export function VerdictBadge({ label }: { label: VerdictLabel }) {
  const s = SEVERITY[label];
  return (
    <span className={`label-tech rounded border px-2 py-1 ${s.className}`}>
      {s.label}
    </span>
  );
}

export function EmergentBadge() {
  return (
    <span
      className="label-tech rounded border border-on-surface bg-[rgba(255,255,255,0.1)] px-2 py-1 text-on-surface"
      title="Neither modality alone crosses threshold; the signal appears only in combination"
    >
      Emergent
    </span>
  );
}

/**
 * Confidence must never look like certainty. The numeral is always shown
 * alongside the bar so 0.51 and 0.97 cannot read the same at a glance.
 */
export function ConfidenceBar({
  value,
  label = "Confidence",
}: {
  value: number;
  label?: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label-tech text-outline">{label}</span>
        <span className="numeric font-display text-sm font-bold text-on-surface">
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-surface-highest">
        <div
          className="animate-bar h-full rounded-full bg-on-surface"
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
    </div>
  );
}

export function Stat({
  label,
  value,
  suffix,
}: {
  label: string;
  value: string | number;
  suffix?: string;
}) {
  return (
    <div className="glass rounded-lg p-6">
      <p className="label-tech text-outline">{label}</p>
      <p className="numeric mt-3 font-display text-4xl font-extrabold text-on-surface">
        {value}
        {suffix && (
          <span className="ml-1 text-lg font-normal text-outline">{suffix}</span>
        )}
      </p>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded bg-surface-high ${className}`}
      aria-hidden
    />
  );
}

export function relativeAge(seconds: number): string {
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
