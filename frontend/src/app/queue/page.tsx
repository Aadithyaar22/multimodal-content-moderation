"use client";

/**
 * Review queue — where moderators actually work.
 *
 * Ranked by priority, never chronological: the ordering is the product.
 *
 * Motion here is deliberately restrained. A moderator scans this list for hours
 * through distressing content, and a flourish costing 300ms is paid on every
 * item. The spectacle belongs on the detail and analyze views; the queue's calm
 * is what makes those land.
 */

import Link from "next/link";
import { useEffect, useState } from "react";
import { getQueue } from "@/lib/api";
import type { QueueFilters, QueueItem } from "@/lib/types";
import {
  ConfidenceBar,
  EmergentBadge,
  GlassPanel,
  Skeleton,
  VerdictBadge,
  relativeAge,
} from "@/components/ui";

export default function QueuePage() {
  const [items, setItems] = useState<QueueItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<QueueFilters>({});

  // Clearing on the interaction rather than in the effect keeps the skeleton
  // appearing the moment a filter is pressed, with no intermediate stale frame.
  const updateFilters = (next: (f: QueueFilters) => QueueFilters) => {
    setItems(null);
    setError(null);
    setFilters(next);
  };

  useEffect(() => {
    let cancelled = false;

    // State transitions live inside the async flow rather than running
    // synchronously in the effect body, so a filter change cannot tear the
    // render between clearing the list and the fetch resolving.
    const load = async () => {
      try {
        const r = await getQueue(filters);
        if (!cancelled) setItems(r.items);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [filters]);

  const emergentCount = items?.filter((i) => i.is_emergent).length ?? 0;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="display-vanguard text-[clamp(2.5rem,7vw,5rem)] text-on-surface">
          Review queue
        </h1>
        <p className="subtitle-athelas mt-4 max-w-2xl text-xl text-on-surface-variant">
          Ranked by priority, not arrival. Every item is a recommendation for a
          human decision — nothing here has been actioned automatically.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <Toggle
          active={!!filters.emergent_only}
          onClick={() =>
            updateFilters((f) => ({ ...f, emergent_only: !f.emergent_only }))
          }
        >
          Emergent only{emergentCount ? ` (${emergentCount})` : ""}
        </Toggle>
        <Toggle
          active={filters.head === "toxicity"}
          onClick={() =>
            updateFilters((f) => ({
              ...f,
              head: f.head === "toxicity" ? undefined : "toxicity",
            }))
          }
        >
          Harassment
        </Toggle>
        <Toggle
          active={filters.head === "misinformation"}
          onClick={() =>
            updateFilters((f) => ({
              ...f,
              head: f.head === "misinformation" ? undefined : "misinformation",
            }))
          }
        >
          Misinformation
        </Toggle>
      </div>

      {error && (
        <GlassPanel title="Could not load queue">
          <p className="text-sm text-on-surface-variant">{error}</p>
        </GlassPanel>
      )}

      {items === null && !error && (
        <div className="space-y-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      )}

      {items?.length === 0 && (
        <GlassPanel title="Queue empty">
          <p className="text-sm text-on-surface-variant">
            No items match the current filters.
          </p>
        </GlassPanel>
      )}

      <ul className="space-y-4">
        {items?.map((item, i) => (
          <li
            key={item.item_id}
            className="animate-rise"
            style={{ animationDelay: `${Math.min(i, 8) * 40}ms` }}
          >
            <QueueRow item={item} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function QueueRow({ item }: { item: QueueItem }) {
  return (
    <Link
      href={`/items/${item.item_id}`}
      className="glass glass-hover block rounded-lg p-5"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="numeric font-display text-sm font-bold text-outline">
            {item.item_id.replace("itm_", "").toUpperCase()}
          </span>
          <VerdictBadge label={item.verdict.label} />
          {item.is_emergent && <EmergentBadge />}
        </div>
        <span className="label-tech text-outline">
          {relativeAge(item.age_seconds)}
        </span>
      </div>

      <p className="mt-4 line-clamp-2 text-on-surface">{item.text_preview}</p>

      <div className="mt-5 grid gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
        <ConfidenceBar value={item.verdict.confidence} />
        <div className="text-right">
          <span className="label-tech text-outline">Priority</span>
          <p className="numeric font-display text-2xl font-extrabold">
            {item.verdict.priority_score.toFixed(2)}
          </p>
        </div>
      </div>
    </Link>
  );
}

function Toggle({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`label-tech rounded border px-3 py-2 transition-colors ${
        active
          ? "border-on-surface bg-on-surface text-black"
          : "border-outline-variant text-outline hover:border-outline hover:text-on-surface"
      }`}
    >
      {children}
    </button>
  );
}
