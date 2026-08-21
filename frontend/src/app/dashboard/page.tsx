"use client";

/**
 * Dashboard — aggregate metrics.
 *
 * Follows the Stitch bento layout, but the tiles report what this system
 * actually measures. The export's metrics ("threat class distribution",
 * "ensemble agreement", "median TTD") were placeholders for a generic forensics
 * product; showing invented numbers on a dashboard is worse than showing none,
 * because a dashboard is exactly where a reader stops checking.
 *
 * The emergent-case rate is given a tile of its own. It is the share of items
 * where neither modality alone crossed threshold, which is the one figure that
 * distinguishes this system from a pair of ordinary classifiers, and it belongs
 * where it can be read at a glance.
 */

import { useEffect, useState } from "react";
import { getStats } from "@/lib/api";
import type { Stats } from "@/lib/types";
import { GlassPanel, Skeleton } from "@/components/ui";

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await getStats();
        if (!cancelled) setStats(s);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <GlassPanel title="Could not load metrics">
        <p className="text-sm text-on-surface-variant">{error}</p>
      </GlassPanel>
    );
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="display-vanguard text-[clamp(2.5rem,7vw,5rem)] text-on-surface">
          System dashboard
        </h1>
        <p className="subtitle-athelas mt-4 max-w-2xl text-xl text-on-surface-variant">
          Aggregate moderation throughput and model behaviour.
        </p>
      </header>

      {!stats ? (
        <div className="grid gap-6 lg:grid-cols-3">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-56 w-full" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <BigStat
                label="Pending review"
                value={stats.queue.pending.toLocaleString()}
                caption="items awaiting a human decision"
              />
            </div>

            <GlassPanel title="Median time to decision" className="animate-rise">
              <p className="numeric font-display text-5xl font-extrabold text-on-surface">
                {formatDuration(stats.queue.median_time_to_decision_s)}
              </p>
              <p className="label-tech mt-4 text-outline">
                {stats.queue.resolved_24h.toLocaleString()} resolved in 24h
              </p>
            </GlassPanel>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <GlassPanel title="Emergent cases" className="animate-rise">
              <Donut value={stats.model.emergent_case_rate} />
              <p className="mt-4 text-sm leading-relaxed text-outline">
                Share of flagged items where neither modality alone crossed
                threshold. These are the cases a single-signal classifier misses.
              </p>
            </GlassPanel>

            <GlassPanel title="Moderator agreement" className="animate-rise">
              <Donut value={stats.model.agreement_rate} />
              <p className="mt-4 text-sm leading-relaxed text-outline">
                How often a moderator decision matched the model recommendation.
              </p>
            </GlassPanel>

            <GlassPanel title="Explanation rated useful" className="animate-rise">
              <Donut value={stats.model.explanation_useful_rate} />
              <p className="mt-4 text-sm leading-relaxed text-outline">
                Optional feedback, so this is a rate over the subset who
                answered, not over every decision.
              </p>
            </GlassPanel>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <GlassPanel title="Harassment head" className="animate-rise">
              <Distribution data={stats.distribution.toxicity} />
            </GlassPanel>
            <GlassPanel title="Misinformation head" className="animate-rise">
              <Distribution data={stats.distribution.misinformation} />
            </GlassPanel>
          </div>
        </>
      )}
    </div>
  );
}

function BigStat({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <section className="glass animate-rise flex h-full flex-col justify-center rounded-lg p-10 text-center">
      <p className="label-tech-lg text-outline">{label}</p>
      <p className="numeric mt-4 font-display text-7xl font-black text-on-surface md:text-8xl">
        {value}
      </p>
      <p className="label-tech mt-4 text-outline">{caption}</p>
    </section>
  );
}

/**
 * Ring gauge. The numeral is the primary reading and the arc is support — a
 * ring alone is hard to read to better than about ten points, and these
 * differences matter at finer resolution than that.
 */
function Donut({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const r = 46;
  const circumference = 2 * Math.PI * r;

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <svg width="128" height="128" viewBox="0 0 128 128" aria-hidden>
          <circle
            cx="64"
            cy="64"
            r={r}
            fill="none"
            stroke="var(--color-surface-highest)"
            strokeWidth="8"
          />
          <circle
            cx="64"
            cy="64"
            r={r}
            fill="none"
            stroke="var(--color-on-surface)"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - value)}
            transform="rotate(-90 64 64)"
            style={{ transition: "stroke-dashoffset 900ms var(--ease-entrance)" }}
          />
        </svg>
        <span className="numeric absolute inset-0 flex items-center justify-center font-display text-2xl font-extrabold text-on-surface">
          {pct}%
        </span>
      </div>
    </div>
  );
}

function Distribution({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-4">
      {entries.map(([label, value]) => (
        <div key={label}>
          <div className="flex items-baseline justify-between">
            <span className="label-tech text-outline">{label}</span>
            <span className="numeric font-display text-sm font-bold text-on-surface">
              {(value * 100).toFixed(0)}%
            </span>
          </div>
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-surface-highest">
            <div
              className="animate-bar h-full rounded-full bg-on-surface"
              style={{ width: `${value * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s ? `${m}m ${s}s` : `${m}m`;
}
