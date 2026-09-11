"use client";

/**
 * "Is this true?" — a live, web-search-grounded check of the caption's own
 * claim, deliberately separate from the misinformation score shown elsewhere
 * on the page. That score is a learned pattern (recycled image plus urgency
 * framing looks like past misinformation); this checks the specific claim
 * against current search results, and the two are meant to disagree
 * sometimes.
 *
 * Opt-in, not auto-fetched: a real search round-trip costs more than every
 * other call on this page, so nothing here runs until someone clicks the
 * button — reusing useAsyncResource by simply not giving it a key until then.
 */

import { useState } from "react";
import { getFactCheck } from "@/lib/api";
import { useAsyncResource } from "@/hooks/useAsyncResource";
import type { FactCheckVerdict } from "@/lib/types";
import { GlassPanel, Skeleton } from "@/components/ui";

const VERDICT_LABEL: Record<FactCheckVerdict, string> = {
  supported: "Supported",
  contradicted: "Contradicted",
  unclear: "Unclear",
  no_factual_claim: "No claim to check",
};

// Red is the system's only accent hue (see ui.tsx's SEVERITY comment) and is
// reserved for the one genuinely concerning outcome here; everything else
// stays neutral, the same treatment "benign" and "review" get on the verdict
// badge.
const VERDICT_CLASS: Record<FactCheckVerdict, string> = {
  contradicted: "border-[var(--color-harm)] text-[var(--color-harm)]",
  supported: "border-outline-variant text-outline",
  unclear: "border-outline-variant text-outline",
  no_factual_claim: "border-outline-variant text-outline",
};

export function FactCheckPanel({ itemId }: { itemId: string }) {
  const [triggered, setTriggered] = useState(false);
  const state = useAsyncResource(triggered ? itemId : null, getFactCheck);

  if (!triggered) {
    return (
      <GlassPanel title="Is this true?" className="animate-rise">
        <p className="text-sm text-on-surface-variant">
          Check the caption&apos;s own claim against live search results —
          independent of the misinformation score above, and often slower:
          a real search takes longer than explaining a score the model
          already computed.
        </p>
        <button
          onClick={() => setTriggered(true)}
          className="label-tech mt-4 rounded border border-outline-variant px-4 py-2 text-on-surface hover:border-outline"
        >
          Check this claim
        </button>
      </GlassPanel>
    );
  }

  return (
    <GlassPanel title="Is this true?" className="animate-rise">
      {state.phase === "loading" ? (
        <div className="space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-11/12" />
          <Skeleton className="h-4 w-4/5" />
          <p className="label-tech pt-2 text-outline">Searching…</p>
        </div>
      ) : state.phase === "error" ? (
        <div className="space-y-3">
          <p className="text-sm text-on-surface-variant">
            Could not check this claim ({state.message}).
          </p>
          <button
            onClick={state.retry}
            className="label-tech rounded border border-outline-variant px-3 py-2 text-on-surface hover:border-outline"
          >
            Retry
          </button>
        </div>
      ) : state.data.status !== "ready" || !state.data.verdict ? (
        <p className="text-sm text-on-surface-variant">
          Claim checking is not configured on this deployment.
        </p>
      ) : (
        <>
          <span
            className={`label-tech inline-block rounded-full border px-3 py-1 ${VERDICT_CLASS[state.data.verdict]}`}
          >
            {VERDICT_LABEL[state.data.verdict]}
          </span>
          {state.data.summary && (
            <p className="mt-3 text-on-surface">{state.data.summary}</p>
          )}
          {state.data.sources.length > 0 && (
            <ul className="mt-4 space-y-1">
              {state.data.sources.map((s) => (
                <li key={s.url}>
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="label-tech text-outline underline hover:text-on-surface"
                  >
                    {s.domain ?? s.title}
                  </a>
                </li>
              ))}
            </ul>
          )}
          <p className="label-tech mt-4 text-outline">
            {state.data.model} · {state.data.latency_ms}ms · independent of
            the misinformation score above
          </p>
        </>
      )}
    </GlassPanel>
  );
}
