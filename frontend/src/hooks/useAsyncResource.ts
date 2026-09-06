"use client";

/**
 * Fetch-with-retry for data whose absence must never look identical to
 * "still loading".
 *
 * A bare `T | null` state — the pattern both the analyze and item-detail pages
 * used for the explanation and attributions calls — cannot tell "the fetch
 * hasn't resolved yet" apart from "it failed and was caught into null". That
 * is exactly how a transient network failure on /explanation became a
 * permanent "Generating..." spinner sitting in front of an answer the backend
 * had already computed and cached: confirmed live against the deployed
 * service on 2026-09-06, where curl returned status="ready" with a full
 * narrative for an item whose page had been stuck on the loading skeleton for
 * over a minute. The failure was almost certainly caused by the backend being
 * hammered with redeploys and load-testing traffic at that exact moment, not a
 * bug in the fetch itself — but *some* transient failure reaching this state
 * was inevitable, and the old code had no way to show it or recover from it
 * short of a full page reload.
 */

import { useCallback, useEffect, useState } from "react";

export type AsyncState<T> =
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "done"; data: T };

export function useAsyncResource<T>(
  key: string | null,
  fetcher: (key: string) => Promise<T>,
): AsyncState<T> & { retry: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ phase: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!key) return;
    let cancelled = false;

    // The loading reset lives inside the async flow rather than as the first
    // statement in the effect body, so a key change cannot tear the render
    // between clearing state and the fetch resolving.
    const load = async () => {
      setState({ phase: "loading" });
      try {
        const data = await fetcher(key);
        if (!cancelled) setState({ phase: "done", data });
      } catch (err: unknown) {
        if (!cancelled) {
          setState({
            phase: "error",
            message: err instanceof Error ? err.message : "request failed",
          });
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
    // fetcher is a stable module-level export (getExplanation, getAttributions)
    // in every call site; omitted to avoid retriggering on an inline arrow.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, attempt]);

  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...state, retry };
}
