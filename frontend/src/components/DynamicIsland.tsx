"use client";

/**
 * Dynamic Island navigation.
 *
 * A floating pill that stays compact while a moderator works and expands when
 * it has something to say — the system warming up, a decision landing, a
 * request failing. The morph is a width/height transition on a single rounded
 * container, so the island never unmounts and its content cross-fades inside a
 * shape that is continuously animating rather than swapping between two states.
 *
 * It floats over the liquid-metal field rather than sitting on an opaque bar,
 * which is what keeps the background continuous behind it. The previous fixed
 * header cut the field in half.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, USE_MOCK } from "@/lib/api";

const LINKS = [
  { href: "/queue", label: "Queue" },
  { href: "/analyze", label: "Analyze" },
  { href: "/dashboard", label: "Dashboard" },
];

type IslandState =
  | { kind: "idle" }
  | { kind: "warming" }
  | { kind: "message"; text: string }
  // Loading finished and failed (backend reports status="error"), or the
  // health call itself couldn't be reached at all. Both are distinct from a
  // cold start: they will not resolve by waiting, so the copy and the retry
  // cadence must not pretend they will.
  | { kind: "error"; text: string };

const UNREACHABLE_AFTER = 5; // consecutive network failures before giving up

export function DynamicIsland() {
  const pathname = usePathname();
  const [state, setState] = useState<IslandState>({ kind: "warming" });
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  /** Expand the island with a transient message, then settle back. */
  const announce = useCallback((text: string, ms = 2600) => {
    clearTimeout(timer.current);
    setState({ kind: "message", text });
    timer.current = setTimeout(() => setState({ kind: "idle" }), ms);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let poll: ReturnType<typeof setTimeout>;
    let consecutiveFailures = 0;

    // A cold backend takes 15-20s to wake. The island holds an explicit warming
    // state for that whole window so the wait never looks like a hang — but
    // that is only correct for models_loaded:false. A permanent load failure
    // (status:"error") or an unreachable host looks identical to a slow cold
    // start on the same field, so both are checked for explicitly rather than
    // folded into the same "still warming, keep polling every 2s" branch,
    // which would otherwise retry forever and never tell the moderator the
    // wait is never going to end.
    const check = async () => {
      try {
        const h = await getHealth();
        if (cancelled) return;
        consecutiveFailures = 0;

        if (h.status === "error") {
          setState({ kind: "error", text: h.error ?? "Model failed to load" });
          poll = setTimeout(check, 15_000); // still worth checking after a redeploy
        } else if (h.models_loaded) {
          setState({ kind: "idle" });
        } else {
          setState({ kind: "warming" });
          poll = setTimeout(check, 2000);
        }
      } catch {
        if (cancelled) return;
        consecutiveFailures += 1;
        if (consecutiveFailures >= UNREACHABLE_AFTER) {
          setState({ kind: "error", text: "Backend unreachable" });
          poll = setTimeout(check, 15_000);
        } else {
          setState({ kind: "warming" });
          poll = setTimeout(check, 2000);
        }
      }
    };
    check();

    return () => {
      cancelled = true;
      clearTimeout(poll);
      clearTimeout(timer.current);
    };
  }, []);

  // Any part of the app can ask the island to speak.
  useEffect(() => {
    const onAnnounce = (e: Event) => {
      const detail = (e as CustomEvent<string>).detail;
      if (detail) announce(detail);
    };
    window.addEventListener("island:announce", onAnnounce);
    return () => window.removeEventListener("island:announce", onAnnounce);
  }, [announce]);

  const expanded = state.kind !== "idle";

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-50 flex justify-center px-4">
      <nav
        aria-label="Primary"
        data-expanded={expanded}
        className="island pointer-events-auto"
      >
        <Link
          href="/"
          className="shrink-0 font-display text-sm font-black tracking-tight text-white"
        >
          SENTINEL
        </Link>

        <span className="island-divider" aria-hidden />

        <div className="flex shrink-0 items-center gap-0.5">
          {LINKS.map((l) => {
            const active = pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`label-tech rounded-full px-3 py-1.5 transition-colors ${
                  active
                    ? "bg-white text-black"
                    : "text-white/55 hover:text-white"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>

        {/* The expanding region. Width is animated on the island itself; this
            only fades its contents so text never appears mid-morph. */}
        <div
          className={`island-slot ${expanded ? "island-slot-open" : ""}`}
          role="status"
          aria-live="polite"
        >
          <span className="island-divider" aria-hidden />
          {state.kind === "warming" && (
            <span className="flex items-center gap-2 whitespace-nowrap">
              <span className="relative flex h-1.5 w-1.5 shrink-0">
                <span className="animate-ring absolute inline-flex h-full w-full rounded-full bg-white/70" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-white" />
              </span>
              <span className="label-tech text-white/70">Waking models</span>
            </span>
          )}
          {state.kind === "error" && (
            <span
              className="flex items-center gap-2 whitespace-nowrap"
              title={state.text}
            >
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--color-harm)]"
                aria-hidden
              />
              <span className="label-tech text-[var(--color-harm)]">
                {state.text}
              </span>
            </span>
          )}
          {state.kind === "message" && (
            <span className="label-tech whitespace-nowrap text-white">
              {state.text}
            </span>
          )}
        </div>

        {USE_MOCK && (
          <>
            <span className="island-divider" aria-hidden />
            <span className="label-tech shrink-0 text-white/40">Mock</span>
          </>
        )}
      </nav>
    </div>
  );
}

/** Ask the island to expand with a transient message from anywhere. */
export function announceToIsland(text: string) {
  window.dispatchEvent(new CustomEvent("island:announce", { detail: text }));
}
