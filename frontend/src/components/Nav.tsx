"use client";

/**
 * Top navigation.
 *
 * Sticky rather than fixed-and-overlapping: the original Stitch screens used a
 * floating bar that sat on top of the page content in both exported
 * screenshots, hiding the first rows of the queue and the report header.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getHealth, USE_MOCK } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Queue" },
  { href: "/analyze", label: "Analyze" },
  { href: "/dashboard", label: "Dashboard" },
];

export function Nav() {
  const pathname = usePathname();
  const [warm, setWarm] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    // The backend sleeps on free tier and takes 30-50s to wake. Polling here
    // means a cold start shows as an explicit warming state instead of the
    // first real request appearing to hang.
    const poll = async () => {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setWarm(h.models_loaded);
        if (!h.models_loaded) timer = setTimeout(poll, 2000);
      } catch {
        if (cancelled) return;
        setWarm(false);
        timer = setTimeout(poll, 2000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--color-glass-border)] bg-[rgba(0,0,0,0.72)] backdrop-blur-2xl">
      <nav className="mx-auto flex max-w-[1440px] items-center gap-8 px-6 py-4 md:px-12">
        <Link href="/" className="font-display text-xl font-black tracking-tight">
          SENTINEL
        </Link>

        <div className="flex gap-1">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`label-tech rounded px-3 py-2 transition-colors ${
                  active
                    ? "bg-[rgba(255,255,255,0.1)] text-on-surface"
                    : "text-outline hover:text-on-surface"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-3">
          {USE_MOCK && (
            <span className="label-tech rounded border border-outline-variant px-2 py-1 text-outline">
              Mock data
            </span>
          )}
          <StatusDot warm={warm} />
        </div>
      </nav>
    </header>
  );
}

function StatusDot({ warm }: { warm: boolean | null }) {
  const text =
    warm === null ? "Connecting" : warm ? "Models ready" : "Warming up";
  return (
    <span className="flex items-center gap-2" title={text}>
      <span className="relative flex h-2 w-2">
        {warm === false && (
          <span className="animate-ring absolute inline-flex h-full w-full rounded-full bg-outline" />
        )}
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${
            warm ? "bg-on-surface" : "bg-outline"
          }`}
        />
      </span>
      {/* Status is never conveyed by the dot alone. */}
      <span className="label-tech hidden text-outline sm:inline">{text}</span>
    </span>
  );
}
