"use client";

/**
 * The landing page's one call to action, aware of sign-in state.
 *
 * Split out of page.tsx (a server component) because reading auth state
 * needs a client component — the rest of the hero has no reason to pay for
 * that. Signed out, this is the front door to signing in rather than a
 * silent trip to a queue that would just bounce off DecisionBar's own
 * sign-in prompt later; signed in, it's exactly the original "Start Review".
 */

import Link from "next/link";
import { useAuth } from "@/lib/auth";

export function HeroCta() {
  const { user } = useAuth();
  const signedIn = !!user;

  return (
    <Link
      href={signedIn ? "/queue" : "/login"}
      className="group relative mt-4 overflow-hidden rounded-full bg-white px-12 py-4 text-black transition-transform duration-300 hover:scale-105 focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black focus:outline-none active:scale-95"
    >
      <span className="label-tech-lg relative z-10 flex items-center gap-2 font-bold">
        {signedIn ? "Start Review" : "Log in / Sign up"}
        <span aria-hidden className="text-[18px] leading-none">
          →
        </span>
      </span>
      <span
        aria-hidden
        className="absolute inset-0 -translate-x-full -skew-x-12 bg-black/10 group-hover:animate-[shimmer_1.5s_infinite]"
      />
    </Link>
  );
}
