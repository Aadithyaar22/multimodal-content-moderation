"use client";

/**
 * Route-aware chrome.
 *
 * The landing page owns its own full-strength liquid-metal field and shows no
 * navigation — it is a threshold, not a screen you work in. Every other route
 * gets the island and the same field dialled down to ambient, so the surface is
 * continuous across the whole product rather than restarting at each page.
 */

import { usePathname } from "next/navigation";
import { DynamicIsland } from "@/components/DynamicIsland";
import { LiquidMetal } from "@/components/LiquidMetal";

export function Chrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLanding = pathname === "/";

  if (isLanding) return <>{children}</>;

  return (
    <>
      <LiquidMetal intensity="ambient" />
      <DynamicIsland />
      {/* Top padding clears the floating island, which overlays rather than
          occupying layout space. */}
      <main className="mx-auto w-full max-w-[1440px] flex-1 px-6 pt-28 pb-10 md:px-12">
        {children}
      </main>
    </>
  );
}
