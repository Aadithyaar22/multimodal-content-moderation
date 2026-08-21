/**
 * Landing page.
 *
 * Three elements only: the mark, one line, one way in.
 *
 * The type is outlined rather than difference-blended. Difference blending is
 * the obvious choice for a black-and-white surface, but it drives contrast to
 * zero over mid-grey, which meant routing the molten veins around the text to
 * keep it readable. An outlined letterform is legible over any value, so the
 * white is free to flow straight through the centre of the frame instead.
 */

import Link from "next/link";
import { LiquidMetal } from "@/components/LiquidMetal";

export default function LandingPage() {
  return (
    <>
      <LiquidMetal intensity="full" />

      <div className="relative flex min-h-[100svh] flex-col items-center justify-center px-6 text-center">
        <h1 className="display-vanguard outlined-display animate-rise text-[clamp(3.5rem,15vw,13rem)]">
          Vanguard
        </h1>

        <p
          className="subtitle-athelas outlined animate-rise mt-8 max-w-3xl text-[clamp(1.15rem,2.6vw,2rem)] leading-snug"
          style={{ animationDelay: "160ms" }}
        >
          Absolute clarity in the face of complex data.
        </p>

        <Link
          href="/queue"
          className="animate-rise group mt-14 inline-flex items-center gap-3 rounded-full bg-white px-8 py-4 text-black shadow-[0_8px_40px_rgba(0,0,0,0.55)] transition-transform duration-300 hover:scale-[1.03] active:scale-100"
          style={{ animationDelay: "320ms" }}
        >
          <span className="label-tech-lg">Start review</span>
          <span
            aria-hidden
            className="transition-transform duration-300 group-hover:translate-x-1"
          >
            →
          </span>
        </Link>
      </div>
    </>
  );
}
