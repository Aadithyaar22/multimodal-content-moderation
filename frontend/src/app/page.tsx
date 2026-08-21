/**
 * Landing page.
 *
 * The liquid-metal field runs full strength here. The display title sits in
 * difference blend mode so it inverts as the molten boundary drifts beneath it
 * — one continuous surface rather than a dark page with a light panel on it.
 *
 * Everything smaller uses `.on-liquid` instead. Difference blending drives
 * contrast to zero over mid-grey, which a 13rem letterform survives and body
 * copy does not, so the effect is spent where it reads and dropped where it
 * would cost legibility.
 *
 * The page also states plainly what the system is for. A moderation tool that
 * opens by implying it decides things sets the wrong expectation before anyone
 * has seen a verdict.
 */

import Link from "next/link";
import { LiquidMetal } from "@/components/LiquidMetal";

export default function LandingPage() {
  return (
    <>
      <LiquidMetal intensity="full" />

      <div className="relative flex min-h-[100svh] flex-col items-center justify-center px-6 text-center">
        <p className="label-tech on-liquid animate-rise mb-8 opacity-80">
          Multimodal content moderation
        </p>

        <h1
          className="display-vanguard blend-invert animate-rise text-[clamp(3.5rem,15vw,13rem)]"
          style={{ animationDelay: "80ms" }}
        >
          Vanguard
        </h1>

        <p
          className="subtitle-athelas on-liquid animate-rise mt-8 max-w-3xl text-[clamp(1.15rem,2.6vw,2rem)] leading-snug"
          style={{ animationDelay: "200ms" }}
        >
          Absolute clarity in the face of complex data.
        </p>

        <p
          className="on-liquid animate-rise mt-6 max-w-xl text-sm leading-relaxed opacity-85"
          style={{ animationDelay: "300ms" }}
        >
          Harm is often a property of the relationship between an image and its
          caption, not of either one alone. Vanguard surfaces those cases and
          explains them — then a person decides.
        </p>

        <Link
          href="/queue"
          className="animate-rise group mt-14 inline-flex items-center gap-3 rounded-full bg-white px-8 py-4 text-black shadow-[0_8px_40px_rgba(0,0,0,0.5)] transition-transform duration-300 hover:scale-[1.03] active:scale-100"
          style={{ animationDelay: "420ms" }}
        >
          <span className="label-tech-lg">Start review</span>
          <span
            aria-hidden
            className="transition-transform duration-300 group-hover:translate-x-1"
          >
            →
          </span>
        </Link>

        <div
          className="animate-rise absolute bottom-10 flex flex-col items-center gap-3"
          style={{ animationDelay: "560ms" }}
        >
          <span className="h-10 w-px bg-white/40" aria-hidden />
          <span className="label-tech on-liquid opacity-70">
            Decision support · not automated removal
          </span>
        </div>
      </div>
    </>
  );
}
