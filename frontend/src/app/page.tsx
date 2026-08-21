/**
 * Landing page — ported from the Stitch vanguard export, structure intact.
 *
 * Type sits in difference blend mode over the shader, which is what the export
 * intended: the letterforms invert as the liquid surface flows beneath them, so
 * they belong to the surface rather than floating above it. That works here in
 * a way it did not over the earlier CSS field, because this shader keeps its
 * values near the extremes — deep black ground with bright specular plumes —
 * rather than spending most of the frame in the mid-greys where difference
 * blending collapses to no contrast.
 */

import Link from "next/link";
import { LiquidGlassShader } from "@/components/LiquidGlassShader";

export default function LandingPage() {
  return (
    <>
      <LiquidGlassShader />

      <main className="relative z-10 flex h-[100svh] flex-grow flex-col items-center justify-center px-12">
        <div className="mx-auto flex w-full max-w-[1440px] flex-col items-center text-center text-white mix-blend-difference">
          <h1 className="mb-6 font-display text-[clamp(3rem,13vw,160px)] font-black uppercase italic leading-[0.88] tracking-tighter text-white drop-shadow-2xl">
            Vanguard
          </h1>

          <p className="mx-auto mb-20 max-w-3xl font-editorial text-[clamp(1.1rem,3vw,32px)] italic leading-[1.25] tracking-wide text-white/90">
            Absolute clarity in the face of complex data.
          </p>

          <Link
            href="/queue"
            className="group relative mt-4 overflow-hidden rounded-full bg-white px-12 py-4 text-black transition-transform duration-300 hover:scale-105 focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black focus:outline-none active:scale-95"
          >
            <span className="label-tech-lg relative z-10 flex items-center gap-2 font-bold">
              Start Review
              <span aria-hidden className="text-[18px] leading-none">
                →
              </span>
            </span>
            <span
              aria-hidden
              className="absolute inset-0 -translate-x-full -skew-x-12 bg-black/10 group-hover:animate-[shimmer_1.5s_infinite]"
            />
          </Link>
        </div>

        <div className="absolute bottom-12 flex w-full flex-col items-center gap-3 text-center text-white opacity-60 mix-blend-difference">
          <p className="label-tech tracking-widest">Sentinel AI Forensics</p>
          <div className="h-px w-12 bg-white/40" />
        </div>
      </main>
    </>
  );
}
