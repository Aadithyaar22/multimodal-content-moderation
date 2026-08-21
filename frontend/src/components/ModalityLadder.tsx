"use client";

/**
 * The modality comparison — the one component this whole interface exists for.
 *
 * Three scores stacked against a shared threshold: vision alone, language
 * alone, and the two fused. When an item is emergent, both unimodal bars stop
 * short of the threshold and the fused bar clears it, and the reader can see in
 * one glance that the harm is a property of the combination rather than of
 * either signal.
 *
 * Rendered as a luminosity ladder rather than in two accent hues, per the
 * Sentinel Noir rule that meaning is carried by brightness and stroke weight.
 * Dim for vision, mid for language, pure white for the fused result.
 *
 * Everything here is readable with animation disabled: the bar widths, the
 * numerals, and the emergent callout are all static facts. Motion only sets
 * the order in which they arrive.
 */

import type { FusionSignal, ModalityScores } from "@/lib/types";

interface Props {
  scores: ModalityScores;
  signal: FusionSignal;
  task: "toxicity" | "misinformation";
  threshold?: number;
}

const RUNGS = [
  { key: "cv_only", label: "Vision only", tone: "var(--color-stream-vision)" },
  { key: "nlp_only", label: "Language only", tone: "var(--color-stream-language)" },
  { key: "fusion", label: "Fused", tone: "var(--color-stream-fusion)" },
] as const;

export function ModalityLadder({ scores, signal, task, threshold = 0.5 }: Props) {
  /**
   * Undefined, not zero, when an arm has no score.
   *
   * The backend omits a single-modality arm when its modality was not supplied,
   * because running it on a null input yields the head's bias rather than a
   * reading. Defaulting that to 0 would draw an empty bar and state "below
   * threshold" — presenting an absent measurement as a confident negative,
   * which is exactly the fabricated signal the API is careful not to emit.
   */
  const value = (k: (typeof RUNGS)[number]["key"]): number | undefined =>
    scores[k]?.[task];

  return (
    <section
      className="glass rounded-lg p-6"
      aria-label="Score by modality"
    >
      <header className="flex items-baseline justify-between border-b border-[var(--color-glass-border)] pb-3">
        <h2 className="label-tech-lg text-on-surface">Signal by modality</h2>
        <span className="label-tech text-outline">
          threshold {threshold.toFixed(2)}
        </span>
      </header>

      <div className="mt-6 space-y-5">
        {RUNGS.map((rung, i) => {
          const v = value(rung.key);
          const isFusion = rung.key === "fusion";
          const missing = v === undefined;
          const clears = !missing && v >= threshold;

          return (
            <div key={rung.key}>
              <div className="mb-2 flex items-baseline justify-between">
                <span
                  className={`label-tech ${isFusion ? "text-on-surface" : "text-outline"}`}
                >
                  {rung.label}
                </span>
                <span
                  className={`numeric font-display text-lg font-bold ${
                    missing
                      ? "text-outline-variant"
                      : isFusion
                        ? "text-on-surface"
                        : "text-on-surface-variant"
                  }`}
                >
                  {missing ? "—" : v.toFixed(2)}
                </span>
              </div>

              <div className="relative h-2 w-full overflow-visible rounded-full bg-surface-highest">
                {!missing && (
                  <div
                    className="animate-bar h-full rounded-full"
                    style={{
                      width: `${Math.min(100, v * 100)}%`,
                      background: rung.tone,
                      animationDelay: `${i * 140}ms`,
                      boxShadow:
                        isFusion && signal.is_emergent
                          ? "0 0 16px rgba(255,255,255,0.55)"
                          : undefined,
                    }}
                  />
                )}
                {/* Shared threshold marker — the reference every bar is read against */}
                <div
                  className="pointer-events-none absolute inset-y-[-4px] w-px bg-outline"
                  style={{ left: `${threshold * 100}%` }}
                  aria-hidden
                />
              </div>

              {/* Never colour alone: state is stated in words too. */}
              <p className="mt-1.5 text-xs text-outline">
                {missing
                  ? "modality not supplied"
                  : clears
                    ? "clears threshold"
                    : "below threshold"}
              </p>
            </div>
          );
        })}
      </div>

      {signal.is_emergent && (
        <div className="animate-rise mt-6 border-l-2 border-on-surface bg-[rgba(255,255,255,0.06)] p-4">
          <p className="label-tech text-on-surface">Emergent signal</p>
          <p className="mt-2 text-sm leading-relaxed text-on-surface-variant">
            Neither modality alone crosses the threshold. The fused score exceeds
            the better of them by{" "}
            <span className="numeric font-display font-bold text-on-surface">
              {signal.delta_over_best_unimodal.toFixed(2)}
            </span>
            . This is harm that exists only in the combination — the case a
            single-signal classifier misses.
          </p>
        </div>
      )}
    </section>
  );
}
