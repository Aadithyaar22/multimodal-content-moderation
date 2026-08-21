/**
 * Token-level attribution display.
 *
 * SHAP scores are *signed*: positive pushes toward harmful, negative toward
 * benign. A single-hue intensity ramp would render a strongly-exonerating token
 * identically to a strongly-incriminating one, silently inverting the meaning
 * of half the data — so sign is carried by two separate channels here.
 *
 * Sentinel Noir forbids hue as a carrier of meaning, so the diverging scale is
 * built from luminosity plus a border side: tokens pushing toward harmful are
 * bright with a left accent, tokens pushing toward benign are recessed with a
 * right accent. Every token also exposes its numeric score to assistive tech,
 * so nothing depends on the visual encoding at all.
 */

interface Props {
  tokens: Array<{ token: string; score: number }>;
  maxAbs?: number;
}

export function ShapTokens({ tokens, maxAbs }: Props) {
  const scale = maxAbs ?? Math.max(0.01, ...tokens.map((t) => Math.abs(t.score)));

  return (
    <div>
      <div className="flex flex-wrap gap-1.5 leading-relaxed">
        {tokens.map((t, i) => {
          const norm = Math.min(1, Math.abs(t.score) / scale);
          const pushesHarmful = t.score > 0;
          const intensity = 0.04 + norm * 0.22;

          return (
            <span
              key={`${t.token}-${i}`}
              title={`${t.token}: ${t.score >= 0 ? "+" : ""}${t.score.toFixed(2)}`}
              aria-label={`${t.token}, attribution ${t.score >= 0 ? "plus" : "minus"} ${Math.abs(t.score).toFixed(2)}, pushes toward ${pushesHarmful ? "harmful" : "benign"}`}
              className={`rounded px-1.5 py-0.5 text-sm transition-colors ${
                pushesHarmful
                  ? "border-l-2 border-on-surface text-on-surface"
                  : "border-r-2 border-outline-variant text-outline"
              }`}
              style={{
                background: pushesHarmful
                  ? `rgba(255,255,255,${intensity})`
                  : `rgba(255,255,255,${intensity * 0.25})`,
              }}
            >
              {t.token}
            </span>
          );
        })}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-5 border-t border-[var(--color-glass-border)] pt-4">
        <Legend side="left" label="Pushes toward harmful" bright />
        <Legend side="right" label="Pushes toward benign" />
        <span className="label-tech ml-auto text-outline">
          max |score| {scale.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function Legend({
  side,
  label,
  bright = false,
}: {
  side: "left" | "right";
  label: string;
  bright?: boolean;
}) {
  return (
    <span className="flex items-center gap-2">
      <span
        className={`h-4 w-6 rounded ${
          side === "left"
            ? "border-l-2 border-on-surface"
            : "border-r-2 border-outline-variant"
        }`}
        style={{
          background: bright ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.05)",
        }}
        aria-hidden
      />
      <span className="label-tech text-outline">{label}</span>
    </span>
  );
}
