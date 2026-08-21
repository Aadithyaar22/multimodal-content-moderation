"use client";

/**
 * Item detail — evidence, scores, and reasoning.
 *
 * The explanation loads on its own slower call and is streamed in. Blocking the
 * whole page on a 2-5s LLM round trip would make every verdict wait on the
 * slowest optional component; the scores are the verdict and arrive first.
 */

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { getAttributions, getExplanation, getItem, resolveApiUrl } from "@/lib/api";
import type { Attributions, Explanation, ItemDetail } from "@/lib/types";
import { ModalityLadder } from "@/components/ModalityLadder";
import { ShapTokens } from "@/components/ShapTokens";
import { EvidencePanel } from "@/components/EvidencePanel";
import { DecisionBar } from "@/components/DecisionBar";
import {
  EmergentBadge,
  GlassPanel,
  Skeleton,
  VerdictBadge,
} from "@/components/ui";

export default function ItemPage({ params }: PageProps<"/items/[id]">) {
  const { id } = use(params);
  const [item, setItem] = useState<ItemDetail | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  // Attributions live on their own endpoint in the contract, so they are
  // fetched separately rather than read off the item record — that way the
  // panel appears as soon as the backend starts returning them.
  const [attributions, setAttributions] = useState<Attributions | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getItem(id)
      .then((d) => !cancelled && setItem(d))
      .catch((e) => !cancelled && setError(e.message));
    // Fired in parallel, rendered when it lands.
    getExplanation(id)
      .then((e) => !cancelled && setExplanation(e))
      .catch(() => {});
    getAttributions(id)
      .then((a) => !cancelled && setAttributions(a))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    return (
      <GlassPanel title="Not found">
        <p className="text-sm text-on-surface-variant">{error}</p>
        <Link href="/queue" className="label-tech mt-4 inline-block underline">
          Back to queue
        </Link>
      </GlassPanel>
    );
  }

  if (!item) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-14 w-2/3" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Skeleton className="h-80" />
          <Skeleton className="h-80" />
        </div>
      </div>
    );
  }

  const task =
    (item.heads.toxicity?.score ?? 0) >= (item.heads.misinformation?.score ?? 0)
      ? "toxicity"
      : "misinformation";

  // Only render the evidence panel when there is something in it. The backend
  // returns a well-formed but empty payload until step 6 lands.
  const hasAttributions = Boolean(
    attributions &&
      (attributions.text ||
        attributions.image ||
        attributions.cross_attention?.available),
  );

  return (
    <div className="space-y-8">
      <header className="animate-rise">
        <Link href="/queue" className="label-tech text-outline hover:text-on-surface">
          ← Queue
        </Link>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <h1 className="display-vanguard text-[clamp(2rem,5vw,3.5rem)]">
            {item.item_id.replace("itm_", "").toUpperCase()}
          </h1>
          <VerdictBadge label={item.verdict.label} />
          {item.fusion_signal.is_emergent && <EmergentBadge />}
        </div>
        <p className="mt-2 label-tech text-outline">
          Analysed {new Date(item.created_at).toUTCString()} ·{" "}
          {item.latency_ms.total}ms
        </p>
      </header>

      {/* The modality comparison leads, because it is the finding. */}
      <div className="animate-rise" style={{ animationDelay: "60ms" }}>
        <ModalityLadder
          scores={item.modality_scores}
          signal={item.fusion_signal}
          task={task}
        />
      </div>

      {/* Evidence leads the explainability section: the Grad-CAM regions and the
          attention connectors are the most direct evidence of what the model
          actually related, so they sit above the per-head breakdown. */}
      {hasAttributions && (
        <div className="animate-rise">
          <EvidencePanel
            attributions={attributions!}
            imageUrl={resolveApiUrl(item.input.image_url)}
            caption={item.input.text}
          />
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <GlassPanel title="Content" className="animate-rise">
          <p className="font-editorial text-lg leading-relaxed text-on-surface">
            {item.input.text}
          </p>
          {item.input.ocr_text && (
            <p className="mt-4 border-l-2 border-outline-variant pl-3 text-sm text-outline">
              Text detected in image: {item.input.ocr_text}
            </p>
          )}
        </GlassPanel>

        <GlassPanel title="Classifier heads" className="animate-rise">
          <div className="space-y-6">
            <HeadBlock
              name="Harassment"
              label={item.heads.toxicity.label}
              classes={item.heads.toxicity.classes}
            />
            <HeadBlock
              name="Misinformation"
              label={item.heads.misinformation.label}
              classes={item.heads.misinformation.classes}
            />
            {item.deepfake.checked && (
              <div className="border-t border-[var(--color-glass-border)] pt-4">
                <div className="flex items-baseline justify-between">
                  <span className="label-tech text-outline">
                    Manipulation check
                  </span>
                  <span className="numeric font-display font-bold">
                    {item.deepfake.label} · {item.deepfake.score.toFixed(2)}
                  </span>
                </div>
                <p className="mt-2 text-xs text-outline">
                  Scored separately and combined at verdict level — pixel
                  artefacts are unrelated to caption text.
                </p>
              </div>
            )}
          </div>
        </GlassPanel>
      </div>

      {attributions?.text && (
        <GlassPanel title="Token attribution (SHAP)" className="animate-rise">
          <ShapTokens tokens={attributions.text.tokens} />
        </GlassPanel>
      )}

      <GlassPanel title="Model reasoning" className="animate-rise">
        {explanation === null ? (
          <div className="space-y-3">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-11/12" />
            <Skeleton className="h-4 w-4/5" />
            <p className="label-tech pt-2 text-outline">Generating…</p>
          </div>
        ) : explanation.status === "ready" && explanation.narrative ? (
          <>
            <p className="font-editorial text-lg leading-relaxed text-on-surface">
              {explanation.narrative}
            </p>
            {explanation.key_factors.length > 0 && (
              <ul className="mt-6 space-y-2">
                {explanation.key_factors.map((f) => (
                  <li
                    key={f.factor}
                    className="flex items-center justify-between border-l-2 border-on-surface bg-[rgba(255,255,255,0.05)] px-4 py-3"
                  >
                    <span className="text-sm">
                      <span className="label-tech mr-3 text-outline">
                        {f.modality}
                      </span>
                      {f.factor}
                    </span>
                    <span className="numeric font-display text-sm font-bold">
                      {f.weight.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <p className="label-tech mt-6 text-outline">
              {explanation.model} · {explanation.latency_ms}ms
            </p>
          </>
        ) : (
          // The verdict never depends on the LLM, so its absence degrades
          // quietly rather than blocking the page.
          <p className="text-sm text-on-surface-variant">
            No narrative available for this item. The scores and attributions
            above are unaffected.
          </p>
        )}
      </GlassPanel>

      <DecisionBar itemId={item.item_id} />
    </div>
  );
}

function HeadBlock({
  name,
  label,
  classes,
}: {
  name: string;
  label: string;
  classes: Record<string, number>;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label-tech text-outline">{name}</span>
        <span className="label-tech text-on-surface">{label}</span>
      </div>
      <div className="mt-3 space-y-2">
        {Object.entries(classes).map(([cls, p]) => (
          <div key={cls} className="flex items-center gap-3">
            <span className="w-24 text-xs text-outline">{cls}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-highest">
              <div
                className="animate-bar h-full rounded-full bg-on-surface-variant"
                style={{ width: `${p * 100}%` }}
              />
            </div>
            <span className="numeric w-12 text-right text-xs text-on-surface-variant">
              {p.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
