"use client";

/**
 * Analyze — manual submission.
 *
 * This is the demo surface: the one screen where someone can put in their own
 * image and caption and watch the system reason about the pair. It gets the
 * fullest treatment for that reason.
 *
 * Results render in place rather than routing away, so the input that produced
 * a verdict stays on screen beside it. Sending the user to a detail page would
 * break exactly the comparison the demo exists to make.
 */

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { analyze, getAttributions, getExplanation } from "@/lib/api";
import type { Attributions, Explanation, ItemDetail } from "@/lib/types";
import { ModalityLadder } from "@/components/ModalityLadder";
import { ShapTokens } from "@/components/ShapTokens";
import { EmergentBadge, GlassPanel, Skeleton, VerdictBadge } from "@/components/ui";

const MAX_BYTES = 10 * 1024 * 1024;

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [runOcr, setRunOcr] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ItemDetail | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [attributions, setAttributions] = useState<Attributions | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback((f: File | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("That file is not an image.");
      return;
    }
    if (f.size > MAX_BYTES) {
      setError(`Image is ${(f.size / 1e6).toFixed(1)}MB; the limit is 10MB.`);
      return;
    }
    setError(null);
    setFile(f);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  }, []);

  const canSubmit = (!!file || text.trim().length > 0) && !busy;

  async function run() {
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setExplanation(null);
    setAttributions(null);

    const form = new FormData();
    if (file) form.append("image", file);
    if (text.trim()) form.append("text", text.trim());
    form.append("run_ocr", String(runOcr));

    try {
      const item = await analyze(form);
      setResult(item);
      // Second, slower call. The verdict is already on screen by the time this
      // resolves; it must never gate the scores.
      getExplanation(item.item_id)
        .then(setExplanation)
        .catch(() => setExplanation(null));
      getAttributions(item.item_id)
        .then(setAttributions)
        .catch(() => setAttributions(null));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const task =
    result &&
    (result.heads.toxicity?.score ?? 0) >= (result.heads.misinformation?.score ?? 0)
      ? "toxicity"
      : "misinformation";

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <header className="text-center">
        <h1 className="display-vanguard text-[clamp(2.5rem,8vw,5.5rem)] text-on-surface">
          Analyze
        </h1>
        <p className="subtitle-athelas mt-3 text-xl text-on-surface-variant">
          Manual submission and multimodal testing
        </p>
      </header>

      <GlassPanel title="Media source" className="animate-rise">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            accept(e.dataTransfer.files?.[0]);
          }}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          role="button"
          tabIndex={0}
          aria-label="Add an image by dropping a file here or browsing"
          className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded border-2 border-dashed p-8 text-center transition-colors ${
            dragging
              ? "border-on-surface bg-[rgba(255,255,255,0.06)]"
              : "border-outline-variant hover:border-outline"
          }`}
        >
          {preview ? (
            <>
              {/* Grayscale by default, per the design system, and it does real
                  work here: evidence is often distressing and desaturating it
                  takes the edge off without hiding anything. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Selected evidence"
                className="max-h-64 rounded object-contain grayscale transition-[filter] duration-500 hover:grayscale-0"
              />
              <p className="label-tech mt-4 text-outline">
                {file?.name} · {((file?.size ?? 0) / 1e6).toFixed(1)}MB · click to replace
              </p>
            </>
          ) : (
            <>
              <span aria-hidden className="text-3xl text-outline">
                ⬆
              </span>
              <p className="mt-4 text-on-surface">Drag and drop an image here</p>
              <p className="label-tech mt-2 text-outline">
                or click to browse · max 10MB
              </p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => accept(e.target.files?.[0])}
          />
        </div>
      </GlassPanel>

      <GlassPanel title="Context & caption" className="animate-rise">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Enter the caption or post text. Leave blank to rely on the image alone."
          className="w-full resize-y rounded border border-[var(--color-glass-border)] bg-surface-dim p-4 text-on-surface placeholder:text-outline focus:border-outline focus:outline-none"
        />

        <label className="mt-4 flex cursor-pointer items-center justify-between rounded border border-[var(--color-glass-border)] p-4">
          <span>
            <span className="block font-medium text-on-surface">
              Run OCR extraction
            </span>
            <span className="text-sm text-outline">
              Detect and read text embedded in the image
            </span>
          </span>
          <input
            type="checkbox"
            checked={runOcr}
            onChange={(e) => setRunOcr(e.target.checked)}
            className="sr-only"
          />
          <span
            aria-hidden
            className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${
              runOcr ? "bg-on-surface" : "bg-surface-highest"
            }`}
          >
            <span
              className={`absolute top-1 h-4 w-4 rounded-full transition-transform ${
                runOcr ? "translate-x-6 bg-black" : "translate-x-1 bg-outline"
              }`}
            />
          </span>
        </label>
      </GlassPanel>

      {error && (
        <p className="label-tech text-center text-[var(--color-harm)]">{error}</p>
      )}

      <div className="flex flex-col items-center gap-4">
        <button
          onClick={run}
          disabled={!canSubmit}
          className="label-tech-lg rounded-full bg-white px-12 py-4 text-black transition-transform duration-300 hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
        >
          {busy ? "Analysing…" : "Run analysis"}
        </button>
        <p className="label-tech text-outline">
          Cold start: the first analysis can take 30–50s while inference nodes wake
        </p>
      </div>

      {busy && (
        <div className="space-y-4">
          <Skeleton className="h-56 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      )}

      {result && (
        <section className="space-y-6 border-t border-[var(--color-glass-border)] pt-8">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="display-vanguard text-3xl text-on-surface">Verdict</h2>
            <VerdictBadge label={result.verdict.label} />
            {result.fusion_signal.is_emergent && <EmergentBadge />}
            <Link
              href={`/items/${result.item_id}`}
              className="label-tech ml-auto text-outline underline hover:text-on-surface"
            >
              Open full record
            </Link>
          </div>

          <div className="animate-rise">
            <ModalityLadder
              scores={result.modality_scores}
              signal={result.fusion_signal}
              task={task}
            />
          </div>

          {attributions?.text && (
            <GlassPanel title="Token attribution (SHAP)">
              <ShapTokens tokens={attributions.text.tokens} />
            </GlassPanel>
          )}

          <GlassPanel title="Model reasoning">
            {explanation === null ? (
              <div className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-10/12" />
                <p className="label-tech pt-2 text-outline">Generating…</p>
              </div>
            ) : explanation.status === "ready" && explanation.narrative ? (
              <p className="font-editorial text-lg leading-relaxed text-on-surface">
                {explanation.narrative}
              </p>
            ) : (
              <p className="text-sm text-on-surface-variant">
                No narrative available. The scores above are unaffected.
              </p>
            )}
          </GlassPanel>

          <p className="label-tech text-center text-outline">
            Flagged for human review · nothing has been actioned automatically
          </p>
        </section>
      )}
    </div>
  );
}
