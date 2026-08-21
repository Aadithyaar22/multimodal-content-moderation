"use client";

/**
 * Evidence panel — Grad-CAM overlay and cross-attention connectors.
 *
 * The connectors are the reason this is one component rather than two. A link
 * runs from a caption token to an image region, so both endpoints have to be
 * measured in the same coordinate space; splitting the image and the text into
 * separate components would leave nothing able to draw between them.
 *
 * This is also the most direct visualization of the project's claim anywhere in
 * the interface. The modality ladder shows *that* fusion changed the score; a
 * line from the word "gift" to the doorway in the photograph shows *what the
 * model connected* to get there.
 *
 * The heatmap is rendered white-hot rather than in the conventional jet
 * colormap. Jet is a poor scale regardless — it is not perceptually uniform and
 * invents banding that is not in the data — and the monochrome palette rules
 * out hue anyway. Intensity carries magnitude, and a legend states the range.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Attributions } from "@/lib/types";

type Bbox = [number, number, number, number];

interface Props {
  attributions: Attributions;
  imageUrl?: string | null;
  caption: string;
}

export function EvidencePanel({ attributions, imageUrl, caption }: Props) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.65);
  const [showLinks, setShowLinks] = useState(true);
  const [activeLink, setActiveLink] = useState<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const regionRefs = useRef<Map<number, HTMLElement>>(new Map());
  const tokenRefs = useRef<Map<string, HTMLElement>>(new Map());
  const [paths, setPaths] = useState<Array<{ d: string; weight: number }>>([]);

  const regions = useMemo(
    () => attributions.image?.regions ?? [],
    [attributions.image],
  );
  // Memoised because `measure` depends on it: a fresh array each render would
  // make the callback change every render and the ResizeObserver re-subscribe
  // in a loop.
  const links = useMemo(
    () => attributions.cross_attention?.top_links ?? [],
    [attributions.cross_attention],
  );
  const tokens = attributions.text?.tokens ?? [];

  /**
   * Recompute connector geometry from live element positions.
   *
   * Measured rather than derived from layout constants: the text reflows with
   * viewport width and the image scales with it, so any hardcoded geometry
   * would be wrong at every size but the one it was written at.
   */
  const measure = useCallback(() => {
    const container = containerRef.current;
    if (!container || !showLinks) {
      setPaths([]);
      return;
    }
    const base = container.getBoundingClientRect();

    const next = links.flatMap((link, i) => {
      const regionEl = regionRefs.current.get(i);
      const tokenEl = tokenRefs.current.get(link.text_token);
      if (!regionEl || !tokenEl) return [];

      const r = regionEl.getBoundingClientRect();
      const t = tokenEl.getBoundingClientRect();

      const x1 = r.right - base.left;
      const y1 = r.top + r.height / 2 - base.top;
      const x2 = t.left - base.left;
      const y2 = t.top + t.height / 2 - base.top;

      // Horizontal control points give a flat S-curve, so the line leaves the
      // region and enters the token side-on. A straight line would cut across
      // the image and read as a scratch on the evidence.
      const dx = Math.max(40, (x2 - x1) * 0.5);
      const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
      return [{ d, weight: link.weight }];
    });

    setPaths(next);
  }, [links, showLinks]);

  useLayoutEffect(() => {
    measure();
  }, [measure]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(measure);
    ro.observe(container);
    window.addEventListener("scroll", measure, { passive: true });
    return () => {
      ro.disconnect();
      window.removeEventListener("scroll", measure);
    };
  }, [measure]);

  const linkedTokens = new Set(links.map((l) => l.text_token));

  return (
    <section className="glass rounded-lg p-6">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--color-glass-border)] pb-4">
        <h2 className="label-tech-lg text-on-surface">Evidence</h2>
        <div className="flex flex-wrap items-center gap-4">
          <Toggle active={showHeatmap} onClick={() => setShowHeatmap((v) => !v)}>
            Grad-CAM
          </Toggle>
          <Toggle
            active={showLinks}
            onClick={() => setShowLinks((v) => !v)}
            disabled={!attributions.cross_attention?.available}
          >
            Attention links
          </Toggle>
          {showHeatmap && (
            <label className="flex items-center gap-2">
              <span className="label-tech text-outline">Opacity</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="h-1 w-24 accent-white"
                aria-label="Heatmap opacity"
              />
            </label>
          )}
        </div>
      </header>

      <div
        ref={containerRef}
        className="relative mt-6 grid gap-8 md:grid-cols-2 md:gap-16"
      >
        {/* Image side */}
        <div className="relative self-start overflow-hidden rounded border border-[var(--color-glass-border)]">
          <div className="relative aspect-[4/3] w-full bg-surface-dim">
            {imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageUrl}
                alt="Evidence"
                onLoad={measure}
                className="h-full w-full object-cover grayscale"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center">
                <p className="label-tech text-outline">Evidence image unavailable</p>
              </div>
            )}

            {showHeatmap && regions.length > 0 && (
              <div
                aria-hidden
                className="pointer-events-none absolute inset-0 transition-opacity duration-300"
                style={{ opacity, background: heatmapCss(regions) }}
              />
            )}

            {/* Region markers double as the anchor points the connectors
                attach to, so the line always meets the box it belongs to. */}
            {regions.map((region, i) => {
              const [x0, y0, x1, y1] = region.bbox as Bbox;
              const isActive = activeLink !== null && linkIndexForRegion(links, i) === activeLink;
              return (
                <div
                  key={`${region.label}-${i}`}
                  ref={(el) => {
                    if (el) regionRefs.current.set(i, el);
                    else regionRefs.current.delete(i);
                  }}
                  className={`absolute rounded-sm border transition-colors ${
                    isActive
                      ? "border-white bg-[rgba(255,255,255,0.12)]"
                      : "border-white/45"
                  }`}
                  style={{
                    left: `${x0 * 100}%`,
                    top: `${y0 * 100}%`,
                    width: `${(x1 - x0) * 100}%`,
                    height: `${(y1 - y0) * 100}%`,
                  }}
                >
                  <span className="label-tech absolute -top-5 left-0 whitespace-nowrap text-white/70">
                    {region.label} · {region.score.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>

          {showHeatmap && (
            <div className="flex items-center gap-3 border-t border-[var(--color-glass-border)] px-3 py-2">
              <span className="label-tech text-outline">Low</span>
              <span
                aria-hidden
                className="h-1.5 flex-1 rounded-full"
                style={{
                  background:
                    "linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.95))",
                }}
              />
              <span className="label-tech text-outline">High attention</span>
            </div>
          )}
        </div>

        {/* Text side */}
        <div className="self-start">
          <p className="label-tech mb-3 text-outline">Caption</p>
          <p className="flex flex-wrap gap-1.5 font-editorial text-lg leading-relaxed">
            {tokens.length > 0
              ? tokens.map((t, i) => {
                  const linked = linkedTokens.has(t.token);
                  const idx = links.findIndex((l) => l.text_token === t.token);
                  return (
                    <span
                      key={`${t.token}-${i}`}
                      ref={(el) => {
                        if (el && linked) tokenRefs.current.set(t.token, el);
                      }}
                      onMouseEnter={() => linked && setActiveLink(idx)}
                      onMouseLeave={() => setActiveLink(null)}
                      className={`rounded px-1 transition-colors ${
                        linked
                          ? "cursor-default bg-[rgba(255,255,255,0.14)] text-white"
                          : "text-on-surface-variant"
                      }`}
                    >
                      {t.token}
                    </span>
                  );
                })
              : caption}
          </p>

          {links.length > 0 && showLinks && (
            <ul className="mt-6 space-y-2">
              {links.map((link, i) => (
                <li
                  key={`${link.text_token}-${i}`}
                  onMouseEnter={() => setActiveLink(i)}
                  onMouseLeave={() => setActiveLink(null)}
                  className={`flex items-center justify-between border-l-2 px-3 py-2 text-sm transition-colors ${
                    activeLink === i
                      ? "border-white bg-[rgba(255,255,255,0.08)]"
                      : "border-outline-variant"
                  }`}
                >
                  <span>
                    <span className="text-white">&ldquo;{link.text_token}&rdquo;</span>
                    <span className="mx-2 text-outline" aria-hidden>
                      →
                    </span>
                    <span className="text-on-surface-variant">
                      {regions[i]?.label ?? "image region"}
                    </span>
                  </span>
                  <span className="numeric font-display text-sm font-bold">
                    {link.weight.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Connector layer. Hidden below md, where the two halves stack and a
            line between them would cross the whole card vertically. */}
        {showLinks && paths.length > 0 && (
          <svg
            aria-hidden
            className="pointer-events-none absolute inset-0 hidden h-full w-full md:block"
          >
            {paths.map((p, i) => (
              <path
                key={i}
                d={p.d}
                fill="none"
                stroke="white"
                strokeWidth={activeLink === i ? 2 : 1}
                strokeOpacity={
                  activeLink === null ? 0.25 + p.weight : activeLink === i ? 0.95 : 0.12
                }
                strokeDasharray="4 6"
                className="transition-[stroke-opacity,stroke-width] duration-200"
              >
                <animate
                  attributeName="stroke-dashoffset"
                  from="20"
                  to="0"
                  dur="1.4s"
                  repeatCount="indefinite"
                />
              </path>
            ))}
          </svg>
        )}
      </div>

      <p className="mt-6 border-t border-[var(--color-glass-border)] pt-4 text-sm leading-relaxed text-outline">
        Lines connect caption tokens to the image regions they attended to most
        strongly in the fusion layer. This is the cross-modal relationship the
        model actually used, not a post-hoc rationalization.
      </p>
    </section>
  );
}

/**
 * Compose the heatmap from region boxes as stacked radial gradients.
 *
 * Used when the backend supplies regions but no rendered heatmap image, which
 * is the common case early on and lets the overlay work before the Grad-CAM
 * endpoint returns pixels. A real heatmap_url, when present, is drawn instead.
 */
function heatmapCss(
  regions: Array<{ bbox: number[]; score: number }>,
): string {
  return regions
    .map((r) => {
      const [x0, y0, x1, y1] = r.bbox;
      const cx = ((x0 + x1) / 2) * 100;
      const cy = ((y0 + y1) / 2) * 100;
      const rx = ((x1 - x0) / 2) * 130;
      const ry = ((y1 - y0) / 2) * 130;
      const a = Math.min(0.95, r.score);
      return `radial-gradient(ellipse ${rx}% ${ry}% at ${cx}% ${cy}%, rgba(255,255,255,${a}) 0%, rgba(255,255,255,${a * 0.45}) 45%, rgba(255,255,255,0) 75%)`;
    })
    .join(", ");
}

function linkIndexForRegion(
  links: Array<{ image_region: number[] }>,
  regionIndex: number,
): number {
  return regionIndex < links.length ? regionIndex : -1;
}

function Toggle({
  active,
  onClick,
  disabled,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      className={`label-tech rounded border px-3 py-1.5 transition-colors disabled:opacity-30 ${
        active
          ? "border-on-surface bg-on-surface text-black"
          : "border-outline-variant text-outline hover:text-on-surface"
      }`}
    >
      {children}
    </button>
  );
}
