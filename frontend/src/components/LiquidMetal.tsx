/**
 * Liquid-metal field — the shared background for the whole product.
 *
 * Built from heavily-blurred radial gradients in pure black and pure white,
 * drifting slowly against each other. The blur is what produces the soft,
 * flexible boundaries: there is no edge anywhere, only a gradient between
 * extremes, so the "surface" reads as molten rather than as flat panels.
 *
 * Content laid over this uses `mix-blend-difference`, which inverts type as it
 * crosses a boundary. That is the effect that makes the boundaries feel alive —
 * a word can be white on the dark side and black on the light side of the same
 * line — and it also guarantees contrast everywhere without any per-region
 * colour logic.
 *
 * All motion is transform-only on a fixed number of layers, so it composites on
 * the GPU and costs nothing per frame. It stops entirely under
 * prefers-reduced-motion, where the field simply becomes a static gradient.
 */

export function LiquidMetal({
  intensity = "full",
}: {
  /** "full" for the landing hero, "ambient" for working screens. */
  intensity?: "full" | "ambient";
}) {
  const ambient = intensity === "ambient";

  return (
    <div
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-black"
      aria-hidden
    >
      {/* The molten body: large white masses over black, blurred past any edge. */}
      <div
        className={ambient ? "liquid-layer liquid-a opacity-25" : "liquid-layer liquid-a"}
      />
      <div
        className={ambient ? "liquid-layer liquid-b opacity-20" : "liquid-layer liquid-b"}
      />
      <div
        className={ambient ? "liquid-layer liquid-c opacity-15" : "liquid-layer liquid-c"}
      />

      {/* Specular sheen — the highlight that reads as a metallic surface rather
          than as fog. Kept thin and moving at a different rate to the body. */}
      <div className={ambient ? "liquid-sheen opacity-30" : "liquid-sheen"} />

      {/* Fine grain breaks up the gradient banding that large soft blurs
          produce on 8-bit displays. */}
      <div className="liquid-grain" />
    </div>
  );
}
