/**
 * Per-browser rater identity for the human-agreement study (PROJECT_CONTEXT.md
 * Sec. 6: "human agreement rate, 10-15 raters").
 *
 * Every decision used to be submitted as moderator_id="mod_demo" regardless of
 * who was actually clicking — harmless for the pooled agreement_rate GET
 * /stats reports (it isn't grouped by rater), but it means a real multi-rater
 * study would have no way to tell its raters apart afterwards, or even count
 * how many actually took part. Asked once per browser, not per decision — the
 * decision bar is deliberately one-click (see DecisionBar.tsx's own comment),
 * and a per-decision prompt would defeat that.
 */

const KEY = "mcm_moderator_id";

export function getModeratorId(): string {
  try {
    const existing = localStorage.getItem(KEY);
    if (existing) return existing;

    const entered = window.prompt(
      "Rater ID for this session (used only to group your decisions in the study — anything memorable works):",
    );
    const id = entered?.trim() || `mod_${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem(KEY, id);
    return id;
  } catch {
    // Private browsing or a blocked storage API: fall back to a per-decision
    // random id rather than throwing — a submitted decision must still work.
    return `mod_${Math.random().toString(36).slice(2, 8)}`;
  }
}
