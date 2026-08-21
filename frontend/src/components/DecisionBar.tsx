"use client";

/**
 * The moderator's action bar.
 *
 * Language matters here. The original Stitch screens used BLOCK / REJECT /
 * ALLOW, which reads as the system acting. These are the moderator's decisions,
 * recorded by the system, and the wording says so.
 *
 * The two feedback toggles feed the human-agreement metric in report Sec. 6.
 * They are deliberately optional and one-click: a required field on a bar a
 * moderator hits hundreds of times a day gets clicked through, and the data
 * becomes worthless.
 */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { submitDecision } from "@/lib/api";
import { announceToIsland } from "@/components/DynamicIsland";
import type { DecisionAction } from "@/lib/types";

const ACTIONS: Array<{ action: DecisionAction; label: string; key: string }> = [
  { action: "approve", label: "Approve", key: "a" },
  { action: "remove", label: "Remove", key: "r" },
  { action: "escalate", label: "Escalate", key: "e" },
  { action: "defer", label: "Defer", key: "d" },
];

export function DecisionBar({ itemId }: { itemId: string }) {
  const router = useRouter();
  const [pending, setPending] = useState<DecisionAction | null>(null);
  const [done, setDone] = useState<DecisionAction | null>(null);
  const [agreed, setAgreed] = useState<boolean | null>(null);
  const [useful, setUseful] = useState<boolean | null>(null);

  async function decide(action: DecisionAction) {
    if (pending || done) return;
    setPending(action);
    try {
      await submitDecision(itemId, {
        action,
        moderator_id: "mod_demo",
        agreed_with_model: agreed ?? undefined,
        explanation_was_useful: useful ?? undefined,
      });
      setDone(action);
      announceToIsland(`Recorded: ${action}`);
      setTimeout(() => router.push("/queue"), 700);
    } finally {
      setPending(null);
    }
  }

  // Moderators work fast; the keyboard path is the real one.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;
      const hit = ACTIONS.find((a) => a.key === e.key.toLowerCase());
      if (hit) {
        e.preventDefault();
        decide(hit.action);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending, done, agreed, useful]);

  return (
    <div className="sticky bottom-0 z-40 -mx-6 mt-10 border-t border-[var(--color-glass-border)] bg-[rgba(0,0,0,0.8)] px-6 py-4 backdrop-blur-2xl md:-mx-12 md:px-12">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-4">
        <span className="label-tech text-outline">
          {done ? "Decision recorded" : "Your decision"}
        </span>

        <div className="flex flex-wrap gap-2">
          {ACTIONS.map(({ action, label, key }) => (
            <button
              key={action}
              onClick={() => decide(action)}
              disabled={!!pending || !!done}
              className={`label-tech rounded border px-4 py-2.5 transition-colors disabled:opacity-40 ${
                done === action
                  ? "border-on-surface bg-on-surface text-black"
                  : action === "remove"
                    ? "border-[var(--color-harm)] text-[var(--color-harm)] hover:bg-[rgba(255,180,171,0.1)]"
                    : "border-outline-variant text-on-surface hover:border-outline"
              }`}
            >
              {pending === action ? "…" : label}
              <kbd className="ml-2 text-[10px] text-outline">{key}</kbd>
            </button>
          ))}
        </div>

        <div className="ml-auto flex flex-wrap gap-2">
          <FeedbackToggle
            active={agreed}
            onToggle={setAgreed}
            label="Agreed with model"
          />
          <FeedbackToggle
            active={useful}
            onToggle={setUseful}
            label="Explanation useful"
          />
        </div>
      </div>
    </div>
  );
}

function FeedbackToggle({
  active,
  onToggle,
  label,
}: {
  active: boolean | null;
  onToggle: (v: boolean | null) => void;
  label: string;
}) {
  return (
    <button
      onClick={() => onToggle(active === true ? null : true)}
      aria-pressed={active === true}
      className={`label-tech rounded border px-3 py-2 transition-colors ${
        active
          ? "border-on-surface text-on-surface"
          : "border-outline-variant text-outline hover:text-on-surface"
      }`}
    >
      {active ? "✓ " : ""}
      {label}
    </button>
  );
}
