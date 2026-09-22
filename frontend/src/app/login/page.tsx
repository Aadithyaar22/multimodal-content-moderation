"use client";

/**
 * Sign in — required only to submit a moderator decision.
 *
 * Analyzing content and checking claims stay open to everyone with no
 * account; this page exists because recording *who decided what* on an item
 * is a permanent record, and until this session anyone with the URL could
 * write one as anyone by typing any moderator_id they liked. Google does the
 * actual authentication here — this page only hosts Google's own button and
 * hands the resulting token to the backend, which is the only thing that
 * verifies it (mcm.serving.auth).
 */

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth, renderGoogleButton } from "@/lib/auth";
import { GlassPanel } from "@/components/ui";

export default function LoginPage() {
  const { user, ready, configured } = useAuth();
  const router = useRouter();
  const buttonRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (user) router.replace("/queue");
  }, [user, router]);

  useEffect(() => {
    if (ready && buttonRef.current) {
      renderGoogleButton(buttonRef.current);
    }
  }, [ready]);

  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-8 py-24 text-center">
      <header>
        <h1 className="display-vanguard text-5xl text-on-surface">Sign in</h1>
        <p className="subtitle-athelas mt-3 text-lg text-on-surface-variant">
          Recording a moderator decision requires a verified Google account.
          Analyzing content and checking claims stay open to everyone.
        </p>
      </header>

      <GlassPanel className="w-full">
        {configured ? (
          <div className="flex flex-col items-center gap-3 py-2">
            <div ref={buttonRef} />
            {!ready && (
              <p className="label-tech text-outline">Loading sign-in…</p>
            )}
          </div>
        ) : (
          <p className="text-sm text-on-surface-variant">
            Sign-in is not configured on this deployment (
            <code className="text-outline">GOOGLE_CLIENT_ID</code> /{" "}
            <code className="text-outline">NEXT_PUBLIC_GOOGLE_CLIENT_ID</code>{" "}
            unset) — decisions cannot be recorded until it is.
          </p>
        )}
      </GlassPanel>
    </div>
  );
}
