"use client";

/**
 * Sign in — required only to submit a moderator decision.
 *
 * Analyzing content and checking claims stay open to everyone with no
 * account; this page exists because recording *who decided what* on an item
 * is a permanent record, and until this session anyone with the URL could
 * write one as anyone by typing any moderator_id they liked.
 *
 * Two independent ways to sign in, both landing in the same session:
 * Google (its own button, hosted by Google — this page never sees a
 * password), or an email/password account this service issues and verifies
 * itself (mcm.serving.accounts). Neither is "the real one" — pick whichever
 * you have.
 */

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth, renderGoogleButton } from "@/lib/auth";
import { GlassPanel } from "@/components/ui";

type Mode = "login" | "register";

export default function LoginPage() {
  const { user, ready, configured, registerWithPassword, loginWithPassword } = useAuth();
  const router = useRouter();
  const buttonRef = useRef<HTMLDivElement>(null);

  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user) router.replace("/queue");
  }, [user, router]);

  useEffect(() => {
    if (ready && buttonRef.current) {
      renderGoogleButton(buttonRef.current);
    }
  }, [ready]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!email.trim() || !password) {
      setError("Enter an email and password.");
      return;
    }
    if (mode === "register" && !name.trim()) {
      setError("Enter a name.");
      return;
    }
    if (mode === "register" && password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setBusy(true);
    try {
      if (mode === "register") {
        await registerWithPassword(email.trim(), password, name.trim());
      } else {
        await loginWithPassword(email.trim(), password);
      }
      // Redirect happens in the effect above once `user` updates.
    } catch (err) {
      setError((err as Error).message || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-8 py-24 text-center">
      <header>
        <h1 className="display-vanguard text-5xl text-on-surface">Sign in</h1>
        <p className="subtitle-athelas mt-3 text-lg text-on-surface-variant">
          Recording a moderator decision requires signing in. Analyzing
          content and checking claims stay open to everyone.
        </p>
      </header>

      <GlassPanel className="w-full">
        <div className="flex flex-col gap-6 py-2">
          {configured ? (
            <div className="flex flex-col items-center gap-3">
              <div ref={buttonRef} />
              {!ready && <p className="label-tech text-outline">Loading sign-in…</p>}
            </div>
          ) : (
            <p className="text-sm text-on-surface-variant">
              Google Sign-In is not configured on this deployment (
              <code className="text-outline">GOOGLE_CLIENT_ID</code> /{" "}
              <code className="text-outline">NEXT_PUBLIC_GOOGLE_CLIENT_ID</code> unset) — use
              email and password below instead.
            </p>
          )}

          <div className="flex items-center gap-3">
            <span className="h-px flex-1 bg-[var(--color-glass-border)]" />
            <span className="label-tech text-outline">or</span>
            <span className="h-px flex-1 bg-[var(--color-glass-border)]" />
          </div>

          <div className="flex justify-center gap-2">
            <button
              type="button"
              onClick={() => {
                setMode("login");
                setError(null);
              }}
              className={`label-tech rounded-full px-4 py-1.5 transition-colors ${
                mode === "login"
                  ? "bg-white text-black"
                  : "border border-outline-variant text-on-surface hover:border-outline"
              }`}
            >
              Log in
            </button>
            <button
              type="button"
              onClick={() => {
                setMode("register");
                setError(null);
              }}
              className={`label-tech rounded-full px-4 py-1.5 transition-colors ${
                mode === "register"
                  ? "bg-white text-black"
                  : "border border-outline-variant text-on-surface hover:border-outline"
              }`}
            >
              Create account
            </button>
          </div>

          <form onSubmit={submit} className="flex flex-col gap-3 text-left">
            {mode === "register" && (
              <label className="flex flex-col gap-1">
                <span className="label-tech text-outline">Name</span>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  autoComplete="name"
                  className="rounded border border-outline-variant bg-transparent px-3 py-2 text-on-surface outline-none focus:border-outline"
                />
              </label>
            )}
            <label className="flex flex-col gap-1">
              <span className="label-tech text-outline">Email</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                className="rounded border border-outline-variant bg-transparent px-3 py-2 text-on-surface outline-none focus:border-outline"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="label-tech text-outline">Password</span>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "register" ? "new-password" : "current-password"}
                className="rounded border border-outline-variant bg-transparent px-3 py-2 text-on-surface outline-none focus:border-outline"
              />
            </label>

            {error && (
              <p className="label-tech text-[var(--color-harm)]">{error}</p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="label-tech-lg mt-2 rounded-full bg-white px-6 py-3 font-bold text-black transition-transform hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
            >
              {busy ? "…" : mode === "register" ? "Create account" : "Log in"}
            </button>

            {mode === "register" && (
              <p className="text-xs text-on-surface-variant">
                No email verification yet — anyone can register with any
                email address. Don&apos;t reuse a password you use elsewhere.
              </p>
            )}
          </form>
        </div>
      </GlassPanel>
    </div>
  );
}
