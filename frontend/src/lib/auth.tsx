"use client";

/**
 * Sign-in, client-side — two independent methods, one shared session shape.
 *
 * Google Identity Services (GIS) hands back a signed ID token directly in
 * the browser — no backend-for-frontend, no server-side OAuth callback, no
 * client secret anywhere in this frontend (Google's client id identifies
 * the app, it isn't a credential, so it's fine to ship it in client code).
 * Email/password accounts (registerWithPassword / loginWithPassword) call
 * the backend's own /auth endpoints (mcm.serving.accounts) and get back a
 * self-issued token in the same shape. Either way, the token is sent as-is
 * to the backend on every decision; the backend is the only thing that
 * actually verifies it (mcm.serving.auth tries both verification paths) —
 * this file's job is only to obtain a token and hold it for the session,
 * never to authorize anything on its own.
 *
 * The token is mirrored into localStorage purely for "stay signed in across
 * a reload" convenience. It is per-browser, never sent anywhere but this
 * app's own backend, and every write action re-verifies it server-side on
 * every request — nothing here treats localStorage as a security boundary,
 * only as a UX one.
 */

import Script from "next/script";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { loginWithPassword as apiLogin, registerWithPassword as apiRegister } from "./api";

export interface AuthUser {
  email: string;
  name: string;
  picture?: string;
}

interface AuthState {
  user: AuthUser | null;
  /** Non-null only when `user` is also non-null (a currently-valid token). */
  idToken: string | null;
  /** The GIS script has loaded and google.accounts.id is callable. */
  ready: boolean;
  /** Google Sign-In specifically is configured (NEXT_PUBLIC_GOOGLE_CLIENT_ID
   * set) — email/password accounts have no equivalent client-side gate,
   * since they need no config on this side at all. */
  configured: boolean;
  registerWithPassword: (email: string, password: string, name: string) => Promise<void>;
  loginWithPassword: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthState | null>(null);
const STORAGE_KEY = "mcm_auth_token";
export const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

function decodeIdToken(token: string): AuthUser | null {
  try {
    const [, payload] = token.split(".");
    const json = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
    // Expiry is enforced server-side on every request that matters — this
    // client-side check only avoids showing a stale "signed in" state for a
    // token that would just get a 401 on the next real request.
    if (typeof json.exp === "number" && json.exp * 1000 < Date.now()) return null;
    if (typeof json.email !== "string") return null;
    return { email: json.email, name: json.name ?? json.email, picture: json.picture };
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [rawToken, setRawToken] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Deliberately not a useState lazy initializer: this component renders
    // server-side first (where localStorage doesn't exist), so the state
    // must start at its SSR-safe default (signed out) and only pick up a
    // stored token after mount — reading localStorage during the initial
    // render would either throw under SSR or hydrate to a value the server
    // never rendered, both worse than the one extra render this causes.
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (stored) setRawToken(stored);
    } catch {
      /* private browsing or blocked storage — falls back to signed out */
    }
  }, []);

  const applyToken = useCallback((token: string) => {
    setRawToken(token);
    try {
      localStorage.setItem(STORAGE_KEY, token);
    } catch {
      /* ignore — the session just won't survive a reload */
    }
  }, []);

  const handleCredential = useCallback(
    (response: { credential: string }) => applyToken(response.credential),
    [applyToken],
  );

  // Deliberately NOT a useEffect keyed on `ready`: GIS's own script.onLoad
  // callback is where this has to live. React runs a descendant's effects
  // before its ancestor's for the same commit, and the login page's own
  // "render the button once ready" effect is exactly such a descendant —
  // keying both on the same `ready` flag raced initialize() against
  // renderButton(), and renderButton always lost, failing silently but for
  // one console warning ("Failed to render button before calling
  // initialize()") that's easy to miss in a deploy nobody's watching the
  // console on. Calling initialize() synchronously inside onLoad, before
  // setReady(true) even runs, guarantees it has already happened by the
  // time any consumer sees ready=true and tries to render a button.
  const onGoogleScriptLoad = useCallback(() => {
    window.google?.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: handleCredential,
    });
    setReady(true);
  }, [handleCredential]);

  const registerWithPassword = useCallback(
    async (email: string, password: string, name: string) => {
      const res = await apiRegister(email, password, name);
      applyToken(res.token);
    },
    [applyToken],
  );

  const loginWithPassword = useCallback(
    async (email: string, password: string) => {
      const res = await apiLogin(email, password);
      applyToken(res.token);
    },
    [applyToken],
  );

  const signOut = useCallback(() => {
    setRawToken(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    window.google?.accounts.id.disableAutoSelect();
  }, []);

  const user = useMemo(() => (rawToken ? decodeIdToken(rawToken) : null), [rawToken]);

  const value = useMemo<AuthState>(
    () => ({
      user,
      idToken: user ? rawToken : null,
      ready,
      configured: !!GOOGLE_CLIENT_ID,
      registerWithPassword,
      loginWithPassword,
      signOut,
    }),
    [user, rawToken, ready, registerWithPassword, loginWithPassword, signOut],
  );

  return (
    <AuthContext.Provider value={value}>
      {GOOGLE_CLIENT_ID && (
        <Script
          src="https://accounts.google.com/gsi/client"
          strategy="afterInteractive"
          onLoad={onGoogleScriptLoad}
        />
      )}
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}

/**
 * Renders Google's own Sign In With Google button into `el` once GIS is
 * ready. A thin wrapper so every call site (the login page, the inline
 * prompt in DecisionBar) renders the identical, Google-styled control
 * rather than a hand-drawn stand-in — the real button carries Google's own
 * account picker and is what makes this a *secure* login, not a decorative
 * one.
 */
export function renderGoogleButton(
  el: HTMLElement,
  options: Record<string, unknown> = {},
): void {
  window.google?.accounts.id.renderButton(el, {
    theme: "filled_black",
    size: "large",
    shape: "pill",
    text: "signin_with",
    ...options,
  });
}

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: {
            client_id: string;
            callback: (response: { credential: string }) => void;
          }) => void;
          renderButton: (parent: HTMLElement, options: Record<string, unknown>) => void;
          prompt: () => void;
          disableAutoSelect: () => void;
        };
      };
    };
  }
}
