# Running the human-agreement study

PROJECT_CONTEXT.md Sec. 6 calls for "human agreement rate (small Likert-scale
study, 10-15 raters)" as one of two checks — alongside explanation
faithfulness — most likely to get skipped under time pressure. The
infrastructure for it is built and tested; what's left is a real pass with
real people, which only you can run. This is the exact procedure.

## What's already built

- Every item's detail page (`/items/{id}`) ends in a decision bar: Approve /
  Remove / Escalate / Defer, plus two optional one-click toggles, "Agreed with
  model" and "Explanation useful" (`frontend/src/components/DecisionBar.tsx`).
- The toggles are deliberately optional — a required field on a bar clicked
  hundreds of times a day gets clicked through on autopilot, and the data
  becomes worthless.
- Raters are distinguishable by their real, verified Google account
  (`frontend/src/lib/auth.tsx`, `mcm.serving.auth`) — every decision requires
  signing in, and the backend rejects the request outright if the identity
  doesn't verify. This superseded an earlier, lighter mechanism (a
  self-chosen nickname stored in the browser, no verification at all) that
  was itself a fix for a worse bug: the bar originally hardcoded
  `moderator_id: "mod_demo"` for everyone, which would have made a
  15-person study indistinguishable from one person clicking 15 times. Real
  sign-in is a strict improvement for a study specifically — a rater can no
  longer be impersonated or double-counted under a different nickname — at
  the cost of the earlier version's two conveniences: a rater needs a Google
  account, and their real email now appears in the stored decision rather
  than an anonymous nickname they chose. If full rater anonymity actually
  matters for a given study (participants who'd answer differently if their
  identity were attached to it), account for that when recruiting rather
  than assuming the old anonymous flow is still there.
- `POST /items/{id}/decision` records every decision; `GET /stats` aggregates
  `agreement_rate` and `explanation_useful_rate` over every resolved item that
  answered (`src/mcm/serving/store.py::aggregate_stats`), tested in
  `tests/test_serving.py::TestStore::test_rates_are_over_responders_not_all_decisions`.

## Running it

1. **Seed the queue.** Raters need real items to react to, not an empty
   queue. Run 20-40 items through `/analyze` beforehand — a mix of clearly
   benign, clearly harmful, and the borderline "review" band, since agreement
   is a more meaningful signal on ambiguous cases than on obvious ones. The
   Hateful Memes / Fakeddit test-split images already on disk
   (`data/processed/*/test.parquet`) are a ready source if you don't want to
   source fresh content.

2. **Recruit 10-15 raters.** Anyone unfamiliar with this specific queue's
   items is fine — classmates, colleagues, whoever you can get a few minutes
   from. Send them the deployed frontend URL directly at `/queue`.

3. **Brief them in one sentence:** "Open a few items, read the verdict and
   explanation, then click Approve/Remove/Escalate/Defer as you genuinely
   would, and use the two toggles honestly — skip a toggle if you're not
   sure rather than guessing." They'll need to sign in with a Google account
   the first time they try to record a decision (an inline button right on
   the decision bar, no separate step) — this requires `GOOGLE_CLIENT_ID` /
   `NEXT_PUBLIC_GOOGLE_CLIENT_ID` to be configured first, see
   `docs/deployment.md`'s Google Sign-In section. Tell raters up front that
   their Google account email is what gets recorded against their ratings,
   not an anonymous handle — see the anonymity note above.

4. **Let each rater cover at least 5-8 items.** Below that, one rater's
   personal quirks dominate their contribution to the pooled rate.

5. **Pull the results:**

   ```bash
   curl -s https://<your-service-url>/api/v1/stats | python3 -m json.tool
   ```

   `model.agreement_rate` and `model.explanation_useful_rate` are what Sec. 6
   asks for, already computed over responders only (a skipped toggle is not
   scored as a "no").

6. **Report the actual rater count, not just the pooled rate.** The rate
   alone doesn't say whether it came from 2 raters or 15 — state both. If you
   want a per-rater breakdown for the report (e.g. to check no single rater's
   pattern dominates the pooled number), query the raw decisions directly:

   ```bash
   curl -s "https://<your-service-url>/api/v1/queue?status=resolved&limit=100" \
     | python3 -c "
   import sys, json
   items = json.load(sys.stdin)['items']
   print(f'{len(items)} resolved items')
   "
   ```

   The per-decision `moderator_id` isn't currently surfaced through a
   dedicated endpoint — `GET /items/{id}` on each resolved item's `decisions`
   list has it, since every stored decision carries it (set server-side from
   the verified Google sign-in — see `DecisionResponse` in
   `src/mcm/serving/schemas.py`, not the request body, which no longer
   accepts one at all). For 10-15 raters across a few dozen items this is
   few enough calls to just script directly against `/items/{id}` for each
   id returned by `/queue?status=resolved`.

## What this can't do for you

This assistant cannot recruit raters, click through the UI as a rater, or
invent Likert-scale responses to make the number come back — that would be
reporting fabricated data as a measured result, which is exactly the kind of
inflation the ablation table's significance testing (`scripts/ablation_table.py`)
and the faithfulness eval's random-region control (`scripts/faithfulness_eval.py`)
exist to keep this project's other numbers honest about. If time runs out
before a real pass happens, the report's honest option is to say so directly
in the limitations chapter — "human-agreement data was not collected" reads
far better under scrutiny than a number nobody can trace back to real raters.
