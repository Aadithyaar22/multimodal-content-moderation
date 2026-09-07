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
- Each browser gets its own rater id on first decision (a one-time prompt,
  stored in `localStorage` — see `frontend/src/lib/moderator.ts`), so
  decisions from different raters are distinguishable afterwards. This was a
  real gap until this pass: the bar previously hardcoded `moderator_id:
  "mod_demo"` for everyone, which would have made a 15-person study
  indistinguishable from one person clicking 15 times.
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
   sure rather than guessing." They'll be prompted once for a rater id on
   their first decision; anything memorable is fine, it never leaves their
   browser except attached to their own decisions.

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
   list has it, since `moderator_id` is part of every stored decision
   (`DecisionRequest` in `src/mcm/serving/schemas.py`). For 10-15 raters
   across a few dozen items this is few enough calls to just script directly
   against `/items/{id}` for each id returned by `/queue?status=resolved`.

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
