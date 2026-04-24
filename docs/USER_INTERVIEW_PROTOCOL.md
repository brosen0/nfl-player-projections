# User-interview protocol — start/sit paper prototype

**Provenance:** action #5 of the 2026-04-23 council's Critical Next
Steps (`council-transcript-20260423-051434.md`). Both peer reviewers
flagged "no user defined" as a joint blind spot; this protocol
exists so the next session is a real conversation, not another
internal review.

## Chairman's success signal (literal)

> User can articulate a decision the tool changed or confirmed, and
> names the format/scoring/league-size constraints you must support.

Two outcomes that satisfy this:

1. User points at a specific 2024/2025 week and says "the model
   would have flipped my start/sit at RB2, I started X and it says
   start Y, and actually yeah Y went off that week." That's a
   confirmation signal.
2. User points at a specific 2024/2025 week and says "the model's
   swap makes no sense because it doesn't know about [injury tag
   / practice report / Vegas line that moved Saturday / bye week]."
   That's a feature-gap signal.

Either one closes the chairman's gate. A 30-minute session with
one user yields both.

## Before the session — operator prep (~15 minutes)

1. Confirm the user picks a specific league format. V1 supports
   **PPR, one-player-per-core-slot (QB:1, RB:2, WR:2, TE:1), no
   FLEX**. If the user's league is half-PPR or has a FLEX, note
   the constraint and keep going — you're gathering format
   requirements, not trying to match every format on v1.
2. Ask the user to pick 1–3 weeks from 2024 or 2025 that they
   **remember well** — a pivotal start/sit, a regret, a lucky
   call. Live memory beats random sampling.
3. For each chosen week, ask the user to write down their full
   15-player roster AND which 6 they started. Fill out the roster
   JSON template:
   ```
   cp docs/roster_template.json /tmp/user_roster_week10.json
   # edit to match the user's actual roster
   ```
4. Sanity-check the JSON loads:
   ```
   python scripts/start_sit_prototype.py \
       --roster /tmp/user_roster_week10.json \
       --season 2024 --week 10
   ```
   If any roster entries are UNMATCHED, the user's spelling
   didn't match the walk-forward CSV — tell them, fix together,
   re-run. UNMATCHED rows are surfaced loudly on purpose.

## During the session — what to capture

### Structured capture (first 10 minutes)

For each week:

- **Model's recommended swap** (from the tool output): swap in
  player X for player Y.
- **User's reaction (exact quote):** "yes — I knew I should have
  done that" / "no — here's why" / "what?"
- **Retrospective delta** from the tool: actual FP model-started vs
  user-started. Note sign + magnitude.
- **Feature the user referenced when disagreeing:** "the injury
  report on Thursday was …" / "the weather was …" / "the Vegas
  line moved because of …" / "the beat reporter said …" — these
  are feature candidates.

### Open-ended (next 15 minutes)

- What info do you use in your live start/sit decision that the
  tool didn't show?
- If you only had this tool (nothing else) for a week, would you
  trust it? What's the threshold where you'd override it?
- What would make you stop using it entirely?
- What scoring format is yours (PPR / half / standard / custom)?
  What league size?
- What's the single decision you care about most — start/sit,
  draft, waiver, trade? Which is lowest-effort for the tool to
  change?

### Close (5 minutes)

- Tell the user: "I'm keeping a list of features the tool
  doesn't know about. Yours are: [list from the feature-reference
  capture above]." Get them to confirm or add.
- Ask: "If I email you a week's start/sit recs every Sunday
  morning for 4 weeks, would you use them?" That's the
  willingness-to-commit signal — if no, the product isn't ready
  for that user yet.

## After the session — operator output (~15 minutes)

Write up the session as `docs/USER_SESSION_{YYYYMMDD}_{handle}.md`
with:

1. **Header** — date, user handle, league format, scoring, roster
   size.
2. **Decisions reviewed** — one row per week with the model's
   recommended swap, user reaction, retrospective delta, and the
   feature reference.
3. **Feature backlog** — every feature the user mentioned that
   isn't in `CAUSAL_FEATURES`. Tag each with
   (a) whether it's a real data feed (injury report, Vegas, news
   API) or a derived signal (streak, matchup-specific pattern),
   (b) hypothesized impact (high / medium / low), (c) acquisition
   cost (hours to days).
4. **Product constraints** — format/scoring/roster-size/bye-week
   handling / multi-league support the tool must accept on v1.
5. **User verdict** — would they use it? At what stake? What
   would make them stop?

Commit the file to main. The writeup is the deliverable.

## Kill criteria for the session itself

- **User disengages within 10 minutes.** The tool is confusing;
  redesign before more sessions.
- **Zero feature references in 30 minutes.** The user is either
  not the right user (not an engaged fantasy player) or the tool
  isn't surfacing anything worth reacting to. Recruit a different
  user before iterating.
- **User cannot articulate a decision the tool changed or confirmed.**
  Chairman's success signal fails. Don't spin up a second session
  — fix the tool's decision surface first.

## What this protocol does NOT claim

- It is not a N=1 decision-quality measurement. One user's week is
  not a statistical signal; it's a product-readiness signal.
- It is not a pricing study. Willingness-to-pay comes after
  product-market-fit conversations, not in the first session.
- It is not a UX review. The v1 CLI is the thinnest possible
  surface; the full UI conversation comes after the first real
  user sees real output.

## Tool quick-reference

```bash
# Run a paper-prototype session
python scripts/start_sit_prototype.py \
    --roster docs/roster_template.json \
    --season 2024 --week 10

# Force a specific predictions CSV (not the auto-latest)
python scripts/start_sit_prototype.py \
    --roster /tmp/my_roster.json \
    --season 2024 --week 10 \
    --predictions-csv data/backtest_results/ts_backtest_2024_20260421_034959_predictions.csv
```

The tool matches roster names against the walk-forward predictions
CSV. Unmatched entries (the user's spelling didn't match) are
surfaced loudly before the decision table — this is intentional,
silent auto-matching hides errors in a paper prototype.
