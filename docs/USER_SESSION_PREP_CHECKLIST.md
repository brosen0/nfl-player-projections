# User-session prep checklist (Step 5)

**Use alongside** `docs/USER_INTERVIEW_PROTOCOL.md` (the long-form
protocol). This is the operator's pre-session checklist — do these
items in order and the session itself becomes a 30-minute
conversation, not a debug session.

## Day-before / morning-of (15 min)

### 1. Send the user this in advance, in plain text

> Hi! Quick prep for our 30 min:
>
> **Pick 1-3 weeks from 2024 or 2025 you remember well** — a pivotal
> start/sit, a regret, a lucky call. For each chosen week, write
> down:
> 1. Your full 15-player roster (just names + positions: QB, RB, WR,
>    TE — no K or DST for v1).
> 2. Which 6 you actually started: 1 QB, 2 RB, 2 WR, 1 TE.
> 3. The format: PPR, half-PPR, or standard scoring? Roster size?
>    League size (8 / 10 / 12)?
> 4. Optional: any specific decision you wrestled with that week.
>
> The tool only supports **PPR, 12-team, no FLEX** in v1 — if your
> league is different, just tell me; we'll capture the format gap
> rather than try to match.

Two outcomes both close the chairman's success gate:
- "The model would have flipped my start at RB2; X would have outscored Y" → confirmation
- "The model's swap is wrong because [injury / weather / Vegas / news]" → feature gap

Either is a win.

### 2. Get the user's roster JSON

Once they reply, copy the template:

```bash
cp docs/roster_template.json /tmp/user_roster_w<X>.json
```

Edit `/tmp/user_roster_w<X>.json` with the user's names. Use either
format:
- `"C.McCaffrey"` (FirstInitial.Last) — what walk-forward CSVs use
- `"Christian McCaffrey"` (full name) — what most users will write

The matcher handles both. The recent fix
(`scripts/start_sit_prototype.py`, commit pending) disambiguates
same-last-name collisions like B.Robinson vs K.Robinson by using
first-initial.

### 3. Sanity-check the JSON loads

```bash
python scripts/start_sit_prototype.py \
    --roster /tmp/user_roster_w<X>.json \
    --season 2024 --week 10
```

Check the output for **UNMATCHED** rows. Common causes:
- User wrote "Aiyuk" but CSV uses "B.Aiyuk" → try both formats
- User picked a 2024 rookie not in the prior-season pool → expected, surface to user
- User's spelling has a typo

If unmatched, fix the JSON together with the user during the
session — that itself is a useful 2-min teaching moment.

### 4. Verify the predictions CSVs are post-fix

The auto-loader picks the latest CSV. As of commit `7ffd54c` the
defaults are:
- 2024: `ts_backtest_2024_20260424_165248_predictions.csv`
- 2025: `ts_backtest_2025_20260424_165230_predictions.csv`

Both include the share-feature fix, `prev_season_ppg`, and rookie
priors. Don't override unless the user wants to test against a
specific older CSV.

## During the session (30 min)

### Structured capture — first 10 min

For each of the user's chosen weeks, run the tool live and capture:

- **What the model recommends:** swap in player X for player Y
  (printed in the tool's "Best lineup" section)
- **User's literal reaction:** quote them exactly. "Yes, I knew it"
  / "No, I trust X" / "What?"
- **Retrospective delta:** the tool prints actual model-lineup vs
  user-lineup totals. Note the sign and magnitude.
- **Which feature would have changed it for them:** "the Thursday
  practice report said …", "the Vegas line moved …", "the beat
  reporter said …". These are feature candidates. Write them
  down verbatim.

### Open-ended — next 15 min

Five questions, in order:

1. What info do you use in your real start/sit decisions that
   the tool didn't show?
2. If this tool was your only source for a week, would you trust
   it? What's your override threshold?
3. What would make you stop using it entirely?
4. What's your league's scoring (PPR/half/std)? Roster size?
   FLEX rules?
5. Single decision you care about most: start/sit, draft, waiver,
   or trade? Which one would change your week if the tool got it
   right?

### Close — last 5 min

Read back the feature-candidates list and ask: "Did I get all of
them? Anything missing?" Then ask the willingness-to-commit signal:
"If I emailed you Sunday-morning recs for 4 weeks, would you use
them?"

## After the session (15 min)

Write up `docs/USER_SESSION_<YYYYMMDD>_<handle>.md` with these
exact sections (per the protocol):

1. **Header** — date, user handle, league format, scoring, roster
   size.
2. **Decisions reviewed** — table: week / model swap / user
   reaction / retrospective delta / feature reference.
3. **Feature backlog** — every feature the user mentioned that
   isn't in `CAUSAL_FEATURES`. Tag each: (a) data feed (injury /
   Vegas / news API) vs derived (streak, matchup pattern); (b)
   hypothesized impact (high / med / low); (c) acquisition cost
   (hours / days).
4. **Product constraints** — format / scoring / roster size /
   bye-week handling / multi-league support v1 must accept.
5. **User verdict** — would they use it? At what stake? Stop
   condition?

Commit to `main`. The writeup is the deliverable.

## Kill criteria for the session itself

(From the protocol — re-iterating because they matter)

- **User disengages within 10 min** → tool is confusing, redesign
  before more sessions.
- **Zero feature references in 30 min** → user isn't engaged
  enough or tool isn't surfacing anything reactable. Recruit
  someone else before iterating.
- **User can't articulate a decision the tool changed or
  confirmed** → fix the decision surface before more sessions.

## What I (Claude) need to do post-session

When you bring back the session output, I'll:
1. Read your raw notes / quotes.
2. Help draft `docs/USER_SESSION_<date>_<handle>.md` per the
   protocol structure.
3. If specific feature candidates are tractable (e.g., "Vegas line
   movement" → we already have schedule/Vegas tables, just need to
   compute), scope the next iteration.
4. If the user has a non-PPR format, scope a v2 of the tool with
   their scoring rules.

## Last sanity check before the session starts

```bash
# All tests green
python -m pytest tests/test_start_sit_prototype.py -q
# 7 passed expected.

# Tool can run end-to-end
python scripts/start_sit_prototype.py \
    --roster docs/roster_template.json \
    --season 2024 --week 10
# Should print starters, bench, swap recommendation, retrospective delta.
```

If both pass: ready.
