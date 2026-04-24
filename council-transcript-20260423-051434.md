# LLM Council Transcript — 2026-04-23 05:14:34 UTC

**Question:** We are only a few months from season — what are the critical actions to prove this model is ready to give users an edge in (a) the season-long fantasy draft (August) AND (b) the weekly season-long start/sit + lineup decisions?

**Session type:** Bucket 4 partial council — 2 advisors (Contrarian + Executor), 2 peer reviewers (Outsider + First Principles Thinker), 1 chairman synthesis. Partial because prior 2026-04-22 re-council already verdicted on weekly cash-H2H deployment readiness; the delta this council addresses is **draft scope + user-facing product + 4.5-month timeline**.

---

## Chairman's Verdict

### Bottom Line
**Do:** Pin the symmetric walk-forward number and run a draft-sim-vs-ADP backtest on 2024 as the single go/no-go gate before committing any August ship date.
**Don't:** Build user-facing draft or start/sit tooling on top of weekly-H2H projections until calibration, season-long transfer validity, and at least one real user's decision context are established.

### Critical Next Steps
1. **DO TODAY:** Finish the symmetric walk-forward re-run to collapse the [55.8 %, 83.7 %] prospective-replay bound to a point estimate with honest CI. Time: 1–2 days. Success signal: a single win-rate number with bootstrap CI committed to the repo; if p5 < 52.38 %, everything downstream halts. ✓ **done (2026-04-23)** — two-stage result: (a) wide-mode symmetric pinned at **23-18 = 56.10 %, bootstrap p5 = 43.90 %, initially FAIL by 8.5 pp** (commit `020e7c8`, [`docs/SYMMETRIC_WALK_FORWARD_PINNED.md`](../docs/SYMMETRIC_WALK_FORWARD_PINNED.md)). Diagnosis ([`docs/INACTIVE_PICK_GAP_DIAGNOSIS.md`](../docs/INACTIVE_PICK_GAP_DIAGNOSIS.md), commit `26960a2`) found 83.6 % of model inactive picks had no `player_injuries` row — a missing-signal problem, not a model problem. (b) **Active-roster filter fix** ([`docs/ACTIVE_ROSTER_FILTER.md`](../docs/ACTIVE_ROSTER_FILTER.md), commit `83daae9`) — backfilled `weekly_rosters` from nflverse (379,802 rows, 2018–2025), wired `status == 'ACT'` gate into both `scripts/paper_trade_lock.py` and the symmetric-replay scoring. With the filter on, model inactive rate drops 22.4 % → 0.4 %, opp drops 15.0 % → 1.6 %, and **wide-symmetric flips to 29-12 = 70.73 %, bootstrap p5 = 58.54 %, PASS by 6.2 pp**. The 70.73 % filtered number slightly exceeds the narrow active-only hindsight baseline (69.77 %). Chairman's halt does not fire on the filtered (harness-correct) configuration — Steps 2-5 unblocked under the standing condition that every live lock carries the filter.
2. Scrape 2024 + 2025 ADP history (FantasyPros or Sleeper public API) into a flat CSV and stand up a snake-draft simulator that drafts against ADP using your current projections as season-long totals. Time: 3–5 days. Success signal: simulator runs a full 2024 draft end-to-end and outputs roster + projected season points for both your-bot and ADP-bot. ✓ **done (2026-04-24)** — FantasyPros ECR backfilled via the reachable `dynastyprocess/data` GitHub mirror (FantasyPros + Sleeper are both blocked in this sandbox). 97,369 rows across 2024 + 2025 landed in a new `adp_history` table; `scripts/snake_draft_sim.py` runs 12-team × 15-round snake drafts with QB/RB/WR/TE + FLEX slots, ModelBot using summed walk-forward per-week predictions, ADPBots using ECR. 2024 and 2025 drafts both run end-to-end and write JSON to `data/draft_sim_results/`. 7 unit tests landed (`tests/test_snake_draft_sim.py`). **Important caveat:** v1's ModelBot uses in-season walk-forward aggregation, so the ModelBot-wins-both-seasons result is a structural check, not a draft-time lift measurement — Step 3 is the proper hindsight gate.
3. Run the draft-sim-vs-ADP hindsight backtest on 2024 (and 2025 where data permits). Time: 2–3 days. Success signal: your bot beats ADP-bot by a statistically meaningful margin on realized 2024 season totals across N ≥ 200 simulated league configurations; if it loses or ties, kill the draft product and redirect those months to start/sit only. ✗ **NOT RUN — kill criterion fired upstream (2026-04-24)** — single-draw matcher-fixed pre-draft sim loses to ADP 2-for-2 (`week1` mode: −9.0 % 2024, −15.2 % 2025; commit `242d85f`). N ≥ 200 cannot flip a structural 2-for-2 directional loss; the gap is missing pre-draft features (offseason news, camp signals, depth charts), not noise. Per this transcript's own kill criterion below, August 2026 draft product is shelved. See `docs/PHASE_4B_VORP_FINDINGS.md`.
4. Add calibration + uncertainty to the projection layer: per-player prediction intervals, injury-adjusted variance, and a positional-replacement (VORP-style) score. Time: 1–2 weeks. Success signal: 80 % intervals contain ~80 % of 2024 + 2025 weekly actuals on a held-out slice; draft-sim using VORP-ranked picks beats the naive-sum version from step 2. **PARTIAL (2026-04-24)** — Phase 4C rookie priors landed (`docs/PHASE_4C_ROOKIE_PRIORS_FINDINGS.md`, commit `0eba000`); rookie R² +0.011 each season, draft rank unchanged. Phase 4B VORP landed (`docs/PHASE_4B_VORP_FINDINGS.md`, commit `242d85f`); +11.7 % 2024 / −4.0 % 2025 vs season_sum, mixed on a single draw. Phase 4A (conformal intervals) and 4D (injury variance) **kept in scope as in-season start/sit improvements** since they're not draft-specific. Draft-sim VORP gate is moot now — see Step 3 above.
5. Recruit ONE real user (a fantasy player in a specific league with a specific scoring format) and run a paper prototype of the start/sit tool against their actual 2024/2025 roster weeks before touching UI. Time: 1 week elapsed, ~4 hours of work. Success signal: user can articulate a decision the tool changed or confirmed, and names the format/scoring/league-size constraints you must support.

### Council Convergence
- The weekly cash-H2H edge does NOT automatically transfer to draft or start/sit; assuming it does is the load-bearing unproven claim.
- ADP data is a hard prerequisite for any draft product and is cheap to acquire; there is no reason it isn't already in the repo.

### Council Disagreement
**The real tradeoff:** ship-by-August momentum vs. prove-the-premise discipline.
**Side A (The Executor):** Treat draft and start/sit as separable shipping tracks with concrete weekly gates, starting from ADP ingestion Monday. The draft-sim-vs-ADP result in week 2 is itself the reality check — build the gate, don't debate it.
**Side B (The Contrarian):** The projections stack was built and validated for one contest type; extending it to drafts and start/sit without calibration, variance modeling, or replacement value produces a confidently-wrong tool users will trust. Ship nothing user-facing until the symmetric re-run pins the win rate and uncertainty is first-class in the model.
**Chairman's call:** Side B, because the Executor's own gate (step 3 above) only works if the projections it consumes are honest — and the Contrarian correctly flags that summed weekly means are the wrong object for season-long ranking. Sequence the Contrarian's discipline, then use the Executor's gates.

### Blind Spots Caught in Review
Both advisors skipped the user. Before August, define: (1) target user persona — redraft, best-ball, DFS cash, or DFS GPP (pick one, not all); (2) league format and scoring you will support on v1 (PPR vs. half vs. standard, roster size, bye handling); (3) the single decision the tool is meant to change (who to pick at pick 47? who to start at FLEX week 6?); (4) an explicit kill criterion for the product itself — if the symmetric re-run lands at ~52 % or the draft-sim ties ADP, what ships, what dies, and what do you tell the one real user you recruited. Put these four answers in writing this week.

### Kill Criteria
- Symmetric walk-forward re-run lands below 55 % win rate or bootstrap p5 < 52.38 %: halt user-facing work, return to model-level diagnosis. **NOT FIRED** — post-audit symmetric re-run lands at 75.6 % (p5=65.85 %), commit `4ee4709`. Start/sit gate cleared.
- Draft-sim-vs-ADP on 2024 fails to beat ADP-only bot by a meaningful margin across ≥ 200 simulated leagues: kill the August draft product, ship start/sit only (or nothing) with no marketing claim about drafts. ✗ **FIRED (2026-04-24, commit `242d85f`)** — pre-draft `week1` ranking mode loses to ADP on BOTH 2024 (−9.0 %) and 2025 (−15.2 %) on the matcher-fixed snake-draft sim. Per the 2026-04-23 council transcript directive ("if it loses or ties, kill the August draft product and redirect those months to start/sit only"), the August 2026 draft product is shelved. Start/sit (75.6 % H2H win rate) becomes the sole user-facing claim. The N≥200 randomization gate (Step 3) was not run because it cannot flip a 2-for-2 directional loss into a win — the gap is structural (no offseason news, camp reports, depth charts in the feature set), not noise. Phase 4A (conformal intervals) and Phase 4D (injury variance) are kept in scope as in-season start/sit improvements. Phase 4B (VORP) and 4C (rookie priors) shipped as base layers but neither closes the pre-draft information gap. Detailed verdict: `docs/PHASE_4B_VORP_FINDINGS.md`.

---

## Original Question

> Council this: we are only a few months of season, what are the critical actions i can take to prove this model is ready to give users an edge in their fantasy football draft and season.

## Framed Question

The model has a validated weekly cash-H2H edge at 69.8 % cross-season hindsight (p = 0.007, ROI +25.6 %) and a committed paper-trade protocol + harness skeleton. Production default is Ridge α=10 000 with Vegas + injury merges live. Phase 4 ensemble killed on runtime. The 2026-04-22 re-council's 5-step deployment sequence (bootstrap, silent-fallback audit, prospective replay, Phase 4 pre-reg, paper-trade spec) landed this week. Season kickoff is ~4.5 months away (~2026-09-10).

**What critical actions should the project take between now and week 1 to prove the model is ready to give users an edge in:**
- (a) the season-long fantasy draft (August), AND
- (b) the weekly season-long start/sit + lineup decisions?

Assume the validated workstream (Ridge weekly H2H) is complete to the sequence's current point; the question is specifically about what's still missing for user-facing product readiness across draft + in-season. Constraints: solo developer, no real-money stakes yet, 2025 + 2024 historical NFL data, no ADP / expert-consensus feeds currently integrated. Do not re-litigate the 2026-04-22 re-council's 5-step sequence.

**Advisor roster (Bucket 4 — 2 advisors):**
- The Contrarian — hunt for what will fail when a lineup-validated model hits a draft-format or user-facing product.
- The Executor — sequence concrete Monday-morning actions against 4.5 months of calendar.

**Peer review roster (Bucket 4 — 2 reviewers):**
- The Outsider
- The First Principles Thinker

---

## Advisor Responses

### The Contrarian

You have validated ONE contest type (weekly cash H2H), and are about to extrapolate that edge to two fundamentally different decision problems. That's the fatal flaw.

**The draft gap is not a gap, it's a chasm.** You have zero season-long projections validated. Weekly walk-forward Ridge predicting next-week points has nothing to say about whether Player X will outscore Player Y across 17 games. Variance, injury priors, age curves, rookie priors, bye-week structure, positional replacement value — none of that is in your stack. You cannot ship a draft tool in August. You can ship a *wrapper around consensus ADP with your name on it*, which is worse than nothing because users will trust it.

**The prospective-replay result should terrify you, not reassure you.** The bound is [55.8 %, 83.7 %]. That's not a confidence interval, that's "we don't know." Until the symmetric walk-forward re-run pins it, every downstream claim to users is built on a number you admit you haven't measured. Ship nothing user-facing until that pin lands.

**"Start/sit" is not weekly-H2H.** In a redraft league the user owns a roster and is choosing between their own options, not picking a portfolio against a generic opponent pool. Your edge metric doesn't transfer. Best-ball and GPP are different games entirely — GPP rewards variance, your mean-projection model actively hurts there.

**What you're avoiding:** calibration. Do your projections have honest intervals? A start/sit UI without uncertainty bands is a liability. Users will anchor on point estimates and blame you for the tail.

**Also missing:** legal/ToS review for any paid product, a data-rights audit on your scraped sources, and a single real user watching a prototype before you commit an August ship date.

### The Executor

Stop treating "draft" and "in-season start/sit" as one project. They ship on different dates with different data.

**Monday (week 1):** Pull the 2024 + 2025 ADP history from FantasyPros or Sleeper's public API into a flat CSV. That's the single blocker. No ADP = no draft product. One afternoon of scraping.

**Weeks 2–4 — Draft backtest skeleton:**
- Convert your weekly Ridge projections into season-long totals by summing 2024 actuals-as-if-predicted. Crude, but it's a baseline.
- Build a snake-draft simulator that drafts against ADP. Score it by final season points vs. an ADP-only bot across the 2024 season you already have on disk. If your model can't beat ADP in hindsight on a season you've already seen, kill the draft product now and save yourself three months.

**Weeks 5–8 — In-season start/sit:** You already have the projections. The product is a ranked table filtered by roster. Build the thinnest possible Streamlit/FastAPI page that ingests a user's roster (CSV paste, not league API integration — that's a rathole) and outputs start/sit for the week. One weekend of work.

**Weeks 9–12:** ADP feed live, roster/schedule ingestion for 2026 when NFL releases it (late April/May), draft tool UI.

**Weeks 13–18:** Mock drafts against the tool. You personally. Every week. If the tool picks players you'd reject, that's the signal.

**Items with no Monday first step — kill or defer:**
- "DFS GPP support." No contest data, no ownership projections, no path. Defer.
- "Best-ball." Different optimization entirely. Defer.
- "Live-data freshness monitoring." No users = no SLA. Defer until someone is paying.

Draft-sim vs. ADP in week 2 is the go/no-go gate. Everything else is downstream.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Executor
- Response B = The Contrarian

### Review 1 — The Outsider

1. **B is stronger.** A hands me a 12-week Gantt chart. B tells me the plan is built on sand. A assumes the weekly edge transfers to draft and start/sit — B names that as the core unproven claim. A ships a tool; B asks if the tool should exist.
2. **A's blind spot: calibration and uncertainty.** A treats "convert weekly projections to season totals by summing" as a baseline. It's not — variance, injury priors, and replacement value are load-bearing for drafts. A also waves at "mock drafts, you personally" as validation. One dev's gut is not a signal.
3. **Both miss the user.** No one asks who this is for, what league format, what scoring, what problem users actually have. Both also skip a kill criterion for the whole product: if the symmetric re-run lands at 52 %, what gets shipped? Neither defines the abort condition.

### Review 2 — The First Principles Thinker

1. **B is strongest.** A answers "how do I ship by week 1." B answers "should I ship by week 1." The validated edge is weekly cash H2H, n = 43 weeks, one contest type. A's draft-sim-vs-ADP gate is clever but uses summed weekly projections as season-long truth — that's the wrong object. B correctly identifies this as a different estimation problem.
2. **A's blind spot: calibration and transfer validity.** A treats the Ridge projections as a trusted primitive and sprints to UI. The whole premise — that a weekly mean-projection model trained for H2H transfers to draft ranking and start/sit — is asserted, not tested. B names this; A builds on top of it.
3. **Both miss the real question: who is the user and what decision are they making?** No user interviews, no JTBD, no willingness-to-pay signal. Shipping any tool in August without one real user in the loop means optimizing a product nobody asked for. The 4.5-month clock is artificial.
