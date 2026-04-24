"""Smoke tests for scripts/snake_draft_sim.py — Step 2 of the
2026-04-23 council's Critical Next Steps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "snake_draft_sim",
    PROJECT_ROOT / "scripts" / "snake_draft_sim.py",
)
sd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sd  # dataclass needs the module resolvable
spec.loader.exec_module(sd)


def _player(name, pos, ecr=100.0, pred=0.0, actual=0.0, modelable=True):
    return sd.DraftPlayer(
        name=name,
        position=pos,
        team="X",
        ecr=float(ecr),
        pred_total=float(pred),
        actual_total=float(actual),
        is_modelable=modelable,
        model_rank_value=float(pred),
    )


def test_snake_pick_order_alternates_per_round():
    order = sd.snake_pick_order(teams=4, rounds=3)
    assert order == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]


def test_model_bot_prefers_projected_over_adp():
    board = [
        _player("QB_adp_favorite", "QB", ecr=1, pred=200, actual=0),
        _player("QB_model_favorite", "QB", ecr=30, pred=400, actual=0),
        _player("RB_adp_favorite", "RB", ecr=2, pred=150, actual=0),
        _player("WR_adp_favorite", "WR", ecr=3, pred=150, actual=0),
    ]
    t = sd.Team(name="M", is_model_bot=True, slot=1)
    # Modeled QB2 has higher pred_total; ADP favorite is "QB1".
    # rank_key_for returns -pred_total for modelable players, so the
    # model bot should prefer the high-pred QB.
    pick = sd._select_pick(t, board)
    assert pick.name == "QB_model_favorite"


def test_adp_bot_prefers_lowest_ecr():
    board = [
        _player("A", "QB", ecr=5, pred=500),
        _player("B", "RB", ecr=1, pred=50),
        _player("C", "WR", ecr=3, pred=100),
    ]
    t = sd.Team(name="A", is_model_bot=False, slot=1)
    pick = sd._select_pick(t, board)
    assert pick.name == "B"  # lowest ECR, regardless of pred


def test_position_cap_blocks_over_stack():
    t = sd.Team(name="A", is_model_bot=False, slot=1)
    # Give the team 3 QBs already — at the QB cap.
    for i in range(3):
        t.roster.append(_player(f"QB{i}", "QB", ecr=1 + i))
    board = [
        _player("QB4", "QB", ecr=10),
        _player("RB1", "RB", ecr=100),
    ]
    # Despite QB4 having lower ECR than RB1, QB is capped -> RB1.
    pick = sd._select_pick(t, board)
    assert pick.name == "RB1"


def test_score_team_picks_optimal_starters():
    t = sd.Team(name="A", is_model_bot=False, slot=1)
    t.roster = [
        _player("QB1", "QB", actual=300),
        _player("QB2", "QB", actual=100),     # bench
        _player("RB1", "RB", actual=200),
        _player("RB2", "RB", actual=180),
        _player("RB3", "RB", actual=50),      # flex? no — lower than WR3
        _player("WR1", "WR", actual=220),
        _player("WR2", "WR", actual=210),
        _player("WR3", "WR", actual=100),     # flex candidate
        _player("TE1", "TE", actual=150),
    ]
    total, starters = sd.score_team(t, "actual_total")
    names = {p.name for p in starters}
    # Expected starters: QB1 + RB1 + RB2 + WR1 + WR2 + TE1
    # + FLEX = best remaining among RB3 (50), WR3 (100) -> WR3
    assert names == {"QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "WR3"}
    assert total == 300 + 200 + 180 + 220 + 210 + 150 + 100


def test_run_draft_assigns_all_picks_and_model_bot_slot():
    # 12 teams × 15 rounds = 180 picks. Provide 300-player board.
    board = []
    for i in range(80):
        board.append(_player(f"QB{i}", "QB", ecr=i + 1,
                             pred=400 - i, actual=400 - i))
    for i in range(100):
        board.append(_player(f"RB{i}", "RB", ecr=i + 1 + 100,
                             pred=300 - i, actual=300 - i))
    for i in range(100):
        board.append(_player(f"WR{i}", "WR", ecr=i + 1 + 200,
                             pred=250 - i, actual=250 - i))
    for i in range(50):
        board.append(_player(f"TE{i}", "TE", ecr=i + 1 + 400,
                             pred=150 - i, actual=150 - i))
    board.sort(key=lambda p: p.ecr)
    teams = sd.run_draft(board, model_slot=3)
    assert len(teams) == sd.TEAMS
    total_picks = sum(len(t.roster) for t in teams)
    assert total_picks == sd.TEAMS * sd.ROUNDS
    model_team = next(t for t in teams if t.is_model_bot)
    assert model_team.slot == 3
    # Every team respects roster-size invariant.
    assert all(len(t.roster) == sd.ROUNDS for t in teams)


def test_build_draft_board_matches_last_name_and_position():
    import pandas as pd
    adp = pd.DataFrame([
        {"name": "C.McCaffrey", "position": "RB", "team": "SF", "ecr": 1.0,
         "sd": 0.0, "best": 1.0, "worst": 2.0},
        {"name": "T.Hill", "position": "WR", "team": "MIA", "ecr": 5.0,
         "sd": 0.5, "best": 3.0, "worst": 8.0},
        {"name": "X.Ghost", "position": "WR", "team": "FA", "ecr": 300.0,
         "sd": 10.0, "best": 200.0, "worst": 400.0},
    ])
    preds = pd.DataFrame([
        {"player_id": "P1", "name": "Christian McCaffrey", "position": "RB",
         "team": "SF", "pred_total": 350.0, "actual_total": 450.0, "weeks": 17},
        {"player_id": "P2", "name": "Tyreek Hill", "position": "WR",
         "team": "MIA", "pred_total": 280.0, "actual_total": 310.0, "weeks": 17},
    ])
    board = sd.build_draft_board(adp, preds)
    by_name = {p.name: p for p in board}
    assert by_name["C.McCaffrey"].is_modelable
    assert by_name["C.McCaffrey"].pred_total == 350.0
    assert by_name["T.Hill"].is_modelable
    assert by_name["T.Hill"].actual_total == 310.0
    assert not by_name["X.Ghost"].is_modelable
    assert by_name["X.Ghost"].pred_total == 0.0


def test_vorp_subtracts_position_replacement_level():
    """VORP for the top QB = QB1_proj - QB14_proj;
       VORP for the top RB = RB1_proj - RB35_proj.
       Cross-position ordering should favor RB scarcity."""
    import pandas as pd
    rows = []
    # 20 QBs: 30, 29, 28, ..., 11 (QB14 = 30 - 13 = 17)
    for i in range(20):
        rows.append({"player_id": f"QB{i}", "name": f"QB{i}", "position": "QB",
                     "team": "X", "pred_total": 30 - i,
                     "actual_total": 0.0, "weeks": 17})
    # 40 RBs: 25, 24.5, 24, ..., 5.5 (RB35 = 25 - 0.5*34 = 8)
    for i in range(40):
        rows.append({"player_id": f"RB{i}", "name": f"RB{i}", "position": "RB",
                     "team": "X", "pred_total": 25 - 0.5 * i,
                     "actual_total": 0.0, "weeks": 17})
    agg = pd.DataFrame(rows)
    vorp = sd._apply_vorp(agg, basis_col="pred_total")
    # QB1 (proj 30) replacement QB14 (proj 17) -> VORP 13
    assert abs(vorp.iloc[0] - 13.0) < 1e-6, f"QB1 VORP = {vorp.iloc[0]}"
    # RB1 (proj 25) replacement RB35 (proj 8) -> VORP 17
    rb1_idx = agg.index[agg["player_id"] == "RB0"][0]
    assert abs(vorp.iloc[rb1_idx] - 17.0) < 1e-6, f"RB1 VORP = {vorp.iloc[rb1_idx]}"
    # RB has higher VORP than QB despite lower raw projection.
    assert vorp.iloc[rb1_idx] > vorp.iloc[0]


def test_vorp_model_bot_prefers_scarce_position():
    """When two players project the same raw points but one is at a
    scarce position (RB) and the other at deep (QB), VORP should
    push the scarce-position one to the top of ModelBot's queue."""
    qb = _player("QB_rare", "QB", ecr=5, pred=25, modelable=True)
    rb = _player("RB_rare", "RB", ecr=6, pred=25, modelable=True)
    # In VORP mode, model_rank_value is set externally from pred_total
    # minus replacement. Simulate: QB14 projects 17, RB35 projects 8.
    # So QB_rare VORP = 8; RB_rare VORP = 17.
    qb.model_rank_value = 25 - 17  # 8
    rb.model_rank_value = 25 - 8   # 17
    t = sd.Team(name="M", is_model_bot=True, slot=1)
    pick = sd._select_pick(t, [qb, rb])
    assert pick.name == "RB_rare"
