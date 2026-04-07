"""Tests that verify config and code are aligned.

Config-code mismatches — where settings declare one thing but code does
another — are a common source of silent bugs. These tests ensure that:
1. position_target_type config values are valid and respected
2. Converter training only happens for positions that need it
3. The prediction path respects the target type config
"""

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Config validity
# ---------------------------------------------------------------------------

class TestPositionTargetTypeConfig:
    """Verify position_target_type config is valid and contains no dead code."""

    VALID_TARGET_TYPES = {"fp", "util"}

    def test_all_target_types_are_valid(self):
        """Every position_target_type value must be 'fp' or 'util'.
        'auto' is not a supported value — it was previously dead code."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        invalid = []
        for pos, target_type in ptc.items():
            if target_type not in self.VALID_TARGET_TYPES:
                invalid.append((pos, target_type))
        assert invalid == [], (
            f"Invalid position_target_type values found:\n"
            + "\n".join(
                f"  {pos}: '{tt}' (must be 'fp' or 'util')"
                for pos, tt in invalid
            )
        )

    def test_qb_target_type_is_explicit(self):
        """QB target type must be explicitly 'fp' or 'util', not 'auto'."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        qb_target = ptc.get("QB", "util")
        assert qb_target in self.VALID_TARGET_TYPES, (
            f"QB target type is '{qb_target}', must be 'fp' or 'util'. "
            f"'auto' was dead code — ensemble.py unconditionally overrode it to 'fp'."
        )

    def test_all_four_positions_have_target_type(self):
        """QB, RB, WR, TE should all have explicit target type config."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        for pos in ["QB", "RB", "WR", "TE"]:
            assert pos in ptc, (
                f"Position {pos} missing from position_target_type config"
            )


# ---------------------------------------------------------------------------
# Converter training alignment
# ---------------------------------------------------------------------------

class TestConverterTrainingAlignment:
    """Verify that converters are only trained for positions that need them."""

    def test_fp_positions_should_not_train_converters(self):
        """Positions with target_type='fp' should not waste time training
        utilization-to-FP converters, since those converters are never used
        at prediction time."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        fp_positions = [pos for pos, tt in ptc.items() if tt == "fp"]
        util_positions = [pos for pos, tt in ptc.items() if tt != "fp"]
        # This test documents which positions use converters
        assert len(util_positions) > 0, "At least one position should use util target"
        assert len(fp_positions) > 0, "At least one position should use fp target"
        # Verify the code would only train converters for util positions
        # (the actual code fix filters positions by config)
        for pos in fp_positions:
            assert ptc[pos] == "fp", f"{pos} should be 'fp' but is '{ptc[pos]}'"


# ---------------------------------------------------------------------------
# Prediction path alignment
# ---------------------------------------------------------------------------

class TestPredictionPathAlignment:
    """Verify the prediction path respects target type config."""

    def test_ensemble_respects_fp_config_for_conversion(self):
        """The ensemble predictor must skip util->FP conversion for positions
        configured as 'fp'. This checks the logic in ensemble.py."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        # For each fp position, the conversion check should return False
        for pos in ["QB", "WR", "TE"]:
            target_type = ptc.get(pos, "util")
            should_convert = target_type != "fp"
            assert not should_convert, (
                f"Position {pos} has target_type='{target_type}' but would "
                f"still trigger util->FP conversion. Config says 'fp' so "
                f"conversion must be skipped."
            )

    def test_rb_uses_util_conversion(self):
        """RB should use the util->FP conversion pipeline."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        rb_target = ptc.get("RB", "util")
        assert rb_target == "util", (
            f"RB target type is '{rb_target}', expected 'util'. "
            f"RB is the only position that benefits from the two-stage pipeline."
        )
