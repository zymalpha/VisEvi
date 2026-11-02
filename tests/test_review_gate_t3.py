"""Regression checks for function routing and conservative T3 geometry."""
import numpy as np
import pytest

from visevi.functions.device_programs import _crossing_state, _point
from visevi.vqa.gate import RuleFunctionGate


def test_explicit_unknown_id_cannot_be_overridden_by_alias():
    route = RuleFunctionGate().route("Q99 projection")
    assert route.status == "unknown"
    assert route.matched_by == "explicit_id"


def test_explicit_function_id_precedes_conflicting_alias():
    route = RuleFunctionGate().route("P9 projection")
    assert route.status == "matched"
    assert route.selected_function_ids == ("P9",)
    assert route.matched_by == "explicit_id"


def test_ascii_alias_uses_word_boundaries():
    route = RuleFunctionGate().route("electrocardiogram review")
    assert route.status == "unknown"
    assert route.selected_function_ids == ()


def test_alias_and_catalogue_candidates_are_merged_before_selection():
    route = RuleFunctionGate().route("projection and CTR")
    assert route.status == "ambiguous"
    assert {candidate.function_id for candidate in route.candidates} == {"Q4", "P9"}


def test_t3_disconnected_fragments_on_opposite_sides_are_indeterminate():
    line = np.zeros((20, 20), dtype=bool)
    line[4, 8] = True
    line[16, 8] = True
    assert _crossing_state(line, [[0.0, 10.0], [19.0, 10.0]]) == \
        "indeterminate_discontinuous_line"


def test_t3_connected_adjacent_pixels_crossing_curve_are_detected():
    line = np.zeros((20, 20), dtype=bool)
    line[9, 8:12] = True
    line[10, 8:12] = True
    assert _crossing_state(line, [[0.0, 9.5], [19.0, 9.5]]) == "crosses"


def test_t3_connected_line_staying_on_one_side_does_not_cross():
    line = np.zeros((20, 20), dtype=bool)
    line[5, 8:12] = True
    assert _crossing_state(line, [[0.0, 10.0], [19.0, 10.0]]) == "does_not_cross"


def test_point_coordinates_use_pixel_centers():
    assert _point([19, 17], "point", width=20, height=18) == [19.0, 17.0]
    with pytest.raises(ValueError, match="outside"):
        _point([20, 17], "point", width=20, height=18)
    with pytest.raises(ValueError, match="outside"):
        _point([19, 18], "point", width=20, height=18)
