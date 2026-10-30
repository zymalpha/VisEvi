"""Independent boundary cases for raster geometry definitions."""
import numpy as np
import pytest

from visevi.perception import geometry as geo


def test_nested_regions_have_zero_set_distance_and_positive_boundary_distance():
    outer = np.zeros((9, 9), bool)
    inner = np.zeros_like(outer)
    outer[1:8, 1:8] = True
    inner[3:6, 3:6] = True
    result = geo.relate_masks(outer, inner)
    assert result['a_covers_b']
    assert result['gap_distance_px'] == 0
    assert result['boundary_distance_px'] == 2
    assert result['pixel_center_distance_px'] == 0


def test_holes_are_retained_in_area_perimeter_and_contours():
    mask = np.ones((5, 5), bool)
    mask[1:4, 1:4] = False
    result = geo.describe_shape(mask)
    assert result['largest_component']['area_px'] == 16
    assert result['largest_component']['perimeter_px'] == 32
    assert len(result['components'][0]['contours_pixel_edges'][0]['holes_xy']) == 1


def test_reflection_keeps_outside_pixels_in_dice_denominator():
    right = np.zeros((3, 8), bool)
    left = np.zeros_like(right)
    right[0, 0] = right[0, 7] = True
    left[0, 2] = True
    value = geo.mirror_dice(right, left, 1)
    assert value['mirror_dice'] == pytest.approx(2 / 3)
    assert value['reflected_outside_frame_px'] == 1


def test_fractional_midline_requires_an_explicit_resampling_choice():
    mask = np.ones((2, 3), bool)
    with pytest.raises(ValueError, match='half-integer'):
        geo.mirror_dice(mask, mask, 0.25)


@pytest.mark.parametrize('bad', [np.zeros((0, 2)), np.full((2, 2), 0.2), np.array([[np.nan]])])
def test_bad_masks_are_rejected(bad):
    with pytest.raises(ValueError):
        geo.area(bad)


def test_overlapping_lung_sides_do_not_double_count_regions():
    mask = np.ones((3, 3), bool)
    with pytest.raises(ValueError, match='overlap'):
        geo.assign_lung_zones(mask, mask, mask)


def test_zone_partition_uses_pixel_centers_for_short_lungs():
    right = np.zeros((2, 5), bool)
    left = np.zeros_like(right)
    right[:, 0] = True
    left[:, 4] = True
    value = geo.assign_lung_zones(right, left, right)
    assert value['right']['zone_overlap_px'] == {'upper': 1, 'middle': 0, 'lower': 1}


def test_duplicate_candidates_are_visible_but_union_burden_is_not_duplicated():
    right = np.zeros((4, 8), bool)
    left = np.zeros_like(right)
    right[:, :3] = True
    left[:, 5:] = True
    region = np.zeros_like(right)
    region[1, 1] = True
    value = geo.summarize_bilateral_distribution(right, left, [region, region, np.zeros_like(right)])
    assert value['component_count'] == 2
    assert value['union_region_area_px'] == 1
    assert value['empty_region_indices'] == [2]
