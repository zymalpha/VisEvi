"""Hand-checkable tests for deterministic appearance features."""
import numpy as np
import pytest
import cv2

from visevi.perception.appearance import (measure_tenengrad,
                                         summarize_intensity,
                                         extract_texture_features)


def test_intensity_statistics_use_exact_mask_pixels():
    image = np.array([[0, 10, 20], [30, 40, 50]], dtype=np.uint8)
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    result = summarize_intensity(image, mask)
    assert result["pixel_count"] == 3
    assert result["mean"] == 20
    assert result["std_population"] == pytest.approx(np.sqrt(800 / 3))


def test_intensity_statistics_support_inclusive_bbox():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    result = summarize_intensity(image, (1, 1, 2, 2))
    assert result["pixel_count"] == 4
    assert result["mean"] == 7.5


def test_tenengrad_is_scale_normalized_and_flat_image_has_zero_energy():
    image = np.zeros((32, 32), dtype=np.float32)
    image[:, 16:] = 100
    scaled = image * 2
    result = measure_tenengrad(image, intensity_scale=100)
    result_scaled = measure_tenengrad(scaled, intensity_scale=200)
    assert result["tenengrad_mean"] > 0
    assert result["tenengrad_mean"] == pytest.approx(result_scaled["tenengrad_mean"])
    assert measure_tenengrad(np.ones((8, 8), dtype=np.uint8))["tenengrad_mean"] == 0


def test_blurring_reduces_tenengrad_for_a_step_edge():
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 32:] = 1
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    assert measure_tenengrad(blurred)["tenengrad_mean"] < measure_tenengrad(image)["tenengrad_mean"]


def test_empty_roi_is_rejected():
    with pytest.raises(ValueError):
        summarize_intensity(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))


def test_glcm_features_match_a_constant_roi():
    image = np.full((3, 3), 32, dtype=np.uint8)
    result = extract_texture_features(image, np.ones((3, 3), dtype=bool), bin_width=16)
    for feature in ("contrast", "dissimilarity"):
        assert result["features"][feature] == 0
    assert result["features"]["homogeneity"] == 1
    assert result["features"]["angular_second_moment"] == 1
    assert result["features"]["correlation"] == 1
    assert result["directional_features"]["horizontal"]["valid_pair_count"] == 6


def test_glcm_features_match_an_alternating_horizontal_pattern():
    image = np.array([[0, 16], [0, 16]], dtype=np.uint8)
    result = extract_texture_features(image, np.ones((2, 2), dtype=bool), bin_width=16)
    horizontal = result["directional_features"]["horizontal"]["features"]
    assert horizontal["contrast"] == 1
    assert horizontal["dissimilarity"] == 1
    assert horizontal["homogeneity"] == 0.5
    assert horizontal["angular_second_moment"] == 0.5
    assert horizontal["correlation"] == -1


def test_glcm_rejects_roi_without_neighbor_pairs():
    image = np.array([[0, 16], [32, 48]], dtype=np.uint8)
    roi = np.array([[1, 0], [0, 0]], dtype=bool)
    with pytest.raises(ValueError, match="neighboring pixel pairs"):
        extract_texture_features(image, roi)


def test_glcm_agrees_with_independent_scikit_image_reference():
    feature = pytest.importorskip('skimage.feature')
    image = np.array([[0, 1, 1, 2], [1, 3, 2, 0], [2, 1, 0, 3]], dtype=np.uint8)
    reference = feature.graycomatrix(image, [1], [0, -np.pi/4, -np.pi/2, -3*np.pi/4],
                                    levels=4, symmetric=True, normed=True)
    value = extract_texture_features(image, np.ones_like(image, dtype=bool), bin_width=1)
    names = {'contrast': 'contrast', 'dissimilarity': 'dissimilarity',
             'homogeneity': 'homogeneity', 'angular_second_moment': 'ASM',
             'energy': 'energy', 'correlation': 'correlation'}
    for actual, ref in names.items():
        assert value['features'][actual] == pytest.approx(feature.graycoprops(reference, ref).mean())


def test_roi_bbox_does_not_truncate_fractional_coordinates():
    with pytest.raises(ValueError, match='integers'):
        summarize_intensity(np.zeros((4, 4)), (0.5, 0, 2, 2))


def test_one_row_image_has_a_nonzero_horizontal_gradient():
    assert measure_tenengrad(np.array([[0, 10, 20]], dtype=float))['tenengrad_mean'] > 0
