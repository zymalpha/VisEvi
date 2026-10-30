"""Deterministic image appearance measurements (A15–A17)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..geometry import _mask


def load_grayscale(path: str | Path) -> tuple[np.ndarray, dict]:
    """Decode a raster image, preserving native grayscale precision."""
    with Image.open(path) as image:
        mode = image.mode
        if image.getexif().get(0x0112, 1) != 1:  # EXIF orientation
            raise ValueError("EXIF orientation is not identity; transpose the image explicitly first")
        if mode in ("1", "L", "I", "F") or mode.startswith("I;16"):
            pixels = np.asarray(image).copy()
            conversion = "native grayscale samples"
        else:
            pixels = np.asarray(image.convert("L"))
            conversion = "Pillow grayscale conversion after palette/color decoding"
    return _grayscale(pixels), {
        "source_mode": mode, "source_dtype": str(pixels.dtype),
        "conversion": conversion, "windowing": "none",
    }


def _grayscale(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise ValueError("image must contain real numeric grayscale values")
    if np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError("image samples must be finite real numbers")
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    elif value.ndim == 3 and value.shape[2] in (3, 4):
        value = np.dot(value[:, :, :3].astype(np.float64), [0.299, 0.587, 0.114])
    if value.ndim != 2:
        raise ValueError("image must be a 2D grayscale or HxWx3/4 numeric array")
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("image must be non-empty and contain only finite values")
    return value.astype(np.float64, copy=False)


def _roi_mask(shape: tuple[int, int], roi: np.ndarray | Mapping | Sequence | None) -> np.ndarray:
    if roi is None:
        return np.ones(shape, dtype=bool)
    if isinstance(roi, Mapping):
        try:
            coords = tuple(roi[key] for key in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ROI mapping must contain integer x0, y0, x1, y1") from exc
    elif isinstance(roi, Sequence) and not isinstance(roi, (str, bytes)):
        if len(roi) != 4:
            raise ValueError("ROI coordinates must be (x0, y0, x1, y1), inclusive")
        coords = tuple(roi)
    else:
        mask = np.asarray(roi)
        if mask.ndim != 2 or mask.shape != shape:
            raise ValueError("ROI mask must be 2D and match the image shape")
        result = _mask(mask, "ROI mask")
        if not result.any():
            raise ValueError("ROI must contain at least one pixel")
        return result
    try:
        if any(isinstance(value, (str, bool)) or not np.isfinite(value)
               or int(value) != value for value in coords):
            raise ValueError("ROI coordinates must be finite integers")
        x0, y0, x1, y1 = map(int, coords)
    except (TypeError, OverflowError) as exc:
        raise ValueError("ROI coordinates must be finite integers") from exc
    height, width = shape
    if x0 < 0 or y0 < 0 or x1 < x0 or y1 < y0 or x1 >= width or y1 >= height:
        raise ValueError("inclusive ROI coordinates must be ordered and inside the image")
    result = np.zeros(shape, dtype=bool)
    result[y0:y1 + 1, x0:x1 + 1] = True
    return result


def summarize_intensity(image: np.ndarray,
                        roi: np.ndarray | Mapping | Sequence | None = None) -> dict:
    """A15: raw grayscale distribution in a mask, inclusive bbox, or full image."""
    gray = _grayscale(image)
    selected = gray[_roi_mask(gray.shape, roi)]
    quantiles = np.quantile(selected, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "pixel_count": int(selected.size),
        "minimum": float(selected.min()),
        "maximum": float(selected.max()),
        "mean": float(selected.mean()),
        "std_population": float(selected.std(ddof=0)),
        "variance_population": float(selected.var(ddof=0)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "intensity_units": "input grayscale units; no windowing or normalization applied",
    }


def measure_tenengrad(image: np.ndarray,
                      roi: np.ndarray | Mapping | Sequence | None = None,
                      intensity_scale: float = 1.0) -> dict:
    """A17: mean squared Sobel gradient (Tenengrad sharpness proxy).

    Gradients are computed on the full image, then averaged over the ROI to
    avoid treating the crop boundary as an image edge. The optional scale
    divides image intensities before the gradient calculation.
    """
    gray = _grayscale(image)
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("intensity_scale must be finite and positive")
    selected = _roi_mask(gray.shape, roi)
    normalized = gray / float(intensity_scale)
    gx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3,
                   scale=1 / 8, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=3,
                   scale=1 / 8, borderType=cv2.BORDER_REFLECT)
    energy = gx * gx + gy * gy
    return {
        "tenengrad_mean": float(energy[selected].mean()),
        "gradient_magnitude_mean": float(np.sqrt(energy[selected]).mean()),
        "pixel_count": int(selected.sum()),
        "method": "mean(Sobel_x^2 + Sobel_y^2), 3x3 kernel, scale=1/8",
        "intensity_scale": float(intensity_scale),
        "interpretation": "relative sharpness proxy; not a calibrated quality grade",
    }


def extract_texture_features(image: np.ndarray, roi: np.ndarray | Mapping | Sequence,
                             bin_width: float = 16.0) -> dict:
    """A16: sparse 2D GLCM features with fixed-width grayscale quantization."""
    gray = _grayscale(image)
    selected = _roi_mask(gray.shape, roi)
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    quantized_float = np.floor(gray / float(bin_width))
    limit = np.iinfo(np.int64).max // 2
    if (not np.isfinite(quantized_float).all()
            or quantized_float.min() < -limit or quantized_float.max() > limit):
        raise ValueError("quantized grayscale values exceed supported integer range")
    quantized = quantized_float.astype(np.int64)
    offsets = {
        "horizontal": (0, 1),
        "diagonal_up_right": (-1, 1),
        "vertical_up": (-1, 0),
        "diagonal_up_left": (-1, -1),
    }
    directional = {}
    for name, (dy, dx) in offsets.items():
        y0, y1 = max(0, -dy), min(gray.shape[0], gray.shape[0] - dy)
        x0, x1 = max(0, -dx), min(gray.shape[1], gray.shape[1] - dx)
        source = quantized[y0:y1, x0:x1]
        target = quantized[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        valid = selected[y0:y1, x0:x1] & selected[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        pairs = np.stack((source[valid], target[valid]), axis=1)
        if not len(pairs):
            directional[name] = {"valid_pair_count": 0, "features": None}
            continue
        pairs = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
        gray_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        probability = counts.astype(np.float64) / counts.sum()
        first = gray_pairs[:, 0].astype(np.float64)
        second = gray_pairs[:, 1].astype(np.float64)
        delta = first - second
        mean_first = float(np.sum(probability * first))
        mean_second = float(np.sum(probability * second))
        std_first = float(np.sqrt(np.sum(probability * (first - mean_first) ** 2)))
        std_second = float(np.sqrt(np.sum(probability * (second - mean_second) ** 2)))
        correlation = (float(np.sum(probability * (first - mean_first) * (second - mean_second))
                              / (std_first * std_second))
                       if std_first and std_second else 1.0)
        asm = float(np.sum(probability ** 2))
        directional[name] = {
            "valid_pair_count": int(len(pairs) // 2),
            "features": {
                "contrast": float(np.sum(probability * delta ** 2)),
                "dissimilarity": float(np.sum(probability * np.abs(delta))),
                "homogeneity": float(np.sum(probability / (1 + delta ** 2))),
                "angular_second_moment": asm,
                "energy": float(np.sqrt(asm)),
                "correlation": correlation,
            },
        }
    feature_names = ("contrast", "dissimilarity", "homogeneity",
                     "angular_second_moment", "energy", "correlation")
    valid_directions = [item["features"] for item in directional.values()
                        if item["features"] is not None]
    if not valid_directions:
        raise ValueError("ROI has no neighboring pixel pairs for 2D texture features")
    features = {name: float(np.mean([direction[name] for direction in valid_directions]))
                for name in feature_names}
    config = {"version": 1, "bin_width": float(bin_width), "origin": 0,
              "offsets": offsets, "symmetric": True,
              "aggregation": "unweighted mean of nonempty directions",
              "constant_correlation": 1.0}
    return {
        "features": features,
        "directional_features": directional,
        "grayscale_bin_width": float(bin_width),
        "quantization_origin": 0.0,
        "quantization": "floor(input_grayscale / bin_width)",
        "directions": offsets,
        "matrix_symmetry": "each neighboring pair is included in both orders",
        "feature_aggregation": "unweighted mean across directions with at least one valid pair",
        "constant_correlation_convention": 1.0,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "intensity_units": "input grayscale units; no windowing or normalization applied",
    }
