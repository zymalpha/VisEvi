"""Registered deterministic measurement and spatial tools (A9–A17, A19–A23)."""
from __future__ import annotations

from itertools import combinations
import hashlib
import inspect
import json
from numbers import Real
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.spec import ToolSpec, parameter_schema, evidence_result_schema
from ..core.units import EvidenceUnit, ToolRefusal, fov_refusal
from . import appearance, geometry


def _mask(session, name: str, tool: str):
    masks, refusal = _masks(session, [name], tool)
    return (None, refusal) if refusal else (masks[name], None)


def _masks(session, names: list[str], tool: str):
    missing = [f"mask:{name}" for name in names if session.masks.get(name) is None]
    if missing:
        return None, ToolRefusal(tool=tool, reason="required masks are not available",
                                 missing=missing)
    truncated = session.fov_truncated(names)
    if truncated:
        return None, fov_refusal(tool, truncated)
    return {name: geometry._mask(session.masks[name], name) for name in names}, None


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_parameter(value, spec, name):
    expected = spec.get('type')
    types = expected if isinstance(expected, list) else [expected]
    if value is None and 'null' in types:
        return
    if 'string' in types and isinstance(value, str) and value:
        return
    if 'number' in types and isinstance(value, Real) and not isinstance(value, bool):
        if not np.isfinite(value) or ('exclusiveMinimum' in spec and value <= spec['exclusiveMinimum']):
            raise ValueError(f'{name} must be finite and in the declared range')
        return
    if 'array' in types and isinstance(value, (list, tuple, np.ndarray)):
        if len(value) < spec.get('minItems', 0) or len(value) > spec.get('maxItems', float('inf')):
            raise ValueError(f'{name} has an invalid number of items')
        for index, item in enumerate(value):
            _validate_parameter(item, spec['items'], f'{name}[{index}]')
        return
    raise ValueError(f'{name} must match declared type {expected}')


class _RegisteredMetric:
    def __init__(self, name: str, capability: str, description: str,
                 schema: dict, operation: Callable):
        self.name = name
        self.capability = capability
        self.description = description
        self.schema = schema
        self.operation = operation
        source_files = (Path(__file__), Path(geometry.__file__), Path(appearance.__file__))
        self.implementation_sha256 = hashlib.sha256(b''.join(path.read_bytes() for path in source_files)).hexdigest()

    def describe(self) -> ToolSpec:
        return ToolSpec(name=self.name, category="perception",
                        description=f"{self.capability}: {self.description}",
                        input_schema=self.schema,
                        output_schema=evidence_result_schema())

    def run(self, session, **params) -> list[EvidenceUnit] | ToolRefusal:
        try:
            missing = set(self.schema['required']) - params.keys()
            extra = params.keys() - self.schema['properties'].keys()
            if missing or extra:
                raise ValueError(f'invalid parameters: missing={sorted(missing)}, unexpected={sorted(extra)}')
            for name, value in params.items():
                _validate_parameter(value, self.schema['properties'][name], name)
            inspect.signature(self.operation).bind(session, **params)
            value, targets, evidence_type = self.operation(session, **params)
            if not isinstance(value, ToolRefusal):
                value = _json_value(value)
                value['capability'] = self.capability
                value['implementation_sha256'] = self.implementation_sha256
                value['parameters'] = _json_value(params)
                value['input_masks'] = {}
                for target in targets:
                    mask = geometry._mask(session.masks[target], target)
                    value['input_masks'][target] = {
                        'shape_hw': list(mask.shape),
                        'binary_sha256': hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
                        'source': session.mask_metadata.get(target, {}),
                    }
                json.dumps(value, allow_nan=False)
        except (ValueError, TypeError, RuntimeError, OSError, OverflowError) as exc:
            return ToolRefusal(tool=self.name, reason=str(exc))
        if isinstance(value, ToolRefusal):
            value.tool = self.name
            return value
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type=evidence_type, finding_or_anatomy=self.capability,
            value=value, source_tool=self.name,
            provenance=session.mask_provenance(targets),
            validity_conditions=["deterministic computation; input masks/annotations are not validated ground truth"])
        session.add_observation(unit)
        return [unit]


def _array2(minimum: int = 2) -> dict:
    return {"type": "array", "items": {"type": "number"},
            "minItems": minimum, "maxItems": minimum}


def build_metric_tools() -> list:
    """Create one registry interface per deterministic capability."""
    def single_mask(name: str, function: Callable, evidence_type: str,
                    capability: str, description: str, spacing: bool = False):
        properties = {"mask_name": {"type": "string"}}
        required = ("mask_name",)
        if spacing:
            properties["spacing_row_col"] = _array2()
        schema = parameter_schema(properties, required)

        def operation(session, mask_name="Heart", spacing_row_col=None):
            mask, refusal = _mask(session, mask_name, name)
            if refusal:
                return refusal, [], evidence_type
            kwargs = {}
            if spacing:
                kwargs["spacing"] = spacing_row_col
            return function(mask, **kwargs), [mask_name], evidence_type

        return _RegisteredMetric(name, capability, description, schema, operation)

    def area_operation(session, mask_name="Heart", spacing_row_col=None):
        mask, refusal = _mask(session, mask_name, "measure_region_area")
        if refusal:
            return refusal, [], "measurement"
        value = {"area_px": geometry.area(mask), "method": "foreground pixel count"}
        if spacing_row_col is not None:
            value["area_physical"] = geometry.area_physical(mask, spacing_row_col)
            value["spacing_row_col"] = list(map(float, spacing_row_col))
        return value, [mask_name], "measurement"

    def distance_operation(session, mask_a="Heart", mask_b="Right_Lung",
                           spacing_row_col=None):
        masks, refusal = _masks(session, [mask_a, mask_b], "measure_region_distance")
        if refusal:
            return refusal, [], "measurement"
        a, b = masks[mask_a], masks[mask_b]
        physical = spacing_row_col is not None
        return {"gap_distance_px": geometry.gap_distance(a, b),
                "pixel_center_distance_px": geometry.min_distance(a, b),
                "gap_distance_physical": geometry.gap_distance(a, b, spacing_row_col) if physical else None,
                "pixel_center_distance_physical": geometry.min_distance(a, b, spacing_row_col) if physical else None,
                "physical_unit": "caller-supplied spacing unit" if physical else None,
                "spacing_row_col": list(map(float, spacing_row_col)) if physical else None,
                "definition": "gap: minimum distance between pixel-cell regions (0 when touching or overlapping), "
                              "same as A22 gap_distance_px; pixel_center: nearest foreground pixel centres"}, [mask_a, mask_b], "measurement"

    def ctr_operation(session, thoracic_inner_borders_x, heart_borders_x=None):
        targets = []
        source = "explicit caller-supplied heart boundary coordinates"
        if heart_borders_x is None:
            heart, refusal = _mask(session, "Heart", "compute_ctr_from_borders")
            if refusal:
                return refusal, [], "measurement"
            bounds = geometry.bbox(heart)
            heart_borders_x = (bounds["x0"] - 0.5, bounds["x1"] + 0.5)
            targets = ["Heart"]
            source = "outer edges of supplied Heart mask pixel cells"
        value = geometry.ctr_from_borders(heart_borders_x, thoracic_inner_borders_x)
        value["heart_boundary_source"] = source
        value["thoracic_boundary_source"] = "explicit caller-supplied coordinates; localization not performed"
        value["clinical_applicability"] = "not assessed; projection and measurement protocol require separate validation"
        value["implementation_status"] = "A12 partial: thoracic inner border localization is not implemented"
        return value, targets, "measurement"

    def compare_lungs(session):
        masks, refusal = _masks(session, ["Right_Lung", "Left_Lung"], "compare_lung_geometry")
        if refusal:
            return refusal, [], "measurement"
        return geometry.lung_area_compare(masks["Right_Lung"], masks["Left_Lung"]), ["Right_Lung", "Left_Lung"], "measurement"

    def appearance_operation(function, use_roi_mask: bool):
        def operation(session, mask_name=None, intensity_scale=1.0):
            roi = None
            targets = []
            if use_roi_mask and mask_name:
                roi, refusal = _mask(session, mask_name, "appearance_metric")
                if refusal:
                    return refusal, [], "appearance"
                targets = [mask_name]
            gray, decoding = appearance.load_grayscale(session.image_ref)
            if function is appearance.extract_texture_features and roi is None:
                raise ValueError("A16 requires mask_name for a defined ROI")
            if function is appearance.measure_tenengrad:
                value = function(gray, roi=roi, intensity_scale=intensity_scale)
            else:
                value = function(gray, roi=roi)
            value['image_decoding'] = decoding
            value['image_sha256'] = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
            return value, targets, "appearance"
        return operation

    def symmetry_operation(session, midline_x=None):
        masks, refusal = _masks(session, ["Right_Lung", "Left_Lung"], "measure_lung_symmetry")
        if refusal:
            return refusal, [], "measurement"
        midline_source = "caller"
        if midline_x is None:
            union = masks["Right_Lung"] | masks["Left_Lung"]
            bounds = geometry.bbox(union)
            midline_x = (bounds["x0"] + bounds["x1"]) / 2
            midline_source = "lung union bbox"
        value = geometry.mirror_dice(masks["Right_Lung"], masks["Left_Lung"], float(midline_x))
        value["midline_source"] = midline_source
        return value, ["Right_Lung", "Left_Lung"], "measurement"

    def zones_operation(session, region_mask_name):
        names = ["Right_Lung", "Left_Lung", region_mask_name]
        masks, refusal = _masks(session, names, "assign_lung_zones")
        if refusal:
            return refusal, [], "spatial_relation"
        value = geometry.assign_lung_zones(masks["Right_Lung"], masks["Left_Lung"], masks[region_mask_name])
        return value, names, "spatial_relation"

    def lesion_anatomy_operation(session, lesion_mask_name, anatomy_mask_name):
        names = [lesion_mask_name, anatomy_mask_name]
        masks, refusal = _masks(session, names, "relate_lesion_to_anatomy")
        if refusal:
            return refusal, [], "spatial_relation"
        return geometry.relate_lesion_to_anatomy(masks[lesion_mask_name], masks[anatomy_mask_name]), names, "spatial_relation"

    def multi_region_operation(session, region_mask_names):
        if len(region_mask_names) < 2 or len(region_mask_names) != len(set(region_mask_names)):
            raise ValueError("region_mask_names must contain at least two unique names")
        masks, refusal = _masks(session, region_mask_names, "relate_multiple_regions")
        if refusal:
            return refusal, [], "spatial_relation"
        relations = [{"region_a": first, "region_b": second,
                      **geometry.relate_masks(masks[first], masks[second])}
                     for first, second in combinations(region_mask_names, 2)]
        return {"relations": relations, "geometry_convention": "unit-square pixel cells"}, region_mask_names, "spatial_relation"

    def distribution_operation(session, lesion_mask_names):
        names = ["Right_Lung", "Left_Lung", *lesion_mask_names]
        if len(lesion_mask_names) != len(set(lesion_mask_names)):
            raise ValueError("lesion_mask_names must be unique")
        masks, refusal = _masks(session, names, "summarize_bilateral_distribution")
        if refusal:
            return refusal, [], "distribution"
        value = geometry.summarize_bilateral_distribution(
            masks["Right_Lung"], masks["Left_Lung"],
            [masks[key] for key in lesion_mask_names])
        return value, names, "distribution"

    tools = [
        single_mask("measure_region_extent", geometry.measure_extent, "measurement", "A9", "inclusive pixel bounding-box extents and pixel-cell maximum Feret diameter", True),
        _RegisteredMetric("measure_region_area", "A10", "measure binary region area in pixels or supplied spacing",
                          parameter_schema({"mask_name": {"type": "string"}, "spacing_row_col": _array2()}, ("mask_name",)), area_operation),
        _RegisteredMetric("measure_region_distance", "A11", "gap between pixel-cell regions plus nearest pixel-centre distance",
                          parameter_schema({"mask_a": {"type": "string"}, "mask_b": {"type": "string"}, "spacing_row_col": _array2()}, ("mask_a", "mask_b")), distance_operation),
        _RegisteredMetric("compute_ctr_from_borders", "A12", "PARTIAL: CTR ratio only; caller must supply thoracic inner border x positions (not detected); heart borders supplied or taken from the Heart mask extent",
                          parameter_schema({"thoracic_inner_borders_x": _array2(), "heart_borders_x": _array2()}, ("thoracic_inner_borders_x",)), ctr_operation),
        _RegisteredMetric("compare_lung_geometry", "A13", "compare right and left lung area and maximum row width",
                          parameter_schema(), compare_lungs),
        single_mask("describe_region_shape", geometry.describe_shape, "measurement", "A14", "perimeter, circularity, solidity, and oriented aspect ratio"),
        _RegisteredMetric("summarize_roi_intensity", "A15", "raw grayscale statistics within a mask or the full image",
                          parameter_schema({"mask_name": {"type": "string"}}), appearance_operation(appearance.summarize_intensity, True)),
        _RegisteredMetric("extract_roi_texture", "A16", "six 2D GLCM features, fixed bin width 16 and four symmetric directions",
                          parameter_schema({"mask_name": {"type": "string"}}, ("mask_name",)), appearance_operation(appearance.extract_texture_features, True)),
        _RegisteredMetric("measure_roi_sharpness", "A17", "Tenengrad sharpness proxy in a mask or the full image",
                          parameter_schema({"mask_name": {"type": "string"}, "intensity_scale": {"type": "number", "exclusiveMinimum": 0}}), appearance_operation(appearance.measure_tenengrad, True)),
        _RegisteredMetric("measure_lung_symmetry", "A19", "mirror Dice between lungs at a caller-supplied or mask-derived midline",
                          parameter_schema({"midline_x": {"type": ["number", "null"]}}), symmetry_operation),
        _RegisteredMetric("assign_lung_zones", "A20", "divide each lung's vertical mask extent into three equal row bands",
                          parameter_schema({"region_mask_name": {"type": "string"}}, ("region_mask_name",)), zones_operation),
        _RegisteredMetric("relate_lesion_to_anatomy", "A21", "2D overlap, containment, distance, and image-axis direction",
                          parameter_schema({"lesion_mask_name": {"type": "string"}, "anatomy_mask_name": {"type": "string"}}, ("lesion_mask_name", "anatomy_mask_name")), lesion_anatomy_operation),
        _RegisteredMetric("relate_multiple_regions", "A22", "pairwise 2D topology and distance for named masks",
                          parameter_schema({"region_mask_names": {"type": "array", "items": {"type": "string"}, "minItems": 2}}, ("region_mask_names",)), multi_region_operation),
        _RegisteredMetric("summarize_bilateral_distribution", "A23", "aggregate supplied candidate masks by lung side and connected component",
                          parameter_schema({"lesion_mask_names": {"type": "array", "items": {"type": "string"}}}, ("lesion_mask_names",)), distribution_operation),
    ]
    return tools
