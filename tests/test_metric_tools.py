"""Registry-level tests for the deterministic A-tool interfaces."""
import json

import numpy as np
from PIL import Image

from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_registry, build_a_registry


def _session(tmp_path):
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    image_path = tmp_path / "image.png"
    Image.fromarray(image).save(image_path)
    session = EvidenceSession(image_ref=str(image_path))
    right = np.zeros((10, 10), dtype=bool)
    left = np.zeros((10, 10), dtype=bool)
    heart = np.zeros((10, 10), dtype=bool)
    lesion = np.zeros((10, 10), dtype=bool)
    right[1:9, 1:4] = True
    left[1:9, 6:9] = True
    heart[3:7, 3:7] = True
    lesion[4:6, 2:4] = True
    session.attach_masks({"Heart": heart, "Right_Lung": right,
                          "Left_Lung": left, "Lesion": lesion})
    return session


def test_registry_exposes_each_deterministic_tool():
    registry = build_registry()
    specs = registry.describe_all()
    names = {spec["name"] for spec in specs}
    expected = {
        "measure_region_extent", "measure_region_area", "measure_region_distance",
        "compute_ctr_from_borders", "compare_lung_geometry", "describe_region_shape",
        "summarize_roi_intensity", "extract_roi_texture", "measure_roi_sharpness",
        "measure_lung_symmetry", "assign_lung_zones", "relate_lesion_to_anatomy",
        "relate_multiple_regions", "summarize_bilateral_distribution",
    }
    assert expected <= names
    json.dumps(specs)


def test_registered_tools_run_against_explicit_masks(tmp_path):
    registry = build_registry()
    session = _session(tmp_path)
    cases = [
        ("measure_region_extent", {"mask_name": "Heart"}),
        ("measure_region_area", {"mask_name": "Heart", "spacing_row_col": [2, 0.5]}),
        ("measure_region_distance", {"mask_a": "Heart", "mask_b": "Lesion"}),
        ("compute_ctr_from_borders", {"thoracic_inner_borders_x": [0, 9]}),
        ("compare_lung_geometry", {}),
        ("describe_region_shape", {"mask_name": "Heart"}),
        ("summarize_roi_intensity", {"mask_name": "Right_Lung"}),
        ("measure_roi_sharpness", {"mask_name": "Right_Lung"}),
        ("measure_lung_symmetry", {"midline_x": 4.5}),
        ("assign_lung_zones", {"region_mask_name": "Lesion"}),
        ("relate_lesion_to_anatomy", {"lesion_mask_name": "Lesion", "anatomy_mask_name": "Right_Lung"}),
        ("relate_multiple_regions", {"region_mask_names": ["Heart", "Lesion", "Right_Lung"]}),
        ("summarize_bilateral_distribution", {"lesion_mask_names": ["Lesion"]}),
    ]
    for name, params in cases:
        result = registry.run(name, session, **params)
        assert not isinstance(result, ToolRefusal), (name, result)
        assert len(result) == 1
        assert result[0].source_tool == name
        json.dumps(result[0].to_dict())


def test_registered_tool_refuses_missing_masks_and_extracts_texture(tmp_path):
    registry = build_registry()
    session = _session(tmp_path)
    missing = registry.run("measure_region_area", session, mask_name="Missing")
    assert isinstance(missing, ToolRefusal)
    assert missing.missing == ["mask:Missing"]

    texture = registry.run("extract_roi_texture", session, mask_name="Right_Lung")
    assert not isinstance(texture, ToolRefusal)
    assert set(texture[0].value['features']) == {
        'contrast', 'dissimilarity', 'homogeneity', 'angular_second_moment', 'energy', 'correlation'}
    json.dumps(texture[0].to_dict(), allow_nan=False)


def test_a_registry_covers_a1_to_a26_without_legacy_placeholders():
    from visevi.eval.runner import a_capabilities
    registry = build_a_registry()
    specs = registry.describe_all()
    assert sorted(a_capabilities(registry), key=lambda c: int(c[1:])) == [f'A{i}' for i in range(1, 27)]
    assert a_capabilities(registry)['A12'] == ['compute_ctr_from_borders', 'measure_ctr']
    assert len(specs) == 27
    from jsonschema import Draft202012Validator
    for spec in json.loads(json.dumps(specs)):
        Draft202012Validator.check_schema(spec['input_schema'])
        Draft202012Validator.check_schema(spec['output_schema'])
    assert all(spec['category'] == 'perception' for spec in specs)
    assert not any('compose_evidence' in spec['name'] or spec['name'] == 'carry_view_assumption' for spec in specs)


def test_invalid_tool_parameters_are_structured_refusals(tmp_path):
    registry = build_a_registry()
    session = _session(tmp_path)
    cases = [
        ('measure_region_extent', {}),
        ('measure_region_area', {'mask_name': 'Heart', 'unexpected': True}),
        ('measure_region_distance', {'mask_a': 'Heart', 'mask_b': 'Lesion', 'spacing_row_col': [1]}),
        ('measure_region_distance', {'mask_a': 'Heart', 'mask_b': 'Lesion', 'spacing_row_col': [-1, 1]}),
        ('measure_roi_sharpness', {'intensity_scale': float('nan')}),
        ('relate_multiple_regions', {'region_mask_names': ['Heart', 'Heart']}),
    ]
    for name, params in cases:
        result = registry.run(name, session, **params)
        assert isinstance(result, ToolRefusal), name
    assert not session.graph.nodes
    json.dumps(session.to_dict(), allow_nan=False)


def test_nonbinary_masks_are_not_silently_thresholded(tmp_path):
    session = _session(tmp_path)
    session.attach_masks({'probabilities': np.full((10, 10), 0.2)})
    result = build_a_registry().run('measure_region_area', session, mask_name='probabilities')
    assert isinstance(result, ToolRefusal)
    assert 'binary' in result.reason


def test_ctr_arithmetic_and_trace_preserve_explicit_coordinates(tmp_path):
    session = _session(tmp_path)
    result = build_a_registry().run('compute_ctr_from_borders', session,
                                   heart_borders_x=[15, 85], thoracic_inner_borders_x=[0, 100])
    assert not isinstance(result, ToolRefusal)
    assert result[0].value['CTR'] == 0.7
    assert session.trace[-1]['params']['heart_borders_x'] == [15, 85]
    assert session.trace[-1]['params']['thoracic_inner_borders_x'] == [0, 100]
    assert 'partial' in result[0].value['implementation_status']


def test_ctr_refuses_heart_outside_thorax(tmp_path):
    session = _session(tmp_path)
    result = build_a_registry().run('compute_ctr_from_borders', session,
                                   heart_borders_x=[0, 200], thoracic_inner_borders_x=[50, 150])
    assert isinstance(result, ToolRefusal)
    assert 'within' in result.reason


def test_distance_reports_cell_gap_and_pixel_centre_distance(tmp_path):
    session = _session(tmp_path)
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[0, 0] = True
    b[0, 1] = True
    session.attach_masks({'A': a, 'B': b})
    value = build_a_registry().run('measure_region_distance', session, mask_a='A', mask_b='B')[0].value
    assert value['gap_distance_px'] == 0.0
    assert value['pixel_center_distance_px'] == 1.0


def test_fov_truncated_masks_are_refused_by_measurements(tmp_path):
    session = _session(tmp_path)
    session.mask_metadata['Heart']['fov_truncated_sides'] = ['top']
    for name, params in [('measure_region_extent', {'mask_name': 'Heart'}),
                         ('compute_ctr_from_borders', {'thoracic_inner_borders_x': [0, 9]}),
                         ('relate_multiple_regions', {'region_mask_names': ['Heart', 'Lesion']})]:
        result = build_a_registry().run(name, session, **params)
        assert isinstance(result, ToolRefusal), name
        assert result.missing == ['fov_complete:Heart']
    assert not isinstance(build_a_registry().run('measure_region_area', session, mask_name='Lesion'), ToolRefusal)


def test_provenance_points_only_to_the_current_mask_producer(tmp_path):
    session = _session(tmp_path)
    for observation_id in ['obs-old', 'obs-new']:
        session.attach_masks({'Right_Lung': session.masks['Right_Lung'], 'Left_Lung': session.masks['Left_Lung']},
                             {'Right_Lung': {'observation_id': observation_id},
                              'Left_Lung': {'observation_id': observation_id + '-left'}})
    result = build_a_registry().run('compare_lung_geometry', session)
    assert result[0].provenance == ['obs-new', 'obs-new-left']


def test_grayscale_tool_preserves_native_16_bit_png(tmp_path):
    path = tmp_path / 'sixteen_bit.png'
    Image.fromarray(np.array([[0, 1024], [4096, 65535]], dtype=np.uint16)).save(path)
    session = EvidenceSession(image_ref=str(path))
    result = build_a_registry().run('summarize_roi_intensity', session)
    assert not isinstance(result, ToolRefusal)
    assert result[0].value['maximum'] == 65535
    assert result[0].value['mean'] == (0 + 1024 + 4096 + 65535) / 4
