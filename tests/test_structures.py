"""Hand-computed tests for CXAS-derived rules (A12 thoracic margin, A25 quality)."""
import json

import numpy as np
import pytest

from visevi.perception import structures as st


def test_cxas_masks_map_back_with_torch_nearest_indexing(tmp_path):
    model = np.zeros((2, 4, 4), bool)
    model[0, 1, 2] = True                      # row 1, column 2 of a 4x4 model frame
    np.savez_compressed(tmp_path / 'labels_model_frame.npz', packed=np.packbits(model, axis=-1),
                        shape=np.array(model.shape))
    run = {'artifacts': {'labels_model_frame.npz': 'x'}, 'result': {'labels': {'0': 'a', '1': 'b'}}}
    mask = st.load_labels(run, tmp_path, ['a'], width=8, height=6)['a']
    # src = floor(dst * 4 / size): rows 2 (floor(2*4/6)=1) only; columns 4,5 -> 2
    assert np.argwhere(mask).tolist() == [[2, 4], [2, 5]]


def test_thoracic_inner_margin_uses_medial_edge_of_outermost_rib_run():
    right, left = np.zeros((50, 100), bool), np.zeros((50, 100), bool)
    right[5:40, 10:15] = True                  # lateral arc, inner edge x=14 -> 14.5
    right[5:40, 30:33] = True                  # a more medial rib run must be ignored
    left[5:40, 85:90] = True                   # lateral arc, inner edge x=85 -> 84.5
    left[20:25, 80:90] = True                  # thicker on rows 20-24 -> inner edge 79.5
    result = st.thoracic_inner_margin(right, left, midline=50, max_row=45)
    assert result['candidate_rows'] == 35
    assert (result['right_inner_x'], result['left_inner_x'], result['width_px']) == (14.5, 84.5, 70.0)
    assert result['row_y'] not in range(20, 25)
    with pytest.raises(ValueError, match='rows'):
        st.thoracic_inner_margin(right, left, midline=50, max_row=7)   # only rows 5-7 qualify


def test_rotation_index_from_medial_clavicle_ends():
    right, left, spine = (np.zeros((40, 100), bool) for _ in range(3))
    right[10, 20:40] = True                    # medial end x=39 -> cell edge 39.5
    left[12, 55:80] = True                     # medial end x=55 -> cell edge 54.5
    spine[:, 48:53] = True                     # mean x 50
    result = st.rotation(right, left, spine)
    assert (result['distance_right_px'], result['distance_left_px']) == (10.5, 4.5)
    assert result['asymmetry_index'] == pytest.approx(6 / 15)


def test_ribs_above_dome_column():
    masks = {st.rib_label('posterior', n, 'right'): np.zeros((100, 50), bool) for n in range(1, 13)}
    for n in range(1, 13):
        masks[st.rib_label('posterior', n, 'right')][n * 7: n * 7 + 3, 20:30] = True   # centre y = 7n+1
    result = st.ribs_above(masks, 'posterior', 'right', x=25, y_limit=60, half_window=2)
    assert result['ribs_above'] == [1, 2, 3, 4, 5, 6, 7, 8]                      # 7*8+1 = 57 < 60
    assert result['highest_number_above'] == 8


def test_dome_penetration_and_field_of_view():
    diaphragm = np.zeros((20, 10), bool)
    diaphragm[12:, 2:5] = True
    assert st.dome(diaphragm) == (12, 3.0)
    gray = np.full((40, 10), 10.0)
    heart, spine = np.ones((40, 10), bool), np.zeros((40, 10), bool)
    spine[:, 4:6] = True                       # 80 retrocardiac pixels, 320 beside
    gray[:, 4:6] = 30.0
    assert st.penetration(gray, heart, spine)['difference'] == 20.0
    fov = st.field_of_view({'lung': diaphragm})['lung']
    assert fov['touches_bottom'] and not fov['touches_top']


def test_rotation_undefined_when_medial_end_crosses_midline():
    right, left, spine = (np.zeros((20, 40), bool) for _ in range(3))
    right[5, 5:25] = True                      # medial end x=24 beyond the midline x=20
    left[6, 25:35] = True
    spine[:, 19:22] = True
    with pytest.raises(ValueError, match='midline'):
        st.rotation(right, left, spine)
