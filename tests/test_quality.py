"""Unit tests: A25 quality metrics (deterministic, synthetic masks/images)."""
import unittest

import numpy as np

from visevi.perception.quality import (inspiration_metrics, rotation_metrics,
                                       exposure_metrics)


def lung_pair(h_pair=(2, 20), right_cols=(4, 10), left_cols=(14, 20), w=24, h=24):
    mr = np.zeros((h, w), bool)
    ml = np.zeros((h, w), bool)
    mr[h_pair[0]:h_pair[1], right_cols[0]:right_cols[1]] = True
    ml[h_pair[0]:h_pair[1], left_cols[0]:left_cols[1]] = True
    return mr, ml


class TestInspiration(unittest.TestCase):
    def test_taller_lungs_higher_index(self):
        mr_t, ml_t = lung_pair((2, 20))                 # height 18
        mr_s, ml_s = lung_pair((6, 20))                 # height 14
        vt = inspiration_metrics(mr_t, ml_t, mr_t | ml_t)
        vs = inspiration_metrics(mr_s, ml_s, mr_s | ml_s)
        self.assertEqual(vt["right_height"], 18)
        self.assertEqual(vt["thorax_inner_width"], 16)
        self.assertAlmostEqual(vt["inspiration_index"], 18 / 16)
        self.assertGreater(vt["inspiration_index"], vs["inspiration_index"])

    def test_diaphragm_asymmetry(self):
        mr, ml = lung_pair((2, 20))
        ml_shift = np.roll(ml, 2, axis=0)               # left base 2 rows lower
        v = inspiration_metrics(mr, ml_shift, mr | ml_shift)
        self.assertEqual(v["diaphragm_dy_px"], 2)


class TestRotation(unittest.TestCase):
    def test_symmetric_pair_zero_proxy(self):
        mr, ml = lung_pair()
        v = rotation_metrics(mr, ml, mr | ml, midline_x=11.5)
        self.assertEqual(v["apex_dy_px"], 0)
        self.assertEqual(v["base_dy_px"], 0)
        self.assertEqual(v["rotation_proxy"], 0.0)

    def test_shifted_pair_proxy(self):
        mr, ml = lung_pair()
        v = rotation_metrics(mr, np.roll(ml, 2, axis=0), mr | ml, midline_x=11.5)
        self.assertEqual(v["apex_dy_px"], 2)
        self.assertAlmostEqual(v["rotation_proxy"], 2 / 18, places=3)


class TestExposure(unittest.TestCase):
    def test_stats_and_penetration(self):
        mr, ml = lung_pair()
        g = np.full((24, 24), 200, np.uint8)            # bright body
        g[mr | ml] = np.linspace(60, 100, (mr | ml).sum())  # dark lungs w/ gradient
        g[2:20, 10:14] = 160                            # mediastinum between lungs
        v = exposure_metrics(g, mr | ml, mr, ml)
        self.assertAlmostEqual(v["lung_mean"], 80.0, places=0)
        self.assertEqual(v["mediastinum_strip_mean"], 160.0)
        self.assertAlmostEqual(v["penetration_ratio"], 2.0, places=0)
        self.assertGreater(v["contrast_index"], 0)

    def test_clipping_fractions(self):
        mr, ml = lung_pair()
        g = np.full((24, 24), 128, np.uint8)
        g[0, :] = 0
        g[1, :] = 255
        v = exposure_metrics(g, mr | ml, mr, ml)
        self.assertAlmostEqual(v["clip_low_frac"], 24 / 576, places=4)
        self.assertAlmostEqual(v["clip_high_frac"], 24 / 576, places=4)

    def test_no_gap_rows_nan(self):
        mr, ml = lung_pair(right_cols=(4, 16), left_cols=(8, 20))   # overlapping edges
        g = np.full((24, 24), 128, np.uint8)
        v = exposure_metrics(g, mr | ml, mr, ml)
        self.assertIsNone(v["mediastinum_strip_mean"])
        self.assertIn("不可用", v["penetration_hint"])


if __name__ == "__main__":
    unittest.main()


def test_exposure_refuses_high_bit_depth_input(tmp_path):
    from PIL import Image
    from visevi.core.session import EvidenceSession
    from visevi.core.units import ToolRefusal
    from visevi.perception.quality import AssessExposure

    path = tmp_path / 'sixteen_bit.png'
    Image.fromarray(np.full((10, 10), 4000, dtype=np.uint16)).save(path)
    session = EvidenceSession(image_ref=str(path))
    lungs = np.zeros((10, 10), bool)
    lungs[2:8, 1:4] = True
    session.attach_masks({'Right_Lung': lungs, 'Left_Lung': lungs[:, ::-1]})
    result = AssessExposure().run(session)
    assert isinstance(result, ToolRefusal)
    assert 'windowing' in result.reason
