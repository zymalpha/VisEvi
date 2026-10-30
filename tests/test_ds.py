"""Unit tests: DS kernel — numbers from LAYER2_ALGORITHM_SPEC.md §3.3/§6."""
import unittest

from visevi.construction import ds


class TestDS(unittest.TestCase):
    def test_discount(self):
        # spec §6: alpha=0.808, m={NOT_F:.85, THETA:.15}
        m = ds.discount({"NOT_F": 0.85, "THETA": 0.15}, 0.808)
        self.assertAlmostEqual(m["NOT_F"], 0.6868, places=4)
        self.assertAlmostEqual(m["THETA"], 0.3132, places=4)
        self.assertAlmostEqual(m["F"], 0.0)

    def test_dempster_spec_walkthrough(self):
        # spec §6: m1={NOT_F:.687, THETA:.313} x m2={NOT_F:.7, THETA:.3} -> .906
        m, k = ds.dempster({"NOT_F": 0.687, "THETA": 0.313}, {"NOT_F": 0.7, "THETA": 0.3})
        self.assertAlmostEqual(k, 0.0)
        self.assertAlmostEqual(m["NOT_F"], 0.9061, places=4)
        self.assertAlmostEqual(m["THETA"], 0.0939, places=4)

    def test_dempster_conflict(self):
        m, k = ds.dempster({"F": 0.6, "THETA": 0.4}, {"NOT_F": 0.6, "THETA": 0.4})
        self.assertAlmostEqual(k, 0.36, places=6)
        # non-conflicting mass renormalized
        self.assertAlmostEqual(m["F"] + m["NOT_F"] + m["THETA"], 1.0)

    def test_yager_conflict_to_ignorance(self):
        m1, m2 = {"F": 0.8, "THETA": 0.2}, {"NOT_F": 0.9, "THETA": 0.1}
        m, k = ds.yager(m1, m2)
        self.assertAlmostEqual(k, 0.72)
        self.assertAlmostEqual(m["F"], 0.08)
        self.assertAlmostEqual(m["NOT_F"], 0.18)
        self.assertAlmostEqual(m["THETA"], 0.02 + 0.72)   # conflict -> ignorance

    def test_combine_policy(self):
        m1, m2 = {"F": 0.8, "THETA": 0.2}, {"NOT_F": 0.9, "THETA": 0.1}
        m, k, rule = ds.combine(m1, m2)
        self.assertEqual(rule, "yager")                    # K=0.72 > 0.6
        m, k, rule = ds.combine({"NOT_F": 0.7, "THETA": 0.3}, {"NOT_F": 0.6, "THETA": 0.4})
        self.assertEqual(rule, "dempster")
        self.assertAlmostEqual(k, 0.0)

    def test_bel_pl(self):
        iv = ds.bel_pl({"F": 0.2, "NOT_F": 0.5, "THETA": 0.3})
        self.assertAlmostEqual(iv["bel_F"], 0.2)
        self.assertAlmostEqual(iv["pl_F"], 0.5)            # F + THETA
        self.assertAlmostEqual(iv["bel_not_f"], 0.5)
        self.assertAlmostEqual(iv["pl_not_f"], 0.8)
        self.assertAlmostEqual(iv["interval_width"], 0.3)

    def test_calibration_curves(self):
        self.assertEqual(ds.g_ctr(0.383)["NOT_F"], 0.85)   # <=0.45 strong not-F
        self.assertEqual(ds.g_ctr(0.62)["F"], 0.85)        # >=0.55 strong F
        mid = ds.g_ctr(0.50)                               # transition band
        self.assertAlmostEqual(mid["F"], 0.425)
        self.assertAlmostEqual(mid["NOT_F"], 0.425)
        self.assertEqual(ds.g_symmetry(0.936)["F"], 0.8)   # >=0.93 symmetric
        self.assertEqual(ds.g_symmetry(0.70)["NOT_F"], 0.8)

    def test_mass_validation(self):
        with self.assertRaises(ValueError):
            ds._norm({"F": 0.7})


if __name__ == "__main__":
    unittest.main()
