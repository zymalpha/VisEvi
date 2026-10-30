"""Unit tests: deterministic geometry engine (pure functions, synthetic masks)."""
import unittest

import numpy as np

from visevi.perception import geometry as geo


def rect(x0, y0, x1, y1, w=20, h=10):
    m = np.zeros((h, w), bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


class TestGeometry(unittest.TestCase):
    def test_area_bbox(self):
        m = rect(2, 3, 8, 6)                      # 7 x 4
        self.assertEqual(geo.area(m), 28)
        self.assertEqual(geo.bbox(m), {"x0": 2, "y0": 3, "x1": 8, "y1": 6, "w": 7, "h": 4})

    def test_max_row_width(self):
        m = np.zeros((6, 12), bool)
        m[1, 2:5] = True                          # width 3 @ y1
        m[4, 1:8] = True                          # width 7 @ y4
        w, y = geo.max_row_width(m)
        self.assertEqual((w, y), (7, 4))

    def test_min_distance_and_adjacency(self):
        a = rect(0, 0, 3, 3)
        b = rect(5, 0, 8, 3)                      # nearest b pixel x=5, nearest a x=3
        self.assertAlmostEqual(geo.min_distance(a, b), 2.0)
        c = rect(3, 0, 6, 3)                      # shares column x=3 -> overlapping
        self.assertAlmostEqual(geo.min_distance(a, c), 0.0)  # 0 = adjacency evidence

    def test_ctr_from_masks(self):
        heart = rect(8, 4, 11, 7)                 # width 4
        lungs = np.zeros((10, 20), bool)
        lungs[3:8, 1:7] = True                    # width 6
        lungs[3:8, 13:19] = True
        union = lungs[:, :]                       # union spans 1..18 -> width 18? rows 3..7
        r = geo.ctr_from_masks(heart, union)
        self.assertEqual(r["heart_max_width_px"], 4)
        self.assertEqual(r["thorax_max_width_px"], 18)
        self.assertAlmostEqual(r["CTR"], 4 / 18)

    def test_ctr_danzer_strict_vs_single_row(self):
        heart = np.zeros((10, 30), bool)
        heart[2:4, 4:9] = True                      # right portion, rows 2-3 (5 wide)
        heart[6:8, 9:20] = True                     # left portion, rows 6-7 (11 wide)
        thorax = rect(2, 1, 28, 8, w=30, h=10)      # max width 27
        loose = geo.ctr_from_masks(heart, thorax)
        strict = geo.ctr_danzer(heart, thorax)
        self.assertEqual(loose["heart_max_width_px"], 11)     # widest single row
        self.assertEqual(strict["heart_transverse_px"], 15)   # bbox 4..19 telescopes
        self.assertAlmostEqual(strict["d_right_px"] + strict["d_left_px"], 15)
        # midline placement inside the heart span cannot change the diameter
        for mid in (11.5, 16.5):
            self.assertEqual(geo.ctr_danzer(heart, thorax, midline_x=mid)["heart_transverse_px"], 15)
        self.assertAlmostEqual(strict["CTR"], 15 / 27)

    def test_mirror_dice_perfect(self):
        w = h = 10
        m = np.zeros((h, w), bool)
        m[2:8, 1:4] = True                        # cols 1,2,3
        left = np.zeros((h, w), bool)
        left[2:8, 6:9] = True                     # x'=9-x: {1,2,3} -> {8,7,6}
        r = geo.mirror_dice(m, left, midline_x=4.5)
        self.assertAlmostEqual(r["mirror_dice"], 1.0)

    def test_mirror_dice_partial_bounded(self):
        m = np.zeros((10, 10), bool)
        m[2:8, 1:4] = True                        # cols 1,2,3
        left = np.zeros((10, 10), bool)
        left[2:8, 5:8] = True                     # cols 5,6,7; reflected {6,7,8} -> 2-col overlap
        r = geo.mirror_dice(m, left, midline_x=4.5)
        self.assertAlmostEqual(r["mirror_dice"], 2 * 12 / 36)  # 2 cols x 6 rows = 12 px
        self.assertLessEqual(r["mirror_dice"], 1.0)

    def test_mirror_dice_disjoint(self):
        m = np.zeros((10, 10), bool)
        m[0:3, 0:2] = True
        left = np.zeros((10, 10), bool)
        left[7:10, 8:10] = True
        r = geo.mirror_dice(m, left, midline_x=4.5)
        self.assertAlmostEqual(r["mirror_dice"], 0.0)

    def test_measure_extent_and_spacing(self):
        m = rect(1, 2, 3, 4, w=8, h=8)
        value = geo.measure_extent(m, spacing=(2.0, 0.5))
        self.assertEqual(value["bbox_width_px"], 3)
        self.assertEqual(value["bbox_height_px"], 3)
        self.assertEqual(value["max_row_width_px"], 3)
        # pixel cells x 0.5..3.5, y 1.5..4.5: corner-to-corner diagonal of a 3x3 square
        self.assertAlmostEqual(value["feret_max_px"], np.sqrt(18))
        self.assertAlmostEqual(geo.area_physical(m, (2, 0.5)), 9.0)
        self.assertAlmostEqual(value["feret_max_physical"], np.sqrt(1.5 ** 2 + 6 ** 2))

    def test_feret_matches_bbox_cell_convention(self):
        bar = rect(5, 2, 14, 2, w=20, h=5)
        value = geo.measure_extent(bar)
        self.assertEqual(value["bbox_width_px"], 10)
        self.assertAlmostEqual(value["feret_max_px"], np.sqrt(101))
        self.assertGreaterEqual(value["feret_max_px"], value["bbox_width_px"])
        self.assertAlmostEqual(geo.feret_max_diameter(rect(2, 2, 2, 2)), np.sqrt(2))

    def test_gap_distance_uses_pixel_cells(self):
        a, b = rect(0, 0, 0, 0), rect(2, 1, 2, 1)
        self.assertAlmostEqual(geo.gap_distance(a, b), 1.0)
        self.assertAlmostEqual(geo.gap_distance(a, b, spacing=(0.5, 2)), 2.0)
        self.assertEqual(geo.gap_distance(a, rect(1, 0, 1, 0)), 0.0)  # edge contact
        self.assertAlmostEqual(geo.relate_masks(a, b)["gap_distance_px"], geo.gap_distance(a, b))

    def test_distance_spacing_overlap_and_empty(self):
        a, b = rect(0, 0, 0, 0), rect(2, 1, 2, 1)
        self.assertAlmostEqual(geo.min_distance(a, b), np.sqrt(5))
        self.assertAlmostEqual(geo.min_distance(a, b, spacing=(0.5, 2)), np.sqrt(16.25))
        self.assertEqual(geo.min_distance(a, a), 0.0)
        with self.assertRaises(ValueError):
            geo.min_distance(np.zeros_like(a), b)

    def test_ctr_from_explicit_boundaries(self):
        result = geo.ctr_from_borders((65, 135), (20, 180))
        self.assertEqual(result["heart_width_px"], 70)
        self.assertEqual(result["thoracic_inner_width_px"], 160)
        self.assertAlmostEqual(result["CTR"], 70 / 160)
        with self.assertRaises(ValueError):
            geo.ctr_from_borders((1, 1), (0, 100))
        with self.assertRaises(ValueError):
            geo.ctr_from_borders((0, 200), (50, 150))  # heart wider than thorax

    def test_lung_comparison_missing_side_is_explicit(self):
        right, left = rect(1, 1, 3, 3, w=10, h=10), np.zeros((10, 10), bool)
        result = geo.lung_area_compare(right, left)
        self.assertEqual(result["right_area_px"], 9)
        self.assertEqual(result["area_ratio_R_over_L"], None)
        self.assertEqual(result["status"], "one_or_both_sides_empty")

    def test_lung_comparison_known_area_and_widths(self):
        right = rect(1, 1, 3, 3, w=10, h=10)
        left = rect(6, 1, 7, 4, w=10, h=10)
        result = geo.lung_area_compare(right, left)
        self.assertEqual(result["right_area_px"], 9)
        self.assertEqual(result["left_area_px"], 8)
        self.assertEqual(result["area_difference_right_minus_left_px"], 1)
        self.assertEqual(result["right_max_width_px"], 3)
        self.assertEqual(result["left_max_width_px"], 2)
        self.assertAlmostEqual(result["area_ratio_R_over_L"], 9 / 8)

    def test_shape_descriptors_for_square(self):
        result = geo.describe_shape(rect(2, 2, 5, 5, w=10, h=10))
        largest = result["largest_component"]
        self.assertEqual(largest["area_px"], 16)
        self.assertEqual(largest["perimeter_px"], 16.0)
        self.assertAlmostEqual(largest["circularity"], np.pi / 4)
        self.assertAlmostEqual(largest["rotated_aspect_ratio"], 1)

    def test_pixel_cell_relations_distinguish_contact_kinds(self):
        a = rect(1, 1, 2, 2)
        edge = rect(3, 1, 4, 2)
        corner = rect(3, 3, 4, 4)
        gap = rect(4, 1, 5, 2)
        self.assertEqual(geo.relate_masks(a, edge)["topology"], "edge_adjacent")
        self.assertEqual(geo.relate_masks(a, corner)["topology"], "corner_adjacent")
        self.assertEqual(geo.relate_masks(a, gap)["topology"], "disjoint")
        self.assertEqual(geo.relate_masks(a, edge)["boundary_distance_px"], 0)
        self.assertAlmostEqual(geo.relate_masks(a, gap)["boundary_distance_px"], 1)
        self.assertEqual(geo.relate_masks(a, rect(1, 1, 1, 1))["topology"], "overlap")

    def test_lesion_anatomy_reports_image_axes_and_overlap(self):
        anatomy = rect(2, 2, 5, 5, w=10, h=10)
        lesion = rect(4, 4, 6, 6, w=10, h=10)
        result = geo.relate_lesion_to_anatomy(lesion, anatomy)
        self.assertEqual(result["topology"], "overlap")
        self.assertEqual(result["intersection_px"], 4)
        self.assertEqual(result["relative_image_horizontal"], "image_right")
        self.assertEqual(result["relative_image_vertical"], "below")
        self.assertEqual(result["laterality_note"], "image-axis direction only; patient laterality requires verified orientation")

    def test_lung_zones_partition_each_lung_height(self):
        right, left = rect(1, 0, 2, 8, w=10, h=10), rect(7, 1, 8, 9, w=10, h=10)
        region = np.zeros((10, 10), bool)
        region[0:3, 1] = True
        region[3:6, 1] = True
        region[6:9, 1] = True
        result = geo.assign_lung_zones(right, left, region)
        self.assertEqual(result["right"]["zone_overlap_px"],
                         {"upper": 3, "middle": 3, "lower": 3})
        self.assertEqual(result["left"]["overlap_px"], 0)

    def test_bilateral_distribution_counts_components(self):
        right, left = rect(1, 1, 3, 8, w=10, h=10), rect(6, 1, 8, 8, w=10, h=10)
        lesions = np.zeros((10, 10), bool)
        lesions[2, 2] = lesions[6, 7] = lesions[0, 5] = True
        result = geo.summarize_bilateral_distribution(right, left, [lesions])
        self.assertEqual(result["pattern"], "bilateral")
        self.assertEqual(result["component_count"], 3)
        self.assertEqual(result["outside_lung_components"], 1)


if __name__ == "__main__":
    unittest.main()


class TestRotatedRectangleTieBreak(unittest.TestCase):
    def test_diagonal_pair_is_reported_as_elongated(self):
        mask = np.zeros((6, 6), bool)
        mask[2, 2] = mask[3, 3] = True
        comp = geo.describe_shape(mask)['components'][0]
        self.assertAlmostEqual(comp['minimum_rotated_rectangle_long_side_px'], 2 * np.sqrt(2))
        self.assertAlmostEqual(comp['minimum_rotated_rectangle_short_side_px'], np.sqrt(2))
        self.assertAlmostEqual(comp['rotated_aspect_ratio'], 2.0)
