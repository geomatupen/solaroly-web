import unittest

from pvrt.dataops.row_sequence_alignment import (
    ALIGNMENT_QUALITY_PRESETS,
    ALIGNMENT_STRICTNESS_PRESETS,
    _candidate_pairs,
)


class ImageAlignmentOptionTests(unittest.TestCase):
    def test_high_quality_increases_resolution_and_features(self):
        standard = ALIGNMENT_QUALITY_PRESETS["standard"]
        high = ALIGNMENT_QUALITY_PRESETS["high"]
        self.assertGreater(high["analysis_max_dimension"], standard["analysis_max_dimension"])
        self.assertGreater(high["maximum_features"], standard["maximum_features"])

    def test_strictness_presets_change_all_acceptance_thresholds_consistently(self):
        strict = ALIGNMENT_STRICTNESS_PRESETS["strict"]
        balanced = ALIGNMENT_STRICTNESS_PRESETS["balanced"]
        lenient = ALIGNMENT_STRICTNESS_PRESETS["lenient"]
        self.assertGreater(strict["filter_threshold"], balanced["filter_threshold"])
        self.assertGreater(balanced["filter_threshold"], lenient["filter_threshold"])
        self.assertGreater(strict["minimum_matches"], balanced["minimum_matches"])
        self.assertGreater(balanced["minimum_matches"], lenient["minimum_matches"])
        self.assertGreater(strict["minimum_inlier_ratio"], balanced["minimum_inlier_ratio"])
        self.assertGreater(balanced["minimum_inlier_ratio"], lenient["minimum_inlier_ratio"])
        self.assertLess(strict["ransac_px"], balanced["ransac_px"])
        self.assertLess(balanced["ransac_px"], lenient["ransac_px"])

    def test_temporal_neighbor_count_controls_candidate_reach(self):
        names = [f"frame_{index}.jpg" for index in range(6)]
        entries = {
            name: {
                "timestamp": float(index),
                "lat": 48.0,
                "lon": 16.0,
                "meters_per_pixel": 0.05,
            }
            for index, name in enumerate(names)
        }
        records = {name: {"prepared_image_size": [640, 512]} for name in names}
        one_neighbor = _candidate_pairs(
            names, entries, records, temporal_neighbors=1, lateral_neighbors=0,
        )
        three_neighbors = _candidate_pairs(
            names, entries, records, temporal_neighbors=3, lateral_neighbors=0,
        )
        self.assertEqual(len(one_neighbor), 5)
        self.assertEqual(len(three_neighbors), 12)


if __name__ == "__main__":
    unittest.main()
