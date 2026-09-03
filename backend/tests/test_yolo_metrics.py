import unittest
from types import SimpleNamespace

from pvrt.backends.yolo.train import _map50_column, _map50_from_result


class YoloMetricTests(unittest.TestCase):
    def setUp(self):
        self.columns = [
            "epoch",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def test_detection_selects_box_ap50(self):
        self.assertEqual(_map50_column(self.columns, False), "metrics/mAP50(B)")

    def test_segmentation_selects_mask_ap50(self):
        self.assertEqual(_map50_column(self.columns, True), "metrics/mAP50(M)")

    def test_result_object_uses_task_specific_metric_branch(self):
        result = SimpleNamespace(
            box=SimpleNamespace(map50=0.81),
            seg=SimpleNamespace(map50=0.67),
        )
        self.assertEqual(_map50_from_result(result, False), 0.81)
        self.assertEqual(_map50_from_result(result, True), 0.67)

    def test_result_dictionary_does_not_mix_box_and_mask_metrics(self):
        result = SimpleNamespace(results_dict={
            "metrics/mAP50(B)": 0.91,
            "metrics/mAP50(M)": 0.72,
        })
        self.assertEqual(_map50_from_result(result, False), 0.91)
        self.assertEqual(_map50_from_result(result, True), 0.72)


if __name__ == "__main__":
    unittest.main()
