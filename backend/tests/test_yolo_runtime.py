import unittest
from types import SimpleNamespace

from pvrt.backends.yolo.runtime import resolve_yolo_device


class YoloRuntimeTests(unittest.TestCase):
    def test_cpu_only_pytorch_uses_cpu(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        self.assertEqual(resolve_yolo_device(torch), "cpu")

    def test_working_cuda_build_uses_first_gpu(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda="12.1"),
            cuda=SimpleNamespace(is_available=lambda: True),
        )
        self.assertEqual(resolve_yolo_device(torch), 0)

    def test_broken_cuda_install_does_not_silently_fallback(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda="12.1"),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
            resolve_yolo_device(torch)


if __name__ == "__main__":
    unittest.main()
