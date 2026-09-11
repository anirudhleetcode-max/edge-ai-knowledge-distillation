import unittest

try:
    import torch
    from ml_pipeline import DistillationConfig, distillation_loss, metric_state
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "Install api/requirements.txt to run PyTorch pipeline tests")
class PipelineTests(unittest.TestCase):
    def test_distillation_loss_returns_hard_and_soft_components(self):
        student = torch.tensor([[1.0, 0.2], [0.1, 1.4]], requires_grad=True)
        teacher = torch.tensor([[1.2, 0.1], [0.2, 1.7]])
        labels = torch.tensor([0, 1])
        loss, detail = distillation_loss(student, teacher, labels, temperature=4.0, alpha=0.5)
        self.assertGreater(float(loss), 0)
        self.assertIn("hardCrossEntropy", detail)
        self.assertIn("softKLDivergence", detail)

    def test_config_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            DistillationConfig(temperature=0.5).validate()

    def test_unmeasured_values_are_explicit(self):
        self.assertEqual(metric_state(None), "Not measured")
        self.assertEqual(metric_state(0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
