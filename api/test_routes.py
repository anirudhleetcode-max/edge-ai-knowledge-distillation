import io
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from main import app
    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "Install api/requirements.txt to run FastAPI route tests")
class RouteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_and_metadata_are_versioned(self):
        health = self.client.get("/health")
        metadata = self.client.get("/api/v1/metadata")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["schemaVersion"], "2.0")
        self.assertEqual(metadata.status_code, 200)
        self.assertIn("metricPolicy", metadata.json())

    def test_distillation_validation_rejects_invalid_temperature(self):
        response = self.client.post("/api/v1/distillation/validate", json={"temperature": 0.5})
        self.assertEqual(response.status_code, 422)

    def test_dataset_inspect_uses_real_csv_shape(self):
        rows = ["text,label"] + [f"message {index},{index % 2}" for index in range(20)]
        csv = ("\n".join(rows) + "\n").encode()
        response = self.client.post("/api/v1/dataset/inspect", files={"file": ("fixture.csv", io.BytesIO(csv), "text/csv")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["columns"], ["text", "label"])
        self.assertEqual(sum(response.json()["rows"].values()), 20)

    def test_missing_artifact_benchmark_is_explicitly_unavailable(self):
        response = self.client.post("/api/v1/benchmark", json={"model_path": "/missing/model", "dataset_path": "/missing/data.csv"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertEqual(response.json()["fp32"]["value"], "Not measured")

    @patch("main.benchmark_local_model", side_effect=RuntimeError("runtime failure"))
    @patch("main.Path.exists", return_value=True)
    def test_benchmark_adapter_failure_is_http_400(self, _exists, _benchmark):
        response = self.client.post("/api/v1/benchmark", json={"model_path": "fixture", "dataset_path": "fixture.csv"})
        self.assertEqual(response.status_code, 400)

    def test_quantize_missing_artifact_is_unavailable(self):
        response = self.client.post("/api/v1/optimization/quantize", json={"model_path": "/missing"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")

    def test_prune_missing_artifact_is_unavailable(self):
        response = self.client.post("/api/v1/optimization/prune", json={"model_path": "/missing", "pruning_amount": 0.2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")

    @patch("main.transform_local_model", side_effect=RuntimeError("quantization failure"))
    def test_quantize_adapter_failure_is_http_400(self, _transform):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/optimization/quantize", json={"model_path": "fixture"})
        self.assertEqual(response.status_code, 400)

    @patch("main.transform_local_model", side_effect=RuntimeError("pruning failure"))
    def test_prune_adapter_failure_is_http_400(self, _transform):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/optimization/prune", json={"model_path": "fixture", "pruning_amount": 0.2})
        self.assertEqual(response.status_code, 400)

    @patch("main.evaluate_local_model", return_value={"measured": True, "f1": 0.8, "accuracy": 0.9})
    def test_evaluate_route_returns_adapter_result(self, _evaluate):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/evaluate?model_path=fixture&dataset_path=fixture.csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "measured")
        self.assertEqual(response.json()["student"]["f1"], 0.8)

    @patch("main.transform_local_model", return_value={"measured": True, "artifactPath": "fixture-int8"})
    def test_quantize_route_returns_artifact_contract(self, _transform):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/optimization/quantize", json={"model_path": "fixture"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["artifactPath"], "fixture-int8")

    @patch("main.transform_local_model", return_value={"measured": True, "artifactPath": "fixture-pruned"})
    def test_prune_route_returns_artifact_contract(self, _transform):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/optimization/prune", json={"model_path": "fixture", "pruning_amount": 0.2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["artifactPath"], "fixture-pruned")

    @patch("main.benchmark_local_model", side_effect=[{"measured": True, "medianMs": 4.0}, {"measured": True, "medianMs": 2.0}])
    @patch("main.transform_local_model", return_value={"measured": True, "artifactPath": "fixture-int8"})
    def test_benchmark_route_returns_measured_variants(self, _transform, _benchmark):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/benchmark", json={"model_path": "fixture", "dataset_path": "fixture.csv"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "measured")
        self.assertEqual(response.json()["fp32"]["medianMs"], 4.0)
        self.assertEqual(response.json()["int8"]["medianMs"], 2.0)

    def test_distillation_run_missing_dataset_is_unavailable(self):
        response = self.client.post("/api/v1/distillation/run?dataset_path=/missing/dataset.csv", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")

    @patch("main.distill_student", return_value={"measured": True, "artifactPath": "fixture-student"})
    def test_distillation_run_returns_artifact_contract(self, _distill):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/distillation/run?dataset_path=fixture.csv&output_dir=fixture-student", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "measured")
        self.assertEqual(response.json()["result"]["artifactPath"], "fixture-student")

    @patch("main.export_onnx", return_value="fixture.onnx")

    @patch("main.load_local_model", return_value=(lambda *args, **kwargs: {}, object()))
    def test_export_route_returns_output_path(self, _load, _export):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/export/onnx", json={"model_path": "fixture", "output_path": "fixture.onnx"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["artifactPath"], "fixture.onnx")

    @patch("main.export_torchscript", return_value="fixture.pt")
    @patch("main.load_local_model", return_value=(lambda *args, **kwargs: {}, object()))
    def test_torchscript_export_route_returns_output_path(self, _load, _export):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/export/torchscript", json={"model_path": "fixture", "output_path": "fixture.pt"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["artifactPath"], "fixture.pt")

    @patch("main.predict_text", return_value={"measured": True, "label": "spam", "confidence": 0.91, "probabilities": {"ham": 0.09, "spam": 0.91}})
    def test_prediction_route_returns_measured_label_and_confidence(self, _predict):
        with patch("main.Path.exists", return_value=True):
            response = self.client.post("/api/v1/predict", json={"model_path": "fixture", "text": "claim your prize"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "measured")
        self.assertEqual(response.json()["result"]["label"], "spam")
        self.assertAlmostEqual(response.json()["result"]["confidence"], 0.91)

    def test_prediction_missing_artifact_is_explicitly_unavailable(self):
        response = self.client.post("/api/v1/predict", json={"model_path": "/missing", "text": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")

    def test_export_missing_artifact_returns_400_or_unavailable(self):
        response = self.client.post("/api/v1/export/onnx", json={"model_path": "/missing", "output_path": "fixture.onnx"})
        self.assertIn(response.status_code, (400, 404))


if __name__ == "__main__":
    unittest.main()
