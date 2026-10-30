"""Exercise HTTP contracts without evaluating the held-out MRI test split."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
from threading import Lock
import time
import unittest
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import torch
from ultralytics.engine.results import Results

from neurovision_ai_part3 import api


class FakeDetector:
    """Use real Ultralytics Results to check the API's tensor conversion."""

    def __init__(self):
        self.calls = []
        self.fail = False
        self.active = 0
        self.max_active = 0
        self.counter_lock = Lock()

    def predict(self, **kwargs):
        with self.counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            self.calls.append(kwargs)
            if self.fail:
                raise RuntimeError("Deliberate inference failure")
            boxes = [[1, 2, 15, 18, 0.8, 2]] if kwargs["conf"] <= 0.8 else []
            return [Results(
                orig_img=kwargs["source"], path="upload",
                names={0: "Glioma", 1: "Meningioma", 2: "Pituitary"},
                boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6),
            )]
        finally:
            with self.counter_lock:
                self.active -= 1


class APITests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "NEUROVISION_MODEL_PATH": "models/best.pt",
            "NEUROVISION_DEVICE": "cpu",
            "NEUROVISION_MAX_UPLOAD_MB": "1",
            "NEUROVISION_MAX_IMAGE_PIXELS": "1024",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.detector = FakeDetector()
        self.info = api.ModelInfo(
            checkpoint="best.pt", sha256="a" * 64,
            classes={0: "Glioma", 1: "Meningioma", 2: "Pituitary"},
        )
        self.loader_patch = patch.object(api, "load_model", return_value=(self.detector, self.info))
        self.loader = self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)
        # A colored image checks that OpenCV BGR channels are preserved.
        self.image = np.full((20, 24, 3), [12, 34, 56], dtype=np.uint8)
        self.png = cv2.imencode(".png", self.image)[1].tobytes()

    def upload(self, client, contents=None, filename="mri.png", confidence=0.25):
        return client.post(
            "/predict", params={"confidence": confidence},
            files={"file": (filename, self.png if contents is None else contents, "application/octet-stream")},
        )

    def test_loads_once_and_returns_real_result_fields(self):
        app = api.create_app()
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["model"]["source"], "local_checkpoint")
            response = self.upload(client)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual((data["image_width"], data["image_height"]), (24, 20))
            self.assertEqual(data["detections"][0]["class_name"], "Pituitary")
            self.assertEqual(data["detections"][0]["bbox_xyxy"], [1, 2, 15, 18])
            self.assertAlmostEqual(data["detections"][0]["confidence"], 0.8, places=6)
            self.assertEqual(self.upload(client).status_code, 200)
            self.loader.assert_called_once_with(api.PROJECT_DIR / "models/best.pt", "cpu")
            call = self.detector.calls[0]
            self.assertEqual(call["conf"], 0.25)
            self.assertFalse(call["save"])
            np.testing.assert_array_equal(call["source"], self.image)
        self.assertIsNone(app.state.model)

    def test_jpeg_png_bmp_and_uppercase_extensions(self):
        with TestClient(api.create_app()) as client:
            for suffix, filename in ((".png", "mri.png"), (".jpg", "mri.JPEG"), (".bmp", "mri.bmp")):
                with self.subTest(filename=filename):
                    image_bytes = cv2.imencode(suffix, self.image)[1].tobytes()
                    self.assertEqual(self.upload(client, image_bytes, filename).status_code, 200)

    def test_empty_detection_is_not_a_diagnosis(self):
        with TestClient(api.create_app()) as client:
            response = self.upload(client, confidence=0.9)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["detections"], [])
            self.assertEqual(response.json()["message"], "No detections at this confidence threshold.")

    def test_missing_file_is_422(self):
        with TestClient(api.create_app()) as client:
            self.assertEqual(client.post("/predict").status_code, 422)

    def test_empty_and_corrupt_images_are_422(self):
        with TestClient(api.create_app()) as client:
            for contents in (b"", b"not an image", self.png[:32]):
                with self.subTest(contents=contents[:10]):
                    self.assertEqual(self.upload(client, contents).status_code, 422)
            self.assertEqual(self.detector.calls, [])

    def test_extension_and_actual_format_are_checked(self):
        gif = BytesIO()
        Image.new("RGB", (8, 8)).save(gif, format="GIF")
        with TestClient(api.create_app()) as client:
            self.assertEqual(self.upload(client, filename="mri.txt").status_code, 400)
            self.assertEqual(self.upload(client, gif.getvalue(), "disguised.png").status_code, 422)
            self.assertEqual(self.detector.calls, [])

    def test_invalid_confidence_is_422(self):
        with TestClient(api.create_app()) as client:
            for threshold in (-0.1, 1.1, "nan", "inf", "wrong"):
                with self.subTest(threshold=threshold):
                    self.assertEqual(self.upload(client, confidence=threshold).status_code, 422)
            self.assertEqual(self.detector.calls, [])

    def test_confidence_endpoints_are_allowed(self):
        with TestClient(api.create_app()) as client:
            for threshold in (0, 1):
                self.assertEqual(self.upload(client, confidence=threshold).status_code, 200)

    def test_upload_byte_limit(self):
        with TestClient(api.create_app()) as client:
            response = self.upload(client, b"x" * (1024 * 1024 + 1))
            self.assertEqual(response.status_code, 413)
            self.assertEqual(self.detector.calls, [])

    def test_image_pixel_limit(self):
        large = cv2.imencode(".png", np.zeros((33, 33, 3), dtype=np.uint8))[1].tobytes()
        with TestClient(api.create_app()) as client:
            self.assertEqual(self.upload(client, large).status_code, 413)
            self.assertEqual(self.detector.calls, [])

    def test_unavailable_model_keeps_health_and_docs_accessible(self):
        self.loader.side_effect = FileNotFoundError("Missing checkpoint")
        with self.assertLogs(api.LOGGER, level="ERROR"):
            with TestClient(api.create_app()) as client:
                self.assertEqual(client.get("/health").status_code, 503)
                self.assertFalse(client.get("/health").json()["model_loaded"])
                self.assertEqual(client.get("/docs").status_code, 200)
                self.assertEqual(self.upload(client).status_code, 503)

    def test_inference_failure_is_a_clear_500(self):
        self.detector.fail = True
        with TestClient(api.create_app()) as client:
            with self.assertLogs(api.LOGGER, level="ERROR"):
                response = self.upload(client)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("Deliberate", response.json()["detail"])

    def test_parallel_requests_serialize_the_shared_model(self):
        with TestClient(api.create_app()) as client:
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(lambda _: self.upload(client), range(4)))
            self.assertTrue(all(response.status_code == 200 for response in responses))
            self.assertEqual(self.detector.max_active, 1)
            self.loader.assert_called_once()

    def test_openapi_describes_upload_and_confidence(self):
        with TestClient(api.create_app()) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            operation = client.get("/openapi.json").json()["paths"]["/predict"]["post"]
            self.assertIn("multipart/form-data", operation["requestBody"]["content"])
            self.assertTrue(operation["requestBody"]["required"])
            self.assertEqual(operation["parameters"][0]["name"], "confidence")

    def test_bad_limits_fail_at_startup(self):
        with patch.dict(os.environ, {"NEUROVISION_MAX_UPLOAD_MB": "0"}):
            with self.assertRaises(ValueError):
                with TestClient(api.create_app()):
                    pass


if __name__ == "__main__":
    unittest.main()
