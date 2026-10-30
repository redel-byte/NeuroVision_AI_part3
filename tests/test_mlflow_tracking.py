"""Verify tracking contracts without training or evaluating the MRI test set."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlflow
from mlflow import MlflowClient

from neurovision_ai_part3.mlflow_tracking import (
    log_dataset_artifacts,
    log_training_results,
    log_yolo_artifacts,
    log_yolo_metrics,
)


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri("sqlite:///" + (cls.root / "tracking.db").as_posix())
        cls.client = MlflowClient()
        cls.experiment_id = cls.client.create_experiment(
            "tracking-tests", artifact_location=(cls.root / "artifacts").as_uri()
        )

    @classmethod
    def tearDownClass(cls):
        # Close the actual cached store before removing SQLite files on Windows.
        cls.client._tracking_client.store.engine.dispose()
        mlflow.set_tracking_uri(cls.previous_uri)
        cls.temp.cleanup()

    def test_logging_requires_an_explicit_run(self):
        with self.assertRaises(RuntimeError):
            log_yolo_artifacts(self.root)
        self.assertIsNone(mlflow.active_run())

    def test_per_class_metrics_follow_measured_ids(self):
        metrics = SimpleNamespace(
            names={0: "Glioma", 1: "Meningioma", 2: "Pituitary"},
            box=SimpleNamespace(
                mp=0.8, mr=0.7, map50=0.9, map75=0.6, map=0.5,
                ap_class_index=[0, 2], p=[0.75, 0.85], r=[0.6, 0.8],
                ap50=[0.8, 1.0], ap=[0.4, 0.6],
            ),
        )
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            log_yolo_metrics(metrics)
        result = self.client.get_run(run.info.run_id)
        self.assertEqual(result.data.metrics["val/class_2/mAP50_95"], 0.6)
        self.assertNotIn("val/class_1/mAP50_95", result.data.metrics)
        self.assertEqual(result.info.status, "FINISHED")

    def test_dataset_counts_and_artifact_roundtrip(self):
        dataset = self.root / "dataset_v1"
        dataset.mkdir()
        (dataset / "data.yaml").write_text("names: [Glioma]\n", encoding="utf-8")
        (dataset / "manifest.csv").write_text("split,hash\nval,example\n", encoding="utf-8")
        (dataset / "preprocessing_summary.json").write_text(
            '{"splits": {"train": {"images": 7, "negatives": 2}, '
            '"val": {"images": 2, "negatives": 1}}}', encoding="utf-8"
        )
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            log_dataset_artifacts(dataset)
        result = self.client.get_run(run.info.run_id)
        self.assertEqual(result.data.metrics["data/train/images"], 7)
        self.assertEqual(result.data.metrics["data/val/images"], 2)
        self.assertEqual(len(result.data.params["manifest_csv_sha256"]), 64)
        output = self.root / "download"
        output.mkdir()
        downloaded = self.client.download_artifacts(
            run.info.run_id, "dataset/data.yaml", str(output)
        )
        self.assertEqual(Path(downloaded).read_bytes(), (dataset / "data.yaml").read_bytes())

    def test_history_has_epoch_steps_and_actual_seed(self):
        training = self.root / "training"
        training.mkdir()
        (training / "args.yaml").write_text("seed: 0\nmodel: yolov8m.pt\n", encoding="utf-8")
        (training / "results.csv").write_text(
            "epoch,time,metrics/mAP50(B),train/box_loss\n"
            "1,5.0,0.5,1.0\n2,9.0,0.7,nan\n", encoding="utf-8"
        )
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            log_training_results(training)
        result = self.client.get_run(run.info.run_id)
        history = self.client.get_metric_history(run.info.run_id, "history/metrics/mAP50B")
        self.assertEqual([point.step for point in history], [1, 2])
        self.assertEqual(result.data.params["seed"], "0")
        self.assertEqual(result.data.metrics["training_seconds_from_csv"], 9.0)
        self.assertNotIn("val/mAP50", result.data.metrics)

    def test_missing_checkpoint_marks_run_failed_and_closes_it(self):
        with self.assertRaises(FileNotFoundError):
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                log_yolo_artifacts(self.root, checkpoint=self.root / "missing.pt")
        self.assertEqual(self.client.get_run(run.info.run_id).info.status, "FAILED")
        self.assertIsNone(mlflow.active_run())


if __name__ == "__main__":
    unittest.main()
