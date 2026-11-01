"""Small, manual MLflow helpers for the YOLOv8 notebook."""

import csv
import hashlib
import json
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path

import mlflow
import yaml


def setup_mlflow(tracking_uri=None, experiment_name=None):
    """Connect to the tracking server; call before creating a YOLO model."""
    from ultralytics import settings

    tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
    )
    experiment_name = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "NeuroVision-YOLOv8"
    )
    # Use only our manual logging, so automatic integrations do not create runs.
    mlflow.autolog(disable=True)
    settings.update({"mlflow": False})
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    print("MLflow server:", tracking_uri)
    print("Experiment:", experiment.name)
    return experiment


def require_active_run():
    """Avoid accidentally creating an unnamed run from a logging helper."""
    if mlflow.active_run() is None:
        raise RuntimeError("Start a run with 'with mlflow.start_run(...)' first.")


def log_dataset_artifacts(processed_dir):
    """Save the dataset configuration and reports, without copying MRI files."""
    require_active_run()
    processed_dir = Path(processed_dir)
    data_yaml = processed_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    mlflow.log_param("dataset_version", processed_dir.name)
    for filename in ("data.yaml", "manifest.csv", "audit.csv", "preprocessing_summary.json"):
        path = processed_dir / filename
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path="dataset")
            if filename in ("data.yaml", "manifest.csv"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                mlflow.log_param(filename.replace(".", "_") + "_sha256", digest)

    summary_path = processed_dir / "preprocessing_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for split, counts in summary["splits"].items():
            mlflow.log_metric(f"data/{split}/images", counts["images"])
            mlflow.log_metric(f"data/{split}/negatives", counts["negatives"])


def log_yolo_metrics(metrics, split="val"):
    """Log global and measured per-class detection metrics from model.val()."""
    require_active_run()
    if split not in ("val", "test"):
        raise ValueError("Use 'val' or 'test' to identify the evaluation split.")

    mlflow.log_metrics({
        f"{split}/precision": float(metrics.box.mp),
        f"{split}/recall": float(metrics.box.mr),
        f"{split}/mAP50": float(metrics.box.map50),
        f"{split}/mAP75": float(metrics.box.map75),
        f"{split}/mAP50_95": float(metrics.box.map),
    })
    for index, class_id in enumerate(metrics.box.ap_class_index):
        prefix = f"{split}/class_{int(class_id)}"
        mlflow.log_metrics({
            f"{prefix}/precision": float(metrics.box.p[index]),
            f"{prefix}/recall": float(metrics.box.r[index]),
            f"{prefix}/mAP50": float(metrics.box.ap50[index]),
            f"{prefix}/mAP50_95": float(metrics.box.ap[index]),
        })
    names = {str(key): value for key, value in metrics.names.items()}
    mlflow.log_dict(names, f"{split}/class_names.json")


def log_yolo_artifacts(save_dir, checkpoint=None):
    """Upload only this run's reports, figures and available checkpoints."""
    require_active_run()
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise NotADirectoryError(save_dir)
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

    for path in sorted(save_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".csv", ".yaml", ".json", ".txt"):
            mlflow.log_artifact(str(path), artifact_path="yolo")

    for filename in ("best.pt", "last.pt"):
        path = save_dir / "weights" / filename
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path="weights")
    if checkpoint is not None:
        path = Path(checkpoint)
        mlflow.log_artifact(str(path), artifact_path="evaluated_checkpoint")
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        mlflow.log_param("checkpoint_sha256", digest)


def log_training_results(save_dir):
    """Import actual training settings, epoch history and files after training."""
    require_active_run()
    save_dir = Path(save_dir)
    args_path = save_dir / "args.yaml"
    csv_path = save_dir / "results.csv"
    if not args_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("Training directory needs args.yaml and results.csv.")

    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    mlflow.log_params({key: value for key, value in args.items() if value is not None})
    last_row = None
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file, skipinitialspace=True):
            row = {key.strip(): value for key, value in row.items()}
            epoch = int(row["epoch"])
            values = {}
            for key, value in row.items():
                if key == "epoch":
                    continue
                number = float(value)
                if math.isfinite(number):
                    name = key.replace("(", "").replace(")", "")
                    values["history/" + name] = number
            mlflow.log_metrics(values, step=epoch)
            last_row = row
    if last_row is None:
        raise ValueError("Training results.csv has no epoch rows.")
    mlflow.log_metric("epochs_completed", int(last_row["epoch"]))
    if "time" in last_row and math.isfinite(float(last_row["time"])):
        mlflow.log_metric("training_seconds_from_csv", float(last_row["time"]))
    log_yolo_artifacts(save_dir)


def log_environment():
    """Record the logging machine's environment, including GPU availability."""
    import torch

    require_active_run()
    environment = {"python": platform.python_version()}
    for package in ("mlflow", "ultralytics", "torch"):
        environment[package] = version(package)
    environment["cuda"] = torch.version.cuda
    environment["gpu"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    mlflow.log_dict(environment, "environment/logging_machine.json")
