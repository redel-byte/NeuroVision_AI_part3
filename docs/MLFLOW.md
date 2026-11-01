# MLflow walkthrough

The notebook uses small functions in `src/neurovision_ai_part3/mlflow_tracking.py`.
Your existing training and validation settings are preserved. MLflow records
what happened; it does not change the detector or its augmentations.

## 1. Start the server

Open a terminal **at the repository root**, beside `pyproject.toml`. On Windows
PowerShell, use the existing virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Keep this terminal running. Open [MLflow](http://127.0.0.1:5000) in your browser.
Stop the server with Ctrl+C; restart with the same command from the same folder
to see the saved runs again.

- `mlflow.db` stores run names, parameters, metrics and artifact locations.
- `mlartifacts/` stores the uploaded files.
- `--serve-artifacts` routes file uploads/downloads through the HTTP server.
- `127.0.0.1:5000` is the address used by the notebook on the same machine.

Keep both the database and artifact directory when backing up the experiment.
Neither belongs in Git. Old file-based runs under `mlruns/` or
`artifacts/mlflow_tracking/` are preserved, but are not automatically imported
into this new database. A new server experiment uses the artifact proxy.

Using the existing interpreter avoids resolving the entire project's dependencies
just to start MLflow. In particular, Airflow needs its own compatible environment.
The inspected local versions are MLflow 3.17.2 and Ultralytics 8.4.193.

## 2. Connect the notebook

Use the project's Python kernel. Run the imports, paths and MLflow setup cells
at the top of `notebooks/workflow.ipynb`. The setup prints the server address
and experiment name, which defaults to `NeuroVision-YOLOv8`.

`setup_mlflow()` reads `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` if you
set them before starting the notebook. Otherwise it uses the local server above.
It disables MLflow autologging and sets Ultralytics' persistent `mlflow` setting
to false so these manual helpers own the runs. Restart an old notebook kernel
first if an existing YOLO instance already has automatic callbacks attached.

For a first check, use the optional **existing training import** block at the
bottom. Uncomment that block and run it after setup. It reads `train-2/args.yaml`
and `train-2/results.csv`; it does not train or evaluate a model. Only available
checkpoints are uploaded. In the inspected checkout, `train-2/weights/best.pt`
is absent and `models/best.pt` exists separately; the validation run logs the
checkpoint it actually evaluates. The import environment describes the local
machine, not the original training GPU. Do not attach the current dataset to
historical training until you have verified it is the same dataset.

**Do not use Run All just to add tracking.** Preprocessing refuses to overwrite
its output, and the existing notebook contains a final-test evaluation. Use
individual cells. This MLflow integration does not require repeating that test.

## 3. Understand the functions

| Function | What it records |
| --- | --- |
| `setup_mlflow` | Selects the tracking server and experiment |
| `log_dataset_artifacts` | YAML, manifest, audit, summary, report hashes and split counts |
| `log_yolo_metrics` | Precision, recall, mAP50, mAP75, mAP50–95 and measured per-class metrics |
| `log_yolo_artifacts` | One output directory's figures/reports, available weights, optional evaluated checkpoint and its hash |
| `log_training_results` | Actual `args.yaml` parameters, epoch-by-epoch CSV history, completed epochs, elapsed CSV time and training files |
| `log_environment` | Python/package versions, CUDA and GPU on the machine logging the run |

All logging helpers require an active run. The notebook uses
`with mlflow.start_run(...)` to group related information. The run ends when
the block finishes; an exception marks it failed. Helpers do not create hidden
unnamed runs. Missing explicitly requested files raise errors.

Think of a run as one experiment folder:

- **Parameter:** a setting, such as `imgsz = 640`. It stays fixed within a run.
- **Metric:** a measurement, such as validation mAP50–95 = `0.722`. Epoch metrics
  can have a sequence of steps, such as epoch 1, 2, 3.
- **Artifact:** a file, such as `best.pt`, `results.csv`, or a confusion matrix.

For example, a validation mAP change from `0.700` to `0.722` is an increase of
`0.022`, or **2.2 percentage points**. Compare runs on the same validation split
and settings. A parameter such as `split` cannot be successively changed from
`train` to `val` to `test` in one run; dataset counts instead use distinct metric
names such as `data/train/images` and `data/val/images`.

The dataset hashes identify the saved YAML and manifest, not a new full scan of
MRI bytes. Per-class metrics use the IDs actually present in the metric object;
we do not invent a score for a class absent from that evaluation.

## 4. Track the workflow

1. After successful preprocessing, the separate tracking cell uploads dataset
   reports. Counting and printing splits no longer open MLflow runs.
2. The commented training block opens a run, trains with the existing settings,
   then logs the actual trainer output folder and validation metrics. Uncomment
   only when you intend to train. CSV history is uploaded **after training**;
   this deliberately simple integration does not stream live epoch metrics.
3. The validation cell opens its own run, evaluates `models/best.pt` on `val`,
   then records metrics, duration, plots, dataset reports and that checkpoint.
   Its output folder includes the run ID so the uploaded plots belong to that
   exact evaluation. Subsequent metric-printing cells still work.
4. The historical import records epoch history under `history/`. The final CSV
   epoch is not necessarily the best checkpoint: use the separately measured
   `val/` metrics to assess a checkpoint.

Runs importing older data have the import time as their MLflow run time. The
CSV-derived training duration is recorded separately. Validation comparisons,
threshold tuning and future promotion use **validation**, never test results.

The saved `.pt` is an artifact. Model Registry packaging and `champion` promotion
remain a later milestone; they require a loadable inference contract and a
validation-based selection policy.

## 5. Verify in the UI

Open `NeuroVision-YOLOv8`, select a run and check:

1. Parameters show the actual settings (`seed` comes from training `args.yaml`,
   not an assumed value).
2. Metrics contain `val/precision`, `val/recall`, `val/mAP50_95` for a validation
   run, or stepped `history/` curves for a training import.
3. Artifacts contain the expected `dataset/`, `yolo/`, `weights/` or
   `evaluated_checkpoint/` folders. Download one report to verify it opens.
4. Compare two comparable validation runs only after they exist. A smoke test
   or a training-history import alone does not complete the MLflow milestone.

Learning check: for a confusion-matrix PNG, should MLflow use a parameter,
a metric, or an artifact?

## 6. Running on Vast.ai

Start MLflow from the project root in a persistent terminal on the GPU machine,
using its installed Python environment:

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

If the notebook also runs on that machine, its default tracking address is
already correct. To view the remote UI locally, add port forwarding to your
existing Vast.ai SSH connection: forward local port 5000 to remote
`127.0.0.1:5000`. Use the SSH host, user and port supplied by your instance.
Then open the same local browser address. If local port 5000 is busy, forward
local 5001 instead and browse port 5001; the remote notebook still uses 5000.

`127.0.0.1` always means the machine running that process. A local Windows
server is not automatically reachable as localhost from a Vast.ai notebook.
Keep the remote database and artifact directory on persistent storage and use
the SSH tunnel for access, consistent with this repository's deployment policy.

## Troubleshooting

- **Connection refused:** keep the server terminal running and check that the
  notebook URI refers to that same machine and port.
- **Port 5000 occupied:** choose another server port and pass its HTTP URI to
  `setup_mlflow`, or set `MLFLOW_TRACKING_URI` before opening the notebook.
- **Old runs missing:** check the server's working directory and database;
  the old file store and the new SQLite database are separate.
- **Unexpected nested/automatic runs:** restart the kernel, run setup, then
  create a fresh YOLO instance. Do not reuse models with old logging callbacks.
- **Artifacts not downloading:** use the server command with
  `--artifacts-destination` and `--serve-artifacts`; an experiment created with
  an old artifact location retains that location. Use a new experiment for
  the new proxy configuration.

Primary references: [MLflow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/),
[manual tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api/),
[Ultralytics MLflow integration](https://docs.ultralytics.com/integrations/mlflow/).
