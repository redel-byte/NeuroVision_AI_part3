# NeuroVision AI Part 3 — project context and learning path

Last reviewed: 2026-10-26. This file is the handoff for a future conversation: ask the assistant to read it and the current repository before continuing. Update the progress log after each completed step. The source brief was supplied in French; this guide uses English for teaching.

## Project contract

Build an **educational computer vision prototype** that detects and localizes brain tumors in MRI images. Parts 1 and 2 classified whole images; Part 3 predicts bounding boxes, dataset classes, and confidence scores. The requested stack is Kaggle data, Ultralytics **YOLOv8**, OpenCV, a VastAI GPU, MLflow, FastAPI, Streamlit, and Airflow. The final work includes reproducible code, a README, and the existing Jira plan.

The dataset named in the brief is [MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes). Its actual directory names, label format, class mapping, negative images, and patient identifiers must be **inspected after download**. Do not infer them from its title.

This is a research/learning project. No model output should be presented as a clinical diagnosis.

## Current repository state

- `pyproject.toml` defines Python >=3.11 but has no dependencies yet; `.python-version` currently says 3.11.
- `notebooks/workflow.ipynb` contains one empty code cell. `app.py` is a placeholder. The package initializer is also a placeholder.
- No dataset, trained weights, or completed pipeline were found in the checkout on 2026-10-26.
- `NeuroVision_AI_Part3_Jira_1week.csv` contains a one-week Jira plan.
- The working tree already has user changes, including a deleted `README.md`. Inspect before editing; do not assume these changes should be discarded.

## How to work together

For each stage, follow the interactive teaching agreement in `AGENTS.md`: explain one concept, work through a small example, pose one focused exercise, then build and check a small increment together. Adapt the next explanation to the user's answer. Record decisions and observed results below. The learner can request a faster pace or a complete implementation at any time.

## Roadmap and completion gates

| Stage | Learn and build | Gate before moving on |
| --- | --- | --- |
| 0. Environment | Python 3.11, reproducible dependencies, repository layout, GPU/CUDA plan | Dependencies and execution environment documented; no credentials committed |
| 1. Data discovery | Download and inventory the real Kaggle archive; understand YOLO boxes; show at least 9 annotated images; count images and boxes per class | Saved audit with observed paths, class mapping, distributions, and representative visualizations |
| 2. Data quality | Match image/label stems, decode images, validate every label, count corrupt/orphan/unsupported files, decide how confirmed negatives are represented | Explicit reject log and counts; no silent repairs |
| 3. Splits and YAML | Pair images and labels; preserve existing valid splits or make deterministic patient-level splits where possible; check duplicates and leakage | Train/val/test counts, class distributions, leakage check, valid `data.yaml` |
| 4. Augmentation | Inspect modest rotation, translation, scale, and intensity changes; assess horizontal flip and Mosaic against MRI anatomy | A documented policy and sample augmented images; validation/test untouched |
| 5. Training | Test one epoch with `yolov8n.pt`; compare controlled full runs; record seed, compute, epochs, batch, image size, patience, time | `best.pt`, `last.pt`, curves, run configuration and reproducible command |
| 6. Evaluation | Explain IoU, precision, recall, AP and mAP; inspect per-class errors, false positives and missed tumors | Choose using validation; report a single final test evaluation and limitations |
| 7. MLflow | Track parameters, metrics, duration and artifacts; register a loadable inference package and manage the `champion` alias | At least two comparable runs and a documented promotion rule |
| 8. API and UI | FastAPI loads the champion once; `/health` and `/predict`; Streamlit uploads an image and calls the API | Valid/invalid/empty-prediction cases tested end to end |
| 9. Airflow and delivery | One task per pipeline stage, dependencies and retries; README, pinned versions and Jira evidence | DAG dry run and documented end-to-end reproduction |

Two decisions matter throughout: use **validation**, not test, for tuning and champion promotion; and check for patient/study overlap before splitting MRI slices. If patient IDs are unavailable, report that limitation and at least search for exact or near-duplicate images.

## Stage 1: first guided exercise

1. Obtain the archive through Kaggle in a local `data/raw` directory or supply its location. The [Kaggle CLI dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) describe `kaggle datasets files <owner/dataset>` and `kaggle datasets download <owner/dataset> --path <directory> --unzip`. Authentication may be required. Record the dataset version/date and license before using it.
2. List the top-level directories and count each extension. Open a small set of images and label files. Record the *observed* class IDs and names; do not create `data.yaml` from guesses.
3. Parse one annotation line: `class_id x_center y_center width height`. The four box values are fractions of the image dimensions. For example, with a 200 × 100 pixel image and line `0 0.5 0.5 0.2 0.4`, the box is `(x_min=80, y_min=30, x_max=120, y_max=70)`. Derive this from `x_min = (x_center-width/2)*W` and the corresponding other edges.
4. Draw boxes on at least 9 images and show class names. Count both images containing each class and boxes of each class; these differ if an image has multiple boxes.
5. Produce an audit table: total files, usable images, labels, images missing labels, labels missing images, empty labels, malformed lines, unreadable images, and counts by class/split.

Starter parsing example (illustrative; it does not yet check class mapping, image decoding, or all malformed-value cases):

```python
def yolo_line_to_xyxy(line: str, image_width: int, image_height: int):
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields, got {len(fields)}: {line!r}")
    class_id = int(fields[0])
    xc, yc, bw, bh = map(float, fields[1:])
    if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
        raise ValueError(f"Invalid normalized box: {line!r}")
    xmin, xmax = xc - bw / 2, xc + bw / 2
    ymin, ymax = yc - bh / 2, yc + bh / 2
    if min(xmin, ymin) < 0 or max(xmax, ymax) > 1:
        raise ValueError(f"Box crosses the image boundary: {line!r}")
    return class_id, (xmin * image_width, ymin * image_height,
                      xmax * image_width, ymax * image_height)
```

Questions to answer from the real data: Are the labels actually YOLO `.txt` files? What are the class names and counts? Are there patient or study IDs? Are empty/missing labels intentional negatives or mistakes? Are pre-existing Train/Valid/Test partitions trustworthy?

## Design notes for later stages

- A missing YOLO label file can mean an image has no objects in Ultralytics' format, but **this project** must distinguish a confirmed negative from an accidental missing annotation. A zero-byte `.txt` is a useful explicit convention after review. See the [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect).
- Start with YOLOv8n for a pipeline smoke test, then compare with YOLOv8s or a single augmentation change under the same split and evaluation settings. The [YOLOv8 model page](https://docs.ultralytics.com/models/yolov8) lists the detection weights.
- Treat class recall and missed tumors as important analysis points, but select any inference confidence threshold on validation data and report the precision trade-off. See [Ultralytics detection metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics).
- MLflow run artifacts such as `best.pt` alone do not establish a loadable registered model contract. Before implementing the registry, decide how a model version packages the checkpoint, class mapping, preprocessing and inference code. See [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) and [Model Registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/).
- Use FastAPI's [file upload](https://fastapi.tiangolo.com/tutorial/request-files/) and [lifespan](https://fastapi.tiangolo.com/advanced/events/) patterns for the inference service. Use Streamlit's [file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) for the client. Follow the installed major version of [Airflow DAG documentation](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html) when writing the DAG.

## Progress log

| Date | Completed | Evidence | Next |
| --- | --- | --- | --- |
| 2026-10-26 | Read project brief and inspected starter repository; created this roadmap and a repository `AGENTS.md` for interactive mentoring | Repository inventory above; dataset absent | Download or locate the dataset, then perform Stage 1 audit |
