# Use the NeuroVision API

The API in `src/neurovision_ai_part3/api.py` exposes your existing YOLOv8m
checkpoint through FastAPI. A client uploads one image and receives JSON
detections. It does not need to import YOLO, run a notebook, or contact MLflow.

The current checkpoint is `models/best.pt`. Its observed classes are Glioma,
Meningioma and Pituitary. The API reads the class mapping from the model and
returns its SHA-256 to identify exactly which checkpoint served the request.
MLflow Registry packaging and a validated `champion` alias remain pending;
the API explicitly reports `local_checkpoint` as its model source.

## Start locally

Open PowerShell at the repository root and use the project's existing Python:

```powershell
.\.venv\Scripts\python.exe -m uvicorn neurovision_ai_part3.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Keep the terminal open. Stop with Ctrl+C. The model loads once at startup,
not once for every upload. The first prediction can still take longer while
Ultralytics initializes its predictor. Use one worker for this local baseline;
each additional worker would load its own copy of the checkpoint.

Open [interactive API documentation](http://127.0.0.1:8000/docs):

1. Expand `GET /health`, click **Try it out**, then **Execute**. Expect `200`,
   `model_loaded: true`, the class mapping and the checkpoint hash.
2. Expand `POST /predict` and click **Try it out**.
3. Leave `confidence` at `0.25`, choose a PNG/JPEG/BMP image, and execute.
   Use an existing validation sample for this check, not the reserved test split.
4. Inspect `detections` and the original image width and height in the response.

The API dependencies are declared directly in `pyproject.toml` and locked in
`uv.lock`. They were already installed in the inspected local environment.
On a separately prepared environment, install the project's dependencies first.
Keep Airflow in its own compatible environment when configuring a fresh Windows
or GPU setup; starting this API does not require running Airflow or MLflow.

## Endpoints

| Method | URL | Purpose |
| --- | --- | --- |
| GET | `/health` | Readiness, device, checkpoint hash and classes |
| POST | `/predict?confidence=0.25` | Upload one image in the multipart field `file` |
| GET | `/docs` | Interactive Swagger documentation and upload form |
| GET | `/openapi.json` | Machine-readable contract for other clients |

The `file` field is a binary file upload, not a JSON path to a file on your
computer. Confidence is a query parameter between 0 and 1. Missing fields and
invalid values are validated by FastAPI.

### Call with curl from PowerShell

Replace the example image path with a real image on your computer:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST "http://127.0.0.1:8000/predict?confidence=0.25" -F "file=@C:\path\to\mri.jpg"
```

Use `curl.exe` on Windows so PowerShell does not substitute an older `curl`
alias. On Linux/Vast.ai, the same request uses `curl` and the image's Linux path.

### Call from Python

This example uses `httpx`, which is declared in the development dependency group.
Replace `path/to/mri.jpg` with your image path:

```python
import httpx

with open("path/to/mri.jpg", "rb") as image:
    response = httpx.post(
        "http://127.0.0.1:8000/predict",
        params={"confidence": 0.25},
        files={"file": ("mri.jpg", image, "image/jpeg")},
        timeout=60,
    )

response.raise_for_status()
result = response.json()
print(result["detections"])
```

A future Streamlit app can make this same HTTP request and draw the returned
boxes on the image. It should not load YOLO itself.

## Understand a response

Illustrative values, not a reported model evaluation:

```json
{
  "message": "Detected 1 region(s).",
  "model": {
    "source": "local_checkpoint",
    "checkpoint": "best.pt",
    "sha256": "<64-character checkpoint hash>",
    "classes": {"0": "Glioma", "1": "Meningioma", "2": "Pituitary"}
  },
  "image_width": 512,
  "image_height": 512,
  "confidence_threshold": 0.25,
  "detections": [
    {
      "class_id": 0,
      "class_name": "Glioma",
      "confidence": 0.91,
      "bbox_xyxy": [40.0, 50.0, 140.0, 150.0]
    }
  ]
}
```

`bbox_xyxy` means `[left, top, right, bottom]` in **original-image pixels**.
The example box is 100 pixels wide and 100 pixels high. YOLO handles its own
resize/letterbox step, and its result coordinates map back to the uploaded image.

`confidence` is the detector's score, not the probability that a patient has
a disease. At a threshold of `0.70`, a score of `0.62` is filtered out. The
default threshold is a starting setting, not a clinically validated cutoff;
choose thresholds using validation data.

When nothing survives the threshold, the API returns `200`, `detections: []`
and `No detections at this confidence threshold.` This is a research result,
not proof of absence of a tumor.

## Input handling and errors

The API accepts `.jpg`, `.jpeg`, `.png` and `.bmp`, checks actual image contents,
and decodes them to the OpenCV BGR representation expected by Ultralytics.
DICOM, TIFF, HEIF and 3D MRI volumes are not inputs to this endpoint.

| Status | Meaning |
| --- | --- |
| 200 | Prediction completed, including an empty detection list |
| 400 | File extension is unsupported |
| 413 | File exceeds the byte limit or decoded dimensions exceed the pixel limit |
| 422 | Missing, empty, corrupt or unsupported image contents, or invalid confidence |
| 500 | Inference failed; details are in server logs |
| 503 | Checkpoint/device could not load; inspect `/health` and server logs |

The file limit is checked before reading all file bytes into memory, and image
dimensions are checked before OpenCV decodes the pixels. FastAPI's multipart
parser may spool uploads to temporary disk before this file-level check; this
is not a transport-level total request limit. The endpoint closes the upload
and does not retain source images or save predicted images to `runs/`.
Only the supported Pillow decoders are used, avoiding Ultralytics' optional
image-plugin installation fallback on corrupt uploads.

The shared model is protected by a lock: requests can be accepted concurrently,
but model inference runs one request at a time in the worker. A synchronous
route lets FastAPI execute the work in its thread pool.

## Configuration

Set these environment variables before starting Uvicorn:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEUROVISION_MODEL_PATH` | `models/best.pt` | Trusted local checkpoint; relative paths are resolved from the repository root |
| `NEUROVISION_DEVICE` | `cpu` | `cpu`, `0` for the first CUDA GPU, or a PyTorch device such as `cuda:0` |
| `NEUROVISION_MAX_UPLOAD_MB` | `10` | Positive integer limit, in MiB, for the uploaded image file |
| `NEUROVISION_MAX_IMAGE_PIXELS` | `16000000` | Positive integer limit for image width times height |

For a GPU machine with a CUDA-enabled Torch installation:

```powershell
$env:NEUROVISION_DEVICE = "0"
```

Then start the same server command. On Linux, set it with
`export NEUROVISION_DEVICE=0` and use `python -m uvicorn` in the installed
environment. The API disables Ultralytics automatic package installation when
loading the model; install required runtime dependencies during environment setup.

The current command binds locally. To use a remote Vast.ai API from your laptop,
forward local port 8000 to the instance's `127.0.0.1:8000` through your existing
SSH connection, then use the same local URLs. Browser clients hosted at another
origin would also need an explicit CORS policy; Swagger on this API's own origin
and Python/Streamlit clients work without that addition.

## Verification

The API tests use a lightweight detector stub with real Ultralytics result
objects, covering upload validation, serialization, startup/shutdown, missing
model, empty results, inference errors, docs and concurrent requests. Run them
from the root:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

A separate live Uvicorn smoke test used the real checkpoint and one 512 × 512
validation image. It returned one detection and verified pixel bounds, class
mapping, health/docs and HTTP error cases. No training or test-split evaluation
was performed. This verifies the API flow, not clinical performance or the
pending MLflow champion-selection gate.

Primary documentation: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[file uploads](https://fastapi.tiangolo.com/tutorial/request-files/),
[testing](https://fastapi.tiangolo.com/tutorial/testing/),
[Ultralytics prediction results](https://docs.ultralytics.com/modes/predict/).
