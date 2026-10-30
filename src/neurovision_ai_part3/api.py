"""A small HTTP API for the existing YOLOv8 detection checkpoint."""

from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
import numpy as np
from PIL import BmpImagePlugin, Image, JpegImagePlugin, PngImagePlugin, UnidentifiedImageError
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ModelInfo(BaseModel):
    source: Literal["local_checkpoint"] = "local_checkpoint"
    checkpoint: str
    sha256: str
    classes: dict[int, str]


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]


class PredictionResponse(BaseModel):
    message: str
    model: ModelInfo
    image_width: int
    image_height: int
    confidence_threshold: float
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    device: str
    model: ModelInfo | None


def positive_setting(name, default):
    """Read a positive integer configuration value at startup."""
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def load_model(model_path, device):
    """Load once; check the local file before YOLO can try any download."""
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError("The API requires a YOLO detection checkpoint.")
    torch_device = "cuda:" + device if device.isdigit() else device
    model.to(torch_device)
    with model_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    info = ModelInfo(
        checkpoint=model_path.name,
        sha256=digest,
        classes=model.names,
    )
    return model, info


def decode_image(contents, max_pixels):
    """Validate real image contents and dimensions before decoding to BGR."""
    # Use just the supported Pillow decoders. Ultralytics patches Image.open
    # to install an optional HEIF plugin when it encounters unidentified bytes.
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        decoder = PngImagePlugin.PngImageFile
    elif contents.startswith(b"\xff\xd8"):
        decoder = JpegImagePlugin.JpegImageFile
    elif contents.startswith(b"BM"):
        decoder = BmpImagePlugin.BmpImageFile
    else:
        raise HTTPException(422, "Image contents must be a decodable JPEG, PNG or BMP.")
    try:
        with decoder(BytesIO(contents)) as source:
            width, height = source.size
            if width * height > max_pixels:
                raise HTTPException(413, "Image dimensions exceed the pixel limit.")
            source.verify()
        # OpenCV produces the BGR array expected by Ultralytics for numpy input.
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Image.DecompressionBombError as error:
        raise HTTPException(413, "Image dimensions exceed the pixel limit.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, cv2.error) as error:
        raise HTTPException(422, "The uploaded image cannot be decoded.") from error
    if image is None:
        raise HTTPException(422, "The uploaded image cannot be decoded.")
    return image


def predict_image(model, image, confidence, device):
    """Convert YOLO results to ordinary JSON-compatible detection values."""
    result = model.predict(
        source=image,
        conf=confidence,
        device=device,
        imgsz=640,
        save=False,
        save_txt=False,
        save_crop=False,
        verbose=False,
    )[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes.cpu():
            class_id = int(box.cls.item())
            detections.append(Detection(
                class_id=class_id,
                class_name=result.names[class_id],
                confidence=float(box.conf.item()),
                bbox_xyxy=box.xyxy[0].tolist(),
            ))
    return detections


@asynccontextmanager
async def lifespan(app):
    """Keep one model in memory per server process, shared by all requests."""
    model_path = Path(os.getenv("NEUROVISION_MODEL_PATH", "models/best.pt"))
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path
    app.state.device = os.getenv("NEUROVISION_DEVICE", "cpu")
    app.state.max_upload_bytes = positive_setting("NEUROVISION_MAX_UPLOAD_MB", 10) * 1024 * 1024
    app.state.max_pixels = positive_setting("NEUROVISION_MAX_IMAGE_PIXELS", 16_000_000)
    app.state.model = None
    app.state.model_info = None
    app.state.inference_lock = Lock()
    try:
        app.state.model, app.state.model_info = load_model(model_path, app.state.device)
    except Exception:
        # Keep health/docs available when the checkpoint or device is unavailable.
        LOGGER.exception("Model startup failed; prediction is unavailable.")
    try:
        yield
    finally:
        app.state.model = None
        app.state.model_info = None


def create_app():
    app = FastAPI(
        title="NeuroVision AI Detection API",
        version="0.1.0",
        description="Research prototype for MRI object detection; outputs are not clinical diagnoses.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, responses={503: {"description": "Model unavailable"}})
    def health(request: Request, response: Response):
        ready = request.app.state.model is not None
        if not ready:
            response.status_code = 503
        return HealthResponse(
            status="ok" if ready else "unavailable",
            model_loaded=ready,
            device=request.app.state.device,
            model=request.app.state.model_info,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            400: {"description": "Unsupported image format"},
            413: {"description": "File or image dimensions exceed the limit"},
            422: {"description": "Missing, empty or invalid image, or invalid threshold"},
            503: {"description": "Model unavailable"},
        },
    )
    def predict(
        request: Request,
        file: UploadFile = File(..., description="One JPEG, PNG or BMP image"),
        confidence: float = Query(0.25, ge=0, le=1, allow_inf_nan=False),
    ):
        state = request.app.state
        try:
            if state.model is None:
                raise HTTPException(503, "Model unavailable. Check /health and the server logs.")
            if Path(file.filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
                raise HTTPException(400, "Allowed file extensions: .jpg, .jpeg, .png, .bmp.")
            contents = file.file.read(state.max_upload_bytes + 1)
        finally:
            file.file.close()

        if not contents:
            raise HTTPException(422, "The uploaded image is empty.")
        if len(contents) > state.max_upload_bytes:
            raise HTTPException(413, "The uploaded file exceeds the size limit.")
        image = decode_image(contents, state.max_pixels)
        try:
            # Sync routes run in FastAPI's thread pool. Serialize the shared YOLO
            # instance, including result extraction, to avoid concurrent mutation.
            with state.inference_lock:
                detections = predict_image(state.model, image, confidence, state.device)
        except Exception as error:
            LOGGER.exception("Model inference failed.")
            raise HTTPException(500, "Prediction failed. Check the server logs.") from error
        height, width = image.shape[:2]
        message = (
            f"Detected {len(detections)} region(s)."
            if detections else "No detections at this confidence threshold."
        )
        return PredictionResponse(
            message=message,
            model=state.model_info,
            image_width=width,
            image_height=height,
            confidence_threshold=confidence,
            detections=detections,
        )

    return app


app = create_app()
