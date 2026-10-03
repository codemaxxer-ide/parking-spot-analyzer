"""Local FastAPI adapter for the verified parking pipeline; one active job."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import csv
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock
import time
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.calibration import draw_reference
from src.parking_slots import load_parking_config
from src.video_processor import process_video

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web/static"
MAX_VIDEO_BYTES = 500 * 1024 * 1024
ARTIFACTS = {
    name: name.replace("_", " ").rsplit(".", 1)[0].title()
    for name in ("cross_dataset_comparison.png", "baseline_external_confusion_matrix.png",
                 "mobilenet_external_confusion_matrix.png", "training_curve_baseline.png",
                 "training_curve_mobilenet.png", "failure_cases.jpg", "shadow_failures.jpg",
                 "occlusion_failures.jpg", "lighting_analysis.png", "occlusion_analysis.png",
                 "weather_analysis.png", "model_comparison.png", "results_summary.md")
}


def find_ffmpeg():
    """Prefer PATH, then the optional packaged binary; never require a download."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def video_properties(path):
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Video could not be decoded")
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        return {"size": frame.shape[:2], "fps": capture.get(cv2.CAP_PROP_FPS),
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "codec": "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).lower(),
                "pixel_format": int(capture.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))}
    finally:
        capture.release()


def encode_browser_video(source: Path, destination: Path):
    """Atomically publish H.264; retain the original as a downloadable fallback."""
    if source.resolve() == destination.resolve():
        raise ValueError("Browser output must differ from the original video.")
    original = video_properties(source)
    # OpenCV exposes the pixel format as a FourCC. Only copy 8-bit planar 4:2:0.
    if (original["codec"] in ("avc1", "h264")
            and original["pixel_format"] == cv2.VideoWriter_fourcc(*"I420")):
        return {"filename": source.name, "browser_playable": True, "warning": None}
    executable = find_ffmpeg()
    fallback = {"filename": source.name, "browser_playable": False,
                "warning": "FFmpeg unavailable. Analysis completed; download the original MP4 to play in VLC. Install imageio-ffmpeg for browser playback."}
    if not executable:
        return fallback
    temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}.mp4")
    try:
        subprocess.run([executable, "-nostdin", "-y", "-i", str(source), "-map", "0:v:0",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
                       check=True, capture_output=True, timeout=600,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        encoded = video_properties(temporary)
        if (encoded["size"] != original["size"] or encoded["frames"] != original["frames"]
                or abs(encoded["fps"] - original["fps"]) > 0.01
                or encoded["codec"] not in ("avc1", "h264")):
            raise RuntimeError("Encoded video properties changed")
        temporary.replace(destination)
        return {"filename": destination.name, "browser_playable": True, "warning": None}
    except (OSError, subprocess.SubprocessError, RuntimeError):
        logging.exception("Browser conversion failed; original analysis retained")
        fallback["warning"] = "Browser video conversion failed. Analysis completed; download the original MP4 to play in VLC. Check FFmpeg and free disk space."
        return fallback
    finally:
        temporary.unlink(missing_ok=True)


def inspect_input(video_path, slots_path, reference_path):
    configuration = load_parking_config(slots_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Unable to open video. Choose a readable MP4 and try again.")
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError("Video dimensions must be even. Export an MP4 with even width and height.")
        configuration.validate_frame(width, height)
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = fps if math.isfinite(fps) and fps > 0 else None
        count = int(count) if math.isfinite(count) and count > 0 else None
        if not cv2.imwrite(str(reference_path), draw_reference(frame, configuration.slots)):
            raise RuntimeError("Unable to save reference image")
        return {"width": width, "height": height, "fps": fps, "frame_count": count,
                "duration": count / fps if count and fps else None,
                "slots": [{"id": slot.slot_id, "label": slot.label, "points": slot.polygon.tolist()}
                          for slot in configuration.slots]}
    finally:
        capture.release()


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = Lock()
        self.active = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cloudforge")

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "Analysis not found. Start a new analysis after a server restart.")
            return json.loads(json.dumps(self.jobs[job_id]))

    def update(self, job_id, **changes):
        with self.lock:
            self.jobs[job_id].update(changes)

    def run(self, job_id):
        directory = self.directory / job_id
        job = self.get(job_id)
        settings = job["settings"]
        stage = "model"
        try:
            engine = settings["engine"]
            self.update(job_id, state="processing", started_at=time.time(),
                        stage="Loading MobileNetV2 slot classifier" if engine == "classifier" else "Loading YOLO11n")
            import torch
            torch.set_num_threads(4)
            detector = classifier = None
            if engine == "classifier":
                from src.slot_classifier import SlotClassifier
                classifier = SlotClassifier()
            else:
                if not (ROOT / "yolo11n.pt").is_file():
                    raise FileNotFoundError("YOLO11n weights missing")
                from src.detector import VehicleDetector
                detector = VehicleDetector(str(ROOT / "yolo11n.pt"), settings["confidence"])
            stage = "processing"

            def progress(snapshot):
                self.update(job_id, stage="Reading → detecting → mapping → stabilising", **snapshot)

            summary = process_video(directory / "input.mp4", directory / "annotated.mp4",
                                    directory / "slots.json", detector=detector,
                                    confidence=settings["confidence"], occupancy_method=settings["occupancy_method"],
                                    overlap_threshold=settings["overlap_threshold"],
                                    smoothing_window=settings["smoothing_window"],
                                    smoothing_required=settings["smoothing_required"],
                                    process_every=settings["process_every"], progress_callback=progress,
                                    engine=engine, classifier=classifier,
                                    classification_threshold=settings["classification_threshold"],
                                    show_rois=settings["show_rois"], show_ids=settings["show_ids"],
                                    show_boxes=settings["show_boxes"])
            stage = "encoding"
            self.update(job_id, stage="Generating browser video")
            video_output = encode_browser_video(directory / "annotated.mp4", directory / "output.mp4")
            final = self.get(job_id)
            states = final.get("occupied", {})
            result = {"total": summary.total_spaces, "occupied": summary.occupied_last_frame,
                      "available": summary.available_last_frame,
                      "occupancy_percent": 100 * summary.occupied_last_frame / summary.total_spaces,
                      "frames_written": summary.frames_written, "inference_frames": summary.inference_frames,
                      "fps": summary.fps, "width": summary.width, "height": summary.height,
                      "scope": "Final processed frame", "settings": settings, "video": video_output,
                      "engine": engine,
                      "slots": [{**slot, "occupied": states.get(slot["id"]),
                                 "occupied_probability": (summary.slot_probabilities or {}).get(slot["id"])}
                                for slot in job["metadata"]["slots"]]}
            with (directory / "analysis.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["slot_id", "label", "state", "scope", "frame"])
                for slot in result["slots"]:
                    writer.writerow([slot["id"], slot["label"], "OCCUPIED" if slot["occupied"] else "AVAILABLE",
                                     "final_processed_frame", summary.frames_written])
            (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            self.update(job_id, state="complete", stage="Analysis complete", result=result, completed_at=time.time())
        except Exception:
            logging.exception("CloudForge job %s failed", job_id)
            message = {"model": ("Slot classifier unavailable. Check models/mobilenetv2_parking.pt and PyTorch."
                                 if settings["engine"] == "classifier" else
                                 "YOLO model unavailable. Check yolo11n.pt and the installed CV dependencies."),
                       "processing": "Video processing failed. Check that the MP4 is complete and the ROI configuration matches it.",
                       "encoding": "Browser video encoding failed. Check free disk space and imageio-ffmpeg, then retry."}[stage]
            self.update(job_id, state="failed", stage="Analysis stopped", error=message)
        finally:
            with self.lock:
                self.active = False


async def save_upload(upload, destination, limit):
    size = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "File is too large. Use an MP4 under 500 MiB and a JSON under 1 MiB.")
            stream.write(chunk)
    if size == 0:
        raise HTTPException(400, "The uploaded file is empty. Choose a valid file.")


def create_app(job_directory=None):
    store = JobStore(job_directory or ROOT / "artifacts/web_jobs")

    @asynccontextmanager
    async def lifespan(app):
        yield
        store.pool.shutdown(wait=True)

    app = FastAPI(title="CloudForge Parking Analysis", lifespan=lifespan)
    app.state.store = store
    app.mount("/static", StaticFiles(directory=ASSETS), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exception):
        return JSONResponse(status_code=422, content={"detail": "Select an MP4 and a parking JSON, and check the analysis settings."})

    @app.get("/")
    def home():
        return FileResponse(ASSETS / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "online", "model_file_available": (ROOT / "yolo11n.pt").is_file(),
                "classifier_available": (ROOT / "models" / "mobilenetv2_parking.pt").is_file(), "busy": store.active}

    @app.get("/api/default-config")
    def default_config():
        return FileResponse(ROOT / "config/parking_slots.json", media_type="application/json")

    @app.post("/api/validate-config")
    async def validate_config(slots: UploadFile = File(...)):
        try:
            if Path(slots.filename or "").suffix.lower() != ".json":
                raise HTTPException(415, "Choose a parking ROI configuration in JSON format.")
            with tempfile.TemporaryDirectory(prefix="cloudforge-roi-") as temporary:
                path = Path(temporary) / "slots.json"
                await save_upload(slots, path, 1024 * 1024)
                from starlette.concurrency import run_in_threadpool
                config = await run_in_threadpool(load_parking_config, path)
                return {"width": config.frame_width, "height": config.frame_height,
                        "slots": [{"id": slot.slot_id, "label": slot.label, "points": slot.polygon.tolist()}
                                  for slot in config.slots]}
        except (ValueError, UnicodeError):
            raise HTTPException(400, "Invalid ROI JSON. Use the calibration tool to export valid parking polygons and frame dimensions.")
        finally:
            await slots.close()

    @app.get("/api/insights")
    def insights():
        try:
            results = {name: json.loads((ROOT / "metrics" / f"{name}_metrics.json").read_text())
                       for name in ("baseline", "mobilenet")}
            inventory = json.loads((ROOT / "data/manifests/dataset_inventory.json").read_text())
            def table(name):
                with (ROOT / "metrics" / name).open(newline="") as stream:
                    return list(csv.DictReader(stream))
            return {"models": results, "dataset": inventory["dataset_b"],
                    "accuracy_gain_pp": 100 * (results["mobilenet"]["external"]["accuracy"] - results["baseline"]["external"]["accuracy"]),
                    "f1_gain_pp": 100 * (results["mobilenet"]["external"]["f1"] - results["baseline"]["external"]["f1"]),
                    "lighting": table("lighting_analysis.csv"), "occlusion": table("occlusion_analysis.csv"),
                    "artifacts": [{"name": name, "title": title, "url": f"/api/artifacts/{name}"}
                                  for name, title in ARTIFACTS.items() if (ROOT / "artifacts/evaluation" / name).is_file()]}
        except (OSError, ValueError, KeyError):
            raise HTTPException(503, "Evaluation files are unavailable. Restore the verified metrics and artifacts folders.")

    @app.get("/api/artifacts/{name}")
    def artifact(name: str):
        path = ROOT / "artifacts/evaluation" / name
        if name not in ARTIFACTS or not path.is_file():
            raise HTTPException(404, "Artifact not available.")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/process-video", status_code=202)
    async def start(video: UploadFile = File(...), slots: UploadFile = File(...),
                    confidence: float = Form(0.35, ge=0, le=1),
                    engine: str = Form("yolo"), classification_threshold: float = Form(0.50, gt=0, lt=1),
                    show_rois: bool = Form(True), show_ids: str = Form("auto"), show_boxes: bool = Form(True),
                    occupancy_method: str = Form("center"), overlap_threshold: float = Form(0.20, gt=0, le=1),
                    smoothing: bool = Form(True), smoothing_window: int = Form(5, ge=1, le=100),
                    smoothing_required: int = Form(3, ge=1, le=100), process_every: int = Form(1, ge=1, le=120)):
        if Path(video.filename or "").suffix.lower() != ".mp4":
            raise HTTPException(415, "Choose an MP4 video file.")
        if Path(slots.filename or "").suffix.lower() != ".json":
            raise HTTPException(415, "Choose a parking ROI configuration in JSON format.")
        if engine not in ("yolo", "classifier") or show_ids not in ("auto", "true", "false"):
            raise HTTPException(422, "Choose a valid analysis engine and label option.")
        if occupancy_method not in ("center", "overlap") or not all(map(math.isfinite, (confidence, overlap_threshold))):
            raise HTTPException(422, "Choose a valid occupancy method and finite thresholds.")
        if smoothing and smoothing_required > smoothing_window:
            raise HTTPException(422, "Smoothing votes cannot exceed the smoothing window.")
        with store.lock:
            if store.active:
                raise HTTPException(409, "An analysis is already running. Wait for it to finish before starting another.")
            store.active = True
        job_id = uuid4().hex
        directory = store.directory / job_id
        try:
            directory.mkdir()
            await save_upload(video, directory / "input.mp4", MAX_VIDEO_BYTES)
            await save_upload(slots, directory / "slots.json", 1024 * 1024)
            # Decode/validate outside the event loop; inference never runs in routes.
            from starlette.concurrency import run_in_threadpool
            metadata = await run_in_threadpool(inspect_input, directory / "input.mp4", directory / "slots.json", directory / "reference.jpg")
            slot_total = len(metadata["slots"])
            settings = dict(engine=engine, classification_threshold=classification_threshold,
                            show_rois=show_rois, show_boxes=show_boxes and engine == "yolo",
                            show_ids=(slot_total <= 50) if show_ids == "auto" else show_ids == "true",
                            confidence=confidence, occupancy_method=occupancy_method, overlap_threshold=overlap_threshold,
                            smoothing_window=smoothing_window if smoothing else 1,
                            smoothing_required=smoothing_required if smoothing else 1, process_every=process_every)
            with store.lock:
                store.jobs[job_id] = {"job_id": job_id, "state": "accepted", "stage": "Input validated", "metadata": metadata,
                                      "settings": settings, "filename": Path(video.filename).name, "frames_written": 0}
            store.pool.submit(store.run, job_id)
            return {"job_id": job_id, "metadata": metadata, "settings": settings}
        except Exception as exc:
            with store.lock:
                store.active = False
            # UUID-created directory is verified before removing rejected uploads.
            if directory.exists() and directory.resolve().parent == store.directory.resolve():
                shutil.rmtree(directory)
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, ValueError):
                text = str(exc).replace(str(directory), "uploaded files")
                raise HTTPException(400, text)
            logging.exception("Upload validation failed")
            raise HTTPException(400, "Unable to prepare this video and configuration. Check the files and free disk space.")
        finally:
            await video.close()
            await slots.close()

    @app.get("/api/status/{job_id}")
    def status(job_id: str):
        return store.get(job_id)

    @app.get("/api/result/{job_id}")
    def result(job_id: str):
        job = store.get(job_id)
        if job["state"] != "complete":
            raise HTTPException(409, job.get("error", "Analysis has not completed yet."))
        return job["result"]

    @app.get("/api/output/{job_id}")
    def output(job_id: str, download: bool = False):
        summary = result(job_id)
        path = store.directory / job_id / summary["video"]["filename"]
        if not path.is_file():
            raise HTTPException(404, "Output video is missing. Run the analysis again.")
        return FileResponse(path, media_type="video/mp4",
                            filename="cloudforge-parking.mp4" if download else None)

    @app.get("/api/files/{job_id}/{kind}")
    def job_file(job_id: str, kind: str):
        store.get(job_id)
        names = {"config": "slots.json", "reference": "reference.jpg", "csv": "analysis.csv"}
        if kind not in names:
            raise HTTPException(404, "File not found.")
        path = store.directory / job_id / names[kind]
        if not path.is_file():
            raise HTTPException(409, "This file is not ready yet.")
        return FileResponse(path, filename=path.name if kind != "reference" else None)

    return app


app = create_app()
