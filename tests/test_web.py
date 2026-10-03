"""Local API boundaries and browser output, reusing the real CV pipeline."""

import json
from pathlib import Path
import subprocess
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

from src.detector import Detection
from src.video_processor import process_video
from web.app import create_app, encode_browser_video, find_ffmpeg, video_properties


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "input.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 240))
        for _ in range(4):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()
        self.roi = {"frame_width": 320, "frame_height": 240, "slots": [
            {"id": "P1", "points": [[10, 100], [100, 100], [100, 220], [10, 220]]},
            {"id": "P2", "points": [[150, 100], [250, 100], [250, 220], [150, 220]]}]}
        self.app = create_app(self.root / "jobs")
        self.client = TestClient(self.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def files(self, video_name="input.mp4", video=None, slots=None):
        return {"video": (video_name, self.video.read_bytes() if video is None else video, "video/mp4"),
                "slots": ("slots.json", json.dumps(self.roi).encode() if slots is None else slots, "application/json")}

    def completed(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/status/{job_id}").json()
            if job["state"] in ("complete", "failed"):
                return job
            time.sleep(.01)
        self.fail("Background job did not finish")

    def test_health_page_and_static_script(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "online")
        self.assertIn("CloudForge", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_verified_insights_and_artifacts(self):
        data = self.client.get("/api/insights").json()
        self.assertEqual(data["models"]["mobilenet"]["external"]["accuracy"], .92225)
        self.assertEqual(data["dataset"]["selected"], 4000)
        for artifact in data["artifacts"]:
            self.assertEqual(self.client.get(artifact["url"]).status_code, 200)
        self.assertEqual(self.client.get("/api/artifacts/not-allowed.json").status_code, 404)

    def test_invalid_video_extension(self):
        self.assertEqual(self.client.post("/api/process-video", files=self.files(video_name="x.exe")).status_code, 415)

    def test_missing_roi_file(self):
        files = self.files(); del files["slots"]
        response = self.client.post("/api/process-video", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()["detail"], str)

    def test_malformed_roi_and_invalid_video_clean_up(self):
        for files in (self.files(slots=b"{bad}"), self.files(video=b"not a video"), self.files(video=b"")):
            self.assertEqual(self.client.post("/api/process-video", files=files).status_code, 400)
            self.assertFalse(self.app.state.store.active)
            self.assertFalse(list((self.root / "jobs").iterdir()))

    def test_roi_resolution_mismatch(self):
        self.roi["frame_width"] = 640
        response = self.client.post("/api/process-video", files=self.files())
        self.assertEqual(response.status_code, 400)
        self.assertIn("calibrated at", response.json()["detail"])

    def test_invalid_settings(self):
        for data in ({"occupancy_method": "invented"}, {"smoothing_required": 6, "smoothing_window": 5},
                     {"process_every": 0}, {"confidence": "nan"}, {"overlap_threshold": 0}):
            self.assertEqual(self.client.post("/api/process-video", files=self.files(), data=data).status_code, 422)

    def test_upload_limit(self):
        with patch("web.app.MAX_VIDEO_BYTES", 10):
            self.assertEqual(self.client.post("/api/process-video", files=self.files()).status_code, 413)
        self.assertFalse(list((self.root / "jobs").iterdir()))

    def test_roi_validation_uses_existing_polygon_rules(self):
        valid = self.client.post("/api/validate-config", files={"slots": self.files()["slots"]})
        self.assertEqual(len(valid.json()["slots"]), 2)
        for contents in (b"{}", b"{bad", b'{"P1":[[0,0],[1,1],[2,2]]}', b'\xff'):
            response = self.client.post("/api/validate-config", files={"slots": ("slots.json", contents)})
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(self.temp.name, response.text)

    def test_background_progress_busy_guard_and_fallback_output(self):
        entered, release = Event(), Event()
        detector = Mock()
        def detect(frame):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test timed out")
            return [Detection((20, 120, 80, 180), "car", .9)]
        detector.detect.side_effect = detect
        with patch("src.detector.VehicleDetector", return_value=detector), patch("web.app.find_ffmpeg", return_value=None):
            try:
                response = self.client.post("/api/process-video", files=self.files(), data={"smoothing": "false"})
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.client.get(f"/api/status/{job_id}").json()["state"], "processing")
                self.assertEqual(self.client.get(f"/api/output/{job_id}").status_code, 409)
                self.assertEqual(self.client.post("/api/process-video", files=self.files()).status_code, 409)
                self.assertEqual(self.client.get("/api/health").status_code, 200)
            finally:
                release.set()
            job = self.completed(job_id)
        self.assertEqual(job["state"], "complete")
        self.assertEqual(job["frames_written"], 4)
        self.assertEqual(job["occupied"], {"P1": True, "P2": False})
        result = job["result"]
        self.assertEqual((result["total"], result["occupied"], result["available"]), (2, 1, 1))
        self.assertFalse(result["video"]["browser_playable"])
        self.assertIn("FFmpeg unavailable", result["video"]["warning"])
        output = self.client.get(f"/api/output/{job_id}?download=true")
        self.assertEqual(output.status_code, 200)
        self.assertIn("attachment", output.headers["content-disposition"])
        self.assertEqual(self.client.get(f"/api/output/{job_id}", headers={"Range": "bytes=0-99"}).status_code, 206)
        self.assertIn("OCCUPIED", self.client.get(f"/api/files/{job_id}/csv").text)
        (self.root / "jobs" / job_id / "annotated.mp4").unlink()
        self.assertEqual(self.client.get(f"/api/output/{job_id}").status_code, 404)

    def test_model_failure_is_friendly_and_releases_worker(self):
        with patch("src.detector.VehicleDetector", side_effect=RuntimeError("private stack detail")), self.assertLogs(level="ERROR"):
            response = self.client.post("/api/process-video", files=self.files())
            job = self.completed(response.json()["job_id"])
        self.assertEqual(job["state"], "failed")
        self.assertIn("YOLO model unavailable", job["error"])
        self.assertNotIn("private stack", json.dumps(job))
        self.assertFalse(self.app.state.store.active)

    def test_processing_failure_is_friendly(self):
        with patch("src.detector.VehicleDetector"), patch("web.app.process_video", side_effect=RuntimeError("private detail")), self.assertLogs(level="ERROR"):
            response = self.client.post("/api/process-video", files=self.files())
            job = self.completed(response.json()["job_id"])
        self.assertEqual(job["state"], "failed")
        self.assertIn("Video processing failed", job["error"])

    def test_unknown_job_and_file_kind(self):
        self.assertEqual(self.client.get("/api/status/unknown").status_code, 404)
        self.assertEqual(self.client.get("/api/output/unknown").status_code, 404)

    def test_callback_reports_all_frames_and_actual_occupancy(self):
        slots = self.root / "slots.json"; slots.write_text(json.dumps(self.roi))
        detector = Mock(); detector.detect.return_value = [Detection((20, 120, 80, 180), "car", .9)]
        snapshots = []
        process_video(self.video, self.root / "processed.mp4", slots, detector=detector,
                      process_every=2, progress_callback=snapshots.append)
        self.assertEqual([s["frames_written"] for s in snapshots], [1, 2, 3, 4])
        self.assertEqual(snapshots[-1]["expected_frames"], 4)
        self.assertEqual(snapshots[-1]["occupied"], {"P1": True, "P2": False})

    def test_no_ffmpeg_preserves_source(self):
        original = self.video.read_bytes()
        with patch("web.app.find_ffmpeg", return_value=None):
            result = encode_browser_video(self.video, self.root / "browser.mp4")
        self.assertFalse(result["browser_playable"])
        self.assertEqual(self.video.read_bytes(), original)

    def test_failed_conversion_preserves_source_and_previous_destination(self):
        destination = self.root / "browser.mp4"; destination.write_bytes(b"previous")
        original = self.video.read_bytes()
        def fail(command, **kwargs):
            Path(command[-1]).write_bytes(b"incomplete")
            raise subprocess.CalledProcessError(1, command)
        with patch("web.app.find_ffmpeg", return_value="ffmpeg"), patch("web.app.subprocess.run", side_effect=fail), self.assertLogs(level="ERROR"):
            result = encode_browser_video(self.video, destination)
        self.assertFalse(result["browser_playable"])
        self.assertEqual(destination.read_bytes(), b"previous")
        self.assertEqual(self.video.read_bytes(), original)
        self.assertFalse(list(self.root.glob(".browser-*.mp4")))

    def test_real_h264_conversion_and_compatible_video_skips_reencoding(self):
        if not find_ffmpeg():
            self.skipTest("FFmpeg unavailable; fallback covered separately")
        destination = self.root / "browser.mp4"
        result = encode_browser_video(self.video, destination)
        self.assertTrue(result["browser_playable"])
        before, after = video_properties(self.video), video_properties(destination)
        self.assertEqual(before["size"], after["size"])
        self.assertEqual(before["frames"], after["frames"])
        self.assertAlmostEqual(before["fps"], after["fps"])
        self.assertIn(after["codec"], ("avc1", "h264"))
        with patch("web.app.subprocess.run") as run:
            result = encode_browser_video(destination, self.root / "copy.mp4")
        run.assert_not_called()
        self.assertEqual(result["filename"], destination.name)

    def test_same_output_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_browser_video(self.video, self.video)


if __name__ == "__main__":
    unittest.main()
