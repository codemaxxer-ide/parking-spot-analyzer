"""Real YOLO/CLI check using a bundled image; not real parking-scene validation."""

import json
from pathlib import Path
import subprocess
import sys

import cv2
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector import VehicleDetector
from src.occupancy import calculate_occupancy
from src.parking_slots import load_parking_slots
from src.calibration import CalibrationState, DisplayTransform


def main():
    directory = ROOT / "artifacts" / "smoke"
    directory.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise RuntimeError("Ultralytics bundled bus.jpg is unavailable.")
    image = cv2.resize(image, (480, 640))
    input_path, output_path = directory / "input.mp4", directory / "output.mp4"
    slots_path = directory / "slots.json"
    slots_path.write_text(json.dumps({
        "bus_demo": [[5, 210], [475, 210], [475, 570], [5, 570]],
        "empty_demo": [[350, 600], [470, 600], [470, 635], [350, 635]],
    }, indent=2), encoding="utf-8")
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (480, 640))
    if not writer.isOpened():
        raise RuntimeError("Smoke input MP4 writer failed to open.")
    try:
        for _ in range(4):
            writer.write(image)
    finally:
        writer.release()
    detector = VehicleDetector(device="cpu")
    detections = detector.detect(image)
    occupancy = calculate_occupancy(load_parking_slots(slots_path), detections)
    if not any(d.class_name == "bus" for d in detections):
        raise RuntimeError("Real YOLO did not detect the bus in the bundled demo image.")
    if (occupancy.total, occupancy.occupied_count, occupancy.available_count) != (2, 1, 1):
        raise RuntimeError(f"Unexpected demo occupancy: {occupancy}")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "process_video.py"),
                             "--input", str(input_path), "--output", str(output_path),
                             "--slots", str(slots_path), "--device", "cpu", "--overwrite",
                             "--no-smoothing"],
                            check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    if "TOTAL: 2 | OCCUPIED: 1 | AVAILABLE: 1" not in result.stdout:
        raise RuntimeError("CLI counts differ from the expected 2/1/1 demo result.")
    capture = cv2.VideoCapture(str(output_path))
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 5) > 0.01:
            raise RuntimeError("Output video failed open/FPS validation.")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (640, 480):
                raise RuntimeError("Output resolution changed.")
            if count == 0:
                if not cv2.imwrite(str(directory / "preview.jpg"), frame):
                    raise RuntimeError("Cannot write smoke preview.")
                if not int(frame[570, 200, 2]) > int(frame[570, 200, 1]):
                    raise RuntimeError("Occupied polygon red annotation is missing.")
                if not int(frame[635, 400, 1]) > int(frame[635, 400, 2]):
                    raise RuntimeError("Vacant polygon green annotation is missing.")
            count += 1
        if count != 4:
            raise RuntimeError(f"Expected four output frames; decoded {count}.")
    finally:
        capture.release()
    print(f"PASS: real YOLO bus detection; counts 2/1/1; {count} decoded frames, 480x640, 5 FPS.")

    # Exercise scaled calibration edits and metadata export without pretending
    # that this bundled demo image is actual parking footage.
    state = CalibrationState(DisplayTransform.fit(480, 640, 240, 320))
    for slot in json.loads(slots_path.read_text()).values():
        for x, y in slot:
            state.add_point(round(x / 2), round(y / 2))
        if not state.complete_slot():
            raise RuntimeError(state.message)
    calibrated_path = directory / "calibrated_slots.json"
    state.save(calibrated_path, directory / "calibrated_reference.jpg", image)
    for name, options in (
        ("output_smoothed.mp4", []),
        ("output_overlap.mp4", ["--occupancy-method", "overlap", "--overlap-threshold", "0.20",
                                "--process-every", "2", "--smoothing-window", "3", "--smoothing-required", "2"]),
    ):
        output = directory / name
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "process_video.py"),
                                 "--input", str(input_path), "--output", str(output),
                                 "--slots", str(calibrated_path), "--device", "cpu", "--overwrite", *options],
                                check=True, capture_output=True, text=True)
        print(result.stdout, end="")
        if "TOTAL: 2 | OCCUPIED: 1 | AVAILABLE: 1" not in result.stdout:
            raise RuntimeError(f"Unexpected stable counts in {name}.")
        capture = cv2.VideoCapture(str(output))
        try:
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 5) > 0.01:
                raise RuntimeError(f"Cannot decode {name} at 5 FPS.")
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (640, 480):
                    raise RuntimeError(f"Resolution changed in {name}.")
                frames.append(frame)
            if len(frames) != 4:
                raise RuntimeError(f"Expected 4 frames in {name}, got {len(frames)}.")
            # Voting starts available, then establishes occupancy by frame 3.
            if not int(frames[0][570, 200, 1]) > int(frames[0][570, 200, 2]):
                raise RuntimeError(f"Startup availability annotation missing in {name}.")
            if not int(frames[-1][570, 200, 2]) > int(frames[-1][570, 200, 1]):
                raise RuntimeError(f"Stable occupied annotation missing in {name}.")
            cv2.imwrite(str(directory / (Path(name).stem + "_preview.jpg")), frames[-1])
        finally:
            capture.release()
        print(f"PASS: metadata calibration + {name}; stable counts 2/1/1; 4 frames, 480x640, 5 FPS.")
    print(f"Smoke artifacts: {directory}")


if __name__ == "__main__":
    main()
