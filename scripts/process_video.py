"""CLI entry point; can be invoked from any working directory."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate a parking video with vehicle detections and slot availability.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--slots", default=str(ROOT / "config" / "parking_slots.json"), help="Parking polygon JSON")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum vehicle confidence, 0..1 (default: 0.35)")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained detection weights (default: yolo11n.pt)")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0 (default: automatic)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (default: 640); output resolution is preserved")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output only after successful processing")
    parser.add_argument("--occupancy-method", choices=("center", "overlap"), default="center")
    parser.add_argument("--overlap-threshold", type=float, default=0.20,
                        help="Minimum intersection area / slot area for one vehicle (default: 0.20)")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Recent detection observations (default: 5)")
    parser.add_argument("--smoothing-required", type=int, default=3, help="Positive votes required (default: 3)")
    parser.add_argument("--no-smoothing", action="store_true", help="Use raw per-observation occupancy")
    parser.add_argument("--process-every", type=int, default=1,
                        help="Run detection once every N frames; write all frames (default: 1)")
    parser.add_argument("--engine", choices=("yolo", "classifier"), default="yolo",
                        help="yolo = YOLO11n + ROI; classifier = MobileNetV2 slot classifier (default: yolo)")
    parser.add_argument("--classification-threshold", type=float, default=0.5,
                        help="Classifier mode: P(occupied) needed to mark a slot occupied (default: 0.5)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from src.video_processor import process_video

        summary = process_video(args.input, args.output, args.slots, model_path=args.model,
                                confidence=args.confidence, device=args.device,
                                image_size=args.imgsz, overwrite=args.overwrite,
                                occupancy_method=args.occupancy_method,
                                overlap_threshold=args.overlap_threshold,
                                smoothing_window=1 if args.no_smoothing else args.smoothing_window,
                                smoothing_required=1 if args.no_smoothing else args.smoothing_required,
                                process_every=args.process_every, engine=args.engine,
                                classification_threshold=args.classification_threshold)
    except ImportError as exc:
        print(f"ERROR: Missing/unusable dependency: {exc}. Run python -m pip install -r requirements.txt.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; video resources released and incomplete output discarded.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {summary.frames_written} frames to {summary.output_path} "
          f"({summary.width}x{summary.height}, {summary.fps:g} FPS).")
    print(f"Last frame: TOTAL: {summary.total_spaces} | OCCUPIED: {summary.occupied_last_frame} | "
          f"AVAILABLE: {summary.available_last_frame}")
    print(f"{args.engine} inference frames: {summary.inference_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
