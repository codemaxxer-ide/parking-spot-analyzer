"""Click arbitrary parking polygons and export original-resolution ROIs."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate parking polygons using a video reference frame.")
    parser.add_argument("--input", required=True, help="Local parking video")
    parser.add_argument("--output", default=str(ROOT / "config" / "parking_slots.json"), help="JSON output path")
    parser.add_argument("--reference", default=str(ROOT / "artifacts" / "parking_reference.jpg"),
                        help="Annotated reference JPEG path")
    parser.add_argument("--max-width", type=int, default=1280, help="Preview window width limit (>=700)")
    parser.add_argument("--max-height", type=int, default=720, help="Preview window height limit (>112)")
    args = parser.parse_args()
    try:
        from src.calibration import run_calibration

        run_calibration(args.input, args.output, args.reference,
                        max_width=args.max_width, max_height=args.max_height)
    except ImportError as exc:
        print(f"ERROR: {exc}. Install requirements.txt in your active environment.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Calibration interrupted; window closed and unsaved edits discarded.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
