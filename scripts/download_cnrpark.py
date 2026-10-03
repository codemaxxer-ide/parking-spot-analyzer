"""Download the author's CNR-EXT crop archive and official label metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests


BASE_URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/dataset_b"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for filename in ("CNRPark+EXT.csv", "CNR-EXT-Patches-150x150.zip"):
        target = args.output / filename
        url = BASE_URL + filename
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            with requests.get(url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("Content-Length", 0))
                if expected > 500_000_000:
                    raise RuntimeError("Unexpected download size exceeds the 500 MB per-file limit.")
                written, last_report = 0, time.monotonic()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        if written > 500_000_000:
                            raise RuntimeError("Download exceeds the verified size limit.")
                        if time.monotonic() - last_report > 10:
                            print(f"{filename}: {written / 1e6:.1f} / {expected / 1e6:.1f} MB", flush=True)
                            last_report = time.monotonic()
                if expected and written != expected:
                    raise RuntimeError("Incomplete download; rerun to retry.")
            partial.replace(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append({"url": url, "file": filename, "bytes": target.stat().st_size, "sha256": digest})
        print(f"Available: {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    with zipfile.ZipFile(args.output / "CNR-EXT-Patches-150x150.zip") as archive:
        print(f"Archive contains {len(archive.infolist())} entries; preparation extracts only the selected subset.")
    (args.output / "source.json").write_text(json.dumps({
        "dataset": "CNR-EXT subset of CNRPark+EXT", "official_page": "http://cnrpark.it/",
        "license_as_stated_by_official_page": "Open Data Commons Open Database License (ODbL) v1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/", "downloads": sources,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
