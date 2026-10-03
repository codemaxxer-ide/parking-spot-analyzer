"""Measured failure tables and image contact sheets; no inferred annotation tags."""

from collections import defaultdict
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .datasets import ROOT


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label"], row["prediction"] = int(row["label"]), int(row["prediction"])
        row["confidence"] = float(row["confidence"])
    return rows


def read_analysis_tags(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    tags = {}
    for row in rows:
        if row["file_path"] in tags:
            raise ValueError("Duplicate analysis-tag image path.")
        for field, allowed in (("lighting_tag", {"NORMAL", "SHADOW", "BRIGHT", "DARK"}),
                               ("occlusion_tag", {"NONE", "PARTIAL", "HEAVY"})):
            if row.get(field) and row[field] not in allowed:
                raise ValueError(f"Invalid {field}: {row[field]}")
        if (row.get("lighting_tag") or row.get("occlusion_tag")) and not row.get("annotation_source"):
            raise ValueError("Reviewed condition tags require annotation provenance.")
        tags[row["file_path"]] = row
    return tags


def condition_metrics(rows: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[(row["model"], row["dataset_source"], row[field])].append(row)
    output = []
    for (model, dataset, condition), samples in sorted(groups.items()):
        correct = sum(row["label"] == row["prediction"] for row in samples)
        output.append({"model": model, "dataset_source": dataset, "dimension": field,
                       "condition": condition, "sample_count": len(samples), "correct": correct,
                       "incorrect": len(samples) - correct, "accuracy": correct / len(samples),
                       "small_sample_warning": len(samples) < 30})
    return output


def write_table(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        names = list(rows[0]) if rows else ["model", "condition", "sample_count", "correct", "incorrect", "accuracy"]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def contact_sheet(rows: list[dict], path: Path, title: str, *, limit: int = 32, columns: int = 4):
    if not rows:
        return False
    selected = rows[:limit]
    tile_width, tile_height, heading = 250, 235, 50
    sheet = Image.new("RGB", (columns * tile_width, heading + ((len(selected) + columns - 1) // columns) * tile_height), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
        heading_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
    except OSError:
        font = heading_font = ImageFont.load_default()
    draw.text((12, 10), title, font=heading_font, fill="#123247")
    for index, row in enumerate(selected):
        x, y = index % columns * tile_width, heading + index // columns * tile_height
        with Image.open(ROOT / row["file_path"]) as image:
            image = ImageOps.contain(image.convert("RGB"), (230, 170))
        sheet.paste(image, (x + 10, y))
        if "review_id" in row:
            lines = [f"Review {row['review_id']} | camera {row['camera']}",
                     f"{row['weather']} | {row['sequence_day']}", f"Source occupancy: {row['label']}"]
        else:
            truth = "VACANT" if row["label"] == 0 else "OCCUPIED"
            predicted = "VACANT" if row["prediction"] == 0 else "OCCUPIED"
            lines = [f"{row['model']} / {row['dataset_source']}", f"True {truth} / predicted {predicted}",
                     f"conf {row['confidence']:.2f} | {row.get('lighting_tag') or row.get('weather') or 'untagged'}"]
        for line_index, line in enumerate(lines):
            draw.text((x + 10, y + 175 + line_index * 18), line, font=font, fill="#122b39")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)
    return True
