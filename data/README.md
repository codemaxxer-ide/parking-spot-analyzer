# CloudForge datasets and measured experiments

Labels are `0 = VACANT`, `1 = OCCUPIED`. The two crop classifiers are experiments;
the final video application still uses YOLO11n, calibrated ROIs, geometry, and
temporal smoothing. Classifier scores do not measure video occupancy accuracy.

## Dataset A: Parking Lot Detection Counter

Source: [Kaggle dataset](https://www.kaggle.com/datasets/iasadpanwhar/parking-lot-detection-counter).
The user supplied `C:\Users\srika\Downloads\archive (1).zip` (261,781,689 bytes).
It is extracted under `data/raw/dataset_a/parking/` in this repository.
No Kaggle credentials were required. No license file was found in the supplied
archive; the Kaggle reuse terms were not verified from its contents.

Actual contents:

- `clf-data/empty/`: 3,045 JPEG crops, VACANT.
- `clf-data/not_empty/`: 3,045 JPEG crops, OCCUPIED.
- `mask_1920_1080.png` and `mask_crop.png`: parking-region masks.
- Four MP4s: `parking_1920_1080.mp4`, `parking_1920_1080_loop.mp4`,
  `parking_crop.mp4`, `parking_crop_loop.mp4`.
- `util.py`: legacy crop/occupancy utility, inspected as text, not executed.
- `model/model.p`: legacy serialized classifier, not loaded or used.

There are no separate full-scene JPEGs or vehicle bounding-box annotations.
Crops are small and variable in shape; common dimensions are about 69 by 29.
The original scene video is short (849 frames at 30 FPS); loop videos repeat it.

## Dataset B: CNR-EXT from CNRPark+EXT

Chosen for occupancy labels, multiple perspectives/cameras, weather variation,
and visible shadows/occlusion. The [official dataset page](http://cnrpark.it/)
links to the [authors' archive release](https://github.com/fabiocarrara/deep-parking/releases/tag/archive).
It states Open Data Commons Open Database License (ODbL) v1.0;
the [license text](https://opendatacommons.org/licenses/odbl/1-0/) is linked in
`manifests/source_provenance.json`. Source URLs, byte sizes, and SHA-256 hashes
are recorded there. Occupancy labels and crop/slot coordinates are not vehicle
bounding-box ground truth.

Downloaded `CNR-EXT-Patches-150x150.zip` (449,502,403 bytes) and
`CNRPark+EXT.csv` (18,132,695 bytes) under `data/raw/dataset_b/`.
The archive contains 144,965 JPEG crops matched to official CSV metadata.
Only the selected 4,000 crops were extracted under `data/raw/dataset_b/subset/`.

Seed 42 selects 2,000 VACANT and 2,000 OCCUPIED crops by round-robin
weather/camera/day strata. The subset covers nine cameras and 23 days:
OVERCAST 1,226, RAINY 1,035, SUNNY 1,739. It is used exclusively as external
test data: no B training, validation, threshold selection, or mixed-data model.

## Leakage precautions and manifests

Dataset A filenames have two numeric tokens. There are 140 first-token groups
and 100 second-token groups. Frame/slot meanings are inferred from filenames;
camera, source sequence, and day metadata are absent and remain null.
Every slot-token group has a constant class label, making random crop splitting
especially misleading. With seed 42, slots are assigned by class to disjoint
70/15/15% groups, then intersected with chronological first-token ranges:

| Split | Frame-token range | VACANT | OCCUPIED | Total |
| --- | --- | ---: | ---: | ---: |
| Train | <=487 | 1,215 | 1,225 | 2,440 |
| Validation | 527–656 | 65 | 70 | 135 |
| Test | >=696 | 90 | 90 | 180 |
| Excluded | Boundary bands or incompatible slot/time group | 1,675 | 1,660 | 3,335 |

Boundaries are 507 and 676, with an embargo of 20 token units on each side.
Token units must not be interpreted as verified capture seconds. The retained
total is 2,755: 1,370 VACANT and 1,385 OCCUPIED. Paths, SHA-256 hashes, slot
tokens, and frame tokens are checked for cross-split overlap. A/B exact hashes
are also checked. Holdout examples are unseen slot/time groups from the same
repetitive scene, not an unseen parking lot. Near-duplicate appearance can
remain despite these checks; external samples are also temporally correlated.

`manifests/dataset_a.csv` includes excluded rows; only its specified split is
loaded for each stage. `manifests/dataset_b.csv` contains external rows only.
The adapters preserve source labels and join real camera/day/weather metadata.
`manifests/dataset_inventory.json` records the actual inventory and counts.

## Preprocessing, training, and evaluation

Both models use RGB, 128x128 bilinear square resizing, tensor conversion to
[0,1], and ImageNet mean `[0.485, 0.456, 0.406]` / standard deviation
`[0.229, 0.224, 0.225]`. Validation/test/external transforms are deterministic.
128x128 is a CPU speed choice; official MobileNetV2 ImageNet evaluation uses
224x224. Dataset A's aspect ratio is changed by square resizing.

Training only: mild random resized crop (scale 0.90–1.0, aspect 0.95–1.05),
rotation ±7 degrees, brightness ±0.25, contrast ±0.20, saturation ±0.12,
hue ±0.015, and 15% probability of 3x3 Gaussian blur (sigma 0.1–0.8).
No synthetic condition tags are created from these transformations.

Baseline: Conv(3→16), ReLU, MaxPool; Conv(16→32), ReLU, MaxPool;
Conv(32→64), ReLU, adaptive 4x4 pool; Linear(1024→128), ReLU,
Dropout(0.25), Linear(128→2). Eight epochs completed; best checkpoint epoch 8.

Transfer: torchvision MobileNetV2 `IMAGENET1K_V2` weights, new 1280→2 head.
Five head-training epochs with frozen features and frozen feature BatchNorm,
then two epochs with the last three feature blocks unfrozen at 0.0001 LR.
Best checkpoint: epoch 7. No fallback to random weights was used.

Both trained on CPU (CUDA unavailable), four torch threads, batch 32, seed 42,
cross-entropy, AdamW, initial LR 0.001, early-stopping patience 3, and selection
by minimum validation loss. The test/external sets do not select checkpoints.
Saved `.history.csv` files contain actual losses, validation accuracy, and epoch
times; `.training.json` contains run configuration and completed epoch counts.

Predictions use argmax of two logits; OCCUPIED is positive. Confusion matrices
have true rows and predicted columns, ordered `[VACANT, OCCUPIED]`.
`metrics/*.json` retains unrounded metrics and checkpoint/manifest hashes;
per-image prediction CSVs make the counts auditable. The external confusion
matrices are baseline `[[1253,747],[199,1801]]` and MobileNetV2
`[[1971,29],[282,1718]]`. Both A test matrices are `[[90,0],[0,90]]`.

## Lighting and occlusion review

`analysis_tags.csv` contains 96 unique external crops (48 per source class),
selected with seed 17 before inspecting individual model errors. Six numbered
review contact sheets were visually inspected. Tags record subjective assistant
visual review, not official annotations or independent human ground truth.
Source occupancy labels were not changed. Weather is official metadata and is
reported separately: SUNNY does not automatically mean BRIGHT or SHADOW.

Lighting: NORMAL 74, SHADOW 10, DARK 8, BRIGHT 4. Occlusion: NONE 65,
PARTIAL 23, HEAVY 7; one almost-black image has unknown occlusion. Visible
foreground branches/posts/people/other vehicles define occlusion; cropping a
vehicle at the image border alone does not. Unknowns are omitted from condition
groups. Small groups (<30) are flagged and do not support strong conclusions.
Tags can overlap across lighting and occlusion dimensions. Contact sheets and
condition tables use actual saved predictions, including failure examples.

## Reproduce from the repository root (PowerShell)

Install dependencies if needed: `python -m pip install -r requirements-ml.txt`.
Extract the supplied A archive to `data/raw/dataset_a/` first. Existing downloads
are reused; preparation extracts only the selected B crops.

```powershell
.\.venv\Scripts\python.exe scripts/download_cnrpark.py
.\.venv\Scripts\python.exe scripts/prepare_datasets.py --seed 42 --embargo 20 --external-count 4000
.\.venv\Scripts\python.exe scripts/train_baseline.py --epochs 8
.\.venv\Scripts\python.exe scripts/train_transfer.py --epochs 5 --fine-tune-epochs 2
.\.venv\Scripts\python.exe scripts/evaluate_models.py --only both
.\.venv\Scripts\python.exe scripts/analyze_failures.py
.\.venv\Scripts\python.exe scripts/audit_ml_results.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Keep the existing reviewed tag CSV to reproduce condition tables. For a *new*
review, `scripts/create_review_subset.py` creates blank tags and contact sheets;
it refuses to overwrite existing tags. Fill real visual-review tags and
provenance before running failure analysis; blank tags produce no fabricated
condition groups. Runtime package versions are recorded in `metrics/audit.json`.

Presentation results are in `artifacts/evaluation/results_summary.md` and its
PNG/JPEG charts/contact sheets. Custom YOLO mAP was not measured: neither
source supplies verified vehicle bounding-box ground truth. No video pipeline
redesign or optional mixed-data experiment was performed.
