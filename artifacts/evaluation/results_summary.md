# CloudForge measured ML results

## DATASETS

Dataset A: Parking Lot Detection Counter. Raw: 6090 crops; used: 2755 (VACANT 1370, OCCUPIED 1385).
Train: 2440; validation: 135; test: 180; excluded: 3335.
Dataset B: CNR-EXT subset of CNRPark+EXT. External test: 4000 (VACANT 2000, OCCUPIED 2000); 9 cameras; 23 days.

OCCUPIED (1) is the positive class. Confusion matrices have true-label rows, predicted-label columns, class order [VACANT, OCCUPIED].

## BASELINE CNN

Dataset A held-out: Accuracy: 100.000% | Precision: 100.000% | Recall: 100.000% | F1: 100.000%
Epochs completed: 8; best validation-loss checkpoint: epoch 8.
Confusion matrix: [[90, 0], [0, 90]].
Dataset B external: Accuracy: 76.350% | Precision: 70.683% | Recall: 90.050% | F1: 79.200%
External confusion matrix: [[1253, 747], [199, 1801]].

## MOBILENETV2 TRANSFER LEARNING

Dataset A held-out: Accuracy: 100.000% | Precision: 100.000% | Recall: 100.000% | F1: 100.000%
Epochs completed: 7; best validation-loss checkpoint: epoch 7.
Confusion matrix: [[90, 0], [0, 90]].
Dataset B external: Accuracy: 92.225% | Precision: 98.340% | Recall: 85.900% | F1: 91.700%
External confusion matrix: [[1971, 29], [282, 1718]].

## CROSS-DATASET TEST

baseline: Dataset A F1 100.00%; Dataset B F1 79.20%. No Dataset B retraining or threshold selection.
mobilenet: Dataset A F1 100.00%; Dataset B F1 91.70%. No Dataset B retraining or threshold selection.

## LIGHTING FINDINGS

Visual-review subset: 96 unique external images, selected using seed 17 before inspecting model errors.
Tags are subjective assistant visual review of source crops, not official annotations. Weather categories are provided by the dataset; SUNNY is not equated with BRIGHT or SHADOW.
baseline / weather / OVERCAST: 959/1226 correct (78.22%).
baseline / weather / RAINY: 809/1035 correct (78.16%).
baseline / weather / SUNNY: 1286/1739 correct (73.95%).
mobilenet / weather / OVERCAST: 1149/1226 correct (93.72%).
mobilenet / weather / RAINY: 960/1035 correct (92.75%).
mobilenet / weather / SUNNY: 1580/1739 correct (90.86%).
baseline / lighting_tag / BRIGHT: 2/4 correct (50.00%). SMALL SUBSET (<30).
baseline / lighting_tag / DARK: 7/8 correct (87.50%). SMALL SUBSET (<30).
baseline / lighting_tag / NORMAL: 56/74 correct (75.68%).
baseline / lighting_tag / SHADOW: 6/10 correct (60.00%). SMALL SUBSET (<30).
mobilenet / lighting_tag / BRIGHT: 3/4 correct (75.00%). SMALL SUBSET (<30).
mobilenet / lighting_tag / DARK: 5/8 correct (62.50%). SMALL SUBSET (<30).
mobilenet / lighting_tag / NORMAL: 69/74 correct (93.24%).
mobilenet / lighting_tag / SHADOW: 10/10 correct (100.00%). SMALL SUBSET (<30).

## OCCLUSION FINDINGS

Crop-level foreground intrusion from branches/posts/other objects was visually tagged. Cutting a car at the crop edge alone is not treated as physical occlusion. An almost-black crop has unknown occlusion and is excluded from occlusion categories.
baseline / HEAVY: 6/7 correct (85.71%). SMALL SUBSET (<30).
baseline / NONE: 48/65 correct (73.85%).
baseline / PARTIAL: 17/23 correct (73.91%). SMALL SUBSET (<30).
mobilenet / HEAVY: 3/7 correct (42.86%). SMALL SUBSET (<30).
mobilenet / NONE: 63/65 correct (96.92%).
mobilenet / PARTIAL: 21/23 correct (91.30%). SMALL SUBSET (<30).

## YOLO

Model: YOLO11n pretrained.
Role: vehicle detection in the unchanged video pipeline.
mAP: NOT MEASURED ON A CUSTOM BOUNDING-BOX DATASET.
Slot masks, occupancy labels, and crop coordinates are not vehicle bounding-box ground truth.

## LIMITATIONS

Dataset A is a short, repetitive scene with constant labels within every slot-token group. Disjoint slot/frame groups and temporal embargo reduce leakage, but the test is not an unseen parking lot. Its high accuracy must be interpreted alongside external results.
Dataset B crops have different viewpoints and aspect ratios; any domain-shift cause is a hypothesis, not established by these experiments. Nearby external frames and slot views are correlated.
128x128 square resizing was used for CPU speed for both models; official pretrained ImageNet evaluation uses 224x224. Source-label issues and almost-black images are retained, not silently corrected.
Condition tags are subjective, small, and may overlap. These classifier metrics do not measure YOLO/video occupancy accuracy.
