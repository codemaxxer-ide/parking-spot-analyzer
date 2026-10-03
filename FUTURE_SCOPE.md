# CloudForge — Future Scope & Limitations

CloudForge — Vacant Parking Space Analyser · TEAM-13 · Problem ID JIG26_11

## Known limitations

**Evidence**
- The only end-to-end video run on real footage used Dataset A's top-down clip. The MobileNetV2 slot classifier was trained on crops from that same video, so the result shows it works on this scene, not that it generalizes. The generalization evidence is the external CNR-EXT crop score (92.225% accuracy, 85.90% occupied recall).
- Crop-classifier scores do not measure video occupancy accuracy. No per-slot ground truth exists for the demo video, so video accuracy is not measured.
- Custom YOLO mAP is not measured; no vehicle bounding-box ground truth exists.

**Detection**
- Pretrained YOLO11n has limited sensitivity to tiny, top-down vehicles. On the Dataset A video it found essentially none, which is why the classifier engine exists.
- The classifier recalls 85.90% of occupied crops externally, so some occupied spaces are predicted vacant. Heavily occluded slots are weakest (3/7 correct in the reviewed set).

**Configuration**
- ROIs assume a fixed camera and the video's original resolution; moving the camera needs recalibration.
- `config/dataset_a_slots.json` is derived from the dataset's mask, not hand-calibrated.
- Dataset B (CNR-EXT) has crops and a CSV only, so there is no video demo for it.

**Application**
- One analysis runs at a time; jobs are held in memory and lost on server restart.
- Reported counts describe the final processed frame, not a live per-frame readout.
- Processing is CPU-bound: about 1.4 s per frame for 396 slots with the classifier, so long footage needs "process every N frames".
- Output video has no audio. If H.264 conversion fails, the original OpenCV MP4 is kept for download.
- No authentication; the server is meant for local use.

## Future scope

1. **Real-footage validation** — hand-label a sample of frames from a deployed camera and report per-slot accuracy, precision and recall for both engines.
2. **Fine-tune on target views** — train the slot classifier on crops from the deployment camera, and fine-tune a detector for top-down vehicles so YOLO mode works on aerial footage.
3. **Live streams** — accept RTSP/CCTV input and update availability continuously instead of processing uploaded files.
4. **Automatic slot discovery** — detect parking bays automatically instead of manual ROI clicking.
5. **Robustness** — handle night, rain, shadows and occlusion with targeted augmentation and per-slot confidence thresholds, flagging uncertain slots instead of guessing.
6. **Faster inference** — GPU, ONNX/TensorRT export and frame-difference skipping to reach real-time on large lots.
7. **Driver-facing features** — availability map, nearest-free-space guidance and occupancy history/analytics.
8. **Deployment** — persistent job storage, a multi-user queue, authentication and a hosted version.
