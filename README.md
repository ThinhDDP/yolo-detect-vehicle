# Motorcycle-heavy traffic: YOLO26s-seg + O-Quickhull

A Python baseline for vehicle detection and classification in images, videos,
cameras and RTSP streams. Default checkpoint: **yolo26s-seg.pt**, the small
pretrained instance-segmentation model requested by you.

YOLO supplies vehicle classes, confidence scores and instance masks. The
O-Quickhull implementation constructs a rectilinear hull for each connected
visible mask component. It does not predict extra vehicles or replace YOLO's
classifier. Hull-based accuracy improvements have not been established on
traffic footage. This package is a starting point for local evaluation.

## Install and run

Use Python 3.10+ in a virtual environment. Install PyTorch for your CPU or GPU
first using https://pytorch.org/get-started/locally/ if you need a specific build.
Then, from the extracted `traffic_och` folder:

```bash
python -m pip install -r requirements.txt
python traffic.py --source traffic.mp4 --device 0 --output runs/traffic_01
```

The pretrained weights download on first use. Internet is needed for that
first download. Use `--device cpu` for a CPU; omit `--device` for automatic
selection. CPU processing at 1280 pixels may be slow. Start with a short clip:

```bash
python traffic.py --source traffic.mp4 --max-frames 100 --device 0 --output runs/trial
python traffic.py --source intersection.jpg --device cpu --output runs/image_01
python traffic.py --source 0 --max-frames 300 --output runs/camera_01
```

Each output directory must be new to prevent overwriting earlier results.

Default vehicle labels are `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.
The pretrained COCO model groups scooters and motorcycles under `motorcycle`;
it has no separate labels for scooter, tuk-tuk, electric bike, or van.
People are excluded from vehicle counts, so riders are not separately counted.
The motorcycle mask ordinarily represents the vehicle rather than the whole
rider-plus-vehicle silhouette.

## Dense traffic settings

The default configuration uses an input size of 1280, confidence threshold
0.20, model NMS IoU 0.70, class-aware suppression, original-resolution masks,
and a maximum of 1000 detections. These are initial settings, not calibrated
thresholds. Lower confidence can improve recall while increasing false
positives; higher NMS IoU can retain adjacent overlapping motorcycles while
also retaining duplicates.

For distant motorcycles in high-resolution footage, optionally add crops:

```bash
python traffic.py --source traffic.mp4 --device 0 --imgsz 1280 \
  --tile-size 960 --tile-overlap 0.25 --tile-nms 0.65 \
  --output runs/tiled_01
```

The script includes one full-frame pass plus overlapping crop passes and
class-aware cross-pass box-IoU deduplication. This costs additional inference
time. Crops can truncate large vehicles and box NMS can confuse crowded,
overlapping instances. Evaluate tiled and full-frame modes separately on your
footage. Tiling is disabled by default and no mask fusion is attempted.
The model's default inference head is used; this code does not force the
YOLO26 NMS-free head, so `--iou` applies to the default NMS path.

## Results

- `annotated.mp4` for videos/cameras, or `annotated.jpg` for an image.
- `detections.jsonl`: one JSON object per frame, with class counts, bounding
  boxes, confidence, visible mask area, and hull polygons.
- `metadata.json`: settings, class names, and installed Ultralytics version.

Counts are detections visible in each frame. **Do not sum them to obtain
unique vehicle totals.** Tracking and directional line-crossing counts are
not implemented. No FPS or detection accuracy claim is made.
`output_timestamp_s` refers to the output video's nominal constant-FPS
 timeline, not actual capture times or source presentation timestamps.
For streams without reported FPS, output uses 25 FPS; override with `--fps`.
The geometry is measured in image pixels, not physical road area or distance.

Each detection can have several `orthogonal_hulls`, because occlusion can
split a mask into disconnected fragments. All fragments still belong to
one detection. Component hulls may overlap; adding their areas is not the
area of their union. `visible_mask_area_px` includes all foreground mask
pixels inside the detection box; `--min-component-area` only filters hull
fragments. Its default of 1 preserves tiny visible pieces; increasing it
can remove noise but also genuine small motorcycle parts.

## How the uploaded paper is implemented

Reference: Nguyen Kieu Linh, Phan Thanh An, Tran Van Hoai (2022),
“A fast and efficient algorithm for determining the connected orthogonal
convex hulls”, Applied Mathematics and Computation 429, 127183.
https://doi.org/10.1016/j.amc.2022.127183

`orthogonal_hull.py` implements:

1. The four cases and the right orthogonal-line vertex from Section 3.
2. Strict right-side tests on both directed segments (Definition 10).
3. Farthest-point selection using squared Euclidean distance to the vertex
   (Definition 11), followed by Algorithm 1's two same-case subproblems.
4. Eight extreme endpoints and four chains from Figure 5 and Section 5.2.
5. Rectilinear reconstruction by inserting the right-line elbow between
   adjacent extreme points.

An explicit stack executes the paper's recursive subdivision and preserves
its output order. It avoids Python recursion-depth failures. The paper gives
expected O(n log n) and worst-case O(n^2) for O-Quickhull under its assumptions;
this package also deduplicates points and extracts mask components, so it is
not a reproduction of the paper's C++ timing results.

Important adaptation: the paper assumes a finite point set with no
semi-isolated points (assumption A); arbitrary fragmented masks do not justify
claiming a unique connected hull. This pipeline processes **4-connected mask
components separately**. Pixels are modeled as unit squares, and boundary
pixel cell corners form the finite point input. Image y coordinates are
negated before Cartesian geometry and restored afterward. No contour
simplification or point subsampling is used. This preserves thin structures
and axis-coordinate ties. Disconnected fragments are not bridged over an
occluder. The generic `hull_polygon(points)` function requires assumption A;
it should not be applied to arbitrary sparse or disconnected point clouds
without checking that assumption.

The hull fills horizontal/vertical concavities and holes within a component.
It is an enclosing geometric descriptor of the predicted visible mask, not
a reconstructed hidden vehicle and not a collision or road-occupancy model.
Its orientation follows the image axes. It does not estimate vehicle heading.

## Improving performance on Southeast Asian traffic

For reliable operation, annotate representative footage from your target
camera, with consistent instance masks and labels. Include packed scooters,
partial occlusion, small distant bikes, nighttime scenes, rain, and local
vehicle types. Split train/validation/test by recording session or camera,
not randomly by neighboring video frames, to reduce leakage.

If standard classes are sufficient, keep the five vehicle labels. If you
need scooter, motorbike, tuk-tuk, van, or electric-bike subtypes, define them
consistently and fine-tune. Do not infer those subclasses from hull aspect
ratio alone: perspective, riders and occlusion distort the shapes.
Detection-only box labels cannot directly train this segmentation pipeline;
use instance polygon/mask annotations in YOLO segmentation format.

After creating your segmentation dataset YAML:

```bash
yolo segment train model=yolo26s-seg.pt data=traffic-seg.yaml \
  imgsz=1280 epochs=100 device=0 batch=4

yolo segment val model=runs/segment/train/weights/best.pt \
  data=traffic-seg.yaml split=test imgsz=1280 device=0

python traffic.py --source traffic.mp4 \
  --model runs/segment/train/weights/best.pt \
  --classes bicycle car motorcycle bus truck \
  --device 0 --output runs/finetuned_01
```

Adjust batch size to available GPU memory. If you changed the taxonomy, pass
the exact names recorded in the custom checkpoint with `--classes`. Choose
confidence/NMS settings using validation data and report motorcycle precision,
recall and AP separately, plus results for small/occluded bikes. Compare
YOLO masks to hull overlays for geometric usefulness; the hull computation
alone does not improve class predictions.

## Verification

Run the included tests from this folder:

```bash
python -m unittest discover -s tests -v
```

Geometry was verified against an independent row/column-convex closure oracle
for 200 randomly generated 4-connected masks, rectangles, one-pixel shapes,
L/U shapes and holes. Farthest-point output was also compared with brute-force
quadrant maxima in all four rotations. Coordinate-offset handling is tested
with simulated model results. These tests do not measure neural-model accuracy.
Pretrained inference on real traffic footage has not been run in the creation
environment, which had no PyTorch/Ultralytics installation or sample traffic
video. Install dependencies and run the short-clip command above first.

## Official API references

- YOLO26 models: https://docs.ultralytics.com/models/yolo26/
- Segmentation weights/results: https://docs.ultralytics.com/tasks/segment/
- Prediction arguments and retina masks: https://docs.ultralytics.com/modes/predict/
- COCO labels: https://docs.ultralytics.com/datasets/detect/coco/
- Segmentation annotations: https://docs.ultralytics.com/datasets/segment/

Consult the Ultralytics license terms when distributing this project or
integrating the model into a commercial product:
https://www.ultralytics.com/license
