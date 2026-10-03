"""Vehicle instance segmentation + paper-based component orthogonal hulls."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from orthogonal_hull import mask_hulls


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Image, video, camera index, or RTSP URL')
    p.add_argument('--model', default='yolo26x-seg.pt')
    p.add_argument('--output', default='runs/traffic')
    p.add_argument('--classes', nargs='+', default=['bicycle','car','motorcycle','bus','truck'],
                   help='Exact names in model.names; replace for custom checkpoints')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--conf', type=float, default=0.20)
    p.add_argument('--iou', type=float, default=0.70, help='Model NMS threshold')
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--device', default=None, help='0 for CUDA, cpu, or mps')
    p.add_argument('--min-component-area', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0, help='0 means unlimited')
    p.add_argument('--fps', type=float, default=None, help='Override output video FPS')
    p.add_argument('--tile-size', type=int, default=0, help='Optional source-pixel tile size')
    p.add_argument('--tile-overlap', type=float, default=0.25)
    p.add_argument('--tile-nms', type=float, default=0.65,
                   help='Class-aware cross-tile box IoU duplicate threshold')
    return p


def starts(length, size, overlap):
    if length <= size:
        return [0]
    step = max(1, round(size*(1-overlap)))
    return sorted(set([*range(0, length-size+1, step), length-size]))


def box_iou(a, b):
    x1, y1 = max(a[0],b[0]), max(a[1],b[1])
    x2, y2 = min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,x2-x1)*max(0,y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(aa+bb-intersection, 1e-9)


def deduplicate(detections, threshold):
    kept = []
    for d in sorted(detections, key=lambda v:v['confidence'], reverse=True):
        if any(d['class_id'] == k['class_id'] and box_iou(d['bbox_xyxy'],k['bbox_xyxy']) > threshold
               for k in kept):
            continue
        kept.append(d)
    return kept


def infer(model, frame, args, class_ids):
    h, w = frame.shape[:2]
    # Always include full-frame context. Tiles add close-up passes for distant
    # motorcycles. Sequential calls bound GPU memory to a single crop.
    crops = [(0,0,w,h)]
    if args.tile_size and (w > args.tile_size or h > args.tile_size):
        s = args.tile_size
        crops += [(x,y,min(x+s,w),min(y+s,h))
                  for y in starts(h,s,args.tile_overlap)
                  for x in starts(w,s,args.tile_overlap)]
    detections = []
    for x,y,x2,y2 in crops:
        crop = frame[y:y2,x:x2]
        kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                      max_det=args.max_det, classes=class_ids,
                      retina_masks=True, agnostic_nms=False, verbose=False)
        if args.device is not None:
            kwargs['device'] = args.device
        result = model.predict(crop, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        if result.masks is None:
            raise RuntimeError('Detections have no masks; use an instance-segmentation checkpoint')
        if tuple(result.masks.data.shape[1:]) != crop.shape[:2]:
            raise RuntimeError('retina_masks must return masks in original crop coordinates')
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for i, (box, score, cls) in enumerate(zip(boxes,scores,classes)):
            # Transfer/crop ONE mask at a time; avoid storing N full-frame masks.
            bx1,by1 = np.floor(box[:2]).astype(int)
            bx2,by2 = np.ceil(box[2:]).astype(int)
            bx1,by1 = max(0,bx1),max(0,by1)
            bx2,by2 = min(crop.shape[1],bx2),min(crop.shape[0],by2)
            mask = (result.masks.data[i,by1:by2,bx1:bx2].cpu().numpy() > 0.5)
            components = mask_hulls(mask, args.min_component_area) if mask.size else []
            for component in components:
                component['polygon'] = (np.array(component['polygon']) + [x+bx1,y+by1]).tolist()
            detections.append(dict(class_id=int(cls), class_name=model.names[int(cls)],
                                   confidence=float(score),
                                   bbox_xyxy=(box+[x,y,x,y]).tolist(),
                                   visible_mask_area_px=int(mask.sum()),
                                   orthogonal_hulls=components))
    return deduplicate(detections,args.tile_nms) if len(crops)>1 else detections


def annotate(frame, detections):
    import cv2
    out = frame.copy()
    for d in detections:
        cls = d['class_id']
        color = ((37*cls+80)%230+25, (71*cls+30)%230+25, (113*cls+10)%230+25)
        for c in d['orthogonal_hulls']:
            points = np.rint(c['polygon']).astype(np.int32)
            cv2.polylines(out,[points],True,color,2)
        x1,y1,x2,y2 = np.rint(d['bbox_xyxy']).astype(int)
        # cv2.rectangle(out,(x1,y1),(x2,y2),color,1)
        cv2.putText(out,f"{d['class_name']} {d['confidence']:.2f}",
                    (max(0,x1),max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
    counts = Counter(d['class_name'] for d in detections)
    text = 'In frame: ' + ', '.join(f'{k}={v}' for k,v in sorted(counts.items()))
    cv2.putText(out,text,(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
    return out


def main():
    args = get_parser().parse_args()
    if not (0 < args.conf <= 1 and 0 < args.iou <= 1 and 0 < args.tile_nms <= 1):
        raise SystemExit('Confidence and IoU thresholds must be in (0,1]')
    if (args.imgsz < 32 or args.max_det < 1 or args.min_component_area < 1
        or args.tile_size < 0 or not 0 <= args.tile_overlap < 1 or args.max_frames < 0
        or (args.fps is not None and args.fps <= 0)):
        raise SystemExit('Invalid size, count, overlap, or FPS argument')
    import cv2
    import ultralytics
    from ultralytics import YOLO
    model = YOLO(args.model)
    if model.task != 'segment':
        raise SystemExit('Use a -seg.pt checkpoint or custom instance segmentation weights')
    name_to_id = {v:k for k,v in model.names.items()}
    missing = set(args.classes)-name_to_id.keys()
    if missing:
        raise SystemExit(f'Classes not in checkpoint: {sorted(missing)}. Available: {model.names}')
    ids = [name_to_id[n] for n in args.classes]
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)  # Prevent accidental overwrite.
    image_extensions = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
    is_image = Path(args.source).suffix.lower() in image_extensions
    cap = writer = None
    frame_number = 0
    started = time.perf_counter()
    metadata = {**vars(args), 'ultralytics_version':ultralytics.__version__,
                'class_names':model.names, 'counts_meaning':'per-frame detections, not unique vehicles',
                'hull_input':'pixel cell corners, each 4-connected component separately'}
    # Do not serialize credentials possibly embedded in a stream URL.
    if '://' in metadata['source']:
        metadata['source'] = '<stream URL redacted>'
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    try:
        if is_image:
            first = cv2.imread(args.source)
            if first is None:
                raise RuntimeError('Could not read source image')
            fps = None
        else:
            source = int(args.source) if args.source.isdecimal() else args.source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError('Could not open video/camera/stream')
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = args.fps or (native_fps if np.isfinite(native_fps) and native_fps>0 else 25.0)
        with (output/'detections.jsonl').open('w') as log:
            while True:
                if is_image:
                    if frame_number:
                        break
                    frame = first
                else:
                    ok,frame = cap.read()
                    if not ok:
                        break
                detections = infer(model,frame,args,ids)
                counts = Counter(d['class_name'] for d in detections)
                row = dict(frame=frame_number, output_timestamp_s=None if is_image else frame_number/fps,
                           counts=dict(counts), detections=detections)
                log.write(json.dumps(row)+'\n')
                rendered = annotate(frame,detections)
                if is_image:
                    if not cv2.imwrite(str(output/'annotated.jpg'),rendered):
                        raise RuntimeError('Failed to write annotated image')
                else:
                    if writer is None:
                        writer = cv2.VideoWriter(str(output/'annotated.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('MP4 encoder unavailable')
                    writer.write(rendered)
                frame_number += 1
                if frame_number % 30 == 0:
                    print(f'{frame_number} frames processed',flush=True)
                if args.max_frames and frame_number >= args.max_frames:
                    break
        if not frame_number:
            raise RuntimeError('Source yielded no frames')
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
    print(f'Saved {frame_number} frames in {time.perf_counter()-started:.1f}s to {output}')


if __name__ == '__main__':
    main()
