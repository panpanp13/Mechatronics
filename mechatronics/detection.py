import onnxruntime as ort
import cv2
import threading
import queue
import numpy as np
import torch
import time
from ultralytics.utils.ops import non_max_suppression

# ─── 1) ONNX Runtime session setup ─────────────────────────────────────────
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.execution_mode          = ort.ExecutionMode.ORT_PARALLEL
so.intra_op_num_threads    = 4
so.inter_op_num_threads    = 1
sess = ort.InferenceSession("yolov8n.onnx", sess_options=so)

# ─── 2) Video source helper ─────────────────────────────────────────────────
def get_video_source(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {'camera' if source == 0 else source}")
    return cap

# ─── 3) Preprocess function ────────────────────────────────────────────────
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))  # Resize for YOLO model input
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None, :]

# ─── 4) Camera reader thread ───────────────────────────────────────────────
def camera_reader(cap, frame_q):
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_q.put(None)
            break
        frame_q.put(frame)  # Blocks if queue already has one unprocessed frame

# ─── 5) Point-in-polygon test ──────────────────────────────────────────────
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# ─── 6) Main processing loop ───────────────────────────────────────────────
def run(lane_lines, island, source, ROI):
    cap = get_video_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Precompute scaled lines & polygons once
    scaled = [
        {
            'pt1': (int(ln['pt1'][0]), int(ln['pt1'][1])),
            'pt2': (int(ln['pt2'][0]), int(ln['pt2'][1]))
        }
        for ln in lane_lines
    ]
    print("scaled",scaled)
    polygons = [
        [scaled[i]['pt1'], scaled[i]['pt2'],
         scaled[i+1]['pt2'], scaled[i+1]['pt1']]
        for i in range(len(scaled) - 1)
    ]
    print("polygon",polygons)
    # Choose frame source
    if source == 0:
        frame_q = queue.Queue(maxsize=1)
        threading.Thread(target=camera_reader, args=(cap, frame_q), daemon=True).start()
        get_frame = frame_q.get
    else:
        def get_frame():
            ret, frame = cap.read()
            return frame if ret else None

    prev_time = time.time()

    while True:
        frame = get_frame()
        if frame is None:
            break

        # ─── Resize the frame to 640x360 before processing ───────────────
        frame_resized = cv2.resize(frame, (640, 360))  # Resize the video frame to 640x360

        # ─── Run detection on full frame (no cropping for now) ────────────
        x = preprocess(frame_resized)
        outputs = sess.run(None, {"images": x})
        pred = torch.from_numpy(outputs[0])
        dets = non_max_suppression(pred, 0.25, 0.45)[0]

        centers = []
        if dets is not None:
            for *xyxy, conf, cls in dets.cpu().numpy():
                cls_id = int(cls)
                if cls_id in [2, 3, 5, 7]:  # e.g. car, motorbike, bus, truck
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Rescale from 160 → original resolution
                    x1 = int(x1 * frame_resized.shape[1] / 320)
                    y1 = int(y1 * frame_resized.shape[0] / 320)
                    x2 = int(x2 * frame_resized.shape[1] / 320)
                    y2 = int(y2 * frame_resized.shape[0] / 320)

                    # Draw red bounding box on the full frame
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Draw red label
                    label = f"{cls_id}:{conf:.2f}"
                    cv2.putText(
                        frame_resized,
                        label,
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )

                    centers.append(((x1 + x2) / 2, y2)) #BOTTOM.CENTER

        # ─── Lane occupancy check ───────────────────────
        lane_status = [False] * len(polygons)
        for p in centers:
            for idx, poly in enumerate(polygons):
                if island and idx == (island - 1):
                    continue
                if point_in_polygon(p, poly):
                    lane_status[idx] = True
                    break

        # ─── Draw lane lines ────────────────────────────
        for ln in scaled:
            cv2.line(frame_resized, ln['pt1'], ln['pt2'], (0, 255, 0), 2)

        # ─── Draw LED status box ────────────────────────
        num = len(lane_status)
        rad, mrg = 20, 10
        total_w = num * (rad * 2 + mrg)
        bx = (frame_resized.shape[1] - total_w) // 2
        cv2.rectangle(frame_resized, (bx, 0), (bx + total_w, 60), (0, 0, 0), -1)
        for i, active in enumerate(lane_status):
            cx = bx + rad + i * (2 * rad + mrg)
            cy = 30
            if island and i == (island - 1):
                cv2.circle(frame_resized, (cx, cy), rad, (0, 181, 247), -1)
                cv2.line(frame_resized, (cx - rad, cy - rad), (cx + rad, cy + rad), (0, 0, 0), 3)
                cv2.line(frame_resized, (cx - rad, cy + rad), (cx + rad, cy - rad), (0, 0, 0), 3)
            else:
                color = (0, 0, 255) if active else (0, 255, 0)
                cv2.circle(frame_resized, (cx, cy), rad, color, -1)

        # ─── Compute & overlay FPS ──────────────────────
        now = time.time()
        fps_proc = 1 / (now - prev_time)
        prev_time = now
        cv2.putText(frame_resized, f"FPS: {fps_proc:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ─── Display the final frame ────────────────────
        cv2.imshow("YOLOv8 ONNX (Optimized)", frame_resized)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    example_lane_lines = [{'x_target': 0.0, 'pt1': (246, 119), 'pt2': (87, 299)},{'x_target': 4.34614782333374, 'pt1': (262, 119), 'pt2': (184, 299)}, {'x_target': 8.995922660827636, 'pt1': (279, 119), 'pt2': (290, 300)}, {'x_target': 15.450153732299803, 'pt1': (303, 119), 'pt2': (438, 302)}, {'x_target': 19.573165130615234, 'pt1': (319, 119), 'pt2': (534, 303)}, {'x_target': 24.0, 'pt1': (336, 119), 'pt2': (639, 304)}]
    #example_lane_lines = [{'x_target': 0.0, 'pt1': (260, 76), 'pt2': (111, 351)}, {'x_target': 12.0, 'pt1': (288, 75), 'pt2': (282, 352)}, {'x_target': 24.0, 'pt1': (318, 74), 'pt2': (464, 354)}]
    ROI = np.array([(260, 76), (318, 74), (464, 354), (111, 351)])
    run(example_lane_lines, 0, "test_video/vehicles.mp4", ROI)