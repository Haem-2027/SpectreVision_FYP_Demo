"""
Loading & unloading counter  -  box-carrying detection + line-crossing count.

The counting RULES are those of the original `run_box_loading_counter.py` (unchanged):

  * A custom YOLO model (box_loading.pt) detects two classes: "box" and "person carrying box".
  * Every detection is tracked across frames (ByteTrack), so each box / person keeps an ID.
  * A counting line is drawn at the truck. For each track, the side of the line its centre is
    on is remembered. When the centre moves to the OTHER side AND the real bounding box touches
    the finite line segment, that is a candidate crossing.
  * A crossing only counts as a LOADED box if a "person carrying box" was seen in the last
    CARRY_LOOKBACK_SEC seconds (so a box drifting across the line on its own is not counted),
    the track has not been counted before, and COOLDOWN_SEC has passed since the last count
    (a carrier and the box he carries cross together, and must count once, not twice).

What this file adds around those rules: GPU inference at the model's NATIVE size (imgsz=640),
a non-interactive line, a presentation overlay, a CSV log of every counted load, and H.264
video output so the result plays in a web browser.

USAGE (from the 'code' folder):
    python Loading_Counter.py                       # default demo window
    python Loading_Counter.py --start 25 --end 265  # choose the part of the footage
    python Loading_Counter.py --pick-line           # click two points to place the line yourself
"""

import os
import csv
import time
import argparse
import subprocess
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "models", "box_loading.pt")                                                  # weights variable contain location of yol trained
SOURCE_CANDIDATES = [
    os.path.join(HERE, "..", "source", "IP28_Loading - Trim.mp4"),
    r"D:\Work\Computer Vision\Allied Cables Project\Loading and Unloading\IP28_Loading - Trim.mp4",
]
OUT_VIDEO = os.path.join(HERE, "..", "videos", "loading_demo.mp4")
OUT_EVENTS = os.path.join(HERE, "..", "videos", "loading_events.csv")

# Counting line at the back of the truck, in 1920x1080 pixel coordinates.
LINE_P1 = (380, 305)         # measured: carried boxes travel from y~480 up to y~260-290 at the truck,             1st point of counting line
LINE_P2 = (700, 305)         # so a line at y=305 is crossed by every box that is loaded

IMGSZ = 640                  # the size the model was trained at. 960 makes it miss almost everything.
CARRY_LOOKBACK_SEC = 2.0     # "person carrying box" must have been seen within this many seconds
COOLDOWN_SEC = 1.5           # minimum time between two counted loads

BOX_CLASS_NAME = "box"
CARRYING_CLASS_NAME = "person carrying box"

OUT_W = 1280                 # width of the output video


# ── counting geometry (unchanged from the original) ───────────────────────────

def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (c[0] - a[0]) * (b[1] - a[1])


def segments_intersect(p1, p2, p3, p4):
    """True if segment p1->p2 crosses segment p3->p4. Works for any orientation/angle."""
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def side_of_line(p, a, b):
    """Signed area of (a, b, p): its sign says which side of the infinite line a->b the point is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def point_in_rect(p, x1, y1, x2, y2):
    return x1 <= p[0] <= x2 and y1 <= p[1] <= y2


def rect_intersects_segment(bbox, p1, p2):
    """True if the finite segment p1->p2 touches the rectangle bbox (crosses an edge, or an
    endpoint lies inside it): the real bounding-box-vs-line test, not just a centre-point test."""
    x1, y1, x2, y2 = bbox                                   #unpacking
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        if segments_intersect(p1, p2, corners[i], corners[(i + 1) % 4]):
            return True
    return point_in_rect(p1, x1, y1, x2, y2) or point_in_rect(p2, x1, y1, x2, y2)


def pick_line_interactively(frame):
    points = []
    window = "Click two points for the counting line, then press Enter/c (r to redo)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 0, 255), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points = []
        elif key in (13, ord("c")) and len(points) == 2:
            break
        elif key == ord("q"):
            raise SystemExit("Line setup cancelled")
    cv2.destroyWindow(window)
    return points[0], points[1]


# ── presentation overlay ──────────────────────────────────────────────────────

def mmss(sec):
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"                                    # f string 


def draw_overlay(frame, boxes, count, last_load_t, flash_until, t_now, t_clip, p1, p2, history):
    """boxes: [(x1,y1,x2,y2,track_id,is_carry)]. Returns the OUT_W-wide annotated frame."""
    s = OUT_W / frame.shape[1]
    out = cv2.resize(frame, (OUT_W, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
    H, W = out.shape[:2]

    for x1, y1, x2, y2, tid, carry in boxes:
        col = (60, 60, 235) if carry else (0, 200, 255)                                                         # This is called a conditional expression or ternary expression.             # red = carrying person, amber = box
        a, b = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        cv2.rectangle(out, a, b, col, 2)
        label = ("carrying" if carry else "box") + (f" #{tid}" if tid is not None else "")
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, a[1] - th - bl - 4)
        cv2.rectangle(out, (a[0], ty), (a[0] + tw + 6, ty + th + bl + 4), col, -1)
        cv2.putText(out, label, (a[0] + 3, ty + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    flashing = t_now < flash_until
    line_col = (80, 230, 80) if flashing else (60, 60, 235)
    cv2.line(out, (int(p1[0] * s), int(p1[1] * s)), (int(p2[0] * s), int(p2[1] * s)), line_col, 3)
    cv2.putText(out, "LOADING LINE", (int(p1[0] * s), int(p1[1] * s) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_col, 1, cv2.LINE_AA)
    if flashing:
        cv2.putText(out, "+1 LOADED", (int(p1[0] * s) + 60, int(p1[1] * s) + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 230, 80), 2, cv2.LINE_AA)

    # panel, top right
    pw, ph = 330, 150
    x, y = W - pw - 16, 16
    roi = out[y:y + ph, x:x + pw]
    cv2.addWeighted(np.full_like(roi, (22, 22, 22)), 0.74, roi, 0.26, 0, roi)
    cv2.putText(out, "BOXES LOADED", (x + 14, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(out, str(count), (x + 14, y + 92), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (80, 230, 80), 4, cv2.LINE_AA)
    last = f"last load  {mmss(last_load_t)}" if last_load_t is not None else "no load yet"
    cv2.putText(out, last, (x + 14, y + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, f"clip time  {mmss(t_clip)}", (x + 14, y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

    # legend, bottom left
    cv2.rectangle(out, (16, H - 52), (28, H - 40), (0, 200, 255), -1)
    cv2.putText(out, "box", (34, H - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(out, (84, H - 52), (96, H - 40), (60, 60, 235), -1)
    cv2.putText(out, "person carrying box", (102, H - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--start", type=float, default=25.0, help="start second of the footage")
    ap.add_argument("--end", type=float, default=265.0, help="end second of the footage")
    ap.add_argument("--pick-line", action="store_true")
    ap.add_argument("--out", default=OUT_VIDEO)
    ap.add_argument("--no-video", action="store_true", help="count only, no video (fast)")
    args = ap.parse_args()

    source = args.source or next((p for p in SOURCE_CANDIDATES if os.path.isfile(p)), None)
    if not source or not os.path.isfile(source):
        raise SystemExit("Source footage not found. Pass --source path\\to\\IP28_Loading.mp4")
    if not os.path.isfile(WEIGHTS):
        raise SystemExit(f"Model not found: {WEIGHTS}")

    model = YOLO(WEIGHTS)
    names = model.names
    box_idx = {i for i, n in names.items() if n.lower() == BOX_CLASS_NAME}
    carry_idx = {i for i, n in names.items() if n.lower() == CARRYING_CLASS_NAME}
    countable = box_idx | carry_idx

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f, end_f = int(args.start * fps), min(total, int(args.end * fps))
    p1, p2 = LINE_P1, LINE_P2
    if args.pick_line:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ok, f0 = cap.read()
        p1, p2 = pick_line_interactively(f0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    print(f"source: {os.path.basename(source)}  {args.start:.0f}s -> {args.end:.0f}s   line {p1} -> {p2}", flush=True)

    # H.264 writer through the ffmpeg bundled with imageio-ffmpeg (plays in every browser)
    ff = None
    if not args.no_video:
        import imageio_ffmpeg
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        W_out, H_out = OUT_W, int(1080 * OUT_W / 1920)
        ff = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W_out}x{H_out}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
             "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)

    carry_lookback_frames = max(1, int(CARRY_LOOKBACK_SEC * fps))
    cooldown_frames = max(1, int(COOLDOWN_SEC * fps))
    recent_carrying = deque(maxlen=carry_lookback_frames)
    track_last_side, track_prev_bbox = {}, {}
    counted_track_ids = set()
    cooldown_remaining = 0
    loaded_count = 0
    last_load_t, flash_until = None, 0.0
    events = []

    for _ in range(start_f):                 # skip to the start by reading (safe for any codec)
        if not cap.grab():
            break
    t_wall0 = last_print = time.time()
    frame_i = start_f

    while frame_i < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        t_clip = (frame_i - start_f) / fps

        result = model.track(frame, persist=True, device=0 if __import__("torch").cuda.is_available() else "cpu",
                             imgsz=IMGSZ, verbose=False, tracker="bytetrack.yaml")[0]
        carrying_seen_this_frame = False
        crossed_this_frame = False
        draw_list = []

        if result.boxes is not None and result.boxes.id is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            for (x1, y1, x2, y2), track_id, cls_id in zip(xyxy, track_ids, class_ids):
                bbox = (x1, y1, x2, y2)
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                is_carry = cls_id in carry_idx
                if is_carry:
                    carrying_seen_this_frame = True
                draw_list.append((x1, y1, x2, y2, track_id, is_carry))

                current_side = side_of_line(centroid, p1, p2)
                prev_side = track_last_side.get(track_id)
                prev_bbox = track_prev_bbox.get(track_id)
                side_flipped = (prev_side is not None and current_side != 0 and (prev_side > 0) != (current_side > 0))

                if side_flipped and (rect_intersects_segment(bbox, p1, p2)
                                     or (prev_bbox is not None and rect_intersects_segment(prev_bbox, p1, p2))):
                    if (cls_id in countable and track_id not in counted_track_ids
                            and cooldown_remaining == 0 and any(recent_carrying)):
                        loaded_count += 1
                        counted_track_ids.add(track_id)
                        crossed_this_frame = True
                        last_load_t, flash_until = t_clip, t_clip + 1.2
                        events.append((round(t_clip, 2), round(frame_i / fps, 2), loaded_count, track_id, names[cls_id]))
                        print(f"[+] load #{loaded_count} at clip {mmss(t_clip)} (source {mmss(frame_i / fps)})", flush=True)

                if current_side != 0:
                    track_last_side[track_id] = current_side
                track_prev_bbox[track_id] = bbox

        recent_carrying.append(carrying_seen_this_frame)
        if crossed_this_frame:
            cooldown_remaining = cooldown_frames
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

        if ff is not None:
            ov = draw_overlay(frame, draw_list, loaded_count, last_load_t, flash_until, t_clip, t_clip, p1, p2, None)
            ff.stdin.write(np.ascontiguousarray(ov).tobytes())

        frame_i += 1
        if time.time() - last_print >= 30:
            last_print = time.time()
            done = (frame_i - start_f) / max(1, end_f - start_f)
            el = time.time() - t_wall0
            print(f"  {done:5.1%} done  {el / 60:4.1f} min elapsed  ~{el / max(done, 1e-6) * (1 - done) / 60:.0f} min left", flush=True)

    cap.release()
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    with open(OUT_EVENTS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_time_s", "source_time_s", "load_number", "track_id", "class"])
        w.writerows(events)
    print(f"\n=== LOADING SUMMARY ===\nBoxes counted as loaded: {loaded_count}")
    print(f"Window: source {args.start:.0f}s -> {args.end:.0f}s   ({(args.end - args.start) / 60:.1f} min)")
    if not args.no_video:
        print("Video:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
