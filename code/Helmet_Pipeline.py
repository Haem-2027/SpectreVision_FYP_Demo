"""
Two-stage helmet detection.

  STAGE 1  find and track every person     (YOLOv8s, two passes: whole frame + full-resolution top strip)
  STAGE 2  for each person, cut out their picture and run the TRAINED helmet model
           (helmet_best.pt, classes: Helmet / No-Helmet) on that crop only.

WHY TWO STAGES
  The helmet model was trained on tight pictures of single people. Running it on the same kind
  of crop matches how it was trained, and a tiny distant helmet becomes a big clear object
  once the crop is enlarged to the model's input size.

HOW A PERSON IS JUDGED
  * Only detections on the person's HEAD area count (a neighbour's helmet is ignored).
  * "Helmet" is accepted only when the model is sure (>= HELMET_STRONG), or moderately sure
    (>= HELMET_WEAK) AND some yellow is visible inside the head box. Reason, measured on the
    demo footage: above 70% confidence a "Helmet" call contains real yellow 98% of the time,
    below that most calls are caps or yellow-green sacks. A wrong "Helmet" hides a violation,
    so it gets the higher bar. "No-Helmet" needs only NOHELMET_MIN.
  * The status shown is a majority vote over the last 5 checks (no flicker). A person with
    no accepted head detection stays grey ("not judged"): never accused.

USAGE (from the 'code' folder)
    python Helmet_Pipeline.py                      # first 5 minutes
    python Helmet_Pipeline.py --start 0 --end 120  # choose the part of the footage
    python Helmet_Pipeline.py --show               # live preview window (q to quit)
"""

import os
import csv
import time
import queue
import argparse
import threading
import subprocess
import collections
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
PERSON_MODEL = os.path.join(HERE, "..", "models", "yolov8s.pt")
HELMET_MODEL = os.path.join(HERE, "..", "models", "helmet_best.pt")
SOURCE_CANDIDATES = [
    os.path.join(HERE, "..", "source", "PPE DETECTION FOOTAGE.mp4"),
    r"D:\Work\Computer Vision\Allied Cables Project\Unit 2 Marketing video\PPE DETECTION FOOTAGE.mp4",
]
OUT_VIDEO = os.path.join(HERE, "..", "videos", "helmet_demo.mp4")
OUT_EVENTS = os.path.join(HERE, "..", "videos", "helmet_events.csv")

# ── CONFIG ────────────────────────────────────────────────────────────────────

# stage 1: people
IMGSZ_NEAR, IMGSZ_FAR, FAR_ZONE_Y = 1280, 1920, 620   # whole frame; top strip at native size   ,  imgsiz people, imgfar, distant_people,  far_zone_y  height of uper region
PERSON_CONF, NMS_IOU, MIN_BOX_H = 0.30, 0.50, 35         # p_conf minimum person detection confidence,  nms_iou thersold removing duplication detection,  min_box_H  Minimum person's bounding-box height
DETECT_EVERY = 8                                      # both stages run on every 8th frame (~3 per second) run detectionafter every  8 frames
MATCH_IOU, MIN_HITS, SHOW_MISSED, DROP_AFTER = 0.25, 2, 2, 8       

# stage 2: helmets
HELMET_IMGSZ = 640
HELMET_STRONG, HELMET_WEAK, NOHELMET_MIN = 0.70, 0.45, 0.30
YELLOW_LOOSE = ((18, 50, 140), (40, 255, 255))        # OpenCV HSV range of (even washed-out) yellow
MIN_YELLOW_PX = 8
CROP_PAD_X, CROP_PAD_Y = 0.15, 0.10
HEAD_TOP, HEAD_BOTTOM = -0.15, 0.45
MIN_PERSON_H = 40
VOTE_LEN = 5
MIN_TRACK_SAMPLES = 5
OUT_W = 1280

NAMES = {0: "Helmet", 1: "No-Helmet"}
GREEN, RED, GREY = (80, 200, 60), (60, 60, 235), (170, 170, 170)


# ── STAGE 1: people ───────────────────────────────────────────────────────────

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr):                     # nms remove duplication detection
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(iou(d[:4], k[:4]) < thr for k in keep):
            keep.append(d)
    return keep


def detect_people(model, frame, device):
    """Two passes (near + far), merged: [(x1, y1, x2, y2, conf)]. Big people are found best at a
    smaller input size, tiny distant people only at full resolution, so one setting cannot do both."""
    dets = []
    near = model.predict(frame, classes=[0], conf=PERSON_CONF, imgsz=IMGSZ_NEAR, device=device, half=False, verbose=False)[0]
    for (x1, y1, x2, y2), c in zip(near.boxes.xyxy.cpu().numpy(), near.boxes.conf.cpu().numpy()):
        dets.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    far = model.predict(frame[:FAR_ZONE_Y], classes=[0], conf=PERSON_CONF, imgsz=IMGSZ_FAR, device=device, half=False, verbose=False)[0]
    for (x1, y1, x2, y2), c in zip(far.boxes.xyxy.cpu().numpy(), far.boxes.conf.cpu().numpy()):
        if y2 < FAR_ZONE_Y - 6:                          # people cut by the strip edge are covered by pass 1
            dets.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    return [d for d in nms(dets, NMS_IOU) if d[3] - d[1] >= MIN_BOX_H]


class SimpleTracker:
    """Stable IDs by box overlap; a person is shown only once seen twice (hides one-frame false alarms)."""

    def __init__(self):
        self.tracks, self.next_id = {}, 1

    def update(self, dets):
        """Returns [(id, x1, y1, x2, y2, fresh)]; fresh = detected in THIS check."""
        
        pairs = sorted(((iou(tr["box"], d[:4]), tid, j) for tid, tr in self.tracks.items()
                        for j, d in enumerate(dets)), reverse=True)   #  enumerate(dets) gets each detection al  ONG WITH ITS INDEX
        used_t, used_d = set(), set()
        for s, tid, j in pairs:
            if s < MATCH_IOU:
                break
            if tid in used_t or j in used_d:
                continue  # SKIP THE CURENT LOOP ITERATION
            tr = self.tracks[tid]
            tr["box"], tr["hits"], tr["missed"] = tuple(dets[j][:4]), tr["hits"] + 1, 0
            used_t.add(tid)
            used_d.add(j)
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1  
                if self.tracks[tid]["missed"] > DROP_AFTER:
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            if j not in used_d:  
                self.tracks[self.next_id] = {"box": tuple(d[:4]), "hits": 1, "missed": 0}
                self.next_id += 1
        return [(tid, *map(int, tr["box"]), tr["missed"] == 0) for tid, tr in self.tracks.items()
                if tr["hits"] >= MIN_HITS and tr["missed"] <= SHOW_MISSED]


# ── STAGE 2: helmet model on each person's crop ──────────────────────────────

def yellow_pixels(frame, box):
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:                   # patch.size — number of elements in the cropped image.
        return 0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return int((cv2.inRange(hsv, YELLOW_LOOSE[0], YELLOW_LOOSE[1]) > 0).sum())


def judge_people(helmet_model, frame, people, device):
    """people: [(id, x1, y1, x2, y2)] -> {id: (class, confidence, head box in frame coords) or None}."""
    H, W = frame.shape[:2]
    crops, metas = [], []
    for pid, x1, y1, x2, y2 in people:
        w, h = x2 - x1, y2 - y1
        if h < MIN_PERSON_H:     
            continue     # skip loop iteration
        cx1, cy1 = max(0, int(x1 - CROP_PAD_X * w)), max(0, int(y1 - CROP_PAD_Y * h))
        cx2, cy2 = min(W, int(x2 + CROP_PAD_X * w)), min(H, int(y2 + CROP_PAD_Y * h))
        crops.append(frame[cy1:cy2, cx1:cx2])
        metas.append((pid, x1, y1, x2, y2, cx1, cy1))
    out = {}
    if not crops: # if no can't find person then no need of helment detection
        return out
    results = helmet_model.predict(crops, imgsz=HELMET_IMGSZ, conf=NOHELMET_MIN, device=device, half=False, verbose=False)
    for (pid, x1, y1, x2, y2, cx1, cy1), r in zip(metas, results):  #--> comine corresponding items into 2 collection
        w, h = x2 - x1, y2 - y1
        best = None
        for (bx1, by1, bx2, by2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            box = (float(bx1 + cx1), float(by1 + cy1), float(bx2 + cx1), float(by2 + cy1))
            mx, my = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not ((x1 - 0.15 * w <= mx <= x2 + 0.15 * w) and (y1 + HEAD_TOP * h <= my <= y1 + HEAD_BOTTOM * h)):
                continue                                   # not on this person's head
            k, c = int(k), float(c)  # k classid,  c --> confidence score
            if k == 0:                                     # "Helmet" must clear the higher bar
                if not (c >= HELMET_STRONG or (c >= HELMET_WEAK and yellow_pixels(frame, box) >= MIN_YELLOW_PX)):  # helment verificationlogic
                    continue
            if best is None or c > best[1]:
                best = (k, c, box)
        out[pid] = best
    return out


class PersonState:
    def __init__(self, t):
        self.first_t = self.last_t = t             # t- c.time
        self.votes = collections.deque(maxlen=VOTE_LEN)    # 0 = Helmet, 1 = No-Helmet
        self.checks = self.helmet_votes = self.bare_votes = 0

    @property         #It allows you to access a method like a normal variable. not a function
    def status(self):
        if len(self.votes) < 2:
            return "?"
        n_bare = sum(self.votes)
        n_helm = len(self.votes) - n_bare
        if n_bare == n_helm:
            return "NO HELMET" if self.votes[-1] == 1 else "HELMET"   # ternary conditional expression
        return "NO HELMET" if n_bare > n_helm else "HELMET"


# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_frame(frame, view, t):  # view Detct info
    s = OUT_W / frame.shape[1]
    out = cv2.resize(frame, (OUT_W, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)  #  s resizeing scalling
    H, W = out.shape[:2] 
    for pid, x1, y1, x2, y2, status, head in view["items"]:
        color = GREEN if status == "HELMET" else RED if status == "NO HELMET" else GREY
        cv2.rectangle(out, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), color, 1)      # person:  1--> thin  rectangle
        if head is not None:                                                                        # head: thick
            k, c, (hx1, hy1, hx2, hy2) = head
            hc = GREEN if k == 0 else RED
            p1, p2 = (int(hx1 * s), int(hy1 * s)), (int(hx2 * s), int(hy2 * s))
            cv2.rectangle(out, p1, p2, hc, 2)
            label = f"{NAMES[k]} {c:.0%}"
            (tw, th), b = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ty = max(0, p1[1] - th - b - 4)  # 4 -> extra space
            cv2.rectangle(out, (p1[0], ty), (p1[0] + tw + 6, ty + th + b + 4), hc, -1)
            cv2.putText(out, label, (p1[0] + 3, ty + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)                       # LINE_AA	Smooth text edges
    c = view["counts"]
    x, y, pw, ph = 16, 56, 300, 128                      # below the camera's own timestamp
    roi = out[y:y + ph, x:x + pw]                         # panel 
    cv2.addWeighted(np.full_like(roi, (22, 22, 22)), 0.74, roi, 0.26, 0, roi)                             # roi rehion of interest
    cv2.putText(out, "HELMET CHECK", (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(out, f"With helmet   {c['helmet']}", (x + 12, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72, GREEN, 2, cv2.LINE_AA)
    cv2.putText(out, f"No helmet     {c['bare']}", (x + 12, y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.72, RED, 2, cv2.LINE_AA)
    cv2.putText(out, f"Not judged    {c['unknown']}", (x + 12, y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    mm, ss = divmod(int(t), 60)                             # consert sec min
    cv2.putText(out, f"clip time {mm}:{ss:02d}", (W - 190, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=300.0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=OUT_VIDEO)
    args = ap.parse_args()

    source = args.source or next((p for p in SOURCE_CANDIDATES if os.path.isfile(p)), None)
    for path, what in ((source, "source footage"), (PERSON_MODEL, "person model"), (HELMET_MODEL, "helmet model")):
        if not path or not os.path.isfile(path):
            raise SystemExit(f"Missing {what}: {path}")
    cap = cv2.VideoCapture(source)                                            # cap  video capture object
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0                                # fps frame per seond
    start_f, end_f = int(args.start * fps), int(args.end * fps)

    person_model, helmet_model = YOLO(PERSON_MODEL), YOLO(HELMET_MODEL)
    device = 0 if __import__("torch").cuda.is_available() else "cpu"
    for _ in range(start_f):               # skip by reading: seeking this HEVC file gives grey frames                     
        if not cap.grab():                                      # cap.grab() = grabs the next frame without fully decoding it for use.
            break    

    tracker = SimpleTracker()
    state = {}
    view = {"items": [], "counts": {"helmet": 0, "bare": 0, "unknown": 0}}
    n_runs = 0

    ff, pool, pending, writeq = None, None, collections.deque(), queue.Queue(maxsize=32)
    wthread = None
    if not args.no_video and not args.show:
        import imageio_ffmpeg
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        ff = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{OUT_W}x{int(1080 * OUT_W / 1920)}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
            stdin=subprocess.PIPE)
        pool = ThreadPoolExecutor(max_workers=3)                                   # 3 drawing task that can run at same time

        def writer_loop():                  # frames arrive in order; ffmpeg encodes H.264
            while True:
                ov = writeq.get()
                if ov is None:
                    break
                ff.stdin.write(np.ascontiguousarray(ov).tobytes())                          # ascontiguousarray() = ensures image data is stored in a contiguous memory layout
        wthread = threading.Thread(target=writer_loop, daemon=True)
        wthread.start()

    t_wall0 = last_print = time.time()
    frame_i = start_f
    print(f"{os.path.basename(source)}: {args.start:.0f}s -> {args.end:.0f}s", flush=True)
    while frame_i < end_f:
        ok, frame = cap.read()
        if not ok:                                  # frame actual img
            break                                           # If the video frame cannot be read, stop the loop.
        t = (frame_i - start_f) / fps

        if (frame_i - start_f) % DETECT_EVERY == 0:
            shown = tracker.update(detect_people(person_model, frame, device))                  # STAGE 1
            fresh = [p[:5] for p in shown if p[5]]
            heads = judge_people(helmet_model, frame, fresh, device)                            # STAGE 2
            items = []
            for pid, x1, y1, x2, y2, is_fresh in shown:
                st = state.setdefault(pid, PersonState(t))
                head = heads.get(pid) if is_fresh else None
                if is_fresh:
                    st.last_t = t
                    if (y2 - y1) >= MIN_PERSON_H:
                        st.checks += 1
                    if head is not None:
                        st.votes.append(head[0])
                        st.helmet_votes += int(head[0] == 0)
                        st.bare_votes += int(head[0] == 1)
                items.append((pid, x1, y1, x2, y2, st.status, head))
            view = {"items": items, "counts": {
                "helmet": sum(1 for i in items if i[5] == "HELMET"),
                "bare": sum(1 for i in items if i[5] == "NO HELMET"),
                "unknown": sum(1 for i in items if i[5] == "?")}}
            n_runs += 1

        if pool is not None:
            pending.append(pool.submit(draw_frame, frame, view, t))
            while pending and (pending[0].done() or len(pending) >= 24):
                writeq.put(pending.popleft().result())
        elif args.show:
            cv2.imshow("Helmet pipeline", draw_frame(frame, view, t))
            if (cv2.waitKey(max(1, int((t_wall0 + t - time.time()) * 1000))) & 0xFF) == ord("q"):
                break

        frame_i += 1
        if time.time() - last_print >= 30:
            last_print = time.time()
            done = (frame_i - start_f) / max(1, end_f - start_f)
            el = time.time() - t_wall0
            print(f"  {done:5.1%} done  {el / 60:4.1f} min elapsed  ~{el / max(done, 1e-6) * (1 - done) / 60:.0f} min left", flush=True)

    if pool is not None:
        while pending:
            writeq.put(pending.popleft().result())
        writeq.put(None)
        wthread.join()                                      # thread  finish
        pool.shutdown()
        ff.stdin.close()
        ff.wait()
    cap.release()                                       # Closes the video file or camera connection
    cv2.destroyAllWindows()                                                 # close open cv window

    real = {p: s for p, s in state.items() if s.checks >= MIN_TRACK_SAMPLES}
    with open(OUT_EVENTS, "w", newline="") as fh:                                                       # fh file hanadle
        w = csv.writer(fh)
        w.writerow(["person_id", "first_seen_s", "last_seen_s", "checks", "helmet_votes", "no_helmet_votes"])
        for p, s in sorted(real.items()):
            w.writerow([p, round(s.first_t, 1), round(s.last_t, 1), s.checks, s.helmet_votes, s.bare_votes])
    print(f"\n=== HELMET SUMMARY ===\nFootage: {args.start:.0f}s -> {args.end:.0f}s   detector runs: {n_runs}")                                                                 #display final sumary
    if not args.no_video and not args.show:
        print("Video:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
