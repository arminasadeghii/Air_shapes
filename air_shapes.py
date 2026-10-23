import math
import os
import time
import glob
import cv2
import numpy as np
import mediapipe as mp

def hx(s):
    return (int(s[5:7], 16), int(s[3:5], 16), int(s[1:3], 16))


BASE, MANTLE, CRUST = hx("#1e1e2e"), hx("#181825"), hx("#11111b")
SURF0, SURF1, SURF2 = hx("#313244"), hx("#45475a"), hx("#585b70")
LAV, MAUVE, PEACH, RED = hx("#730000B4"), hx("#730000B4"), hx("#730000B4"), hx("#730000B4")
SHAPE_COLOR = {
    "circle": hx("#730000B4"), "square": hx("#730000B4"), "rectangle": hx("#730000B4"),
    "triangle": hx("#730000B4"), "star": hx("#730000B4"),
    "heart": hx("#730000B4")}

PW, PH, M = 640, 480, 24
W, H = M * 3 + PW * 2, M * 2 + PH


def smooth_closed(pts, k=2):
    """Moving average over a closed stroke to kill hand jitter."""
    out = np.zeros_like(pts)
    for i in range(-k, k + 1):
        out += np.roll(pts, i, axis=0)
    return out / (2 * k + 1)


def deep_defects(stroke_xy, size, min_depth=0.08):
    """Return [(point, depth_fraction)] for significant concavities."""
    c = np.round(stroke_xy).astype(np.int32).reshape(-1, 1, 2)
    try:
        hidx = cv2.convexHull(c, returnPoints=False)
        d = cv2.convexityDefects(c, hidx)
    except cv2.error:          # self-intersecting stroke
        return []
    if d is None:
        return []
    out = []
    for s, e, f, depth in d[:, 0]:
        frac = (depth / 256.0) / size
        if frac > min_depth:
            out.append((c[f, 0], frac))
    return out


def classify(stroke):
    if len(stroke) < 12:
        return None
    pts = np.array(stroke, np.float32).reshape(-1, 1, 2)
    per = cv2.arcLength(pts, False)
    if per < 150:
        return None
    if np.linalg.norm(pts[0, 0] - pts[-1, 0]) > per * 0.30:
        return None
    hull = cv2.convexHull(pts)
    ha, hp = cv2.contourArea(hull), cv2.arcLength(hull, True)
    if ha < 1:
        return None
    solidity = abs(cv2.contourArea(pts)) / ha
    circ = 4 * math.pi * ha / (hp * hp)
    n = len(cv2.approxPolyDP(hull, 0.045 * hp, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    ratio = max(rw, rh) / max(min(rw, rh), 1e-6)
    rectness = ha / max(rw * rh, 1e-6)

    # ---- heart: one deep notch near top-center, and a fairly solid outline ----
    xy = smooth_closed(pts.reshape(-1, 2))
    x0, y0, bw, bh = cv2.boundingRect(hull)
    size = max(bw, bh)
    defs = deep_defects(xy, size, min_depth=0.08)
    if defs and 0.7 < bw / max(bh, 1) < 1.5 and 0.76 < solidity < 0.96:
        (px, py), depth = max(defs, key=lambda t: t[1])
        in_top = py < y0 + 0.5 * bh
        centered = abs(px - (x0 + bw / 2)) < 0.25 * bw
        if len(defs) <= 2 and in_top and centered and depth < 0.45:
            return "heart", 1

    # ---- star: several notches (looser depth), or low solidity ----
    star_defs = deep_defects(xy, size, min_depth=0.04)
    if len(star_defs) >= 3 or solidity < 0.72:
        return "star", 1
    if n == 3:
        return "triangle", 1
    if circ > 0.88:
        return "circle", 1
    if n == 4:
        return ("square", 1) if ratio < 1.22 else ("rectangle", ratio)
    if rectness < 0.62:
        return "triangle", 1
    if rectness > 0.90 and circ < 0.86:
        return ("square", 1) if ratio < 1.22 else ("rectangle", ratio)
    return "circle", 1


def make_mesh(verts, faces):
    v = np.array(verts, np.float32)
    f = np.array(faces, np.int32)
    for i, (a, b, c) in enumerate(f):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        if np.dot(n, (v[a] + v[b] + v[c]) / 3) < 0:
            f[i] = (a, c, b)
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    edges = {}
    for i, tri in enumerate(f):
        for k in range(3):
            e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
            edges.setdefault(e, []).append(i)
    feature = [(a, b, fs[0], fs[1]) for (a, b), fs in edges.items()
               if len(fs) == 2 and np.dot(nrm[fs[0]], nrm[fs[1]]) < 0.9]
    return {"v": v, "f": f, "n": nrm, "edges": feature}


def prism(poly, depth):
    n, d = len(poly), depth / 2
    verts = [(x, y, d) for x, y in poly] + [(x, y, -d) for x, y in poly] + [(0, 0, d), (0, 0, -d)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, j + n), (i, j + n, i + n), (2 * n, i, j), (2 * n + 1, i + n, j + n)]
    return make_mesh(verts, faces)


def sphere(r=1.4, nlat=14, nlon=28):
    verts = [(0, r, 0), (0, -r, 0)]
    for i in range(1, nlat):
        t = math.pi * i / nlat
        for j in range(nlon):
            p = 2 * math.pi * j / nlon
            verts.append((r * math.sin(t) * math.cos(p), r * math.cos(t), r * math.sin(t) * math.sin(p)))
    idx = lambda i, j: 2 + (i - 1) * nlon + j % nlon
    faces = []
    for j in range(nlon):
        faces.append((0, idx(1, j), idx(1, j + 1)))
        faces.append((1, idx(nlat - 1, j + 1), idx(nlat - 1, j)))
    for i in range(1, nlat - 1):
        for j in range(nlon):
            faces += [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)), (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))]
    return make_mesh(verts, faces)


def build(name, ratio):
    if name == "circle":
        return sphere()
    if name == "square":
        return prism([(-1.1, -1.1), (1.1, -1.1), (1.1, 1.1), (-1.1, 1.1)], 2.2)
    if name == "rectangle":
        w = 2.8; h = w / min(ratio, 3.0)
        return prism([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)], 1.4)
    if name == "triangle":
        pts = [(1.8 * math.cos(math.pi / 2 + k * 2 * math.pi / 3),
                1.8 * math.sin(math.pi / 2 + k * 2 * math.pi / 3)) for k in range(3)]
        return prism(pts, 1.6)
    if name == "heart":
        pts = []
        for k in range(64):
            t = 2 * math.pi * k / 64
            x = 16 * math.sin(t) ** 3
            y = (13 * math.cos(t)
                 - 5 * math.cos(2 * t)
                 - 2 * math.cos(3 * t)
                 - math.cos(4 * t))
            pts.append((x / 9.0, y / 9.0))
        return prism(pts, 0.9)

    pts = []
    for k in range(10):
        a, r = math.pi / 2 + k * math.pi / 5, (1.9 if k % 2 == 0 else 0.8)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return prism(pts, 0.9)


LIGHT = np.array([-0.4, 0.6, 0.8], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def render(mesh, color, yaw, pitch=0.45, D=6.0, F=470.0):
    img = np.full((PH, PW, 3), MANTLE, np.uint8)
    cv2.ellipse(img, (PW // 2, PH - 62), (170, 22), 0, 0, 360, CRUST, -1, cv2.LINE_AA)
    if mesh is None:
        return img
    cy_, sy_, cx_, sx_ = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]], np.float32)
    R = Rx @ Ry
    V, N, f = mesh["v"] @ R.T, mesh["n"] @ R.T, mesh["f"]
    C = V[f].mean(axis=1)
    cam = np.array([0, 0, D], np.float32)
    vis = ((cam - C) * N).sum(axis=1) > 0
    s = F / (D - V[:, 2])
    P = np.stack([PW / 2 + V[:, 0] * s, PH / 2 - V[:, 1] * s - 8], axis=1)
    lam = np.clip(N @ LIGHT, 0, 1)
    idx = np.where(vis)[0]
    for i in idx[np.argsort(C[idx, 2])]:
        col = tuple(int(c * (0.30 + 0.70 * lam[i])) for c in color)
        poly = P[f[i]].astype(np.int32)
        cv2.fillConvexPoly(img, poly, col, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, col, 1, cv2.LINE_AA)
    for a, b, f1, f2 in mesh["edges"]:
        if vis[f1] or vis[f2]:
            cv2.line(img, tuple(P[a].astype(int)), tuple(P[b].astype(int)), CRUST, 2, cv2.LINE_AA)
    return img


def rmask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def paste(canvas, img, x, y, mask):
    roi = canvas[y:y + mask.shape[0], x:x + mask.shape[1]]
    np.copyto(roi, img, where=mask[..., None] > 0)


HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


def find_model():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        exact = os.path.join(folder, "hand_landmarker.task")
        if os.path.exists(exact):
            return exact
        found = glob.glob(os.path.join(folder, "*and*andmark*.task"))
        if found:
            return found[0]
    raise SystemExit("hand_landmarker.task not found. Put it in the same folder as this script.")


class HandTracker:
    def __init__(self):
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.h = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1,
                                              min_detection_confidence=0.6, min_tracking_confidence=0.6)
            return
        path = find_model()
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp_py.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6)
        self.h = vision.HandLandmarker.create_from_options(opts)
        self.t0, self.last_ts = time.time(), -1

    def detect(self, rgb):
        if self.legacy:
            r = self.h.process(rgb).multi_hand_landmarks
            return [(l.x, l.y) for l in r[0].landmark] if r else None
        ts = max(int((time.time() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        r = self.h.detect_for_video(img, ts)
        return [(l.x, l.y) for l in r.hand_landmarks[0]] if r.hand_landmarks else None

    def close(self):
        self.h.close()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PH)

    tracker = HandTracker()

    mask_in, mask_out = rmask(PW, PH, 20), rmask(PW + 4, PH + 4, 22)
    bg = np.full((H, W, 3), BASE, np.uint8)
    for x in (M, M * 2 + PW):
        paste(bg, np.full((PH + 4, PW + 4, 3), SURF0, np.uint8), x - 2, M - 2, mask_out)

    tint = np.full((PH, PW, 3), BASE, np.uint8)
    stroke, drawing, smooth, lost = [], False, None, 0
    stroke_ok = True
    mesh, mesh_color, yaw = None, MAUVE, 0.6

    def finish():
        nonlocal mesh, mesh_color, stroke_ok
        res = classify(stroke)
        stroke_ok = res is not None
        if res:
            mesh, mesh_color = build(*res), SHAPE_COLOR[res[0]]

    win = "Air Shapes 3D"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(cv2.flip(frame, 1), (PW, PH))
        frame = cv2.addWeighted(frame, 0.88, tint, 0.12, 0)
        lm = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if lm:
            lost = 0
            P = [(x * PW, y * PH) for x, y in lm]
            palm = math.dist(P[0], P[9]) + 1e-6
            gap = math.dist(P[4], P[8]) / palm
            tip = ((P[4][0] + P[8][0]) / 2, (P[4][1] + P[8][1]) / 2)
            smooth = tip if smooth is None else (smooth[0] * 0.6 + tip[0] * 0.4, smooth[1] * 0.6 + tip[1] * 0.4)
            if not drawing and gap < 0.28:
                drawing, stroke = True, []
            elif drawing and gap > 0.42:
                drawing = False
                finish()
            if drawing and (not stroke or math.dist(stroke[-1], smooth) > 3):
                stroke.append(smooth)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, (int(P[a][0]), int(P[a][1])), (int(P[b][0]), int(P[b][1])), SURF2, 2, cv2.LINE_AA)
            for p in P:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, LAV, -1, cv2.LINE_AA)
            c = (int(smooth[0]), int(smooth[1]))
            if drawing:
                cv2.circle(frame, c, 9, PEACH, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, c, 11, LAV, 2, cv2.LINE_AA)
        else:
            smooth = None
            lost += 1
            if drawing and lost > 6:
                drawing = False
                finish()

        if len(stroke) > 1:
            pl = np.array(stroke, np.int32).reshape(-1, 1, 2)
            col = MAUVE if stroke_ok or drawing else RED
            glow = frame.copy()
            cv2.polylines(glow, [pl], False, col, 14, cv2.LINE_AA)
            frame = cv2.addWeighted(glow, 0.35, frame, 0.65, 0)
            cv2.polylines(frame, [pl], False, col, 4, cv2.LINE_AA)

        yaw += 0.02
        canvas = bg.copy()
        paste(canvas, frame, M, M, mask_in)
        paste(canvas, render(mesh, mesh_color, yaw), M * 2 + PW, M, mask_in)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("c"):
            stroke, drawing, mesh, stroke_ok = [], False, None, True

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()