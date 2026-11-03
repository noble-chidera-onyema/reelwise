"""The TV: find the screen in the host scene, play the story on it, and push
the camera through it until the story fills the frame.

Screen detection, most reliable first:
  green   the screen is flat bright green (#00FF00). The key gives an exact
          matte, so anything in front of the TV (a head, a couch arm) stays in
          front, and it is re-measured every frame of a moving plate.
  panel   a flat, bright, cream or white screen, like the one in the Worth
          Telling set. Its exact rounded outline becomes the matte.
  quad    the largest TV-shaped four-sided outline in the image.
  manual  four corner points you give.

The push-through. The story is vertical (9:16) and most TV screens are
landscape. On the TV the story plays the way a real TV shows a vertical
video: sharp in the middle, blurred copy filling the sides. The camera then
zooms until that sharp middle panel exactly fills the output frame, at which
point the TV is gone and the story is full screen. No cut, no jump. The
reverse move takes you back out to the hosts.

All geometry is in host-canvas pixels. The story is warped straight from its
own frames at output resolution, so it stays sharp however far the camera
pushes in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GREEN_LO = np.array([40, 110, 90], np.uint8)    # OpenCV HSV, hue 0..180
GREEN_HI = np.array([85, 255, 255], np.uint8)


@dataclass
class Screen:
    corners: np.ndarray          # (4, 2) float32: top-left, top-right, bottom-right, bottom-left
    method: str                  # green | quad | manual
    matte: np.ndarray | None = None   # uint8 0..255 in host canvas, green method only

    @property
    def aspect(self) -> float:
        tl, tr, br, bl = self.corners
        w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        return float(w / max(h, 1e-6))

    def to_dict(self) -> dict:
        return {"corners": self.corners.round(1).tolist(), "method": self.method, "aspect": round(self.aspect, 3)}


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_corners(ap)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(c)))


def green_matte(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Keep only the largest green region (the screen), then soften the edge by one pixel.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == k, 255, 0).astype(np.uint8)
    # Fill holes from anything in front of the screen when computing the outline,
    # but keep the raw matte so foreground objects stay in front.
    return cv2.GaussianBlur(m, (3, 3), 0)


def detect_green(img: np.ndarray, min_area: float = 0.004) -> Screen | None:
    m = green_matte(img)
    if (m > 127).mean() < min_area:
        return None
    filled = cv2.morphologyEx((m > 127).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    q = _quad_from_mask(filled)
    return Screen(q, "green", m) if q is not None else None


def detect_panel(img: np.ndarray) -> Screen | None:
    """A flat, bright, low-saturation screen (a lit TV drawn as a cream or white panel).

    The filled region becomes the matte, so rounded screen corners are kept and the
    bezel is never painted over. Placeholder text on the screen is filled in.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    best, best_area = None, 0
    for vmin in (200, 180, 160):
        m = ((hsv[..., 2] >= vmin) & (hsv[..., 1] <= 90)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (0.02 * w * h < a < 0.8 * w * h):
                continue
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            if a / max(rw * rh, 1) < 0.9:          # rectangular, not the moon
                continue
            s = Screen(order_corners(cv2.boxPoints(((cx, cy), (rw, rh), ang))), "panel")
            if 1.05 <= s.aspect <= 2.4 and a > best_area:
                filled = np.zeros((h, w), np.uint8)
                cv2.drawContours(filled, [cv2.convexHull(c)], -1, 255, -1)
                s.matte = cv2.GaussianBlur(filled, (3, 3), 0)
                best, best_area = s, a
        if best is not None:
            return best
    return None


def detect_quad(img: np.ndarray) -> Screen | None:
    """Largest convex four-sided outline with a TV-like shape. Lower confidence than green."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 40, 40)
    best, best_area = None, 0.0
    for lo, hi in ((30, 90), (50, 150), (80, 200)):
        e = cv2.dilate(cv2.Canny(g, lo, hi), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (0.01 * w * h < a < 0.7 * w * h):
                continue
            ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            s = Screen(order_corners(ap), "quad")
            if 1.1 <= s.aspect <= 2.4 and a > best_area:
                best, best_area = s, a
    return best


def detect_screen(img: np.ndarray, manual: list | None = None) -> Screen | None:
    if manual:
        return Screen(order_corners(np.array(manual, np.float32)), "manual")
    return detect_green(img) or detect_panel(img) or detect_quad(img)


# --------------------------------------------------------------------------- geometry

def panel_rect(screen_aspect: float, story_aspect: float, sh: float = 1.0) -> tuple[np.ndarray, float, float]:
    """Screen-plane coordinates: the screen is [0, sw] x [0, sh]. Returns the sharp
    story panel corners (contain-fit, centred) and sw, sh."""
    sw = screen_aspect * sh
    if story_aspect <= screen_aspect:      # tall story in a wide screen: pillars
        ph, pw = sh, sh * story_aspect
    else:                                  # wide story in a tall screen: bars
        pw, ph = sw, sw / story_aspect
    x0, y0 = (sw - pw) / 2, (sh - ph) / 2
    rect = np.array([[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]], np.float32)
    return rect, sw, sh


def plane_to_host(screen: Screen, sw: float, sh: float) -> np.ndarray:
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    return cv2.getPerspectiveTransform(src, screen.corners.astype(np.float32))


def fit_aspect(rect: tuple[float, float, float, float], aspect: float) -> tuple[float, float, float, float]:
    """Grow (x, y, w, h) to the given w/h aspect around its centre."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    return cx - w / 2, cy - h / 2, w, h


def fit_inside(rect: tuple[float, float, float, float], aspect: float) -> tuple[float, float, float, float]:
    """Largest (x, y, w, h) of the given aspect inside rect, centred."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    return cx - w / 2, cy - h / 2, w, h


def zoom_rect(r0, r1, u: float):
    """Camera crop at progress u (0 = r0, 1 = r1). Width changes geometrically and the
    target stays put on screen, which reads as one steady push."""
    e = u * u * (3 - 2 * u)
    w0, w1 = r0[2], r1[2]
    w = float(np.exp(np.log(w0) + (np.log(w1) - np.log(w0)) * e))
    k = (w - w1) / (w0 - w1) if abs(w0 - w1) > 1e-6 else 1 - e
    return tuple(r1[i] + (r0[i] - r1[i]) * k for i in range(4))


def crop_matrix(rect, out_w: int, out_h: int) -> np.ndarray:
    x, y, w, h = rect
    return np.array([[out_w / w, 0, -x * out_w / w], [0, out_h / h, -y * out_h / h], [0, 0, 1]], np.float64)


# --------------------------------------------------------------------------- compositor

class TVCompositor:
    """Renders host-canvas frames with the story playing on the TV, under any camera crop."""

    def __init__(self, canvas_size: tuple[int, int], screen: Screen, out_size: tuple[int, int]):
        self.cw, self.ch = canvas_size
        self.screen = screen
        self.ow, self.oh = out_size
        self.story_aspect = self.ow / self.oh
        self.panel, self.sw, self.sh = panel_rect(screen.aspect, self.story_aspect)
        self.H = plane_to_host(screen, self.sw, self.sh)
        pc = cv2.perspectiveTransform(self.panel[None], self.H)[0]
        x0, y0 = pc.min(0)
        x1, y1 = pc.max(0)
        self.full = fit_inside((0.0, 0.0, float(self.cw), float(self.ch)), self.story_aspect)
        self.target = fit_aspect((float(x0), float(y0), float(x1 - x0), float(y1 - y0)), self.story_aspect)
        if screen.matte is not None and screen.method == "panel":
            self.static_matte = screen.matte
        else:
            poly = np.zeros((self.ch, self.cw), np.uint8)
            cv2.fillConvexPoly(poly, screen.corners.astype(np.int32), 255)
            self.static_matte = cv2.GaussianBlur(poly, (3, 3), 0)

    def despill(self, host_frame: np.ndarray, matte: np.ndarray) -> np.ndarray:
        """Remove green spill on pixels around the screen edge (G clamped to max(R, B))."""
        if self.screen.method != "green":
            return host_frame
        band = cv2.dilate((matte > 8).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        f = host_frame.copy()
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        cap = np.maximum(r, b)
        g[band] = np.minimum(g[band], cap[band])
        return f

    def matte_for(self, host_frame: np.ndarray) -> np.ndarray:
        if self.screen.method == "green":
            return green_matte(host_frame)
        return self.static_matte

    def render(self, host_frame: np.ndarray, story_frame: np.ndarray, u: float,
               glow: float = 0.08, overlay: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """One output frame. u is zoom progress: 0 = whole host scene, 1 = story full frame.

        host_frame is the set (with the TV). overlay is (BGR, alpha 0..1) drawn over the TV,
        both in canvas pixels: the hosts when they are composited as separate layers."""
        rect = zoom_rect(self.full, self.target, u)
        C = crop_matrix(rect, self.ow, self.oh)
        m_host = self.matte_for(host_frame)
        if self.screen.method == "green":
            # Grow the matte a pixel so the keyed edge sits under the picture, then despill.
            m_host = cv2.dilate(m_host, np.ones((3, 3), np.uint8))
            host_frame = self.despill(host_frame, m_host)
        out = cv2.warpPerspective(host_frame, C, (self.ow, self.oh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        matte = cv2.warpPerspective(m_host, C, (self.ow, self.oh),
                                    flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
        # Blurred side fill: the story stretched over the whole screen, softened and dimmed.
        small = cv2.resize(story_frame, (64, int(64 / self.screen.aspect)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        small = (small.astype(np.float32) * 0.55).astype(np.uint8)
        sh_h, sh_w = small.shape[:2]
        S_bg = np.array([[self.sw / sh_w, 0, 0], [0, self.sh / sh_h, 0], [0, 0, 1]])
        bg = cv2.warpPerspective(small, C @ self.H @ S_bg, (self.ow, self.oh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        fh, fw = story_frame.shape[:2]
        src = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], np.float32)
        S_fg = cv2.getPerspectiveTransform(src, self.panel)
        M = C @ self.H @ S_fg
        fg = cv2.warpPerspective(story_frame, M, (self.ow, self.oh), flags=cv2.INTER_LINEAR)
        fgm = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), M, (self.ow, self.oh),
                                  flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
        screen_img = fg.astype(np.float32) * fgm + bg.astype(np.float32) * (1 - fgm)
        # A faint screen glow so the picture reads as lit by the TV, fading as we enter it.
        if glow > 0:
            screen_img = screen_img * (1 + glow * (1 - u))
        comp = out.astype(np.float32) * (1 - matte) + screen_img * matte
        if overlay is not None:
            ob = cv2.warpPerspective(overlay[0], C, (self.ow, self.oh), flags=cv2.INTER_LINEAR)
            oa = cv2.warpPerspective(overlay[1].astype(np.float32), C, (self.ow, self.oh),
                                     flags=cv2.INTER_LINEAR)[..., None]
            comp = comp * (1 - oa) + ob.astype(np.float32) * oa
        # Near the end of the push, blend to the untouched story frame so perspective
        # leftovers from an angled TV never show.
        if u > 0.9:
            a = (u - 0.9) / 0.1
            full = cv2.resize(story_frame, (self.ow, self.oh), interpolation=cv2.INTER_AREA)
            comp = comp * (1 - a) + full.astype(np.float32) * a
        return np.clip(comp, 0, 255).astype(np.uint8)
