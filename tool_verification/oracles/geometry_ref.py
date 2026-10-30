"""Independent reference implementations of the deterministic A tools (V2 oracle).

Written from the definitions in doc/项目介绍.md and the tool descriptions, by
brute force where possible. Must not import visevi. Masks are 2D bool arrays,
x = column, y = row; each foreground pixel is a unit square centred on (x, y).
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np


# ---------------------------------------------------------------- basics
def pixels(mask):
    ys, xs = np.nonzero(mask)
    return list(zip(xs.tolist(), ys.tolist()))


def bbox(mask):
    pts = pixels(mask)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return {'x0': min(xs), 'y0': min(ys), 'x1': max(xs), 'y1': max(ys),
            'w': max(xs) - min(xs) + 1, 'h': max(ys) - min(ys) + 1}


def max_row_width(mask):
    best, best_y = -1, -1
    for y in range(mask.shape[0]):
        xs = [x for x in range(mask.shape[1]) if mask[y, x]]
        if xs and max(xs) - min(xs) + 1 > best:
            best, best_y = max(xs) - min(xs) + 1, y
    return best, best_y


def corners(mask, row=1.0, col=1.0):
    pts = set()
    for x, y in pixels(mask):
        for dx in (-0.5, 0.5):
            for dy in (-0.5, 0.5):
                pts.add(((x + dx) * col, (y + dy) * row))
    return list(pts)


def feret(mask, row=1.0, col=1.0):
    pts = corners(mask, row, col)
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = max(best, math.dist(pts[i], pts[j]))
    return best


def cell_gap(p, q, row=1.0, col=1.0):
    dx = max(abs(p[0] - q[0]) - 1, 0) * col
    dy = max(abs(p[1] - q[1]) - 1, 0) * row
    return math.hypot(dx, dy)


def gap_distance(a, b, row=1.0, col=1.0):
    pa, pb = pixels(a), pixels(b)
    if set(pa) & set(pb):
        return 0.0
    return min(cell_gap(p, q, row, col) for p in pa for q in pb)


def center_distance(a, b, row=1.0, col=1.0):
    return min(math.hypot((p[0] - q[0]) * col, (p[1] - q[1]) * row) for p in pixels(a) for q in pixels(b))


# ---------------------------------------------------------------- components
def components8(mask):
    seen = np.zeros(mask.shape, bool)
    comps = []
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] and not seen[y, x]:
                comp, queue = [], deque([(x, y)])
                seen[y, x] = True
                while queue:
                    cx, cy = queue.popleft()
                    comp.append((cx, cy))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                queue.append((nx, ny))
                comps.append(comp)
    return comps   # in raster order of first pixel (same as scipy.ndimage.label numbering)


def exposed_edges(cells):
    s = set(cells)
    edges = []
    for x, y in cells:
        if (x - 1, y) not in s: edges.append(((x - .5, y - .5), (x - .5, y + .5)))
        if (x + 1, y) not in s: edges.append(((x + .5, y - .5), (x + .5, y + .5)))
        if (x, y - 1) not in s: edges.append(((x - .5, y - .5), (x + .5, y - .5)))
        if (x, y + 1) not in s: edges.append(((x - .5, y + .5), (x + .5, y + .5)))
    return edges


def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def min_area_rect(hull):
    best = None
    for i in range(len(hull)):
        p, q = hull[i], hull[(i + 1) % len(hull)]
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        c, s = math.cos(ang), math.sin(ang)
        rot = [(x * c + y * s, -x * s + y * c) for x, y in hull]
        w = max(r[0] for r in rot) - min(r[0] for r in rot)
        h = max(r[1] for r in rot) - min(r[1] for r in rot)
        area, long, short = w * h, max(w, h), min(w, h)
        if best is None or area < best[0] * (1 - 1e-9):
            best = (area, long, short)
        elif area <= best[0] * (1 + 1e-9) and long / short > best[1] / best[2]:   # tie: most elongated
            best = (best[0], long, short)
    return best


def shape(mask):
    comps = components8(mask)
    out = []
    for cells in comps:
        area = len(cells)
        perim = len(exposed_edges(cells))
        hull = convex_hull([(x + dx, y + dy) for x, y in cells for dx in (-.5, .5) for dy in (-.5, .5)])
        hull_area = polygon_area(hull)
        _, long, short = min_area_rect(hull)
        out.append({'area_px': area, 'perimeter_px': float(perim),
                    'circularity': 4 * math.pi * area / perim ** 2, 'convex_hull_area_px': hull_area,
                    'solidity': area / hull_area, 'long': long, 'short': short, 'aspect': long / short})
    return out


# ---------------------------------------------------------------- relations
def _segment_distance(s, t):
    """Distance between two axis-aligned segments."""
    (ax0, ay0), (ax1, ay1) = s
    (bx0, by0), (bx1, by1) = t
    dx = max(0.0, max(min(ax0, ax1), min(bx0, bx1)) - min(max(ax0, ax1), max(bx0, bx1)))
    dy = max(0.0, max(min(ay0, ay1), min(by0, by1)) - min(max(ay0, ay1), max(by0, by1)))
    return math.hypot(dx, dy)


def relate(a, b):
    pa, pb = set(pixels(a)), set(pixels(b))
    inter = len(pa & pb)
    union = len(pa | pb)
    if inter:
        topology = 'overlap'
    elif any((x + dx, y + dy) in pb for x, y in pa for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
        topology = 'edge_adjacent'
    elif any((x + dx, y + dy) in pb for x, y in pa for dx in (-1, 1) for dy in (-1, 1)):
        topology = 'corner_adjacent'
    else:
        topology = 'disjoint'
    ea, eb = exposed_edges(list(pa)), exposed_edges(list(pb))
    return {'topology': topology, 'intersection_px': inter, 'union_px': union, 'iou': inter / union,
            'overlap_fraction_of_a': inter / len(pa), 'overlap_fraction_of_b': inter / len(pb),
            'a_covers_b': pb <= pa, 'b_covers_a': pa <= pb,
            'gap_distance_px': gap_distance(a, b),
            'boundary_distance_px': min(_segment_distance(s, t) for s in ea for t in eb),
            'pixel_center_distance_px': 0.0 if inter else center_distance(a, b)}


def lesion_direction(lesion, anatomy):
    ly, lx = np.nonzero(lesion); ay, ax = np.nonzero(anatomy)
    dx, dy = float(lx.mean() - ax.mean()), float(ly.mean() - ay.mean())
    return ('image_right' if dx > 0 else 'image_left' if dx < 0 else 'same_x',
            'below' if dy > 0 else 'above' if dy < 0 else 'same_y')


# ---------------------------------------------------------------- lungs
def zones(lung, region):
    ys = [y for y in range(lung.shape[0]) if lung[y].any()]
    y0, height = min(ys), max(ys) - min(ys) + 1
    counts = [0, 0, 0]
    for x, y in pixels(region & lung):
        band = min(2, max(0, math.floor(3 * (y - y0 + 0.5) / height)))
        counts[band] += 1
    return counts


def mirror_dice(right, left, midline):
    h, w = right.shape
    mirrored = np.zeros_like(right)
    for x, y in pixels(right):
        xr = round(2 * midline - x)
        if 0 <= xr < w:
            mirrored[y, xr] = True
    return 2 * int((mirrored & left).sum()) / int(right.sum() + left.sum())


def distribution(right, left, regions):
    comps = []
    for i, region in enumerate(regions):
        for cells in components8(region):
            r = sum(1 for c in cells if right[c[1], c[0]])
            l_ = sum(1 for c in cells if left[c[1], c[0]])
            side = 'bilateral_overlap' if r and l_ else 'right' if r else 'left' if l_ else 'outside_lungs'
            comps.append((i, len(cells), r, l_, side))
    rc = sum(1 for c in comps if c[2]); lc = sum(1 for c in comps if c[3])
    pattern = ('no_regions' if not comps else 'bilateral' if rc and lc else 'right_only' if rc
               else 'left_only' if lc else 'outside_lungs_only')
    return pattern, comps


# ---------------------------------------------------------------- appearance
def quantile_linear(values, q):
    v = sorted(values)
    h = (len(v) - 1) * q
    lo = math.floor(h)
    return v[lo] + (h - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def intensity(gray, roi):
    vals = [float(gray[y, x]) for x, y in pixels(roi)]
    n = len(vals); mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {'pixel_count': n, 'minimum': min(vals), 'maximum': max(vals), 'mean': mean,
            'variance_population': var, 'std_population': math.sqrt(var),
            **{f'p{int(q * 100):02d}': quantile_linear(vals, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}


def glcm_features(gray, roi, bin_width=16.0):
    q = np.floor(gray / bin_width).astype(np.int64)
    h, w = gray.shape
    dirs = {'horizontal': (0, 1), 'diagonal_up_right': (-1, 1), 'vertical_up': (-1, 0), 'diagonal_up_left': (-1, -1)}
    per = {}
    for name, (dy, dx) in dirs.items():
        counts = {}
        n = 0
        for y in range(h):
            for x in range(w):
                y2, x2 = y + dy, x + dx
                if 0 <= y2 < h and 0 <= x2 < w and roi[y, x] and roi[y2, x2]:
                    for a, b in ((q[y, x], q[y2, x2]), (q[y2, x2], q[y, x])):
                        counts[(a, b)] = counts.get((a, b), 0) + 1
                    n += 1
        if not n:
            per[name] = None
            continue
        total = sum(counts.values())
        P = {k: v / total for k, v in counts.items()}
        mi = sum(p * i for (i, j), p in P.items()); mj = sum(p * j for (i, j), p in P.items())
        si = math.sqrt(sum(p * (i - mi) ** 2 for (i, j), p in P.items()))
        sj = math.sqrt(sum(p * (j - mj) ** 2 for (i, j), p in P.items()))
        asm = sum(p * p for p in P.values())
        per[name] = {'contrast': sum(p * (i - j) ** 2 for (i, j), p in P.items()),
                     'dissimilarity': sum(p * abs(i - j) for (i, j), p in P.items()),
                     'homogeneity': sum(p / (1 + (i - j) ** 2) for (i, j), p in P.items()),
                     'angular_second_moment': asm, 'energy': math.sqrt(asm),
                     'correlation': (sum(p * (i - mi) * (j - mj) for (i, j), p in P.items()) / (si * sj)) if si and sj else 1.0,
                     'pairs': n}
    valid = [v for v in per.values() if v]
    if not valid:
        return None, per
    mean = {k: sum(v[k] for v in valid) / len(valid) for k in valid[0] if k != 'pairs'}
    return mean, per


def tenengrad(gray, roi, scale=1.0):
    g = gray / scale
    h, w = g.shape
    def px(y, x):   # cv2.BORDER_REFLECT: fedcba|abcdefgh|hgfedcb
        y = -y - 1 if y < 0 else (2 * h - y - 1 if y >= h else y)
        x = -x - 1 if x < 0 else (2 * w - x - 1 if x >= w else x)
        return g[y, x]
    vals, mags = [], []
    for x, y in pixels(roi):
        gx = ((px(y - 1, x + 1) + 2 * px(y, x + 1) + px(y + 1, x + 1))
              - (px(y - 1, x - 1) + 2 * px(y, x - 1) + px(y + 1, x - 1))) / 8
        gy = ((px(y + 1, x - 1) + 2 * px(y + 1, x) + px(y + 1, x + 1))
              - (px(y - 1, x - 1) + 2 * px(y - 1, x) + px(y - 1, x + 1))) / 8
        vals.append(gx * gx + gy * gy); mags.append(math.sqrt(gx * gx + gy * gy))
    return sum(vals) / len(vals), sum(mags) / len(mags)


# ---------------------------------------------------------------- CXAS rules (A12 / A25)
def thoracic_inner_margin(right_ribs, left_ribs, midline, max_row):
    rows = []
    h, w = right_ribs.shape
    for y in range(0, min(max_row, h - 1) + 1):
        rx = [x for x in range(w) if x < math.floor(midline) and right_ribs[y, x]]
        lx = [x for x in range(w) if x >= math.ceil(midline) and left_ribs[y, x]]
        if not rx or not lx:
            continue
        r = min(rx)
        while r + 1 < w and right_ribs[y, r + 1]:
            r += 1
        l_ = max(lx)
        while l_ - 1 >= 0 and left_ribs[y, l_ - 1]:
            l_ -= 1
        if l_ - 0.5 > r + 0.5:
            rows.append((y, r + 0.5, l_ - 0.5))
    return rows


def rotation_index(clav_r, clav_l, spine):
    rp = max(pixels(clav_r), key=lambda p: (p[0], -p[1]))
    lp = min(pixels(clav_l), key=lambda p: (p[0], p[1]))
    def mid(y):
        xs = [x for x in range(spine.shape[1]) if spine[y, x]]
        return sum(xs) / len(xs)
    dr = mid(rp[1]) - (rp[0] + 0.5); dl = (lp[0] - 0.5) - mid(lp[1])
    if dr <= 0 or dl <= 0:
        return None                      # undefined: medial end at/beyond midline
    return dr, dl, (dr - dl) / (dr + dl)
