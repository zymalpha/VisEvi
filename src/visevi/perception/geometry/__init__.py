"""Deterministic measurements and spatial relations for 2D masks (A9–A14, A19–A23).

Extents, shapes, and gaps use one convention: each foreground pixel is a unit
square centred on its integer (x=column, y=row) coordinate. Pixel-centre
distances are reported only as explicitly named secondary values.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _mask(value: np.ndarray, name: str = "mask") -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError(f"{name} must be a nonzero-size 2D array")
    if not (np.issubdtype(result.dtype, np.number) or result.dtype == np.bool_):
        raise ValueError(f"{name} must be a numeric or boolean binary mask")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.isin(result, (0, 1, 255)).all():
        raise ValueError(f"{name} must be binary (0/1 or 0/255); threshold probabilities explicitly")
    return result != 0


def _nonempty(mask: np.ndarray, name: str) -> np.ndarray:
    result = _mask(mask, name)
    if not result.any():
        raise ValueError(f"{name} must not be empty")
    return result


def _pixel_spacing(spacing: tuple[float, float] | None) -> tuple[float, float]:
    if spacing is None:
        return 1.0, 1.0
    if len(spacing) != 2:
        raise ValueError("spacing must be (row_spacing, column_spacing)")
    row, col = map(float, spacing)
    if not np.isfinite((row, col)).all() or row <= 0 or col <= 0:
        raise ValueError("pixel spacing values must be finite and positive")
    return row, col


def _lung_pair(right: np.ndarray, left: np.ndarray, *, nonempty: bool = True):
    convert = _nonempty if nonempty else _mask
    right, left = convert(right, "right_lung"), convert(left, "left_lung")
    if right.shape != left.shape:
        raise ValueError("right_lung and left_lung must have the same shape")
    if np.any(right & left):
        raise ValueError("right_lung and left_lung overlap; resolve ambiguous laterality first")
    return right, left


def row_spans(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row [xmin, xmax] for rows where the mask is present (vectorized)."""
    m = _mask(m)
    h, w = m.shape
    rows = np.where(m.any(axis=1))[0]
    xmin = m.argmax(axis=1)[rows]
    xmax = w - 1 - m[:, ::-1].argmax(axis=1)[rows]
    return rows, xmin, xmax


def max_row_width(m: np.ndarray) -> tuple[int, int]:
    """(max transverse span, row y where it occurs)."""
    m = _nonempty(m, "mask")
    rows, xmn, xmx = row_spans(m)
    w = xmx - xmn + 1
    i = int(w.argmax())
    return int(w[i]), int(rows[i])


def bbox(m: np.ndarray) -> dict:
    m = _nonempty(m, "mask")
    ys, xs = np.nonzero(m)
    return {"x0": int(xs.min()), "y0": int(ys.min()),
            "x1": int(xs.max()), "y1": int(ys.max()),
            "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1)}


def area(m: np.ndarray) -> int:
    return int(_mask(m).sum())


def area_physical(m: np.ndarray, spacing: tuple[float, float]) -> float:
    """Area in squared spacing units; ``spacing`` is (row, column)."""
    row, col = _pixel_spacing(spacing)
    return float(area(m) * row * col)


def feret_max_diameter(m: np.ndarray,
                       spacing: tuple[float, float] | None = None) -> float:
    """Maximum Feret diameter of the pixel-cell region (corner to corner).

    A 1x10 run gives sqrt(10**2 + 1), a single pixel sqrt(2): the same unit-cell
    geometry as the inclusive bbox, A14 shapes, and A22 gaps.
    """
    import cv2
    from scipy.spatial.distance import cdist
    m = _nonempty(m, "mask")
    row, col = _pixel_spacing(spacing)
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    # hull(cells) = hull(centres) expanded by the four half-pixel corner offsets
    corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    points = (hull[:, None, :] + corners[None]).reshape(-1, 2) * np.array([col, row])
    return float(cdist(points, points).max())


def measure_extent(m: np.ndarray,
                   spacing: tuple[float, float] | None = None) -> dict:
    """A9: inclusive pixel bounding box and transverse/maximum extents."""
    m = _nonempty(m, "mask")
    row, col = _pixel_spacing(spacing)
    bounds = bbox(m)
    width, row_y = max_row_width(m)
    return {
        "bbox_inclusive_px": bounds,
        "bbox_width_px": bounds["w"],
        "bbox_height_px": bounds["h"],
        "max_row_width_px": width,
        "max_row_width_row_y": row_y,
        "feret_max_px": feret_max_diameter(m),
        "bbox_width_physical": bounds["w"] * col if spacing is not None else None,
        "bbox_height_physical": bounds["h"] * row if spacing is not None else None,
        "feret_max_physical": feret_max_diameter(m, spacing) if spacing is not None else None,
        "spacing_row_col": [row, col] if spacing is not None else None,
        "bbox_convention": "inclusive pixel indices; dimensions count pixels",
        "feret_convention": "pixel-cell corners (unit squares), consistent with bbox dimensions",
        "physical_unit": "caller-supplied spacing unit" if spacing is not None else None,
    }


def _min_area_rectangle_sides(hull: np.ndarray) -> tuple[float, float]:
    """(long, short) sides of the minimum-area enclosing rectangle (rotating calipers).

    The minimum-area rectangle is not unique for some pixel shapes (two
    diagonal pixels admit a 2x2 square and a 2.83x1.41 rectangle). Among
    rectangles within 1e-9 relative of the minimum area, the most elongated one
    is reported, so the aspect ratio reflects the region's elongation.
    """
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        if not np.hypot(*edge):
            continue
        u = edge / np.hypot(*edge)
        v = np.array([-u[1], u[0]])
        a, b = hull @ u, hull @ v
        w, h = a.max() - a.min(), b.max() - b.min()
        area, long, short = w * h, max(w, h), min(w, h)
        if best is None or area < best[0] * (1 - 1e-9):
            best = (area, long, short)
        elif area <= best[0] * (1 + 1e-9) and long / short > best[1] / best[2]:
            best = (best[0], long, short)
    return float(best[1]), float(best[2])


def _shape_component(mask: np.ndarray) -> dict:
    geom = _mask_geometry(mask)
    hull = geom.convex_hull
    long, short = _min_area_rectangle_sides(np.asarray(hull.exterior.coords)[:-1])
    perimeter = float(geom.length)
    area_px = float(geom.area)
    hull_area = float(hull.area)
    polygons = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    contours = [{
        'outer_xy': [[float(x), float(y)] for x, y in polygon.exterior.coords],
        'holes_xy': [[[float(x), float(y)] for x, y in ring.coords]
                     for ring in polygon.interiors],
    } for polygon in polygons]
    return {
        "area_px": int(round(area_px)),
        "perimeter_px": perimeter,
        "circularity": (4 * np.pi * area_px / (perimeter * perimeter)
                        if perimeter else None),
        "convex_hull_area_px": hull_area,
        "solidity": area_px / hull_area if hull_area else None,
        "minimum_rotated_rectangle_long_side_px": long,
        "minimum_rotated_rectangle_short_side_px": short,
        "rotated_aspect_ratio": long / short if short else None,
        "contours_pixel_edges": contours,
    }


def describe_shape(m: np.ndarray) -> dict:
    """A14: perimeter, circularity, solidity, and oriented aspect ratio.

    Each foreground pixel is a unit square. Components use 8-connectivity;
    aggregate descriptors refer to the largest component by pixel area.
    """
    from scipy.ndimage import label

    m = _nonempty(m, "mask")
    labels, count = label(m, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    for component_id in range(1, count + 1):
        descriptor = _shape_component(labels == component_id)
        descriptor["component_id"] = component_id - 1
        components.append(descriptor)
    largest = max(components, key=lambda item: item["area_px"])
    return {"component_count": int(count),
            "components": components,
            "largest_component": {key: value for key, value in largest.items()
                                  if key not in ("component_id", "contours_pixel_edges")},
            "method": "unit-square pixel-cell geometry; 8-connected components"}


def min_distance(a: np.ndarray, b: np.ndarray,
                 spacing: tuple[float, float] | None = None) -> float:
    """Minimum foreground pixel-center distance (secondary A11/A22 value); overlap is zero."""
    a, b = _nonempty(a, "a"), _nonempty(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return float(distance_transform_edt(~a, sampling=_pixel_spacing(spacing))[b].min())


def gap_distance(a: np.ndarray, b: np.ndarray,
                 spacing: tuple[float, float] | None = None) -> float:
    """A11/A22 primary distance: gap between pixel-cell regions; touching or overlap is zero."""
    from shapely.affinity import scale

    a, b = _nonempty(a, "a"), _nonempty(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    row, col = _pixel_spacing(spacing)
    geom_a, geom_b = _mask_geometry(a), _mask_geometry(b)
    if spacing is not None:
        geom_a = scale(geom_a, xfact=col, yfact=row, origin=(0, 0))
        geom_b = scale(geom_b, xfact=col, yfact=row, origin=(0, 0))
    return float(geom_a.distance(geom_b))


def ctr_from_masks(heart: np.ndarray, lung_union: np.ndarray) -> dict:
    """A12 CTR — automated proxy: max per-row span of Heart over lung-union."""
    heart = _nonempty(heart, "heart")
    lung_union = _nonempty(lung_union, "lung_union")
    if heart.shape != lung_union.shape:
        raise ValueError("heart and lung_union must have the same shape")
    heart_w, heart_y = max_row_width(heart)
    thorax_w, thorax_y = max_row_width(lung_union)
    return {"heart_max_width_px": heart_w, "heart_width_row_y": heart_y,
            "thorax_max_width_px": thorax_w, "thorax_width_row_y": thorax_y,
            "CTR": heart_w / thorax_w,
            "method": "max per-row transverse span of Heart over lung-union "
                      "(automated proxy; radiologic CTR uses rib-inner margins)"}


def ctr_danzer(m_heart: np.ndarray, m_thorax: np.ndarray,
               midline_x: float | None = None) -> dict:
    """A12 CTR, strict Danzer (1919) caliber:
    numerator = (midline -> farthest right heart border) + (midline -> farthest
    left heart border), each a horizontal excursion. The two excursions
    telescope: whenever the midline falls between the borders the sum equals
    the heart bbox width, so the transverse diameter — hence CTR — is
    invariant to midline placement. denominator = max internal thoracic width
    (m_thorax max row width, automated proxy of the rib-inner margin).
    Contrast: ctr_from_masks takes the widest SINGLE row of the heart, which
    underestimates the transverse diameter when the two border extremes occur
    at different heights."""
    m_heart = _nonempty(m_heart, "heart")
    m_thorax = _nonempty(m_thorax, "thorax")
    if m_heart.shape != m_thorax.shape:
        raise ValueError("heart and thorax must have the same shape")
    hb = bbox(m_heart)
    if midline_x is None:
        tb = bbox(m_thorax)
        midline_x = (tb["x0"] + tb["x1"]) / 2.0
    d_right = midline_x - hb["x0"]        # image-left side = patient right
    d_left = hb["x1"] - midline_x
    tw, ty = max_row_width(m_thorax)
    transverse = d_right + d_left          # = heart bbox x-span (midline-independent)
    return {"d_right_px": round(float(d_right), 1), "d_left_px": round(float(d_left), 1),
            "heart_transverse_px": round(float(transverse), 1),
            "thorax_max_width_px": tw, "thorax_width_row_y": ty,
            "midline_x": midline_x,
            "CTR": transverse / tw,
            "method": "Danzer 1919 strict: sum of horizontal excursions from "
                      "midline to right/left heart border (midline-independent "
                      "bbox width) over max thorax width (lung-union proxy for "
                      "rib-inner margin)"}


def ctr_from_borders(heart_borders_x: tuple[float, float],
                     thoracic_inner_borders_x: tuple[float, float]) -> dict:
    """A12 (partial): CTR arithmetic from supplied horizontal border positions.

    Coordinates are continuous image x positions of the outer edges (a mask's
    inclusive pixel x0..x1 spans x0-0.5..x1+0.5). Neither the heart nor the
    thoracic inner margins are localized here; a heart extent outside the
    thoracic extent is rejected as an inconsistent measurement.
    """
    if len(heart_borders_x) != 2 or len(thoracic_inner_borders_x) != 2:
        raise ValueError("each boundary pair must contain left and right x coordinates")
    heart_left, heart_right = map(float, heart_borders_x)
    thorax_left, thorax_right = map(float, thoracic_inner_borders_x)
    values = np.array([heart_left, heart_right, thorax_left, thorax_right])
    if not np.isfinite(values).all():
        raise ValueError("boundary coordinates must be finite")
    heart_width = heart_right - heart_left
    thorax_width = thorax_right - thorax_left
    if heart_width <= 0 or thorax_width <= 0:
        raise ValueError("left boundary must be less than right boundary")
    if heart_left < thorax_left or heart_right > thorax_right:
        raise ValueError("heart borders must lie within the thoracic inner borders")
    return {
        "heart_width_px": heart_width,
        "thoracic_inner_width_px": thorax_width,
        "CTR": heart_width / thorax_width,
        "heart_borders_x": [heart_left, heart_right],
        "thoracic_inner_borders_x": [thorax_left, thorax_right],
        "method": "heart horizontal extent / manually supplied internal thoracic extent",
        "scope": "ratio calculation only; no boundary localization performed",
    }


def lung_area_compare(right: np.ndarray, left: np.ndarray) -> dict:
    right, left = _lung_pair(right, left, nonempty=False)
    ra, la = area(right), area(left)
    rw = max_row_width(right)[0] if ra else None
    lw = max_row_width(left)[0] if la else None
    return {"right_area_px": ra, "left_area_px": la,
            "area_difference_right_minus_left_px": ra - la,
            "right_max_width_px": rw, "left_max_width_px": lw,
            "area_ratio_R_over_L": ra / la if la else None,
            "width_difference_right_minus_left_px": rw - lw if rw is not None and lw is not None else None,
            "width_ratio_R_over_L": rw / lw if rw is not None and lw else None,
            "status": "ok" if ra and la else "one_or_both_sides_empty"}


def mirror_dice(m_right: np.ndarray, m_left: np.ndarray, midline_x: float) -> dict:
    """A19 laterality: mirror the right mask onto the left half about midline,
    Dice against the left mask. Dice = 2|A∩B| / (|A|+|B|) — bounded [0,1].
    Radiologic right lung = image-left side is the CALLER's naming convention;
    math is symmetric."""
    m_right, m_left = _mask(m_right, "m_right"), _mask(m_left, "m_left")
    if m_right.shape != m_left.shape:
        raise ValueError("mirror masks must have the same 2D shape")
    if not np.isfinite(midline_x):
        raise ValueError("midline_x must be finite")
    if not np.isclose(2 * midline_x, round(2 * midline_x), rtol=0, atol=1e-8):
        raise ValueError("midline_x must lie on an integer or half-integer pixel coordinate")
    h, w = m_right.shape
    # Coordinates refer to pixel centers, as bbox and all other geometry do.
    # Pixels outside the image are background, never a repeated edge pixel.
    xi = np.rint(2 * midline_x - np.arange(w)).astype(int)
    valid = (xi >= 0) & (xi < w)
    mirrored = np.zeros((h, w), dtype=bool)
    mirrored[:, valid] = m_right[:, xi[valid]]
    inter = (mirrored & m_left).sum()
    denom = m_right.sum() + m_left.sum()
    if denom == 0:
        raise ValueError("mirror Dice is undefined for two empty regions")
    return {"mirror_dice": float(2 * inter / denom),
            "midline_x": midline_x,
            "reflected_outside_frame_px": int(m_right.sum() - mirrored.sum()),
            "method": "right mask reflected about supplied midline; out-of-frame pixels remain in Dice denominator"}


def _mask_geometry(mask: np.ndarray):
    """Build exact pixel-cell polygons from horizontal foreground runs."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    mask = _nonempty(mask, "mask")
    runs = []
    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        runs.extend(box(int(start) - 0.5, y - 0.5,
                        int(end) - 0.5, y + 0.5)
                    for start, end in zip(starts, ends))
    return unary_union(runs)


def relate_masks(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact 2D pixel-cell overlap, containment, contact, and boundary distance."""
    a, b = _nonempty(a, "a"), _nonempty(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    intersection_px = int(np.count_nonzero(a & b))
    union_px = int(np.count_nonzero(a | b))
    geom_a, geom_b = _mask_geometry(a), _mask_geometry(b)
    contact = geom_a.boundary.intersection(geom_b.boundary)
    if intersection_px:
        topology = "overlap"
    elif not contact.is_empty:
        topology = "edge_adjacent" if contact.length > 0 else "corner_adjacent"
    else:
        topology = "disjoint"
    return {
        "topology": topology,
        "intersection_px": intersection_px,
        "union_px": union_px,
        "iou": intersection_px / union_px,
        "overlap_fraction_of_a": intersection_px / int(a.sum()),
        "overlap_fraction_of_b": intersection_px / int(b.sum()),
        "a_covers_b": bool(geom_a.covers(geom_b)),
        "b_covers_a": bool(geom_b.covers(geom_a)),
        "gap_distance_px": float(geom_a.distance(geom_b)),
        "boundary_distance_px": float(geom_a.boundary.distance(geom_b.boundary)),
        "pixel_center_distance_px": min_distance(a, b),
        "geometry_convention": "each foreground pixel is a unit square centered on its integer coordinate",
    }


def relate_lesion_to_anatomy(lesion: np.ndarray, anatomy: np.ndarray) -> dict:
    """A21: spatial relation and image-axis direction for two raster regions."""
    lesion, anatomy = _nonempty(lesion, "lesion"), _nonempty(anatomy, "anatomy")
    if lesion.shape != anatomy.shape:
        raise ValueError("lesion and anatomy must have the same shape")
    relation = relate_masks(lesion, anatomy)
    lesion_y, lesion_x = np.nonzero(lesion)
    anatomy_y, anatomy_x = np.nonzero(anatomy)
    dx = float(lesion_x.mean() - anatomy_x.mean())
    dy = float(lesion_y.mean() - anatomy_y.mean())
    relation.update({
        "relative_image_horizontal": "image_right" if dx > 0 else "image_left" if dx < 0 else "same_x",
        "relative_image_vertical": "below" if dy > 0 else "above" if dy < 0 else "same_y",
        "centroid_delta_xy_px": [dx, dy],
        "laterality_note": "image-axis direction only; patient laterality requires verified orientation",
    })
    return relation


def _lung_zone_masks(lung: np.ndarray) -> dict[str, np.ndarray]:
    lung = _nonempty(lung, "lung")
    ys = np.flatnonzero(lung.any(axis=1))
    y0, y1 = int(ys.min()), int(ys.max())
    height = y1 - y0 + 1
    rows = np.arange(lung.shape[0])[:, None]
    bands = np.clip(((2 * (rows - y0) + 1) * 3 // (2 * height)), 0, 2)
    names = ("upper", "middle", "lower")
    return {name: lung & (bands == i) for i, name in enumerate(names)}


def assign_lung_zones(right_lung: np.ndarray, left_lung: np.ndarray,
                      region: np.ndarray) -> dict:
    """A20: assign region pixels to thirds of each lung's own vertical extent."""
    right, left = _lung_pair(right_lung, left_lung)
    region = _mask(region, "region")
    if right.shape != left.shape or right.shape != region.shape:
        raise ValueError("right_lung, left_lung, and region must have the same shape")
    output = {}
    for side, lung in (("right", right), ("left", left)):
        zones = _lung_zone_masks(lung)
        side_px = int(np.count_nonzero(region & lung))
        output[side] = {
            "overlap_px": side_px,
            "zone_overlap_px": {name: int(np.count_nonzero(region & zone))
                                for name, zone in zones.items()},
            "zone_fraction_of_side_overlap": {
                name: (int(np.count_nonzero(region & zone)) / side_px if side_px else None)
                for name, zone in zones.items()},
        }
    union = right | left
    output["outside_lung_overlap_px"] = int(np.count_nonzero(region & ~union))
    output["method"] = "three equal vertical bands over lung pixel-cell extent; assign by pixel center; image rows increase downward"
    output["status"] = "no_lung_overlap" if not (output["right"]["overlap_px"] or output["left"]["overlap_px"]) else "ok"
    return output


def summarize_bilateral_distribution(right_lung: np.ndarray, left_lung: np.ndarray,
                                     lesion_regions: list[np.ndarray]) -> dict:
    """A23: geometry-only side distribution for independently supplied regions.

    Each input region is treated as one candidate. Eight-connected components
    are retained as subregions; any pixel overlap assigns that component to a
    side, so the result is a spatial summary rather than a disease judgment.
    """
    from scipy.ndimage import label

    right, left = _lung_pair(right_lung, left_lung)
    if right.shape != left.shape:
        raise ValueError("right_lung and left_lung must have the same shape")
    structure = np.ones((3, 3), dtype=np.uint8)
    components = []
    union_regions = np.zeros_like(right)
    empty_regions = []
    for region_index, region in enumerate(lesion_regions):
        region = _mask(region, f"lesion_regions[{region_index}]")
        if region.shape != right.shape:
            raise ValueError("all lesion regions must have the same shape as the lungs")
        union_regions |= region
        if not region.any():
            empty_regions.append(region_index)
        labels, count = label(region, structure=structure)
        for component_index in range(1, count + 1):
            component = labels == component_index
            right_px = int(np.count_nonzero(component & right))
            left_px = int(np.count_nonzero(component & left))
            if right_px and left_px:
                side = "bilateral_overlap"
            elif right_px:
                side = "right"
            elif left_px:
                side = "left"
            else:
                side = "outside_lungs"
            components.append({"region_index": region_index,
                               "component_index": component_index - 1,
                               "area_px": int(component.sum()),
                               "right_lung_overlap_px": right_px,
                               "left_lung_overlap_px": left_px,
                               "laterality": side})
    right_components = sum(c["right_lung_overlap_px"] > 0 for c in components)
    left_components = sum(c["left_lung_overlap_px"] > 0 for c in components)
    if not components:
        pattern = "no_regions"
    elif right_components and left_components:
        pattern = "bilateral"
    elif right_components:
        pattern = "right_only"
    elif left_components:
        pattern = "left_only"
    else:
        pattern = "outside_lungs_only"
    return {
        "pattern": pattern,
        "candidate_region_count": len(lesion_regions),
        "component_count": len(components),
        "right_components": int(right_components),
        "left_components": int(left_components),
        "bilateral_overlap_components": sum(c["laterality"] == "bilateral_overlap" for c in components),
        "outside_lung_components": sum(c["laterality"] == "outside_lungs" for c in components),
        "empty_region_indices": empty_regions,
        "union_region_area_px": int(union_regions.sum()),
        "union_right_overlap_px": int(np.count_nonzero(union_regions & right)),
        "union_left_overlap_px": int(np.count_nonzero(union_regions & left)),
        "union_outside_lungs_px": int(np.count_nonzero(union_regions & ~(right | left))),
        "zones": assign_lung_zones(right, left, union_regions),
        "components": components,
        "method": "8-connected pixel components; any positive overlap assigns a side; geometry only, no clinical threshold",
    }
