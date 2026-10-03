"""O-Quickhull: Linh, An & Hoai (2022), Algorithm 1 and Section 5.2.

The point routine assumes the paper's assumption (A). For masks, work on
4-connected components separately, treating pixels as closed unit cells.
No Euclidean convex hull, bounding-box substitute, or iterative filling is
used to construct the output. Coordinates are Cartesian internally.
"""
from __future__ import annotations
import numpy as np


def case(a, b):
    dx, dy = b - a
    if dx == 0 or dy == 0:
        return 0
    return {(False, True): 1, (False, False): 2,
            (True, False): 3, (True, True): 4}[dx > 0, dy > 0]


def right_vertex(a, b):
    # Choose the elbow v for which orient(a,v,b) < 0 (Definition 9).
    if (b[0] - a[0]) * (b[1] - a[1]) < 0:
        return np.array([b[0], a[1]])
    return np.array([a[0], b[1]])


def orient(a, b, p):
    return (b[0]-a[0])*(p[:, 1]-a[1]) - (b[1]-a[1])*(p[:, 0]-a[0])


def on_right(points, a, b):
    if not len(points) or not case(a, b):
        return points[:0]
    v = right_vertex(a, b)
    return points[(orient(a, v, points) < 0) & (orient(v, b, points) < 0)]


def o_quickhull(a, b, points):
    """Ordered interior extreme points; explicit stack avoids recursion limits.

    points must be P_ab, not the whole input. Squared Euclidean distance to
    the right-line vertex implements Definition 11 without a square root.
    """
    j = case(a, b)
    if not j:
        return []
    output = []
    stack = [(a, b, points)]
    while stack:
        task = stack.pop()
        if len(task) == 1:
            output.append(task[0])
            continue
        left, right, candidates = task
        if not len(candidates):
            continue
        delta = candidates - right_vertex(left, right)
        c = candidates[np.argmax(np.einsum('ij,ij->i', delta, delta))]
        # The same-case guards are essential: Algorithm 1 steps (b) and (c).
        if case(c, right) == j:
            stack.append((c, right, on_right(candidates, c, right)))
        stack.append((c,))
        if case(left, c) == j:
            stack.append((left, c, on_right(candidates, left, c)))
    return output


def hull_polygon(points):
    """Return rectilinear boundary for inputs satisfying assumption (A).

    Not a general solution for disconnected arbitrary point clouds. Closed
    polygon is represented without repeating its initial vertex.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('Expected finite points of shape (N,2)')
    p = np.unique(p, axis=0)
    if len(p) < 2:
        return p
    x, y = p.T
    top, bottom = p[y == y.max()], p[y == y.min()]
    left, right = p[x == x.min()], p[x == x.max()]
    # q1,q1',q2,q2',q3,q3',q4,q4' in Figure 5.
    pairs = [(top[np.argmin(top[:,0])], left[np.argmax(left[:,1])]),
             (left[np.argmin(left[:,1])], bottom[np.argmin(bottom[:,0])]),
             (bottom[np.argmax(bottom[:,0])], right[np.argmin(right[:,1])]),
             (right[np.argmax(right[:,1])], top[np.argmax(top[:,0])])]
    extremes = []
    for a, b in pairs:
        extremes.extend([a, *o_quickhull(a, b, on_right(p, a, b)), b])
    vertices = []
    for a, b in zip(extremes, extremes[1:] + extremes[:1]):
        if not vertices or not np.array_equal(vertices[-1], a):
            vertices.append(a)
        if case(a, b):
            vertices.append(right_vertex(a, b))
    if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
        vertices.pop()
    return np.asarray(vertices)


def polygon_area(p):
    if len(p) < 3:
        return 0.0
    return float(abs(np.dot(p[:,0], np.roll(p[:,1], -1)) -
                     np.dot(p[:,1], np.roll(p[:,0], -1))) / 2)


def mask_hulls(mask, min_component_area=1):
    """One hull per 4-connected mask component, in image pixel-edge coordinates.

    Preserve disconnected visible fragments without inventing bridges across
    occluders. Multiple polygons still belong to ONE YOLO vehicle detection.
    Component hulls can overlap; their areas are not a union-area estimate.
    """
    import cv2
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError('Expected a 2D binary mask')
    if min_component_area < 1:
        raise ValueError('min_component_area must be >= 1')
    mask = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    output = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < min_component_area:
            continue
        component = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        # All boundary pixels, not a simplified polygon. Cell corners preserve
        # horizontal/vertical ties, one-pixel objects, and thin structures.
        eroded = cv2.erode(component, np.ones((3,3), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0)
        yy, xx = np.nonzero(component & ~eroded)
        base = np.column_stack([xx+x, yy+y]).astype(np.float64)
        corners = np.concatenate([base + offset for offset in
                                  [(0,0), (1,0), (0,1), (1,1)]])
        corners[:,1] *= -1  # image y-down -> Cartesian y-up
        polygon = hull_polygon(corners)
        polygon[:,1] *= -1
        output.append({'polygon': polygon.tolist(), 'mask_area_px': area,
                       'hull_area_px': polygon_area(polygon)})
    return output
