"""Continuous polygon/rectangle intersection area, including concave slots.

Sutherland-Hodgman clips the polygon against the rectangle's four half-planes.
Disconnected pieces may share zero-area boundary bridges; their signed areas
still add correctly. No bounding-rectangle approximation or pixel rasterization.
"""

import math

import numpy as np


def polygon_rectangle_intersection_area(polygon: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return 0.0
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    for axis, boundary, direction in ((0, x1, 1), (0, x2, -1), (1, y1, 1), (1, y2, -1)):
        if len(points) == 0:
            return 0.0
        clipped = []
        previous = points[-1]
        previous_inside = direction * (previous[axis] - boundary) >= 0
        for current in points:
            inside = direction * (current[axis] - boundary) >= 0
            if inside != previous_inside:
                fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
                intersection = previous + fraction * (current - previous)
                intersection[axis] = boundary
                clipped.append(intersection)
            if inside:
                clipped.append(current)
            previous, previous_inside = current, inside
        points = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    # Translate before shoelace to reduce cancellation for large coordinates.
    points = points - points[0]
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
