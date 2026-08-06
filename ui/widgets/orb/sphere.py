"""Geometry for the orb.

The Age of Ultron J.A.R.V.I.S. hologram was "an internal network of
connections inside a spherical center", circumscribed by "audio-reactive
exterior rings of data". Both halves are built here as real 3D geometry and
projected, rather than faked with 2D gradients: a great circle on a sphere
projects to a straight-looking arc whose depth genuinely varies, and that
depth is what makes the sphere read as a volume instead of a disc.

All math is vectorized. One (N, P, 3) array is rotated per frame with a
single matmul; the widget only walks the result to emit polylines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def fibonacci_normals(count: int) -> np.ndarray:
    """`count` roughly evenly spaced unit vectors on the sphere.

    Used as great-circle normals. Even spacing matters: random normals
    clump, and clumped great circles read as a tangle rather than a weave.
    """
    i = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(np.clip(1.0 - 2.0 * i / count, -1.0, 1.0))
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def _orthonormal_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to each normal."""
    count = normals.shape[0]
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    # Swap the helper axis where it is nearly parallel to the normal,
    # otherwise the cross product collapses to zero.
    parallel = np.abs(normals[:, 2]) > 0.9
    helper[parallel] = np.array([1.0, 0.0, 0.0])

    u = np.cross(normals, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    v = np.cross(normals, u)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return u, v


@dataclass(frozen=True)
class Filaments:
    """Unit-sphere great circles. `points` is (n_filaments, n_points, 3)."""

    points: np.ndarray
    scales: np.ndarray  # per-filament radius factor, for a layered weave

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    @property
    def samples(self) -> int:
        return int(self.points.shape[1])


def build_filaments(count: int, samples: int, seed: int = 11) -> Filaments:
    """Great circles at slightly varied radii.

    Uniform-radius circles all touch the same shell and the interior looks
    hollow. Staggering the radii fills the volume, which is what gives the
    dense woven core.
    """
    count = max(3, int(count))
    samples = max(12, int(samples))

    normals = fibonacci_normals(count)
    u, v = _orthonormal_basis(normals)

    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=True)
    cos_t = np.cos(t)[None, :, None]
    sin_t = np.sin(t)[None, :, None]
    points = u[:, None, :] * cos_t + v[:, None, :] * sin_t

    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.62, 1.0, size=count)
    # Guarantee a few circles ride the outer shell so the silhouette reads.
    scales[:: max(1, count // 5)] = 1.0
    points = points * scales[:, None, None]

    return Filaments(points=np.ascontiguousarray(points), scales=scales)


# Minor/major axis ratio per ring: how open each ellipse looks. Deliberately
# kept well away from zero so a ring is never seen edge-on.
DEFAULT_RING_RATIOS = (0.34, 0.24, 0.46)


def build_ring_bases(rolls_rad, ratios: tuple[float, ...] | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Plane bases for the exterior rings, as `(u, v)` of shape (count, 3).

    The rings do not tumble with the sphere. They are spinning platters seen
    at a fixed viewing tilt, so each keeps a constant minor/major ratio and
    only precesses in screen space via `rolls_rad`.

    Sharing the sphere's rotation instead puts a ring edge-on periodically,
    where its ellipse collapses to a line and reads as a spike skewering the
    orb rather than as a ring around it.

    `v` keeps a real z component, so depth-based occlusion against the
    sphere stays correct.
    """
    rolls = np.atleast_1d(np.asarray(rolls_rad, dtype=np.float64))
    count = int(rolls.size)
    ratios = ratios or DEFAULT_RING_RATIOS
    k = np.array([ratios[i % len(ratios)] for i in range(count)], dtype=np.float64)

    cos_r, sin_r = np.cos(rolls), np.sin(rolls)
    depth_scale = np.sqrt(np.maximum(0.0, 1.0 - k * k))

    u = np.stack([cos_r, sin_r, np.zeros(count)], axis=1)
    v = np.stack([-sin_r * k, cos_r * k, depth_scale], axis=1)
    return np.ascontiguousarray(u), np.ascontiguousarray(v)


def ring_points(u: np.ndarray, v: np.ndarray, angles: np.ndarray,
                phases: np.ndarray) -> np.ndarray:
    """Sample every ring at `angles + phases[ring]`. Returns (n_rings, n, 3)."""
    theta = angles[None, :] + phases[:, None]
    return u[:, None, :] * np.cos(theta)[..., None] + v[:, None, :] * np.sin(theta)[..., None]


def rotation_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                    roll_deg: float = 0.0) -> np.ndarray:
    """Intrinsic Y (yaw), X (pitch), Z (roll) rotation."""
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def project(points: np.ndarray, rot: np.ndarray, cx: float, cy: float,
            radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate and orthographically project.

    Returns `(xy, depth)` where xy is (..., 2) in widget pixels and depth is
    (...) in roughly [-1, 1]; positive depth is toward the viewer.
    """
    rotated = points @ rot.T
    xy = np.empty(rotated.shape[:-1] + (2,), dtype=np.float64)
    xy[..., 0] = cx + rotated[..., 0] * radius
    # Screen y grows downward, so flip to keep the rotation reading correctly.
    xy[..., 1] = cy - rotated[..., 1] * radius
    return xy, rotated[..., 2]


def depth_bands(depth: np.ndarray, bands: int) -> np.ndarray:
    """Bucket depth into `bands` levels, 0 = farthest.

    Per-point alpha would need a pen change per segment. Bucketing lets a
    whole run of points share one pen, which is the difference between
    ~100 polylines a frame and ~2000 individual line calls.
    """
    normalized = (np.clip(depth, -1.0, 1.0) + 1.0) * 0.5
    return np.clip((normalized * bands).astype(np.int32), 0, bands - 1)


def runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a 1D label array into `(start, stop_exclusive, label)` runs."""
    if values.size == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [values.size]))
    return [(int(a), int(b), int(values[a])) for a, b in zip(starts, stops)]
