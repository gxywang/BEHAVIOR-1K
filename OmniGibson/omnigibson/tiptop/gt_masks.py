"""Oracle object masks from rendered depth + object meshes, without Isaac's instance-segmentation annotator.

Isaac's ``seg_instance`` annotator segfaults on the first render in large BEHAVIOR house scenes (see README/DEPLOYMENT),
so task runs render rgb + depth only. Ground-truth masks are then computed geometrically: a pixel belongs to an object
when its unprojected depth point lies on (within ``tol`` of) that object's surface mesh at its current pose.

No OmniGibson / torch imports: numpy + scipy (a trimesh dependency) + trimesh only, so this is unit-testable.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from omnigibson.tiptop.protocol import depth_to_points

# Triangles longer than this are subdivided before the proximity test: the candidate-pair search radius grows with the
# largest triangle, so a few huge faces (a table top) would otherwise pair every point with every triangle.
MAX_TRIANGLE_EDGE = 0.03


def masks_from_geometry(depth, intrinsics, world_from_cam, meshes: dict, tol: float = 0.008) -> dict:
    """Per-object boolean masks (H, W) from a z-depth image and the objects' surface meshes.

    Args:
        depth: (H, W) float z-depth in metres (distance to the image plane); 0 (or non-finite) = invalid pixel.
        intrinsics: (3, 3) OpenCV pinhole matrix for the same resolution.
        world_from_cam: (4, 4) pose of the OpenCV camera frame (+x right, +y down, +z forward) expressed in the frame
            the meshes are in. Any consistent frame works (world or robot base); mixing frames gives empty masks.
        meshes: {label: trimesh.Trimesh} surface meshes at their current poses, in the ``world_from_cam`` frame.
        tol: surface distance (m) within which a depth point counts as lying on the mesh.

    Returns:
        {label: (H, W) bool} for every requested label, mutually disjoint like instance segmentation; a label whose
        mesh has no candidate pixels (outside the view, fully occluded, empty mesh) gets an all-False mask. Pixel
        counts per label are ``mask.sum()``.

    Method: unproject the depth (``protocol.depth_to_points``), keep the finite points inside the mesh AABB expanded by
    ``tol`` (cheap prefilter), then test the point-to-surface distance (vertex KD-tree first; exact point-triangle
    distance only for the undecided points and the triangles whose centroid is close enough). A pixel within ``tol``
    of several labelled surfaces (two objects in contact: the contact band of each lies within ``tol`` of the other)
    is assigned to the label with the smallest exact surface distance, so masks never overlap; that exact distance is
    only computed for the contested pixels, so the common case costs nothing extra.

    Caveat (contact halo): depth points of an UNLABELLED support surface that lie within ``tol`` of an object's bottom
    edge (the table around a mug's foot) are attributed to the object, giving a halo of ``tol`` / (pixel footprint)
    pixels along the contact line (1-2 px at 720 px / 0.5 m, ~4 px at 1280 px). When the support surface is itself a
    label the closest-surface rule gives it those pixels instead. Occluders are never attributed: their depth points
    are far from the occluded surface. Measured on the Panda tabletop demo (1280x720, fx 667, seg_instance as
    reference): every seg pixel is within 2 mm of the mesh, and tol 2 / 4 / 8 mm gives IoU 0.97 / 0.95 / 0.90 (mug)
    and 1.00 / 0.99 / 0.96 (bowl), the whole difference being the table halo. The 8 mm default keeps headroom for
    coarser cameras (the sampling error grows with the pixel footprint).
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    pts = depth_to_points(depth, np.asarray(intrinsics, dtype=np.float64), np.asarray(world_from_cam, dtype=np.float64))
    flat = pts.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    labels = list(meshes)
    hits = []  # per label: flat pixel indices within tol of the surface
    for label in labels:
        mesh = meshes[label]
        idx = np.zeros(0, dtype=np.intp)
        if mesh is not None and len(mesh.faces) > 0:
            lo, hi = mesh.bounds[0] - tol, mesh.bounds[1] + tol
            with np.errstate(invalid="ignore"):  # NaN rows compare False and are excluded by ``finite`` anyway
                candidates = np.flatnonzero(finite & (flat >= lo).all(axis=1) & (flat <= hi).all(axis=1))
            if candidates.size:
                idx = candidates[points_within_tol(mesh, flat[candidates], tol)]
        hits.append(idx)
    owner = np.full(depth.size, -1, dtype=np.int64)
    n_hits = np.zeros(depth.size, dtype=np.int32)
    for i, idx in enumerate(hits):
        owner[idx] = i
        n_hits[idx] += 1
    contested = np.flatnonzero(n_hits > 1)
    if contested.size:  # objects in contact: the pixel goes to the closest surface (ties / no finite distance: first)
        best = np.full(contested.size, np.inf)
        for i, (label, idx) in enumerate(zip(labels, hits)):
            sel = np.flatnonzero(np.isin(contested, idx, assume_unique=True))
            if sel.size == 0:
                continue
            dist = surface_distances(meshes[label], flat[contested[sel]], tol)
            better = dist < best[sel]
            best[sel[better]] = dist[better]
            owner[contested[sel[better]]] = i
    return {label: (owner == i).reshape(depth.shape) for i, label in enumerate(labels)}


def _prepare(mesh: trimesh.Trimesh):
    """Subdivide oversized triangles; return (triangles, referenced vertices, max edge, centroids, centroid radii)."""
    tris = np.asarray(mesh.triangles, dtype=np.float64)
    edge_max = float(np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max())
    if edge_max > MAX_TRIANGLE_EDGE:
        mesh = mesh.subdivide_to_size(MAX_TRIANGLE_EDGE)
        tris = np.asarray(mesh.triangles, dtype=np.float64)
        edge_max = float(np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max())
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[np.unique(mesh.faces)]  # referenced vertices only
    centroids = tris.mean(axis=1)
    radii = np.linalg.norm(tris - centroids[:, None, :], axis=2).max(axis=1)
    return tris, vertices, edge_max, centroids, radii


def _triangle_distances(tris, centroids, radii, points, tol: float) -> np.ndarray:
    """(N,) exact distance from each point to the nearest triangle when that is <= tol, +inf otherwise.

    Only triangles that can be within tol are tested: |p - centroid| <= tol + (centroid-to-vertex radius).
    """
    dist = np.full(len(points), np.inf)
    pairs = cKDTree(points).sparse_distance_matrix(cKDTree(centroids), tol + float(radii.max()), output_type="ndarray")
    keep = pairs["v"] <= tol + radii[pairs["j"]]
    pi, ti = pairs["i"][keep], pairs["j"][keep]
    if pi.size:
        closest = trimesh.triangles.closest_point(tris[ti], points[pi])
        np.minimum.at(dist, pi, np.linalg.norm(closest - points[pi], axis=1))
        dist[dist > tol] = np.inf
    return dist


def points_within_tol(mesh: trimesh.Trimesh, points, tol: float) -> np.ndarray:
    """(N,) bool: whether each point lies within ``tol`` of the surface of ``mesh`` (exact point-triangle distance)."""
    points = np.asarray(points, dtype=np.float64)
    hit = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or len(mesh.faces) == 0:
        return hit
    tris, vertices, edge_max, centroids, radii = _prepare(mesh)
    # vertices: within tol of a vertex -> hit; farther than tol + edge_max from every vertex -> miss
    # (every point of a triangle is within edge_max of one of its vertices); exact distance only for the rest
    d_vertex, _ = cKDTree(vertices).query(points, distance_upper_bound=tol + edge_max)
    hit[d_vertex <= tol] = True
    undecided = np.flatnonzero(~hit & np.isfinite(d_vertex))
    if undecided.size:
        hit[undecided] = np.isfinite(_triangle_distances(tris, centroids, radii, points[undecided], tol))
    return hit


def surface_distances(mesh: trimesh.Trimesh, points, tol: float) -> np.ndarray:
    """(N,) float: exact distance from each point to the surface of ``mesh`` when <= ``tol``, +inf otherwise."""
    points = np.asarray(points, dtype=np.float64)
    dist = np.full(len(points), np.inf)
    if len(points) == 0 or len(mesh.faces) == 0:
        return dist
    tris, vertices, edge_max, centroids, radii = _prepare(mesh)
    d_vertex, _ = cKDTree(vertices).query(points, distance_upper_bound=tol + edge_max)
    near = np.flatnonzero(np.isfinite(d_vertex))  # the others are farther than tol from every triangle
    if near.size:
        dist[near] = _triangle_distances(tris, centroids, radii, points[near], tol)
    return dist
