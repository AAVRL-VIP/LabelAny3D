"""
3D bounding box estimation utilities.

Functions for estimating oriented 3D bounding boxes from point clouds
with ground plane alignment.
"""

import numpy as np
import trimesh
import os
import json
import math
from sklearn.decomposition import PCA


# =============================================================================
# Flat-object (TV/monitor) handling
# =============================================================================
# Flat panel objects are reconstructed as razor-thin, often tilted/lying sheets,
# which makes the generic axis-aligned box collapse to a sliver with the wrong
# axis as "thickness". For these categories we instead build the box from a
# stable frame (gravity up + camera-facing normal) and clamp the front-to-back
# thickness to a fixed value when it comes out implausibly large.
# See estimate_bbox / _estimate_flat_panel_bbox.
FLAT_CATEGORIES = {
    'tv', 'television', 'monitor', 'tv_monitor', 'tvmonitor',
    'screen', 'computer_monitor', 'display',
}
FLAT_DEPTH_THRESHOLD = 0.20   # meters; clamp thickness only when it exceeds this
FLAT_FIXED_DEPTH = 0.04       # meters; fixed thickness applied when clamping


def _normalize_category(cat_name):
    """Lowercase and normalize a category name for flat-object matching."""
    if not cat_name:
        return ''
    return str(cat_name).strip().lower().replace(' ', '_')


# =============================================================================
# Basic Geometry Functions
# =============================================================================

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def rotate_y(yaw):
    """Generate a rotation matrix for yaw (around the y-axis)."""
    return np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])


def rotation_matrix_from_vectors(vec1, vec2):
    """Compute rotation matrix that rotates vec1 to vec2."""
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)

    axis = np.cross(vec1, vec2)
    cos_theta = np.dot(vec1, vec2)

    skew_symmetric = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = (
        np.eye(3) + skew_symmetric +
        np.dot(skew_symmetric, skew_symmetric) * (1 - cos_theta) / (np.linalg.norm(axis) ** 2)
    )

    return rotation_matrix


def point_to_plane_distance(plane, x, y, z):
    """Calculate the shortest distance from a point to a plane."""
    plane = np.array(plane)
    a, b, c, d = plane
    numerator = abs(a * x + b * y + c * z + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    return numerator / denominator


# =============================================================================
# Bounding Box Functions
# =============================================================================

def convert_box_vertices(center_x, center_y, center_z, l, w, h, yaw):
    """
    Generate 8 corner vertices of a 3D bounding box.

    Args:
        center_x, center_y, center_z: Box center coordinates
        l, w, h: Box dimensions (length, width, height)
        yaw: Rotation angle around y-axis

    Returns:
        8x3 array of corner vertices
    """
    local_corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])

    rotation_matrix = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    rotated_corners = np.dot(local_corners, rotation_matrix.T)
    global_corners = rotated_corners + np.array([center_x, center_y, center_z])

    return global_corners


def estimate_bbox(in_pc, cat_name=None, ground_equ=None, method='pca'):
    """
    Estimate oriented bounding box from point cloud.

    Args:
        in_pc: Input point cloud (N, 3)
        cat_name: Category name (unused, kept for compatibility)
        ground_equ: Ground plane equation [a, b, c, d] or canonical upright direction
        method: 'pca' or 'convex_hull' for yaw estimation

    Returns:
        vertices: 8 bbox vertices in camera coordinates
        center_cam: bbox center in camera coordinates
        dimension: [depth, height, width]
        R_cam: Rotation matrix from canonical to camera coordinates
    """
    # Subsample input point cloud if needed
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]

    # Rotate the point cloud to align with the ground plane
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)

    rotated_pc = np.dot(in_pc, rotation_matrix)

    # Remove NaN points
    valid_mask = ~np.isnan(rotated_pc).any(axis=1)
    rotated_pc = rotated_pc[valid_mask]

    if len(rotated_pc) == 0:
        raise ValueError("No valid points after removing NaN values")

    # Flat panel objects (TV/monitor/screen) are reconstructed as razor-thin
    # sheets that single-view depth alignment often leaves tilted/lying down, so
    # the usual axis-aligned fit puts the thin "thickness" on the wrong axis and
    # the box collapses to a sliver. For these we build the box from a stable
    # frame instead: vertical from the (shared) gravity up, the thin axis facing
    # the camera, and a fixed thickness. See _estimate_flat_panel_bbox.
    if _normalize_category(cat_name) in FLAT_CATEGORIES:
        return _estimate_flat_panel_bbox(rotated_pc, rotation_matrix)

    # Determine yaw using selected method
    if method == 'convex_hull':
        yaw = _estimate_yaw_convex_hull(rotated_pc)
    elif method == 'pca':
        yaw = _estimate_yaw_pca(rotated_pc)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'convex_hull'")

    # Rotate the point cloud to align with the x-axis and z-axis
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T

    rotated_pc_2_np = rotated_pc_2.T
    low, high = 2, 98

    x_min, y_min, z_min = np.percentile(rotated_pc_2_np, low, axis=0)
    x_max, y_max, z_max = np.percentile(rotated_pc_2_np, high, axis=0)

    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2


    print(f"[{method}] dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

    # Generate vertices in aligned space
    vertices = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)

    # Transform vertices back to camera space
    vertices = np.dot(rotate_y(-yaw), vertices.T).T
    vertices = np.dot(vertices, rotation_matrix.T)

    # Calculate center by transforming the center point directly
    center_aligned = np.array([cx, cy, cz])
    center_cam = rotation_matrix.T @ (rotate_y(-yaw) @ center_aligned)

    dimension = [dz, dy, dx]
    R_cam = rotation_matrix.T @ rotate_y(-yaw)

    return vertices, center_cam, dimension, R_cam


def _estimate_flat_panel_bbox(rotated_pc, rotation_matrix):
    """
    Build an oriented box for a flat panel (TV/monitor) in a stable frame.

    The point cloud is already ground-aligned (gravity -> [0, -1, 0]). We orient
    the box with: height along world up, the thin (thickness) axis pointing back
    toward the camera so the panel faces the viewer, and width perpendicular to
    both. Thickness is clamped to a fixed value when implausibly large, which is
    the normal case for these mis-reconstructed sheets.

    Args:
        rotated_pc: ground-aligned point cloud (N, 3), camera at the origin
        rotation_matrix: maps ground-aligned frame back to camera (row convention)

    Returns:
        vertices, center_cam, dimension [depth, height, width], R_cam
    """
    up = np.array([0.0, 1.0, 0.0])

    # Direction from the object back toward the camera (origin), in the
    # ground-aligned frame; its horizontal component is the panel normal.
    cam_dir = -rotated_pc.mean(axis=0)
    normal = cam_dir - np.dot(cam_dir, up) * up
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        # Object sits directly above/below the camera: pick an arbitrary horizontal.
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / norm

    width_axis = np.cross(up, normal)
    width_axis = width_axis / np.linalg.norm(width_axis)
    axes = np.vstack([width_axis, up, normal])  # rows: width, height, thickness

    proj = rotated_pc @ axes.T
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    center_axis = (lo + hi) / 2
    width, height, thick = hi - lo

    if thick > FLAT_DEPTH_THRESHOLD:
        print(f"[flat] clamping thickness {thick:.3f} -> {FLAT_FIXED_DEPTH:.3f}")
        thick = FLAT_FIXED_DEPTH
    print(f"[flat] width={width:.3f}, height={height:.3f}, thick={thick:.3f}")

    # Corners ordered to match convert_box_vertices: local x=width, y=height, z=thickness.
    hw, hh, ht = width / 2, height / 2, thick / 2
    local = np.array([
        [-hw, -hh, -ht], [hw, -hh, -ht], [hw, hh, -ht], [-hw, hh, -ht],
        [-hw, -hh,  ht], [hw, -hh,  ht], [hw, hh,  ht], [-hw, hh,  ht],
    ])
    corners_rot = center_axis @ axes + local @ axes  # ground-aligned frame
    vertices = (corners_rot @ rotation_matrix.T).astype(np.float16)
    center_cam = corners_rot.mean(axis=0) @ rotation_matrix.T

    # R_cam: columns are the width/height/thickness axes expressed in camera frame.
    R_cam = (axes @ rotation_matrix.T).T
    dimension = [float(thick), float(height), float(width)]  # [depth, height, width]

    return vertices, center_cam, dimension, R_cam


def _estimate_yaw_pca(rotated_pc):
    """Estimate yaw angle using PCA."""
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    return np.arctan2(yaw_vec[1], yaw_vec[0])


def _estimate_yaw_convex_hull(rotated_pc):
    """Estimate yaw angle using minimum area bounding box from convex hull."""
    from scipy.spatial import ConvexHull

    points_2d = rotated_pc[:, [0, 2]]  # X and Z coordinates

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        min_area = float('inf')
        best_yaw = 0

        for i in range(len(hull_points)):
            edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
            yaw = np.arctan2(edge[1], edge[0])

            rot_2d = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            rotated_2d = (rot_2d @ points_2d.T).T

            x_min, x_max = rotated_2d[:, 0].min(), rotated_2d[:, 0].max()
            z_min, z_max = rotated_2d[:, 1].min(), rotated_2d[:, 1].max()
            area = (x_max - x_min) * (z_max - z_min)

            if area < min_area:
                min_area = area
                best_yaw = yaw

        return best_yaw

    except Exception as e:
        print(f"ConvexHull failed: {e}, falling back to PCA")
        return _estimate_yaw_pca(rotated_pc)


# =============================================================================
# Scene Processing Functions
# =============================================================================

def _scene_gravity_up(recons_dir, objs):
    """
    Estimate a shared gravity-up direction from the upright (non-flat) objects in
    the scene. Flat panels (TV/monitor) have unreliable per-object up vectors, so
    they are excluded and instead reuse this shared estimate.

    Returns a unit 3-vector, or None if there is no usable reference object.
    """
    ups = []
    for obj in objs:
        parts = obj.split("_", 1)
        if len(parts) < 2:
            continue
        category = parts[1].split(".", 1)[0]
        if _normalize_category(category) in FLAT_CATEGORIES:
            continue
        up_path = os.path.join(recons_dir, f"{obj.split('.', 1)[0]}_canonical_upright.npy")
        if not os.path.exists(up_path):
            continue
        u = np.load(up_path)[:3]
        n = np.linalg.norm(u)
        if n < 1e-6:
            continue
        ups.append(u / n)

    if not ups:
        return None
    ref = ups[0]
    aligned = [u if np.dot(u, ref) >= 0 else -u for u in ups]  # bring to a common hemisphere
    g = np.mean(aligned, axis=0)
    gn = np.linalg.norm(g)
    if gn < 1e-6:
        return None
    return g / gn


def save_3d_with_ground_alignment_bbox(scene_dir, bbox_method='pca'):
    """
    Save 3D bounding boxes with ground alignment for all objects in a scene.

    Args:
        scene_dir: Scene directory path
        bbox_method: Method for bbox estimation - 'pca' (default) or 'convex_hull'

    Returns:
        List of bounding box dictionaries
    """
    recons_dir = os.path.join(scene_dir, "reconstruction")
    files_and_dirs = os.listdir(recons_dir)
    objs = [
        item for item in files_and_dirs
        if item not in ['full_scene.glb', 'background.ply'] and item.endswith('.glb')
    ]
    bbox_list = []

    # Shared gravity-up from upright objects; flat panels reuse it instead of
    # their own unreliable per-object up vector.
    scene_up = _scene_gravity_up(recons_dir, objs)

    for obj in objs:
        obj_dict = {}
        parts = obj.split("_", 1)
        obj_id = parts[0]
        category, _ = parts[1].split(".", 1)

        mesh = trimesh.load(os.path.join(recons_dir, obj))
        canonical_upright = np.load(
            os.path.join(recons_dir, f"{obj.split('.', 1)[0]}_canonical_upright.npy")
        )

        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.dump()
            mesh = meshes[0]

        if mesh.is_empty or mesh.area == 0 or len(mesh.faces) == 0:
            print(f"Invalid mesh at {os.path.join(recons_dir, obj)}, skipping.")
            continue

        point_cloud = mesh.sample(500)
        point_clouds = trimesh.points.PointCloud(point_cloud)

        # Use the shared gravity-up for flat panels when available.
        ground_equ = canonical_upright
        if _normalize_category(category) in FLAT_CATEGORIES and scene_up is not None:
            ground_equ = np.append(scene_up, 0.0)

        try:
            boxes3d, center_cam, dimensions, R_cam = estimate_bbox(
                np.array(point_clouds.vertices),
                category,
                ground_equ,
                method=bbox_method
            )
        except Exception as e:
            print(f"Error estimating bbox for {obj}: {e}")
            continue

        obj_dict["obj_id"] = obj_id
        obj_dict["category_name"] = category
        obj_dict["center_cam"] = center_cam.tolist()
        obj_dict["R_cam"] = R_cam.tolist()
        obj_dict["dimensions"] = dimensions
        obj_dict["bbox3D_cam"] = boxes3d.tolist()
        bbox_list.append(obj_dict)

    with open(os.path.join(scene_dir, '3dbbox_ground.json'), 'w') as json_file:
        json.dump(bbox_list, json_file)

    return bbox_list
