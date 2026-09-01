#!/usr/bin/env python3
"""
Thermal Mosaic from COLMAP Poses (Single-Plane Reprojection)

Goal:
- Create a single orthophoto-like thermal mosaic (not a true orthophoto) by reprojecting raw thermal images
  onto a reference plane using COLMAP-optimized camera poses, preserving raw pixel values for an AI pipeline.

Assumptions:
- Input images are raw thermal images (do not modify intensities).
- Camera intrinsics are known (from COLMAP cameras.txt or provided).
- Camera poses come from COLMAP sparse reconstruction (images.txt) — world->camera quaternion and translation.
- No dense point cloud or true orthorectification.
- Single reference plane Z = Z0 in world frame (or configurable plane).
- Output is one global mosaic image.

Method:
- Parse COLMAP `cameras.txt` (intrinsics) and `images.txt` (poses).
- Convert COLMAP quaternions to rotation matrices R_wc (world->camera). Invert to get R_cw and camera center C.
- For each image pixel (u,v), compute its ray in world coordinates:
    v_c = K^{-1} [u, v, 1], v_w = R_cw @ v_c, origin C
  Intersect ray with plane Z=Z0: lambda = (Z0 - C.z) / v_w.z; P_w = C + lambda * v_w.
- Define a global mosaic grid in plane XY using chosen resolution. Forward-map pixel intensities to mosaic grid.
- Fuse overlaps with simple policy (e.g., last-write-wins or keep-first). Preserve intensities (nearest-neighbor).

Usage:
    python scripts/thermal_mosaic_from_colmap.py \
        --images_path /path/to/images \
        --model_path /path/to/colmap_model_txt \
        --plane_z 0.0 \
        --resolution 0.1 \
        --output /path/to/mosaic.png

Notes:
- This is a minimal example (NumPy + OpenCV optional for IO). It avoids expensive blending to keep thermal values.
- Overlap handling: for thermal imagery, prefer nearest-neighbor, and optionally keep-first or max-intensity.

"""
import argparse
import math
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Optional: use OpenCV for faster IO if available
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def parse_cameras_txt(path: Path) -> Dict[int, Dict[str, float]]:
    """Parse COLMAP cameras.txt to retrieve intrinsics.
    Returns dict: camera_id -> {model, width, height, fx, fy, cx, cy}
    Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL minimally.
    """
    cams: Dict[int, Dict[str, float]] = {}
    if not path.exists():
        raise FileNotFoundError(f"cameras.txt not found at {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2]); height = int(parts[3])
            params = list(map(float, parts[4:]))
            fx = fy = cx = cy = None
            if model == 'SIMPLE_PINHOLE':
                # params: f cx cy
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == 'PINHOLE':
                # params: fx fy cx cy
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == 'SIMPLE_RADIAL':
                # params: f cx cy k
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            cams[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            }
    return cams


def quaternion_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (Hamilton) to rotation matrix. Returns R_wc (world->camera)."""
    # Normalize sign to fix 180° ambiguity
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    x, y, z = qx, qy, qz
    w = qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - s*(yy+zz),     s*(xy - wz),     s*(xz + wy)],
        [    s*(xy + wz), 1 - s*(xx + zz),     s*(yz - wx)],
        [    s*(xz - wy),     s*(yz + wx), 1 - s*(xx + yy)],
    ], dtype=float)
    return R


def parse_images_txt(path: Path) -> Dict[str, Dict]:
    """Parse COLMAP images.txt.
    Returns dict: name -> {camera_id, qw,qx,qy,qz, tx,ty,tz}
    """
    if not path.exists():
        raise FileNotFoundError(f"images.txt not found at {path}")
    data: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.strip() for ln in fh.readlines()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        if not ln or ln.startswith('#'):
            continue
        parts = ln.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        data[name] = {
            'image_id': image_id,
            'camera_id': camera_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
        }
        # skip points2D line
        if i < len(lines):
            if lines[i] and not lines[i].startswith('#'):
                i += 1
    return data


def build_K(cam: Dict[str, float]) -> np.ndarray:
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K


def camera_center_from_pose(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """Camera center C_w in world coordinates: C = -R^T * t."""
    return -R_wc.T @ t_wc


def project_image_to_plane(name: str, img: np.ndarray, K: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray,
                            plane_z: float, origin_xy: Tuple[float, float], resolution: float,
                            mosaic: np.ndarray, weight: np.ndarray) -> None:
    """Forward-project image pixels (nearest-neighbor) onto plane Z=plane_z, fill mosaic.
    origin_xy: (xmin, ymin) defining mosaic origin in plane coordinates.
    mosaic: preallocated HxW array, weight: same shape (for overlap policy).
    """
    H_img, W_img = img.shape[:2]
    # Invert pose to get camera->world rotation
    R_cw = R_wc.T
    # Camera center
    C = camera_center_from_pose(R_wc, t_wc)
    # Precompute inverse intrinsics
    K_inv = np.linalg.inv(K)

    # Build grid of pixel coords
    us = np.arange(W_img)
    vs = np.arange(H_img)
    uu, vv = np.meshgrid(us, vs)

    ones = np.ones_like(uu, dtype=float)
    pix_h = np.stack([uu.astype(float), vv.astype(float), ones], axis=-1)  # HxWx3
    # Rays in camera frame
    v_c = (K_inv @ pix_h.reshape(-1, 3).T).T  # (N,3)
    # Rays in world frame
    v_w = (R_cw @ v_c.T).T  # (N,3)

    Cz = C[2]
    vz = v_w[:, 2]
    # Avoid division by zero (rays parallel to plane)
    mask = np.abs(vz) > 1e-8
    lam = np.zeros_like(vz)
    lam[mask] = (plane_z - Cz) / vz[mask]

    # Intersection points
    P = C.reshape(1, 3) + (lam.reshape(-1, 1) * v_w)  # (N,3)
    X = P[:, 0]
    Y = P[:, 1]

    xmin, ymin = origin_xy
    # Mosaic coordinates
    j = ((X - xmin) / resolution).astype(int)  # x -> column
    i = ((Y - ymin) / resolution).astype(int)  # y -> row

    H_mos, W_mos = mosaic.shape[:2]
    valid = (i >= 0) & (i < H_mos) & (j >= 0) & (j < W_mos) & mask

    # Forward write: last-write-wins policy (minimal blending)
    # Optionally, use weight to keep-first: only write where weight==0
    src_vals = img.reshape(-1)[valid]
    ii = i[valid]
    jj = j[valid]
    # Keep-first: only write where weight==0
    w_subset = weight[ii, jj]
    keep = w_subset == 0
    if np.any(keep):
        mosaic[ii[keep], jj[keep]] = src_vals[keep]
        weight[ii[keep], jj[keep]] = 1


def estimate_mosaic_bounds(images: Dict[str, Dict], cams: Dict[int, Dict], images_path: Path,
                           plane_z: float) -> Tuple[float, float, float, float]:
    """Estimate XY bounds on the plane by projecting image corners.
    Returns (xmin, ymin, xmax, ymax).
    """
    xs = []
    ys = []
    for name, pose in images.items():
        cam = cams[pose['camera_id']]
        K = build_K(cam)
        R_wc = quaternion_to_R(pose['qw'], pose['qx'], pose['qy'], pose['qz'])
        t_wc = np.array([pose['tx'], pose['ty'], pose['tz']], dtype=float)
        C = camera_center_from_pose(R_wc, t_wc)
        R_cw = R_wc.T
        H_img, W_img = cam['height'], cam['width']
        corners = np.array([[0,0,1],[W_img-1,0,1],[0,H_img-1,1],[W_img-1,H_img-1,1]], dtype=float)
        K_inv = np.linalg.inv(K)
        v_c = (K_inv @ corners.T).T
        v_w = (R_cw @ v_c.T).T
        vz = v_w[:,2]
        mask = np.abs(vz) > 1e-8
        lam = np.zeros_like(vz)
        lam[mask] = (plane_z - C[2]) / vz[mask]
        P = C.reshape(1,3) + lam.reshape(-1,1) * v_w
        xs.extend(P[:,0].tolist())
        ys.extend(P[:,1].tolist())
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax


def load_image(path: Path) -> np.ndarray:
    if HAS_CV2:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if img.ndim == 3:
            # Convert to single channel if needed (assume thermal is single channel)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = img[...,0]  # take first channel
        return img


def save_image(path: Path, img: np.ndarray) -> None:
    if HAS_CV2:
        cv2.imwrite(str(path), img)
    else:
        Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser(description="Thermal mosaic from COLMAP poses (single-plane)")
    ap.add_argument('--images_path', required=True, help='Path to raw thermal images')
    ap.add_argument('--model_path', required=True, help='Path to COLMAP text model (contains cameras.txt, images.txt)')
    ap.add_argument('--plane_z', type=float, default=0.0, help='Reference plane Z in world coords (default 0.0)')
    ap.add_argument('--resolution', type=float, default=0.1, help='Mosaic resolution in meters per pixel')
    ap.add_argument('--output', required=True, help='Output mosaic image file')
    ap.add_argument('--overlap', choices=['keep-first','last-wins'], default='keep-first', help='Overlap policy')
    args = ap.parse_args()

    images_path = Path(args.images_path)
    model_path = Path(args.model_path)
    cameras_txt = model_path / 'cameras.txt'
    images_txt = model_path / 'images.txt'

    cams = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)

    # Estimate mosaic bounds
    xmin, ymin, xmax, ymax = estimate_mosaic_bounds(images, cams, images_path, args.plane_z)
    width_m = xmax - xmin
    height_m = ymax - ymin
    W_mos = max(1, int(math.ceil(width_m / args.resolution)))
    H_mos = max(1, int(math.ceil(height_m / args.resolution)))

    print(f"Mosaic bounds XY: [{xmin:.2f},{ymin:.2f}] to [{xmax:.2f},{ymax:.2f}] meters")
    print(f"Mosaic size: {W_mos} x {H_mos} pixels @ {args.resolution} m/px")

    mosaic = np.zeros((H_mos, W_mos), dtype=np.uint16)
    weight = np.zeros_like(mosaic, dtype=np.uint8)

    # Forward project images
    names = list(images.keys())
    for idx, name in enumerate(names):
        pose = images[name]
        cam = cams[pose['camera_id']]
        K = build_K(cam)
        R_wc = quaternion_to_R(pose['qw'], pose['qx'], pose['qy'], pose['qz'])
        t_wc = np.array([pose['tx'], pose['ty'], pose['tz']], dtype=float)
        img_path = images_path / name
        if not img_path.exists():
            print(f"[warn] Image not found: {img_path}")
            continue
        img = load_image(img_path)
        if img.shape[1] != cam['width'] or img.shape[0] != cam['height']:
            print(f"[warn] Image size mismatch for {name}: got {img.shape[::-1]}, expected {(cam['width'], cam['height'])}")
        project_image_to_plane(name, img, K, R_wc, t_wc, args.plane_z, (xmin, ymin), args.resolution, mosaic, weight)
        if (idx+1) % 5 == 0:
            print(f"Projected {idx+1}/{len(names)} images")

    # Save mosaic
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(out_path, mosaic)
    print(f"Saved mosaic to {out_path}")


def create_mosaic_from_rotated_images(
    rotated_images_dir: Path,
    out_mosaic_path: Path,
    plane_z: float = 0.0,
    resolution: float = 0.1,
    camera_meta: dict = None,
    log: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Create a single-plane mosaic from pre-rotated images using grid stitching.
    
    Args:
        rotated_images_dir: Directory containing rotated image files (PNG or JPG).
        out_mosaic_path: Output path for the mosaic TIF/PNG.
        plane_z: Reference plane Z coordinate (default 0.0 m).
        resolution: Mosaic resolution in meters per pixel.
        camera_meta: Camera metadata dict with lat/lon for georeferencing (optional).
        log: Optional callback for mosaic construction details.
    
    Returns:
        Path to the generated mosaic file.
    
    Note:
    - Rotated images are assumed to be aligned to north (camera heading rotated to 0°).
    - A simple grid-based stitching is used: place images at their geo-coordinates and blend/average overlaps.
    - For thermal or grayscale images, nearest-neighbor is preferred to preserve intensity.
    """
    if not rotated_images_dir.exists():
        raise FileNotFoundError(f"Rotated images directory not found: {rotated_images_dir}")
    
    image_files = sorted([p for p in rotated_images_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if not image_files:
        raise ValueError(f"No images found in {rotated_images_dir}")
    
    # Calculate geographic bounds from camera metadata if available
    if camera_meta:
        import math
        
        # Extract coordinates and estimate bounds
        coords = []
        img_to_coords = {}
        img_to_meta = {}
        for key, entry in camera_meta.items():
            if key.startswith('__') or not isinstance(entry, dict):
                continue
            lat = entry.get('latitude') or entry.get('lat')
            lon = entry.get('longitude') or entry.get('lon')
            if lat is not None and lon is not None:
                coords.append((float(lat), float(lon)))
                # Match to image file by name
                for img_path in image_files:
                    # Try matching by stem (filename without extension)
                    if key in img_path.name or img_path.stem in key or key.startswith(img_path.stem):
                        img_to_coords[img_path.name] = (float(lat), float(lon))
                        img_to_meta[img_path.name] = entry
                        break
        
        if coords and len(coords) > 1:
            # Calculate bounds
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            center_lat = (min_lat + max_lat) / 2
            
            # Convert resolution (meters) to degrees
            lat_rad = math.radians(center_lat)
            meters_per_deg_lon = 111320 * math.cos(lat_rad)
            meters_per_deg_lat = 111320
            
            # Keep the requested output resolution, but scale every full source
            # image to its metadata-derived ground footprint on that canvas.
            # This fixes overlap geometry without cropping source content.
            footprint_sizes: Dict[str, Tuple[int, int]] = {}
            footprint_width_m = []
            footprint_height_m = []
            metadata_mpp = []
            for img_path in image_files:
                with Image.open(img_path) as source_image:
                    source_width, source_height = source_image.size
                entry = img_to_meta.get(img_path.name, {})
                try:
                    image_mpp = float(entry.get("meters_per_pixel") or resolution)
                except (TypeError, ValueError):
                    image_mpp = float(resolution)
                if not math.isfinite(image_mpp) or image_mpp <= 0:
                    image_mpp = float(resolution)
                metadata_mpp.append(image_mpp)
                footprint_width_m.append(source_width * image_mpp)
                footprint_height_m.append(source_height * image_mpp)
                footprint_sizes[img_path.name] = (
                    max(1, int(round(source_width * image_mpp / resolution))),
                    max(1, int(round(source_height * image_mpp / resolution))),
                )
            img_width_deg = max(footprint_width_m) / meters_per_deg_lon
            img_height_deg = max(footprint_height_m) / meters_per_deg_lat
            alignment_emit = log or (lambda message: print(f"[mosaic-align] {message}"))
            alignment_emit(
                "Ground footprints: output_resolution="
                f"{resolution:.6f} m/px, metadata_GSD_median={float(np.median(metadata_mpp)):.6f} m/px, "
                f"metadata_GSD_range={min(metadata_mpp):.6f}–{max(metadata_mpp):.6f} m/px."
            )
            
            # Add padding for image extent
            min_lon -= img_width_deg
            max_lon += img_width_deg
            min_lat -= img_height_deg
            max_lat += img_height_deg
            
            # Calculate canvas size in pixels
            canvas_width_deg = max_lon - min_lon
            canvas_height_deg = max_lat - min_lat
            canvas_width = int(canvas_width_deg * meters_per_deg_lon / resolution)
            canvas_height = int(canvas_height_deg * meters_per_deg_lat / resolution)
            
            # Limit canvas size to reasonable bounds (prevent memory issues)
            max_dim = 20000
            if canvas_width > max_dim or canvas_height > max_dim:
                scale = min(max_dim / canvas_width, max_dim / canvas_height)
                canvas_width = int(canvas_width * scale)
                canvas_height = int(canvas_height * scale)
                print(f"Warning: Canvas too large, scaled down to {canvas_width}x{canvas_height}")
            
            print(f"[mosaic] Bounds: lat=[{min_lat:.6f}, {max_lat:.6f}], lon=[{min_lon:.6f}, {max_lon:.6f}]")
            print(f"[mosaic] Canvas size: {canvas_width}x{canvas_height} pixels")
            
            # Create canvas with transparency support
            canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

            final_centers = {}
            for img_path in image_files:
                if img_path.name not in img_to_coords:
                    continue
                lat, lon = img_to_coords[img_path.name]
                final_centers[img_path.name] = (
                    ((lon - min_lon) / canvas_width_deg) * canvas_width,
                    ((max_lat - lat) / canvas_height_deg) * canvas_height,
                )
            alignment_emit("Placing images from final camera metadata; no separate mosaic matcher is run.")
            
            # Voronoi nearest-center with Gaussian blending at edges only
            distance_map = np.full((canvas_height, canvas_width), np.inf, dtype=np.float32)
            second_distance_map = np.full((canvas_height, canvas_width), np.inf, dtype=np.float32)
            image_index_map = np.full((canvas_height, canvas_width), -1, dtype=np.int32)
            
            images_data = []
            for idx, img_path in enumerate(image_files):
                if img_path.name not in img_to_coords:
                    print(f"Warning: No coordinates for {img_path.name}, skipping")
                    continue
                
                try:
                    img = Image.open(img_path)
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    img_array = np.array(img)
                    target_w, target_h = footprint_sizes[img_path.name]
                    if img.size != (target_w, target_h):
                        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    rotation_correction = float(
                        (img_to_meta.get(img_path.name) or {}).get("row_alignment_rotation_deg") or 0.0
                    )
                    if abs(rotation_correction) > 1e-6:
                        # GeoJSON uses positive corrections clockwise because image Y
                        # grows downward. Pillow's positive angle is counter-clockwise.
                        img = img.rotate(-rotation_correction, resample=Image.Resampling.BICUBIC, expand=True)
                    img_array = np.array(img)
                    h, w = img_array.shape[:2]
                    center_x, center_y = final_centers.get(
                        img_path.name,
                        (
                            ((img_to_coords[img_path.name][1] - min_lon) / canvas_width_deg) * canvas_width,
                            ((max_lat - img_to_coords[img_path.name][0]) / canvas_height_deg) * canvas_height,
                        ),
                    )
                    x_px = int(round(center_x - w / 2.0))
                    y_px = int(round(center_y - h / 2.0))
                    
                    # Image center in canvas coordinates
                    img_center_x = x_px + w // 2
                    img_center_y = y_px + h // 2
                    
                    # Store image data
                    images_data.append({
                        'array': img_array,
                        'x_offset': x_px,
                        'y_offset': y_px,
                        'center_x': img_center_x,
                        'center_y': img_center_y,
                        'width': w,
                        'height': h,
                        'name': img_path.name,
                        'index': idx
                    })
                    
                    # Calculate distance from image center for all canvas pixels this image covers
                    y_start = max(0, y_px)
                    y_end = min(canvas_height, y_px + h)
                    x_start = max(0, x_px)
                    x_end = min(canvas_width, x_px + w)
                    
                    if y_start >= y_end or x_start >= x_end:
                        continue
                    
                    # Create grid of canvas coordinates
                    canvas_y, canvas_x = np.mgrid[y_start:y_end, x_start:x_end]
                    
                    # Calculate distance from image center
                    dist_from_center = np.sqrt(
                        (canvas_x - img_center_x)**2 + (canvas_y - img_center_y)**2
                    )
                    
                    # Update distance maps for Voronoi
                    current_dist = distance_map[y_start:y_end, x_start:x_end]
                    mask = dist_from_center < current_dist
                    
                    # Push current closest to second-closest
                    second_distance_map[y_start:y_end, x_start:x_end][mask] = current_dist[mask]
                    
                    # Update closest
                    distance_map[y_start:y_end, x_start:x_end][mask] = dist_from_center[mask]
                    image_index_map[y_start:y_end, x_start:x_end][mask] = idx
                    
                    # Also update second-closest
                    mask2 = (dist_from_center < second_distance_map[y_start:y_end, x_start:x_end]) & ~mask
                    second_distance_map[y_start:y_end, x_start:x_end][mask2] = dist_from_center[mask2]
                    
                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")
            
            print(f"[mosaic] Built Voronoi distance map with {len(images_data)} images")
            
            # Compose with sharp centers and small Gaussian blend at boundaries
            canvas_accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_accum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            blend_width = 25  # Small blend zone in pixels
            
            for img_data in images_data:
                idx = img_data['index']
                img_array = img_data['array']
                x_px = img_data['x_offset']
                y_px = img_data['y_offset']
                
                y_start = max(0, y_px)
                y_end = min(canvas_height, y_px + img_array.shape[0])
                x_start = max(0, x_px)
                x_end = min(canvas_width, x_px + img_array.shape[1])
                
                if y_start >= y_end or x_start >= x_end:
                    continue
                
                img_y_start = y_start - y_px
                img_y_end = img_y_start + (y_end - y_start)
                img_x_start = x_start - x_px
                img_x_end = img_x_start + (x_end - x_start)
                
                # Get regions
                img_region = img_array[img_y_start:img_y_end, img_x_start:img_x_end]
                index_region = image_index_map[y_start:y_end, x_start:x_end]
                dist_region = distance_map[y_start:y_end, x_start:x_end]
                second_dist_region = second_distance_map[y_start:y_end, x_start:x_end]
                
                # Only this image's Voronoi cells
                is_mine = index_region == idx
                
                # Calculate distance to boundary
                dist_diff = second_dist_region - dist_region
                near_boundary = dist_diff < blend_width
                
                # Blend weight: 1.0 far from boundary, smooth falloff near boundary
                blend_weight = np.ones_like(dist_diff, dtype=np.float32)
                blend_weight[near_boundary & is_mine] = np.clip(dist_diff[near_boundary & is_mine] / blend_width, 0, 1)
                blend_weight[~is_mine] = 0
                
                # Apply alpha
                alpha = img_region[:, :, 3].astype(np.float32) / 255.0
                final_weight = blend_weight * alpha
                
                # Accumulate weighted RGB (proper blending without whitening)
                weight_accum[y_start:y_end, x_start:x_end] += final_weight
                for c in range(3):
                    canvas_accum[y_start:y_end, x_start:x_end, c] += img_region[:, :, c].astype(np.float32) * final_weight
                
                print(f"[mosaic] Composited {img_data['name']} ({is_mine.sum()} pixels)")
            
            # Normalize to avoid white blending artifacts
            canvas_array = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
            has_data = weight_accum > 1e-6
            for c in range(3):
                canvas_array[has_data, c] = (canvas_accum[has_data, c] / weight_accum[has_data]).clip(0, 255).astype(np.uint8)
            canvas_array[has_data, 3] = 255  # Full opacity
            
            canvas = Image.fromarray(canvas_array, mode='RGBA')
            print(f"[mosaic] Voronoi mosaic complete with edge blending")
            
            # Store bounds for georeferencing
            geo_bounds = (min_lon, min_lat, max_lon, max_lat)
        else:
            # Fallback: simple horizontal stacking if no coordinates
            print("Warning: Insufficient coordinate data, using fallback horizontal stacking")
            first_img = Image.open(image_files[0])
            img_width, img_height = first_img.size
            img_mode = first_img.mode
            canvas_width = img_width * len(image_files)
            canvas_height = img_height
            canvas = Image.new(img_mode, (canvas_width, canvas_height), 0)
            for idx, img_path in enumerate(image_files):
                try:
                    img = Image.open(img_path)
                    canvas.paste(img, (idx * img_width, 0))
                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")
            geo_bounds = None
    else:
        # No camera metadata - fallback to horizontal stacking
        print("Warning: No camera metadata, using fallback horizontal stacking")
        first_img = Image.open(image_files[0])
        img_width, img_height = first_img.size
        img_mode = first_img.mode
        canvas_width = img_width * len(image_files)
        canvas_height = img_height
        canvas = Image.new(img_mode, (canvas_width, canvas_height), 0)
        for idx, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                canvas.paste(img, (idx * img_width, 0))
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
        geo_bounds = None
    
    # Save mosaic with georeferencing if camera metadata available
    out_mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    
    if camera_meta and geo_bounds and str(out_mosaic_path).endswith('.tif'):
        # Use calculated geographic bounds for georeferencing
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            min_lon, min_lat, max_lon, max_lat = geo_bounds
            
            # Create transform from bounds
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, canvas_width, canvas_height)
            
            # Convert canvas to numpy array
            canvas_array = np.array(canvas)
            
            # Determine number of bands and dtype
            if len(canvas_array.shape) == 2:
                # Grayscale
                count = 1
                canvas_array = canvas_array[np.newaxis, :, :]
            elif canvas_array.shape[2] == 3:
                # RGB
                count = 3
                canvas_array = np.transpose(canvas_array, (2, 0, 1))
            elif canvas_array.shape[2] == 4:
                # RGBA
                count = 4
                canvas_array = np.transpose(canvas_array, (2, 0, 1))
            else:
                count = canvas_array.shape[2]
                canvas_array = np.transpose(canvas_array, (2, 0, 1))
            
            # Write georeferenced GeoTIFF
            with rasterio.open(
                str(out_mosaic_path),
                'w',
                driver='GTiff',
                height=canvas_height,
                width=canvas_width,
                count=count,
                dtype=canvas_array.dtype,
                crs='EPSG:4326',  # WGS84
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(canvas_array)
            
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            print(f"Created georeferenced mosaic: {out_mosaic_path}")
            print(f"  CRS=EPSG:4326, center=({center_lat:.6f}, {center_lon:.6f})")
            print(f"  Bounds: [{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}]")
        except Exception as e:
            print(f"Warning: Failed to georeference mosaic: {e}. Saving without CRS.")
            import traceback
            traceback.print_exc()
            canvas.save(str(out_mosaic_path))
    else:
        # Save as regular image (fallback or non-TIF format)
        canvas.save(str(out_mosaic_path))
        print(f"Created mosaic: {out_mosaic_path} (size={canvas_width}x{canvas_height})")
    
    return out_mosaic_path


if __name__ == '__main__':
    main()
