"""Create a quick georeferenced mosaic from prepared individual images."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

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


