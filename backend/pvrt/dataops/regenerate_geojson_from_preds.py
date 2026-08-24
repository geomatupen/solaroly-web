#!/usr/bin/env python3
import json
from pathlib import Path
import math
from PIL import Image
import shutil

import sys


def _normalize_heading_deg(val):
    try:
        if val is None:
            return None
        heading = float(val)
    except (TypeError, ValueError):
        return None

    # normalize to (-180, 180]
    while heading <= -180.0:
        heading += 360.0
    while heading > 180.0:
        heading -= 360.0
    return heading


def _camera_heading_to_overlay_rotation(camera_heading_deg):
    heading = _normalize_heading_deg(camera_heading_deg)
    return heading if heading is not None else 0.0


def _coerce_float(val):
    try:
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _camera_meta_session_meta(camera_meta):
    if not isinstance(camera_meta, dict):
        return {}
    meta = camera_meta.get('__meta__')
    return meta if isinstance(meta, dict) else {}


def _camera_heading_from_entry(cam_entry, session_meta):
    heading = None
    if cam_entry and isinstance(cam_entry, dict) and cam_entry.get('rotation') is not None:
        heading = _normalize_heading_deg(cam_entry.get('rotation'))
    if heading is None:
        default_rot = _coerce_float(session_meta.get('default_rotation_deg'))
        if default_rot is not None:
            heading = _normalize_heading_deg(default_rot)
    offset = _coerce_float(session_meta.get('rotation_offset_deg'))
    if heading is not None and offset is not None:
        heading = _normalize_heading_deg(heading + offset)
    return heading

# allow passing session directory path as first CLI arg, source images dir as second arg
# First arg must be the full path to the session directory (project structure)
if len(sys.argv) < 2:
    print("[ERROR] Usage: regenerate_geojson_from_preds.py <session_dir_path> <source_images_dir> [--use-thermal]")
    print("[ERROR] session_dir_path: Full path to test/outputs/<session-id>/")
    sys.exit(1)

SESSION_DIR_PATH = sys.argv[1]
SRC_IMAGES_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else None  # optional source images directory
USE_THERMAL = '--use-thermal' in sys.argv  # flag to use thermal images if available

# Use the full session directory path (new project structure)
BASE = Path(SESSION_DIR_PATH)
IMAGES_DIR = BASE / 'rotated_images'
# If rotated_images doesn't exist yet but camera_meta is available, we'll
# attempt to materialize rotated copies from the original `images/` so the
# frontend can use consistent rotated thumbnails. If rotated_images truly
# exists, prefer it; otherwise fall back to `images/` (we may populate
# rotated_images further down).
if not IMAGES_DIR.exists():
    IMAGES_DIR = BASE / 'images'
PRED_DIR = BASE / 'preds'
CAM_META = BASE / 'camera_meta.json'
MANIFEST = BASE / 'manifest.json'
OUT_ANOM = BASE / 'predictions.geojson'
OUT_IMAGES = BASE / 'images.geojson'

# Default GSD - will be overridden by per-image values from camera_meta
DEFAULT_METERS_PER_PIXEL = 0.05

# load camera_meta
camera_meta = {}
if CAM_META.exists():
    try:
        camera_meta = json.loads(CAM_META.read_text(encoding='utf-8'))
    except Exception as e:
        print('failed to load camera_meta', e)
session_meta = _camera_meta_session_meta(camera_meta)

# load manifest for lat/lon hints
manifest = {}
if MANIFEST.exists():
    try:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    except Exception as e:
        print('failed to load manifest', e)

# sizes
sizes = {}
for p in sorted(IMAGES_DIR.glob('*')):
    if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
        try:
            with Image.open(p) as im:
                sizes[p.name] = im.size
        except Exception:
            pass

# After inspecting sizes, if camera_meta exists but rotated_images is missing
# or empty, materialize rotated copies from the originals (`images/`). This
# ensures `rotated_images/` is available even if thumbs/ or rotated thumbs
# were not previously created.
rotated_dir = BASE / 'rotated_images'
try:
    # If camera metadata is present, (re)generate rotated images from source
    # so we ensure correct rotation is applied (overwrite any existing files).
    # Rotation now always happens before inference, so post-inference calls
    # should NOT re-rotate. Only populate when a source images dir is provided.
    need_populate = CAM_META.exists() and SRC_IMAGES_DIR is not None
    print(f"[rotation] CAM_META path: {CAM_META}")
    print(f"[rotation] CAM_META.exists(): {need_populate}")

    if need_populate:
        print(f"[rotation] Starting rotation: camera_meta has {len(camera_meta)} keys")
        cam = camera_meta
        cam_session_meta = session_meta
        # Use source images directory passed from backend, or fall back to session dirs
        if SRC_IMAGES_DIR and SRC_IMAGES_DIR.exists():
            src_images = SRC_IMAGES_DIR
        else:
            # prefer originals if present, otherwise prefer full-size overlays,
            # finally fall back to session thumbs
            src_images = BASE / 'images'
            if not src_images.exists():
                src_images = BASE / 'overlays'
            if not src_images.exists():
                src_images = BASE / 'thumbs'
        print(f"[rotation] src_images: {src_images} (exists={src_images.exists()})")
        if src_images.exists():
            rotated_dir.mkdir(parents=True, exist_ok=True)
            print(f"[rotation] Created rotated_dir: {rotated_dir}")
            src_files = sorted(src_images.glob('*'))
            src_count = len([f for f in src_files if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
            print(f"[rotation] Found {src_count} image files in {src_images.name}")
            
            # Check if source dataset has thermal previews - only use if --use-thermal flag is set
            src_thermal_dir = SRC_IMAGES_DIR / 'thermal' if SRC_IMAGES_DIR else None
            src_thermal_pairs = None
            use_thermal_for_rotation = False
            if USE_THERMAL and src_thermal_dir and (src_thermal_dir / 'pairs.json').exists():
                try:
                    src_thermal_pairs = json.loads((src_thermal_dir / 'pairs.json').read_text(encoding='utf-8'))
                    if src_thermal_pairs:
                        use_thermal_for_rotation = True
                        print(f"[rotation] Using thermal images for rotation ({len(src_thermal_pairs)} pairs)")
                except Exception as e:
                    print(f"[rotation] Error reading thermal pairs.json: {e}")
            
            if not use_thermal_for_rotation:
                print(f"[rotation] Using RGB images for rotation (USE_THERMAL={USE_THERMAL})")
            
            for p in sorted(src_images.glob('*')):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                fname = p.name
                rot = 0.0
                # Flexible camera_meta lookup: try exact name, stem, and fallbacks
                cam_entry = None
                if fname in cam and isinstance(cam.get(fname), dict):
                    cam_entry = cam.get(fname)
                elif p.name in cam and isinstance(cam.get(p.name), dict):
                    cam_entry = cam.get(p.name)
                elif p.stem in cam and isinstance(cam.get(p.stem), dict):
                    cam_entry = cam.get(p.stem)
                else:
                    # try matching by cam key stems or substring
                    for k, v in cam.items():
                        try:
                            if not isinstance(v, dict):
                                continue
                            if str(k).startswith('__'):
                                continue
                            if Path(k).stem == p.stem or k.startswith(p.stem) or k.endswith(p.stem) or p.stem.startswith(Path(k).stem):
                                cam_entry = v
                                break
                        except Exception:
                            continue
                heading = _camera_heading_from_entry(cam_entry, cam_session_meta)
                
                # Smart heading source selection:
                # If gimbal and aircraft headings differ significantly (>90°), they may be 
                # in different reference frames. Use aircraft heading in that case.
                # Otherwise use gimbal heading (standard practice).
                if heading is not None and cam_entry and isinstance(cam_entry, dict):
                    gimbal = _coerce_float(cam_entry.get('rotation_gimbal'))
                    aircraft = _coerce_float(cam_entry.get('rotation_aircraft'))
                    if gimbal is not None and aircraft is not None:
                        diff = abs(gimbal - aircraft)
                        # Large difference (>90°) indicates different reference frame
                        if diff > 90:
                            heading = _normalize_heading_deg(aircraft)
                
                rot = _camera_heading_to_overlay_rotation(heading)
                # Rotate images to north-up orientation
                # Formula: angle = -heading
                angle = -float(rot or 0.0)

                # If thermal mode, load thermal preview and convert to 3-channel RGB
                if use_thermal_for_rotation and fname in src_thermal_pairs:
                    thermal_fname = src_thermal_pairs.get(fname)
                    # thermal_fname includes "thermal/" prefix (e.g., "thermal/DJI_xxx_thermal.tif")
                    # We'll use the _preview.png variant (normalized, matches training)
                    thermal_path = SRC_IMAGES_DIR / thermal_fname
                    thermal_preview_path = thermal_path.with_name(thermal_path.stem + '_preview.png')
                    
                    if thermal_preview_path.exists():
                        try:
                            # Load thermal preview (already normalized grayscale)
                            from PIL import Image as PILImage
                            import numpy as np
                            with PILImage.open(thermal_preview_path) as tim:
                                # Preview is already grayscale, convert to 3-channel RGB
                                if tim.mode != 'L':
                                    tim = tim.convert('L')
                                thermal_arr = np.array(tim, dtype=np.uint8)
                            
                            # Replicate grayscale across 3 channels for RGB
                            thermal_rgb = np.stack([thermal_arr, thermal_arr, thermal_arr], axis=2)
                            p_img = PILImage.fromarray(thermal_rgb, mode='RGB')
                            print(f"[rotation] Using thermal preview for {fname}")
                        except Exception as e:
                            print(f"[rotation] ERROR loading thermal preview for {fname}: {e}, falling back to RGB")
                            p_img = Image.open(p)
                    else:
                        print(f"[rotation] Thermal preview not found for {fname}, falling back to RGB")
                        p_img = Image.open(p)
                else:
                    p_img = Image.open(p)

                try:
                    with p_img as im:
                        if abs(angle) < 1e-6:
                            # No rotation needed, but still convert to RGBA with transparency for consistency
                            if im.mode != "RGBA":
                                im = im.convert("RGBA")
                            out_name = f"{Path(fname).stem}.png"
                            im.save(rotated_dir / out_name)
                            print(f"[rotation] Saved {fname} → {out_name} (angle ~0)")
                        else:
                            # Convert to RGBA first for transparency support
                            if im.mode != "RGBA":
                                im = im.convert("RGBA")
                            
                            rint = int(round(angle % 360)) % 360
                            if rint in (90, 180, 270):
                                # For 90° increments, use transpose (preserves transparency)
                                if rint == 90:
                                    rim = im.transpose(Image.Transpose.ROTATE_90)
                                elif rint == 180:
                                    rim = im.transpose(Image.Transpose.ROTATE_180)
                                else:
                                    rim = im.transpose(Image.Transpose.ROTATE_270)
                            else:
                                # For arbitrary angles, use rotate with transparent fill
                                try:
                                    rim = im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
                                except TypeError:
                                    # Older PIL versions
                                    rim = im.rotate(angle, Image.BICUBIC, expand=True)
                                    # Create transparent background manually
                                    bg = Image.new("RGBA", rim.size, (0, 0, 0, 0))
                                    bg.paste(rim, mask=rim.split()[3] if rim.mode == "RGBA" else None)
                                    rim = bg
                            
                            out_name = f"{Path(fname).stem}.png"
                            rim.save(rotated_dir / out_name)
                            print(f"[rotation] Rotated {fname} → {out_name} (angle={angle:.1f}°)")
                except Exception as e:
                    print(f"[rotation] ERROR processing {fname}: {e}")
                    try:
                        shutil.copy2(p, rotated_dir / fname)
                        print(f"[rotation] Fallback: copied {fname}")
                    except Exception as e2:
                        print(f"[rotation] Fallback failed: {e2}")

            # Prefer using rotated_dir for sizes lookup now that we've created it
            IMAGES_DIR = rotated_dir
            # recompute sizes from rotated_dir
            sizes = {}
            for p in sorted(IMAGES_DIR.glob('*')):
                if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    try:
                        with Image.open(p) as im:
                            sizes[p.name] = im.size
                    except Exception:
                        pass
            print(f"[rotation] COMPLETE: {len(sizes)} images in {IMAGES_DIR.name}, IMAGES_DIR now={IMAGES_DIR}")
        else:
            print(f"[rotation] SKIPPED: src_images does not exist")
    else:
        print(f"[rotation] SKIPPED: CAM_META does not exist")
except Exception as e:
    print('failed to create rotated_images from originals:', e)

# image latlon mapping from manifest entries if present
latlon_map = {}
for k, v in manifest.items():
    if isinstance(v, dict) and 'lat' in v and 'lon' in v:
        latlon_map[k] = (float(v['lat']), float(v['lon']))

# fallback: try camera_meta entries with lat/lon
for k, v in camera_meta.items():
    if not isinstance(v, dict):
        continue
    if str(k).startswith('__'):
        continue
    if 'lat' in v and 'lon' in v:
        latlon_map.setdefault(k, (float(v['lat']), float(v['lon'])))

print('images_dir:', IMAGES_DIR, 'pred_json_dir:', PRED_DIR)
print('loaded sizes:', len(sizes), 'camera_meta entries:', len(camera_meta), 'manifest latlon:', len(latlon_map))

# build images.geojson
imgs_fc = {"type": "FeatureCollection", "features": []}
for fname, (w_px, h_px) in sizes.items():
    latlon = latlon_map.get(fname) or latlon_map.get(Path(fname).stem)
    if not latlon:
        # try flexible matching by stem or substring
        for k in latlon_map.keys():
            try:
                if Path(k).stem == Path(fname).stem or k.startswith(fname) or k.endswith(fname):
                    latlon = latlon_map[k]
                    break
            except Exception:
                continue
    if not latlon:
        continue
    lon_c = latlon[1]
    lat_c = latlon[0]
    # degrees per meter
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_c))
    deg_per_m_lon = 1.0 / meters_per_deg_lon
    deg_per_m_lat = 1.0 / meters_per_deg_lat

    # compute rotation using flexible matching (exact name, stem, or substring)
    rot = 0.0
    cam_entry = None
    if fname in camera_meta and isinstance(camera_meta[fname], dict):
        cam_entry = camera_meta[fname]
    elif Path(fname).name in camera_meta and isinstance(camera_meta[Path(fname).name], dict):
        cam_entry = camera_meta[Path(fname).name]
    elif Path(fname).stem in camera_meta and isinstance(camera_meta[Path(fname).stem], dict):
        cam_entry = camera_meta[Path(fname).stem]
    else:
        for k, v in camera_meta.items():
            try:
                if not isinstance(v, dict):
                    continue
                if str(k).startswith('__'):
                    continue
                if Path(k).stem == Path(fname).stem or k.startswith(Path(fname).stem) or k.endswith(Path(fname).stem) or Path(fname).stem.startswith(Path(k).stem):
                    cam_entry = v
                    break
            except Exception:
                continue
    heading_deg = _camera_heading_from_entry(cam_entry, session_meta)
    # Footprint for rotated images should be north-up (no extra rotation)
    rotation_deg = 0.0  # predictions are on already-rotated (north-up) images
    rot_for_geo = rotation_deg
    rot_overlay = rotation_deg
    rot = rot_overlay

    # Get per-image GSD from camera_meta, fallback to default
    meters_per_pixel = DEFAULT_METERS_PER_PIXEL
    if cam_entry and isinstance(cam_entry, dict):
        img_gsd = cam_entry.get('meters_per_pixel')
        if img_gsd is not None:
            try:
                meters_per_pixel = float(img_gsd)
            except (TypeError, ValueError):
                pass

    # corners using pixel-based math (match preds reprojection)
    cx = float(w_px) / 2.0
    cy = float(h_px) / 2.0
    pix_corners = [(0.0, 0.0), (float(w_px), 0.0), (float(w_px), float(h_px)), (0.0, float(h_px))]
    a = math.radians(float(rot_for_geo))
    ca = math.cos(a)
    sa = math.sin(a)
    out_corners = []
    for (px, py) in pix_corners:
        dx_m = (px - cx) * meters_per_pixel
        dy_m = (py - cy) * meters_per_pixel
        rx = dx_m * ca - dy_m * sa
        ry = dx_m * sa + dy_m * ca
        lon_p = lon_c + (rx * deg_per_m_lon)
        lat_p = lat_c - (ry * deg_per_m_lat)
        out_corners.append([lon_p, lat_p])

    props = {
        'image': fname,
        'w': int(w_px), 'h': int(h_px),
        'width_m': float(w_px) * meters_per_pixel,
        'height_m': float(h_px) * meters_per_pixel,
        'meters_per_pixel': float(meters_per_pixel),
        'rotation': float(rot_for_geo),
        'corners': out_corners,
        'src': fname,
    }
    if heading_deg is not None:
        props['rotation_heading'] = float(heading_deg)
    props['rotation_overlay'] = float(rot_overlay)
    feat = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [lon_c, lat_c]},
        'properties': props,
    }
    imgs_fc['features'].append(feat)

OUT_IMAGES.write_text(json.dumps(imgs_fc, indent=2), encoding='utf-8')
print('Wrote images.geojson features:', len(imgs_fc['features']))

# build predictions.geojson from preds
anom_fc = {"type": "FeatureCollection", "features": []}
if PRED_DIR.exists():
    for jpath in sorted(Path(PRED_DIR).glob('*.json')):
        try:
            jd = json.loads(jpath.read_text(encoding='utf-8'))
        except Exception:
            continue
        boxes = jd.get('boxes', []) or []
        scores = jd.get('scores', []) or []
        classes = jd.get('classes', []) or []
        srcfile = jd.get('file') or (jpath.stem + '.png')
        latlon = latlon_map.get(srcfile) or latlon_map.get(Path(srcfile).name) or latlon_map.get(Path(srcfile).stem)
        # fallback: try to match manifest keys by stem or substring
        if not latlon:
            for k in latlon_map.keys():
                try:
                    if Path(k).stem == Path(srcfile).stem or k.startswith(srcfile) or k.endswith(srcfile):
                        latlon = latlon_map[k]
                        break
                except Exception:
                    continue

        wh = sizes.get(srcfile) or sizes.get(Path(srcfile).name) or sizes.get(Path(srcfile).stem)
        if not wh:
            for k in sizes.keys():
                try:
                    if Path(k).stem == Path(srcfile).stem or k.startswith(srcfile) or k.endswith(srcfile):
                        wh = sizes[k]
                        break
                except Exception:
                    continue
        if not latlon or not wh:
            continue
        lat, lon = latlon
        w_px, h_px = wh
        cx = float(w_px) / 2.0
        cy = float(h_px) / 2.0
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))

        rotation_deg = 0.0
        heading_deg = None
        # Flexible camera_meta lookup: try exact name, name, stem, then fuzzy
        cam_entry = None
        try:
            if srcfile in camera_meta and isinstance(camera_meta[srcfile], dict):
                cam_entry = camera_meta[srcfile]
            elif Path(srcfile).name in camera_meta and isinstance(camera_meta[Path(srcfile).name], dict):
                cam_entry = camera_meta[Path(srcfile).name]
            elif Path(srcfile).stem in camera_meta and isinstance(camera_meta[Path(srcfile).stem], dict):
                cam_entry = camera_meta[Path(srcfile).stem]
            else:
                for k, v in camera_meta.items():
                    try:
                        if not isinstance(v, dict):
                            continue
                        if str(k).startswith('__'):
                            continue
                        if Path(k).stem == Path(srcfile).stem or k.startswith(Path(srcfile).stem) or k.endswith(Path(srcfile).stem) or Path(srcfile).stem.startswith(Path(k).stem):
                            cam_entry = v
                            break
                    except Exception:
                        continue
        except Exception:
            cam_entry = None
        heading_deg = _camera_heading_from_entry(cam_entry, session_meta)
        # Predictions are on rotated (north-up) images -> no additional rotation when mapping boxes
        rotation_deg = 0.0

        # Get per-image GSD from camera_meta, fallback to default
        meters_per_pixel = DEFAULT_METERS_PER_PIXEL
        if cam_entry and isinstance(cam_entry, dict):
            img_gsd = cam_entry.get('meters_per_pixel')
            if img_gsd is not None:
                try:
                    meters_per_pixel = float(img_gsd)
                except (TypeError, ValueError):
                    pass

        a = math.radians(rotation_deg)
        ca = math.cos(a)
        sa = math.sin(a)

        for i, b in enumerate(boxes):
            try:
                x0, y0, x1, y1 = map(float, b)
            except Exception:
                continue
            sc = float(scores[i]) if i < len(scores) else 0.0
            dx0_m = (x0 - cx) * meters_per_pixel
            dx1_m = (x1 - cx) * meters_per_pixel
            dy0_m = (y0 - cy) * meters_per_pixel
            dy1_m = (y1 - cy) * meters_per_pixel
            # build rotated polygon from the four box corners so anomalies align
            # with rotated image footprints (same convention as backend)
            try:
                corners_px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                poly = []
                for (px, py) in corners_px:
                    dx_m = (px - cx) * meters_per_pixel
                    dy_m = (py - cy) * meters_per_pixel
                    if rotation_deg and abs(rotation_deg) > 1e-6:
                        rx = dx_m * ca - dy_m * sa
                        ry = dx_m * sa + dy_m * ca
                    else:
                        rx, ry = dx_m, dy_m
                    lon_p = lon + (rx * deg_per_m_lon)
                    lat_p = lat - (ry * deg_per_m_lat)
                    poly.append([lon_p, lat_p])
                # close polygon
                if poly and poly[0] != poly[-1]:
                    poly.append(poly[0])
            except Exception:
                # fallback to axis-aligned bbox
                if rotation_deg and abs(rotation_deg) > 1e-6:
                    r0x = dx0_m * ca - dy0_m * sa
                    r0y = dx0_m * sa + dy0_m * ca
                    r1x = dx1_m * ca - dy1_m * sa
                    r1y = dx1_m * sa + dy1_m * ca
                else:
                    r0x, r0y, r1x, r1y = dx0_m, dy0_m, dx1_m, dy1_m
                lon0 = lon + (r0x * deg_per_m_lon)
                lon1 = lon + (r1x * deg_per_m_lon)
                lat0 = lat - (r0y * deg_per_m_lat)
                lat1 = lat - (r1y * deg_per_m_lat)
                poly = [[lon0, lat0], [lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]]

            anom_fc['features'].append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [poly]}, 'properties': {'score': sc, 'image': srcfile}})

    OUT_ANOM.write_text(json.dumps(anom_fc, indent=2), encoding='utf-8')
    print('Wrote predictions.geojson features:', len(anom_fc['features']))
else:
    print('No preds directory found at', PRED_DIR)
