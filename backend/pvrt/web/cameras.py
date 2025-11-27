"""Agisoft camera references parser (table-only).

This module parses Agisoft-exported reference tables (e.g. `references.txt`,
CSV or TSV) produced by Agisoft PhotoScan/Metashape. It extracts per-image
positions and orientations (lat/lon/alt, yaw/rotation) and returns a mapping
keyed by a normalized basename (no extension, lowercased, trailing _v/_t
removed).

We intentionally *do not* attempt to parse XML `cameras.xml` here — the
workflow is table-first as requested by the UI and user preferences.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET
from math import atan2, degrees
from typing import Dict, Any, Optional
import csv


def _norm_key(name: str) -> str:
    if not name:
        return ""
    b = name.split('/')[-1].split('\\')[-1]
    noext = b.rsplit('.', 1)[0]
    noext = noext.strip().lower()
    # strip trailing _v or _t (common optical/thermal suffixes)
    noext = noext[:-2] if noext.endswith(('_v', '_t')) else noext
    return noext


def _quat_to_yaw_deg(qw: float, qx: float, qy: float, qz: float) -> float:
    # Convert quaternion (w, x, y, z) to yaw (heading) in degrees.
    try:
        yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return degrees(yaw)
    except Exception:
        return 0.0


def _try_parse_xml(root_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        root = ET.fromstring(root_bytes)
    except Exception:
        return out

    for cam in root.findall('.//camera'):
        try:
            file_el = cam.find('file')
            fname = file_el.text.strip() if file_el is not None and file_el.text else None

            # center/position
            cx = cy = cz = None
            c_el = cam.find('center') or cam.find('position') or cam.find('translation')
            if c_el is not None:
                x_el = c_el.find('x') or c_el.find('lon') or c_el.find('longitude')
                y_el = c_el.find('y') or c_el.find('lat') or c_el.find('latitude')
                z_el = c_el.find('z') or c_el.find('alt') or c_el.find('height')
                try:
                    cx = float(x_el.text) if x_el is not None and x_el.text else None
                    cy = float(y_el.text) if y_el is not None and y_el.text else None
                    cz = float(z_el.text) if z_el is not None and z_el.text else None
                except Exception:
                    cx = cy = cz = None

            # rotation: try quaternion first, else look for yaw/heading
            rot_deg: Optional[float] = None
            q_el = cam.find('quaternion') or cam.find('rotation') or cam.find('orientation')
            if q_el is not None:
                try:
                    qw = float(q_el.find('w').text) if q_el.find('w') is not None and q_el.find('w').text else None
                    qx = float(q_el.find('x').text) if q_el.find('x') is not None and q_el.find('x').text else None
                    qy = float(q_el.find('y').text) if q_el.find('y') is not None and q_el.find('y').text else None
                    qz = float(q_el.find('z').text) if q_el.find('z') is not None and q_el.find('z').text else None
                    if None not in (qw, qx, qy, qz):
                        rot_deg = _quat_to_yaw_deg(qw, qx, qy, qz)
                except Exception:
                    rot_deg = None

            # maybe yaw/heading present as scalar
            if rot_deg is None:
                for tag in ('yaw', 'heading', 'rotation', 'angle'):
                    r_el = cam.find(tag)
                    if r_el is not None and r_el.text:
                        try:
                            rot_deg = float(r_el.text)
                            break
                        except Exception:
                            continue

            # image size
            w_px = h_px = None
            w_el = cam.find('.//width')
            h_el = cam.find('.//height')
            try:
                if w_el is not None and w_el.text:
                    w_px = int(w_el.text)
                if h_el is not None and h_el.text:
                    h_px = int(h_el.text)
            except Exception:
                w_px = h_px = None

            key = _norm_key(fname or '')
            if not key:
                continue
            entry: Dict[str, Any] = {'file': fname, 'x': cx, 'y': cy, 'z': cz, 'rotation': float(rot_deg or 0.0)}
            if w_px:
                entry['w_px'] = int(w_px)
            if h_px:
                entry['h_px'] = int(h_px)

            # map x,y -> lon,lat if values look like degrees
            if cx is not None and cy is not None:
                if -180.0 <= cx <= 180.0 and -90.0 <= cy <= 90.0:
                    entry['lon'] = float(cx)
                    entry['lat'] = float(cy)
                else:
                    entry['x'] = float(cx)
                    entry['y'] = float(cy)
            if cz is not None:
                entry['alt'] = float(cz)

            out[key] = entry
        except Exception:
            continue

    return out


def _try_parse_table(bytes_buf: bytes) -> Dict[str, Dict[str, Any]]:
    """Attempt to parse a CSV/TSV/whitespace-delimited table exported by Agisoft.
    We try to detect delimiter and header names and extract common columns.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        text = bytes_buf.decode('utf-8', errors='replace')
    except Exception:
        return out

    # Quick sniff for delimiter: prefer comma, else tab, else whitespace
    sample = '\n'.join(text.splitlines()[:10])
    delim = ','
    if '\t' in sample and sample.count('\t') >= sample.count(','):
        delim = '\t'
    elif sample.count(',') == 0 and ' ' in sample:
        delim = None  # fallback to split()

    rows = []
    # Preprocess lines: remove empty lines and handle leading metadata/comment lines
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        return out

    # Find a sensible header line. Agisoft often emits a CoordinateSystem line
    # first (starting with '#') and the header as '#Label,...'. We want to
    # locate the header (which contains 'label' or 'yaw' etc.) and strip a
    # leading '#' if present so csv.DictReader sees correct fieldnames.
    header_idx = 0
    for i, ln in enumerate(raw_lines):
        s = ln.lstrip()
        low = s.lower()
        if low.startswith('#'):
            # candidate header if it contains typical column names
            if any(k in low for k in ('label', 'yaw', 'longitude', 'latitude', 'lat', 'lon')):
                # strip leading '#' characters and whitespace
                raw_lines[i] = s.lstrip('#').lstrip()
                header_idx = i
                break
            # otherwise skip this metadata/comment line
            continue
        else:
            # non-comment line: if it looks like a header (contains header tokens), use it
            if any(k in low for k in ('label', 'yaw', 'longitude', 'latitude', 'lat', 'lon')):
                header_idx = i
                break
            # otherwise keep searching
            continue

    lines_for_reader = raw_lines[header_idx:]

    if delim:
        reader = csv.DictReader(lines_for_reader, delimiter=delim)
        rows = list(reader)
    else:
        # whitespace-split: first line header, then map by position
        lines = [ln.strip() for ln in lines_for_reader if ln.strip()]
        if not lines:
            return out
        hdr = lines[0].split()
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) < 2:
                continue
            row = {hdr[i]: parts[i] if i < len(parts) else '' for i in range(len(hdr))}
            rows.append(row)

    # normalize header keys and map common column names
    for r in rows:
        try:
            # lowercase keys
            norm = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}

            # find filename field
            fn = None
            for k in ('filename', 'file', 'image', 'name'):
                if k in norm and norm[k]:
                    fn = norm[k]; break
            if not fn:
                # sometimes first column has no header
                vals = list(r.values())
                if vals:
                    fn = vals[0]
            if not fn:
                continue

            key = _norm_key(str(fn))
            if not key:
                continue
            # ignore spurious summary/header rows that some exports include
            if key.strip().startswith('#') or 'total' in key:
                continue

            # lon/lat candidates
            lon = None; lat = None; alt = None; rot = None
            for lon_k in ('lon', 'longitude', 'x', 'long'):
                if lon_k in norm and norm[lon_k]:
                    try: lon = float(norm[lon_k]); break
                    except Exception: pass
            for lat_k in ('lat', 'latitude', 'y'):
                if lat_k in norm and norm[lat_k]:
                    try: lat = float(norm[lat_k]); break
                    except Exception: pass
            for alt_k in ('alt', 'height', 'z', 'elev', 'elevation'):
                if alt_k in norm and norm[alt_k]:
                    try: alt = float(norm[alt_k]); break
                    except Exception: pass

            # rotation/yaw
            for rot_k in ('yaw', 'heading', 'rotation', 'angle'):
                if rot_k in norm and norm[rot_k]:
                    try:
                        rot = float(norm[rot_k]); break
                    except Exception:
                        rot = None

            # quaternion fallback: qw,qx,qy,qz
            if rot is None and all(k in norm for k in ('qw', 'qx', 'qy', 'qz')):
                try:
                    qw = float(norm.get('qw')); qx = float(norm.get('qx'))
                    qy = float(norm.get('qy')); qz = float(norm.get('qz'))
                    rot = _quat_to_yaw_deg(qw, qx, qy, qz)
                except Exception:
                    rot = None

            entry: Dict[str, Any] = {'file': fn}
            if lon is not None and lat is not None:
                entry['lon'] = float(lon); entry['lat'] = float(lat)
            if alt is not None:
                entry['alt'] = float(alt)
            if rot is not None:
                entry['rotation'] = float(rot)

            # optional pixel dims
            for w_k in ('width', 'w_px', 'w'):
                if w_k in norm and norm[w_k]:
                    try: entry['w_px'] = int(float(norm[w_k])); break
                    except Exception: pass
            for h_k in ('height', 'h_px', 'h'):
                if h_k in norm and norm[h_k]:
                    try: entry['h_px'] = int(float(norm[h_k])); break
                    except Exception: pass

            out[key] = entry
        except Exception:
            continue

    return out


def parse_agisoft_cameras(buf: bytes) -> Dict[str, Dict[str, Any]]:
    """Parse an Agisoft cameras export table (references.txt / CSV / TSV).

    This function only attempts table parsing. It returns a mapping keyed by
    normalized basename to camera metadata.
    """
    return _try_parse_table(buf)
