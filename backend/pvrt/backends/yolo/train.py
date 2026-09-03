"""Training helper for YOLO using ultralytics.YOLO

Provides a `run_train` function called by the YOLOBackend above. This is a
minimal wrapper around `ultralytics.YOLO(...).train(...)` tuned to the project's
conventions (writes outputs into out_dir/run_name).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import logging
import json
from statistics import median

from .runtime import resolve_yolo_device

# Third-party imports moved to module top per cleanup policy. ImportErrors
# will surface at import-time (fail-fast) which is the desired behavior.
from ultralytics import YOLO
import yaml
import shutil
import threading
import time
import csv

log = logging.getLogger("pvrt")


def _metric_name(yolo_seg: bool) -> str:
    return "mask" if yolo_seg else "bbox"


def _metric_key(yolo_seg: bool) -> str:
    return "val_mask_AP50" if yolo_seg else "val_bbox_AP50"


def _map50_column(fieldnames: List[str] | None, yolo_seg: bool) -> str | None:
    """Return the task-specific Ultralytics AP50 column."""
    suffix = "(m)" if yolo_seg else "(b)"
    columns = [column for column in (fieldnames or []) if column]
    for column in columns:
        normalized = column.strip().lower()
        if normalized == f"metrics/map50{suffix}":
            return column
    return None


def _map50_from_result(result: Any, yolo_seg: bool) -> float | None:
    """Extract task-specific AP50 from an Ultralytics metric result."""
    if result is None:
        return None

    metric_branch = getattr(result, "seg" if yolo_seg else "box", None)
    if metric_branch is not None and hasattr(metric_branch, "map50"):
        try:
            return float(metric_branch.map50)
        except (TypeError, ValueError):
            pass

    dictionaries = []
    for attribute in ("results_dict", "metrics", "results"):
        value = getattr(result, attribute, None)
        if isinstance(value, dict):
            dictionaries.append(value)
    if isinstance(result, dict):
        dictionaries.append(result)

    expected = f"metrics/map50({'m' if yolo_seg else 'b'})"
    for metrics in dictionaries:
        for key, value in metrics.items():
            if str(key).strip().lower() == expected:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _discover_num_classes_from_coco(train_dir: Path) -> int:
    # lightweight: look for _annotations.coco.json or any .json that looks like COCO
    import json
    cand = train_dir / "_annotations.coco.json"
    if cand.exists():
        data = json.loads(cand.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        return len(cats)
    for j in train_dir.glob("*.json"):
        data = json.loads(j.read_text(encoding="utf-8"))
        if isinstance(data, dict) and all(k in data for k in ("images", "annotations", "categories")):
            cats = data.get("categories", [])
            return len(cats)
    return 0


def _discover_class_names_from_coco(train_dir: Path) -> List[str]:
    cand = train_dir / "_annotations.coco.json"
    if cand.exists():
        data = json.loads(cand.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        # keep original order if ids are non-numeric; attempt numeric sort first
        try:
            cats = sorted(cats, key=lambda c: int(c.get("id", 0)))
        except (TypeError, ValueError):
            pass
        return [str(c.get("name", f"class_{i}")) for i, c in enumerate(cats)]
    return []


def _discover_class_names_from_yaml(dataset_yaml: Path | None) -> List[str]:
    if not dataset_yaml:
        return []
    try:
        data = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return []
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        try:
            return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
        except (TypeError, ValueError):
            return [str(value) for value in names.values()]
    return []


def run_train(train_dir: Path, val_dir: Path, out_dir: Path, use_thermal: bool, max_iter: int, base_lr: float, ims_per_batch: int, run_name: str = "yolo_run", yolo_family: str = "v8", yolo_seg: bool = False, yolo_size: str = "s", requested_channels: int = 3, dataset_yaml: Path | None = None) -> Dict[str, Any]:
    """Run a YOLO training job.

    Returns a dict with keys: best_weights, final_weights, model_name, num_classes, class_names
    """
    # all required imports are at module top (fail-fast on missing deps)

    # For compatibility with the project's Detectron-style outputs we
    # expose all training artifacts directly inside `out_dir` (no
    # nested `run_name` subfolder). Calling code sets `out_dir` to the
    # desired run folder. This mirrors Detectron's behavior where
    # model_best.pt / model_final.pt live directly in the run folder.
    run_dir = out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # All models use 3-channel training
    requested_channels = 3

    # Attempt to find classes
    class_names = _discover_class_names_from_yaml(dataset_yaml) or _discover_class_names_from_coco(train_dir)
    num_classes = len(class_names) or _discover_num_classes_from_coco(train_dir)

    # Choose model checkpoint from family + seg flag
    family = (yolo_family or "v8").lower()
    size = (yolo_size or "s").lower()
    size = size if size in {"n","s","m","l","x"} else "s"
    if family in {"v8", "8"}:
        model_weights = f"yolov8{size}-seg.pt" if yolo_seg else f"yolov8{size}.pt"
    elif family in {"v9", "9"}:
        model_weights = f"yolov9{size}-seg.pt" if yolo_seg else f"yolov9{size}.pt"
    elif family in {"v10", "10"}:
        model_weights = f"yolov10{size}-seg.pt" if yolo_seg else f"yolov10{size}.pt"
    else:
        model_weights = "yolov8s.pt"

    # NOTE: Do not duplicate/merge the training images into the run folder.
    # The project's datasets live in the provided `train_dir`/`val_dir` and we
    # should reference them directly.
    data_root = train_dir.parent.resolve()

    # For 3-channel training, generate train/val list files that point
    # to the existing RGB or thermal images.
    list_train = None
    list_val = None
    train_rel = None
    val_rel = None

    # If thermal is requested for a 3-channel run, prepare merged thermal-as-RGB
    if use_thermal:
        thermal_folder_names = {"thermal", "ir", "t", "temp"}

        def is_thermal_child(path: Path, root: Path) -> bool:
            return any(part.lower() in thermal_folder_names for part in path.relative_to(root).parts[:-1])

        def find_label_for_src(src: Path):
            sidecar = src.with_suffix('.txt')
            if sidecar.exists():
                return sidecar
            parts = list(src.parts)
            image_indexes = [index for index, part in enumerate(parts) if part.lower() == 'images']
            if image_indexes:
                parts[image_indexes[-1]] = 'labels'
                candidate = Path(*parts).with_suffix('.txt')
                if candidate.exists():
                    return candidate
            return None

        # Prepare symlinked dataset with thermal-as-RGB images
        def _load_pairs_json(images_dir: Path):
            pj = images_dir / "thermal" / "pairs.json"
            if pj.exists():
                try:
                    j = json.loads(pj.read_text(encoding="utf-8"))
                    return {str(k): str(v) for k, v in j.items()} if isinstance(j, dict) else {}
                except Exception:
                    return {}
            return {}

        def find_thermal_for_src(src: Path, images_dir: Path):
            # Respect pairs.json mapping if present
            pairs = _load_pairs_json(images_dir)
            if src.name in pairs:
                cand = Path(pairs[src.name])
                if cand.is_absolute():
                    if cand.exists():
                        return cand
                else:
                    pc = images_dir / cand
                    if pc.exists():
                        return pc

            # Look in common thermal subdirs
            THERMAL_DIR_NAMES = ("thermal", "ir", "t", "temp")
            THERMAL_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            for dname in THERMAL_DIR_NAMES:
                tdir = images_dir / dname
                if tdir.exists():
                    for ext in THERMAL_EXTS:
                        cand = tdir / f"{src.stem}{ext}"
                        if cand.exists():
                            return cand
                        cand2 = tdir / f"{src.stem}_thermal{ext}"
                        if cand2.exists():
                            return cand2

            # Fallback: check sibling files next to the RGB image
            for ext in THERMAL_EXTS:
                cand = images_dir / f"{src.stem}{ext}"
                if cand.exists():
                    return cand
                cand2 = images_dir / f"{src.stem}_thermal{ext}"
                if cand2.exists():
                    return cand2

            # Final fallback: adjacent file named <stem>_thermal.ext next to src
            for ext in THERMAL_EXTS:
                cand = src.with_name(f"{src.stem}_thermal{ext}")
                if cand.exists():
                    return cand
            return None

        def collect_thermal_as_rgb(folder: Path):
            out = []
            missing = 0
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    continue
                if is_thermal_child(p, folder):
                    continue
                t = find_thermal_for_src(p, folder)
                if t and t.exists():
                    out.append(str(t.resolve()))
                else:
                    missing += 1
                    logging.getLogger("pvrt").warning(f"YOLO preproc: no decoded thermal preview for {p.name}; skipping")
            return out, missing

        train_paths, train_missing = collect_thermal_as_rgb(train_dir)
        val_paths, val_missing = collect_thermal_as_rgb(val_dir)

        if train_paths or val_paths:
            # Create a per-run images folder with symlinks named as the
            # original RGB filenames but pointing to the decoded thermal previews
            def _prepare_symlink_tree(src_root: Path, dst_images: Path, dst_labels: Path):
                dst_images.mkdir(parents=True, exist_ok=True)
                dst_labels.mkdir(parents=True, exist_ok=True)
                for p in sorted(src_root.rglob("*")):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                        continue
                    if is_thermal_child(p, src_root):
                        continue
                    t = find_thermal_for_src(p, src_root)
                    if t and t.exists():
                        relative = p.relative_to(src_root)
                        dst_img = dst_images / relative
                        dst_img.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            if dst_img.exists():
                                dst_img.unlink()
                            dst_img.symlink_to(t.resolve())
                        except Exception:
                            # fallback to copying if symlink fails
                            try:
                                from shutil import copy2
                                copy2(str(t.resolve()), str(dst_img))
                            except Exception:
                                log.debug(f"YOLO preproc: failed to link or copy {t} -> {dst_img}")
                        # also link labels if found next to original RGB
                        label_src = find_label_for_src(p)
                        if label_src:
                            dst_lbl = (dst_labels / relative).with_suffix('.txt')
                            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                if dst_lbl.exists():
                                    dst_lbl.unlink()
                                dst_lbl.symlink_to(label_src.resolve())
                            except Exception:
                                try:
                                    from shutil import copy2
                                    copy2(str(label_src.resolve()), str(dst_lbl))
                                except Exception:
                                    log.debug(f"YOLO preproc: failed to link or copy label {label_src} -> {dst_lbl}")

            prepared_root = run_dir / "prepared_dataset"
            train_dst = prepared_root / "images" / "train"
            val_dst = prepared_root / "images" / "val"
            _prepare_symlink_tree(train_dir, train_dst, prepared_root / "labels" / "train")
            _prepare_symlink_tree(val_dir, val_dst, prepared_root / "labels" / "val")

            # Point ultralytics data.yaml to the per-run folders
            data_root = prepared_root
            train_rel = str(train_dst.relative_to(prepared_root))
            val_rel = str(val_dst.relative_to(prepared_root))
            log.info(f"YOLO preproc: prepared symlinked thermal-as-RGB dataset ({train_rel}, {val_rel}); skipped {train_missing}+{val_missing} missing previews")
        else:
            # No thermal previews found; fall back to RGB-only lists
            log.info("YOLO preproc: no decoded thermal previews found; falling back to RGB-only lists")

    # If not using thermal or thermal symlinks, create RGB-only lists
    if not train_rel:
        def collect_rgb_only(folder: Path):
            out = []
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    continue
                relative_folders = {part.lower() for part in p.relative_to(folder).parts[:-1]}
                if relative_folders.intersection({"thermal", "ir", "t", "temp"}):
                    continue
                out.append(str(p.resolve()))
            return out

        train_list = collect_rgb_only(train_dir)
        val_list = collect_rgb_only(val_dir)
        # write list files into run_dir
        list_train = run_dir / "train_images.txt"
        list_val = run_dir / "val_images.txt"
        list_train.write_text("\n".join(train_list), encoding="utf-8")
        list_val.write_text("\n".join(val_list), encoding="utf-8")
        data_root = run_dir
        log.info(f"YOLO preproc: wrote RGB-only train/val lists ({len(train_list)}/{len(val_list)})")

    # Create a temporary data.yaml compatible with ultralytics
    data_yaml = {
        "path": str(data_root),
        "train": (str(list_train) if list_train is not None else (train_rel if train_rel is not None else "train")),
        "val": (str(list_val) if list_val is not None else (val_rel if val_rel is not None else "valid")),
        "test": "test",
        "names": class_names or {i: f"class_{i}" for i in range(num_classes)}
    }
    yaml_path = run_dir / "data.yaml"
    # prefer YAML, fallback to JSON if yaml.safe_dump fails (rare)
    try:
        # yaml.safe_dump expects a file-like stream when provided a second
        # argument. Passing a Path caused a runtime error ('PosixPath' has no
        # attribute 'write'). Open the path explicitly and write via the file
        # handle. Fall back to JSON text if YAML dumping fails.
        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data_yaml, fh)
    except (yaml.YAMLError, TypeError, ValueError):
        yaml_path.write_text(json.dumps(data_yaml), encoding="utf-8")

    # 1-channel models are no longer supported; requested==1 should have been
    # coerced earlier by the backend. No checkpoint conversion is performed.

    # instantiate model and call train
    model = YOLO(model_weights)

    # 1-channel model patching removed: we don't support single-channel models.

    # Map parameters: for YOLO we treat the UI "Iterations" value as epochs.
    # This keeps behavior predictable: specifying 3000 will request 3000 epochs
    # (training may be slow for large epoch counts). To ensure a valid run we
    # coerce to at least 1 epoch when a positive value is provided.
    epochs = max(1, int(max_iter)) if max_iter and max_iter > 0 else 50

    device = resolve_yolo_device()

    # train
    # Log a compact header into the mini-log so users see major run parameters
    test_logger = logging.getLogger("pvrt.test")
    test_logger.info(
        f"UI:INFO:train: YOLO starting: data={yaml_path} epochs={epochs} "
        f"batch={ims_per_batch} lr0={base_lr} device={device} family={family} size={size}"
    )
    # Start a small background poller that tails ultralytics' results.csv and
    # emits per-epoch summaries into the mini-log (pvrt.test). This gives the
    # frontend near-real-time epoch updates without modifying ultralytics internals.
    stop_event = threading.Event()
    last_rows = {"n": 0}

    def _tail_results_csv():
        tlog = logging.getLogger("pvrt.test")
        results_csv = run_dir / "results.csv"
        while not stop_event.is_set():
            if results_csv.exists():
                with open(results_csv, newline='') as fh:
                    rdr = list(csv.DictReader(fh))
                n = len(rdr)
                if n > last_rows["n"]:
                    for i in range(last_rows["n"], n):
                        row = rdr[i]
                        epoch = row.get("epoch") or row.get("Epoch") or str(i + 1)
                        map_col = _map50_column(list(row.keys()), yolo_seg)
                        loss_col = None
                        for k in row.keys():
                            if "loss" in str(k).lower():
                                loss_col = k
                                break
                        parts = [f"epoch={epoch}"]
                        if map_col and row.get(map_col) is not None:
                            parts.append(f"{_metric_name(yolo_seg)}_map50={row.get(map_col)}")
                        if loss_col and row.get(loss_col) is not None:
                            parts.append(f"loss={row.get(loss_col)}")
                        tlog.info("UI:INFO:train: YOLO " + " ".join(parts))
                    last_rows["n"] = n
            stop_event.wait(2.0)

    poller = threading.Thread(target=_tail_results_csv, daemon=True)
    poller.start()

    try:
        results = model.train(
            data=str(yaml_path),
            imgsz=512,
            epochs=epochs,
            batch=ims_per_batch or 8,
            workers=4,
            device=device,
            optimizer="AdamW",
            lr0=base_lr or 0.001,
            # Use basic per-image augmentations (HSV, flip, scale, etc.) but disable
            # mosaic/mixup which compose multiple images and can fail when images
            # have different channel counts (e.g., RGB vs grayscale or RGBA).
            augment=True,
            mosaic=False,
            mixup=False,
            # Ask Ultralytics to write outputs directly into the provided
            # `out_dir` by passing the parent as `project` and the run folder
            # name as `name`. This ensures ultralytics will create
            # project/name == out_dir and not an extra nested folder like
            # out_dir/train.
            project=str(out_dir.parent),
            name=run_dir.name,
            exist_ok=True,
        )
    finally:
        stop_event.set()
        poller.join(timeout=3.0)

    # locate where ultralytics saved results (if results contains files use that,
    # otherwise assume it wrote directly into run_dir)
    save_dir = Path(results.files[0]).parent if hasattr(results, "files") and results.files else run_dir

    # No flattening here: we instruct ultralytics to write directly into
    # out_dir (project=out_dir.parent, name=run_dir.name) so outputs should
    # already live under run_dir. If ultralytics changes behavior in the
    # future we'll handle that as a separate case.

    # possible ultralytics placements: save_dir/weights/{best.pt,last.pt} or
    # save_dir/{best.pt,last.pt}. Search both and then copy/rename the files
    # into the DETECTRON-style names at the run root (model_best.pt, model_final.pt)
    best_candidates = [save_dir / "weights" / "best.pt", save_dir / "best.pt"]
    final_candidates = [save_dir / "weights" / "last.pt", save_dir / "last.pt"]

    found_best = next((p for p in best_candidates if p.exists()), None)
    found_final = next((p for p in final_candidates if p.exists()), None)

    # copy/rename into run_dir as Detectron-style filenames (no nested folders)
    if found_best:
        shutil.copy2(str(found_best), str(run_dir / "model_best.pt"))
        logging.getLogger("pvrt.test").info(f"UI:OK:train: YOLO saved best weights -> {run_dir / 'model_best.pt'}")
    if found_final:
        shutil.copy2(str(found_final), str(run_dir / "model_final.pt"))
        logging.getLogger("pvrt.test").info(f"UI:OK:train: YOLO saved final weights -> {run_dir / 'model_final.pt'}")

    # --- attempt to extract robust training statistics for model_meta.json ---
    def _first_result(obj):
        # ultralytics may return a Results object or a list-like container
        if isinstance(obj, (list, tuple)) and obj:
            return obj[0]
        return obj

    def _get_loss_history(r, save_dir: Path):
        # try common locations for loss history in the results object
        losses = []
        if hasattr(r, "history") and r.history:
            h = r.history
            if isinstance(h, dict):
                for v in h.values():
                    if isinstance(v, (list, tuple)) and v:
                        losses = [float(x) for x in v if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','',1).isdigit())]
                        if losses:
                            pass
            elif isinstance(h, (list, tuple)):
                for e in h:
                    if isinstance(e, dict):
                        for k, v in e.items():
                            if "loss" in str(k).lower():
                                try:
                                    losses.append(float(v))
                                except (TypeError, ValueError):
                                    continue
        # fallback: try to read ultralytics CSV/JSON artifacts if present
        for cand in [save_dir / "results.csv", save_dir / "metrics.csv", save_dir / "results.json", save_dir / "metrics.json"]:
            if cand.exists():
                if cand.suffix.lower() == ".json":
                    j = json.loads(cand.read_text(encoding="utf-8"))
                    if isinstance(j, dict):
                        for v in j.values():
                            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v[:5]):
                                losses = [float(x) for x in v]
                                break
                    elif isinstance(j, list):
                        for entry in j:
                            if isinstance(entry, dict):
                                for k, vv in entry.items():
                                    if "loss" in str(k).lower():
                                        try:
                                            losses.append(float(vv))
                                        except (TypeError, ValueError):
                                            continue
                else:
                    with open(cand, newline='') as fh:
                        rdr = csv.DictReader(fh)
                        for row in rdr:
                            for k, v in row.items():
                                if "loss" in str(k).lower():
                                    try:
                                        losses.append(float(v))
                                    except (TypeError, ValueError):
                                        continue
            if losses:
                break
        return losses

    def _median_last_n(seq, n=20):
        if not seq:
            return None
        sl = [float(x) for x in seq if x is not None]
        if not sl:
            return None
        tail = sl[-n:]
        try:
            return float(median(tail))
        except (TypeError, ValueError):
            # fallback simple median
            tail_sorted = sorted(tail)
            L = len(tail_sorted)
            if L == 0:
                return None
            mid = L // 2
            if L % 2 == 1:
                return float(tail_sorted[mid])
            return float((tail_sorted[mid - 1] + tail_sorted[mid]) / 2.0)

    # prepare defaults
    r0 = _first_result(results)
    save_dir = Path(results.files[0]).parent if hasattr(results, "files") and results.files else run_dir

    # Best / final mapping
    task_metric_key = _metric_key(yolo_seg)
    best_entry = {
        "iter": None,
        "val_AP50": None,
        task_metric_key: None,
        "total_loss_med20": None,
        "total_loss_raw": None,
        "path": str((run_dir / "model_best.pt").name) if (run_dir / "model_best.pt").exists() else (str(found_best.name) if found_best else ""),
    }
    final_entry = {
        "iter": None,
        "val_AP50": None,
        task_metric_key: None,
        "total_loss_med20": None,
        "total_loss_raw": None,
        "path": str((run_dir / "model_final.pt").name) if (run_dir / "model_final.pt").exists() else (str(found_final.name) if found_final else ""),
    }

    try:
        # map mAP from results object
        map50 = _map50_from_result(r0, yolo_seg)
        if map50 is not None:
            # assume this is the final/validation mAP; place in final and best if appropriate
            final_entry["val_AP50"] = map50
            final_entry[task_metric_key] = map50
            # optimistically set best to same if best.pt present
            if (run_dir / "model_best.pt").exists():
                best_entry["val_AP50"] = map50
                best_entry[task_metric_key] = map50

        # attempt to extract loss history and compute stats
        losses = _get_loss_history(r0, save_dir)
        if losses:
            final_entry["total_loss_med20"] = _median_last_n(losses, 20)
            final_entry["total_loss_raw"] = float(losses[-1]) if losses else None
            # copy to best if best exists
            if best_entry.get("path"):
                best_entry["total_loss_med20"] = final_entry["total_loss_med20"]
                best_entry["total_loss_raw"] = final_entry["total_loss_raw"]
    except Exception:
        log.exception("Failed to extract training statistics from ultralytics results")

    # If ultralytics wrote a results.csv we can get precise per-epoch mAP and epoch indices.
    csv_cand = save_dir / "results.csv"
    if csv_cand.exists():
        with open(csv_cand, newline='') as fh:
            rdr = csv.DictReader(fh)
            rows = list(rdr)
            map_col = _map50_column(rdr.fieldnames, yolo_seg)
            epoch_col = None
            for col in rdr.fieldnames or []:
                if col and col.lower() in {'epoch', 'ep', 'e'}:
                    epoch_col = col
                    break

            if rows:
                last = rows[-1]
                if epoch_col and epoch_col in last:
                    try:
                        final_entry['iter'] = int(float(last[epoch_col]))
                    except (TypeError, ValueError):
                        final_entry['iter'] = None
                if map_col and map_col in last:
                    try:
                        value = float(last[map_col])
                        final_entry['val_AP50'] = value
                        final_entry[task_metric_key] = value
                    except (TypeError, ValueError):
                        pass

                if map_col:
                    best_idx = None
                    best_val = None
                    for r in rows:
                        try:
                            v = float(r.get(map_col, 0) or 0)
                        except (TypeError, ValueError):
                            v = 0.0
                        if best_val is None or v > best_val:
                            best_val = v
                            try:
                                best_idx = int(float(r.get(epoch_col))) if epoch_col and r.get(epoch_col) is not None else None
                            except (TypeError, ValueError):
                                best_idx = None
                    if best_idx is not None:
                        best_entry['iter'] = best_idx
                    if best_val is not None:
                        best_entry['val_AP50'] = float(best_val)
                        best_entry[task_metric_key] = float(best_val)

    return {
        "best_weights": str(run_dir / "model_best.pt") if (run_dir / "model_best.pt").exists() else (str(found_best) if found_best else ""),
        "final_weights": str(run_dir / "model_final.pt") if (run_dir / "model_final.pt").exists() else (str(found_final) if found_final else ""),
        "model_name": model_weights.replace('.pt',''),
        "score_thresh_test": 0.25,
        "num_classes": int(num_classes),
        "class_names": class_names,
        "best_model": best_entry,
        "final_model": final_entry,
    }
