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

log = logging.getLogger("pvrt")


def _discover_num_classes_from_coco(train_dir: Path) -> int:
    # lightweight: look for _annotations.coco.json or any .json that looks like COCO
    import json
    cand = train_dir / "_annotations.coco.json"
    if cand.exists():
        data = json.loads(cand.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        return len(cats)
    for j in train_dir.glob("*.json"):
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
            if isinstance(data, dict) and all(k in data for k in ("images", "annotations", "categories")):
                cats = data.get("categories", [])
                return len(cats)
        except Exception:
            continue
    return 0


def _discover_class_names_from_coco(train_dir: Path) -> List[str]:
    import json
    cand = train_dir / "_annotations.coco.json"
    if cand.exists():
        data = json.loads(cand.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        try:
            cats = sorted(cats, key=lambda c: int(c.get("id", 0)))
        except Exception:
            pass
        return [str(c.get("name", f"class_{i}")) for i, c in enumerate(cats)]
    return []


def run_train(train_dir: Path, val_dir: Path, out_dir: Path, use_thermal: bool, max_iter: int, base_lr: float, ims_per_batch: int, run_name: str = "yolo_run", yolo_family: str = "v8", yolo_seg: bool = False, yolo_size: str = "s", requested_channels: int = 3) -> Dict[str, Any]:
    """Run a YOLO training job.

    Returns a dict with keys: best_weights, final_weights, model_name, num_classes, class_names
    """
    # Delay import to keep module import cheap when not training
    from ultralytics import YOLO
    import yaml
    from pathlib import Path

    # For compatibility with the project's Detectron-style outputs we
    # expose all training artifacts directly inside `out_dir` (no
    # nested `run_name` subfolder). Calling code sets `out_dir` to the
    # desired run folder. This mirrors Detectron's behavior where
    # model_best.pt / model_final.pt live directly in the run folder.
    run_dir = out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # If the caller requested thermal channels (1 or 4) but `use_thermal`
    # is False (no thermal available / not enabled), coerce to 3-channel
    # behavior. This prevents creating merged/symlinked train folders that
    # are empty and ensures the dataset loader only sees 3-channel RGB images
    # (avoids mixed [3,H,W] and [1,H,W] tensors in a batch).
    if not use_thermal and requested_channels != 3:
        log.warning(
            f"YOLO preproc: requested_channels={requested_channels} but use_thermal=False; forcing requested_channels=3 to avoid mixed-channel batches"
        )
        requested_channels = 3

    # Attempt to find classes
    class_names = _discover_class_names_from_coco(train_dir)
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
    # should reference them directly. Creating merged copies would produce
    # duplicate image files and is unnecessary. If dataset normalization or
    # filtering is required it should be performed in-place or via symlinks.
    data_root = train_dir.parent.resolve()

    # If the requested model expects 3 channels, prefer NOT to create any
    # merged folder: instead generate small train/val list files that point
    # to the existing RGB images under the provided `train_dir`/`val_dir`.
    # This avoids duplicating image bytes while guaranteeing that only
    # RGB-capable files are presented to the trainer (preventing mixed
    # 1/3-channel batches).
    list_train = None
    list_val = None
    if requested_channels == 3:
        def collect_rgb_only(folder: Path):
            out = []
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    continue
                try:
                    with Image.open(p) as im:
                        # prefer bands detection which handles many TIFF modes
                        bands = im.getbands()
                        if not bands or len(bands) == 1:
                            # single-channel -> skip
                            continue
                        out.append(str(p.resolve()))
                except Exception:
                    continue
            return out

        try:
            from PIL import Image
            train_list = collect_rgb_only(train_dir)
            val_list = collect_rgb_only(val_dir)
            # write list files into run_dir (small text files)
            list_train = run_dir / "train_images.txt"
            list_val = run_dir / "val_images.txt"
            list_train.write_text("\n".join(train_list), encoding="utf-8")
            list_val.write_text("\n".join(val_list), encoding="utf-8")
            data_root = run_dir
            log.info(f"YOLO preproc: wrote RGB-only train/val lists ({len(train_list)}/{len(val_list)})")
        except Exception:
            log.exception("Failed to produce RGB-only lists; falling back to original data root")
            data_root = train_dir.parent.resolve()
    else:
        # For non-3-channel runs (thermal-based training) we still need to
        # prepare per-run inputs. Use the symlinked merged tree to avoid
        # duplicating bytes while presenting a consistent dataset to the
        # trainer.
        try:
            from .preproc import merge_rgb_with_thermal
            merged_root = run_dir / "merged"
            merged_root.mkdir(parents=True, exist_ok=True)
            tcount = merge_rgb_with_thermal(train_dir, merged_root / "train", requested_channels=requested_channels, use_thermal=use_thermal, symlink=True)
            vcount = merge_rgb_with_thermal(val_dir, merged_root / "valid", requested_channels=requested_channels, use_thermal=use_thermal, symlink=True)
            if (tcount or vcount):
                data_root = merged_root
                log.info(f"YOLO preproc (symlink): prepared dataset: train={tcount}, valid={vcount}; using {merged_root}")
            else:
                log.debug(f"YOLO preproc (symlink) produced no outputs; using original data root {data_root}")
        except Exception:
            log.exception("YOLO preproc (symlink) failed; falling back to original data root")
            data_root = train_dir.parent.resolve()

    # Create a temporary data.yaml compatible with ultralytics
    data_yaml = {
        "path": str(data_root),
        "train": str(list_train) if list_train is not None else "train",
        "val": str(list_val) if list_val is not None else "valid",
        "test": "test",
        "names": class_names or {i: f"class_{i}" for i in range(num_classes)}
    }
    yaml_path = run_dir / "data.yaml"
    try:
        yaml.safe_dump(data_yaml, yaml_path)
    except Exception:
        yaml_path.write_text(json.dumps(data_yaml), encoding="utf-8")

    # If requested_channels == 1, attempt to convert the base checkpoint into
    # a 1-channel-first-conv variant that we can pass to YOLO so training
    # produces a true 1-channel model. Conversion is best-effort and falls
    # back to the original checkpoint on failure.
    if requested_channels == 1:
        try:
            from .weights import convert_yolo_checkpoint_to_1ch
            src = Path(model_weights)
            converted = run_dir / "weights" / "converted_1ch.pt"
            converted.parent.mkdir(parents=True, exist_ok=True)
            new_path = convert_yolo_checkpoint_to_1ch(src, converted)
            # If conversion produced a different path, use it. Otherwise,
            # if the source wasn't a local file (e.g., hub name), try the
            # fallback: instantiate YOLO once, extract state_dict, convert
            # that state and write dst.
            if Path(new_path) == src and not src.exists():
                try:
                    temp_model = YOLO(model_weights)
                    # try to locate underlying module's state_dict
                    underlying = None
                    for attr in ("model", "model.model", "module", "net"):
                        try:
                            obj = temp_model
                            for p in attr.split("."):
                                obj = getattr(obj, p)
                            import torch
                            if isinstance(obj, torch.nn.Module):
                                underlying = obj
                                break
                        except Exception:
                            continue
                    if underlying is not None:
                        try:
                            sd = underlying.state_dict()
                            # attempt conversion via the helper by saving
                            # a minimal checkpoint containing 'model'
                            ck = {"model": sd}
                            torch.save(ck, str(converted))
                            new_path = convert_yolo_checkpoint_to_1ch(converted, converted)
                            if Path(new_path) != src:
                                model_weights = str(new_path)
                        except Exception:
                            log.exception("YOLO weights: fallback conversion from instantiated model failed")
                except Exception:
                    log.exception("YOLO weights: fallback instantiation for conversion failed")
            else:
                model_weights = str(new_path)
        except Exception:
            log.exception("YOLO weights: conversion attempt failed; proceeding with original weights")

    # instantiate model and call train
    model = YOLO(model_weights)

    # If requested_channels == 1 we may want to produce a model that truly
    # accepts a single-channel input. Ultralytics' YOLO object wraps a
    # torch.nn.Module; try to locate that module and patch its first Conv2d
    # to accept 1 input channel by averaging pretrained RGB weights. This is
    # best-effort and will continue silently on failure.
    if requested_channels == 1:
        try:
            import torch.nn as _nn

            # locate underlying module
            underlying = None
            for attr in ("model", "model.model", "module", "net"):
                try:
                    parts = attr.split(".")
                    obj = model
                    for p in parts:
                        obj = getattr(obj, p)
                    if isinstance(obj, _nn.Module):
                        underlying = obj
                        break
                except Exception:
                    continue

            def _patch_first_conv_to_1ch_general(mod: _nn.Module) -> bool:
                import torch
                for name, module_ in mod.named_modules():
                    if isinstance(module_, _nn.Conv2d) and module_.in_channels == 3:
                        new_conv = _nn.Conv2d(
                            in_channels=1,
                            out_channels=module_.out_channels,
                            kernel_size=module_.kernel_size,
                            stride=module_.stride,
                            padding=module_.padding,
                            dilation=module_.dilation,
                            groups=module_.groups,
                            bias=(module_.bias is not None),
                            padding_mode=module_.padding_mode,
                        )
                        with torch.no_grad():
                            w = module_.weight  # [out,3,k,k]
                            avg = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                            new_conv.weight[:, :1, :, :] = avg
                            if module_.bias is not None:
                                new_conv.bias.copy_(module_.bias)

                        parent = mod
                        parts = name.split(".")
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], new_conv)
                        return True
                return False

            if underlying is not None:
                try:
                    patched = _patch_first_conv_to_1ch_general(underlying)
                    if patched:
                        log.info("UI:OK:train: YOLO underlying model patched to 1-channel first conv")
                except Exception:
                    log.exception("UI:WARN:train: failed to patch YOLO model to 1-channel; proceeding")
        except Exception:
            # any failure here is non-fatal; training will still run on 3-channel inputs
            try:
                log.exception("UI:WARN:train: exception while attempting YOLO 1-channel patch")
            except Exception:
                pass

    # Map parameters: for YOLO we treat the UI "Iterations" value as epochs
    # (user expects the number they enter to be honored as epochs). This keeps
    # behavior predictable: providing 3000 → 3000 epochs is possible (but may
    # be slowed); to keep a minimum we coerce to at least 1 epoch when >0.
    epochs = max(1, int(max_iter)) if max_iter and max_iter > 0 else 50

    # train
    # Log a compact header into the mini-log so users see major run parameters
    try:
        test_logger = logging.getLogger("pvrt.test")
        test_logger.info(f"UI:INFO:train: YOLO starting: data={yaml_path} epochs={epochs} batch={ims_per_batch} lr0={base_lr} device=0 family={family} size={size}")
    except Exception:
        pass

    # Start a small background poller that tails ultralytics' results.csv and
    # emits per-epoch summaries into the mini-log (pvrt.test). This gives the
    # frontend near-real-time epoch updates without modifying ultralytics internals.
    import threading, time, csv

    stop_event = threading.Event()
    last_rows = {"n": 0}

    def _tail_results_csv():
        tlog = logging.getLogger("pvrt.test")
        results_csv = run_dir / "results.csv"
        while not stop_event.is_set():
            try:
                if results_csv.exists():
                    with open(results_csv, newline='') as fh:
                        rdr = list(csv.DictReader(fh))
                    n = len(rdr)
                    if n > last_rows["n"]:
                        for i in range(last_rows["n"], n):
                            row = rdr[i]
                            # attempt to extract epoch, map50, loss
                            epoch = row.get("epoch") or row.get("Epoch") or str(i + 1)
                            # common mAP column names
                            map_col = None
                            for k in row.keys():
                                lk = str(k).lower()
                                if "map" in lk and ("0.5" in lk or "50" in lk or "map50" in lk):
                                    map_col = k
                                    break
                            loss_col = None
                            for k in row.keys():
                                if "loss" in str(k).lower():
                                    loss_col = k
                                    break
                            parts = [f"epoch={epoch}"]
                            if map_col and row.get(map_col) is not None:
                                parts.append(f"map50={row.get(map_col)}")
                            if loss_col and row.get(loss_col) is not None:
                                parts.append(f"loss={row.get(loss_col)}")
                            tlog.info("UI:INFO:train: YOLO " + " ".join(parts))
                        last_rows["n"] = n
            except Exception:
                logging.getLogger("pvrt").exception("Failed while tailing YOLO results.csv")
            stop_event.wait(2.0)

    poller = threading.Thread(target=_tail_results_csv, daemon=True)
    poller.start()

    results = model.train(
        data=str(yaml_path),
        imgsz=512,
        epochs=epochs,
        batch=ims_per_batch or 8,
        workers=4,
        device=0,
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
    try:
        import shutil
        if found_best:
            shutil.copy2(str(found_best), str(run_dir / "model_best.pt"))
            logging.getLogger("pvrt.test").info(f"UI:OK:train: YOLO saved best weights -> {run_dir / 'model_best.pt'}")
        if found_final:
            shutil.copy2(str(found_final), str(run_dir / "model_final.pt"))
            logging.getLogger("pvrt.test").info(f"UI:OK:train: YOLO saved final weights -> {run_dir / 'model_final.pt'}")
    except Exception:
        log.exception("Failed to copy/rename ultralytics weights into run root")

    # --- attempt to extract robust training statistics for model_meta.json ---
    def _first_result(obj):
        # ultralytics may return a Results object or a list-like container
        try:
            if isinstance(obj, (list, tuple)) and obj:
                return obj[0]
        except Exception:
            pass
        return obj

    def _get_map50_from_result(r):
        try:
            # try common attributes/dicts
            cand_metrics = None
            if hasattr(r, "metrics") and isinstance(r.metrics, dict):
                cand_metrics = r.metrics
            elif isinstance(r, dict):
                cand_metrics = r
            elif hasattr(r, "results") and isinstance(r.results, dict):
                cand_metrics = r.results

            if isinstance(cand_metrics, dict):
                # look for any key that describes mAP@0.5
                for k, v in cand_metrics.items():
                    lk = str(k).lower()
                    if "map" in lk and ("0.5" in lk or "50" in lk):
                        try:
                            return float(v)
                        except Exception:
                            pass
                # nested keys like 'box' -> 'map50'
                for v in cand_metrics.values():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            lk2 = str(k2).lower()
                            if "map" in lk2 and ("0.5" in lk2 or "50" in lk2):
                                try:
                                    return float(v2)
                                except Exception:
                                    pass
            # common attribute names
            for attr in ("map50", "mAP_0.5", "mAP50", "map_0.5"):
                if hasattr(r, attr):
                    try:
                        return float(getattr(r, attr))
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    def _get_loss_history(r, save_dir: Path):
        # try common locations for loss history in the results object
        losses = []
        try:
            if hasattr(r, "history") and r.history:
                # history may be a list of dicts or dict of lists
                h = r.history
                if isinstance(h, dict):
                    # try to find the first list-like value of numeric items
                    for v in h.values():
                        if isinstance(v, (list, tuple)) and v:
                            try:
                                losses = [float(x) for x in v if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','',1).isdigit())]
                                if losses:
                                    break
                            except Exception:
                                continue
                elif isinstance(h, (list, tuple)):
                    for e in h:
                        if isinstance(e, dict):
                            for k, v in e.items():
                                if "loss" in str(k).lower():
                                    try:
                                        losses.append(float(v))
                                    except Exception:
                                        pass
        except Exception:
            pass

        # fallback: try to read ultralytics CSV/JSON artifacts if present
        try:
            for cand in [save_dir / "results.csv", save_dir / "metrics.csv", save_dir / "results.json", save_dir / "metrics.json"]:
                if cand.exists():
                    try:
                        if cand.suffix.lower() == ".json":
                            j = json.loads(cand.read_text(encoding="utf-8"))
                            # j may be a dict with 'metrics' or a list of per-epoch dicts
                            if isinstance(j, dict):
                                # try to find loss series
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
                                                except Exception:
                                                    pass
                        else:
                            # csv
                            import csv
                            with open(cand, newline='') as fh:
                                rdr = csv.DictReader(fh)
                                for row in rdr:
                                    for k, v in row.items():
                                        if "loss" in str(k).lower():
                                            try:
                                                losses.append(float(v))
                                            except Exception:
                                                pass
                    except Exception:
                        continue
                if losses:
                    break
        except Exception:
            pass

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
        except Exception:
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
    best_entry = {
        "iter": None,
        "val_bbox_AP50": None,
        "total_loss_med20": None,
        "total_loss_raw": None,
        "path": str((run_dir / "model_best.pt").name) if (run_dir / "model_best.pt").exists() else (str(found_best.name) if found_best else ""),
    }
    final_entry = {
        "iter": None,
        "val_bbox_AP50": None,
        "total_loss_med20": None,
        "total_loss_raw": None,
        "path": str((run_dir / "model_final.pt").name) if (run_dir / "model_final.pt").exists() else (str(found_final.name) if found_final else ""),
    }

    try:
        # map mAP from results object
        map50 = _get_map50_from_result(r0)
        if map50 is not None:
            # assume this is the final/validation mAP; place in final and best if appropriate
            final_entry["val_bbox_AP50"] = map50
            # optimistically set best to same if best.pt present
            if (run_dir / "model_best.pt").exists():
                best_entry["val_bbox_AP50"] = map50

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
    try:
        import csv
        csv_cand = save_dir / "results.csv"
        if csv_cand.exists():
            with open(csv_cand, newline='') as fh:
                rdr = csv.DictReader(fh)
                rows = list(rdr)
                # find a column name that looks like mAP50 (case-insensitive)
                map_col = None
                for col in rdr.fieldnames or []:
                    if col and 'map' in col.lower() and ('0.5' in col.lower() or '50' in col.lower()):
                        map_col = col
                        break
                # also find epoch column
                epoch_col = None
                for col in rdr.fieldnames or []:
                    if col and col.lower() in {'epoch', 'ep', 'e'}:
                        epoch_col = col
                        break

                if rows:
                    # final = last row
                    last = rows[-1]
                    if epoch_col and epoch_col in last:
                        try:
                            final_entry['iter'] = int(float(last[epoch_col]))
                        except Exception:
                            final_entry['iter'] = None
                    # read final mAP50 if available
                    if map_col and map_col in last:
                        try:
                            final_entry['val_bbox_AP50'] = float(last[map_col])
                        except Exception:
                            final_entry['val_bbox_AP50'] = final_entry.get('val_bbox_AP50')

                    # determine best epoch by max map_col
                    if map_col:
                        best_idx = None
                        best_val = None
                        for r in rows:
                            try:
                                v = float(r.get(map_col, 0) or 0)
                            except Exception:
                                v = 0.0
                            if best_val is None or v > best_val:
                                best_val = v
                                try:
                                    best_idx = int(float(r.get(epoch_col))) if epoch_col and r.get(epoch_col) is not None else None
                                except Exception:
                                    best_idx = None
                        if best_idx is not None:
                            best_entry['iter'] = best_idx
                        if best_val is not None:
                            best_entry['val_bbox_AP50'] = float(best_val)
    except Exception:
        log.exception('Failed to parse results.csv for precise epoch/mAP values')
    except Exception:
        log.exception("Failed to extract training statistics from ultralytics results")

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
