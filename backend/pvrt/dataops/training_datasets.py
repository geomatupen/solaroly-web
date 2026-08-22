"""Training-dataset upload, validation, and registry helpers."""
from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
UPLOAD_EXTS = IMAGE_EXTS | {".json", ".coco", ".yaml", ".yml", ".txt"}
COCO_NAMES = (
    "_annotations.coco.json", "_annotations.coco", "annotations.json",
    "train.json", "valid.json", "val.json",
)
MAX_UPLOAD_FILES = int(os.getenv("PVRT_TRAIN_UPLOAD_MAX_FILES", "100000"))
MAX_UPLOAD_BYTES = int(os.getenv("PVRT_TRAIN_UPLOAD_MAX_BYTES", str(20 * 1024**3)))


class DatasetUploadError(ValueError):
    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_relative_path(value: str) -> Path:
    normalized = str(value or "").replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetUploadError(f"Unsafe upload path: {value!r}")
    if any(":" in part for part in path.parts):
        raise DatasetUploadError(f"Unsafe upload path: {value!r}")
    return Path(*path.parts)


def _allowed_file(path: Path) -> bool:
    return path.suffix.lower() in UPLOAD_EXTS


def _ignored_file(path: Path) -> bool:
    return (
        path.name in {".DS_Store", "Thumbs.db"}
        or path.suffix.lower() in {".cache", ".pyc"}
        or "__MACOSX" in path.parts
    )


def _write_stream(stream: Any, destination: Path, byte_state: list[int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            byte_state[0] += len(chunk)
            if byte_state[0] > MAX_UPLOAD_BYTES:
                raise DatasetUploadError("Training upload exceeds the configured size limit.")
            target.write(chunk)


def _extract_zip(stream: Any, staging: Path) -> None:
    try:
        archive = zipfile.ZipFile(stream)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetUploadError(f"Invalid ZIP archive: {exc}") from exc
    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise DatasetUploadError("The ZIP archive is empty.")
        if len(members) > MAX_UPLOAD_FILES:
            raise DatasetUploadError("Training upload contains too many files.")
        total = sum(max(0, int(item.file_size)) for item in members)
        if total > MAX_UPLOAD_BYTES:
            raise DatasetUploadError("Training upload exceeds the configured size limit.")
        destinations: set[str] = set()
        for item in members:
            relative = _safe_relative_path(item.filename)
            mode = (item.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise DatasetUploadError(f"ZIP symlinks are not allowed: {item.filename}")
            if _ignored_file(relative):
                continue
            if not _allowed_file(relative):
                raise DatasetUploadError(f"Unsupported training file: {item.filename}")
            destination_key = relative.as_posix().casefold()
            if destination_key in destinations:
                raise DatasetUploadError(f"Duplicate training file path: {item.filename}")
            destinations.add(destination_key)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def _flatten_single_wrapper(staging: Path) -> None:
    children = [p for p in staging.iterdir() if p.name not in {"__MACOSX", ".DS_Store"}]
    if len(children) != 1 or not children[0].is_dir():
        return
    wrapper = children[0]
    for child in list(wrapper.iterdir()):
        child.replace(staging / child.name)
    wrapper.rmdir()


def stage_upload(files: Iterable[Any], staging: Path) -> None:
    uploads = list(files)
    if not uploads:
        raise DatasetUploadError("Choose a ZIP file or a dataset folder.")
    if len(uploads) > MAX_UPLOAD_FILES:
        raise DatasetUploadError("Training upload contains too many files.")
    if len(uploads) == 1 and str(uploads[0].filename or "").lower().endswith(".zip"):
        _extract_zip(uploads[0].file, staging)
    else:
        if any(str(item.filename or "").lower().endswith(".zip") for item in uploads):
            raise DatasetUploadError("Upload one ZIP archive, or select a folder—not both.")
        byte_state = [0]
        destinations: set[str] = set()
        for upload in uploads:
            relative = _safe_relative_path(upload.filename or "")
            if _ignored_file(relative):
                continue
            if not _allowed_file(relative):
                raise DatasetUploadError(f"Unsupported training file: {upload.filename}")
            destination_key = relative.as_posix().casefold()
            if destination_key in destinations:
                raise DatasetUploadError(f"Duplicate training file path: {upload.filename}")
            destinations.add(destination_key)
            _write_stream(upload.file, staging / relative, byte_state)
    _flatten_single_wrapper(staging)


def _images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _folder_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _unreadable_images(paths: Iterable[Path]) -> int:
    count = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            count += 1
    return count


def _split(root: Path, *names: str) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _coco_json(split: Path) -> Path | None:
    for name in COCO_NAMES:
        candidate = split / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(split.glob("*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict) and all(key in data for key in ("images", "annotations", "categories")):
                return candidate
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def validate_coco(root: Path) -> dict[str, Any]:
    train = _split(root, "train")
    valid = _split(root, "valid", "val", "validation")
    detected = bool((train and _coco_json(train)) or (valid and _coco_json(valid)))
    errors: list[str] = []
    warnings: list[str] = []
    split_info: dict[str, Any] = {}
    category_sets: list[list[str]] = []
    if not train:
        errors.append("Missing train/ directory.")
    if not valid:
        errors.append("Missing valid/ or val/ directory.")
    for label, split in (("train", train), ("valid", valid)):
        if not split:
            continue
        annotation = _coco_json(split)
        if not annotation:
            errors.append(f"Missing COCO annotation JSON in {label}/.")
            continue
        try:
            data = json.loads(annotation.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid COCO JSON in {label}/: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"COCO annotation in {label}/ must be a JSON object.")
            continue
        images = data.get("images") if isinstance(data.get("images"), list) else []
        annotations = data.get("annotations") if isinstance(data.get("annotations"), list) else []
        categories = data.get("categories") if isinstance(data.get("categories"), list) else []
        if not images:
            errors.append(f"COCO {label} split has no image records.")
        if not categories:
            errors.append(f"COCO {label} split has no categories.")
        image_ids = {item.get("id") for item in images if isinstance(item, dict)}
        category_ids = {item.get("id") for item in categories if isinstance(item, dict)}
        if len(image_ids) != len(images):
            errors.append(f"COCO {label} contains missing or duplicate image IDs.")
        if len(category_ids) != len(categories):
            errors.append(f"COCO {label} contains missing or duplicate category IDs.")
        missing_files = []
        existing_files: list[Path] = []
        for item in images:
            if not isinstance(item, dict) or not item.get("file_name"):
                errors.append(f"COCO {label} contains an image without file_name.")
                continue
            try:
                relative = _safe_relative_path(str(item["file_name"]))
            except DatasetUploadError:
                missing_files.append(str(item["file_name"]))
                continue
            if not (split / relative).is_file():
                missing_files.append(str(item["file_name"]))
            else:
                existing_files.append(split / relative)
        invalid_annotations = 0
        for item in annotations:
            if not isinstance(item, dict):
                invalid_annotations += 1
                continue
            bbox = item.get("bbox")
            if item.get("image_id") not in image_ids or item.get("category_id") not in category_ids:
                invalid_annotations += 1
            elif bbox is not None and (
                not isinstance(bbox, list) or len(bbox) != 4 or
                any(not isinstance(value, (int, float)) for value in bbox) or bbox[2] < 0 or bbox[3] < 0
            ):
                invalid_annotations += 1
        if missing_files:
            errors.append(f"COCO {label} references {len(missing_files)} missing image files.")
        unreadable = _unreadable_images(existing_files)
        if unreadable:
            errors.append(f"COCO {label} references {unreadable} unreadable image files.")
        if invalid_annotations:
            errors.append(f"COCO {label} contains {invalid_annotations} invalid annotations.")
        names = [str(item.get("name", "")) for item in categories if isinstance(item, dict)]
        category_sets.append(names)
        split_info[label] = {
            "images": len(images), "annotations": len(annotations),
            "annotation_file": str(annotation.relative_to(root)),
            "missing_images": len(missing_files), "unreadable_images": unreadable,
        }
    if len(category_sets) == 2 and category_sets[0] != category_sets[1]:
        errors.append("COCO train and validation categories do not match.")
    classes = category_sets[0] if category_sets else []
    return {
        "detected": detected, "valid": detected and not errors,
        "errors": errors, "warnings": warnings, "splits": split_info,
        "classes": classes,
    }


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _yaml_image_dir(root: Path, yaml_dir: Path, base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip() or value.endswith(".txt"):
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    return candidate if _inside(root, candidate) and candidate.is_dir() else None


def _label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    indexes = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if indexes:
        parts[indexes[-1]] = "labels"
        return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def validate_yolo(root: Path) -> dict[str, Any]:
    yamls = [p for p in (root / "data.yaml", root / "data.yml") if p.is_file()]
    detected = bool(yamls)
    errors: list[str] = []
    warnings: list[str] = []
    split_info: dict[str, Any] = {}
    if not yamls:
        return {"detected": False, "valid": False, "errors": ["Missing data.yaml."], "warnings": [], "splits": {}, "classes": []}
    yaml_path = yamls[0]
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return {"detected": True, "valid": False, "errors": [f"Invalid data.yaml: {exc}"], "warnings": [], "splits": {}, "classes": []}
    if not isinstance(data, dict):
        return {"detected": True, "valid": False, "errors": ["data.yaml must contain a mapping."], "warnings": [], "splits": {}, "classes": []}
    names_value = data.get("names")
    if isinstance(names_value, list):
        classes = [str(value) for value in names_value]
    elif isinstance(names_value, dict):
        try:
            classes = [str(value) for _, value in sorted(names_value.items(), key=lambda item: int(item[0]))]
        except (TypeError, ValueError):
            classes = [str(value) for value in names_value.values()]
    else:
        classes = []
    if not classes:
        errors.append("data.yaml has no class names.")
    configured_root = data.get("path")
    base = yaml_path.parent
    if isinstance(configured_root, str) and configured_root.strip():
        path_value = Path(configured_root)
        base = (yaml_path.parent / path_value).resolve() if not path_value.is_absolute() else path_value.resolve()
        if not _inside(root, base):
            errors.append("data.yaml path points outside the uploaded dataset.")
            base = yaml_path.parent
    for label, keys in (("train", ("train",)), ("valid", ("val", "valid"))):
        value = next((data.get(key) for key in keys if data.get(key) is not None), None)
        image_dir = _yaml_image_dir(root, yaml_path.parent, base, value)
        if not image_dir:
            errors.append(f"YOLO {label} image directory is missing or outside the dataset.")
            continue
        label_dir = _label_dir(image_dir)
        if not label_dir.is_dir() or not _inside(root, label_dir):
            errors.append(f"YOLO {label} labels directory is missing: {label_dir.name}")
            continue
        images = _images(image_dir)
        if not images:
            errors.append(f"YOLO {label} split has no images.")
        unreadable = _unreadable_images(images)
        if unreadable:
            errors.append(f"YOLO {label} contains {unreadable} unreadable image files.")
        missing_labels = 0
        invalid_rows = 0
        label_files = 0
        for image in images:
            relative = image.relative_to(image_dir).with_suffix(".txt")
            label_path = label_dir / relative
            if not label_path.is_file():
                missing_labels += 1
                continue
            label_files += 1
            try:
                rows = label_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                invalid_rows += 1
                continue
            for row in rows:
                values = row.split()
                try:
                    class_id = int(values[0])
                    coords = [float(value) for value in values[1:]]
                    valid_shape = len(coords) == 4 or (len(coords) >= 6 and len(coords) % 2 == 0)
                    if not valid_shape or not 0 <= class_id < len(classes) or any(not 0 <= value <= 1 for value in coords):
                        invalid_rows += 1
                except (IndexError, TypeError, ValueError):
                    invalid_rows += 1
        if not label_files:
            errors.append(f"YOLO {label} split has no label files.")
        if invalid_rows:
            errors.append(f"YOLO {label} split contains {invalid_rows} invalid label rows.")
        if missing_labels:
            warnings.append(f"YOLO {label} has {missing_labels} images without label files (treated as background).")
        split_info[label] = {
            "images": len(images), "label_files": label_files,
            "missing_labels": missing_labels,
            "unreadable_images": unreadable,
            "image_dir": str(image_dir.relative_to(root)),
            "label_dir": str(label_dir.relative_to(root)),
        }
    return {
        "detected": detected, "valid": detected and not errors,
        "errors": errors, "warnings": warnings, "splits": split_info,
        "classes": classes, "data_yaml": str(yaml_path.relative_to(root)),
    }


def validate_yolo_sidecars(root: Path, coco: dict[str, Any]) -> dict[str, Any]:
    """Check YOLO label files stored beside images in a COCO split layout."""
    if not coco.get("valid"):
        return {"detected": False, "valid": False, "errors": [], "warnings": [], "splits": {}}
    class_count = len(coco.get("classes") or [])
    errors: list[str] = []
    warnings: list[str] = []
    split_info: dict[str, Any] = {}
    detected_labels = 0
    for label, split in (("train", _split(root, "train")), ("valid", _split(root, "valid", "val", "validation"))):
        if not split:
            continue
        images = _images(split)
        label_files = 0
        missing_labels = 0
        invalid_rows = 0
        invalid_class_ids = 0
        invalid_shapes = 0
        out_of_range_rows = 0
        unreadable_rows = 0
        for image in images:
            label_path = image.with_suffix(".txt")
            if not label_path.is_file():
                missing_labels += 1
                continue
            label_files += 1
            detected_labels += 1
            try:
                rows = label_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                invalid_rows += 1
                continue
            for row in rows:
                values = row.split()
                try:
                    class_id = int(values[0])
                    coords = [float(value) for value in values[1:]]
                    valid_shape = len(coords) == 4 or (len(coords) >= 6 and len(coords) % 2 == 0)
                    row_invalid = False
                    if not valid_shape:
                        invalid_shapes += 1
                        row_invalid = True
                    if not 0 <= class_id < class_count:
                        invalid_class_ids += 1
                        row_invalid = True
                    if any(not 0 <= value <= 1 for value in coords):
                        out_of_range_rows += 1
                        row_invalid = True
                    if row_invalid:
                        invalid_rows += 1
                except (IndexError, TypeError, ValueError):
                    unreadable_rows += 1
                    invalid_rows += 1
        if invalid_rows:
            reasons = []
            if invalid_class_ids:
                reasons.append(f"{invalid_class_ids} invalid class IDs")
            if invalid_shapes:
                reasons.append(f"{invalid_shapes} malformed coordinate lists")
            if out_of_range_rows:
                reasons.append(f"{out_of_range_rows} rows outside the 0–1 coordinate range")
            if unreadable_rows:
                reasons.append(f"{unreadable_rows} unreadable rows")
            errors.append(
                f"YOLO-compatible {label} split contains {invalid_rows} invalid label rows"
                f" ({', '.join(reasons)})."
            )
        if not label_files:
            errors.append(f"YOLO-compatible {label} split has no label files.")
        if missing_labels:
            warnings.append(
                f"YOLO-compatible {label} split has {missing_labels} images without label files "
                "(valid background images; add labels only if those images contain objects)."
            )
        split_info[label] = {
            "images": len(images), "label_files": label_files,
            "missing_labels": missing_labels, "image_dir": str(split.relative_to(root)),
            "label_dir": str(split.relative_to(root)),
            "invalid_label_rows": invalid_rows,
            "invalid_class_ids": invalid_class_ids,
            "invalid_coordinate_lists": invalid_shapes,
            "out_of_range_rows": out_of_range_rows,
        }
    if not detected_labels:
        errors.append("No YOLO .txt label files were found beside the COCO images.")
    return {
        "detected": detected_labels > 0,
        "valid": detected_labels > 0 and not errors,
        "errors": errors, "warnings": warnings, "splits": split_info,
        "classes": coco.get("classes", []),
    }


def validate_dataset(root: Path, requested_format: str = "auto") -> dict[str, Any]:
    requested = str(requested_format or "auto").strip().lower()
    if requested not in {"unified", "auto", "coco", "yolo"}:
        raise DatasetUploadError("Dataset format must be unified, auto, coco, or yolo.")
    coco = validate_coco(root)
    yolo = validate_yolo(root)
    yolo_sidecar = validate_yolo_sidecars(root, coco)
    validators = {"coco": coco, "yolo": yolo}
    compatible = [name for name, result in validators.items() if result["valid"]]
    errors: list[str] = []
    warnings: list[str] = []
    if requested == "unified":
        if not coco["valid"]:
            errors.extend(f"COCO: {message}" for message in coco["errors"])
        elif not yolo_sidecar["valid"]:
            errors.extend(f"YOLO sidecars: {message}" for message in yolo_sidecar["errors"])
        if coco["valid"]:
            warnings.extend(coco["warnings"])
        if yolo_sidecar["valid"]:
            warnings.extend(yolo_sidecar["warnings"])
    elif requested == "auto":
        if not compatible:
            for name, result in validators.items():
                if result["detected"]:
                    errors.extend(f"{name.upper()}: {message}" for message in result["errors"])
            if not errors:
                errors.append("No supported COCO or YOLO training layout was detected.")
        for name, result in validators.items():
            # The project layout stores valid YOLO labels beside the COCO images.
            # A stray or incompatible native data.yaml must not imply that YOLO
            # training is unavailable when those sidecars already validate.
            if name == "yolo" and yolo_sidecar["valid"]:
                continue
            if result["detected"] and not result["valid"] and compatible:
                warnings.extend(f"{name.upper()} compatibility: {message}" for message in result["errors"])
        if coco["valid"] and yolo_sidecar["detected"] and not yolo_sidecar["valid"]:
            warnings.extend(f"YOLO compatibility: {message}" for message in yolo_sidecar["errors"])
    else:
        selected = validators[requested]
        if not selected["valid"]:
            errors.extend(selected["errors"])
    if requested != "unified":
        for name in compatible:
            warnings.extend(validators[name]["warnings"])
    primary = "coco" if requested == "unified" else requested if requested != "auto" else (compatible[0] if compatible else None)
    primary_result = validators.get(primary or "", {})
    training_backends: list[str] = []
    if coco["valid"]:
        training_backends.append("detectron")
    if yolo["valid"] or yolo_sidecar["valid"]:
        training_backends.append("yolo")
    return {
        "valid": not errors and bool(compatible),
        "requested_format": requested,
        "compatible_formats": compatible,
        "errors": errors[:50], "warnings": warnings[:50],
        "classes": primary_result.get("classes", []),
        "splits": primary_result.get("splits", {}),
        "format_details": validators,
        "yolo_sidecar": yolo_sidecar,
        "training_backends": training_backends,
        "file_count": sum(1 for p in root.rglob("*") if p.is_file()),
        "size_bytes": _folder_size(root),
        "validated_at": _now(),
    }


def _registry_path(project_train_dir: Path) -> Path:
    return Path(project_train_dir) / "training_datasets.json"


def load_registry(project_train_dir: Path) -> dict[str, dict[str, Any]]:
    path = _registry_path(project_train_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("datasets", {}) if isinstance(data, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def save_registry(project_train_dir: Path, datasets: dict[str, dict[str, Any]]) -> None:
    path = _registry_path(project_train_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"datasets": datasets}, indent=2), encoding="utf-8")
    temporary.replace(path)


def list_datasets(project_train_dir: Path) -> list[dict[str, Any]]:
    registry = load_registry(project_train_dir)
    changed = False
    entries = list(registry.values())
    for entry in entries:
        root = Path(str(entry.get("storage_path", "")))
        entry["available"] = root.is_dir()
        # Upgrade registries created before backend compatibility was recorded.
        if root.is_dir() and "training_backends" not in entry.get("validation", {}):
            requested = str(entry.get("validation", {}).get("requested_format") or "auto")
            entry["validation"] = validate_dataset(root, requested)
            registry[str(entry.get("id"))] = entry
            changed = True
    if changed:
        save_registry(project_train_dir, registry)
    return sorted(entries, key=lambda item: str(item.get("created_at", "")), reverse=True)


def get_dataset(project_train_dir: Path, dataset_id: str) -> dict[str, Any] | None:
    entry = load_registry(project_train_dir).get(str(dataset_id))
    if entry:
        entry["available"] = Path(str(entry.get("storage_path", ""))).is_dir()
    return entry


def rename_dataset(project_train_dir: Path, dataset_id: str, display_name: str) -> dict[str, Any]:
    """Rename a dataset's display label while preserving its stable ID and path."""
    name = str(display_name or "").strip()[:128]
    if not name:
        raise DatasetUploadError("Training dataset name cannot be empty.")
    registry = load_registry(project_train_dir)
    entry = registry.get(str(dataset_id))
    if not entry:
        raise DatasetUploadError("Training dataset was not found.")
    entry["display_name"] = name
    entry["modified_at"] = _now()
    registry[str(dataset_id)] = entry
    save_registry(project_train_dir, registry)
    return entry


def delete_dataset(project_train_dir: Path, dataset_id: str) -> dict[str, Any]:
    """Delete one uploaded dataset directory and remove its registry entry."""
    project_train_dir = Path(project_train_dir).resolve()
    registry = load_registry(project_train_dir)
    entry = registry.get(str(dataset_id))
    if not entry:
        raise DatasetUploadError("Training dataset was not found.")
    if entry.get("source") == "legacy_project_data":
        raise DatasetUploadError("Project training data cannot be deleted from the uploaded-data library.")
    raw_path = str(entry.get("storage_path") or "").strip()
    if not raw_path:
        raise DatasetUploadError("Training dataset has no storage path.")
    root = Path(raw_path)
    if root.is_symlink():
        raise DatasetUploadError("Refusing to delete a dataset stored through a symbolic link.")
    root = root.resolve()
    project_root = project_train_dir.parent
    protected = {Path(root.anchor), Path.home().resolve(), project_train_dir, project_root}
    if root in protected or root in project_train_dir.parents or project_train_dir.is_relative_to(root):
        raise DatasetUploadError("Refusing to delete an unsafe training dataset path.")

    registry.pop(str(dataset_id), None)
    if not root.exists():
        save_registry(project_train_dir, registry)
        return entry
    if not root.is_dir():
        raise DatasetUploadError("Training dataset storage path is not a directory.")

    temporary = root.parent / f".{root.name}.pvrt-delete-{str(dataset_id)[:8]}"
    if temporary.exists():
        raise DatasetUploadError("A previous deletion is still pending for this dataset.")
    root.rename(temporary)
    try:
        save_registry(project_train_dir, registry)
        shutil.rmtree(temporary)
    except Exception:
        if temporary.exists() and not root.exists():
            temporary.rename(root)
        registry[str(dataset_id)] = entry
        save_registry(project_train_dir, registry)
        raise
    return entry


def ensure_legacy_dataset(project_train_dir: Path) -> None:
    """Register the project's existing train/data folder once for continuity."""
    project_train_dir = Path(project_train_dir)
    legacy_root = project_train_dir / "data"
    if not (legacy_root / "train").is_dir() or not (
        (legacy_root / "valid").is_dir() or (legacy_root / "val").is_dir()
    ):
        return
    registry = load_registry(project_train_dir)
    resolved = str(legacy_root.resolve())
    for dataset_id, entry in registry.items():
        if str(entry.get("storage_path", "")) != resolved:
            continue
        if "training_backends" not in entry.get("validation", {}):
            entry["validation"] = validate_dataset(legacy_root, "auto")
            entry["available"] = True
            registry[dataset_id] = entry
            save_registry(project_train_dir, registry)
        return
    report = validate_dataset(legacy_root, "auto")
    dataset_id = uuid.uuid5(uuid.NAMESPACE_URL, f"pvrt-training:{resolved}").hex
    registry[dataset_id] = {
        "id": dataset_id,
        "display_name": "Project training data",
        "storage_path": resolved,
        "created_at": _now(),
        "validation": report,
        "available": True,
        "source": "legacy_project_data",
    }
    save_registry(project_train_dir, registry)


def resolve_dataset_for_training(
    project_train_dir: Path,
    dataset_id: str,
    backend: str,
) -> dict[str, Any]:
    """Revalidate and resolve concrete split paths for one training backend."""
    project_train_dir = Path(project_train_dir)
    registry = load_registry(project_train_dir)
    entry = registry.get(str(dataset_id))
    if not entry:
        raise DatasetUploadError("Selected training dataset was not found.")
    root = Path(str(entry.get("storage_path", "")))
    if not root.is_dir():
        raise DatasetUploadError("Selected training dataset folder is missing.")
    requested = str(entry.get("validation", {}).get("requested_format") or "auto")
    report = validate_dataset(root, requested)
    entry["validation"] = report
    entry["available"] = True
    registry[str(dataset_id)] = entry
    save_registry(project_train_dir, registry)
    backend_name = str(backend or "").strip().lower()
    if backend_name not in report.get("training_backends", []):
        raise DatasetUploadError(
            f"Dataset is not valid for {backend_name or 'the selected backend'} training.",
            report,
        )
    if backend_name == "detectron":
        train_dir = _split(root, "train")
        valid_dir = _split(root, "valid", "val", "validation")
        dataset_yaml = None
        dataset_format = "coco"
    elif report.get("format_details", {}).get("yolo", {}).get("valid"):
        yolo_info = report["format_details"]["yolo"]
        train_dir = root / yolo_info["splits"]["train"]["image_dir"]
        valid_dir = root / yolo_info["splits"]["valid"]["image_dir"]
        dataset_yaml = root / yolo_info["data_yaml"]
        dataset_format = "yolo"
    else:
        train_dir = _split(root, "train")
        valid_dir = _split(root, "valid", "val", "validation")
        dataset_yaml = None
        dataset_format = "coco_yolo_sidecar"
    if not train_dir or not valid_dir or not train_dir.is_dir() or not valid_dir.is_dir():
        raise DatasetUploadError("Training or validation split folder is missing.")
    return {
        "entry": entry,
        "root": root,
        "train_dir": train_dir,
        "valid_dir": valid_dir,
        "dataset_yaml": dataset_yaml,
        "dataset_format": dataset_format,
    }


def install_dataset(
    *,
    files: Iterable[Any],
    destination: Path,
    display_name: str,
    requested_format: str,
    project_train_dir: Path,
) -> dict[str, Any]:
    destination = Path(destination)
    if destination.exists():
        raise DatasetUploadError("Destination folder already exists; choose a new folder.")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pvrt-training-upload-", dir=str(parent)))
    try:
        stage_upload(files, staging)
        report = validate_dataset(staging, requested_format)
        if not report["valid"]:
            raise DatasetUploadError("Training dataset validation failed.", report)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    dataset_id = uuid.uuid4().hex
    entry = {
        "id": dataset_id,
        "display_name": display_name,
        "storage_path": str(destination),
        "created_at": _now(),
        "validation": report,
        "available": True,
        "source": "uploaded",
    }
    registry = load_registry(project_train_dir)
    registry[dataset_id] = entry
    save_registry(project_train_dir, registry)
    return entry
