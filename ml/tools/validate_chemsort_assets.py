#!/usr/bin/env python3
"""Validate the locked ChemSort model contract and Flutter flashcard assets.

The validator uses only Python's standard library. If TensorFlow or
``tflite_runtime`` is installed, it also checks the model tensor metadata.

The files may live in different project directories. The validator searches
the project recursively by exact filename, so layouts such as
``models/tflite`` plus ``data/content`` are supported.

Example project layout::

    assets/chemsort/
      yolo11n-cls_data_v2_float32.tflite
      labels_9.txt
      chemsort_mapping.json
      flashcards.json
    tools/
      validate_chemsort_assets.py

Run from the Flutter project root::

    python ml/tools/validate_chemsort_assets.py --require-flashcards

Or from ``ml/`` with the project root explicitly::

    python -m tools.validate_chemsort_assets --project-root .. --require-flashcards

Pass ``--export-report reports/experiments/<run>/export_report.json`` to also
check that the model's SHA-256 is one that passed the export parity gate.

Every file path can also be supplied separately; run with ``--help`` for the
full command syntax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


MODEL_FILE = "yolo11n-cls_data_v2_float32.tflite"
LABELS_FILE = "labels_9.txt"
MAPPING_FILE = "chemsort_mapping.json"
FLASHCARDS_FILE = "flashcards.json"

ACCEPTED_INPUT_SHAPES = ([1, 224, 224, 3], [1, 3, 224, 224])
EXPECTED_LABELS = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]
EXPECTED_MATERIAL_ORDER = [
    "paper_cardboard",
    "organic",
    "glass",
    "metal",
    "other",
    "plastic",
]
EXPECTED_APP_CLASS_NAMES = [
    "Paper/Cardboard",
    "Organic",
    "Glass",
    "Metal",
    "Other",
    "Plastic",
]
EXPECTED_APP_GROUPS = {
    "Paper/Cardboard": ["Cardboard", "Paper"],
    "Organic": ["Food Organics"],
    "Glass": ["Glass"],
    "Metal": ["Metal"],
    "Other": ["Miscellaneous Trash", "Textile Trash", "Vegetation"],
    "Plastic": ["Plastic"],
}
EXPECTED_FLASHCARD_KEYS = [
    "id",
    "material_key",
    "plastic_code",
    "title_vi",
    "content_vi",
    "source_title",
    "source_url",
    "review_status",
]
CONTENT_LENGTH_RANGE = (60, 400)
IGNORED_DISCOVERY_DIRECTORIES = {
    "ml",
    ".dart_tool",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "build",
}


def configure_utf8_console() -> None:
    """Keep Vietnamese diagnostics printable on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, validation: Validation) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        validation.errors.append(f"Không đọc được JSON {path.name}: {error}")
        return None


def is_https_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def absolute_from_root(value: Path, project_root: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_root / expanded).resolve()


def discover_asset(
    filename: str,
    project_root: Path,
    validation: Validation,
) -> Path:
    common_locations = [
        project_root / filename,
        project_root / "assets" / filename,
        project_root / "assets" / "chemsort" / filename,
        project_root / "assets" / "ml" / filename,
        project_root / "assets" / "models" / filename,
        project_root / "assets" / "data" / filename,
        project_root / "models" / filename,
        project_root / "models" / "tflite" / filename,
        project_root / "data" / filename,
        project_root / "data" / "content" / filename,
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()

    # The first conventional location wins (assets/chemsort for the app).
    for candidate in common_locations:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    try:
        recursive_matches = project_root.rglob(filename)
        for candidate in recursive_matches:
            relative_parts = candidate.relative_to(project_root).parts[:-1]
            if any(part in IGNORED_DISCOVERY_DIRECTORIES for part in relative_parts):
                continue
            resolved = candidate.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    except OSError as error:
        validation.warnings.append(
            f"Không thể tìm toàn bộ project cho {filename}: {error}"
        )

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        choices = "; ".join(str(path) for path in candidates)
        validation.errors.append(
            f"Tìm thấy nhiều file {filename}: {choices}. "
            "Hãy chỉ định đường dẫn riêng trên dòng lệnh."
        )
        return candidates[0]

    return (project_root / filename).resolve()


def resolve_asset_paths(
    args: argparse.Namespace,
    validation: Validation,
) -> tuple[Path, Path, Path, Path, Path]:
    project_root = args.project_root.expanduser().resolve()
    validation.require(project_root.is_dir(), f"Project root không tồn tại: {project_root}")

    assets_dir = None
    if args.assets_dir is not None:
        assets_dir = absolute_from_root(args.assets_dir, project_root)

    def resolve_one(explicit: Path | None, filename: str) -> Path:
        if explicit is not None:
            return absolute_from_root(explicit, project_root)
        if assets_dir is not None:
            return (assets_dir / filename).resolve()
        return discover_asset(filename, project_root, validation)

    return (
        project_root,
        resolve_one(args.model, MODEL_FILE),
        resolve_one(args.labels, LABELS_FILE),
        resolve_one(args.mapping, MAPPING_FILE),
        resolve_one(args.flashcards, FLASHCARDS_FILE),
    )


def accepted_model_hashes(
    report_path: Path | None,
    validation: Validation,
) -> set[str] | None:
    """SHA-256 values of formats that passed an export parity gate."""
    if report_path is None:
        validation.warnings.append(
            "Không có --export-report; SHA-256 chỉ được báo cáo, chưa đối chiếu"
        )
        return None
    if not report_path.is_file():
        validation.errors.append(f"Không tìm thấy export report: {report_path}")
        return None
    report = read_json(report_path, validation)
    if not isinstance(report, dict):
        return None
    hashes: set[str] = set()
    # New pipeline: {"formats": {"float32": {...}, "float16": {...}}}
    for entry in (report.get("formats") or {}).values():
        if isinstance(entry, dict) and entry.get("passed_gate") is True:
            digest = entry.get("model_sha256")
            if isinstance(digest, str) and len(digest) == 64:
                hashes.add(digest.lower())
    # Legacy pipeline: {"float32": {...}, "float16": {...}}
    for key in ("float32", "float16"):
        entry = report.get(key)
        if isinstance(entry, dict) and entry.get("passed_gate") is True:
            digest = entry.get("sha256")
            if isinstance(digest, str) and len(digest) == 64:
                hashes.add(digest.lower())
    if not hashes:
        validation.errors.append("Export report không có định dạng nào passed_gate=true")
    return hashes


def validate_model(
    path: Path,
    accepted_hashes: set[str] | None,
    validation: Validation,
) -> dict[str, Any]:
    result: dict[str, Any] = {"sha256": None, "tensor_check": "not_run", "input_shape": None}
    if not path.is_file():
        validation.errors.append(f"Thiếu model: {path}")
        return result

    size = path.stat().st_size
    validation.require(size > 1_000_000, "Model TFLite nhỏ bất thường")
    with path.open("rb") as handle:
        header = handle.read(8)
    validation.require(
        len(header) >= 8 and header[4:8] == b"TFL3",
        "Model không có định danh FlatBuffer TFL3 hợp lệ",
    )

    digest = sha256_file(path)
    result["sha256"] = digest
    if accepted_hashes:
        validation.require(
            digest in accepted_hashes,
            "SHA-256 của model không khớp format nào đã qua export parity gate",
        )

    interpreter_class = None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf  # type: ignore

        interpreter_class = tf.lite.Interpreter
    except ImportError:
        for module in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
            try:
                interpreter_class = __import__(module, fromlist=["Interpreter"]).Interpreter
                break
            except ImportError:
                continue
        else:
            validation.warnings.append(
                "Không có TensorFlow/LiteRT/tflite_runtime; đã kiểm tra model bằng TFL3 và SHA-256"
            )

    if interpreter_class is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*tf.lite.Interpreter is deprecated.*",
                    category=UserWarning,
                )
                interpreter = interpreter_class(model_path=str(path), num_threads=1)
            interpreter.allocate_tensors()
            inputs = interpreter.get_input_details()
            outputs = interpreter.get_output_details()
            validation.require(len(inputs) == 1, "Model phải có đúng một input tensor")
            validation.require(len(outputs) == 1, "Model phải có đúng một output tensor")
            if inputs:
                input_shape = [int(value) for value in inputs[0]["shape"]]
                input_dtype = getattr(inputs[0]["dtype"], "__name__", str(inputs[0]["dtype"]))
                result["input_shape"] = input_shape
                validation.require(
                    input_shape in ACCEPTED_INPUT_SHAPES,
                    "Input tensor phải là [1, 224, 224, 3] (NHWC) hoặc "
                    f"[1, 3, 224, 224] (NCHW), hiện là {input_shape}",
                )
                validation.require(
                    input_dtype == "float32",
                    f"Input tensor phải là float32, hiện là {input_dtype}",
                )
            if outputs:
                output_shape = [int(value) for value in outputs[0]["shape"]]
                output_dtype = getattr(outputs[0]["dtype"], "__name__", str(outputs[0]["dtype"]))
                output_size = 1
                for value in output_shape:
                    output_size *= value
                validation.require(
                    output_size == 9,
                    f"Output tensor phải có 9 giá trị, hiện là {output_shape}",
                )
                validation.require(
                    output_dtype == "float32",
                    f"Output tensor phải là float32, hiện là {output_dtype}",
                )
            result["tensor_check"] = "passed"
        except Exception as error:  # Runtime-specific model loading errors.
            validation.errors.append(f"Không kiểm tra được tensor TFLite: {error}")
            result["tensor_check"] = "failed"

    return result


def validate_labels(path: Path, validation: Validation) -> list[str]:
    if not path.is_file():
        validation.errors.append(f"Thiếu labels: {path}")
        return []
    try:
        labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as error:
        validation.errors.append(f"Không đọc được labels: {error}")
        return []

    validation.require(labels == EXPECTED_LABELS, "Thứ tự labels_9.txt không đúng")
    validation.require(len(labels) == len(set(labels)), "labels_9.txt có nhãn trùng")
    return labels


def validate_mapping(
    path: Path,
    labels: list[str],
    validation: Validation,
    model_input_shape: list[int] | None = None,
) -> dict[str, Any] | None:
    mapping = read_json(path, validation)
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        validation.errors.append("chemsort_mapping.json phải là một JSON object")
        return None

    model_output = mapping.get("model_output_9class")
    validation.require(
        isinstance(model_output, dict),
        "Mapping thiếu object model_output_9class",
    )
    if isinstance(model_output, dict):
        validation.require(
            model_output.get("classes_in_index_order") == labels,
            "model_output_9class không khớp labels_9.txt",
        )

    app_output = mapping.get("app_output_6class")
    validation.require(
        isinstance(app_output, dict),
        "Mapping thiếu object app_output_6class",
    )
    if not isinstance(app_output, dict):
        return mapping
    validation.require(
        app_output.get("classes_in_index_order") == EXPECTED_APP_CLASS_NAMES,
        "app_output_6class không đúng thứ tự 6 nhóm",
    )
    groups = app_output.get("groups")
    validation.require(
        groups == EXPECTED_APP_GROUPS,
        "app_output_6class.groups không khớp contract 9→6",
    )

    input_contract = mapping.get("input")
    validation.require(isinstance(input_contract, dict), "Mapping thiếu object input")
    if isinstance(input_contract, dict):
        shape = input_contract.get("shape")
        validation.require(
            shape in ACCEPTED_INPUT_SHAPES,
            f"Mapping input.shape không hợp lệ: {shape}",
        )
        if model_input_shape is not None:
            validation.require(
                shape == model_input_shape,
                f"Mapping input.shape {shape} khác input tensor của model {model_input_shape}",
            )
        validation.require(
            input_contract.get("dtype") == "float32",
            "Mapping input.dtype phải là float32",
        )
        validation.require(
            input_contract.get("color_order") == "RGB",
            "Mapping input.color_order phải là RGB",
        )

    return mapping


def validate_flashcards(
    path: Path,
    mapping: dict[str, Any] | None,
    validation: Validation,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "records": 0,
        "material_counts": {},
        "plastic_general": 0,
        "plastic_code_counts": {},
    }
    flashcards = read_json(path, validation)
    if flashcards is None:
        return result
    if not isinstance(flashcards, list):
        validation.errors.append("flashcards.json phải là một JSON array")
        return result

    result["records"] = len(flashcards)
    validation.require(len(flashcards) > 0, "flashcards.json không có bản ghi nào")

    ids: list[str] = []
    titles: list[str] = []
    contents: list[str] = []
    material_counts: Counter[str] = Counter()
    plastic_general = 0
    plastic_code_counts: Counter[int] = Counter()
    forbidden_keys = {"tags", "question_vi", "answer_vi", "explanation_vi"}

    for row_number, card in enumerate(flashcards, start=1):
        prefix = f"Flashcard #{row_number}"
        if not isinstance(card, dict):
            validation.errors.append(f"{prefix} phải là object")
            continue

        validation.require(
            list(card.keys()) == EXPECTED_FLASHCARD_KEYS,
            f"{prefix} phải có đúng 8 trường theo đúng thứ tự",
        )
        validation.require(
            not forbidden_keys.intersection(card),
            f"{prefix} chứa trường đã loại bỏ: {sorted(forbidden_keys.intersection(card))}",
        )

        card_id = card.get("id")
        material_key = card.get("material_key")
        plastic_code = card.get("plastic_code")
        title = card.get("title_vi")
        content = card.get("content_vi")
        source_title = card.get("source_title")
        source_url = card.get("source_url")
        review_status = card.get("review_status")

        validation.require(isinstance(card_id, str) and bool(card_id.strip()), f"{prefix}: id bị trống")
        validation.require(
            material_key in EXPECTED_MATERIAL_ORDER,
            f"{prefix}: material_key không hợp lệ: {material_key}",
        )
        validation.require(
            isinstance(title, str) and len(title.strip()) >= 12,
            f"{prefix}: title_vi quá ngắn hoặc không hợp lệ",
        )
        low, high = CONTENT_LENGTH_RANGE
        validation.require(
            isinstance(content, str) and low <= len(content) <= high,
            f"{prefix}: content_vi phải dài {low}–{high} ký tự",
        )
        validation.require(
            isinstance(source_title, str) and bool(source_title.strip()),
            f"{prefix}: source_title bị trống",
        )
        validation.require(is_https_url(source_url), f"{prefix}: source_url phải là URL HTTPS")
        validation.require(review_status == "reviewed", f"{prefix}: review_status phải là reviewed")

        if isinstance(card_id, str):
            ids.append(card_id)
        if isinstance(title, str):
            titles.append(title)
        if isinstance(content, str):
            contents.append(content)
        if isinstance(material_key, str):
            material_counts[material_key] += 1

        if material_key == "plastic":
            if plastic_code is None:
                plastic_general += 1
            else:
                validation.require(
                    type(plastic_code) is int and 1 <= plastic_code <= 7,
                    f"{prefix}: plastic_code phải là null hoặc số nguyên 1–7",
                )
                if type(plastic_code) is int:
                    plastic_code_counts[plastic_code] += 1
        else:
            validation.require(
                plastic_code is None,
                f"{prefix}: plastic_code chỉ được dùng cho material_key=plastic",
            )

    validation.require(len(ids) == len(set(ids)), "id bị trùng")
    validation.require(len(titles) == len(set(titles)), "title_vi bị trùng")
    validation.require(len(contents) == len(set(contents)), "content_vi bị trùng")
    missing_materials = [key for key in EXPECTED_MATERIAL_ORDER if material_counts[key] == 0]
    validation.require(
        not missing_materials,
        f"Nhóm vật liệu chưa có fact nào: {missing_materials}",
    )
    validation.require(plastic_general > 0, "Cần ít nhất một fact nhựa chung (plastic_code=null)")
    missing_codes = [code for code in range(1, 8) if plastic_code_counts[code] == 0]
    validation.require(
        not missing_codes,
        f"Mã nhựa chưa có fact nào: {missing_codes}",
    )

    if mapping is not None:
        app_order = mapping.get("app_output_6class", {}).get(
            "classes_in_index_order"
        )
        mapping_order = (
            [name.lower().replace("/", "_") for name in app_order]
            if isinstance(app_order, list)
            else None
        )
        validation.require(
            mapping_order == EXPECTED_MATERIAL_ORDER,
            "material_key trong flashcards không đồng bộ với mapping",
        )

    result["material_counts"] = dict(material_counts)
    result["plastic_general"] = plastic_general
    result["plastic_code_counts"] = dict(sorted(plastic_code_counts.items()))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ChemSort TFLite, labels, mapping and flashcards assets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Thư mục gốc project; mặc định là thư mục hiện tại",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Dùng khi cả bốn file nằm chung một thư mục",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"Đường dẫn riêng tới {MODEL_FILE}",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=f"Đường dẫn riêng tới {LABELS_FILE}",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=f"Đường dẫn riêng tới {MAPPING_FILE}",
    )
    parser.add_argument(
        "--flashcards",
        type=Path,
        default=None,
        help=f"Đường dẫn riêng tới {FLASHCARDS_FILE}",
    )
    parser.add_argument(
        "--require-flashcards",
        action="store_true",
        help="Bắt buộc kiểm tra flashcards; dùng cho bundle ứng dụng hoàn chỉnh",
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=None,
        help="export_report.json để đối chiếu SHA-256 của model",
    )
    return parser.parse_args()


def main() -> int:
    configure_utf8_console()
    args = parse_args()
    validation = Validation()
    project_root, model_path, labels_path, mapping_path, flashcards_path = (
        resolve_asset_paths(args, validation)
    )

    report_path = (
        absolute_from_root(args.export_report, project_root)
        if args.export_report is not None
        else None
    )
    accepted_hashes = accepted_model_hashes(report_path, validation)
    model_result = validate_model(model_path, accepted_hashes, validation)
    labels = validate_labels(labels_path, validation)
    mapping = validate_mapping(
        mapping_path, labels, validation, model_result.get("input_shape")
    )
    content_required = bool(
        args.require_flashcards
        or args.flashcards is not None
        or args.assets_dir is not None
    )
    content_validated = flashcards_path.is_file() or content_required
    if content_validated:
        flashcard_result = validate_flashcards(
            flashcards_path,
            mapping,
            validation,
        )
    else:
        flashcard_result = {
            "records": 0,
            "material_counts": {},
            "plastic_general": 0,
            "plastic_code_counts": {},
        }
        validation.warnings.append(
            "Không tìm thấy flashcards.json; đã bỏ qua content validation. "
            "Dùng --require-flashcards khi kiểm tra bundle ứng dụng hoàn chỉnh."
        )

    print("ChemSort asset validation")
    print(f"Project root: {project_root}")
    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")
    print(f"Mapping: {mapping_path}")
    print(
        f"Flashcards: {flashcards_path}"
        if content_validated
        else "Flashcards: <skipped>"
    )
    print(f"Model SHA-256: {model_result['sha256'] or '<unavailable>'}")
    print(f"Tensor check: {model_result['tensor_check']}")
    print(f"Label count: {len(labels)}")
    if content_validated:
        print(f"Flashcard count: {flashcard_result['records']}")
        print(
            "Material counts: "
            + json.dumps(
                flashcard_result["material_counts"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        print(f"Plastic general: {flashcard_result['plastic_general']}")
        print(
            "Plastic codes: "
            + json.dumps(
                flashcard_result["plastic_code_counts"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    if validation.warnings:
        print("\nCẢNH BÁO:")
        for warning in validation.warnings:
            print(f"- {warning}")

    if validation.errors:
        print("\nKHÔNG ĐẠT:")
        for error in validation.errors:
            print(f"- {error}")
        return 1

    if content_validated:
        print("\nĐẠT: model assets và flashcards đồng bộ.")
    else:
        print("\nĐẠT: model, labels và mapping đồng bộ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
