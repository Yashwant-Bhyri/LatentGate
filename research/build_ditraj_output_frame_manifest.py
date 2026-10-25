#!/usr/bin/env python3
"""Build or validate the evaluator-side DiTraj 102-frame manifest.

This helper deliberately does not run a model, copy frames, create an archive,
score annotations, or compute background classifications. Its build mode reads
six completed frame directories and writes only a blind frame manifest plus a
separate owner-only blind-ID/source mapping. Its validate mode is read-only and
checks a finalized blind ZIP against the manifest and detached integrity receipt.

Do not invoke build mode until the registered Gate 1, no-paid, full-output, and
packet-release prerequisites have passed. Build mode also requires the exact
run-report bytes to match a separately reviewed SHA-256 pin in the project
registry; the report cannot attest its own integrity. This source has not been
executed or tested; review and isolated validation are still required before
operational use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image


MANIFEST_SCHEMA = "ditraj.generated_output_frame_manifest.v1"
INTEGRITY_SCHEMA = "ditraj.packet_integrity.v1"
PRIVATE_MAP_SCHEMA = "ditraj.private_blind_run_map.v1"
RUN_REPORT_EXPERIMENT_ID = "EXP-DITRAJ-WAN13B-PAIRED-PATH-PILOT-002"
RUN_REPORT_PIN_REGISTRY_FIELD = "generated_output_frame_manifest_builder_run_report_sha256"
FULL_SCHEDULE_STEPS = 30
REGISTERED_SEED = 2026
REGISTERED_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REGISTERED_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
REGISTERED_DITRAJ_COMMIT = "6fa8e613c9e7c40ec2db456ab2ba34367a076886"
# Pin to the runner source inspected in the 2026-10-17 crosswalk. Change only
# alongside a reviewed runner/schema update and a new registry pin.
PINNED_RUNNER_SHA256 = "c00e1470c3e67da3573f61614f90e3ec9ab525135040ca3928a4bc3db4432984"
CLIP_COUNT = 6
FRAMES_PER_CLIP = 17
WIDTH = 832
HEIGHT = 480
PIXEL_FORMAT = "RGB8"
FRAME_NAMES = tuple(f"frame_{i:03d}.png" for i in range(FRAMES_PER_CLIP))

MANIFEST_KEYS = (
    "schema_version",
    "packet_id",
    "expected_clip_count",
    "expected_frames_per_clip",
    "pixel_format",
    "width",
    "height",
    "frames",
)
FRAME_KEYS = (
    "blind_clip_id",
    "frame_index",
    "image_file",
    "sha256",
    "width",
    "height",
)
INTEGRITY_KEYS = (
    "schema_version",
    "packet_id",
    "frame_manifest_sha256",
    "annotator_archive_sha256",
    "frame_count",
)
PRIVATE_MAP_KEYS = (
    "schema_version",
    "packet_id",
    "run_report_sha256",
    "run_report_path",
    "blind_to_source",
)
PRIVATE_MAP_ENTRY_KEYS = ("blind_clip_id", "source_label", "source_dir")

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
EXPECTED_RUN_LABELS = {
    f"{direction}_{arm}"
    for direction in ("left", "right")
    for arm in ("base", "sg_only", "ditraj")
}
EXPECTED_SETTINGS = {
    "base": {"mask_step": 0, "fixRope_step": 0},
    "sg_only": {"mask_step": 18, "fixRope_step": 0},
    "ditraj": {"mask_step": 18, "fixRope_step": 3},
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Pixel data plus ordinary color/display metadata only. Text, EXIF, timestamps,
# private chunks, and unknown ancillary chunks are rejected to avoid condition
# leakage through the blind image packet.
ALLOWED_PNG_CHUNKS = {
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    b"sRGB",
    b"gAMA",
    b"cHRM",
    b"pHYs",
}
ALLOWED_ZIP_DIRS = {"images/"}


class ManifestError(ValueError):
    """Raised when source frames or a manifest violate the frozen contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_int(value: Any) -> bool:
    # bool subclasses int in Python, but JSON booleans are not schema integers.
    return isinstance(value, int) and not isinstance(value, bool)


def _expect_keys(value: Any, expected: Iterable[str], where: str) -> None:
    expected_tuple = tuple(expected)
    if not isinstance(value, dict) or tuple(value.keys()) != expected_tuple:
        raise ManifestError(f"{where}: keys/order do not match the registered schema")


def _uuid4(value: Any, where: str) -> str:
    if not isinstance(value, str) or not UUID4_RE.fullmatch(value):
        raise ManifestError(f"{where}: expected a lowercase canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ManifestError(f"{where}: malformed UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ManifestError(f"{where}: expected a lowercase canonical UUIDv4")
    return value


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"JSON contains duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ManifestError(f"non-standard JSON numeric token {value!r} is invalid")


def _load_strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_nofollow(path, "JSON artifact")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ManifestError("JSON artifacts must not contain a UTF-8 BOM")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid UTF-8 JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError("JSON artifact root must be an object")
    return value, raw


def _read_regular_nofollow(path: Path, description: str) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ManifestError("artifact reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot read {description} as a non-symlink file: {path.name}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ManifestError(f"{description} is not a regular file: {path.name}")
        with os.fdopen(fd, "rb", closefd=True) as stream:
            fd = -1
            return stream.read()
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise


def _serialize_manifest(manifest: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError("manifest cannot be serialized as strict JSON") from exc
    return (text + "\n").encode("utf-8")


def _image_relpath(blind_clip_id: str, frame_index: int) -> str:
    return f"images/{blind_clip_id}/frame_{frame_index:03d}.png"


def _duplicate_hash_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_hash.setdefault(row["sha256"], []).append(
            {"blind_clip_id": row["blind_clip_id"], "frame_index": row["frame_index"]}
        )
    return [
        {"sha256": digest, "frame_keys": keys}
        for digest, keys in sorted(by_hash.items())
        if len(keys) > 1
    ]


def _check_png_chunks(raw: bytes, where: str) -> None:
    if not raw.startswith(PNG_SIGNATURE):
        raise ManifestError(f"{where}: not a PNG file")
    offset = len(PNG_SIGNATURE)
    saw_ihdr = False
    saw_iend = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            raise ManifestError(f"{where}: truncated PNG chunk")
        length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(raw):
            raise ManifestError(f"{where}: PNG chunk length exceeds file")
        if chunk_type not in ALLOWED_PNG_CHUNKS:
            raise ManifestError(f"{where}: disallowed PNG metadata/chunk {chunk_type!r}")
        if offset == len(PNG_SIGNATURE):
            if chunk_type != b"IHDR" or length != 13:
                raise ManifestError(f"{where}: PNG must begin with a valid IHDR")
            ihdr = raw[offset + 8 : offset + 8 + length]
            png_width = int.from_bytes(ihdr[0:4], "big")
            png_height = int.from_bytes(ihdr[4:8], "big")
            bit_depth, color_type = ihdr[8], ihdr[9]
            compression, filtering, interlace = ihdr[10], ihdr[11], ihdr[12]
            if (png_width, png_height) != (WIDTH, HEIGHT):
                raise ManifestError(f"{where}: PNG IHDR dimensions are not 832x480")
            if (bit_depth, color_type) != (8, 2):
                raise ManifestError(f"{where}: PNG IHDR must be 8-bit truecolor RGB")
            if compression != 0 or filtering != 0 or interlace not in (0, 1):
                raise ManifestError(f"{where}: unsupported PNG IHDR encoding")
            saw_ihdr = True
        if chunk_type == b"IEND":
            if length != 0 or end != len(raw):
                raise ManifestError(f"{where}: malformed or trailing data after PNG IEND")
            saw_iend = True
            break
        offset = end
    if not saw_ihdr or not saw_iend:
        raise ManifestError(f"{where}: PNG is missing IHDR or IEND")


def _inspect_png(raw: bytes, where: str) -> tuple[int, int]:
    _check_png_chunks(raw, where)
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ManifestError(f"{where}: decoded format is not PNG")
            if image.mode != "RGB":
                raise ManifestError(f"{where}: PNG must decode as RGB without alpha")
            if image.size != (WIDTH, HEIGHT):
                raise ManifestError(f"{where}: expected {WIDTH}x{HEIGHT}, got {image.size}")
            image.load()
            if image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
                raise ManifestError(f"{where}: decoded pixels violate RGB8 dimensions")
    except ManifestError:
        raise
    except Exception as exc:  # Pillow raises several decode/CRC exception types.
        raise ManifestError(f"{where}: PNG decode failed") from exc
    return WIDTH, HEIGHT


def _validate_manifest_object(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    _expect_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ManifestError("manifest: unsupported schema_version")
    packet_id = _uuid4(manifest["packet_id"], "manifest.packet_id")
    constants = {
        "expected_clip_count": CLIP_COUNT,
        "expected_frames_per_clip": FRAMES_PER_CLIP,
        "pixel_format": PIXEL_FORMAT,
        "width": WIDTH,
        "height": HEIGHT,
    }
    for key, expected in constants.items():
        actual = manifest[key]
        if isinstance(expected, int):
            if not _is_int(actual) or actual != expected:
                raise ManifestError(f"manifest.{key}: expected integer {expected}")
        elif actual != expected:
            raise ManifestError(f"manifest.{key}: expected {expected!r}")

    rows = manifest["frames"]
    if not isinstance(rows, list) or len(rows) != CLIP_COUNT * FRAMES_PER_CLIP:
        raise ManifestError("manifest.frames: expected exactly 102 rows")

    ids: set[str] = set()
    keys: set[tuple[str, int]] = set()
    paths: set[str] = set()
    previous_sort_key: tuple[str, int] | None = None
    for position, row in enumerate(rows):
        where = f"manifest.frames[{position}]"
        _expect_keys(row, FRAME_KEYS, where)
        clip_id = _uuid4(row["blind_clip_id"], f"{where}.blind_clip_id")
        frame_index = row["frame_index"]
        if not _is_int(frame_index) or not 0 <= frame_index < FRAMES_PER_CLIP:
            raise ManifestError(f"{where}.frame_index: expected integer 0..16")
        key = (clip_id, frame_index)
        if key in keys:
            raise ManifestError(f"{where}: duplicate blind key")
        keys.add(key)
        ids.add(clip_id)

        expected_path = _image_relpath(clip_id, frame_index)
        image_file = row["image_file"]
        if not isinstance(image_file, str) or image_file != expected_path:
            raise ManifestError(f"{where}.image_file: path must match blind ID/frame index")
        pure_path = PurePosixPath(image_file)
        if (
            pure_path.is_absolute()
            or "\\" in image_file
            or "\x00" in image_file
            or any(part in {"", ".", ".."} for part in image_file.split("/"))
        ):
            raise ManifestError(f"{where}.image_file: unsafe/non-normalized relative path")
        if image_file in paths:
            raise ManifestError(f"{where}.image_file: duplicate image path")
        paths.add(image_file)

        digest = row["sha256"]
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ManifestError(f"{where}.sha256: expected lowercase SHA-256")
        for dimension, expected in (("width", WIDTH), ("height", HEIGHT)):
            actual = row[dimension]
            if not _is_int(actual) or actual != expected:
                raise ManifestError(f"{where}.{dimension}: expected integer {expected}")

        sort_key = (clip_id, frame_index)
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ManifestError("manifest.frames: rows are not in canonical ID/index order")
        previous_sort_key = sort_key

    if len(ids) != CLIP_COUNT:
        raise ManifestError("manifest.frames: expected exactly six distinct blind IDs")
    expected_keys = {
        (clip_id, index)
        for clip_id in ids
        for index in range(FRAMES_PER_CLIP)
    }
    if keys != expected_keys:
        raise ManifestError("manifest.frames: blind key set is not the complete 6x17 product")
    return rows, ids


def _read_source_frame(path: Path, label: str, frame_name: str) -> tuple[bytes, str]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ManifestError(f"source frame missing for {label}/{frame_name}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ManifestError(f"source frame is not a regular non-symlink file: {frame_name}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ManifestError("source frame reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot read source frame {frame_name}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ManifestError(f"source frame is not a regular file: {frame_name}")
        with os.fdopen(fd, "rb", closefd=True) as stream:
            fd = -1
            raw = stream.read()
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise
    _inspect_png(raw, f"source {label}/{frame_name}")
    return raw, _sha256(raw)


def _resolve_source_dir(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser()
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ManifestError(f"source directory for {label} must not be a symlink")
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"source directory unavailable for {label}") from exc
    if not resolved.is_dir():
        raise ManifestError(f"source path for {label} is not a directory")
    return resolved


def _parse_source_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ManifestError("each --source-run must use LABEL=FRAME_DIRECTORY")
    label, path = value.split("=", 1)
    if not SOURCE_LABEL_RE.fullmatch(label):
        raise ManifestError("source labels must be 1..64 safe ASCII characters")
    if not path:
        raise ManifestError("source frame directory must not be empty")
    return label, path


def _verify_full_run_report(
    report_path: Path,
    source_specs: list[tuple[str, str]],
    expected_report_sha256: str,
) -> tuple[str, Path]:
    report_path = report_path.expanduser()
    if report_path.name != "pilot_report.json":
        raise ManifestError("run report input must be named pilot_report.json")
    if not SHA256_RE.fullmatch(expected_report_sha256):
        raise ManifestError("a separately registered lowercase run-report SHA-256 is required")
    try:
        if stat.S_ISLNK(report_path.lstat().st_mode):
            raise ManifestError("run report must not be a symlink")
        resolved_report = report_path.resolve(strict=True)
    except OSError as exc:
        raise ManifestError("full-schedule pilot report is unavailable") from exc
    report, report_bytes = _load_strict_json(resolved_report)
    report_sha256 = _sha256(report_bytes)
    if report_sha256 != expected_report_sha256:
        raise ManifestError("run report bytes do not match the independently registered report hash")
    if report.get("experiment_id") != RUN_REPORT_EXPERIMENT_ID:
        raise ManifestError("run report experiment_id does not match the registered six-arm pilot")
    if report.get("status") != "completed_single_seed_screen":
        raise ManifestError("run report is not a completed full-schedule six-arm screen")

    model = report.get("model")
    ditraj = report.get("ditraj")
    if not isinstance(model, dict) or (
        model.get("repo_id"), model.get("revision")
    ) != (REGISTERED_MODEL_ID, REGISTERED_MODEL_REVISION):
        raise ManifestError("run report model identity/revision differs from the registered pilot")
    if not isinstance(ditraj, dict) or ditraj.get("commit") != REGISTERED_DITRAJ_COMMIT:
        raise ManifestError("run report DiTraj commit differs from the registered pilot")

    generation = report.get("generation")
    if not isinstance(generation, dict):
        raise ManifestError("run report is missing generation metadata")
    expected_generation = {
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": FRAMES_PER_CLIP,
        "num_inference_steps": FULL_SCHEDULE_STEPS,
        "seed": REGISTERED_SEED,
    }
    for key, expected in expected_generation.items():
        value = generation.get(key)
        if not _is_int(value) or value != expected:
            raise ManifestError(f"run report generation.{key} is not the registered full schedule")
    noise_hash = generation.get("same_initial_noise_sha256_across_arms")
    if not isinstance(noise_hash, str) or not SHA256_RE.fullmatch(noise_hash):
        raise ManifestError("run report has no valid common initial-noise hash")

    adaptations = report.get("local_source_adaptations")
    if not isinstance(adaptations, dict) or adaptations.get("runner_sha256") != PINNED_RUNNER_SHA256:
        raise ManifestError("run report runner hash does not match the reviewed source pin")
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != EXPECTED_RUN_LABELS:
        raise ManifestError("run report must contain the exact mirrored Base/SG-only/DiTraj six-arm set")

    report_parent = resolved_report.parent
    for source_label, source_path_text in source_specs:
        source_dir = _resolve_source_dir(source_path_text, source_label)
        if source_dir.parent != report_parent or source_dir.name != source_label:
            raise ManifestError("source frame directory must be the matching run-report child directory")
        arm_record = arms.get(source_label)
        if not isinstance(arm_record, dict):
            raise ManifestError("run report is missing an expected arm record")
        direction, arm = source_label.split("_", 1)
        if arm_record.get("path_condition") != direction:
            raise ManifestError("run report arm path does not match its source label")
        if arm_record.get("settings") != EXPECTED_SETTINGS[arm]:
            raise ManifestError("run report arm settings differ from the registered controls")
        arm_noise = arm_record.get("initial_noise_sha256")
        if arm_noise != noise_hash:
            raise ManifestError("run report arms do not share the recorded initial-noise hash")
        expected_frame_paths = [f"{source_label}/{name}" for name in FRAME_NAMES]
        if arm_record.get("frame_paths") != expected_frame_paths:
            raise ManifestError("run report frame paths are incomplete, extra, or reordered")

    return report_sha256, resolved_report


def _registered_run_report_sha256() -> str:
    registry_path = Path(__file__).resolve().parent.parent / "latentgate_experiment_registry.yaml"
    raw = _read_regular_nofollow(registry_path, "project experiment registry")
    try:
        registry_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("experiment registry is not UTF-8") from exc
    pin_lines = re.findall(
        rf"^    {re.escape(RUN_REPORT_PIN_REGISTRY_FIELD)}:\s*(.*?)\s*$",
        registry_text,
        flags=re.MULTILINE,
    )
    if len(pin_lines) != 1:
        raise ManifestError("registry must contain exactly one run-report SHA-256 pin field")
    match = re.fullmatch(r'"([0-9a-f]{64})"', pin_lines[0])
    if match is None:
        raise ManifestError("registered run-report SHA-256 is absent or not a lowercase pinned digest")
    return match.group(1)


def _build_documents(
    source_specs: list[tuple[str, str]], report_sha256: str, report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(source_specs) != CLIP_COUNT:
        raise ManifestError("build requires exactly six --source-run arguments")
    labels = [label for label, _ in source_specs]
    if len(set(labels)) != CLIP_COUNT or set(labels) != EXPECTED_RUN_LABELS:
        raise ManifestError("source labels must be the exact registered mirrored six-arm set")

    resolved_sources: list[tuple[str, Path]] = []
    for label, path_text in source_specs:
        resolved_sources.append((label, _resolve_source_dir(path_text, label)))
    resolved_dirs = [path for _, path in resolved_sources]
    if len(set(resolved_dirs)) != CLIP_COUNT:
        raise ManifestError("each source run must use a distinct frame directory")

    packet_id = str(uuid.uuid4())
    frame_rows: list[dict[str, Any]] = []
    blind_to_source: list[dict[str, str]] = []
    generated_ids: set[str] = set()

    for source_label, source_dir in resolved_sources:
        clip_id = str(uuid.uuid4())
        if clip_id in generated_ids:
            raise ManifestError("UUIDv4 collision; refuse rather than reuse a blind ID")
        generated_ids.add(clip_id)

        try:
            entries = list(source_dir.iterdir())
        except OSError as exc:
            raise ManifestError(f"cannot enumerate source frame directory for {source_label}") from exc
        entry_names = [entry.name for entry in entries]
        if len(entry_names) != FRAMES_PER_CLIP or set(entry_names) != set(FRAME_NAMES):
            raise ManifestError(
                f"source {source_label}: directory must contain exactly frame_000.png through frame_016.png"
            )

        for frame_index, frame_name in enumerate(FRAME_NAMES):
            raw, digest = _read_source_frame(source_dir / frame_name, source_label, frame_name)
            del raw
            frame_rows.append(
                {
                    "blind_clip_id": clip_id,
                    "frame_index": frame_index,
                    "image_file": _image_relpath(clip_id, frame_index),
                    "sha256": digest,
                    "width": WIDTH,
                    "height": HEIGHT,
                }
            )
        blind_to_source.append(
            {
                "blind_clip_id": clip_id,
                "source_label": source_label,
                "source_dir": str(source_dir),
            }
        )

    frame_rows.sort(key=lambda row: (row["blind_clip_id"], row["frame_index"]))
    blind_to_source.sort(key=lambda row: row["blind_clip_id"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "packet_id": packet_id,
        "expected_clip_count": CLIP_COUNT,
        "expected_frames_per_clip": FRAMES_PER_CLIP,
        "pixel_format": PIXEL_FORMAT,
        "width": WIDTH,
        "height": HEIGHT,
        "frames": frame_rows,
    }
    private_map = {
        "schema_version": PRIVATE_MAP_SCHEMA,
        "packet_id": packet_id,
        "run_report_sha256": report_sha256,
        "run_report_path": str(report_path),
        "blind_to_source": blind_to_source,
    }
    _validate_manifest_object(manifest)
    _validate_private_map_object(private_map, manifest)
    return manifest, private_map


def _validate_private_map_object(
    private_map: dict[str, Any], manifest: dict[str, Any]
) -> None:
    _expect_keys(private_map, PRIVATE_MAP_KEYS, "private mapping")
    if private_map["schema_version"] != PRIVATE_MAP_SCHEMA:
        raise ManifestError("private mapping: unsupported schema_version")
    if private_map["packet_id"] != manifest["packet_id"]:
        raise ManifestError("private mapping packet_id differs from the manifest")
    report_sha = private_map["run_report_sha256"]
    if not isinstance(report_sha, str) or not SHA256_RE.fullmatch(report_sha):
        raise ManifestError("private mapping has an invalid run-report hash")
    report_path = private_map["run_report_path"]
    if not isinstance(report_path, str) or not Path(report_path).is_absolute():
        raise ManifestError("private mapping run-report path must be absolute")
    entries = private_map["blind_to_source"]
    if not isinstance(entries, list) or len(entries) != CLIP_COUNT:
        raise ManifestError("private mapping must contain exactly six blind-to-source entries")
    manifest_ids = {row["blind_clip_id"] for row in manifest["frames"]}
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    previous_id = ""
    for index, entry in enumerate(entries):
        where = f"private mapping.blind_to_source[{index}]"
        _expect_keys(entry, PRIVATE_MAP_ENTRY_KEYS, where)
        blind_id = _uuid4(entry["blind_clip_id"], f"{where}.blind_clip_id")
        label = entry["source_label"]
        source_dir = entry["source_dir"]
        if not isinstance(label, str) or label not in EXPECTED_RUN_LABELS:
            raise ManifestError(f"{where}.source_label is not a registered source label")
        if not isinstance(source_dir, str) or not Path(source_dir).is_absolute():
            raise ManifestError(f"{where}.source_dir must be an absolute private provenance path")
        if blind_id in seen_ids or label in seen_labels or blind_id < previous_id:
            raise ManifestError("private mapping contains duplicate or unsorted entries")
        seen_ids.add(blind_id)
        seen_labels.add(label)
        previous_id = blind_id
    if seen_ids != manifest_ids or seen_labels != EXPECTED_RUN_LABELS:
        raise ManifestError("private mapping must bijectively cover the blind IDs and six run labels")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_paths(
    manifest_path: Path, private_map_path: Path, source_dirs: list[Path]
) -> tuple[Path, Path]:
    manifest_path = Path(os.path.abspath(manifest_path.expanduser()))
    private_map_path = Path(os.path.abspath(private_map_path.expanduser()))
    if manifest_path.name != "frame_manifest.json":
        raise ManifestError("manifest output must be named frame_manifest.json")
    if private_map_path.name != "blind_run_map.json":
        raise ManifestError("private mapping output must be named blind_run_map.json")
    try:
        manifest_parent = manifest_path.parent.resolve(strict=True)
        private_map_parent = private_map_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ManifestError("output parent directories must already exist") from exc
    # Compare real paths, not spellings: symlinked ancestors must not make two
    # outputs appear separate when they actually share a directory or source tree.
    manifest_path = manifest_parent / manifest_path.name
    private_map_path = private_map_parent / private_map_path.name
    if manifest_path == private_map_path:
        raise ManifestError("manifest and private mapping must be different files")
    if not manifest_path.parent.is_dir() or not private_map_path.parent.is_dir():
        raise ManifestError("output parent directories must already exist")
    if manifest_path.exists() or private_map_path.exists():
        raise ManifestError("refusing to overwrite an existing evaluator artifact")
    for output_path in (manifest_path, private_map_path):
        for source_dir in source_dirs:
            if _is_within(output_path, source_dir):
                raise ManifestError("evaluator outputs must be outside all source frame directories")
    if manifest_path.parent == private_map_path.parent:
        raise ManifestError("blind manifest and private run mapping must use separate directories")
    if _is_within(private_map_path.parent, manifest_path.parent) or _is_within(
        manifest_path.parent, private_map_path.parent
    ):
        raise ManifestError("blind manifest and private mapping directories must not be nested")

    if os.name != "posix":
        raise ManifestError("owner-only private-map storage requires a POSIX filesystem")
    private_parent = private_map_path.parent.stat()
    if private_parent.st_uid != os.getuid() or stat.S_IMODE(private_parent.st_mode) != 0o700:
        raise ManifestError("private-map parent must be owned by the current user and mode 0700")
    return manifest_path, private_map_path


def _write_new_private_file(path: Path, content: bytes) -> tuple[int, int]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ManifestError("artifact writes require O_NOFOLLOW support")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ManifestError(f"cannot create evaluator artifact without overwrite: {path.name}") from exc
    identity: tuple[int, int] | None = None
    try:
        metadata = os.fstat(fd)
        identity = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            fd = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if identity is None:
            raise ManifestError("could not verify newly created artifact identity")
        return identity
    except Exception:
        if fd >= 0:
            os.close(fd)
        if identity is not None:
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == identity:
                    path.unlink()
            except OSError:
                pass
        raise


def _remove_if_same_file(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
        if (current.st_dev, current.st_ino) == identity:
            path.unlink()
    except OSError:
        pass


def build_manifest(
    source_specs: list[tuple[str, str]],
    run_report_path: Path,
    manifest_out: Path,
    private_map_out: Path,
) -> tuple[str, list[dict[str, Any]]]:
    if len(source_specs) != CLIP_COUNT or {label for label, _ in source_specs} != EXPECTED_RUN_LABELS:
        raise ManifestError("build requires the exact six registered mirrored-arm source directories")
    expected_run_report_sha256 = _registered_run_report_sha256()
    report_sha256, resolved_report = _verify_full_run_report(
        run_report_path, source_specs, expected_run_report_sha256
    )
    manifest, private_map = _build_documents(source_specs, report_sha256, resolved_report)
    source_dirs = [Path(path).expanduser().resolve(strict=True) for _, path in source_specs]
    manifest_path, private_map_path = _validate_output_paths(
        manifest_out, private_map_out, source_dirs
    )
    manifest_bytes = _serialize_manifest(manifest)
    private_bytes = (
        json.dumps(private_map, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=False)
        + "\n"
    ).encode("utf-8")
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        # The unblind mapping is written only to the separately protected path.
        created.append((private_map_path, _write_new_private_file(private_map_path, private_bytes)))
        created.append((manifest_path, _write_new_private_file(manifest_path, manifest_bytes)))
    except Exception:
        for path, identity in reversed(created):
            _remove_if_same_file(path, identity)
        raise
    return _sha256(manifest_bytes), _duplicate_hash_groups(manifest["frames"])


def _load_and_validate_manifest(path: Path) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    manifest, raw = _load_strict_json(path)
    rows, _ = _validate_manifest_object(manifest)
    if raw != _serialize_manifest(manifest):
        raise ManifestError("manifest bytes are not in the registered canonical serialization")
    return manifest, raw, rows


def _validate_integrity_receipt(
    path: Path, manifest: dict[str, Any], manifest_bytes: bytes, archive_bytes: bytes
) -> None:
    receipt, _ = _load_strict_json(path)
    if not isinstance(receipt, dict) or set(receipt) != set(INTEGRITY_KEYS):
        raise ManifestError("packet_integrity: keys do not match the registered schema")
    if receipt["schema_version"] != INTEGRITY_SCHEMA:
        raise ManifestError("packet_integrity: unsupported schema_version")
    if receipt["packet_id"] != manifest["packet_id"]:
        raise ManifestError("packet_integrity: packet_id mismatch")
    if not _is_int(receipt["frame_count"]) or receipt["frame_count"] != 102:
        raise ManifestError("packet_integrity: frame_count must be integer 102")
    manifest_sha = receipt["frame_manifest_sha256"]
    archive_sha = receipt["annotator_archive_sha256"]
    if not isinstance(manifest_sha, str) or not SHA256_RE.fullmatch(manifest_sha):
        raise ManifestError("packet_integrity: invalid frame_manifest_sha256")
    if not isinstance(archive_sha, str) or not SHA256_RE.fullmatch(archive_sha):
        raise ManifestError("packet_integrity: invalid annotator_archive_sha256")
    if manifest_sha != _sha256(manifest_bytes):
        raise ManifestError("packet_integrity: manifest hash mismatch")
    if archive_sha != _sha256(archive_bytes):
        raise ManifestError("packet_integrity: archive hash mismatch")


def _zip_member_is_regular(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    return file_type in (0, stat.S_IFREG)


def _validate_archive_members(archive_bytes: bytes, rows: list[dict[str, Any]]) -> None:
    expected_paths = {row["image_file"] for row in rows}
    expected_dirs = set(ALLOWED_ZIP_DIRS)
    clip_ids = {row["blind_clip_id"] for row in rows}
    expected_dirs.update("images/" + clip_id + "/" for clip_id in clip_ids)
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes), "r") as archive:
            if archive.comment:
                raise ManifestError("annotator ZIP comment is not allowed")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ManifestError("annotator ZIP contains duplicate member names")
            regular_files: set[str] = set()
            for info in infos:
                if info.comment or info.extra:
                    raise ManifestError("annotator ZIP member comments/extra fields are not allowed")
                name = info.filename
                pure = PurePosixPath(name)
                if pure.is_absolute() or "\\" in name or "\x00" in name:
                    raise ManifestError("annotator ZIP contains an unsafe member path")
                if any(part in {"", ".", ".."} for part in name.rstrip("/").split("/")):
                    raise ManifestError("annotator ZIP contains a non-normalized member path")
                if info.is_dir():
                    if name not in expected_dirs:
                        raise ManifestError("annotator ZIP contains an unexpected directory")
                    mode = (info.external_attr >> 16) & 0xFFFF
                    if stat.S_IFMT(mode) not in (0, stat.S_IFDIR):
                        raise ManifestError("annotator ZIP directory member has invalid file type")
                    continue
                if not _zip_member_is_regular(info):
                    raise ManifestError("annotator ZIP contains a non-regular file member")
                if name not in expected_paths:
                    raise ManifestError("annotator ZIP contains an extra/unlisted file")
                regular_files.add(name)
            if regular_files != expected_paths:
                raise ManifestError("annotator ZIP file members do not equal the manifest paths")

            for row in rows:
                try:
                    payload = archive.read(row["image_file"])
                except (KeyError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
                    raise ManifestError("an expected PNG could not be read from the ZIP") from exc
                if _sha256(payload) != row["sha256"]:
                    raise ManifestError("PNG SHA-256 does not match its manifest row")
                width, height = _inspect_png(payload, row["image_file"])
                if width != row["width"] or height != row["height"]:
                    raise ManifestError("decoded PNG dimensions do not match its manifest row")
    except ManifestError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ManifestError("annotator archive is unreadable or invalid") from exc


def validate_archive(
    manifest_path: Path, archive_path: Path, integrity_path: Path
) -> tuple[str, list[dict[str, Any]]]:
    if manifest_path.name != "frame_manifest.json":
        raise ManifestError("manifest input must be named frame_manifest.json")
    if integrity_path.name != "packet_integrity.json":
        raise ManifestError("integrity input must be named packet_integrity.json")
    if archive_path.suffix.lower() != ".zip":
        raise ManifestError("archive input must have a .zip suffix")
    manifest, manifest_bytes, rows = _load_and_validate_manifest(manifest_path)
    archive_bytes = _read_regular_nofollow(archive_path, "finalized annotator archive")
    _validate_integrity_receipt(integrity_path, manifest, manifest_bytes, archive_bytes)
    _validate_archive_members(archive_bytes, rows)
    return _sha256(archive_bytes), _duplicate_hash_groups(rows)


def _source_specs(values: list[str]) -> list[tuple[str, str]]:
    return [_parse_source_spec(value) for value in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser(
        "build",
        help="read six completed frame directories; write only manifest and separate private mapping",
    )
    build_parser.add_argument(
        "--source-run",
        action="append",
        required=True,
        metavar="LABEL=FRAME_DIRECTORY",
        help="repeat exactly six times; LABEL is retained only in the owner-only mapping",
    )
    build_parser.add_argument("--run-report", required=True, type=Path)
    build_parser.add_argument("--manifest-out", required=True, type=Path)
    build_parser.add_argument("--private-map-out", required=True, type=Path)

    validate_parser = commands.add_parser(
        "validate",
        help="read-only validation of a finalized archive, manifest, and detached integrity receipt",
    )
    validate_parser.add_argument("--manifest", required=True, type=Path)
    validate_parser.add_argument("--archive", required=True, type=Path)
    validate_parser.add_argument("--integrity", required=True, type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest_sha, duplicate_groups = build_manifest(
                _source_specs(args.source_run),
                args.run_report,
                args.manifest_out,
                args.private_map_out,
            )
            print(f"Evaluator manifest created; SHA-256={manifest_sha}")
            print("The separate owner-only mapping is not part of the blind packet or scorer input.")
            print("No image files or archive were created; run was not scored.")
            for group in duplicate_groups:
                print(f"EXACT_FRAME_DUPLICATE_DIAGNOSTIC: {group}")
            return 0
        archive_sha, duplicate_groups = validate_archive(args.manifest, args.archive, args.integrity)
        print(f"Manifest/archive integrity validation passed; archive SHA-256={archive_sha}")
        print("This validates packet structure only; it creates no score or background classification.")
        for group in duplicate_groups:
            print(f"EXACT_FRAME_DUPLICATE_DIAGNOSTIC: {group}")
        return 0
    except (ManifestError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
