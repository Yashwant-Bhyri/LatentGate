#!/usr/bin/env python3
"""Unverified Gate 1 custody/hash draft; never parses CSV cells or prints file data."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Iterator


HOME = Path.home()
BASE = HOME / "Library" / "Application Support" / "LatentGate" / "Gate1"
ORIGINALS = BASE / "originals"
STAGING = BASE / "staging"
RECEIPTS = BASE / "receipts"
PRIVATE_ROSTER = BASE / "private_roster"

EXPECTED_PACKETS = {
    "synthetic_reliability/rater_01.zip": "0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8",
    "synthetic_reliability/rater_02.zip": "8a660acb5bd77855b82c13b27cc951f8d2e4d6131e38e0144e83638e2f92e7ff",
}
ATTESTATION_VERSION = "gate1_independence_v1"
ATTESTATION_TEXT = (
    "I completed this packet independently, did not inspect another submission, "
    "and did not coordinate labels."
)
MAX_ENVELOPE_BYTES = 12_000_000
MAX_CSV_BYTES = 10_000_000
MAX_PRIVATE_JSON_BYTES = 64_000
CHUNK_BYTES = 1_048_576
OPAQUE_ID_RE = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# Darwin <sys/acl.h> values. The source is Apple Libc's public header:
# https://github.com/apple-oss-distributions/Libc/blob/main/include/sys/acl.h
ACL_TYPE_EXTENDED = 0x00000100
ACL_FIRST_ENTRY = 0
ACL_NEXT_ENTRY = -1
ACL_EXTENDED_ALLOW = 1
ACL_EXTENDED_DENY = 2


class PreflightFailure(Exception):
    """An intentionally detail-free fail-closed condition."""


def require(condition: bool) -> None:
    if not condition:
        raise PreflightFailure


def _private_file_mode(mode: int) -> bool:
    return stat.S_IMODE(mode) in {0o400, 0o600}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in obj)
        obj[key] = value
    return obj


_ACL_FUNCTIONS: tuple[Any, Any, Any, Any] | None = None


def _darwin_acl_functions() -> tuple[Any, Any, Any, Any]:
    global _ACL_FUNCTIONS
    require(sys.platform == "darwin")
    if _ACL_FUNCTIONS is None:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            get_fd = libc.acl_get_fd_np
            get_entry = libc.acl_get_entry
            get_tag_type = libc.acl_get_tag_type
            free_acl = libc.acl_free
            get_fd.argtypes = (ctypes.c_int, ctypes.c_int)
            get_fd.restype = ctypes.c_void_p
            get_entry.argtypes = (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            )
            get_entry.restype = ctypes.c_int
            get_tag_type.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
            get_tag_type.restype = ctypes.c_int
            free_acl.argtypes = (ctypes.c_void_p,)
            free_acl.restype = ctypes.c_int
        except (AttributeError, OSError, TypeError):
            raise PreflightFailure from None
        _ACL_FUNCTIONS = (get_fd, get_entry, get_tag_type, free_acl)
    return _ACL_FUNCTIONS


def _acl_summary(fd: int) -> tuple[bool, bool]:
    """Return (has_entries, has_allow) from the ACL attached to this fd."""
    get_fd, get_entry, get_tag_type, free_acl = _darwin_acl_functions()
    ctypes.set_errno(0)
    acl = get_fd(fd, ACL_TYPE_EXTENDED)
    require(acl is not None)
    try:
        entry = ctypes.c_void_p()
        ctypes.set_errno(0)
        result = get_entry(acl, ACL_FIRST_ENTRY, ctypes.byref(entry))
        if result == -1:
            # On a valid ACL and valid FIRST_ENTRY selector, EINVAL means the
            # ACL has no entries. Every other error is a fail-closed result.
            require(ctypes.get_errno() == errno.EINVAL)
            return False, False
        require(result == 0 and entry.value is not None)

        has_allow = False
        while True:
            tag = ctypes.c_int()
            require(get_tag_type(entry, ctypes.byref(tag)) == 0)
            require(tag.value in {ACL_EXTENDED_ALLOW, ACL_EXTENDED_DENY})
            has_allow = has_allow or tag.value == ACL_EXTENDED_ALLOW

            ctypes.set_errno(0)
            result = get_entry(acl, ACL_NEXT_ENTRY, ctypes.byref(entry))
            if result == -1:
                # Darwin reports end-of-list as EINVAL after the final entry.
                require(ctypes.get_errno() == errno.EINVAL)
                break
            require(result == 0 and entry.value is not None)
        return True, has_allow
    finally:
        require(free_acl(acl) == 0)


def _require_no_acl(fd: int) -> None:
    has_entries, _ = _acl_summary(fd)
    require(not has_entries)


def _require_no_allow_acl(fd: int) -> None:
    """Permit deny-only ancestor ACLs, but never an ACL grant."""
    _, has_allow = _acl_summary(fd)
    require(not has_allow)


def _check_private_directory_fd(fd: int) -> os.stat_result:
    info = os.fstat(fd)
    require(stat.S_ISDIR(info.st_mode))
    require(info.st_uid == os.getuid())
    require(stat.S_IMODE(info.st_mode) == 0o700)
    _require_no_acl(fd)
    return info


@contextmanager
def _open_owner_only_directories() -> Iterator[dict[str, int]]:
    """Open the custody tree component-by-component and retain its dirfds.

    Descriptor-relative access prevents pathname redirection after opening.
    It does not lock the owner account against concurrent same-UID mutation.
    """
    require(sys.platform == "darwin")
    require(hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"))
    require(os.open in os.supports_dir_fd)
    require(os.listdir in os.supports_fd)
    require(os.stat in os.supports_dir_fd and os.stat in os.supports_follow_symlinks)
    require(HOME.is_absolute() and len(HOME.parts) > 1)
    require(not any(part in {"", ".", ".."} for part in HOME.parts[1:]))

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    opened: list[tuple[int, os.stat_result, bool]] = []
    descriptors: list[int] = []
    retained: dict[str, int] = {}
    try:
        root_fd = os.open(os.sep, directory_flags)
        descriptors.append(root_fd)
        root_info = os.fstat(root_fd)
        opened.append((root_fd, root_info, False))
        require(stat.S_ISDIR(root_info.st_mode))
        require(root_info.st_uid == 0)
        require((stat.S_IMODE(root_info.st_mode) & 0o022) == 0)
        _require_no_allow_acl(root_fd)

        current_fd = root_fd
        current_path = Path(os.sep)
        for component in HOME.parts[1:]:
            current_path = current_path / component
            child_fd = os.open(component, directory_flags, dir_fd=current_fd)
            descriptors.append(child_fd)
            info = os.fstat(child_fd)
            opened.append((child_fd, info, False))
            require(stat.S_ISDIR(info.st_mode))
            if current_path == HOME:
                require(info.st_uid == os.getuid())
                require(stat.S_IMODE(info.st_mode) == 0o750)
            else:
                require(info.st_uid == 0)
                require((stat.S_IMODE(info.st_mode) & 0o022) == 0)
            _require_no_allow_acl(child_fd)
            current_fd = child_fd

        current_path = HOME
        for component in ("Library", "Application Support", "LatentGate", "Gate1"):
            current_path = current_path / component
            child_fd = os.open(component, directory_flags, dir_fd=current_fd)
            descriptors.append(child_fd)
            info = os.fstat(child_fd)
            is_private = component == "Gate1"
            opened.append((child_fd, info, is_private))
            require(stat.S_ISDIR(info.st_mode))
            require(info.st_uid == os.getuid())
            if component in {"Library", "Application Support", "Gate1"}:
                require(stat.S_IMODE(info.st_mode) == 0o700)
            else:
                require((stat.S_IMODE(info.st_mode) & 0o022) == 0)
            _require_no_allow_acl(child_fd)
            if is_private:
                _require_no_acl(child_fd)
                retained["base"] = child_fd
            current_fd = child_fd

        for name in ("originals", "staging", "receipts", "private_roster"):
            current_path = BASE / name
            child_fd = os.open(name, directory_flags, dir_fd=current_fd)
            descriptors.append(child_fd)
            info = os.fstat(child_fd)
            opened.append((child_fd, info, True))
            _check_private_directory_fd(child_fd)
            _require_no_allow_acl(child_fd)
            retained[name] = child_fd

        require(set(retained) == {"base", "originals", "staging", "receipts", "private_roster"})
        yield retained
    finally:
        try:
            for fd, before, strict_acl in opened:
                after = os.fstat(fd)
                require(stat.S_ISDIR(after.st_mode))
                require(before.st_dev == after.st_dev and before.st_ino == after.st_ino)
                require(before.st_uid == after.st_uid)
                require(stat.S_IMODE(before.st_mode) == stat.S_IMODE(after.st_mode))
                _require_no_allow_acl(fd)
                if strict_acl:
                    _require_no_acl(fd)
        finally:
            for fd in reversed(descriptors):
                os.close(fd)


@contextmanager
def _open_private_regular(
    directory_fd: int, name: str, maximum: int
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    require(name not in {"", ".", ".."})
    require("/" not in name and "\\" not in name)
    require(hasattr(os, "O_NOFOLLOW"))
    _check_private_directory_fd(directory_fd)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    stream: BinaryIO | None = None
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode))
        require(before.st_uid == os.getuid())
        require(_private_file_mode(before.st_mode))
        require(0 <= before.st_size <= maximum)
        _require_no_acl(fd)
        stream = os.fdopen(fd, "rb", closefd=True)
        fd = -1
        yield stream, before
    finally:
        if stream is not None:
            stream.close()
        elif fd >= 0:
            os.close(fd)


def _unchanged(before: os.stat_result, after: os.stat_result, bytes_read: int) -> None:
    require(stat.S_ISREG(after.st_mode))
    require(before.st_dev == after.st_dev and before.st_ino == after.st_ino)
    require(before.st_size == after.st_size == bytes_read)
    require(before.st_mtime_ns == after.st_mtime_ns)
    require(before.st_ctime_ns == after.st_ctime_ns)


def _read_private_json(directory_fd: int, name: str) -> Any:
    with _open_private_regular(
        directory_fd, name, MAX_PRIVATE_JSON_BYTES
    ) as (stream, before):
        data = stream.read(MAX_PRIVATE_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
        require(len(data) <= MAX_PRIVATE_JSON_BYTES)
        _unchanged(before, after, len(data))
    return json.loads(
        data.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
    )


def _hash_stream(stream: BinaryIO, maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        block = stream.read(min(CHUNK_BYTES, maximum - total + 1))
        if not block:
            break
        total += len(block)
        require(total <= maximum)
        digest.update(block)
    return digest.hexdigest(), total


def _hash_private_file(
    directory_fd: int, name: str, maximum: int
) -> tuple[str, int]:
    with _open_private_regular(directory_fd, name, maximum) as (stream, before):
        digest, total = _hash_stream(stream, maximum)
        after = os.fstat(stream.fileno())
        _unchanged(before, after, total)
    return digest, total


def _regular_names(directory_fd: int) -> set[str]:
    _check_private_directory_fd(directory_fd)
    names: set[str] = set()
    for name in os.listdir(directory_fd):
        require(name not in {"", ".", ".."})
        require("/" not in name and "\\" not in name)
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        require(stat.S_ISREG(info.st_mode))
        require(info.st_uid == os.getuid())
        require(_private_file_mode(info.st_mode))
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            opened_info = os.fstat(fd)
            require(stat.S_ISREG(opened_info.st_mode))
            require(opened_info.st_dev == info.st_dev and opened_info.st_ino == info.st_ino)
            require(opened_info.st_uid == os.getuid())
            require(_private_file_mode(opened_info.st_mode))
            _require_no_acl(fd)
        finally:
            os.close(fd)
        names.add(name)
    _check_private_directory_fd(directory_fd)
    return names


def _validate_roster(value: Any) -> list[dict[str, str]]:
    require(isinstance(value, dict))
    require(set(value) == {"schema", "assignments"})
    require(value["schema"] == "gate1_assignment_roster_v1")
    assignments = value["assignments"]
    require(isinstance(assignments, list) and len(assignments) == 2)
    cleaned: list[dict[str, str]] = []
    for assignment in assignments:
        require(isinstance(assignment, dict))
        require(set(assignment) == {"rater_token", "assignee_ref", "packet_id", "packet_sha256"})
        token = assignment["rater_token"]
        assignee = assignment["assignee_ref"]
        packet_id = assignment["packet_id"]
        packet_hash = assignment["packet_sha256"]
        require(isinstance(token, str) and TOKEN_RE.fullmatch(token) is not None)
        require(
            isinstance(assignee, str)
            and 1 <= len(assignee) <= 256
            and assignee == assignee.strip()
        )
        require(isinstance(packet_id, str) and packet_id in EXPECTED_PACKETS)
        require(isinstance(packet_hash, str) and SHA256_RE.fullmatch(packet_hash) is not None)
        require(packet_hash == EXPECTED_PACKETS[packet_id])
        cleaned.append(assignment)
    require(len({row["rater_token"] for row in cleaned}) == 2)
    require(len({row["assignee_ref"] for row in cleaned}) == 2)
    require({row["packet_id"] for row in cleaned} == set(EXPECTED_PACKETS))
    return cleaned


def _validate_receipt(value: Any, filename: str) -> dict[str, Any]:
    require(isinstance(value, dict))
    required = {
        "schema", "receipt_id", "packet_id", "packet_sha256", "rater_token",
        "received_utc", "received_format", "received_file_sha256",
        "independence_attestation_version", "independence_attestation",
        "scored_csv_sha256",
    }
    require(set(value) in (required, required | {"csv_member_ordinal"}))
    require(value["schema"] == "gate1_submission_receipt_v1")
    receipt_id = value["receipt_id"]
    require(isinstance(receipt_id, str) and OPAQUE_ID_RE.fullmatch(receipt_id) is not None)
    require(filename == f"{receipt_id}.json")
    packet_id = value["packet_id"]
    require(isinstance(packet_id, str) and packet_id in EXPECTED_PACKETS)
    packet_hash = value["packet_sha256"]
    require(isinstance(packet_hash, str) and packet_hash == EXPECTED_PACKETS[packet_id])
    token = value["rater_token"]
    require(isinstance(token, str) and TOKEN_RE.fullmatch(token) is not None)
    received_format = value["received_format"]
    require(received_format in {"csv", "zip"})
    require(value["independence_attestation_version"] == ATTESTATION_VERSION)
    require(value["independence_attestation"] == ATTESTATION_TEXT)
    try:
        received_time = datetime.fromisoformat(value["received_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise PreflightFailure from None
    require(received_time.tzinfo is not None and received_time.utcoffset() is not None)
    require(received_time.utcoffset().total_seconds() == 0)
    for field in ("received_file_sha256", "scored_csv_sha256"):
        require(isinstance(value[field], str) and SHA256_RE.fullmatch(value[field]) is not None)
    if received_format == "zip":
        require(value.get("csv_member_ordinal") == 1)
    else:
        require("csv_member_ordinal" not in value)
    return value


def _safe_zip_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    require(not info.is_dir())
    require(name.lower().endswith(".csv"))
    require("/" not in name and "\\" not in name and ":" not in name)
    require(not posix.is_absolute() and not windows.is_absolute())
    require(".." not in posix.parts and ".." not in windows.parts)
    upper_name = name.upper()
    require("UNBLIND" not in upper_name and "PRIVATE_KEY" not in upper_name)
    # Bit 0 marks traditional encryption; bit 6 marks strong encryption.
    require((info.flag_bits & (0x1 | 0x40)) == 0)
    # Some DOS-origin entries signal directories in external attributes
    # without a trailing slash, which ZipInfo.is_dir() alone does not catch.
    require((info.external_attr & 0x10) == 0)
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    require(file_type in (0, stat.S_IFREG))
    require(0 <= info.file_size <= MAX_CSV_BYTES)


def _hash_original_and_member(
    receipt: dict[str, Any], originals_fd: int, receipt_id: str
) -> tuple[str, str | None]:
    received_format = receipt["received_format"]
    maximum = MAX_ENVELOPE_BYTES if received_format == "zip" else MAX_CSV_BYTES
    original_name = f"{receipt_id}.envelope"
    with _open_private_regular(
        originals_fd, original_name, maximum
    ) as (stream, before):
        envelope_hash, envelope_bytes = _hash_stream(stream, maximum)
        member_hash: str | None = None
        if received_format == "zip":
            stream.seek(0)
            with zipfile.ZipFile(stream, "r") as archive:
                infos = archive.infolist()
                require(len(infos) == 1)
                info = infos[0]
                _safe_zip_member(info)
                with archive.open(info, "r") as member:
                    member_hash, member_bytes = _hash_stream(member, MAX_CSV_BYTES)
                require(member_bytes == info.file_size)
        after = os.fstat(stream.fileno())
        _unchanged(before, after, envelope_bytes)
    return envelope_hash, member_hash


def _preflight() -> None:
    with _open_owner_only_directories() as directory_fds:
        roster = _validate_roster(
            _read_private_json(
                directory_fds["private_roster"], "assignments.json"
            )
        )
        receipt_names = sorted(_regular_names(directory_fds["receipts"]))
        require(len(receipt_names) == 2)
        require(all(name.endswith(".json") for name in receipt_names))
        receipts = [
            _validate_receipt(
                _read_private_json(directory_fds["receipts"], name), name
            )
            for name in receipt_names
        ]
        require(len({receipt["receipt_id"] for receipt in receipts}) == 2)
        require(len({receipt["rater_token"] for receipt in receipts}) == 2)
        require({receipt["packet_id"] for receipt in receipts} == set(EXPECTED_PACKETS))

        roster_by_token = {row["rater_token"]: row for row in roster}
        require(set(roster_by_token) == {receipt["rater_token"] for receipt in receipts})
        for receipt in receipts:
            assignment = roster_by_token[receipt["rater_token"]]
            require(receipt["packet_id"] == assignment["packet_id"])
            require(receipt["packet_sha256"] == assignment["packet_sha256"])

        receipt_ids = {receipt["receipt_id"] for receipt in receipts}
        require(
            _regular_names(directory_fds["originals"])
            == {f"{receipt_id}.envelope" for receipt_id in receipt_ids}
        )
        require(
            _regular_names(directory_fds["staging"])
            == {f"{receipt_id}.csv" for receipt_id in receipt_ids}
        )
        require(
            _regular_names(directory_fds["receipts"])
            == {f"{receipt_id}.json" for receipt_id in receipt_ids}
        )
        require(
            _regular_names(directory_fds["private_roster"])
            == {"assignments.json"}
        )

        received_hashes: list[str] = []
        for receipt in receipts:
            receipt_id = receipt["receipt_id"]
            original_hash, member_hash = _hash_original_and_member(
                receipt, directory_fds["originals"], receipt_id
            )
            staged_hash, _ = _hash_private_file(
                directory_fds["staging"],
                f"{receipt_id}.csv",
                MAX_CSV_BYTES,
            )
            require(secrets.compare_digest(original_hash, receipt["received_file_sha256"]))
            require(secrets.compare_digest(staged_hash, receipt["scored_csv_sha256"]))
            if receipt["received_format"] == "csv":
                require(member_hash is None)
                require(secrets.compare_digest(original_hash, staged_hash))
            else:
                require(member_hash is not None)
                require(secrets.compare_digest(member_hash, receipt["scored_csv_sha256"]))
                require(secrets.compare_digest(member_hash, staged_hash))
            received_hashes.append(original_hash)
        require(len(set(received_hashes)) == 2)


def main() -> int:
    run_id = secrets.token_hex(16)
    passed = False
    try:
        require(len(sys.argv) == 1)
        _preflight()
        passed = True
    except (Exception, KeyboardInterrupt):
        passed = False
    sys.stdout.write(json.dumps({"run_id": run_id, "result": "PASS" if passed else "FAIL"}) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
