"""Fail-closed validation for a manually reviewed, no-paid DiTraj admission receipt.

This module does not query Kaggle billing, quota, or account state. It validates
that an operator's fresh attestation is complete and matches the intended arm,
the locally pinned notebook hash, and the exact runner bytes before any runner
imports model, CUDA, or upstream-patching code.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "latentgate.ditraj_free_admission.v1"
DESIGNATED_KAGGLE_ACCOUNT = "bvyashwantkumar"
MAX_RECEIPT_AGE = timedelta(minutes=15)
MAX_FUTURE_SKEW = timedelta(minutes=2)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("receipt checked_at_utc must be an ISO-8601 UTC timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("receipt checked_at_utc must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("receipt checked_at_utc must include UTC timezone information")
    return parsed.astimezone(timezone.utc)


def validate_admission_receipt(
    receipt_path: Path,
    *,
    intended_arm: str,
    expected_notebook_sha256: str,
    runner_path: Path | None = None,
    expected_runner_sha256: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a manual receipt; returns a JSON-safe, explicit provenance record.

    The notebook file is not available to a Kaggle kernel for independent
    hashing. Therefore `expected_notebook_sha256` must be transcribed from the
    reviewed local source pin; the receipt records that operator check. During
    pre-install notebook setup, the runner hash is compared with its pinned
    local hash. At CLI invocation, the executing runner file is re-hashed.
    """
    if not intended_arm or not intended_arm.strip():
        raise ValueError("an explicit intended arm is required")
    if not SHA256_RE.fullmatch(expected_notebook_sha256):
        raise ValueError("expected notebook SHA-256 must be 64 lowercase hex characters")
    if not receipt_path.is_file():
        raise ValueError(f"no admission receipt at {receipt_path}; execution remains blocked")
    if expected_runner_sha256 is not None and not SHA256_RE.fullmatch(expected_runner_sha256):
        raise ValueError("expected runner SHA-256 must be 64 lowercase hex characters")
    if runner_path is None and expected_runner_sha256 is None:
        raise ValueError("an expected runner hash or runner file is required")
    if runner_path is not None and not runner_path.is_file():
        raise ValueError(f"runner source is missing at {runner_path}; execution remains blocked")

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("admission receipt is unreadable or invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("admission receipt must be a JSON object")

    required = {
        "schema",
        "account",
        "checked_at_utc",
        "notebook_sha256",
        "runner_sha256",
        "intended_arm",
        "free_t4_admission_confirmed",
        "paid_prompt_seen",
        "admission_provenance",
        "evidence_sha256",
    }
    missing = sorted(required - receipt.keys())
    if missing:
        raise ValueError(f"admission receipt is missing required fields: {', '.join(missing)}")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise ValueError("admission receipt schema is not supported")
    if receipt["account"] != DESIGNATED_KAGGLE_ACCOUNT:
        raise ValueError("receipt account does not match the designated Kaggle account")
    if receipt["intended_arm"] != intended_arm:
        raise ValueError("receipt intended_arm does not match this run")
    if receipt["free_t4_admission_confirmed"] is not True:
        raise ValueError("fresh free T4 admission is not attested; execution remains blocked")
    if receipt["paid_prompt_seen"] is not False:
        raise ValueError("a paid/upgrade prompt was seen or not explicitly ruled out")
    if receipt["notebook_sha256"] != expected_notebook_sha256:
        raise ValueError("receipt notebook SHA-256 does not match the reviewed local notebook pin")
    if not isinstance(receipt["runner_sha256"], str) or not SHA256_RE.fullmatch(receipt["runner_sha256"]):
        raise ValueError("receipt runner SHA-256 must be 64 lowercase hex characters")

    actual_runner_sha256 = sha256_file(runner_path) if runner_path is not None else expected_runner_sha256
    if receipt["runner_sha256"] != actual_runner_sha256:
        raise ValueError("receipt runner SHA-256 does not match the executing runner bytes")
    if not isinstance(receipt["evidence_sha256"], str) or not SHA256_RE.fullmatch(receipt["evidence_sha256"]):
        raise ValueError("receipt evidence_sha256 must identify the reviewed admission evidence")
    provenance = receipt["admission_provenance"]
    if not isinstance(provenance, str) or len(provenance.strip()) < 12:
        raise ValueError("receipt admission_provenance must describe the manual UI evidence")

    checked_at = _parse_utc(receipt["checked_at_utc"])
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = now_utc - checked_at
    if age < -MAX_FUTURE_SKEW:
        raise ValueError("receipt check time is implausibly in the future")
    if age > MAX_RECEIPT_AGE:
        raise ValueError("admission receipt is older than 15 minutes; execution remains blocked")

    return {
        "verification": "manual_operator_attestation_no_provider_quota_api",
        "account": DESIGNATED_KAGGLE_ACCOUNT,
        "checked_at_utc": checked_at.isoformat().replace("+00:00", "Z"),
        "notebook_sha256": expected_notebook_sha256,
        "runner_sha256": actual_runner_sha256,
        "intended_arm": intended_arm,
        "free_t4_admission_confirmed_by_operator": True,
        "paid_prompt_seen_by_operator": False,
        "admission_provenance": provenance.strip(),
        "evidence_sha256": receipt["evidence_sha256"],
        "notebook_hash_verification": "operator_verified_local_pin_not_kernel_rehashed",
        "runner_hash_verification": (
            "rehash_executing_file" if runner_path is not None else "matched_pinned_hash_before_install"
        ),
        "provider_quota_api_queried": False,
    }
