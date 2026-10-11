"""Restore the submission assets from a Hugging Face model repository.

The Hugging Face repository must preserve the same relative paths as this
submission directory. After downloading, this script verifies every asset
listed in ASSETS.json by size and SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def asset_records() -> list[dict]:
    payload = json.loads((ROOT / "ASSETS.json").read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("assets", "files"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if isinstance(payload, dict) and all(isinstance(value, list) for value in payload.values()):
        records: list[dict] = []
        for task, task_assets in payload.items():
            for asset in task_assets:
                record = dict(asset)
                record["path"] = f"{task}/{relative_path(record)}"
                records.append(record)
        return records
    raise ValueError("ASSETS.json does not contain an assets/files list")


def relative_path(record: dict) -> str:
    for key in ("path", "relative_path", "file"):
        value = record.get(key)
        if value:
            return str(value).replace("\\", "/")
    raise ValueError(f"Asset record has no path: {record}")


def expected_hash(record: dict) -> str | None:
    for key in ("sha256", "sha256sum", "hash"):
        value = record.get(key)
        if value:
            return str(value).lower().removeprefix("sha256:")
    return None


def verify(records: list[dict]) -> list[str]:
    errors: list[str] = []
    for record in records:
        relative = relative_path(record)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            errors.append(f"outside submission root: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        size = record.get("size") or record.get("size_bytes") or record.get("bytes")
        if size is not None and path.stat().st_size != int(size):
            errors.append(f"size mismatch: {relative}")
        checksum = expected_hash(record)
        if checksum and digest(path) != checksum:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Hugging Face model repo, e.g. org/nyx-robosyn-2026")
    parser.add_argument("--revision", default="main", help="Pinned branch, tag, or commit")
    parser.add_argument("--token", default=None, help="Token for a private repository; prefer HF_TOKEN")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    records = asset_records()
    if not args.verify_only:
        if not args.repo_id:
            parser.error("--repo-id is required unless --verify-only is used")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub before downloading assets") from exc
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            local_dir=ROOT,
            token=args.token,
        )

    errors = verify(records)
    print(json.dumps({"assets_checked": len(records), "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

