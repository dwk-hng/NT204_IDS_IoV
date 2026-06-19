from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from create_merged_7_percent_dataset import COLUMN_ALIASES


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "merged_intrusion_dataset_7_percent.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "merged_intrusion_dataset_7_percent_canonical.csv"


EXTRA_ALIASES = {
    # CIC-IDS-2017 includes this duplicated feature name in some exports.
    "Fwd Header Length.1": "Fwd Header Length",
}

LABEL_VALUE_ALIASES = {
    "BENIGN": "Benign",
}


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def canonical_column(name: Any) -> str:
    stripped = str(name).lstrip("\ufeff").strip()
    return EXTRA_ALIASES.get(stripped, COLUMN_ALIASES.get(stripped, stripped))


def is_empty(value: str) -> bool:
    stripped = value.strip()
    return stripped == "" or stripped.lower() in {"nan", "none", "null"}


def normalize_label(value: str) -> str:
    label = value.strip()
    if "," in label:
        label = label.rsplit(",", 1)[-1].strip()
    if label.lower() == "label":
        return ""
    return LABEL_VALUE_ALIASES.get(label, label)


def build_column_plan(
    header: list[str],
) -> tuple[list[str], list[int], list[tuple[int, list[int]]], OrderedDict[str, list[str]]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    raw_groups: OrderedDict[str, list[str]] = OrderedDict()

    for idx, raw_name in enumerate(header):
        canonical = canonical_column(raw_name)
        groups.setdefault(canonical, []).append(idx)
        raw_groups.setdefault(canonical, []).append(raw_name)

    output_columns = list(groups.keys())
    index_groups = list(groups.values())
    primary_indexes = [indexes[0] for indexes in index_groups]
    duplicate_index_groups = [
        (out_idx, indexes)
        for out_idx, indexes in enumerate(index_groups)
        if len(indexes) > 1
    ]
    duplicate_groups = OrderedDict(
        (name, raw_names)
        for name, raw_names in raw_groups.items()
        if len(raw_names) > 1
    )
    return output_columns, primary_indexes, duplicate_index_groups, duplicate_groups


def coalesce_row(
    row: list[str],
    primary_indexes: list[int],
    duplicate_index_groups: list[tuple[int, list[int]]],
    label_output_index: int,
) -> list[str]:
    output_row = [
        row[idx] if idx < len(row) else ""
        for idx in primary_indexes
    ]

    for out_idx, indexes in duplicate_index_groups:
        for idx in indexes:
            if idx >= len(row):
                continue
            value = row[idx]
            if is_empty(value):
                continue
            output_row[out_idx] = value
            break

    if label_output_index >= 0:
        output_row[label_output_index] = normalize_label(output_row[label_output_index])

    return output_row


def canonicalize_csv(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    overwrite: bool,
    progress_every: int,
    encoding: str,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows_written = 0
    rows_skipped_missing_label = 0
    malformed_rows = 0

    with input_path.open("r", newline="", encoding=encoding, errors="replace") as in_f:
        reader = csv.reader(in_f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {input_path}") from exc

        output_columns, primary_indexes, duplicate_index_groups, duplicate_groups = build_column_plan(header)
        label_output_index = output_columns.index("Label") if "Label" in output_columns else -1

        with output_path.open("w", newline="", encoding=encoding) as out_f:
            writer = csv.writer(out_f, lineterminator="\n")
            writer.writerow(output_columns)

            for row in reader:
                if len(row) != len(header):
                    malformed_rows += 1

                output_row = coalesce_row(
                    row,
                    primary_indexes,
                    duplicate_index_groups,
                    label_output_index,
                )
                if label_output_index >= 0 and is_empty(output_row[label_output_index]):
                    rows_skipped_missing_label += 1
                    continue

                writer.writerow(output_row)
                rows_written += 1

                if progress_every and rows_written % progress_every == 0:
                    elapsed = time.time() - started
                    rate = rows_written / max(elapsed, 1e-9)
                    print(f"rows={rows_written:,} elapsed={elapsed:.1f}s rate={rate:,.0f} rows/s")

    elapsed = time.time() - started
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "output": str(output_path),
        "input_columns": len(header),
        "output_columns": len(output_columns),
        "rows_written": rows_written,
        "rows_skipped_missing_label": rows_skipped_missing_label,
        "malformed_rows": malformed_rows,
        "elapsed_seconds": elapsed,
        "duplicate_groups": duplicate_groups,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Rows={rows_written:,}; columns {len(header)} -> {len(output_columns)}; elapsed={elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize and coalesce duplicate/alias columns in the merged IoV IDS CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    raise_csv_field_limit()
    args = parse_args()
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    canonicalize_csv(
        input_path=args.input,
        output_path=args.output,
        manifest_path=manifest,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
