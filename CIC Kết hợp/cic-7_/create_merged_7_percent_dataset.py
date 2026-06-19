from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CIC_2017 = (
    PROJECT_ROOT
    / "cic-ids-2017"
    / "MachineLearningCVE"
    / "cic-ids-2017-merge.csv"
)
DEFAULT_CIC_2018 = (
    PROJECT_ROOT
    / "Dataset2018"
    / "CSV"
    / "cic-ddos-2018-merge-repaired.csv"
)
DEFAULT_CIC_2019_DIR = PROJECT_ROOT / "cic-iot-2019"
DEFAULT_OUTPUT = PROJECT_ROOT / "merged_intrusion_dataset_7_percent.csv"


# CIC-IDS-2017 / CIC-DDoS-2019 mostly use long feature names.
# CSE-CIC-IDS-2018 uses shorter aliases. Canonical mode maps those aliases
# into the long names so the merged output is one usable table.
COLUMN_ALIASES = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


@dataclass
class InputCsv:
    dataset_name: str
    path: Path
    header: list[str]
    output_indexes: list[int]
    row_count: int = 0


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 7 percent global sample from CIC-IDS-2017, "
            "CSE-CIC-IDS-2018, and CIC-DDoS-2019 without loading them into RAM."
        )
    )
    parser.add_argument("--cic2017", type=Path, default=DEFAULT_CIC_2017)
    parser.add_argument("--cic2018", type=Path, default=DEFAULT_CIC_2018)
    parser.add_argument("--cic2019-dir", type=Path, default=DEFAULT_CIC_2019_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frac", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schema",
        choices=("canonical", "raw-union"),
        default="canonical",
        help=(
            "canonical: strip/map common CIC feature aliases; "
            "raw-union: preserve raw headers like pandas.concat."
        ),
    )
    parser.add_argument(
        "--count-method",
        choices=("fast", "csv"),
        default="fast",
        help=(
            "fast counts newline bytes and is much quicker for CIC CSV files; "
            "csv parses rows and is safer for unusual quoted newlines."
        ),
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--output-encoding", default="utf-8")
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-count",
        action="store_true",
        help="With --dry-run, only inspect inputs and schema without counting rows.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path for a JSON manifest. Defaults to <output>.manifest.json.",
    )
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args()


def read_header(path: Path, encoding: str) -> list[str]:
    with path.open("r", newline="", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc
    if header:
        header[0] = header[0].lstrip("\ufeff")
    return header


def canonical_column(name: str) -> str:
    stripped = name.lstrip("\ufeff").strip()
    return COLUMN_ALIASES.get(stripped, stripped)


def schema_column(name: str, schema_mode: str) -> str:
    if schema_mode == "canonical":
        return canonical_column(name)
    return name.lstrip("\ufeff")


def discover_inputs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    output_path = args.output.resolve()
    inputs: list[tuple[str, Path]] = [
        ("CIC_IDS_2017", args.cic2017),
        ("CSE_CIC_IDS_2018", args.cic2018),
    ]

    if not args.cic2019_dir.exists():
        raise FileNotFoundError(f"CIC-DDoS-2019 directory not found: {args.cic2019_dir}")

    cic2019_files = sorted(
        (p for p in args.cic2019_dir.rglob("*.csv") if p.resolve() != output_path),
        key=lambda p: str(p).lower(),
    )
    inputs.extend(("CIC_DDoS_2019", p) for p in cic2019_files)

    missing = [p for _, p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(f"- {p}" for p in missing)
        )
    if not cic2019_files:
        raise FileNotFoundError(f"No CSV files found under: {args.cic2019_dir}")

    return inputs


def build_schema(
    discovered: Iterable[tuple[str, Path]], schema_mode: str, encoding: str
) -> tuple[list[str], list[InputCsv]]:
    output_columns: list[str] = []
    column_to_index: dict[str, int] = {}
    input_csvs: list[InputCsv] = []

    for dataset_name, path in discovered:
        header = read_header(path, encoding)
        output_indexes: list[int] = []

        for raw_col in header:
            out_col = schema_column(raw_col, schema_mode)
            if out_col not in column_to_index:
                column_to_index[out_col] = len(output_columns)
                output_columns.append(out_col)
            output_indexes.append(column_to_index[out_col])

        input_csvs.append(
            InputCsv(
                dataset_name=dataset_name,
                path=path,
                header=header,
                output_indexes=output_indexes,
            )
        )

    return output_columns, input_csvs


def count_rows_fast(path: Path) -> int:
    size = path.stat().st_size
    if size == 0:
        return 0

    newline_count = 0
    last_byte = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    physical_lines = newline_count if last_byte in (b"\n", b"\r") else newline_count + 1
    return max(0, physical_lines - 1)


def count_rows_csv(path: Path, encoding: str) -> int:
    with path.open("r", newline="", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def count_inputs(input_csvs: list[InputCsv], method: str, encoding: str) -> int:
    total = 0
    print(f"Counting rows with method={method} ...")
    for i, item in enumerate(input_csvs, start=1):
        started = time.time()
        if method == "fast":
            item.row_count = count_rows_fast(item.path)
        else:
            item.row_count = count_rows_csv(item.path, encoding)
        total += item.row_count
        elapsed = time.time() - started
        print(
            f"[{i:02d}/{len(input_csvs):02d}] {item.dataset_name}: "
            f"{item.row_count:,} rows - {item.path.name} ({elapsed:.1f}s)"
        )
    print(f"Total rows: {total:,}")
    return total


def sample_target(total_rows: int, frac: float) -> int:
    if not 0 < frac <= 1:
        raise ValueError("--frac must be in the range (0, 1].")
    return int(round(total_rows * frac))


def sample_streaming(
    input_csvs: list[InputCsv],
    output_columns: list[str],
    args: argparse.Namespace,
    total_rows: int,
    target_rows: int,
) -> dict[str, int]:
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {output}")

    rng = random.Random(args.seed)
    remaining = total_rows
    needed = target_rows
    seen = 0
    selected = 0
    malformed_width = 0

    final_columns = list(output_columns)
    source_dataset_index = -1
    source_file_index = -1
    if args.include_source:
        source_dataset_index = len(final_columns)
        final_columns.append("source_dataset")
        source_file_index = len(final_columns)
        final_columns.append("source_file")

    print(
        f"Sampling {target_rows:,} rows from {total_rows:,} "
        f"({args.frac:.4%}), seed={args.seed}"
    )
    print(f"Writing output: {output}")

    with output.open("w", newline="", encoding=args.output_encoding) as out_f:
        writer = csv.writer(out_f, lineterminator="\n")
        writer.writerow(final_columns)

        for file_no, item in enumerate(input_csvs, start=1):
            file_seen = 0
            file_selected = 0
            started = time.time()
            print(f"[{file_no:02d}/{len(input_csvs):02d}] Reading {item.path}")

            with item.path.open(
                "r", newline="", encoding=args.encoding, errors="replace"
            ) as in_f:
                reader = csv.reader(in_f)
                next(reader, None)

                for row in reader:
                    if remaining <= 0:
                        break

                    choose = False
                    if needed > 0:
                        choose = needed == remaining or rng.random() < (needed / remaining)

                    if choose:
                        out_row = [""] * len(final_columns)
                        limit = min(len(row), len(item.output_indexes))
                        if len(row) != len(item.output_indexes):
                            malformed_width += 1
                        for src_idx in range(limit):
                            out_row[item.output_indexes[src_idx]] = row[src_idx]
                        if args.include_source:
                            out_row[source_dataset_index] = item.dataset_name
                            out_row[source_file_index] = str(item.path)
                        writer.writerow(out_row)
                        selected += 1
                        file_selected += 1
                        needed -= 1

                    seen += 1
                    file_seen += 1
                    remaining -= 1

                    if args.progress_every and seen % args.progress_every == 0:
                        print(
                            f"  progress: seen={seen:,}, selected={selected:,}, "
                            f"needed={needed:,}, remaining={remaining:,}"
                        )

            elapsed = time.time() - started
            if file_seen != item.row_count:
                print(
                    f"  warning: counted {item.row_count:,} rows but parsed "
                    f"{file_seen:,} rows in {item.path.name}"
                )
            print(
                f"  done: parsed={file_seen:,}, selected={file_selected:,} "
                f"({elapsed:.1f}s)"
            )

            if needed == 0:
                print("Target sample size reached.")
                break

    if selected != target_rows:
        print(
            f"warning: selected {selected:,} rows, expected {target_rows:,}. "
            "If this happens with --count-method fast, retry with --count-method csv."
        )
    if malformed_width:
        print(f"warning: {malformed_width:,} selected rows had a non-header column count.")

    return {
        "seen_rows": seen,
        "selected_rows": selected,
        "remaining_rows": remaining,
        "remaining_needed": needed,
        "malformed_width_selected_rows": malformed_width,
    }


def write_manifest(
    args: argparse.Namespace,
    input_csvs: list[InputCsv],
    output_columns: list[str],
    total_rows: int | None,
    target_rows: int | None,
    sample_stats: dict[str, int] | None,
) -> None:
    if args.no_manifest:
        return

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")

    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output": str(args.output),
        "frac": args.frac,
        "seed": args.seed,
        "schema": args.schema,
        "count_method": args.count_method,
        "total_rows": total_rows,
        "target_rows": target_rows,
        "sample_stats": sample_stats,
        "columns": output_columns,
        "input_files": [
            {
                "dataset_name": item.dataset_name,
                "path": str(item.path),
                "row_count": item.row_count,
                "column_count": len(item.header),
            }
            for item in input_csvs
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Manifest written: {manifest_path}")


def print_schema_summary(input_csvs: list[InputCsv], output_columns: list[str]) -> None:
    print(f"Input files: {len(input_csvs)}")
    for item in input_csvs:
        print(
            f"- {item.dataset_name}: {item.path.name} "
            f"({len(item.header)} source columns)"
        )
    print(f"Output columns: {len(output_columns)}")
    print("First 20 output columns:")
    for col in output_columns[:20]:
        print(f"  - {col}")
    if len(output_columns) > 20:
        print(f"  ... {len(output_columns) - 20} more columns")


def main() -> None:
    configure_stdio()
    raise_csv_field_limit()
    args = parse_args()

    discovered = discover_inputs(args)
    output_columns, input_csvs = build_schema(
        discovered=discovered,
        schema_mode=args.schema,
        encoding=args.encoding,
    )
    print_schema_summary(input_csvs, output_columns)

    total_rows = None
    target_rows = None
    sample_stats = None

    if not args.no_count:
        total_rows = count_inputs(input_csvs, args.count_method, args.encoding)
        target_rows = sample_target(total_rows, args.frac)
        print(f"Target rows ({args.frac:.4%}): {target_rows:,}")

    if args.dry_run:
        print("Dry run complete. No output CSV was written.")
        if args.manifest is not None:
            write_manifest(args, input_csvs, output_columns, total_rows, target_rows, None)
        return

    if total_rows is None or target_rows is None:
        raise ValueError("Cannot write a sample with --no-count.")

    sample_stats = sample_streaming(
        input_csvs=input_csvs,
        output_columns=output_columns,
        args=args,
        total_rows=total_rows,
        target_rows=target_rows,
    )
    write_manifest(args, input_csvs, output_columns, total_rows, target_rows, sample_stats)
    print("Done.")


if __name__ == "__main__":
    main()
