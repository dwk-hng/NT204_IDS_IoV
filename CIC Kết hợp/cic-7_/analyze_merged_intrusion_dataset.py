#!/usr/bin/env python3
"""
Analyze the merged 7 percent intrusion dataset.

Outputs:
  - label_counts.csv
  - imbalance_summary.json
  - correlation_heatmap_matrix.csv
  - label_distribution.svg
  - imbalance_focus.svg
  - feature_correlation_heatmap.svg
  - analysis_report.html
  - analysis_report.pdf

The script intentionally avoids matplotlib/reportlab so it can run in this
project environment with only pandas/numpy/sklearn installed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import re
import sys
import textwrap
import time
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import DtypeWarning


warnings.filterwarnings("ignore", category=DtypeWarning)


DEFAULT_DATASET = Path(
    r"D:\Tài liệu UIT\Tài liệu kỳ 6\IDPS\Project\merged_intrusion_dataset_7_percent_canonical.csv"
)

ID_LIKE_COLUMNS = {
    "label",
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
    "simillarhttp",
    "similarhttp",
    "unnamed: 0",
}


def log(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, flush=True)


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.4f}%"


def slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def short_label(text: str, limit: int = 28) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def ascii_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return normalized.encode("ascii", "ignore").decode("ascii")


def esc_pdf(text: object) -> str:
    text = ascii_text(text)
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def read_manifest(dataset_path: Path) -> dict:
    manifest_path = Path(str(dataset_path) + ".manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_label_column(columns: Iterable[str]) -> str:
    columns = list(columns)
    for candidate in ("Label", "label", "Class", "class"):
        if candidate in columns:
            return candidate
    lowered = {col.lower(): col for col in columns}
    for candidate in ("label", "attack", "class"):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("Cannot find a label column. Expected a column named Label.")


def clean_label_series(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "<missing>").astype(str).str.strip()


def read_label_counts(
    dataset_path: Path,
    label_col: str,
    chunksize: int,
    expected_rows: int | None,
) -> Counter:
    counts: Counter[str] = Counter()
    start = time.time()
    seen = 0
    for index, chunk in enumerate(
        pd.read_csv(dataset_path, usecols=[label_col], chunksize=chunksize), start=1
    ):
        labels = clean_label_series(chunk[label_col])
        counts.update(labels.tolist())
        seen += len(chunk)
        if index == 1 or index % 5 == 0:
            suffix = ""
            if expected_rows:
                suffix = f" ({seen / expected_rows * 100:.1f}%)"
            log(f"[labels] read {fmt_int(seen)} rows{suffix} in {time.time() - start:.1f}s")
    return counts


def infer_numeric_columns(dataset_path: Path, label_col: str, nrows: int = 5000) -> list[str]:
    sample = pd.read_csv(dataset_path, nrows=nrows)
    numeric_cols: list[str] = []
    for col in sample.columns:
        key = col.strip().lower()
        if key in ID_LIKE_COLUMNS or col == label_col:
            continue
        converted = pd.to_numeric(sample[col], errors="coerce")
        valid_ratio = float(converted.notna().mean())
        if valid_ratio >= 0.75 and converted.nunique(dropna=True) > 1:
            numeric_cols.append(col)
    return numeric_cols


def sample_numeric_data(
    dataset_path: Path,
    label_col: str,
    numeric_cols: list[str],
    sample_rows: int,
    chunksize: int,
    total_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if not numeric_cols:
        raise ValueError("No numeric columns were detected for heatmap analysis.")

    usecols = [label_col] + numeric_cols
    frames: list[pd.DataFrame] = []
    collected = 0
    frac = min(1.0, max(sample_rows / max(total_rows, 1), 1 / max(total_rows, 1)))
    rng_seed = int(random_state)
    start = time.time()

    for index, chunk in enumerate(
        pd.read_csv(dataset_path, usecols=usecols, chunksize=chunksize), start=1
    ):
        remaining = sample_rows - collected
        if remaining <= 0:
            break

        take = max(1, int(math.ceil(len(chunk) * frac * 1.15)))
        take = min(take, len(chunk), remaining)
        sampled = chunk.sample(n=take, random_state=rng_seed + index)
        sampled[label_col] = clean_label_series(sampled[label_col])

        for col in numeric_cols:
            sampled[col] = pd.to_numeric(sampled[col], errors="coerce")
        sampled.replace([np.inf, -np.inf], np.nan, inplace=True)

        frames.append(sampled)
        collected += len(sampled)
        if index == 1 or index % 5 == 0:
            log(
                f"[sample] collected {fmt_int(collected)} / {fmt_int(sample_rows)} rows "
                f"in {time.time() - start:.1f}s"
            )

    if not frames:
        raise ValueError("Could not collect any sample rows.")

    sample = pd.concat(frames, ignore_index=True)
    if len(sample) > sample_rows:
        sample = sample.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)
    return sample


def build_label_count_frame(counts: Counter[str]) -> pd.DataFrame:
    total = sum(counts.values())
    rows = []
    for label, count in counts.most_common():
        rows.append(
            {
                "label": label,
                "count": int(count),
                "percentage": count / total * 100 if total else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["cumulative_percentage"] = df["percentage"].cumsum()
    if not df.empty:
        max_count = float(df["count"].max())
        df["ratio_to_majority"] = max_count / df["count"].clip(lower=1)
    else:
        df["ratio_to_majority"] = []
    return df


def imbalance_summary(label_df: pd.DataFrame) -> dict:
    total = int(label_df["count"].sum())
    class_count = int(len(label_df))
    if label_df.empty:
        return {"total_rows": 0, "class_count": 0}

    counts = label_df["count"].to_numpy(dtype=float)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p + 1e-15)).sum())
    normalized_entropy = float(entropy / math.log(class_count)) if class_count > 1 else 1.0
    effective_classes = float(math.exp(entropy))
    gini_impurity = float(1.0 - np.square(p).sum())

    majority = label_df.iloc[0]
    minority = label_df.sort_values("count", ascending=True).iloc[0]
    median_count = float(label_df["count"].median())
    rare_1 = label_df[label_df["percentage"] < 1.0]
    rare_01 = label_df[label_df["percentage"] < 0.1]
    under_median_10 = label_df[label_df["count"] < median_count / 10.0]

    return {
        "total_rows": total,
        "class_count": class_count,
        "majority_label": str(majority["label"]),
        "majority_count": int(majority["count"]),
        "majority_percentage": float(majority["percentage"]),
        "minority_label": str(minority["label"]),
        "minority_count": int(minority["count"]),
        "minority_percentage": float(minority["percentage"]),
        "imbalance_ratio_majority_to_minority": float(
            majority["count"] / max(float(minority["count"]), 1.0)
        ),
        "median_class_count": median_count,
        "labels_below_1_percent": rare_1["label"].astype(str).tolist(),
        "labels_below_0_1_percent": rare_01["label"].astype(str).tolist(),
        "labels_below_one_tenth_median": under_median_10["label"].astype(str).tolist(),
        "top_3_labels": label_df.head(3).to_dict(orient="records"),
        "bottom_10_labels": label_df.sort_values("count", ascending=True)
        .head(10)
        .to_dict(orient="records"),
        "normalized_entropy": normalized_entropy,
        "effective_number_of_classes": effective_classes,
        "gini_impurity": gini_impurity,
    }


def select_heatmap_features(
    sample: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    limit: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    usable_cols: list[str] = []
    numeric = sample[numeric_cols].copy()
    missing_ratio = numeric.isna().mean()

    for col in numeric_cols:
        series = numeric[col]
        if missing_ratio[col] > 0.70:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        usable_cols.append(col)

    if not usable_cols:
        raise ValueError("No usable numeric columns remained after cleaning missing values.")

    numeric = numeric[usable_cols]
    medians = numeric.median(numeric_only=True)
    numeric = numeric.fillna(medians).fillna(0.0)

    labels = sample[label_col].astype(str)
    codes, _ = pd.factorize(labels)

    scores = None
    try:
        from sklearn.feature_selection import f_classif

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_scores, _ = f_classif(numeric.to_numpy(dtype=float), codes)
        scores = pd.Series(raw_scores, index=usable_cols).replace([np.inf, -np.inf], np.nan)
    except Exception:
        codes_series = pd.Series(codes, index=numeric.index)
        scores = numeric.corrwith(codes_series).abs()

    scores = scores.fillna(0.0).sort_values(ascending=False)
    selected = scores.head(limit).index.tolist()
    corr = numeric[selected].corr().fillna(0.0)

    score_df = pd.DataFrame(
        {
            "feature": scores.index,
            "selection_score": scores.values,
        }
    )
    return corr, selected, score_df


def color_gradient(value: float) -> str:
    value = max(-1.0, min(1.0, float(value)))
    if value < 0:
        t = value + 1.0
        r = int(62 + (255 - 62) * t)
        g = int(113 + (255 - 113) * t)
        b = int(191 + (255 - 191) * t)
    else:
        t = value
        r = int(255 + (178 - 255) * t)
        g = int(255 + (24 - 255) * t)
        b = int(255 + (43 - 255) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<style>"
        "text{font-family:Arial,DejaVu Sans,sans-serif;fill:#1f2933}"
        ".small{font-size:12px}.label{font-size:13px}.title{font-size:22px;font-weight:700}"
        ".axis{stroke:#7b8794;stroke-width:1}.grid{stroke:#d9e2ec;stroke-width:1}"
        "</style>\n"
    )


def write_label_distribution_svg(label_df: pd.DataFrame, path: Path) -> None:
    plot_df = label_df.sort_values("count", ascending=False).copy()
    n = len(plot_df)
    row_h = 27
    width = 1120
    height = 120 + row_h * max(n, 1)
    left = 270
    right = 250
    top = 72
    max_bar = width - left - right
    max_count = float(plot_df["count"].max()) if n else 1.0
    max_log = math.log10(max_count + 1.0)

    parts = [svg_header(width, height)]
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>\n')
    parts.append(
        f'<text class="title" x="{left}" y="36" text-anchor="middle">Label distribution (log-scaled bars)</text>\n'
    )
    parts.append(
        f'<text class="small" x="{left}" y="58" text-anchor="middle">Total rows: {html.escape(fmt_int(int(plot_df["count"].sum())))} | Classes: {n}</text>\n'
    )

    for i, row in enumerate(plot_df.itertuples(index=False), start=0):
        y = top + i * row_h
        bar_w = max_bar * math.log10(float(row.count) + 1.0) / max_log if max_log else 0
        color = "#2f80ed" if i else "#d64545"
        parts.append(f'<line class="grid" x1="{left}" y1="{y + 9}" x2="{width - right}" y2="{y + 9}"/>\n')
        parts.append(
            f'<text class="label" x="18" y="{y + 14}">{html.escape(short_label(row.label, 34))}</text>\n'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" fill="{color}" rx="3"/>\n'
        )
        parts.append(
            f'<text class="small" x="{left + bar_w + 8}" y="{y + 14}">'
            f'{html.escape(fmt_int(row.count))} ({row.percentage:.4f}%)</text>\n'
        )

    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_imbalance_focus_svg(label_df: pd.DataFrame, path: Path) -> None:
    plot_df = label_df.sort_values("count", ascending=True).copy()
    n = len(plot_df)
    row_h = 28
    width = 1120
    height = 130 + row_h * max(n, 1)
    left = 290
    right = 260
    top = 80
    max_bar = width - left - right
    max_count = float(plot_df["count"].max()) if n else 1.0
    median_count = float(plot_df["count"].median()) if n else 0

    parts = [svg_header(width, height)]
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>\n')
    parts.append(
        '<text class="title" x="560" y="36" text-anchor="middle">Imbalance focus: minority to majority classes</text>\n'
    )
    parts.append(
        f'<text class="small" x="560" y="58" text-anchor="middle">Linear bars sorted ascending | Median class count: {html.escape(fmt_int(median_count))}</text>\n'
    )

    for i, row in enumerate(plot_df.itertuples(index=False), start=0):
        y = top + i * row_h
        bar_w = max_bar * float(row.count) / max_count if max_count else 0
        color = "#f2994a" if row.percentage < 1.0 else "#27ae60"
        parts.append(f'<line class="grid" x1="{left}" y1="{y + 10}" x2="{width - right}" y2="{y + 10}"/>\n')
        parts.append(
            f'<text class="label" x="18" y="{y + 15}">{html.escape(short_label(row.label, 36))}</text>\n'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="19" fill="{color}" rx="3"/>\n'
        )
        parts.append(
            f'<text class="small" x="{left + bar_w + 8}" y="{y + 15}">'
            f'{html.escape(fmt_int(row.count))} ({row.percentage:.4f}%, 1:{row.ratio_to_majority:.1f})</text>\n'
        )

    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_heatmap_svg(corr: pd.DataFrame, path: Path) -> None:
    labels = list(corr.columns)
    n = len(labels)
    cell = 32
    left = 250
    top = 210
    width = left + cell * n + 150
    height = top + cell * n + 95
    parts = [svg_header(width, height)]
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>\n')
    parts.append(
        f'<text class="title" x="{width / 2:.0f}" y="36" text-anchor="middle">Correlation heatmap of selected numeric features</text>\n'
    )
    parts.append(
        f'<text class="small" x="{width / 2:.0f}" y="58" text-anchor="middle">Features selected by ANOVA F-score against class labels</text>\n'
    )

    for i, label in enumerate(labels):
        x = left + i * cell + cell / 2
        parts.append(
            f'<text class="small" x="{x:.1f}" y="{top - 12}" text-anchor="end" '
            f'transform="rotate(-55 {x:.1f},{top - 12})">{html.escape(short_label(label, 22))}</text>\n'
        )
        y = top + i * cell + cell / 2 + 4
        parts.append(
            f'<text class="small" x="{left - 8}" y="{y:.1f}" text-anchor="end">{html.escape(short_label(label, 28))}</text>\n'
        )

    for r, row_label in enumerate(labels):
        for c, col_label in enumerate(labels):
            value = float(corr.loc[row_label, col_label])
            x = left + c * cell
            y = top + r * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{color_gradient(value)}" stroke="#ffffff" stroke-width="1"/>\n'
            )

    legend_x = left + cell * n + 42
    legend_y = top
    for j in range(101):
        value = 1 - 2 * j / 100
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y + j * 2}" width="18" height="2" fill="{color_gradient(value)}"/>\n'
        )
    parts.append(f'<text class="small" x="{legend_x + 28}" y="{legend_y + 8}">+1</text>\n')
    parts.append(f'<text class="small" x="{legend_x + 28}" y="{legend_y + 105}">0</text>\n')
    parts.append(f'<text class="small" x="{legend_x + 28}" y="{legend_y + 204}">-1</text>\n')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


class PdfPage:
    def __init__(self, width: int = 595, height: int = 842) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    @staticmethod
    def _rgb(color: tuple[float, float, float]) -> str:
        return " ".join(f"{max(0.0, min(1.0, c)):.4f}" for c in color)

    def text(
        self,
        x: float,
        y: float,
        text: object,
        size: int = 10,
        color: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        self.commands.append(f"{self._rgb(color)} rg")
        self.commands.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({esc_pdf(text)}) Tj ET")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float] = (0, 0, 0),
        width: float = 1.0,
    ) -> None:
        self.commands.append(f"{self._rgb(color)} RG {width:.2f} w")
        self.commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: tuple[float, float, float],
        stroke: tuple[float, float, float] | None = None,
    ) -> None:
        self.commands.append(f"{self._rgb(fill)} rg")
        if stroke is None:
            self.commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")
        else:
            self.commands.append(f"{self._rgb(stroke)} RG")
            self.commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re B")

    def stream(self) -> bytes:
        return ("\n".join(self.commands) + "\n").encode("latin-1", errors="replace")


class SimplePDF:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pages: list[PdfPage] = []

    def add_page(self) -> PdfPage:
        page = PdfPage()
        self.pages.append(page)
        return page

    def save(self) -> None:
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

        page_ids = []
        next_id = 4
        for _ in self.pages:
            page_ids.append(next_id)
            next_id += 2

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>".encode("ascii"))
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        for page_id, page in zip(page_ids, self.pages):
            content_id = page_id + 1
            page_obj = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page.width} {page.height}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
            stream = page.stream()
            content_obj = (
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream
                + b"endstream"
            )
            objects.append(page_obj)
            objects.append(content_obj)

        with self.path.open("wb") as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            offsets = [0]
            for number, obj in enumerate(objects, start=1):
                offsets.append(f.tell())
                f.write(f"{number} 0 obj\n".encode("ascii"))
                f.write(obj)
                f.write(b"\nendobj\n")
            xref = f.tell()
            f.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
            f.write(b"0000000000 65535 f \n")
            for offset in offsets[1:]:
                f.write(f"{offset:010d} 00000 n \n".encode("ascii"))
            f.write(
                (
                    f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                    f"startxref\n{xref}\n%%EOF\n"
                ).encode("ascii")
            )


def wrap_for_pdf(text: object, max_chars: int) -> list[str]:
    return textwrap.wrap(ascii_text(text), width=max_chars, break_long_words=False) or [""]


def pdf_color(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )


def draw_wrapped(
    page: PdfPage,
    x: float,
    y: float,
    text: object,
    size: int = 10,
    max_chars: int = 80,
    leading: int = 13,
    color: tuple[float, float, float] = (0, 0, 0),
) -> float:
    for line in wrap_for_pdf(text, max_chars):
        page.text(x, y, line, size=size, color=color)
        y -= leading
    return y


def build_pdf_report(
    path: Path,
    dataset_path: Path,
    manifest: dict,
    label_df: pd.DataFrame,
    summary: dict,
    corr: pd.DataFrame,
    selected_features: list[str],
    sample_size: int,
    output_dir: Path,
) -> None:
    pdf = SimplePDF(path)

    page = pdf.add_page()
    page.text(52, 795, "Merged Intrusion Dataset 7 Percent - Data Analysis Report", 17)
    page.line(52, 780, 543, 780, (0.20, 0.25, 0.32), 1.2)
    y = 754
    y = draw_wrapped(page, 52, y, f"Dataset: {dataset_path}", 10, 86)
    y = draw_wrapped(page, 52, y - 5, f"Output folder: {output_dir}", 10, 86)
    page.text(52, y - 12, f"Generated at: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 10)
    y -= 45

    page.text(52, y, "Key findings", 13)
    y -= 20
    findings = [
        f"Total rows analyzed: {fmt_int(summary['total_rows'])}; number of classes: {summary['class_count']}.",
        (
            f"Majority class is {summary['majority_label']} with "
            f"{fmt_int(summary['majority_count'])} rows ({summary['majority_percentage']:.4f}%)."
        ),
        (
            f"Minority class is {summary['minority_label']} with "
            f"{fmt_int(summary['minority_count'])} rows ({summary['minority_percentage']:.4f}%)."
        ),
        (
            "Majority/minority imbalance ratio: "
            f"{summary['imbalance_ratio_majority_to_minority']:.2f}:1."
        ),
        (
            f"Effective number of classes: {summary['effective_number_of_classes']:.2f} "
            f"out of {summary['class_count']} actual classes."
        ),
        (
            "Labels below 1 percent: "
            + (", ".join(summary["labels_below_1_percent"]) or "none")
            + "."
        ),
        f"Heatmap computed from {fmt_int(sample_size)} sampled rows and {len(selected_features)} selected numeric features.",
    ]
    for item in findings:
        page.text(64, y, "-", 10)
        y = draw_wrapped(page, 78, y, item, 10, 72, 13)
        y -= 4

    if manifest:
        y -= 10
        page.text(52, y, "Manifest metadata", 13)
        y -= 20
        for key in ("source_file", "rows_written", "output_columns", "created_at"):
            if key in manifest:
                y = draw_wrapped(page, 64, y, f"{key}: {manifest[key]}", 9, 80, 12)

    page = pdf.add_page()
    page.text(52, 795, "Label Distribution (log-scaled)", 16)
    page.line(52, 780, 543, 780, (0.20, 0.25, 0.32), 1.0)
    plot_df = label_df.sort_values("count", ascending=False)
    left = 172
    top_y = 750
    row_h = min(24, max(13, int(600 / max(len(plot_df), 1))))
    max_bar = 280
    max_log = math.log10(float(plot_df["count"].max()) + 1.0)
    for i, row in enumerate(plot_df.itertuples(index=False)):
        y_row = top_y - i * row_h
        if y_row < 70:
            break
        bar_w = max_bar * math.log10(float(row.count) + 1.0) / max_log if max_log else 0
        color = (0.84, 0.27, 0.27) if i == 0 else (0.18, 0.50, 0.84)
        page.text(52, y_row + 3, short_label(row.label, 22), 8)
        page.rect(left, y_row, bar_w, 11, color)
        page.text(left + bar_w + 6, y_row + 3, f"{fmt_int(row.count)} ({row.percentage:.3f}%)", 8)

    page = pdf.add_page()
    page.text(52, 795, "Where the Imbalance Happens", 16)
    page.line(52, 780, 543, 780, (0.20, 0.25, 0.32), 1.0)
    y = 752
    page.text(52, y, "Smallest classes", 12)
    y -= 22
    page.text(52, y, "Label", 9)
    page.text(265, y, "Count", 9)
    page.text(350, y, "Percent", 9)
    page.text(435, y, "Ratio to majority", 9)
    y -= 10
    page.line(52, y, 543, y, (0.75, 0.78, 0.82), 0.8)
    y -= 18
    for row in label_df.sort_values("count", ascending=True).head(22).itertuples(index=False):
        page.text(52, y, short_label(row.label, 32), 8)
        page.text(265, y, fmt_int(row.count), 8)
        page.text(350, y, f"{row.percentage:.5f}%", 8)
        page.text(435, y, f"1:{row.ratio_to_majority:.1f}", 8)
        y -= 18

    y -= 12
    page.text(52, y, "Interpretation", 12)
    y -= 20
    interpretation = (
        "The imbalance is concentrated in classes whose percentage is below 1 percent "
        "of the dataset and especially below 0.1 percent. During training, these classes "
        "can be under-learned because weighted aggregate scores are dominated by high-support labels."
    )
    draw_wrapped(page, 52, y, interpretation, 10, 84)

    page = pdf.add_page()
    page.text(52, 795, "Feature Correlation Heatmap", 16)
    page.line(52, 780, 543, 780, (0.20, 0.25, 0.32), 1.0)
    labels = list(corr.columns)
    n = len(labels)
    cell = min(22, max(11, int(390 / max(n, 1))))
    left = 160
    top = 710
    for i, label in enumerate(labels):
        page.text(52, top - i * cell + 3, short_label(label, 24), 6)
        page.text(left + i * cell, top + 12, str(i + 1), 6)
    for r, row_label in enumerate(labels):
        for c, col_label in enumerate(labels):
            value = float(corr.loc[row_label, col_label])
            x = left + c * cell
            y_cell = top - r * cell
            page.rect(x, y_cell, cell, cell, pdf_color(color_gradient(value)), (1, 1, 1))
    y = top - n * cell - 25
    page.text(52, y, "Selected feature order", 12)
    y -= 18
    for idx, feature in enumerate(selected_features, start=1):
        if y < 55:
            break
        y = draw_wrapped(page, 60, y, f"{idx}. {feature}", 7, 95, 9)

    page = pdf.add_page()
    page.text(52, 795, "Method Notes and Recommendations", 16)
    page.line(52, 780, 543, 780, (0.20, 0.25, 0.32), 1.0)
    y = 750
    notes = [
        (
            "Label counts are exact because the script scans the Label column across the full CSV."
        ),
        (
            "The heatmap is sample-based to keep memory usage stable on a multi-gigabyte CSV. "
            "Increase --sample-rows if the machine has more RAM."
        ),
        (
            "For modelling reports, do not rely only on weighted precision/recall/F1. Weighted metrics can "
            "look strong when majority classes dominate. Add per-class recall/F1 for the rare labels listed above."
        ),
        (
            "Suggested mitigations: stratified train/test split, class weights, focal loss or re-sampling, "
            "and threshold tuning for rare attack classes."
        ),
        (
            "Generated files include SVG plots, CSV tables, JSON summary, HTML report, and this PDF report."
        ),
    ]
    for item in notes:
        page.text(64, y, "-", 10)
        y = draw_wrapped(page, 78, y, item, 10, 72, 14)
        y -= 8

    pdf.save()


def build_html_report(
    path: Path,
    dataset_path: Path,
    label_df: pd.DataFrame,
    summary: dict,
    selected_features: list[str],
    sample_size: int,
    svg_names: list[str],
) -> None:
    top_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.label))}</td>"
        f"<td>{fmt_int(row.count)}</td>"
        f"<td>{row.percentage:.5f}%</td>"
        f"<td>1:{row.ratio_to_majority:.1f}</td>"
        "</tr>"
        for row in label_df.itertuples(index=False)
    )
    rare = ", ".join(summary["labels_below_1_percent"]) or "Không có"
    svgs = "\n".join(
        f'<section><img src="{html.escape(name)}" alt="{html.escape(name)}"></section>'
        for name in svg_names
    )
    feature_list = "".join(f"<li>{html.escape(f)}</li>" for f in selected_features)
    content = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Dataset analysis report</title>
<style>
body{{font-family:Arial,DejaVu Sans,sans-serif;margin:32px;color:#1f2933;line-height:1.45}}
h1,h2{{color:#102a43}} img{{max-width:100%;border:1px solid #d9e2ec;margin:14px 0 28px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #d9e2ec;padding:7px;text-align:left}}
.metric{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}
.metric div{{border:1px solid #d9e2ec;padding:12px;border-radius:6px;background:#f8fafc}}
</style>
</head>
<body>
<h1>Báo cáo phân tích dataset merged_intrusion_dataset_7_percent_canonical.csv</h1>
<p><b>Dataset:</b> {html.escape(str(dataset_path))}</p>
<div class="metric">
<div><b>Tổng số dòng</b><br>{fmt_int(summary['total_rows'])}</div>
<div><b>Số classes</b><br>{summary['class_count']}</div>
<div><b>Majority class</b><br>{html.escape(summary['majority_label'])}: {summary['majority_percentage']:.4f}%</div>
<div><b>Imbalance ratio</b><br>{summary['imbalance_ratio_majority_to_minority']:.2f}:1</div>
</div>
<h2>Kết luận nhanh</h2>
<p>Mất cân bằng tập trung ở các nhãn có tỷ lệ dưới 1%: {html.escape(rare)}.</p>
<p>Nhãn ít nhất là <b>{html.escape(summary['minority_label'])}</b> với {fmt_int(summary['minority_count'])} dòng ({summary['minority_percentage']:.5f}%).</p>
<p>Heatmap được tính trên mẫu {fmt_int(sample_size)} dòng, chọn {len(selected_features)} feature số có khả năng phân tách nhãn cao nhất.</p>
<h2>Plots</h2>
{svgs}
<h2>Feature dùng trong heatmap</h2>
<ol>{feature_list}</ol>
<h2>Bảng số lượng nhãn</h2>
<table>
<thead><tr><th>Label</th><th>Count</th><th>Percentage</th><th>Ratio to majority</th></tr></thead>
<tbody>{top_table}</tbody>
</table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze label distribution, class imbalance and feature correlations."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--sample-rows", type=int, default=120_000)
    parser.add_argument("--heatmap-features", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_path.with_name(dataset_path.stem + "_analysis")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(dataset_path)
    expected_rows = manifest.get("rows_written")
    expected_rows = int(expected_rows) if isinstance(expected_rows, int) else None

    header = pd.read_csv(dataset_path, nrows=0)
    label_col = detect_label_column(header.columns)
    log(f"Dataset: {dataset_path}")
    log(f"Output:  {output_dir}")
    log(f"Columns: {len(header.columns)} | Label column: {label_col}")

    counts = read_label_counts(dataset_path, label_col, args.chunksize, expected_rows)
    label_df = build_label_count_frame(counts)
    total_rows = int(label_df["count"].sum())
    label_counts_path = output_dir / "label_counts.csv"
    label_df.to_csv(label_counts_path, index=False, encoding="utf-8-sig")

    summary = imbalance_summary(label_df)
    summary_path = output_dir / "imbalance_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Exact rows counted: {fmt_int(total_rows)}")
    log(f"Classes: {summary['class_count']}")
    log(
        "Majority/minority: "
        f"{summary['majority_label']} / {summary['minority_label']} "
        f"= {summary['imbalance_ratio_majority_to_minority']:.2f}:1"
    )

    numeric_cols = infer_numeric_columns(dataset_path, label_col)
    log(f"Numeric feature candidates: {len(numeric_cols)}")
    sample = sample_numeric_data(
        dataset_path,
        label_col,
        numeric_cols,
        sample_rows=args.sample_rows,
        chunksize=args.chunksize,
        total_rows=total_rows,
        random_state=args.random_state,
    )
    log(f"Sample rows for heatmap: {fmt_int(len(sample))}")

    corr, selected_features, feature_scores = select_heatmap_features(
        sample,
        label_col,
        numeric_cols,
        limit=args.heatmap_features,
    )
    corr_path = output_dir / "correlation_heatmap_matrix.csv"
    score_path = output_dir / "feature_selection_scores.csv"
    corr.to_csv(corr_path, encoding="utf-8-sig")
    feature_scores.to_csv(score_path, index=False, encoding="utf-8-sig")

    label_svg = output_dir / "label_distribution.svg"
    imbalance_svg = output_dir / "imbalance_focus.svg"
    heatmap_svg = output_dir / "feature_correlation_heatmap.svg"
    write_label_distribution_svg(label_df, label_svg)
    write_imbalance_focus_svg(label_df, imbalance_svg)
    write_heatmap_svg(corr, heatmap_svg)

    html_path = output_dir / "analysis_report.html"
    build_html_report(
        html_path,
        dataset_path,
        label_df,
        summary,
        selected_features,
        len(sample),
        [label_svg.name, imbalance_svg.name, heatmap_svg.name],
    )

    pdf_path = output_dir / "analysis_report.pdf"
    build_pdf_report(
        pdf_path,
        dataset_path,
        manifest,
        label_df,
        summary,
        corr,
        selected_features,
        len(sample),
        output_dir,
    )

    log("\nDone. Generated files:")
    for path in (
        label_counts_path,
        summary_path,
        corr_path,
        score_path,
        label_svg,
        imbalance_svg,
        heatmap_svg,
        html_path,
        pdf_path,
    ):
        log(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
