from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parent
CASE_ROOT = CODE_ROOT.parent
STUDY_CASES_ROOT = CASE_ROOT.parent
PROJECT_ROOT = STUDY_CASES_ROOT.parent


def _prepend_sys_path(path: Path | None) -> None:
    if path is None:
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _find_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded
    return None


def _build_input_candidates() -> list[Path]:
    candidates: list[Path] = []
    input_path = os.environ.get("PICOTA_ENERGIGRAN_DATA")
    if input_path:
        candidates.append(Path(input_path))
    candidates.extend(
        [
            CASE_ROOT / "data" / "energigran.tsv",
            CASE_ROOT / "data" / "energigran.csv",
            CASE_ROOT / "data" / "energigran" / "energigran.tsv",
            CASE_ROOT / "data" / "energigran" / "energigran.csv",
            PROJECT_ROOT / "runtime.test" / "data" / "energigran.tsv",
            PROJECT_ROOT.parent / "picota" / "runtime.test" / "data" / "energigran.tsv",
        ]
    )
    return candidates


def _default_input_path() -> Path:
    candidates = _build_input_candidates()
    existing = _find_existing_path(candidates)
    return existing or candidates[0]


def _load_training_dependencies():
    global TimeSeriesDataset, compute_violation_report

    try:
        from AlternativeKanTrainer import AlternativeKanTrainer
        from MetamorphicAlternativeKanTrainer import MetamorphicAlternativeKanTrainer
        from SolarPlantRuleCatalog import (
            DEFAULT_SOLAR_PLANT_RULE_WEIGHT_MAP,
            build_solar_plant_active_power_rule_specs,
        )
        from TabNetAlternativeTrainer import TabNetAlternativeTrainer
        from metamorphic_evaluation import compute_violation_report as _compute_violation_report
        from kan.MetamorphicCatalog import summarize_rule_specs
        from kan.TimeSeriesDataset import TimeSeriesDataset as _TimeSeriesDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import Energigran training dependencies from the local code directory."
        ) from exc

    TimeSeriesDataset = _TimeSeriesDataset
    compute_violation_report = _compute_violation_report
    return (
        AlternativeKanTrainer,
        MetamorphicAlternativeKanTrainer,
        TabNetAlternativeTrainer,
        DEFAULT_SOLAR_PLANT_RULE_WEIGHT_MAP,
        build_solar_plant_active_power_rule_specs,
        summarize_rule_specs,
    )


_prepend_sys_path(CODE_ROOT)

TimeSeriesDataset = None
compute_violation_report = None

TIME_FEATURE_NAMES = [
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos",
    "quarter_sin",
    "quarter_cos",
]

ALL_NUMERIC_COLUMNS = [
    "cellTemperature",
    "Infecar.temperature",
    "Infecar.radiation",
    "grid",
    "consumption",
    "generation",
]

TARGET_COLUMN = "generation"
FORBIDDEN_INPUTS = {TARGET_COLUMN}
INPUT_NUMERIC_COLUMNS = [c for c in ALL_NUMERIC_COLUMNS if c not in FORBIDDEN_INPUTS]
TARGET_FUTURE_COLUMN = "generation_future"


def detect_table_delimiter(table_path: Path) -> str:
    if table_path.suffix.lower() == ".tsv":
        return "\t"
    with table_path.open("r", encoding="utf-8", newline="") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                return "\t"
            return ","
    return ","


def parse_utc_instant(value: str) -> datetime:
    ts = value.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def sin_cos(value: float, period: float) -> tuple[float, float]:
    angle = (2.0 * math.pi * value) / period
    return math.sin(angle), math.cos(angle)


def encode_time_features(dt_utc: datetime) -> list[float]:
    month_idx = dt_utc.month - 1
    day_idx = dt_utc.day - 1
    hour_idx = dt_utc.hour
    quarter_idx = (dt_utc.month - 1) // 3  # year quarter: 0..3

    month_sin, month_cos = sin_cos(month_idx, 12.0)
    day_sin, day_cos = sin_cos(day_idx, 31.0)
    hour_sin, hour_cos = sin_cos(hour_idx, 24.0)
    quarter_sin, quarter_cos = sin_cos(quarter_idx, 4.0)

    return [
        month_sin,
        month_cos,
        day_sin,
        day_cos,
        hour_sin,
        hour_cos,
        quarter_sin,
        quarter_cos,
    ]


def load_hourly_means(table_path: Path) -> list[dict]:
    def make_bucket() -> dict[str, float]:
        bucket = {"count": 0.0}
        for col in ALL_NUMERIC_COLUMNS:
            bucket[col] = 0.0
        return bucket

    buckets: dict[datetime, dict[str, float]] = defaultdict(make_bucket)
    delimiter = detect_table_delimiter(table_path)

    with table_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        required = {"instant", *ALL_NUMERIC_COLUMNS}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Input table missing required columns: {missing}. "
                f"Found={reader.fieldnames}, delimiter='{delimiter}'"
            )

        for row in reader:
            dt = parse_utc_instant(row["instant"])
            hour_dt = dt.replace(minute=0, second=0, microsecond=0)
            bucket = buckets[hour_dt]
            try:
                parsed_values = {col: float(row[col]) for col in ALL_NUMERIC_COLUMNS}
            except (TypeError, ValueError):
                continue
            try:
                for col in ALL_NUMERIC_COLUMNS:
                    bucket[col] += parsed_values[col]
            except (TypeError, ValueError):
                continue
            bucket["count"] += 1.0

    hourly_rows: list[dict] = []
    for hour_dt in sorted(buckets.keys()):
        bucket = buckets[hour_dt]
        count = int(bucket.get("count", 0.0))
        if count <= 0:
            continue
        row = {"instant": hour_dt}
        for col in ALL_NUMERIC_COLUMNS:
            row[col] = float(bucket[col] / count)
        hourly_rows.append(row)
    return hourly_rows


def build_horizon_examples(hourly_rows: list[dict], horizon_hours: int) -> list[dict]:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")
    by_instant = {row["instant"]: row for row in hourly_rows}
    examples: list[dict] = []
    for row in hourly_rows:
        dst_dt = row["instant"] + timedelta(hours=int(horizon_hours))
        target_row = by_instant.get(dst_dt)
        if target_row is None:
            continue
        enriched = dict(row)
        enriched[TARGET_FUTURE_COLUMN] = float(target_row[TARGET_COLUMN])
        examples.append(enriched)
    return examples


def split_records(
        records: list[dict],
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    if len(records) < 10:
        raise ValueError(f"Need >=10 records after hourly aggregation, got {len(records)}")
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    train_end = max(1, int(len(records) * train_ratio))
    val_end = min(len(records) - 1, train_end + max(1, int(len(records) * val_ratio)))
    if train_end >= val_end:
        train_end = max(1, val_end - 1)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    if len(test_idx) == 0:
        test_idx = indices[-1:]
        val_idx = indices[train_end:-1]
        if len(val_idx) == 0:
            val_idx = indices[train_end - 1: train_end]
            train_idx = indices[: train_end - 1]

    train_rows = [records[int(i)] for i in train_idx]
    val_rows = [records[int(i)] for i in val_idx]
    test_rows = [records[int(i)] for i in test_idx]
    return train_rows, val_rows, test_rows


def compute_feature_stats(train_rows: list[dict]) -> tuple[list[float], list[float]]:
    feature_matrix = np.array(
        [[float(row[col]) for col in INPUT_NUMERIC_COLUMNS] for row in train_rows],
        dtype=np.float64,
    )
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    return means.astype(np.float32).tolist(), stds.astype(np.float32).tolist()


def compute_target_scaler(train_rows: list[dict]) -> tuple[float, float]:
    targets = [float(row[TARGET_FUTURE_COLUMN]) for row in train_rows]
    out_min = float(min(targets))
    out_max = float(max(targets))
    if out_max <= out_min:
        out_max = out_min + 1.0
    return out_min, out_max


def make_kan_items(rows: list[dict], out_min: float, out_max: float) -> list[dict]:
    span = out_max - out_min
    items: list[dict] = []
    for row in rows:
        t_features = encode_time_features(row["instant"])
        numerical_t_features = [float(row[col]) for col in INPUT_NUMERIC_COLUMNS]
        target_raw = float(row[TARGET_FUTURE_COLUMN])
        out = (target_raw - out_min) / span
        items.append(
            {
                "out": float(out),
                "t": [float(v) for v in t_features],
                "categorical_t_features": [],
                "numerical_t_features": [float(v) for v in numerical_t_features],
                "lookback_t": [],
                "categorical_lookback_features": [],
                "numerical_lookback_features": [],
            }
        )
    return items


class ModelFlatWrapper(torch.nn.Module):
    def __init__(self, predictor_model: torch.nn.Module, time_dim: int, numeric_dim: int):
        super().__init__()
        self.predictor_model = predictor_model
        self.time_dim = int(time_dim)
        self.numeric_dim = int(numeric_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        expected = self.time_dim + self.numeric_dim
        if x.size(1) != expected:
            raise ValueError(
                "Input feature width does not match configured dimensions "
                f"({x.size(1)} != {expected})"
            )
        t_end = self.time_dim
        n_end = t_end + self.numeric_dim
        batch_size = x.size(0)
        zeros = torch.zeros((batch_size, 0), dtype=x.dtype, device=x.device)
        batch = {
            "t": x[:, :t_end],
            "numerical_t_features": x[:, t_end:n_end],
            "categorical_t_features": zeros,
            "lookback_t": zeros,
            "categorical_lookback_features": zeros,
            "numerical_lookback_features": zeros,
        }
        return self.predictor_model(batch)


def flatten_items(items: list[dict]) -> np.ndarray:
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    flat: list[list[float]] = []
    for item in items:
        flat.append(list(item["t"]) + list(item["numerical_t_features"]))
    return np.asarray(flat, dtype=np.float32)


def sample_rows(matrix: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if matrix.shape[0] == 0:
        return matrix
    if max_rows <= 0 or matrix.shape[0] <= max_rows:
        return matrix
    rng = np.random.default_rng(seed)
    idx = rng.choice(matrix.shape[0], size=int(max_rows), replace=False)
    return matrix[idx]


def compute_shap_importance(
    model: torch.nn.Module,
    input_variables: list[str],
    train_items: list[dict],
    test_items: list[dict],
    time_dim: int,
    numeric_dim: int,
    target_min: float,
    target_max: float,
    background_size: int,
    eval_size: int,
    top_k: int,
    seed: int,
    output_path: Path,
) -> list[dict[str, float | str]]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP no está instalado. Instala 'shap' en el venv para habilitar el análisis."
        ) from exc

    train_matrix = flatten_items(train_items)
    test_matrix = flatten_items(test_items)
    if train_matrix.shape[0] == 0 or test_matrix.shape[0] == 0:
        raise ValueError("No hay muestras suficientes para SHAP.")

    background = sample_rows(train_matrix, max_rows=int(background_size), seed=int(seed))
    explained = sample_rows(test_matrix, max_rows=int(eval_size), seed=int(seed) + 1)

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    wrapped = ModelFlatWrapper(
        predictor_model=model_cpu,
        time_dim=time_dim,
        numeric_dim=numeric_dim,
    ).to("cpu").eval()

    background_t = torch.tensor(background, dtype=torch.float32)
    explained_t = torch.tensor(explained, dtype=torch.float32)

    explainer = shap.GradientExplainer(wrapped, background_t)
    shap_values = explainer.shap_values(explained_t)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[0], dtype=np.float64)
    else:
        shap_array = np.asarray(shap_values, dtype=np.float64)

    if shap_array.ndim == 3:
        if shap_array.shape[-1] == 1:
            shap_array = shap_array[:, :, 0]
        elif shap_array.shape[0] == 1:
            shap_array = shap_array[0]
    if shap_array.ndim != 2:
        raise ValueError(f"Forma SHAP inesperada: {shap_array.shape}")

    if shap_array.shape[1] != len(input_variables):
        if shap_array.shape[0] == len(input_variables):
            shap_array = shap_array.T
        else:
            raise ValueError(
                f"Ancho SHAP ({shap_array.shape[1]}) distinto a input_variables ({len(input_variables)})"
            )

    mean_abs = np.mean(np.abs(shap_array), axis=0)
    raw_span = target_max - target_min if target_max > target_min else 1.0
    mean_abs_raw = mean_abs * raw_span
    total = float(np.sum(mean_abs))
    share = np.zeros_like(mean_abs) if total <= 0.0 else (mean_abs / total) * 100.0

    order = np.argsort(-mean_abs)
    rows: list[dict[str, float | str]] = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": float(rank),
                "feature": str(input_variables[int(idx)]),
                "mean_abs_shap_model": float(mean_abs[int(idx)]),
                "mean_abs_shap_raw": float(mean_abs_raw[int(idx)]),
                "share_percent": float(share[int(idx)]),
            }
        )

    output_path = output_path.resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "feature", "mean_abs_shap_model", "mean_abs_shap_raw", "share_percent"])
        for row in rows:
            writer.writerow(
                [
                    int(row["rank"]),
                    row["feature"],
                    f"{float(row['mean_abs_shap_model']):.10f}",
                    f"{float(row['mean_abs_shap_raw']):.10f}",
                    f"{float(row['share_percent']):.6f}",
                ]
            )

    k = max(1, int(top_k))
    return rows[:k]


def print_violation_report(split_name: str, report: dict | None) -> None:
    if report is None:
        print(f"rule_violations[{split_name}]: not_available", flush=True)
        return
    total_violations = int(report.get("total_violations", 0))
    total_cases = int(report.get("total_cases", 0))
    overall_rate = float(report.get("overall_violation_rate", float("nan")))
    overall_pct = overall_rate * 100.0 if math.isfinite(overall_rate) else float("nan")
    print(
        f"rule_violations[{split_name}]: "
        f"overall_rate={overall_rate:.6f} "
        f"overall_pct={overall_pct:.2f}% "
        f"cases={total_cases} "
        f"violations={total_violations}",
        flush=True,
    )
    by_test = report.get("by_test", {})
    for rule_name, stats in sorted(by_test.items()):
        violations = int(stats.get("violations", 0))
        total = int(stats.get("total", 0))
        rate = float(stats.get("violation_rate", float("nan")))
        pct = rate * 100.0 if math.isfinite(rate) else float("nan")
        print(
            f"rule_violations[{split_name}] "
            f"rule={rule_name} "
            f"rate={rate:.6f} "
            f"pct={pct:.2f}% "
            f"violations={violations}/{total}",
            flush=True,
        )


def evaluate_rule_violations(
        model,
        items: list[dict],
        batch_size: int,
        rule_specs: list,
        atol: float,
        rtol: float,
) -> dict | None:
    relation_tests = [spec.relation_test for spec in rule_specs if getattr(spec, "relation_test", None) is not None]
    if not relation_tests:
        return None
    data_loader = DataLoader(TimeSeriesDataset(items), batch_size=int(batch_size), shuffle=False)
    return compute_violation_report(
        model=model,
        data_loader=data_loader,
        metamorphic_tests=relation_tests,
        atol=float(atol),
        rtol=float(rtol),
    )


def print_model_summary(
        label: str,
        best_val_metrics: dict[str, float | int | str],
        test_metrics: dict[str, float | int | str],
        val_violation_report: dict | None,
        test_violation_report: dict | None,
) -> None:
    print(
        f"{label} best_val: "
        f"n={best_val_metrics['n_samples']} "
        f"mae_model={float(best_val_metrics['mae_model']):.6f} "
        f"rmse_model={float(best_val_metrics['rmse_model']):.6f} "
        f"r2={float(best_val_metrics['r2']):.6f} "
        f"mae_raw={float(best_val_metrics['mae_raw']):.6f} "
        f"rmse_raw={float(best_val_metrics['rmse_raw']):.6f} "
        f"max_abs_err_raw={float(best_val_metrics['max_abs_err_raw']):.6f} "
        f"p95_abs_err_raw={float(best_val_metrics['p95_abs_err_raw']):.6f} "
        f"p99_abs_err_raw={float(best_val_metrics['p99_abs_err_raw']):.6f} "
        f"tail5_mean_abs_err_raw={float(best_val_metrics['tail5_mean_abs_err_raw']):.6f}",
        flush=True,
    )
    print(
        f"{label} test: "
        f"n={test_metrics['n_samples']} "
        f"mae_model={float(test_metrics['mae_model']):.6f} "
        f"rmse_model={float(test_metrics['rmse_model']):.6f} "
        f"r2={float(test_metrics['r2']):.6f} "
        f"mae_raw={float(test_metrics['mae_raw']):.6f} "
        f"rmse_raw={float(test_metrics['rmse_raw']):.6f} "
        f"max_abs_err_raw={float(test_metrics['max_abs_err_raw']):.6f} "
        f"p95_abs_err_raw={float(test_metrics['p95_abs_err_raw']):.6f} "
        f"p99_abs_err_raw={float(test_metrics['p99_abs_err_raw']):.6f} "
        f"tail5_mean_abs_err_raw={float(test_metrics['tail5_mean_abs_err_raw']):.6f}",
        flush=True,
    )
    print_violation_report(f"{label}:val", val_violation_report)
    print_violation_report(f"{label}:test", test_violation_report)


def overall_violation_percent(report: dict | None) -> float | None:
    if report is None:
        return None
    try:
        rate = float(report.get("overall_violation_rate", float("nan")))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate):
        return None
    return rate * 100.0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train KAN from energigran TSV/CSV (hourly mean aggregation), "
            "target=generation, excluding target column from inputs."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_default_input_path(),
        help="Path to energigran TSV/CSV file",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=PROJECT_ROOT / "temp" / "test-models-alternative" / "SolarPlant" / "generation_h24_from_tsv.bin",
        help="Where to save trained model weights",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument(
        "--trainer-mode",
        choices=("KAN", "KAN-Mm", "tabnet", "tabnet-mr", "all"),
        default="all",
        help="Choose KAN, KAN-Mm, TabNet, TabNet-MR, or all",
    )
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--relation-constraint-weight", type=float, default=0.25)
    parser.add_argument("--worst-case-over-T-weight", type=float, default=0.0)
    parser.add_argument("--violation-atol", type=float, default=1e-6)
    parser.add_argument("--violation-rtol", type=float, default=1e-4)
    parser.add_argument("--tabnet-n-steps", type=int, default=4)
    parser.add_argument("--tabnet-n-d", type=int, default=24)
    parser.add_argument("--tabnet-n-a", type=int, default=24)
    parser.add_argument("--tabnet-gamma", type=float, default=1.3)
    parser.add_argument("--tabnet-dropout", type=float, default=0.05)
    parser.add_argument("--tabnet-mask-temperature", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.64)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--disable-shap",
        action="store_true",
        help="Disable SHAP feature attribution analysis.",
    )
    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=256,
        help="Max number of train samples used as SHAP background.",
    )
    parser.add_argument(
        "--shap-eval-size",
        type=int,
        default=256,
        help="Max number of test samples explained with SHAP.",
    )
    parser.add_argument(
        "--shap-top-k",
        type=int,
        default=15,
        help="How many top variables to print from SHAP ranking.",
    )
    parser.add_argument(
        "--shap-output",
        type=Path,
        default=PROJECT_ROOT / "temp" / "analysis" / "solarPlant" / "generation_h24_shap_importance.csv",
        help="CSV path for global SHAP importance report.",
    )
    parser.add_argument(
        "--limit-hours",
        type=int,
        default=None,
        help="Optional cap on number of hourly rows after aggregation (for quick tests)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    (
        AlternativeKanTrainer,
        MetamorphicAlternativeKanTrainer,
        TabNetAlternativeTrainer,
        DEFAULT_SOLAR_PLANT_RULE_WEIGHT_MAP,
        build_solar_plant_active_power_rule_specs,
        summarize_rule_specs,
    ) = _load_training_dependencies()

    table_path = args.csv.resolve()
    if not table_path.exists():
        raise FileNotFoundError(f"Input table not found: {table_path}")

    hourly_rows = load_hourly_means(table_path)
    if args.limit_hours is not None:
        hourly_rows = hourly_rows[: int(args.limit_hours)]
    if len(hourly_rows) < 10:
        raise ValueError(f"Need >=10 hourly rows, got {len(hourly_rows)}")
    horizon_examples = build_horizon_examples(hourly_rows, horizon_hours=int(args.horizon_hours))
    if len(horizon_examples) < 10:
        raise ValueError(
            f"Need >=10 horizon examples for +{args.horizon_hours}h forecast, got {len(horizon_examples)}"
        )

    train_rows, val_rows, test_rows = split_records(
        records=horizon_examples,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    means, stds = compute_feature_stats(train_rows)
    out_min, out_max = compute_target_scaler(train_rows)
    train_items = make_kan_items(train_rows, out_min=out_min, out_max=out_max)
    val_items = make_kan_items(val_rows, out_min=out_min, out_max=out_max)
    test_items = make_kan_items(test_rows, out_min=out_min, out_max=out_max)

    input_variables = list(TIME_FEATURE_NAMES) + list(INPUT_NUMERIC_COLUMNS)
    lookback = 0
    device = get_device()

    print(
        "config: "
        f"input={table_path} "
        f"horizon_hours={args.horizon_hours} "
        f"trainer_mode={args.trainer_mode} "
        f"epochs={args.epochs} "
        f"batch_size={args.batch_size} "
        f"lr={args.lr} "
        f"seed={args.seed} "
        f"shap_enabled={not args.disable_shap} "
        f"shap_background_size={args.shap_background_size} "
        f"shap_eval_size={args.shap_eval_size} "
        f"device={device}",
        flush=True,
    )
    print(
        "data: "
        f"hourly_rows={len(hourly_rows)} "
        f"horizon_examples={len(horizon_examples)} "
        f"train={len(train_rows)} "
        f"val={len(val_rows)} "
        f"test={len(test_rows)}",
        flush=True,
    )
    print(
        "features: "
        f"time={TIME_FEATURE_NAMES} "
        f"numerical_inputs={INPUT_NUMERIC_COLUMNS} "
        f"excluded={sorted(FORBIDDEN_INPUTS)} "
        f"target={TARGET_COLUMN}+{args.horizon_hours}h",
        flush=True,
    )
    print(
        "target_scaler: "
        f"out_min={out_min:.6f} "
        f"out_max={out_max:.6f}",
        flush=True,
    )

    rule_specs, effective_rule_weights, inactive_rule_weights = build_solar_plant_active_power_rule_specs(
        numerical_t_feature_names=INPUT_NUMERIC_COLUMNS,
        rule_weight_map=DEFAULT_SOLAR_PLANT_RULE_WEIGHT_MAP,
    )
    rule_summary = summarize_rule_specs(rule_specs)
    print(f"rule_weight_map={effective_rule_weights}", flush=True)
    if inactive_rule_weights:
        print(f"rule_weight_map_inactive={inactive_rule_weights}", flush=True)
    print(
        "rule_config: "
        f"supervised_weight={args.supervised_weight} "
        f"relation_constraint_weight={args.relation_constraint_weight} "
        f"worst_case_over_T_weight={args.worst_case_over_T_weight} "
        f"tabnet_steps={args.tabnet_n_steps} "
        f"tabnet_n_d={args.tabnet_n_d} "
        f"tabnet_n_a={args.tabnet_n_a} "
        f"tabnet_gamma={args.tabnet_gamma} "
        f"tabnet_dropout={args.tabnet_dropout}",
        flush=True,
    )
    print(
        "catalog: "
        f"specs={rule_summary['num_specs']} "
        f"relation_tests={rule_summary['num_relation_tests']} "
        f"over_T_transforms={rule_summary['num_over_T_transforms']} "
        f"by_category={rule_summary['by_category']}",
        flush=True,
    )

    trainer_name = "Energigran"
    model_out = args.model_out.resolve()
    os.makedirs(model_out.parent, exist_ok=True)
    branch_times_sec: dict[str, float] = {}
    branch_summary: dict[str, dict[str, float | None]] = {}
    shap_model = None
    shap_label = None

    if args.trainer_mode in ("KAN", "all"):
        t0 = time.perf_counter()
        baseline_trainer = AlternativeKanTrainer(
            name=f"{trainer_name}[KAN]",
            input_variables=input_variables,
            output_variable=f"{TARGET_COLUMN}+{args.horizon_hours}h",
            lookback=lookback,
            means=means,
            stds=stds,
            out_min=out_min,
            out_max=out_max,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            device=device,
            lr=float(args.lr),
            seed=int(args.seed),
        )
        baseline_model, baseline_best_val_metrics = baseline_trainer.train(train_items, val_items)
        baseline_test_metrics = baseline_trainer.evaluate(baseline_model, test_items)
        baseline_val_viol = evaluate_rule_violations(
            model=baseline_model,
            items=val_items,
            batch_size=int(args.batch_size),
            rule_specs=rule_specs,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        baseline_test_viol = evaluate_rule_violations(
            model=baseline_model,
            items=test_items,
            batch_size=int(args.batch_size),
            rule_specs=rule_specs,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        baseline_out = (
            model_out.with_name(f"{model_out.stem}_kan{model_out.suffix}")
            if args.trainer_mode == "all"
            else model_out
        )
        torch.save(baseline_model.state_dict(), baseline_out)
        print_model_summary(
            label="KAN",
            best_val_metrics=baseline_best_val_metrics,
            test_metrics=baseline_test_metrics,
            val_violation_report=baseline_val_viol,
            test_violation_report=baseline_test_viol,
        )
        print(f"model_saved[KAN]={baseline_out}", flush=True)
        elapsed = time.perf_counter() - t0
        branch_times_sec["KAN"] = elapsed
        branch_summary["KAN"] = {
            "mae_raw": float(baseline_test_metrics["mae_raw"]),
            "r2": float(baseline_test_metrics["r2"]),
            "max_abs_err_raw": float(baseline_test_metrics["max_abs_err_raw"]),
            "tail5_mean_abs_err_raw": float(baseline_test_metrics["tail5_mean_abs_err_raw"]),
            "violations_pct": overall_violation_percent(baseline_test_viol),
            "time_sec": float(elapsed),
        }
        print(f"elapsed[KAN]={elapsed:.2f}s", flush=True)
        shap_model = baseline_model
        shap_label = "KAN"

    if args.trainer_mode in ("KAN-Mm", "all"):
        t0 = time.perf_counter()
        metamorphic_trainer = MetamorphicAlternativeKanTrainer(
            name=f"{trainer_name}[KAN-Mm]",
            input_variables=input_variables,
            output_variable=f"{TARGET_COLUMN}+{args.horizon_hours}h",
            lookback=lookback,
            means=means,
            stds=stds,
            out_min=out_min,
            out_max=out_max,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            device=device,
            lr=float(args.lr),
            seed=int(args.seed),
            rule_specs=rule_specs,
            supervised_weight=float(args.supervised_weight),
            relation_constraint_weight=float(args.relation_constraint_weight),
            worst_case_over_T_weight=float(args.worst_case_over_T_weight),
        )
        metamorphic_model, metamorphic_best_val_metrics = metamorphic_trainer.train(train_items, val_items)
        _, metamorphic_val_viol = metamorphic_trainer.evaluate_with_rule_violations(
            model=metamorphic_model,
            items=val_items,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        metamorphic_test_metrics, metamorphic_test_viol = metamorphic_trainer.evaluate_with_rule_violations(
            model=metamorphic_model,
            items=test_items,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        metamorphic_out = (
            model_out.with_name(f"{model_out.stem}_kan_mm{model_out.suffix}")
            if args.trainer_mode == "all"
            else model_out
        )
        torch.save(metamorphic_model.state_dict(), metamorphic_out)
        print_model_summary(
            label="KAN-Mm",
            best_val_metrics=metamorphic_best_val_metrics,
            test_metrics=metamorphic_test_metrics,
            val_violation_report=metamorphic_val_viol,
            test_violation_report=metamorphic_test_viol,
        )
        print(f"model_saved[KAN-Mm]={metamorphic_out}", flush=True)
        elapsed = time.perf_counter() - t0
        branch_times_sec["KAN-Mm"] = elapsed
        branch_summary["KAN-Mm"] = {
            "mae_raw": float(metamorphic_test_metrics["mae_raw"]),
            "r2": float(metamorphic_test_metrics["r2"]),
            "max_abs_err_raw": float(metamorphic_test_metrics["max_abs_err_raw"]),
            "tail5_mean_abs_err_raw": float(metamorphic_test_metrics["tail5_mean_abs_err_raw"]),
            "violations_pct": overall_violation_percent(metamorphic_test_viol),
            "time_sec": float(elapsed),
        }
        print(f"elapsed[KAN-Mm]={elapsed:.2f}s", flush=True)
        if shap_model is None:
            shap_model = metamorphic_model
            shap_label = "KAN-Mm"

    if args.trainer_mode in ("tabnet", "all"):
        t0 = time.perf_counter()
        input_dim = len(input_variables)
        tabnet_trainer = TabNetAlternativeTrainer(
            name=f"{trainer_name}[tabnet]",
            out_min=out_min,
            out_max=out_max,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            device=device,
            lr=float(args.lr),
            seed=int(args.seed),
            input_dim=input_dim,
            n_d=int(args.tabnet_n_d),
            n_a=int(args.tabnet_n_a),
            n_steps=int(args.tabnet_n_steps),
            gamma=float(args.tabnet_gamma),
            dropout=float(args.tabnet_dropout),
            mask_temperature=float(args.tabnet_mask_temperature),
        )
        tabnet_model, tabnet_best_val_metrics = tabnet_trainer.train(train_items, val_items)
        tabnet_test_metrics = tabnet_trainer.evaluate(tabnet_model, test_items)
        tabnet_val_viol = evaluate_rule_violations(
            model=tabnet_model,
            items=val_items,
            batch_size=int(args.batch_size),
            rule_specs=rule_specs,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        tabnet_test_viol = evaluate_rule_violations(
            model=tabnet_model,
            items=test_items,
            batch_size=int(args.batch_size),
            rule_specs=rule_specs,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        tabnet_out = (
            model_out.with_name(f"{model_out.stem}_tabnet{model_out.suffix}")
            if args.trainer_mode == "all"
            else model_out
        )
        torch.save(tabnet_model.state_dict(), tabnet_out)
        print_model_summary(
            label="tabnet",
            best_val_metrics=tabnet_best_val_metrics,
            test_metrics=tabnet_test_metrics,
            val_violation_report=tabnet_val_viol,
            test_violation_report=tabnet_test_viol,
        )
        print(f"model_saved[tabnet]={tabnet_out}", flush=True)
        elapsed = time.perf_counter() - t0
        branch_times_sec["tabnet"] = elapsed
        branch_summary["tabnet"] = {
            "mae_raw": float(tabnet_test_metrics["mae_raw"]),
            "r2": float(tabnet_test_metrics["r2"]),
            "max_abs_err_raw": float(tabnet_test_metrics["max_abs_err_raw"]),
            "tail5_mean_abs_err_raw": float(tabnet_test_metrics["tail5_mean_abs_err_raw"]),
            "violations_pct": overall_violation_percent(tabnet_test_viol),
            "time_sec": float(elapsed),
        }
        print(f"elapsed[tabnet]={elapsed:.2f}s", flush=True)
        if shap_model is None:
            shap_model = tabnet_model
            shap_label = "tabnet"

    if args.trainer_mode in ("tabnet-mr", "all"):
        t0 = time.perf_counter()
        input_dim = len(input_variables)
        tabnet_mr_trainer = TabNetAlternativeTrainer(
            name=f"{trainer_name}[tabnet-mr]",
            out_min=out_min,
            out_max=out_max,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            device=device,
            lr=float(args.lr),
            seed=int(args.seed),
            input_dim=input_dim,
            n_d=int(args.tabnet_n_d),
            n_a=int(args.tabnet_n_a),
            n_steps=int(args.tabnet_n_steps),
            gamma=float(args.tabnet_gamma),
            dropout=float(args.tabnet_dropout),
            mask_temperature=float(args.tabnet_mask_temperature),
            rule_specs=rule_specs,
            supervised_weight=float(args.supervised_weight),
            relation_constraint_weight=float(args.relation_constraint_weight),
            worst_case_over_T_weight=float(args.worst_case_over_T_weight),
        )
        tabnet_mr_model, tabnet_mr_best_val_metrics = tabnet_mr_trainer.train(train_items, val_items)
        _, tabnet_mr_val_viol = tabnet_mr_trainer.evaluate_with_rule_violations(
            model=tabnet_mr_model,
            items=val_items,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        tabnet_mr_test_metrics, tabnet_mr_test_viol = tabnet_mr_trainer.evaluate_with_rule_violations(
            model=tabnet_mr_model,
            items=test_items,
            atol=float(args.violation_atol),
            rtol=float(args.violation_rtol),
        )
        tabnet_mr_out = (
            model_out.with_name(f"{model_out.stem}_tabnet_mr{model_out.suffix}")
            if args.trainer_mode == "all"
            else model_out
        )
        torch.save(tabnet_mr_model.state_dict(), tabnet_mr_out)
        print_model_summary(
            label="tabnet-mr",
            best_val_metrics=tabnet_mr_best_val_metrics,
            test_metrics=tabnet_mr_test_metrics,
            val_violation_report=tabnet_mr_val_viol,
            test_violation_report=tabnet_mr_test_viol,
        )
        print(f"model_saved[tabnet-mr]={tabnet_mr_out}", flush=True)
        elapsed = time.perf_counter() - t0
        branch_times_sec["tabnet-mr"] = elapsed
        branch_summary["tabnet-mr"] = {
            "mae_raw": float(tabnet_mr_test_metrics["mae_raw"]),
            "r2": float(tabnet_mr_test_metrics["r2"]),
            "max_abs_err_raw": float(tabnet_mr_test_metrics["max_abs_err_raw"]),
            "tail5_mean_abs_err_raw": float(tabnet_mr_test_metrics["tail5_mean_abs_err_raw"]),
            "violations_pct": overall_violation_percent(tabnet_mr_test_viol),
            "time_sec": float(elapsed),
        }
        print(f"elapsed[tabnet-mr]={elapsed:.2f}s", flush=True)
        if shap_model is None:
            shap_model = tabnet_mr_model
            shap_label = "tabnet-mr"

    if not args.disable_shap and shap_model is not None:
        top_rows = compute_shap_importance(
            model=shap_model,
            input_variables=input_variables,
            train_items=train_items,
            test_items=test_items,
            time_dim=len(TIME_FEATURE_NAMES),
            numeric_dim=len(INPUT_NUMERIC_COLUMNS),
            target_min=out_min,
            target_max=out_max,
            background_size=int(args.shap_background_size),
            eval_size=int(args.shap_eval_size),
            top_k=int(args.shap_top_k),
            seed=int(args.seed),
            output_path=args.shap_output,
        )
        print(f"shap_model={shap_label}", flush=True)
        print(f"shap_saved={args.shap_output.resolve()}", flush=True)
        print("shap_top_features:", flush=True)
        for row in top_rows:
            print(
                f"rank={int(row['rank'])} "
                f"feature={row['feature']} "
                f"mean_abs_shap_model={float(row['mean_abs_shap_model']):.8f} "
                f"mean_abs_shap_raw={float(row['mean_abs_shap_raw']):.8f} "
                f"share_percent={float(row['share_percent']):.4f}",
                flush=True,
            )

    if branch_times_sec:
        compact = " ".join([f"{name}={seconds:.2f}s" for name, seconds in branch_times_sec.items()])
        print(f"elapsed_summary: {compact}", flush=True)
    if branch_summary:
        print("branch_summary:", flush=True)
        print(
            "branch\tMAE_raw\tR2\tmax_abs_err_raw\ttail5_mean_abs_err_raw\tviolations_pct\ttime_s",
            flush=True,
        )
        for branch_name, metrics in branch_summary.items():
            mae_raw = float(metrics.get("mae_raw", float("nan")))
            r2 = float(metrics.get("r2", float("nan")))
            max_abs_err_raw = float(metrics.get("max_abs_err_raw", float("nan")))
            tail5_mean_abs_err_raw = float(metrics.get("tail5_mean_abs_err_raw", float("nan")))
            violations_pct = metrics.get("violations_pct")
            time_sec = float(metrics.get("time_sec", float("nan")))
            if violations_pct is None or not math.isfinite(float(violations_pct)):
                violations_text = "n/a"
            else:
                violations_text = f"{float(violations_pct):.2f}%"
            print(
                f"{branch_name}\t"
                f"{mae_raw:.6f}\t"
                f"{r2:.6f}\t"
                f"{max_abs_err_raw:.6f}\t"
                f"{tail5_mean_abs_err_raw:.6f}\t"
                f"{violations_text}\t"
                f"{time_sec:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
