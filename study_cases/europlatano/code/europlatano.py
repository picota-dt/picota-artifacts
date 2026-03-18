from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import numpy as np
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
    input_path = os.environ.get("PICOTA_EUROPLATANO_DATA")
    if input_path:
        candidates.append(Path(input_path))
    candidates.extend(
        [
            CASE_ROOT / "data" / "europlatano" / "europlatano.tsv",
            CASE_ROOT / "data" / "europlatano.tsv",
            PROJECT_ROOT / "runtime.test" / "data" / "europlatano.tsv",
            PROJECT_ROOT.parent / "picota" / "runtime.test" / "data" / "europlatano.tsv",
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
        from EuroplatanoRuleCatalog import (
            DEFAULT_EUROPLATANO_RULE_WEIGHT_MAP,
            build_europlatano_production_rule_specs,
        )
        from KanTrainer import AlternativeKanTrainer
        from MetamorphicAlternativeKanTrainer import MetamorphicAlternativeKanTrainer
        from TabNetTrainer import TabNetAlternativeTrainer
        from metamorphic_evaluation import compute_violation_report as _compute_violation_report
        from kan.MetamorphicCatalog import summarize_rule_specs
        from kan.TimeSeriesDataset import TimeSeriesDataset as _TimeSeriesDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import Europlátano training dependencies from the local code directory."
        ) from exc

    TimeSeriesDataset = _TimeSeriesDataset
    compute_violation_report = _compute_violation_report
    return (
        AlternativeKanTrainer,
        MetamorphicAlternativeKanTrainer,
        TabNetAlternativeTrainer,
        DEFAULT_EUROPLATANO_RULE_WEIGHT_MAP,
        build_europlatano_production_rule_specs,
        summarize_rule_specs,
    )


_prepend_sys_path(CODE_ROOT)

TimeSeriesDataset = None
compute_violation_report = None


TARGET_COLUMN = "Production"
TIMESTAMP_COLUMN = "instant"

TIME_FEATURE_NAMES = [
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "week_sin",
    "week_cos",
    "quarter_sin",
    "quarter_cos",
]

# We use these columns to link daily records across +28d for the same production unit.
ENTITY_KEY_COLUMNS = ["Category", "Island", "Area", "Altitude"]


@dataclass
class ParsedRow:
    instant: datetime
    numeric: dict[str, float]
    categorical: dict[str, str]
    target: float


@dataclass
class ExampleRow:
    instant: datetime
    numeric: dict[str, float]
    categorical: dict[str, str]
    target_future: float


class KANFlatWrapper(torch.nn.Module):
    def __init__(
        self,
        kan_model: torch.nn.Module,
        time_dim: int,
        numeric_dim: int,
        categorical_dim: int,
    ):
        super().__init__()
        self.kan_model = kan_model
        self.time_dim = int(time_dim)
        self.numeric_dim = int(numeric_dim)
        self.categorical_dim = int(categorical_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != self.time_dim + self.numeric_dim + self.categorical_dim:
            raise ValueError(
                "Input feature width does not match configured dimensions "
                f"({x.size(1)} != {self.time_dim + self.numeric_dim + self.categorical_dim})"
            )
        t_end = self.time_dim
        n_end = t_end + self.numeric_dim
        c_end = n_end + self.categorical_dim
        batch_size = x.size(0)
        zeros = torch.zeros((batch_size, 0), dtype=x.dtype, device=x.device)
        batch = {
            "t": x[:, :t_end],
            "numerical_t_features": x[:, t_end:n_end],
            "categorical_t_features": x[:, n_end:c_end],
            "lookback_t": zeros,
            "categorical_lookback_features": zeros,
            "numerical_lookback_features": zeros,
        }
        return self.kan_model(batch)


def parse_utc_day(value: str) -> datetime:
    ts = value.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_float(value: str) -> float:
    return float(str(value).strip())


def is_float_column(values: list[str]) -> bool:
    for value in values:
        v = str(value).strip()
        if v == "":
            continue
        try:
            float(v)
        except ValueError:
            return False
    return True


def sin_cos(value: float, period: float) -> tuple[float, float]:
    angle = (2.0 * math.pi * value) / period
    return math.sin(angle), math.cos(angle)


def encode_time_features(day_utc: datetime) -> list[float]:
    month_idx = day_utc.month - 1
    day_idx = day_utc.day - 1
    week_idx = day_utc.isocalendar().week - 1
    quarter_idx = (day_utc.month - 1) // 3

    month_sin, month_cos = sin_cos(month_idx, 12.0)
    day_sin, day_cos = sin_cos(day_idx, 31.0)
    week_sin, week_cos = sin_cos(week_idx, 53.0)
    quarter_sin, quarter_cos = sin_cos(quarter_idx, 4.0)

    return [
        month_sin,
        month_cos,
        day_sin,
        day_cos,
        week_sin,
        week_cos,
        quarter_sin,
        quarter_cos,
    ]


def load_rows(tsv_path: Path) -> tuple[list[ParsedRow], list[str], list[str]]:
    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = list(reader.fieldnames or [])
        required = {TIMESTAMP_COLUMN, TARGET_COLUMN}
        missing = required - set(headers)
        if missing:
            raise ValueError(f"TSV missing required columns: {sorted(missing)}")

        raw_rows = list(reader)

    candidate_cols = [c for c in headers if c not in (TIMESTAMP_COLUMN, TARGET_COLUMN)]
    samples_by_col: dict[str, list[str]] = defaultdict(list)
    for row in raw_rows:
        for col in candidate_cols:
            samples_by_col[col].append(row[col])

    numeric_columns = [c for c in candidate_cols if is_float_column(samples_by_col[c])]
    categorical_columns = [c for c in candidate_cols if c not in numeric_columns]

    parsed_rows: list[ParsedRow] = []
    for row in raw_rows:
        try:
            instant = parse_utc_day(row[TIMESTAMP_COLUMN])
            target = parse_float(row[TARGET_COLUMN])
            numeric = {col: parse_float(row[col]) for col in numeric_columns}
            categorical = {col: str(row[col]).strip() for col in categorical_columns}
        except (TypeError, ValueError):
            continue
        parsed_rows.append(
            ParsedRow(
                instant=instant,
                numeric=numeric,
                categorical=categorical,
                target=target,
            )
        )
    return parsed_rows, numeric_columns, categorical_columns


def entity_key(row: ParsedRow) -> tuple[str, ...]:
    key_parts = []
    for col in ENTITY_KEY_COLUMNS:
        if col in row.categorical:
            key_parts.append(row.categorical[col])
        elif col in row.numeric:
            key_parts.append(f"{row.numeric[col]:.6f}")
        else:
            key_parts.append("")
    return tuple(key_parts)


def aggregate_duplicates(rows: list[ParsedRow], numeric_columns: list[str], categorical_columns: list[str]) -> list[ParsedRow]:
    grouped: dict[tuple[datetime, tuple[str, ...]], list[ParsedRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.instant, entity_key(row))].append(row)

    aggregated: list[ParsedRow] = []
    for (_instant, _entity_key), items in grouped.items():
        first = items[0]
        count = float(len(items))
        numeric = {}
        for col in numeric_columns:
            numeric[col] = float(sum(item.numeric[col] for item in items) / count)
        categorical = {}
        for col in categorical_columns:
            counter = Counter(item.categorical[col] for item in items)
            categorical[col] = counter.most_common(1)[0][0]
        target = float(sum(item.target for item in items) / count)
        aggregated.append(
            ParsedRow(
                instant=first.instant,
                numeric=numeric,
                categorical=categorical,
                target=target,
            )
        )
    aggregated.sort(key=lambda r: (r.instant, entity_key(r)))
    return aggregated


def build_horizon_examples(rows: list[ParsedRow], horizon_days: int) -> list[ExampleRow]:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")

    by_entity_day: dict[tuple[tuple[str, ...], datetime], ParsedRow] = {}
    for row in rows:
        by_entity_day[(entity_key(row), row.instant)] = row

    examples: list[ExampleRow] = []
    for row in rows:
        key = entity_key(row)
        future_day = row.instant + timedelta(days=int(horizon_days))
        future_row = by_entity_day.get((key, future_day))
        if future_row is None:
            continue
        examples.append(
            ExampleRow(
                instant=row.instant,
                numeric=row.numeric,
                categorical=row.categorical,
                target_future=float(future_row.target),
            )
        )
    return examples


def split_rows(
    rows: list[ExampleRow],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[ExampleRow], list[ExampleRow], list[ExampleRow]]:
    if len(rows) < 10:
        raise ValueError(f"Need >=10 rows after horizon pairing, got {len(rows)}")
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)

    train_end = max(1, int(len(rows) * train_ratio))
    val_end = min(len(rows) - 1, train_end + max(1, int(len(rows) * val_ratio)))
    if train_end >= val_end:
        train_end = max(1, val_end - 1)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    if len(test_idx) == 0:
        test_idx = idx[-1:]
        val_idx = idx[train_end:-1]
        if len(val_idx) == 0:
            val_idx = idx[train_end - 1 : train_end]
            train_idx = idx[: train_end - 1]

    train = [rows[int(i)] for i in train_idx]
    val = [rows[int(i)] for i in val_idx]
    test = [rows[int(i)] for i in test_idx]
    return train, val, test


def fit_minmax_numeric(train_rows: list[ExampleRow], numeric_columns: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    mins = {}
    maxs = {}
    for col in numeric_columns:
        values = [row.numeric[col] for row in train_rows]
        col_min = float(min(values))
        col_max = float(max(values))
        if col_max <= col_min:
            col_max = col_min + 1.0
        mins[col] = col_min
        maxs[col] = col_max
    return mins, maxs


def fit_minmax_target(train_rows: list[ExampleRow]) -> tuple[float, float]:
    values = [row.target_future for row in train_rows]
    out_min = float(min(values))
    out_max = float(max(values))
    if out_max <= out_min:
        out_max = out_min + 1.0
    return out_min, out_max


def build_one_hot_maps(train_rows: list[ExampleRow], categorical_columns: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for col in categorical_columns:
        values = sorted({row.categorical[col] for row in train_rows})
        mapping[col] = values
    return mapping


def one_hot_encode(values_by_col: dict[str, str], one_hot_map: dict[str, list[str]]) -> list[float]:
    encoded: list[float] = []
    for col, categories in one_hot_map.items():
        current = values_by_col.get(col, "")
        for cat in categories:
            encoded.append(1.0 if current == cat else 0.0)
    return encoded


def normalize_minmax(value: float, min_value: float, max_value: float) -> float:
    return float((value - min_value) / (max_value - min_value))


def make_items(
    rows: list[ExampleRow],
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_mins: dict[str, float],
    numeric_maxs: dict[str, float],
    target_min: float,
    target_max: float,
    one_hot_map: dict[str, list[str]],
) -> list[dict]:
    _ = categorical_columns
    items: list[dict] = []
    for row in rows:
        t_features = encode_time_features(row.instant)
        numerical_t_features = [
            normalize_minmax(row.numeric[col], numeric_mins[col], numeric_maxs[col])
            for col in numeric_columns
        ]
        categorical_t_features = one_hot_encode(row.categorical, one_hot_map)
        out = normalize_minmax(row.target_future, target_min, target_max)
        items.append(
            {
                "out": float(out),
                "t": [float(v) for v in t_features],
                "categorical_t_features": [float(v) for v in categorical_t_features],
                "numerical_t_features": [float(v) for v in numerical_t_features],
                "lookback_t": [],
                "categorical_lookback_features": [],
                "numerical_lookback_features": [],
            }
        )
    return items


def flatten_items(items: list[dict]) -> np.ndarray:
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    flat = []
    for item in items:
        row = list(item["t"]) + list(item["numerical_t_features"]) + list(item["categorical_t_features"])
        flat.append(row)
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
    categorical_dim: int,
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
    wrapped = KANFlatWrapper(
        kan_model=model_cpu,
        time_dim=time_dim,
        numeric_dim=numeric_dim,
        categorical_dim=categorical_dim,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train europlatano KAN baselines and metamorphic KAN from produccion.tsv "
            "with +28d horizon, minmax normalization, cyclical time encoding and optional SHAP."
        )
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=_default_input_path(),
        help="Path to europlatano.tsv",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=PROJECT_ROOT / "temp" / "test-models-alternative" / "Europlatano" / "Production_h28_from_tsv.bin",
        help="Where to save trained model weights",
    )
    parser.add_argument(
        "--trainer-mode",
        choices=("KAN", "KAN-Mm", "tabnet", "tabnet-mm", "all"),
        default="all",
        help="Train KAN, KAN-Mm, tabnet, tabnet-mm, or all.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-days", type=int, default=28)
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
        default=PROJECT_ROOT / "temp" / "analysis" / "europlatano" / "Production_h28_shap_importance.csv",
        help="CSV path for global SHAP importance report.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional cap on raw parsed rows for quick tests",
    )
    return parser


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = build_parser().parse_args()
    (
        AlternativeKanTrainer,
        MetamorphicAlternativeKanTrainer,
        TabNetAlternativeTrainer,
        DEFAULT_EUROPLATANO_RULE_WEIGHT_MAP,
        build_europlatano_production_rule_specs,
        summarize_rule_specs,
    ) = _load_training_dependencies()
    tsv_path = args.tsv.resolve()
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    parsed_rows, numeric_columns, categorical_columns = load_rows(tsv_path)
    if args.limit_rows is not None:
        parsed_rows = parsed_rows[: int(args.limit_rows)]
    if len(parsed_rows) < 10:
        raise ValueError(f"Need >=10 parsed rows, got {len(parsed_rows)}")

    aggregated_rows = aggregate_duplicates(parsed_rows, numeric_columns, categorical_columns)
    horizon_rows = build_horizon_examples(aggregated_rows, horizon_days=int(args.horizon_days))
    if len(horizon_rows) < 10:
        raise ValueError(
            f"Need >=10 rows after +{args.horizon_days}d pairing, got {len(horizon_rows)}"
        )

    train_rows, val_rows, test_rows = split_rows(
        rows=horizon_rows,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    numeric_mins, numeric_maxs = fit_minmax_numeric(train_rows, numeric_columns)
    target_min, target_max = fit_minmax_target(train_rows)
    one_hot_map = build_one_hot_maps(train_rows, categorical_columns)

    train_items = make_items(
        rows=train_rows,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_mins=numeric_mins,
        numeric_maxs=numeric_maxs,
        target_min=target_min,
        target_max=target_max,
        one_hot_map=one_hot_map,
    )
    val_items = make_items(
        rows=val_rows,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_mins=numeric_mins,
        numeric_maxs=numeric_maxs,
        target_min=target_min,
        target_max=target_max,
        one_hot_map=one_hot_map,
    )
    test_items = make_items(
        rows=test_rows,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_mins=numeric_mins,
        numeric_maxs=numeric_maxs,
        target_min=target_min,
        target_max=target_max,
        one_hot_map=one_hot_map,
    )

    one_hot_feature_names: list[str] = []
    for col, categories in one_hot_map.items():
        for cat in categories:
            one_hot_feature_names.append(f"{col}={cat}")

    input_variables = list(TIME_FEATURE_NAMES) + list(numeric_columns) + list(one_hot_feature_names)

    # Numerical features are already minmax-normalized in [0, 1], so keep KAN normalization neutral.
    means = [0.0 for _ in numeric_columns]
    stds = [1.0 for _ in numeric_columns]
    out_min = float(target_min)
    out_max = float(target_max)
    device = get_device()

    print(
        "config: "
        f"tsv={tsv_path} "
        f"horizon_days={args.horizon_days} "
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
        f"parsed_rows={len(parsed_rows)} "
        f"aggregated_rows={len(aggregated_rows)} "
        f"horizon_rows={len(horizon_rows)} "
        f"train={len(train_rows)} "
        f"val={len(val_rows)} "
        f"test={len(test_rows)}",
        flush=True,
    )
    print(
        "features: "
        f"time={TIME_FEATURE_NAMES} "
        f"numerical={numeric_columns} "
        f"categorical={categorical_columns} "
        f"onehot_dim={len(one_hot_feature_names)} "
        f"target={TARGET_COLUMN}+{args.horizon_days}d",
        flush=True,
    )
    print(
        "target_minmax: "
        f"min={target_min:.6f} max={target_max:.6f}",
        flush=True,
    )
    rule_specs, effective_rule_weights, inactive_rule_weights = build_europlatano_production_rule_specs(
        numerical_t_feature_names=numeric_columns,
        rule_weight_map=DEFAULT_EUROPLATANO_RULE_WEIGHT_MAP,
        raw_output_min=out_min,
        raw_output_max=out_max,
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
        f"violation_atol={args.violation_atol} "
        f"violation_rtol={args.violation_rtol} "
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

    trainer_name = "Europlatano"
    model_out = args.model_out.resolve()
    os.makedirs(model_out.parent, exist_ok=True)
    shap_model = None
    shap_label = None
    branch_times_sec: dict[str, float] = {}
    branch_summary: dict[str, dict[str, float | None]] = {}

    if args.trainer_mode in ("KAN", "all"):
        t0 = time.perf_counter()
        baseline_trainer = AlternativeKanTrainer(
            name=f"{trainer_name}[KAN]",
            input_variables=input_variables,
            output_variable=f"{TARGET_COLUMN}+{args.horizon_days}d",
            lookback=0,
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
            output_variable=f"{TARGET_COLUMN}+{args.horizon_days}d",
            lookback=0,
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

    if args.trainer_mode in ("tabnet-mm", "all"):
        t0 = time.perf_counter()
        input_dim = len(input_variables)
        tabnet_mr_trainer = TabNetAlternativeTrainer(
            name=f"{trainer_name}[tabnet-mm]",
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
            label="tabnet-mm",
            best_val_metrics=tabnet_mr_best_val_metrics,
            test_metrics=tabnet_mr_test_metrics,
            val_violation_report=tabnet_mr_val_viol,
            test_violation_report=tabnet_mr_test_viol,
        )
        print(f"model_saved[tabnet-mm]={tabnet_mr_out}", flush=True)
        elapsed = time.perf_counter() - t0
        branch_times_sec["tabnet-mm"] = elapsed
        branch_summary["tabnet-mm"] = {
            "mae_raw": float(tabnet_mr_test_metrics["mae_raw"]),
            "r2": float(tabnet_mr_test_metrics["r2"]),
            "max_abs_err_raw": float(tabnet_mr_test_metrics["max_abs_err_raw"]),
            "tail5_mean_abs_err_raw": float(tabnet_mr_test_metrics["tail5_mean_abs_err_raw"]),
            "violations_pct": overall_violation_percent(tabnet_mr_test_viol),
            "time_sec": float(elapsed),
        }
        print(f"elapsed[tabnet-mm]={elapsed:.2f}s", flush=True)
        if shap_model is None:
            shap_model = tabnet_mr_model
            shap_label = "tabnet-mm"

    if not args.disable_shap and shap_model is not None:
        top_rows = compute_shap_importance(
            model=shap_model,
            input_variables=input_variables,
            train_items=train_items,
            test_items=test_items,
            time_dim=len(TIME_FEATURE_NAMES),
            numeric_dim=len(numeric_columns),
            categorical_dim=len(one_hot_feature_names),
            target_min=target_min,
            target_max=target_max,
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
