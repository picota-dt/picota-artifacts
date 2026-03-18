#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class ParsedInputRow:
    day: str
    month: str
    fca: str
    category: str
    production: float
    area: float | None
    altitude: float | None
    island: str
    weather_values: dict[str, float | None]


def _instant_to_day(instant_value: str) -> str:
    cleaned = instant_value.strip()
    if len(cleaned) >= 10 and cleaned[4] == "-" and cleaned[7] == "-":
        return cleaned[:10]
    return ""


def _day_to_month(day_value: str) -> str:
    if len(day_value) == 10 and day_value[4] == "-" and day_value[7] == "-":
        return day_value[:7]
    return ""


def _day_index_to_iso_instant(day_value: str, index_in_day: int, rows_in_day: int) -> str:
    if rows_in_day <= 0:
        dt = datetime.strptime(day_value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if rows_in_day <= 24:
        hour = index_in_day
        minute = 0
        second = 0
        millisecond = 0
    else:
        total_ms_in_day = 24 * 60 * 60 * 1000
        ms_of_day = int((index_in_day * total_ms_in_day) // rows_in_day)
        hour, rem_ms = divmod(ms_of_day, 60 * 60 * 1000)
        minute, rem_ms = divmod(rem_ms, 60 * 1000)
        second, millisecond = divmod(rem_ms, 1000)

    dt = datetime.strptime(day_value, "%Y-%m-%d").replace(
        tzinfo=timezone.utc,
        hour=hour,
        minute=minute,
        second=second,
        microsecond=millisecond * 1000,
    )
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _month_to_iso_instant(month_value: str) -> str:
    dt = datetime.strptime(f"{month_value}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _month_index_to_iso_instant(month_value: str, index_in_month: int, rows_in_month: int) -> str:
    base = datetime.strptime(f"{month_value}-01", "%Y-%m-%d").replace(
        tzinfo=timezone.utc,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    if rows_in_month <= 0:
        return base.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if base.month == 12:
        next_month = base.replace(year=base.year + 1, month=1, day=1)
    else:
        next_month = base.replace(month=base.month + 1, day=1)

    total_ms_in_month = int((next_month - base).total_seconds() * 1000)
    ms_instant = int((index_in_month * total_ms_in_month) // rows_in_month)
    instant_dt = base + timedelta(milliseconds=ms_instant)
    return instant_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned.replace(",", "."))
    except ValueError:
        return None


def _format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _category_to_column(category_value: str) -> str:
    normalized = " ".join(category_value.strip().split())
    return f"Production:{normalized}"


def _first_existing(fieldnames: list[str], candidates: list[str]) -> str | None:
    fieldname_set = set(fieldnames)
    for candidate in candidates:
        if candidate in fieldname_set:
            return candidate
    return None


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]

    position = (len(sorted_values) - 1) * q
    low_index = int(position)
    high_index = min(low_index + 1, len(sorted_values) - 1)
    low_value = sorted_values[low_index]
    high_value = sorted_values[high_index]
    weight = position - low_index
    return low_value + (high_value - low_value) * weight


def _filter_outliers_iqr(
        rows: list[ParsedInputRow], multiplier: float
) -> tuple[list[ParsedInputRow], int, dict[str, tuple[float, float]]]:
    production_by_category: dict[str, list[float]] = {}
    for row in rows:
        production_by_category.setdefault(row.category, []).append(row.production)

    bounds_by_category: dict[str, tuple[float, float]] = {}
    for category, values in production_by_category.items():
        if len(values) < 4:
            continue
        sorted_values = sorted(values)
        q1 = _quantile(sorted_values, 0.25)
        q3 = _quantile(sorted_values, 0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        bounds_by_category[category] = (lower_bound, upper_bound)

    filtered_rows: list[ParsedInputRow] = []
    removed_rows = 0
    for row in rows:
        bounds = bounds_by_category.get(row.category)
        if bounds is None:
            filtered_rows.append(row)
            continue
        lower_bound, upper_bound = bounds
        if row.production < lower_bound or row.production > upper_bound:
            removed_rows += 1
            continue
        filtered_rows.append(row)

    return filtered_rows, removed_rows, bounds_by_category


def _mode_text(values: list[str]) -> str:
    filtered = [value.strip() for value in values if value and value.strip()]
    if not filtered:
        return ""
    counts = Counter(filtered)
    max_count = max(counts.values())
    top_values = sorted(value for value, count in counts.items() if count == max_count)
    return top_values[0]


def _mode_numeric(values: list[float]) -> str:
    if not values:
        return ""
    as_text = [_format_float(value) for value in values]
    return _mode_text(as_text)


def _new_bucket(weather_columns: list[str], split_category: bool) -> dict[str, object]:
    bucket: dict[str, object] = {
        "area_values": [],
        "altitude_values": [],
        "island_values": [],
        "weather_sum": {column: 0.0 for column in weather_columns},
        "weather_count": {column: 0 for column in weather_columns},
    }
    if split_category:
        bucket["production_by_category"] = {}
    else:
        bucket["production"] = 0.0
    return bucket


def _update_bucket(
        bucket: dict[str, object],
        weather_columns: list[str],
        weather_values: dict[str, float | None],
        area_value: float | None,
        altitude_value: float | None,
        island_value: str,
        production_value: float,
        category_value: str,
        split_category: bool,
) -> None:
    if area_value is not None:
        area_values = bucket["area_values"]
        area_values.append(area_value)
    if altitude_value is not None:
        altitude_values = bucket["altitude_values"]
        altitude_values.append(altitude_value)
    if island_value:
        island_values = bucket["island_values"]
        island_values.append(island_value)

    weather_sum = bucket["weather_sum"]
    weather_count = bucket["weather_count"]
    for column in weather_columns:
        value = weather_values.get(column)
        if value is None:
            continue
        weather_sum[column] = float(weather_sum[column]) + value
        weather_count[column] = int(weather_count[column]) + 1

    if split_category:
        production_by_category = bucket["production_by_category"]
        production_column = _category_to_column(category_value)
        production_by_category[production_column] = float(
            production_by_category.get(production_column, 0.0)) + production_value
    else:
        bucket["production"] = float(bucket["production"]) + production_value


def _weather_mean(bucket: dict[str, object], column: str) -> str:
    weather_sum = bucket["weather_sum"]
    weather_count = bucket["weather_count"]
    count = int(weather_count.get(column, 0))
    if count <= 0:
        return ""
    total = float(weather_sum.get(column, 0.0))
    return _format_float(total / count)


def _write_day_split_tsv(
        output_tsv: Path,
        aggregates: dict[tuple[str, str], dict[str, object]],
        weather_columns: list[str],
        production_columns: list[str],
) -> tuple[int, int]:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(aggregates.keys())
    rows_by_day: Counter[str] = Counter()
    for day_value, _fca in sorted_keys:
        rows_by_day[day_value] += 1
    assigned_by_day: dict[str, int] = {}
    days_over_24_rows = sum(1 for count in rows_by_day.values() if count > 24)

    output_columns = ["instant", "Altitude", "Area", "Island"] + weather_columns + production_columns
    with output_tsv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(output_columns)
        for day_value, fca in sorted_keys:
            bucket = aggregates[(day_value, fca)]
            index_in_day = assigned_by_day.get(day_value, 0)
            assigned_by_day[day_value] = index_in_day + 1
            instant_value = _day_index_to_iso_instant(day_value, index_in_day, rows_by_day[day_value])

            row = [
                instant_value,
                _mode_numeric(bucket["altitude_values"]),
                _mode_numeric(bucket["area_values"]),
                _mode_text(bucket["island_values"]),
            ]
            for weather_column in weather_columns:
                row.append(_weather_mean(bucket, weather_column))
            production_by_category = bucket["production_by_category"]
            for production_column in production_columns:
                row.append(_format_float(float(production_by_category.get(production_column, 0.0))))
            writer.writerow(row)
    return len(sorted_keys), days_over_24_rows


def _write_day_no_split_tsv(
        output_tsv: Path,
        aggregates: dict[tuple[str, str, str], dict[str, object]],
        weather_columns: list[str],
) -> tuple[int, int]:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(aggregates.keys())
    rows_by_day: Counter[str] = Counter()
    for day_value, _fca, _category in sorted_keys:
        rows_by_day[day_value] += 1
    assigned_by_day: dict[str, int] = {}
    days_over_24_rows = sum(1 for count in rows_by_day.values() if count > 24)

    output_columns = ["instant", "Category", "Production", "Altitude", "Area", "Island"] + weather_columns
    with output_tsv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(output_columns)
        for day_value, fca, category in sorted_keys:
            bucket = aggregates[(day_value, fca, category)]
            index_in_day = assigned_by_day.get(day_value, 0)
            assigned_by_day[day_value] = index_in_day + 1
            instant_value = _day_index_to_iso_instant(day_value, index_in_day, rows_by_day[day_value])

            row = [
                instant_value,
                category,
                _format_float(float(bucket["production"])),
                _mode_numeric(bucket["altitude_values"]),
                _mode_numeric(bucket["area_values"]),
                _mode_text(bucket["island_values"]),
            ]
            for weather_column in weather_columns:
                row.append(_weather_mean(bucket, weather_column))
            writer.writerow(row)
    return len(sorted_keys), days_over_24_rows


def _write_month_split_tsv(
        output_tsv: Path,
        aggregates: dict[tuple[str, str], dict[str, object]],
        weather_columns: list[str],
        production_columns: list[str],
) -> int:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(aggregates.keys())
    rows_by_month: Counter[str] = Counter()
    for month_value, _fca in sorted_keys:
        rows_by_month[month_value] += 1
    assigned_by_month: dict[str, int] = {}

    output_columns = ["instant", "Altitude", "Area", "Island"] + weather_columns + production_columns
    with output_tsv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(output_columns)
        for month_value, fca in sorted_keys:
            bucket = aggregates[(month_value, fca)]
            index_in_month = assigned_by_month.get(month_value, 0)
            assigned_by_month[month_value] = index_in_month + 1
            row = [
                _month_index_to_iso_instant(month_value, index_in_month, rows_by_month[month_value]),
                _mode_numeric(bucket["altitude_values"]),
                _mode_numeric(bucket["area_values"]),
                _mode_text(bucket["island_values"]),
            ]
            for weather_column in weather_columns:
                row.append(_weather_mean(bucket, weather_column))
            production_by_category = bucket["production_by_category"]
            for production_column in production_columns:
                row.append(_format_float(float(production_by_category.get(production_column, 0.0))))
            writer.writerow(row)
    return len(sorted_keys)


def _write_month_no_split_tsv(
        output_tsv: Path,
        aggregates: dict[tuple[str, str, str], dict[str, object]],
        weather_columns: list[str],
) -> int:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(aggregates.keys())
    rows_by_month: Counter[str] = Counter()
    for month_value, _fca, _category in sorted_keys:
        rows_by_month[month_value] += 1
    assigned_by_month: dict[str, int] = {}

    output_columns = ["instant", "Category", "Production", "Altitude", "Area", "Island"] + weather_columns
    with output_tsv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(output_columns)
        for month_value, fca, category in sorted_keys:
            bucket = aggregates[(month_value, fca, category)]
            index_in_month = assigned_by_month.get(month_value, 0)
            assigned_by_month[month_value] = index_in_month + 1
            row = [
                _month_index_to_iso_instant(month_value, index_in_month, rows_by_month[month_value]),
                category,
                _format_float(float(bucket["production"])),
                _mode_numeric(bucket["altitude_values"]),
                _mode_numeric(bucket["area_values"]),
                _mode_text(bucket["island_values"]),
            ]
            for weather_column in weather_columns:
                row.append(_weather_mean(bucket, weather_column))
            writer.writerow(row)
    return len(sorted_keys)


def main() -> int:
    default_input = Path(__file__).resolve().parents[2] / "data" / "europlatano" / "europlatano.tsv"
    default_output_root = Path(__file__).resolve().parents[2] / "data" / "europlatano" / "datasets"

    parser = argparse.ArgumentParser(
        description="Generate day/month Europlatano TSV projections with and without category split.")
    parser.add_argument("--input-tsv", default=str(default_input), help="Input TSV path.")
    parser.add_argument("--output-root", default=str(default_output_root),
                        help="Output root directory for projections.")
    parser.add_argument(
        "--outlier-method",
        choices=["iqr", "none"],
        default="iqr",
        help="Outlier detection method applied before generating datasets.",
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for production outlier filtering (per category).",
    )
    args = parser.parse_args()

    input_tsv = Path(args.input_tsv)
    output_root = Path(args.output_root)
    outlier_method = str(args.outlier_method)
    outlier_iqr_multiplier = float(args.outlier_iqr_multiplier)

    if not input_tsv.exists():
        raise SystemExit(f"Input TSV not found: {input_tsv}")
    if outlier_iqr_multiplier <= 0:
        raise SystemExit("--outlier-iqr-multiplier must be > 0.")

    required_columns = {"instant", "fca", "Category", "Production"}
    day_split: dict[tuple[str, str], dict[str, object]] = {}
    day_no_split: dict[tuple[str, str, str], dict[str, object]] = {}
    month_split: dict[tuple[str, str], dict[str, object]] = {}
    month_no_split: dict[tuple[str, str, str], dict[str, object]] = {}
    production_columns: set[str] = set()
    parsed_rows: list[ParsedInputRow] = []
    total_rows = 0
    invalid_rows = 0

    with input_tsv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        missing_required = sorted(required_columns - set(fieldnames))
        if missing_required:
            raise SystemExit("Missing required columns: " + ", ".join(missing_required))

        area_column = _first_existing(fieldnames, ["Area", "Superficie"])
        altitude_column = _first_existing(fieldnames, ["Altitude", "Altura"])
        island_column = _first_existing(fieldnames, ["Island", "ISLA", "Isla"])
        if area_column is None:
            raise SystemExit("Missing required area column (Area or Superficie).")
        if altitude_column is None:
            raise SystemExit("Missing required altitude column (Altitude or Altura).")

        weather_columns = [column for column in fieldnames if column.startswith("Territory.")]

        for row in reader:
            total_rows += 1
            day_value = _instant_to_day(str(row.get("instant", "")))
            month_value = _day_to_month(day_value)
            fca = str(row.get("fca", "") or "").strip()
            category = str(row.get("Category", "") or "").strip()
            production = _parse_float(str(row.get("Production", "") or ""))
            area = _parse_float(str(row.get(area_column, "") or ""))
            altitude = _parse_float(str(row.get(altitude_column, "") or ""))
            island = str(row.get(island_column, "") or "").strip() if island_column else ""

            if not day_value or not month_value or not fca or not category or production is None:
                invalid_rows += 1
                continue

            weather_values = {column: _parse_float(str(row.get(column, "") or "")) for column in weather_columns}
            parsed_rows.append(
                ParsedInputRow(
                    day=day_value,
                    month=month_value,
                    fca=fca,
                    category=category,
                    production=production,
                    area=area,
                    altitude=altitude,
                    island=island,
                    weather_values=weather_values,
                )
            )

    outlier_rows = 0
    outlier_categories_with_bounds = 0
    rows_for_aggregation = parsed_rows
    if outlier_method == "iqr":
        rows_for_aggregation, outlier_rows, bounds_by_category = _filter_outliers_iqr(
            parsed_rows,
            multiplier=outlier_iqr_multiplier,
        )
        outlier_categories_with_bounds = len(bounds_by_category)

    for parsed_row in rows_for_aggregation:
        production_columns.add(_category_to_column(parsed_row.category))

        day_split_key = (parsed_row.day, parsed_row.fca)
        day_split_bucket = day_split.get(day_split_key)
        if day_split_bucket is None:
            day_split_bucket = _new_bucket(weather_columns, split_category=True)
            day_split[day_split_key] = day_split_bucket
        _update_bucket(
            day_split_bucket,
            weather_columns=weather_columns,
            weather_values=parsed_row.weather_values,
            area_value=parsed_row.area,
            altitude_value=parsed_row.altitude,
            island_value=parsed_row.island,
            production_value=parsed_row.production,
            category_value=parsed_row.category,
            split_category=True,
        )

        day_no_split_key = (parsed_row.day, parsed_row.fca, parsed_row.category)
        day_no_split_bucket = day_no_split.get(day_no_split_key)
        if day_no_split_bucket is None:
            day_no_split_bucket = _new_bucket(weather_columns, split_category=False)
            day_no_split[day_no_split_key] = day_no_split_bucket
        _update_bucket(
            day_no_split_bucket,
            weather_columns=weather_columns,
            weather_values=parsed_row.weather_values,
            area_value=parsed_row.area,
            altitude_value=parsed_row.altitude,
            island_value=parsed_row.island,
            production_value=parsed_row.production,
            category_value=parsed_row.category,
            split_category=False,
        )

        month_split_key = (parsed_row.month, parsed_row.fca)
        month_split_bucket = month_split.get(month_split_key)
        if month_split_bucket is None:
            month_split_bucket = _new_bucket(weather_columns, split_category=True)
            month_split[month_split_key] = month_split_bucket
        _update_bucket(
            month_split_bucket,
            weather_columns=weather_columns,
            weather_values=parsed_row.weather_values,
            area_value=parsed_row.area,
            altitude_value=parsed_row.altitude,
            island_value=parsed_row.island,
            production_value=parsed_row.production,
            category_value=parsed_row.category,
            split_category=True,
        )

        month_no_split_key = (parsed_row.month, parsed_row.fca, parsed_row.category)
        month_no_split_bucket = month_no_split.get(month_no_split_key)
        if month_no_split_bucket is None:
            month_no_split_bucket = _new_bucket(weather_columns, split_category=False)
            month_no_split[month_no_split_key] = month_no_split_bucket
        _update_bucket(
            month_no_split_bucket,
            weather_columns=weather_columns,
            weather_values=parsed_row.weather_values,
            area_value=parsed_row.area,
            altitude_value=parsed_row.altitude,
            island_value=parsed_row.island,
            production_value=parsed_row.production,
            category_value=parsed_row.category,
            split_category=False,
        )

    sorted_production_columns = sorted(production_columns)
    day_split_path = output_root / "day" / "split_category" / "produccion_agregada_cat.tsv"
    day_no_split_path = output_root / "day" / "no_split" / "produccion_agregada.tsv"
    month_split_path = output_root / "month" / "split_category" / "produccion_agregada_cat.tsv"
    month_no_split_path = output_root / "month" / "no_split" / "produccion_agregada.tsv"

    day_split_rows, day_split_days_over_24 = _write_day_split_tsv(
        day_split_path,
        day_split,
        weather_columns=weather_columns,
        production_columns=sorted_production_columns,
    )
    day_no_split_rows, day_no_split_days_over_24 = _write_day_no_split_tsv(
        day_no_split_path,
        day_no_split,
        weather_columns=weather_columns,
    )
    month_split_rows = _write_month_split_tsv(
        month_split_path,
        month_split,
        weather_columns=weather_columns,
        production_columns=sorted_production_columns,
    )
    month_no_split_rows = _write_month_no_split_tsv(
        month_no_split_path,
        month_no_split,
        weather_columns=weather_columns,
    )

    print(f"Input rows: {total_rows}")
    print(f"Rows skipped (invalid instant/fca/category/production): {invalid_rows}")
    print(f"Rows parsed as valid before outlier filtering: {len(parsed_rows)}")
    print(
        "Outlier filtering: "
        f"method={outlier_method} "
        f"iqr_multiplier={_format_float(outlier_iqr_multiplier)} "
        f"rows_removed={outlier_rows} "
        f"categories_with_bounds={outlier_categories_with_bounds}"
    )
    print(f"Rows used for aggregation after outlier filtering: {len(rows_for_aggregation)}")
    print(f"Detected category columns: {len(sorted_production_columns)}")
    print(f"Output day split_category rows: {day_split_rows} (days with >24 rows: {day_split_days_over_24})")
    print(f"Output day no_split rows: {day_no_split_rows} (days with >24 rows: {day_no_split_days_over_24})")
    print(f"Output month split_category rows: {month_split_rows}")
    print(f"Output month no_split rows: {month_no_split_rows}")
    print(f"Output TSV (day/split_category): {day_split_path}")
    print(f"Output TSV (day/no_split): {day_no_split_path}")
    print(f"Output TSV (month/split_category): {month_split_path}")
    print(f"Output TSV (month/no_split): {month_no_split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
