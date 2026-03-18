#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _normalize_cat_label(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    aliases = {
        "SEG": "SEGUNDA",
    }
    return aliases.get(cleaned.upper(), cleaned)


def _normalize_fecha_to_iso_instant(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except ValueError:
            pass

    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return cleaned

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize_column_value(column: str, value: Any) -> str:
    normalized = _normalize_value(value)
    if column == "Fecha":
        return _normalize_fecha_to_iso_instant(normalized)
    if column == "Cat":
        return _normalize_cat_label(normalized)
    return normalized


def _reorder_columns(keep_columns: list[str]) -> list[str]:
    preferred = ["Fecha"]
    front = [column for column in preferred if column in keep_columns]
    rest = [column for column in keep_columns if column not in preferred]
    return front + rest


def _first_pass(
        input_jsonl: Path,
        unique_jsonl: Path,
) -> tuple[int, int, int, int, list[str], list[str]]:
    seen_hashes: set[bytes] = set()
    columns_order: list[str] = []
    known_columns: set[str] = set()
    constant_values: dict[str, str] = {}
    variable_columns: set[str] = set()

    total_rows = 0
    empty_rows = 0
    duplicate_rows = 0
    unique_rows = 0

    with input_jsonl.open("r", encoding="utf-8") as source, unique_jsonl.open("w", encoding="utf-8") as unique_out:
        for line_number, raw_line in enumerate(source, start=1):
            total_rows += 1
            line = raw_line.strip()
            if not line:
                empty_rows += 1
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at line {line_number}: {error}") from error

            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at line {line_number}, got {type(row).__name__}")

            canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            row_hash = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).digest()
            if row_hash in seen_hashes:
                duplicate_rows += 1
                continue

            seen_hashes.add(row_hash)
            unique_rows += 1
            unique_out.write(canonical + "\n")

            row_values = {key: _normalize_column_value(key, value) for key, value in row.items()}

            for column in columns_order:
                if column in variable_columns:
                    continue
                value = row_values.get(column, "")
                if constant_values[column] != value:
                    variable_columns.add(column)
                    constant_values.pop(column, None)

            for column, value in row_values.items():
                if column in known_columns:
                    continue

                known_columns.add(column)
                columns_order.append(column)

                # Before this row the column was missing, i.e. value "".
                if unique_rows == 1 or value == "":
                    constant_values[column] = value
                else:
                    variable_columns.add(column)

    keep_columns = [column for column in columns_order if column in variable_columns]
    removed_columns = [column for column in columns_order if column not in variable_columns]
    return total_rows, unique_rows, duplicate_rows, empty_rows, keep_columns, removed_columns


def _sort_tsv_body_by_column(input_tsv_body: Path, output_tsv_body: Path, column_index: int) -> None:
    sort_bin = shutil.which("sort")
    if sort_bin is None:
        raise RuntimeError("System command 'sort' not found; cannot sort rows by date.")

    env = os.environ.copy()
    env["LC_ALL"] = "C"
    sort_key = f"-k{column_index + 1},{column_index + 1}"
    subprocess.run(
        [sort_bin, "-t", "\t", sort_key, str(input_tsv_body), "-o", str(output_tsv_body)],
        check=True,
        env=env,
    )


def _fecha_to_day(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if len(cleaned) >= 10 and cleaned[4] == "-" and cleaned[7] == "-":
        return cleaned[:10]
    normalized = _normalize_fecha_to_iso_instant(cleaned)
    if len(normalized) >= 10 and normalized[4] == "-" and normalized[7] == "-":
        return normalized[:10]
    return ""


def _is_missing_text(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return True
    return cleaned.lower() in {"nan", "none", "null", "na", "n/a"}


def _safe_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _safe_float_from_any(value: Any) -> float | None:
    if value is None:
        return None
    return _safe_float(str(value).replace(",", "."))


def _normalize_join_key(value: Any) -> str:
    text = _normalize_value(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def _build_incidencias_join_key(vale: Any, almacen: Any, empresa: Any) -> tuple[str, str, str] | None:
    vale_key = _normalize_join_key(vale)
    if not vale_key:
        return None
    almacen_key = _normalize_join_key(almacen).strip().upper()
    empresa_key = _normalize_join_key(empresa).strip().upper()
    return vale_key, almacen_key, empresa_key


def _build_incidencias_join_key_vale_almacen(vale: Any, almacen: Any) -> tuple[str, str] | None:
    vale_key = _normalize_join_key(vale)
    if not vale_key:
        return None
    almacen_key = _normalize_join_key(almacen).strip().upper()
    return vale_key, almacen_key


def _build_incidencias_join_key_vale(vale: Any) -> str | None:
    vale_key = _normalize_join_key(vale)
    if not vale_key:
        return None
    return vale_key


def _format_decimal(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


EXCLUDED_INCIDENCIA_CODES = {
    "REV",
    "MNS",
    "MN3",
    "MNC",
    "PC",
    "PFP",
    "SFP",
    "RAJ",
    "AMA",
    "FRI",
    "MTR",
    "GOL",
    "UDT",
    "OTR",
}

ALLOWED_CATEGORY_CODES = {
    "B",
    "BIOK",
    "C",
    "E",
    "G",
    "J",
    "M",
    "N",
    "P",
    "S",
    "U",
    "V",
    "W",
    "X",
    "Y",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _resolve_meteo_station_file(meteo_dir: Path, station_code: str) -> Path | None:
    code = station_code.strip()
    if not code:
        return None
    candidates = [f"{code}.tsv", f"{code.upper()}.tsv", f"{code.lower()}.tsv"]
    for candidate in candidates:
        p = meteo_dir / candidate
        if p.exists():
            return p
    return None


def _load_meteo_station_rows(
        meteo_dir: Path,
        station_code: str,
        meteo_columns: list[str],
) -> dict[str, dict[str, str]] | None:
    file_path = _resolve_meteo_station_file(meteo_dir, station_code)
    if file_path is None:
        return None

    by_day: dict[str, dict[str, str]] = {}
    with file_path.open("r", encoding="utf-8", newline="") as meteo_in:
        reader = csv.DictReader(meteo_in, delimiter="\t")
        if not reader.fieldnames or "date" not in reader.fieldnames:
            return {}
        for meteo_row in reader:
            day = str(meteo_row.get("date", "")).strip()[:10]
            if not day:
                continue
            values: dict[str, str] = {}
            for column in meteo_columns:
                raw = meteo_row.get(column, "")
                text = "" if raw is None else str(raw).strip()
                if text.lower() == "nan":
                    text = ""
                values[column] = text
            by_day[day] = values
    return by_day


def _load_finca_metadata_lookup(fincas_tsv: Path) -> dict[str, dict[str, str]]:
    if not fincas_tsv.exists():
        raise RuntimeError(f"Fincas metadata TSV not found: {fincas_tsv}")

    with fincas_tsv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"Invalid fincas TSV header: {fincas_tsv}")

        field_by_lower = {name.strip().lower(): name for name in reader.fieldnames}
        fca_field = field_by_lower.get("fca")
        isla_field = field_by_lower.get("isla")
        altura_field = field_by_lower.get("altura")

        if not fca_field or not isla_field or not altura_field:
            raise RuntimeError(
                "Fincas TSV must include columns Fca, ISLA and Altura "
                f"(available: {', '.join(reader.fieldnames)})"
            )

        lookup: dict[str, dict[str, str]] = {}
        for row in reader:
            key = str(row.get(fca_field, "")).strip().upper()[:3]
            if not key:
                continue

            current_isla = str(row.get(isla_field, "") or "").strip()
            current_altura = str(row.get(altura_field, "") or "").strip()
            existing = lookup.get(key)
            if existing is None:
                lookup[key] = {"ISLA": current_isla, "Altura": current_altura}
            else:
                if not existing["ISLA"] and current_isla:
                    existing["ISLA"] = current_isla
                if not existing["Altura"] and current_altura:
                    existing["Altura"] = current_altura

    return lookup


def _load_zero_position_fca_codes(fincas_tsv: Path) -> set[str]:
    if not fincas_tsv.exists():
        raise RuntimeError(f"Fincas metadata TSV not found: {fincas_tsv}")

    with fincas_tsv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"Invalid fincas TSV header: {fincas_tsv}")

        field_by_lower = {name.strip().lower(): name for name in reader.fieldnames}
        fca_field = field_by_lower.get("fca")
        x_field = field_by_lower.get("x")
        y_field = field_by_lower.get("y")
        if not fca_field or not x_field or not y_field:
            raise RuntimeError(
                "Fincas TSV must include columns Fca, X and Y "
                f"(available: {', '.join(reader.fieldnames)})"
            )

        zero_codes: set[str] = set()
        for row in reader:
            code = str(row.get(fca_field, "")).strip().upper()[:3]
            if not code:
                continue

            x_value = _safe_float(str(row.get(x_field, "") or "").replace(",", "."))
            y_value = _safe_float(str(row.get(y_field, "") or "").replace(",", "."))
            if x_value is None or y_value is None:
                continue
            if x_value == 0.0 and y_value == 0.0:
                zero_codes.add(code)

    return zero_codes


def _enrich_output_tsv_with_finca_metadata(
        output_tsv: Path,
        fincas_tsv: Path,
        tmp_dir: str | None = None,
        drop_fca: bool = True,
) -> tuple[int, int]:
    if not output_tsv.exists():
        raise RuntimeError(f"Output TSV not found for finca metadata enrichment: {output_tsv}")

    finca_lookup = _load_finca_metadata_lookup(fincas_tsv=fincas_tsv)

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_fincas_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        enriched_path = Path(tmp_out.name)

    metadata_hits = 0
    metadata_missing = 0

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, enriched_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            header = next(reader)
            if "fca" not in header:
                raise RuntimeError("Column 'fca' not found in output TSV; cannot join finca metadata.")

            final_header = list(header)
            if "Altura" not in final_header:
                final_header.append("Altura")
            if "ISLA" not in final_header:
                final_header.append("ISLA")
            if drop_fca:
                final_header = [column for column in final_header if column != "fca"]
            writer.writerow(final_header)

            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                row_map = {header[i]: row[i] for i in range(len(header))}

                key = str(row_map.get("fca", "")).strip().upper()[:3]
                metadata = finca_lookup.get(key)
                if metadata is not None:
                    row_map["Altura"] = metadata.get("Altura", "")
                    row_map["ISLA"] = metadata.get("ISLA", "")
                    metadata_hits += 1
                else:
                    row_map["Altura"] = ""
                    row_map["ISLA"] = ""
                    metadata_missing += 1

                if drop_fca:
                    row_map.pop("fca", None)

                writer.writerow([row_map.get(column, "") for column in final_header])
    finally:
        output_tsv.unlink(missing_ok=True)
        enriched_path.replace(output_tsv)

    return metadata_hits, metadata_missing


def _load_incidencias_lookup(
        incidencias_jsonl: Path,
        excluded_codes: set[str],
) -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
    dict[str, dict[str, str]],
    dict[tuple[str, str, str], str],
    dict[tuple[str, str], str],
    dict[str, str],
    list[str],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    if not incidencias_jsonl.exists():
        raise RuntimeError(f"Incidencias JSONL not found: {incidencias_jsonl}")

    raw_lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    raw_lookup_vale_almacen: dict[tuple[str, str], dict[str, float]] = {}
    raw_lookup_vale: dict[str, dict[str, float]] = {}
    totals_by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    totals_by_key_vale_almacen: dict[tuple[str, str], dict[str, float]] = {}
    totals_by_key_vale: dict[str, dict[str, float]] = {}
    total_rows = 0
    indexed_rows = 0
    skipped_invalid_ratio = 0
    skipped_missing_cod = 0
    skipped_excluded_cod = 0
    clipped_ratio_rows = 0

    with incidencias_jsonl.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue

            total_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON in incidencias at line {line_number}: {error}") from error

            if not isinstance(row, dict):
                raise RuntimeError(
                    "Expected JSON object in incidencias at line "
                    f"{line_number}, got {type(row).__name__}"
                )

            key = _build_incidencias_join_key(
                vale=row.get("Vale"),
                almacen=row.get("Almacen"),
                empresa=row.get("Empresa"),
            )
            if not key:
                continue
            key_vale_almacen = _build_incidencias_join_key_vale_almacen(
                vale=row.get("Vale"),
                almacen=row.get("Almacen"),
            )
            key_vale = _build_incidencias_join_key_vale(row.get("Vale"))

            cod = _normalize_value(row.get("Cod")).strip().upper()
            if not cod:
                skipped_missing_cod += 1
                continue
            if cod in excluded_codes:
                skipped_excluded_cod += 1
                continue

            danadas = _safe_float_from_any(row.get("Dañadas"))
            pinas = _safe_float_from_any(row.get("Piñas"))
            if danadas is None or pinas is None or pinas <= 0:
                skipped_invalid_ratio += 1
                continue

            ratio = danadas / pinas
            if ratio < 0:
                ratio = 0.0
                clipped_ratio_rows += 1
            elif ratio > 1:
                ratio = 1.0
                clipped_ratio_rows += 1

            by_code = raw_lookup.setdefault(key, {})
            previous = by_code.get(cod)
            if previous is None or ratio > previous:
                by_code[cod] = ratio
            if key_vale_almacen is not None:
                by_code_vale_almacen = raw_lookup_vale_almacen.setdefault(key_vale_almacen, {})
                previous_vale_almacen = by_code_vale_almacen.get(cod)
                if previous_vale_almacen is None or ratio > previous_vale_almacen:
                    by_code_vale_almacen[cod] = ratio
            if key_vale is not None:
                by_code_vale = raw_lookup_vale.setdefault(key_vale, {})
                previous_vale = by_code_vale.get(cod)
                if previous_vale is None or ratio > previous_vale:
                    by_code_vale[cod] = ratio

            totals = totals_by_key.setdefault(key, {"danadas": 0.0, "pinas": 0.0})
            totals["danadas"] += danadas
            totals["pinas"] += pinas
            if key_vale_almacen is not None:
                totals_vale_almacen = totals_by_key_vale_almacen.setdefault(
                    key_vale_almacen, {"danadas": 0.0, "pinas": 0.0}
                )
                totals_vale_almacen["danadas"] += danadas
                totals_vale_almacen["pinas"] += pinas
            if key_vale is not None:
                totals_vale = totals_by_key_vale.setdefault(key_vale, {"danadas": 0.0, "pinas": 0.0})
                totals_vale["danadas"] += danadas
                totals_vale["pinas"] += pinas
            indexed_rows += 1

    code_columns = sorted({code for by_code in raw_lookup.values() for code in by_code})
    lookup = {
        key: {code: _format_decimal(value) for code, value in by_code.items()}
        for key, by_code in raw_lookup.items()
    }
    lookup_vale_almacen = {
        key: {code: _format_decimal(value) for code, value in by_code.items()}
        for key, by_code in raw_lookup_vale_almacen.items()
    }
    lookup_vale = {
        key: {code: _format_decimal(value) for code, value in by_code.items()}
        for key, by_code in raw_lookup_vale.items()
    }
    total_ratio_lookup: dict[tuple[str, str, str], str] = {}
    total_ratio_lookup_vale_almacen: dict[tuple[str, str], str] = {}
    total_ratio_lookup_vale: dict[str, str] = {}
    for key, totals in totals_by_key.items():
        pinas_total = totals.get("pinas", 0.0)
        if pinas_total <= 0:
            total_ratio_lookup[key] = "0"
            continue
        ratio = totals.get("danadas", 0.0) / pinas_total
        if ratio < 0:
            ratio = 0.0
        elif ratio > 1:
            ratio = 1.0
        total_ratio_lookup[key] = _format_decimal(ratio)
    for key, totals in totals_by_key_vale_almacen.items():
        pinas_total = totals.get("pinas", 0.0)
        if pinas_total <= 0:
            total_ratio_lookup_vale_almacen[key] = "0"
            continue
        ratio = totals.get("danadas", 0.0) / pinas_total
        if ratio < 0:
            ratio = 0.0
        elif ratio > 1:
            ratio = 1.0
        total_ratio_lookup_vale_almacen[key] = _format_decimal(ratio)
    for key, totals in totals_by_key_vale.items():
        pinas_total = totals.get("pinas", 0.0)
        if pinas_total <= 0:
            total_ratio_lookup_vale[key] = "0"
            continue
        ratio = totals.get("danadas", 0.0) / pinas_total
        if ratio < 0:
            ratio = 0.0
        elif ratio > 1:
            ratio = 1.0
        total_ratio_lookup_vale[key] = _format_decimal(ratio)
    return (
        lookup,
        lookup_vale_almacen,
        lookup_vale,
        total_ratio_lookup,
        total_ratio_lookup_vale_almacen,
        total_ratio_lookup_vale,
        code_columns,
        total_rows,
        indexed_rows,
        skipped_invalid_ratio,
        skipped_missing_cod,
        skipped_excluded_cod,
        clipped_ratio_rows,
    )


def _enrich_output_tsv_with_incidencias(
        output_tsv: Path,
        incidencias_jsonl: Path,
        tmp_dir: str | None = None,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]:
    if not output_tsv.exists():
        raise RuntimeError(f"Output TSV not found for incidencias join: {output_tsv}")

    (
        incidencias_lookup,
        incidencias_lookup_vale_almacen,
        incidencias_lookup_vale,
        incidencias_total_ratio_lookup,
        incidencias_total_ratio_lookup_vale_almacen,
        incidencias_total_ratio_lookup_vale,
        incidencias_codes,
        incidencias_total_rows,
        incidencias_indexed_rows,
        incidencias_skipped_invalid_ratio,
        incidencias_skipped_missing_cod,
        incidencias_skipped_excluded_cod,
        incidencias_clipped_ratio_rows,
    ) = _load_incidencias_lookup(
        incidencias_jsonl=incidencias_jsonl,
        excluded_codes=EXCLUDED_INCIDENCIA_CODES,
    )

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_incidencias_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        joined_path = Path(tmp_out.name)

    production_rows_with_match = 0
    production_rows_with_match_exact = 0
    production_rows_with_match_fallback_vale_almacen = 0
    production_rows_with_match_fallback_vale = 0
    production_rows_without_match = 0
    joined_rows_written = 0

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, joined_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            try:
                header = next(reader)
            except StopIteration as error:
                raise RuntimeError("Output TSV is empty; cannot join incidencias.") from error

            if "Albaran" not in header:
                raise RuntimeError("Column 'Albaran' not found in output TSV; cannot join incidencias.")
            if "Almacen" not in header:
                raise RuntimeError("Column 'Almacen' not found in output TSV; cannot join incidencias.")
            if "Empresa" not in header:
                raise RuntimeError("Column 'Empresa' not found in output TSV; cannot join incidencias.")

            appended_codes = [code for code in incidencias_codes if code not in header]
            final_header = list(header) + appended_codes
            if "incidence_total_ratio" not in final_header:
                final_header.append("incidence_total_ratio")
            writer.writerow(final_header)
            albaran_index = header.index("Albaran")
            almacen_index = header.index("Almacen")
            empresa_index = header.index("Empresa")

            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[: len(header)]

                join_key = _build_incidencias_join_key(
                    vale=row[albaran_index],
                    almacen=row[almacen_index],
                    empresa=row[empresa_index],
                )
                join_key_vale_almacen = _build_incidencias_join_key_vale_almacen(
                    vale=row[albaran_index],
                    almacen=row[almacen_index],
                )
                join_key_vale = _build_incidencias_join_key_vale(row[albaran_index])

                incidencias_by_code = incidencias_lookup.get(join_key)
                incidence_total_ratio = incidencias_total_ratio_lookup.get(join_key, "0") if join_key else "0"
                if incidencias_by_code is not None:
                    production_rows_with_match_exact += 1
                if incidencias_by_code is None and join_key_vale_almacen is not None:
                    incidencias_by_code = incidencias_lookup_vale_almacen.get(join_key_vale_almacen)
                    incidence_total_ratio = incidencias_total_ratio_lookup_vale_almacen.get(join_key_vale_almacen, "0")
                    if incidencias_by_code is not None:
                        production_rows_with_match_fallback_vale_almacen += 1
                if incidencias_by_code is None and join_key_vale is not None:
                    incidencias_by_code = incidencias_lookup_vale.get(join_key_vale)
                    incidence_total_ratio = incidencias_total_ratio_lookup_vale.get(join_key_vale, "0")
                    if incidencias_by_code is not None:
                        production_rows_with_match_fallback_vale += 1
                if incidencias_by_code is None:
                    production_rows_without_match += 1
                    writer.writerow(row + ["0"] * len(appended_codes) + ["0"])
                else:
                    production_rows_with_match += 1
                    writer.writerow(
                        row
                        + [incidencias_by_code.get(code, "0") for code in appended_codes]
                        + [incidence_total_ratio]
                    )
                joined_rows_written += 1
    finally:
        output_tsv.unlink(missing_ok=True)
        joined_path.replace(output_tsv)

    return (
        incidencias_total_rows,
        incidencias_indexed_rows,
        incidencias_skipped_invalid_ratio,
        incidencias_skipped_missing_cod,
        incidencias_skipped_excluded_cod,
        incidencias_clipped_ratio_rows,
        len(appended_codes),
        production_rows_with_match,
        production_rows_with_match_exact,
        production_rows_with_match_fallback_vale_almacen,
        production_rows_with_match_fallback_vale,
        production_rows_without_match,
        joined_rows_written,
    )


def _aggregate_output_tsv_by_day_fca_category(
        output_tsv: Path,
        tmp_dir: str | None = None,
) -> tuple[int, int]:
    if not output_tsv.exists():
        raise RuntimeError(f"Output TSV not found for day/FCA/category aggregation: {output_tsv}")

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_day_fca_cat_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        aggregated_path = Path(tmp_out.name)

    input_rows = 0
    output_rows = 0

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, aggregated_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            try:
                header = next(reader)
            except StopIteration as error:
                raise RuntimeError("Output TSV is empty; cannot aggregate by day/FCA/category.") from error

            fecha_column = "Fecha" if "Fecha" in header else ("fecha" if "fecha" in header else None)
            fca_column = "fca" if "fca" in header else ("Fca" if "Fca" in header else None)
            categoria_column = (
                "categoria" if "categoria" in header else ("Category" if "Category" in header else None)
            )
            if fecha_column is None or fca_column is None or categoria_column is None:
                raise RuntimeError(
                    "Columns required for aggregation by day/FCA/category are missing. "
                    f"Required: Fecha/fecha, fca/Fca, categoria/Category. Available: {', '.join(header)}"
                )

            fecha_index = header.index(fecha_column)
            fca_index = header.index(fca_column)
            categoria_index = header.index(categoria_column)
            kilos_index = header.index("Kilos") if "Kilos" in header else (header.index("Production") if "Production" in header else None)

            grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
            missing_key_counter = 0

            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[: len(header)]

                input_rows += 1

                day_key = _fecha_to_day(row[fecha_index]) or row[fecha_index].strip()
                fca_key = row[fca_index].strip()
                categoria_key = row[categoria_index].strip()
                if not day_key or not fca_key or not categoria_key:
                    missing_key_counter += 1
                    group_key = (f"__missing_{missing_key_counter}", fca_key, categoria_key)
                else:
                    group_key = (day_key, fca_key, categoria_key)

                kilos_value = None
                if kilos_index is not None:
                    kilos_value = _safe_float(row[kilos_index].replace(",", "."))

                state = grouped.get(group_key)
                if state is None:
                    state = {
                        "row": list(row),
                        "best_kilos": kilos_value,
                        "sum_kilos": kilos_value if kilos_value is not None else 0.0,
                        "has_sum_kilos": kilos_value is not None,
                    }
                    grouped[group_key] = state
                    continue

                if kilos_value is not None:
                    if state["has_sum_kilos"]:
                        state["sum_kilos"] += kilos_value
                    else:
                        state["sum_kilos"] = kilos_value
                        state["has_sum_kilos"] = True

                best_kilos = state["best_kilos"]
                if best_kilos is None and kilos_value is not None:
                    state["row"] = list(row)
                    state["best_kilos"] = kilos_value
                elif best_kilos is not None and kilos_value is not None and kilos_value > best_kilos:
                    state["row"] = list(row)
                    state["best_kilos"] = kilos_value

            writer.writerow(header)
            for state in grouped.values():
                final_row = list(state["row"])
                if kilos_index is not None and state["has_sum_kilos"]:
                    final_row[kilos_index] = _format_decimal(state["sum_kilos"])
                writer.writerow(final_row)
                output_rows += 1
    finally:
        output_tsv.unlink(missing_ok=True)
        aggregated_path.replace(output_tsv)

    return input_rows, output_rows


def _enrich_output_tsv_with_meteo(
        output_tsv: Path,
        meteo_dir: Path,
        meteo_columns: list[str],
        tmp_dir: str | None = None,
        drop_fca: bool = True,
        blocked_station_codes: set[str] | None = None,
) -> tuple[int, int, int, int]:
    if not output_tsv.exists():
        raise RuntimeError(f"Output TSV not found for meteo enrichment: {output_tsv}")
    if not meteo_dir.exists():
        raise RuntimeError(f"Meteo directory not found: {meteo_dir}")

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_meteo_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        enriched_path = Path(tmp_out.name)

    station_codes = sorted(path.stem.upper() for path in meteo_dir.glob("*.tsv"))
    station_coords: dict[str, tuple[float, float]] = {}
    default_values = {column: "" for column in meteo_columns}

    for station_code in station_codes:
        station_file = _resolve_meteo_station_file(meteo_dir=meteo_dir, station_code=station_code)
        if station_file is None:
            continue
        with station_file.open("r", encoding="utf-8", newline="") as station_in:
            reader = csv.DictReader(station_in, delimiter="\t")
            for row_index, station_row in enumerate(reader):
                lat_value = _safe_float(str(station_row.get("station_lat", "")))
                lon_value = _safe_float(str(station_row.get("station_lon", "")))
                if lat_value is not None and lon_value is not None:
                    station_coords[station_code] = (lat_value, lon_value)
                for column in meteo_columns:
                    if _is_missing_text(default_values[column]):
                        candidate = "" if station_row.get(column) is None else str(station_row.get(column)).strip()
                        if not _is_missing_text(candidate):
                            default_values[column] = candidate
                if station_code in station_coords and row_index >= 9:
                    break

    meteo_cache: dict[str, dict[str, dict[str, str]] | None] = {}
    nearest_cache: dict[str, list[str]] = {}
    offsets: list[int] = []
    for offset in range(1, 31):
        offsets.append(offset)
        offsets.append(-offset)

    def get_station_rows(station_code: str) -> dict[str, dict[str, str]] | None:
        normalized = station_code.strip().upper()
        if not normalized:
            return None
        if normalized not in meteo_cache:
            meteo_cache[normalized] = _load_meteo_station_rows(
                meteo_dir=meteo_dir,
                station_code=normalized,
                meteo_columns=meteo_columns,
            )
        return meteo_cache[normalized]

    def get_candidates_by_distance(station_code: str) -> list[str]:
        normalized = station_code.strip().upper()
        if not normalized:
            return list(station_codes)
        if normalized in nearest_cache:
            return nearest_cache[normalized]

        if normalized not in station_coords:
            ordered = [normalized] + [code for code in station_codes if code != normalized]
            nearest_cache[normalized] = ordered
            return ordered

        base_lat, base_lon = station_coords[normalized]
        distances: list[tuple[float, str]] = []
        for candidate_code in station_codes:
            if candidate_code == normalized:
                continue
            candidate_coords = station_coords.get(candidate_code)
            if candidate_coords is None:
                continue
            distance_km = _haversine_km(base_lat, base_lon, candidate_coords[0], candidate_coords[1])
            distances.append((distance_km, candidate_code))
        distances.sort(key=lambda item: item[0])

        ordered = [normalized] + [candidate_code for _, candidate_code in distances]
        nearest_cache[normalized] = ordered
        return ordered

    def fill_missing_from_row(
            meteo_values: dict[str, str],
            source_values: dict[str, str],
            missing_columns: list[str],
    ) -> bool:
        filled = False
        for column in list(missing_columns):
            candidate = source_values.get(column, "")
            if not _is_missing_text(candidate):
                meteo_values[column] = candidate
                missing_columns.remove(column)
                filled = True
        return filled

    def resolve_meteo_values(station_code: str, day: str) -> tuple[dict[str, str], str]:
        values = {column: "" for column in meteo_columns}
        normalized_code = station_code.strip().upper()
        direct_day_hit = False
        used_fallback = False

        candidates = get_candidates_by_distance(normalized_code)
        primary_rows = get_station_rows(normalized_code) if normalized_code else None
        if primary_rows and day in primary_rows:
            values.update(primary_rows[day])
            direct_day_hit = True

        missing_columns = [column for column in meteo_columns if _is_missing_text(values[column])]

        if day and missing_columns:
            for candidate_code in candidates[1:]:
                candidate_rows = get_station_rows(candidate_code)
                if not candidate_rows:
                    continue
                source_row = candidate_rows.get(day)
                if not source_row:
                    continue
                if fill_missing_from_row(values, source_row, missing_columns):
                    used_fallback = True
                if not missing_columns:
                    break

        if day and missing_columns:
            try:
                day_obj = datetime.strptime(day, "%Y-%m-%d").date()
            except ValueError:
                day_obj = None

            if day_obj is not None:
                temporal_candidates = candidates[:64]
                for offset in offsets:
                    candidate_day = (day_obj + timedelta(days=offset)).isoformat()
                    for candidate_code in temporal_candidates:
                        candidate_rows = get_station_rows(candidate_code)
                        if not candidate_rows:
                            continue
                        source_row = candidate_rows.get(candidate_day)
                        if not source_row:
                            continue
                        if fill_missing_from_row(values, source_row, missing_columns):
                            used_fallback = True
                        if not missing_columns:
                            break
                    if not missing_columns:
                        break

        if missing_columns:
            for column in list(missing_columns):
                fallback_value = default_values.get(column, "")
                if not _is_missing_text(fallback_value):
                    values[column] = fallback_value
                    missing_columns.remove(column)
                    used_fallback = True

        if missing_columns:
            return values, "incomplete"
        if used_fallback:
            return values, "fallback"
        if direct_day_hit:
            return values, "direct"
        return values, "fallback"

    rows_direct = 0
    rows_fallback = 0
    rows_incomplete = 0
    rows_blocked_zero_position = 0
    blocked_codes = blocked_station_codes or set()

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, enriched_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            header = next(reader)
            if "Fecha" not in header:
                raise RuntimeError("Column 'Fecha' not found in output TSV.")
            if "fca" not in header:
                raise RuntimeError("Column 'fca' not found in output TSV; cannot join meteo.")

            final_header = list(header)
            for column in meteo_columns:
                if column not in final_header:
                    final_header.append(column)
            if drop_fca:
                final_header = [column for column in final_header if column != "fca"]
            writer.writerow(final_header)

            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                row_map = {header[i]: row[i] for i in range(len(header))}

                station_code = row_map.get("fca", "").strip()[:3].upper()
                day = _fecha_to_day(row_map.get("Fecha", ""))
                if station_code in blocked_codes:
                    meteo_values = {column: "" for column in meteo_columns}
                    row_status = "blocked"
                else:
                    meteo_values, row_status = resolve_meteo_values(station_code=station_code, day=day)
                if row_status == "direct":
                    rows_direct += 1
                elif row_status == "fallback":
                    rows_fallback += 1
                elif row_status == "blocked":
                    rows_incomplete += 1
                    rows_blocked_zero_position += 1
                else:
                    rows_incomplete += 1

                for column, value in meteo_values.items():
                    row_map[column] = value
                if drop_fca:
                    row_map.pop("fca", None)

                writer.writerow([row_map.get(column, "") for column in final_header])
    finally:
        output_tsv.unlink(missing_ok=True)
        enriched_path.replace(output_tsv)

    return rows_direct, rows_fallback, rows_incomplete, rows_blocked_zero_position


def _drop_output_tsv_columns(
        output_tsv: Path,
        columns_to_drop: set[str],
        tmp_dir: str | None = None,
) -> tuple[list[str], list[str]]:
    if not output_tsv.exists():
        raise RuntimeError(f"Output TSV not found for column filtering: {output_tsv}")

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_dropcols_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        filtered_path = Path(tmp_out.name)

    kept_header: list[str] = []
    removed_header: list[str] = []

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, filtered_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            try:
                header = next(reader)
            except StopIteration as error:
                raise RuntimeError("Output TSV is empty; cannot filter columns.") from error

            kept_indexes: list[int] = []
            for index, column in enumerate(header):
                if column in columns_to_drop:
                    removed_header.append(column)
                else:
                    kept_indexes.append(index)
                    kept_header.append(column)

            writer.writerow(kept_header)
            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                writer.writerow([row[i] for i in kept_indexes])
    finally:
        output_tsv.unlink(missing_ok=True)
        filtered_path.replace(output_tsv)

    return kept_header, removed_header


def _rename_tsv_headers(output_tsv: Path, rename_map: dict[str, str], tmp_dir: str | None = None) -> list[str]:
    if not rename_map:
        with output_tsv.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source, delimiter="\t")
            return next(reader)

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_renamed_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        renamed_path = Path(tmp_out.name)

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, renamed_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            try:
                header = next(reader)
            except StopIteration as error:
                raise RuntimeError("Output TSV is empty; cannot rename columns.") from error

            renamed_header = [rename_map.get(column, column) for column in header]
            if len(set(renamed_header)) != len(renamed_header):
                raise RuntimeError(
                    "Column rename map creates duplicate header names: " + ", ".join(renamed_header)
                )

            writer.writerow(renamed_header)
            for row in reader:
                writer.writerow(row)
    finally:
        output_tsv.unlink(missing_ok=True)
        renamed_path.replace(output_tsv)

    return renamed_header


def _drop_incomplete_rows(output_tsv: Path, tmp_dir: str | None = None) -> tuple[int, int, int, dict[str, int]]:
    if not output_tsv.exists():
        raise RuntimeError(f"Output TSV not found for incomplete-row filtering: {output_tsv}")

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_complete_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_out:
        filtered_path = Path(tmp_out.name)

    total_rows = 0
    kept_rows = 0
    dropped_rows = 0
    missing_by_column: dict[str, int] = {}

    try:
        with output_tsv.open("r", encoding="utf-8", newline="") as source, filtered_path.open(
                "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")

            try:
                header = next(reader)
            except StopIteration as error:
                raise RuntimeError("Output TSV is empty; cannot filter incomplete rows.") from error

            writer.writerow(header)

            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[: len(header)]

                total_rows += 1
                incomplete = False
                for index, column in enumerate(header):
                    if _is_missing_text(str(row[index])):
                        missing_by_column[column] = missing_by_column.get(column, 0) + 1
                        incomplete = True

                if incomplete:
                    dropped_rows += 1
                    continue

                kept_rows += 1
                writer.writerow(row)
    finally:
        output_tsv.unlink(missing_ok=True)
        filtered_path.replace(output_tsv)

    return total_rows, kept_rows, dropped_rows, missing_by_column


def _second_pass(
        unique_jsonl: Path,
        output_tsv: Path,
        keep_columns: list[str],
        tmp_dir: str | None = None,
) -> tuple[int, int, int, int]:
    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_body_", "suffix": ".tsv", "delete": False}
    if tmp_dir:
        tmp_kwargs["dir"] = tmp_dir

    with tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_body, tempfile.NamedTemporaryFile(**tmp_kwargs) as tmp_sorted:
        body_path = Path(tmp_body.name)
        sorted_body_path = Path(tmp_sorted.name)

    seen_projected_hashes: set[bytes] = set()
    projected_duplicates = 0
    projected_unique_rows = 0
    removed_non_positive_kilos = 0
    removed_disallowed_category = 0
    kilos_index = keep_columns.index("Kilos") if "Kilos" in keep_columns else None
    categoria_index = (
        keep_columns.index("categoria")
        if "categoria" in keep_columns
        else (keep_columns.index("Category") if "Category" in keep_columns else None)
    )
    if categoria_index is None:
        raise RuntimeError(
            "Column 'categoria' not found in projected columns; cannot apply category filter. "
            f"Available: {keep_columns}"
        )

    try:
        with unique_jsonl.open("r", encoding="utf-8") as source, body_path.open(
                "w", encoding="utf-8", newline=""
        ) as body_out:
            writer = csv.writer(body_out, delimiter="\t", lineterminator="\n")
            for line_number, line in enumerate(source, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"Expected JSON object in temporary unique file at line {line_number}, got {type(row).__name__}"
                    )
                projected_row = [_normalize_column_value(column, row.get(column)) for column in keep_columns]

                categoria_value = projected_row[categoria_index].strip().upper()
                if categoria_value not in ALLOWED_CATEGORY_CODES:
                    removed_disallowed_category += 1
                    continue

                if kilos_index is not None:
                    kilos_raw = projected_row[kilos_index].strip()
                    try:
                        kilos_value = float(kilos_raw.replace(",", ".")) if kilos_raw else None
                    except ValueError:
                        kilos_value = None
                    if kilos_value is not None and kilos_value <= 0:
                        removed_non_positive_kilos += 1
                        continue

                projected_canonical = json.dumps(projected_row, ensure_ascii=False, separators=(",", ":"))
                projected_hash = hashlib.blake2b(projected_canonical.encode("utf-8"), digest_size=16).digest()

                if projected_hash in seen_projected_hashes:
                    projected_duplicates += 1
                    continue

                seen_projected_hashes.add(projected_hash)
                projected_unique_rows += 1
                writer.writerow(projected_row)

        if "Fecha" in keep_columns:
            _sort_tsv_body_by_column(
                input_tsv_body=body_path,
                output_tsv_body=sorted_body_path,
                column_index=keep_columns.index("Fecha"),
            )
            source_path = sorted_body_path
        else:
            source_path = body_path

        with output_tsv.open("w", encoding="utf-8", newline="") as tsv_out:
            writer = csv.writer(tsv_out, delimiter="\t", lineterminator="\n")
            writer.writerow(keep_columns)
            with source_path.open("r", encoding="utf-8") as body_in:
                for raw_line in body_in:
                    tsv_out.write(raw_line)
    finally:
        body_path.unlink(missing_ok=True)
        sorted_body_path.unlink(missing_ok=True)

    return projected_unique_rows, projected_duplicates, removed_non_positive_kilos, removed_disallowed_category


def main() -> int:
    default_data_dir = Path("/Users/oroncal/workspace/projects/picota/temp/data/europlatano")
    default_input = default_data_dir / "raw" / "produccion.jsonl"
    default_output = default_data_dir / "europlatano_incidencias.tsv"
    default_incidencias = default_data_dir / "raw" / "incidencias.jsonl"
    default_fincas_tsv = default_data_dir / "raw" / "fincas.tsv"

    parser = argparse.ArgumentParser(
        description=(
            "Transform produccion.jsonl into TSV for incidencias projection: "
            "no meteo-temporal enrichment, with incidencias join by Albaran/Vale."
        )
    )
    parser.add_argument("--input-jsonl", default=str(default_input), help="Input JSONL path.")
    parser.add_argument("--output-tsv", default=str(default_output), help="Output TSV path.")
    parser.add_argument("--incidencias-jsonl", default=str(default_incidencias), help="Incidencias JSONL path.")
    parser.add_argument("--tmp-dir", default=None, help="Temporary directory (default: system temp).")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary deduplicated JSONL.")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_tsv = Path(args.output_tsv)
    incidencias_jsonl = Path(args.incidencias_jsonl)
    fincas_tsv = default_fincas_tsv

    if not input_jsonl.exists():
        raise SystemExit(f"Input file not found: {input_jsonl}")
    if not incidencias_jsonl.exists():
        raise SystemExit(f"Incidencias file not found: {incidencias_jsonl}")

    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    tmp_kwargs: dict[str, Any] = {"prefix": "produccion_unique_", "suffix": ".jsonl"}
    if args.tmp_dir:
        tmp_kwargs["dir"] = args.tmp_dir

    with tempfile.NamedTemporaryFile(delete=False, **tmp_kwargs) as tmp:
        unique_jsonl = Path(tmp.name)

    try:
        total_rows, unique_rows, duplicate_rows, empty_rows, keep_columns, removed_columns = _first_pass(
            input_jsonl=input_jsonl,
            unique_jsonl=unique_jsonl,
        )
        force_drop_columns = {
            "Albaran",
            "fca",
            "Fca",
            "finca",
            "Finca",
            "Semana",
            "Cat",
            "producto",
            "RegLin",
            "Empresa",
            "Almacen",
            "Linea",
            "AlbLin",
            "Entidad",
            "Orden",
            "pescaj",
            "patron",
        }
        keep_columns = _reorder_columns(keep_columns)
        for required_column in ("Albaran", "Almacen", "Empresa"):
            if required_column not in keep_columns:
                keep_columns.append(required_column)
        (
            projected_unique_rows,
            projected_duplicates,
            removed_non_positive_kilos,
            removed_disallowed_category,
        ) = _second_pass(
            unique_jsonl=unique_jsonl,
            output_tsv=output_tsv,
            keep_columns=keep_columns,
            tmp_dir=args.tmp_dir,
        )
        (
            incidencias_total_rows,
            incidencias_indexed_rows,
            incidencias_skipped_invalid_ratio,
            incidencias_skipped_missing_cod,
            incidencias_skipped_excluded_cod,
            incidencias_clipped_ratio_rows,
            incidencias_added_onehot_columns,
            incidencias_join_hit_rows,
            incidencias_join_hit_rows_exact,
            incidencias_join_hit_rows_fallback_vale_almacen,
            incidencias_join_hit_rows_fallback_vale,
            incidencias_join_missing_rows,
            incidencias_join_rows_written,
        ) = _enrich_output_tsv_with_incidencias(
            output_tsv=output_tsv,
            incidencias_jsonl=incidencias_jsonl,
            tmp_dir=args.tmp_dir,
        )
        finca_rows_with_metadata, finca_rows_without_metadata = _enrich_output_tsv_with_finca_metadata(
            output_tsv=output_tsv,
            fincas_tsv=fincas_tsv,
            tmp_dir=args.tmp_dir,
            drop_fca=False,
        )
        grouped_rows_before_aggregation, grouped_rows_after_aggregation = _aggregate_output_tsv_by_day_fca_category(
            output_tsv=output_tsv,
            tmp_dir=args.tmp_dir,
        )

        keep_columns, forced_removed = _drop_output_tsv_columns(
            output_tsv=output_tsv,
            columns_to_drop=force_drop_columns,
            tmp_dir=args.tmp_dir,
        )
        for column in forced_removed:
            if column not in removed_columns:
                removed_columns.append(column)

        header_rename_map = {
            "Fecha": "instant",
            "fecha": "instant",
            "categoria": "Category",
            "tmin": "Territory.Temperature:Min",
            "tmax": "Territory.Temperature:Max",
            "tmed": "Territory.Temperature:Average",
            "velmedia": "Territory.WindSpeed:Average",
            "racha": "Territory.WindSpeed:Max",
            "hrMedia": "Territory.Humidity",
            "prec": "Territory.Precipitation",
            "dir": "Territory.WindDirection",
            "M2": "Area",
            "Altura": "Altitude",
            "Isla": "Island",
            "ISLA": "Island",
            "Kilos": "Production",
        }
        keep_columns = [header_rename_map.get(column, column) for column in keep_columns]
        _rename_tsv_headers(output_tsv=output_tsv, rename_map=header_rename_map, tmp_dir=args.tmp_dir)
        final_total_rows, final_complete_rows, final_dropped_incomplete_rows, missing_by_column = (
            _drop_incomplete_rows(output_tsv=output_tsv, tmp_dir=args.tmp_dir)
        )

        print(f"Input rows: {total_rows}")
        print(f"Empty lines skipped: {empty_rows}")
        print(f"Duplicate rows removed: {duplicate_rows}")
        print(f"Unique rows written: {unique_rows}")
        print(f"Rows removed with Kilos <= 0: {removed_non_positive_kilos}")
        print(f"Duplicate rows removed after projection: {projected_duplicates}")
        print(f"Rows written to TSV after projection: {projected_unique_rows}")
        print(f"Incidencias rows read: {incidencias_total_rows}")
        print(f"Incidencias rows indexed: {incidencias_indexed_rows}")
        print(f"Incidencias rows skipped (invalid dañadas/piñas): {incidencias_skipped_invalid_ratio}")
        print(f"Incidencias rows skipped (missing Cod): {incidencias_skipped_missing_cod}")
        print(f"Incidencias rows skipped (excluded Cod): {incidencias_skipped_excluded_cod}")
        print(f"Incidencias rows clipped to [0,1]: {incidencias_clipped_ratio_rows}")
        print(f"Incidencias one-hot columns added from Cod: {incidencias_added_onehot_columns}")
        print(f"Production rows with incidencias join hit: {incidencias_join_hit_rows}")
        print(f"Production rows with incidencias exact join hit: {incidencias_join_hit_rows_exact}")
        print(
            "Production rows with incidencias join hit by fallback Vale+Almacen: "
            f"{incidencias_join_hit_rows_fallback_vale_almacen}"
        )
        print(
            "Production rows with incidencias join hit by fallback Vale only: "
            f"{incidencias_join_hit_rows_fallback_vale}"
        )
        print(f"Production rows without incidencias join match: {incidencias_join_missing_rows}")
        print(f"Production rows written after incidencias join: {incidencias_join_rows_written}")
        print(f"Rows with finca metadata join hit: {finca_rows_with_metadata}")
        print(f"Rows without finca metadata (fca[:3] not found): {finca_rows_without_metadata}")
        print(f"Rows before aggregation by day/FCA/category: {grouped_rows_before_aggregation}")
        print(f"Rows after aggregation by day/FCA/category: {grouped_rows_after_aggregation}")
        print(f"Rows removed due disallowed categoria code: {removed_disallowed_category}")
        print(f"Final rows checked for completeness: {final_total_rows}")
        print(f"Final incomplete rows removed: {final_dropped_incomplete_rows}")
        print(f"Final complete rows kept: {final_complete_rows}")
        if missing_by_column:
            missing_summary = ", ".join(
                f"{column}={count}" for column, count in sorted(missing_by_column.items(), key=lambda item: item[0])
            )
            print("Missing-value hits by column (pre-final-drop): " + missing_summary)
        print(f"Columns kept in TSV: {len(keep_columns)}")
        print(f"Columns removed (constant + forced): {len(removed_columns)}")
        if removed_columns:
            print("Removed columns: " + ", ".join(removed_columns))
        print(f"Output TSV: {output_tsv}")
    finally:
        if args.keep_temp:
            print(f"Temporary deduplicated JSONL kept at: {unique_jsonl}")
        else:
            unique_jsonl.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
