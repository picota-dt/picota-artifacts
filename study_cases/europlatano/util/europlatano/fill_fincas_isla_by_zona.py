#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def _is_missing(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return True
    return cleaned.lower() in {"nan", "none", "null", "na", "n/a"}


def _find_column(fieldnames: list[str], expected: str) -> str | None:
    expected_lower = expected.strip().lower()
    for name in fieldnames:
        if name.strip().lower() == expected_lower:
            return name
    return None


def main() -> int:
    default_input = Path(__file__).resolve().parents[2] / "data" / "europlatano" / "fincas.tsv"
    default_output = Path(__file__).resolve().parents[2] / "data" / "europlatano" / "fincas_filled.tsv"

    parser = argparse.ArgumentParser(
        description="Fill missing ISLA values from other rows sharing the same Zona."
    )
    parser.add_argument("--input-tsv", default=str(default_input), help="Input fincas TSV path.")
    parser.add_argument("--output-tsv", default=str(default_output), help="Output TSV path.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write changes back to input file (overrides --output-tsv).",
    )
    args = parser.parse_args()

    input_tsv = Path(args.input_tsv)
    output_tsv = input_tsv if args.in_place else Path(args.output_tsv)

    if not input_tsv.exists():
        raise SystemExit(f"Input TSV not found: {input_tsv}")

    with input_tsv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit(f"Invalid TSV header: {input_tsv}")
        rows = list(reader)

    zona_col = _find_column(fieldnames, "Zona")
    isla_col = _find_column(fieldnames, "ISLA")
    if not zona_col or not isla_col:
        raise SystemExit(
            "Required columns not found. "
            f"Expected Zona and ISLA, available: {', '.join(fieldnames)}"
        )

    zona_to_islas: dict[str, dict[str, str]] = {}
    for row in rows:
        zona_raw = str(row.get(zona_col, "") or "").strip()
        isla_raw = str(row.get(isla_col, "") or "").strip()
        if _is_missing(zona_raw) or _is_missing(isla_raw):
            continue

        zona_key = zona_raw.upper()
        isla_key = isla_raw.upper()
        mapping = zona_to_islas.setdefault(zona_key, {})
        if isla_key not in mapping:
            mapping[isla_key] = isla_raw

    zona_unambiguous: dict[str, str] = {}
    ambiguous_zones = 0
    for zona_key, islands in zona_to_islas.items():
        if len(islands) == 1:
            zona_unambiguous[zona_key] = next(iter(islands.values()))
        else:
            ambiguous_zones += 1

    completed_rows = 0
    unresolved_missing_isla = 0
    missing_zona = 0
    ambiguous_zone_rows = 0

    for row in rows:
        zona_raw = str(row.get(zona_col, "") or "").strip()
        isla_raw = str(row.get(isla_col, "") or "").strip()

        if not _is_missing(isla_raw):
            continue

        if _is_missing(zona_raw):
            missing_zona += 1
            unresolved_missing_isla += 1
            continue

        zona_key = zona_raw.upper()
        inferred_isla = zona_unambiguous.get(zona_key)
        if inferred_isla:
            row[isla_col] = inferred_isla
            completed_rows += 1
            continue

        if zona_key in zona_to_islas:
            ambiguous_zone_rows += 1
        unresolved_missing_isla += 1

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input rows: {len(rows)}")
    print(f"Rows completed using Zona->ISLA inference: {completed_rows}")
    print(f"Rows still missing ISLA: {unresolved_missing_isla}")
    print(f"Rows with missing Zona (and missing ISLA): {missing_zona}")
    print(f"Rows with ambiguous Zona (multiple possible ISLA): {ambiguous_zone_rows}")
    print(f"Ambiguous Zona groups: {ambiguous_zones}")
    print(f"Output TSV: {output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
