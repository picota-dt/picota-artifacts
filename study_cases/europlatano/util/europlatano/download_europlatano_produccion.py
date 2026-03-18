#!/usr/bin/env python3
import argparse
import calendar
import gzip
import json
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE_URL = "https://api.europlatano.net/ulpgc/kilos"
BEARER_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3NjQ4Mzk2NjAsImRhdGEiOnsidXNlcm5hbWUiOiJwcnVlYmEifX0.-5D-qd-AHdifktb2cmDjmCFqu3P537RjqhU7oh1Xy7A"


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid date: {value!r} (expected dd/mm/YYYY or YYYY-mm-dd)")


def _format_api_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def _add_months(value: date, months: int) -> date:
    if months < 0:
        raise ValueError("months must be >= 0")
    year = value.year + (value.month - 1 + months) // 12
    month = (value.month - 1 + months) % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date


def _iter_ranges(start: date, until: date, months_step: int) -> list[DateRange]:
    if start > until:
        raise ValueError("start must be <= until")
    if months_step <= 0:
        raise ValueError("months_step must be >= 1")

    ranges: list[DateRange] = []
    cursor = start
    while cursor <= until:
        end = _add_months(cursor, months_step) - timedelta(days=1)
        if end > until:
            end = until
        ranges.append(DateRange(cursor, end))
        cursor = end + timedelta(days=1)
    return ranges


def _build_url(range_: DateRange) -> str:
    params = {
        "fechaMin": _format_api_date(range_.start),
        "fechaMax": _format_api_date(range_.end),
    }
    return f"{API_BASE_URL}?{urlencode(params, safe='/')}"


def _read_response_bytes(
        req: Request, timeout_s: float, ssl_context: ssl.SSLContext | None
) -> tuple[bytes, dict[str, str]]:
    with urlopen(req, timeout=timeout_s, context=ssl_context) as response:
        body = response.read()
        headers = {k.lower(): v for k, v in response.headers.items()}

    if headers.get("content-encoding", "").lower() == "gzip":
        body = gzip.decompress(body)
    return body, headers


def _fetch_json(
        url: str,
        token: str | None,
        ssl_context: ssl.SSLContext | None,
        timeout_s: float,
        max_attempts: int,
        backoff_s: float,
) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "picota-runtime-trainer europlatano kilos downloader",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = Request(url, headers=headers, method="GET")
            body, _headers = _read_response_bytes(req, timeout_s=timeout_s, ssl_context=ssl_context)

            if not body.strip():
                if attempt >= max_attempts:
                    print(
                        f"Warning: empty response body for {url}. Treating as empty data.",
                        file=sys.stderr,
                    )
                    return {"data": []}
                last_error = ValueError("Empty response body")
                raise last_error

            return json.loads(body.decode("utf-8"))
        except HTTPError as error:
            last_error = error
            status_code = getattr(error, "code", None)
            retryable = status_code in (429, 500, 502, 503, 504)
            if not retryable or attempt >= max_attempts:
                raise
        except (TimeoutError, socket.timeout) as error:
            last_error = error
            if attempt >= max_attempts:
                print(
                    f"Warning: timeout for {url}. Treating as empty data.",
                    file=sys.stderr,
                )
                return {"data": []}
        except URLError as error:
            last_error = error
            is_timeout = isinstance(getattr(error, "reason", None), (TimeoutError, socket.timeout))
            if attempt >= max_attempts:
                if is_timeout:
                    print(
                        f"Warning: timeout for {url}. Treating as empty data.",
                        file=sys.stderr,
                    )
                    return {"data": []}
                raise
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= max_attempts:
                raise

        sleep_seconds = backoff_s * (2 ** (attempt - 1))
        print(
            f"Retrying in {sleep_seconds:.1f}s (attempt {attempt}/{max_attempts}) after error: {last_error}",
            file=sys.stderr,
        )
        time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch after {max_attempts} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Europlatano kilos in monthly chunks (from Jan 2023  to today)."
    )
    parser.add_argument("--start", default="01/01/2023", help="Start date (dd/mm/YYYY or YYYY-mm-dd).")
    parser.add_argument(
        "--until",
        default=date.today().strftime("%d/%m/%Y"),
        help="End date inclusive (dd/mm/YYYY or YYYY-mm-dd). Default: today.",
    )
    parser.add_argument("--months-step", type=int, default=1, help="Months per request (must be 1). Default: 1.")
    default_out_jsonl = Path(__file__).resolve().parents[1] / "data" / "europlatano" / "produccion.jsonl"
    parser.add_argument(
        "--out-jsonl",
        default=str(default_out_jsonl),
        help="Output JSONL file (one record per line). Default: ./runtime.test/data/europlatano/produccion.jsonl",
    )
    output_mode_group = parser.add_mutually_exclusive_group()
    output_mode_group.add_argument(
        "--append",
        dest="append",
        action="store_true",
        help="Append to --out-jsonl (default).",
    )
    output_mode_group.add_argument(
        "--overwrite",
        dest="append",
        action="store_false",
        help="Overwrite --out-jsonl before writing.",
    )
    parser.set_defaults(append=True)
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds. Default: 60.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max attempts per request. Default: 5.")
    parser.add_argument("--backoff", type=float, default=1.0, help="Backoff base seconds. Default: 1.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests (seconds). Default: 0.2.")
    ssl_group = parser.add_mutually_exclusive_group()
    ssl_group.add_argument(
        "--insecure",
        dest="verify_ssl",
        action="store_false",
        help="Disable SSL certificate verification (default).",
    )
    ssl_group.add_argument(
        "--secure",
        dest="verify_ssl",
        action="store_true",
        help="Enable SSL certificate verification.",
    )
    parser.set_defaults(verify_ssl=False)
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help="Path to a PEM CA bundle to trust (used with --secure).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only; do not download.")
    args = parser.parse_args()

    if args.months_step != 1:
        parser.error("--months-step must be 1 (monthly requests).")

    start = _parse_date(args.start)
    until = _parse_date(args.until)
    ranges = _iter_ranges(start, until, months_step=args.months_step)

    for range_ in ranges:
        if args.dry_run:
            print(_build_url(range_))

    if args.dry_run:
        return 0

    token = BEARER_TOKEN.strip()
    if token == "PASTE_BEARER_TOKEN_HERE":
        token = ""
    if not token:
        print("No bearer token set. Edit BEARER_TOKEN in the script.", file=sys.stderr)

    verify_ssl = bool(args.verify_ssl or args.ca_bundle)
    if verify_ssl:
        ssl_context = ssl.create_default_context()
        if args.ca_bundle:
            ssl_context.load_verify_locations(cafile=args.ca_bundle)
    else:
        ssl_context = ssl._create_unverified_context()

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    total_records = 0
    with out_jsonl.open(mode, encoding="utf-8") as output_file:
        for range_ in ranges:
            url = _build_url(range_)
            print(f"Fetching {range_.start}..{range_.end}")
            payload = _fetch_json(
                url=url,
                token=token or None,
                ssl_context=ssl_context,
                timeout_s=args.timeout,
                max_attempts=args.max_attempts,
                backoff_s=args.backoff,
            )

            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object response for {url}, got {type(payload).__name__}")

            data = payload.get("data")
            if not isinstance(data, list):
                keys = ", ".join(sorted(payload.keys())[:12])
                raise RuntimeError(
                    f"Expected response['data'] list for {url}. Got {type(data).__name__}. Keys: {keys}"
                )

            for row in data:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()

            total_records += len(data)
            print(f"Wrote {len(data)} records (total: {total_records})")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote {total_records} records to {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
