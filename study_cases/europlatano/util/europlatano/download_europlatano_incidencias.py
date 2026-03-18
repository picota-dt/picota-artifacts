#!/usr/bin/env python3
import argparse
import calendar
import gzip
import json
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

API_BASE_URL = "https://api.europlatano.net/ulpgc/incidencias"

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

    def to_filename(self) -> str:
        return f"incidencias_{self.start:%Y%m%d}_{self.end:%Y%m%d}.json"


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
    # Keep slashes unescaped to match the example URL.
    return f"{API_BASE_URL}?{urlencode(params, safe='/')}"


def _read_response_bytes(
        req: Request, timeout_s: float, ssl_context: ssl.SSLContext | None
) -> tuple[bytes, dict[str, str]]:
    with urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
        raw = resp.read()
        headers = {k.lower(): v for k, v in resp.headers.items()}

    if headers.get("content-encoding", "").lower() == "gzip":
        raw = gzip.decompress(raw)
    return raw, headers


def _fetch_json(
        url: str,
        token: str | None,
        ssl_context: ssl.SSLContext | None,
        timeout_s: float,
        max_attempts: int,
        backoff_s: float,
) -> tuple[bytes, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "picota-runtime-trainer europlatano downloader",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = Request(url, headers=headers, method="GET")
            body, _headers = _read_response_bytes(req, timeout_s=timeout_s, ssl_context=ssl_context)

            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            return body, parsed
        except HTTPError as e:
            last_error = e
            status = getattr(e, "code", None)
            retryable = status in (429, 500, 502, 503, 504)
            if not retryable or attempt >= max_attempts:
                raise
        except URLError as e:
            last_error = e
            if attempt >= max_attempts:
                raise

        sleep_s = backoff_s * (2 ** (attempt - 1))
        print(
            f"Retrying in {sleep_s:.1f}s (attempt {attempt}/{max_attempts}) after error: {last_error}",
            file=sys.stderr,
        )
        time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch after {max_attempts} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Europlatano incidencias in 6-month chunks (from 2014 to today)."
    )
    parser.add_argument("--start", default="01/01/2014", help="Start date (dd/mm/YYYY or YYYY-mm-dd).")
    parser.add_argument(
        "--until",
        default=date.today().strftime("%d/%m/%Y"),
        help="End date inclusive (dd/mm/YYYY or YYYY-mm-dd). Default: today.",
    )
    parser.add_argument("--months-step", type=int, default=6, help="Months per request (max: 6). Default: 6.")
    default_out_jsonl = (
            Path(__file__).resolve().parents[1] / "data" / "europlatano" / "incidencias.jsonl"
    )
    parser.add_argument(
        "--out-jsonl",
        default=str(default_out_jsonl),
        help="Output JSONL file (one incidencia per line). Default: ./runtime.test/data/europlatano/incidencias.jsonl",
    )
    parser.add_argument("--append", action="store_true", help="Append to --out-jsonl instead of overwriting.")
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
        help="Path to a PEM CA bundle to trust (recommended for corporate proxy TLS).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only; do not download.")
    args = parser.parse_args()

    if args.months_step > 6:
        parser.error("--months-step cannot be greater than 6 (API limitation).")

    start = _parse_date(args.start)
    until = _parse_date(args.until)
    ranges = _iter_ranges(start, until, months_step=args.months_step)

    for range_ in ranges:
        url = _build_url(range_)
        if args.dry_run:
            print(url)
            continue

    if args.dry_run:
        return 0

    token = BEARER_TOKEN.strip()
    if token == "PASTE_BEARER_TOKEN_HERE":
        token = ""
    if not token:
        print(
            "No bearer token set. Edit BEARER_TOKEN in the script.",
            file=sys.stderr,
        )

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

    total_items = 0
    with out_jsonl.open(mode, encoding="utf-8") as out_fp:
        for range_ in ranges:
            url = _build_url(range_)

            print(f"Fetching {range_.start}..{range_.end}")
            _body, parsed = _fetch_json(
                url=url,
                token=token or None,
                ssl_context=ssl_context,
                timeout_s=args.timeout,
                max_attempts=args.max_attempts,
                backoff_s=args.backoff,
            )

            if not isinstance(parsed, dict):
                raise RuntimeError(f"Expected JSON object response for {url}, got {type(parsed).__name__}")
            data = parsed.get("data")
            if not isinstance(data, list):
                keys = ", ".join(sorted(parsed.keys())[:12])
                raise RuntimeError(
                    f"Expected response['data'] list for {url}. Got {type(data).__name__}. Keys: {keys}"
                )

            for item in data:
                out_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_fp.flush()

            total_items += len(data)
            print(f"Wrote {len(data)} items (total: {total_items})")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote {total_items} items to {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
