#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga meteorología diaria (2014-01-01 .. hoy) para un conjunto de fincas,
eligiendo automáticamente la mejor fuente/estación entre:
  - SITCAN / GRAFCAN SensorThings API (sensores.grafcan.es)
  - AEMET OpenData (opendata.aemet.es)  <-- requiere API key

Entrada:
  - fincas.tsv (por defecto): columnas esperadas:
        Fca, X, Y     (UTM ETRS89 / UTM 28N, EPSG:25828)
    o bien:
        Fca, lat, lon (WGS84)
Salida:
  - Un TSV por finca, nombre: <Fca>.tsv (en el directorio de salida)

Uso rápido:
  python download_meteo.py \
    --fincas fincas.tsv \
    --out out_meteo

Notas:
- La cobertura histórica del SITCAN puede empezar años después de 2014; por eso
  la selección prioriza SITCAN solo si su cobertura en el periodo solicitado
  supera un umbral configurable (por defecto 90% de días con T o P).
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

try:
    from urllib3.exceptions import InsecureRequestWarning
except Exception:
    InsecureRequestWarning = None

SITCAN_STATIONS_CSV = "https://opendata.sitcan.es/upload/meteorologia/estaciones.csv"
SITCAN_API_BASE = "https://sensores.grafcan.es/api/v1.0"
AEMET_API_BASE = "https://opendata.aemet.es/opendata/api"
AEMET_API_KEY_FILE = os.path.join(os.path.dirname(__file__), "aemet_api_key.txt")
CACHE_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), ".cache_meteo")
REQUESTS_VERIFY: str | bool = True


# -------------------------
# Utilidades generales
# -------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def configure_ssl_verify(ca_bundle: Optional[str], insecure: bool = False) -> str | bool:
    if insecure:
        if InsecureRequestWarning is not None:
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        return False
    if ca_bundle:
        if not os.path.isfile(ca_bundle):
            raise RuntimeError(f"CA bundle no encontrado: {ca_bundle}")
        return ca_bundle
    return True


def request_get_with_retries(
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: int = 60,
        max_attempts: int = 5,
) -> requests.Response:
    request_kwargs = {"params": params, "headers": headers, "timeout": timeout}
    if REQUESTS_VERIFY is not True:
        request_kwargs["verify"] = REQUESTS_VERIFY

    for attempt in range(max_attempts):
        try:
            r = requests.get(url, **request_kwargs)
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                "Error SSL verificando certificado. Usa --ca-bundle /ruta/cacert.pem, "
                "--insecure, o exporta REQUESTS_CA_BUNDLE/SSL_CERT_FILE."
            ) from exc
        except requests.exceptions.RequestException:
            if attempt >= max_attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 15))
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_attempts - 1:
                r.raise_for_status()
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else min(2 ** attempt, 15)
            time.sleep(wait)
            continue

        r.raise_for_status()
        return r

    raise RuntimeError(f"No se pudo completar GET tras {max_attempts} intentos: {url}")


def safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 60) -> dict:
    r = request_get_with_retries(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    # A veces viene con encoding raro: forzamos utf-8 si falla
    try:
        return r.json()
    except Exception:
        return json.loads(r.content.decode("utf-8", errors="replace"))


def safe_get_text(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 60) -> str:
    r = request_get_with_retries(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    if r.encoding is None:
        r.encoding = "utf-8"
    return r.text


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def station_cache_filename(cache_key: Tuple[str, str, str, str]) -> str:
    source, station_ref, start_s, end_s = cache_key
    safe_station = re.sub(r"[^A-Za-z0-9_.-]+", "_", station_ref)[:48] or "unknown"
    raw = f"{source}|{station_ref}|{start_s}|{end_s}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"daily_{source}_{safe_station}_{start_s}_{end_s}_{digest}.pkl"


def station_cache_path(cache_dir: str, cache_key: Tuple[str, str, str, str]) -> str:
    daily_dir = os.path.join(cache_dir, "daily")
    ensure_dir(daily_dir)
    return os.path.join(daily_dir, station_cache_filename(cache_key))


def load_station_daily_cache_disk(cache_dir: str, cache_key: Tuple[str, str, str, str]) -> Optional[pd.DataFrame]:
    path = station_cache_path(cache_dir, cache_key)
    if not os.path.isfile(path):
        return None
    try:
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
    except Exception:
        return None
    return None


def save_station_daily_cache_disk(cache_dir: str, cache_key: Tuple[str, str, str, str], df: pd.DataFrame) -> None:
    path = station_cache_path(cache_dir, cache_key)
    try:
        df.to_pickle(path)
    except Exception:
        # Cache best-effort: no interrumpir ejecución principal.
        pass


def sitcan_coverage_cache_path(cache_dir: str) -> str:
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, "sitcan_coverage_cache.json")


def load_sitcan_coverage_cache_disk(cache_dir: str) -> Dict[Tuple[int, str, str], float]:
    path = sitcan_coverage_cache_path(cache_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[Tuple[int, str, str], float] = {}
        for k, v in raw.items():
            parts = str(k).split("|")
            if len(parts) != 3:
                continue
            thing_id, start_s, end_s = parts
            out[(int(thing_id), start_s, end_s)] = float(v)
        return out
    except Exception:
        return {}


def save_sitcan_coverage_cache_disk(cache_dir: str, cache: Dict[Tuple[int, str, str], float]) -> None:
    path = sitcan_coverage_cache_path(cache_dir)
    raw = {f"{thing_id}|{start_s}|{end_s}": float(v) for (thing_id, start_s, end_s), v in cache.items()}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=True)
    except Exception:
        # Cache best-effort: no interrumpir ejecución principal.
        pass


def summarize_existing_output(path: str) -> Dict[str, object]:
    rows = 0
    source = "existing_file"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            rows = max(sum(1 for _ in f) - 1, 0)
    except Exception:
        rows = 0

    try:
        head = pd.read_csv(path, sep="\t", nrows=1)
        if "source" in head.columns and not head.empty and pd.notna(head.iloc[0].get("source")):
            source = str(head.iloc[0]["source"])
    except Exception:
        pass

    return {"source": source, "rows": int(rows)}


def load_key_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key and not key.startswith("#"):
                return key
    raise RuntimeError(f"Fichero de API key vacío: {path}")


def resolve_aemet_key() -> str:
    if not os.path.isfile(AEMET_API_KEY_FILE):
        raise RuntimeError(f"No existe el fichero de API key: {AEMET_API_KEY_FILE}")
    return load_key_from_file(AEMET_API_KEY_FILE)


def parse_aemet_latlon(s: str) -> Optional[float]:
    """
    AEMET a veces devuelve lat/long como DDMMSSX (ej: '280648N', '0161548W')
    Devuelve decimal con signo. Si ya es decimal, intenta parsearlo también.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    if not s:
        return None

    # Decimal directo
    try:
        return float(s.replace(",", "."))
    except Exception:
        pass

    m = re.match(r"^(\d{2,3})(\d{2})(\d{2})([NSEW])$", s)
    if not m:
        return None
    deg, minute, sec, hemi = m.groups()
    deg = int(deg);
    minute = int(minute);
    sec = int(sec)
    dec = deg + minute / 60.0 + sec / 3600.0
    if hemi in ("S", "W"):
        dec = -dec
    return dec


def utm_to_wgs84(zone_number: int, easting: np.ndarray, northing: np.ndarray, northern: bool = True) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Conversión UTM -> WGS84 en puro Python/NumPy (fallback cuando pyproj no está).
    """
    # WGS84
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    k0 = 0.9996

    x = easting.astype(float) - 500000.0
    y = northing.astype(float)
    if not northern:
        y = y - 10000000.0

    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))

    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512
    fp = mu + j1 * np.sin(2 * mu) + j2 * np.sin(4 * mu) + j3 * np.sin(6 * mu) + j4 * np.sin(8 * mu)

    sin_fp = np.sin(fp)
    cos_fp = np.cos(fp)
    tan_fp = np.tan(fp)

    n1 = a / np.sqrt(1 - e2 * sin_fp ** 2)
    r1 = a * (1 - e2) / (1 - e2 * sin_fp ** 2) ** 1.5
    t1 = tan_fp ** 2
    c1 = ep2 * cos_fp ** 2
    d = x / (n1 * k0)

    lat = fp - (n1 * tan_fp / r1) * (
            d ** 2 / 2
            - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
            + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720
    )

    lon0 = np.deg2rad((zone_number - 1) * 6 - 180 + 3)
    lon = lon0 + (
            d
            - (1 + 2 * t1 + c1) * d ** 3 / 6
            + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120
    ) / cos_fp

    return np.rad2deg(lat), np.rad2deg(lon)


def to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return np.nan
    # AEMET usa coma decimal a veces
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


# -------------------------
# AEMET
# -------------------------

class AemetTransientError(RuntimeError):
    pass


def aemet_hateoas_get(endpoint: str, api_key: str, params: Optional[dict] = None) -> list:
    """
    En AEMET OpenData, una llamada suele devolver {"estado":200, "datos":"<url>", ...}
    Luego hay que hacer GET a ese 'datos' para obtener la lista real.
    """
    transient_status = {429, 500, 502, 503, 504}
    url = f"{AEMET_API_BASE}{endpoint}"
    p = {} if params is None else dict(params)
    p["api_key"] = api_key
    last_error: Optional[BaseException] = None

    for attempt in range(3):
        try:
            meta = safe_get(url, params=p)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in transient_status:
                last_error = exc
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise AemetTransientError(f"AEMET temporal en metadatos ({endpoint}): HTTP {status}") from exc
            raise

        estado = meta.get("estado") if isinstance(meta, dict) else None
        if "datos" not in meta:
            if estado == 404:
                return []
            if estado in transient_status:
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise AemetTransientError(f"AEMET temporal en metadatos ({endpoint}): {meta}")
            raise RuntimeError(f"AEMET respuesta inesperada para {endpoint}: {meta}")

        data_url = meta["datos"]
        try:
            data = safe_get(data_url)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in transient_status:
                last_error = exc
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise AemetTransientError(
                    f"AEMET temporal en datos ({endpoint} -> {data_url}): HTTP {status}"
                ) from exc
            raise

        # A veces devuelve dict con "error"; normalizamos a lista vacía
        if isinstance(data, dict) and "error" in data:
            return []
        if isinstance(data, list):
            return data
        return data.get("datos", []) if isinstance(data, dict) else []

    raise AemetTransientError(f"AEMET temporal en {endpoint}") from last_error


def aemet_inventory(api_key: str) -> pd.DataFrame:
    """
    Inventario de estaciones de valores climatológicos.
    """
    rows = aemet_hateoas_get("/valores/climatologicos/inventarioestaciones/todasestaciones/", api_key)
    df = pd.DataFrame(rows)
    # Normaliza coordenadas
    if "latitud" in df.columns:
        df["lat"] = df["latitud"].apply(parse_aemet_latlon)
    if "longitud" in df.columns:
        df["lon"] = df["longitud"].apply(parse_aemet_latlon)
    return df


def add_months(d: dt.date, months: int) -> dt.date:
    month_idx = (d.month - 1) + months
    year = d.year + month_idx // 12
    month = month_idx % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def iter_date_windows_by_months(start: dt.date, end: dt.date, months: int = 6) -> Iterable[Tuple[dt.date, dt.date]]:
    cur = start
    while cur <= end:
        win_end = add_months(cur, months) - dt.timedelta(days=1)
        if win_end > end:
            win_end = end
        yield cur, win_end
        cur = win_end + dt.timedelta(days=1)


def aemet_daily(api_key: str, indicativo: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Descarga valores climatológicos diarios para estación AEMET.
    Endpoint (según Swagger/ejemplos): /valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/estacion/{}
    """
    all_rows: List[dict] = []
    for win_start, win_end in iter_date_windows_by_months(start, end, months=6):
        fechaini = f"{win_start.isoformat()}T00:00:00UTC"
        fechafin = f"{win_end.isoformat()}T23:59:59UTC"
        endpoint = f"/valores/climatologicos/diarios/datos/fechaini/{fechaini}/fechafin/{fechafin}/estacion/{indicativo}"
        try:
            rows = aemet_hateoas_get(endpoint, api_key)
        except AemetTransientError as exc:
            print(
                f"  !! Aviso AEMET temporal {indicativo} [{win_start}..{win_end}]: {exc}. Se continúa.",
                file=sys.stderr,
            )
            continue
        except RuntimeError as exc:
            msg = str(exc)
            # En ciertos tramos puede no haber datos para la estación.
            if "No hay datos" in msg or "'estado': 404" in msg:
                continue
            raise
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).drop_duplicates()
    # fecha -> date
    if "fecha" in df.columns:
        df["date"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    # Normaliza numéricas típicas (puedes ampliar)
    numeric_cols = [
        "tmed", "tmin", "tmax", "prec", "hrMedia", "hrMax", "hrMin",
        "presMax", "presMin", "velmedia", "racha", "sol", "nieve", "tmin", "tmax"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_float)

    # Selección de columnas útiles (manteniendo todo por si acaso)
    df = df.sort_values("date")
    return df


# -------------------------
# SITCAN / SensorThings
# -------------------------

@dataclass
class SitcanStation:
    station_id: int
    name: str
    lat: float
    lon: float


def parse_wkt_point(s: object) -> Tuple[float, float]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (np.nan, np.nan)
    text = str(s).strip()
    # Formato típico: POINT(lon lat)
    m = re.match(r"^POINT\s*\(\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*\)$", text, flags=re.IGNORECASE)
    if not m:
        return (np.nan, np.nan)
    lon, lat = m.groups()
    return (float(lon), float(lat))


def sitcan_load_stations() -> pd.DataFrame:
    # estaciones.csv se publica en SITCAN Open Data
    csv_text = safe_get_text(SITCAN_STATIONS_CSV)
    df = pd.read_csv(io.StringIO(csv_text))
    # Normalización de nombres de columnas más comunes:
    # En versiones distintas puede variar, intentamos localizar lat/lon y nombre.
    colmap = {c.lower(): c for c in df.columns}
    # candidatos
    name_col = colmap.get("nombre") or colmap.get("name") or colmap.get("thing_name") or colmap.get(
        "location_description") or df.columns[0]
    lat_col = colmap.get("lat") or colmap.get("latitud") or colmap.get("y")
    lon_col = colmap.get("lon") or colmap.get("longitud") or colmap.get("x")
    coord_col = colmap.get("location_coordinates")

    if (lat_col is None or lon_col is None) and coord_col is not None:
        parsed = df[coord_col].apply(parse_wkt_point)
        df["_lon"] = parsed.apply(lambda p: p[0])
        df["_lat"] = parsed.apply(lambda p: p[1])
        lon_col = "_lon"
        lat_col = "_lat"

    # Si no existen lat/lon, abortamos con mensaje claro
    if lat_col is None or lon_col is None:
        raise RuntimeError(
            f"No encuentro columnas de lat/lon (ni location_coordinates) en estaciones.csv; columnas: {list(df.columns)}")

    # Identificador: en dumps recientes es 'thing_id'
    id_col = colmap.get("thing_id") or colmap.get("id") or colmap.get("station_id") or colmap.get("location_id") or \
             df.columns[0]

    out = pd.DataFrame({
        "station_id": pd.to_numeric(df[id_col], errors="coerce"),
        "name": df[name_col].astype(str),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    }).dropna(subset=["station_id", "lat", "lon"])
    out["station_id"] = out["station_id"].astype(int)
    return out


def sitcan_find_nearest_station(stations: pd.DataFrame, lat: float, lon: float) -> SitcanStation:
    d = stations.apply(lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1)
    idx = int(d.idxmin())
    r = stations.loc[idx]
    return SitcanStation(int(r["station_id"]), str(r["name"]), float(r["lat"]), float(r["lon"]))


def sitcan_get_openapi() -> dict:
    # El propio servicio expone OpenAPI
    return safe_get(f"{SITCAN_API_BASE}/?format=openapi")


def sitcan_guess_thing_id_by_station_code(station_id: int) -> Optional[int]:
    """
    Heurística: muchos despliegues asocian el station_id del CSV con Thing@iot.id.
    Si no coincide, el script cae a búsqueda por nombre/ubicación (más costosa).
    """
    # probamos a acceder a Things(<id>)
    url = f"{SITCAN_API_BASE}/Things({station_id})"
    try:
        _ = safe_get(url)
        return station_id
    except Exception:
        return None


def sitcan_find_thing_by_location(lat: float, lon: float, max_candidates: int = 2000) -> Optional[int]:
    """
    Fallback: descarga Things (paginado) y escoge el más cercano por Location.
    Puede ser costoso; úsalo solo si la heurística no funciona.
    """
    # OData: /Things?$expand=Locations&$top=...
    url = f"{SITCAN_API_BASE}/Things"
    params = {"$expand": "Locations", "$top": 200}
    best = (None, 1e18)
    next_url = url
    while next_url and max_candidates > 0:
        js = safe_get(next_url, params=params if next_url == url else None)
        for t in js.get("value", []):
            locs = t.get("Locations", []) or t.get("Locations@iot.navigationLink")
            # si viene expandido:
            if isinstance(locs, list) and locs:
                # GeoJSON-ish: location -> {"type":"Point","coordinates":[lon,lat]}
                loc = locs[0].get("location") or {}
                coords = loc.get("coordinates")
                if coords and len(coords) == 2:
                    tlon, tlat = coords[0], coords[1]
                    dist = haversine_km(lat, lon, tlat, tlon)
                    if dist < best[1]:
                        best = (t.get("@iot.id"), dist)
        next_url = js.get("@iot.nextLink")
        max_candidates -= len(js.get("value", []))
        params = None
    return best[0]


def sitcan_get_datastreams(thing_id: int) -> pd.DataFrame:
    """
    Devuelve datastreams de un Thing con expand de ObservedProperty.
    """
    url = f"{SITCAN_API_BASE}/Things({thing_id})"
    params = {"$expand": "Datastreams($expand=ObservedProperty)"}
    js = safe_get(url, params=params)
    ds = js.get("Datastreams", [])
    rows = []
    for d in ds:
        op = d.get("ObservedProperty", {}) or {}
        rows.append({
            "datastream_id": d.get("@iot.id"),
            "name": d.get("name"),
            "description": d.get("description"),
            "unit": (d.get("unitOfMeasurement") or {}).get("symbol"),
            "observed_property": op.get("name") or op.get("description"),
        })
    return pd.DataFrame(rows)


def sitcan_pick_datastreams(datastreams: pd.DataFrame) -> Dict[str, int]:
    """
    Intenta seleccionar datastreams relevantes para meteorología diaria.
    Mapeo -> id:
      - t (temperatura), rh (humedad), p (precip), ws (viento), sr (radiación)
    """
    wanted = {}
    # patrones simples; ajusta si ves nombres distintos
    patterns = {
        "t": r"temp|temperatura",
        "rh": r"hum|humedad|rh",
        "p": r"precip|lluv|rain|pluv",
        "ws": r"viento|wind|veloc",
        "sr": r"radia|solar|irradi",
        "pres": r"presi|pressure",
    }
    for key, pat in patterns.items():
        m = datastreams["observed_property"].fillna("").str.lower().str.contains(pat, regex=True) | \
            datastreams["name"].fillna("").str.lower().str.contains(pat, regex=True)
        if m.any():
            wanted[key] = int(datastreams[m].iloc[0]["datastream_id"])
    return wanted


def sitcan_fetch_observations(datastream_id: int, start: dt.date, end: dt.date, page_size: int = 2000) -> pd.DataFrame:
    """
    Descarga observaciones para un datastream y las devuelve como dataframe:
    columns: phenomenonTime (datetime), result (float)
    """
    start_iso = f"{start.isoformat()}T00:00:00Z"
    end_iso = f"{end.isoformat()}T23:59:59Z"
    base = f"{SITCAN_API_BASE}/Datastreams({datastream_id})/Observations"
    # OData filter
    flt = f"phenomenonTime ge {start_iso} and phenomenonTime le {end_iso}"
    params = {
        "$select": "phenomenonTime,result",
        "$orderby": "phenomenonTime asc",
        "$filter": flt,
        "$top": page_size
    }
    out_rows = []
    next_url = base
    while next_url:
        js = safe_get(next_url, params=params if next_url == base else None)
        vals = js.get("value", [])
        for o in vals:
            out_rows.append((o.get("phenomenonTime"), o.get("result")))
        next_url = js.get("@iot.nextLink")
        params = None
        # safety break
        if not vals:
            break

    df = pd.DataFrame(out_rows, columns=["phenomenonTime", "result"])
    if df.empty:
        return df
    df["phenomenonTime"] = pd.to_datetime(df["phenomenonTime"], errors="coerce", utc=True)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.dropna(subset=["phenomenonTime"])
    return df


def sitcan_daily_from_datastreams(thing_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Construye un dataframe diario agregando observaciones (si son subdiarias) por fecha.
    Agregación:
      - temperatura: media diaria
      - precipitación: suma diaria
      - humedad: media diaria
      - viento: media diaria
      - radiación: suma diaria (si aplica)
    """
    ds = sitcan_get_datastreams(thing_id)
    if ds.empty:
        return pd.DataFrame()

    chosen = sitcan_pick_datastreams(ds)
    if not chosen:
        return pd.DataFrame()

    daily = None
    agg_map = {
        "t": "mean",
        "rh": "mean",
        "ws": "mean",
        "pres": "mean",
        "p": "sum",
        "sr": "sum",
    }

    for key, ds_id in chosen.items():
        obs = sitcan_fetch_observations(ds_id, start, end)
        if obs.empty:
            continue
        obs["date"] = obs["phenomenonTime"].dt.date
        series = obs.groupby("date")["result"].agg(agg_map.get(key, "mean")).rename(key)
        df = series.reset_index()
        daily = df if daily is None else daily.merge(df, on="date", how="outer")

    if daily is None:
        return pd.DataFrame()
    daily = daily.sort_values("date")
    return daily


def coverage_ratio(df: pd.DataFrame, start: dt.date, end: dt.date, cols: List[str]) -> float:
    """
    % de días con datos (al menos uno de cols no nulo).
    """
    if df.empty or "date" not in df.columns:
        return 0.0
    full_days = pd.date_range(start, end, freq="D").date
    m = df.set_index("date")
    present = []
    for d in full_days:
        if d in m.index:
            row = m.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            present.append(any(pd.notna(row.get(c)) for c in cols if c in m.columns))
        else:
            present.append(False)
    return float(np.mean(present))


# -------------------------
# Pipeline por finca
# -------------------------

def load_fincas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    if "lat" in cols and "lon" in cols:
        df["lat"] = pd.to_numeric(df[cols["lat"]], errors="coerce")
        df["lon"] = pd.to_numeric(df[cols["lon"]], errors="coerce")
    elif "x" in cols and "y" in cols:
        x = pd.to_numeric(df[cols["x"]], errors="coerce").astype(float).values
        y = pd.to_numeric(df[cols["y"]], errors="coerce").astype(float).values
        if Transformer is not None:
            # asumimos EPSG:25828 -> WGS84
            tf = Transformer.from_crs("EPSG:25828", "EPSG:4326", always_xy=True)
            lon, lat = tf.transform(x, y)
            df["lat"] = lat
            df["lon"] = lon
        else:
            # Fallback para entornos sin pyproj.
            lat, lon = utm_to_wgs84(zone_number=28, easting=x, northing=y, northern=True)
            df["lat"] = lat
            df["lon"] = lon
    else:
        raise RuntimeError(f"No encuentro columnas lat/lon ni X/Y en {path}. Columnas: {list(df.columns)}")

    fca_col = cols.get("fca") or cols.get("finca") or df.columns[0]
    df["Fca"] = df[fca_col].astype(str)
    return df[["Fca", "lat", "lon"]].dropna(subset=["lat", "lon"])


def pick_best_source_for_finca(
        lat: float,
        lon: float,
        sitcan_stations: pd.DataFrame,
        aemet_stations: pd.DataFrame,
        aemet_key: str,
        start: dt.date,
        end: dt.date,
        sitcan_min_coverage: float,
        sitcan_coverage_cache: Optional[Dict[Tuple[int, str, str], float]] = None,
) -> Tuple[str, dict]:
    """
    Devuelve ('sitcan'|'aemet', metadata)
    """
    sit_meta = None
    if sitcan_stations is not None and not sitcan_stations.empty:
        try:
            # Nearest SITCAN
            sit = sitcan_find_nearest_station(sitcan_stations, lat, lon)
            thing_id = sitcan_guess_thing_id_by_station_code(sit.station_id)
            if thing_id is None:
                # fallback caro: buscar por localización
                thing_id = sitcan_find_thing_by_location(sit.lat, sit.lon)

            if thing_id is not None:
                cov_cache_key = (int(thing_id), start.isoformat(), end.isoformat())
                if sitcan_coverage_cache is not None and cov_cache_key in sitcan_coverage_cache:
                    cov = sitcan_coverage_cache[cov_cache_key]
                else:
                    sit_daily = sitcan_daily_from_datastreams(thing_id, start, end)
                    cov = coverage_ratio(sit_daily, start, end, cols=["t", "p"])
                    if sitcan_coverage_cache is not None:
                        sitcan_coverage_cache[cov_cache_key] = cov
                if cov >= sitcan_min_coverage:
                    sit_meta = {
                        "thing_id": thing_id,
                        "station_id": sit.station_id,
                        "station_name": sit.name,
                        "station_lat": sit.lat,
                        "station_lon": sit.lon,
                        "distance_km": haversine_km(lat, lon, sit.lat, sit.lon),
                    }
        except Exception:
            sit_meta = None

    if sit_meta is not None:
        return "sitcan", sit_meta

    # AEMET nearest
    a = aemet_stations.dropna(subset=["lat", "lon"]).copy()
    a["dist_km"] = a.apply(lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1)
    r = a.sort_values("dist_km").iloc[0]
    return "aemet", {
        "indicativo": r.get("indicativo"),
        "nombre": r.get("nombre"),
        "provincia": r.get("provincia"),
        "station_lat": float(r["lat"]),
        "station_lon": float(r["lon"]),
        "distance_km": float(r["dist_km"]),
    }


def build_daily_meteo_for_finca(source: str, meta: dict, start: dt.date, end: dt.date, aemet_key: str) -> pd.DataFrame:
    if source == "aemet":
        df = aemet_daily(aemet_key, str(meta["indicativo"]), start, end)
        if df.empty:
            return df
        # Dejamos un set compacto de variables típicas (y conserva otras si existen)
        keep_first = ["date", "tmed", "tmin", "tmax", "prec", "hrMedia", "velmedia", "dir", "racha", "presMax",
                      "presMin", "sol"]
        cols = [c for c in keep_first if c in df.columns] + [c for c in df.columns if c not in keep_first]
        df = df[cols]
        return df

    if source == "sitcan":
        daily = sitcan_daily_from_datastreams(int(meta["thing_id"]), start, end)
        return daily

    raise ValueError(f"source desconocida: {source}")


def main():
    global REQUESTS_VERIFY
    ap = argparse.ArgumentParser()
    ap.add_argument("--fincas", default="fincas.tsv", help="TSV de fincas (Fca + X/Y o lat/lon)")
    ap.add_argument("--out", default="out_meteo", help="Directorio de salida")
    ap.add_argument("--cache-dir", default=os.environ.get("METEO_CACHE_DIR") or CACHE_DIR_DEFAULT,
                    help="Directorio de cache persistente")
    ap.add_argument("--start", default="2014-01-01", help="Fecha inicio (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Fecha fin (YYYY-MM-DD), por defecto hoy")
    ap.add_argument("--ca-bundle", default=os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE"),
                    help="Ruta a CA bundle PEM para validar HTTPS")
    ap.add_argument("--insecure", dest="insecure", action="store_true", default=True,
                    help="Desactiva verificación TLS/SSL (por defecto activado)")
    ap.add_argument("--sitcan-min-coverage", type=float, default=0.90,
                    help="Cobertura mínima de SITCAN para preferirlo (0..1)")
    ap.add_argument("--max-fincas", type=int, default=0, help="Para pruebas: limita nº de fincas (0=sin límite)")
    args = ap.parse_args()

    try:
        aemet_key = resolve_aemet_key()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if not aemet_key:
        print(
            f"ERROR: falta API key en {AEMET_API_KEY_FILE}",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.insecure:
        print("AVISO: SSL/TLS sin verificación (modo por defecto).", file=sys.stderr)
    try:
        REQUESTS_VERIFY = configure_ssl_verify(args.ca_bundle, insecure=args.insecure)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    ensure_dir(args.out)
    ensure_dir(args.cache_dir)

    fincas = load_fincas(args.fincas)
    if args.max_fincas and args.max_fincas > 0:
        fincas = fincas.head(args.max_fincas)

    if all(os.path.isfile(os.path.join(args.out, f"{str(row['Fca'])}.tsv")) for _, row in fincas.iterrows()):
        print(f"Fincas: {len(fincas)} | Periodo: {start} .. {end}")
        print("Todos los ficheros de finca ya existen. Se omiten descargas.")
        decisions = []
        for _, row in fincas.iterrows():
            fca = str(row["Fca"])
            out_path = os.path.join(args.out, f"{fca}.tsv")
            existing = summarize_existing_output(out_path)
            decisions.append({
                "Fca": fca,
                "source": existing["source"],
                "rows": int(existing["rows"]),
                "skipped_existing": True,
            })
        summary_path = os.path.join(args.out, "_summary_stations.tsv")
        pd.DataFrame(decisions).to_csv(summary_path, sep="\t", index=False)
        print(f"\nOK. Resumen: {summary_path}")
        print(f"TSVs por finca en: {args.out}")
        return

    print(f"Fincas: {len(fincas)} | Periodo: {start} .. {end}")
    print("Cargando estaciones SITCAN...")
    try:
        sitcan_stations = sitcan_load_stations()
    except Exception as exc:
        print(f"  !! Aviso SITCAN: {exc}. Se usará AEMET.", file=sys.stderr)
        sitcan_stations = pd.DataFrame(columns=["station_id", "name", "lat", "lon"])

    print("Cargando inventario AEMET...")
    aemet_stations = aemet_inventory(aemet_key)
    if aemet_stations.empty:
        raise RuntimeError("Inventario AEMET vacío. ¿API key correcta?")

    # Resumen global
    decisions = []
    station_daily_cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
    sitcan_coverage_cache: Dict[Tuple[int, str, str], float] = load_sitcan_coverage_cache_disk(args.cache_dir)
    if sitcan_coverage_cache:
        print(f"Cobertura SITCAN cacheada: {len(sitcan_coverage_cache)} entradas")

    for i, row in fincas.iterrows():
        fca = row["Fca"]
        lat = float(row["lat"]);
        lon = float(row["lon"])
        out_path = os.path.join(args.out, f"{fca}.tsv")

        print(f"\n[{i + 1}/{len(fincas)}] Finca {fca} ({lat:.6f},{lon:.6f})")
        if os.path.isfile(out_path):
            existing = summarize_existing_output(out_path)
            print(f"  -> fichero existente: {os.path.basename(out_path)}; se omite descarga.")
            decisions.append({
                "Fca": fca,
                "source": existing["source"],
                "rows": int(existing["rows"]),
                "skipped_existing": True,
            })
            continue

        try:
            source, meta = pick_best_source_for_finca(
                lat, lon, sitcan_stations, aemet_stations, aemet_key,
                start, end, args.sitcan_min_coverage,
                sitcan_coverage_cache=sitcan_coverage_cache,
            )
            print(
                f"  -> fuente elegida: {source} | estación: {meta.get('station_name') or meta.get('nombre')} | dist={meta.get('distance_km'):.2f} km")

            if source == "aemet":
                station_ref = str(meta.get("indicativo", ""))
            elif source == "sitcan":
                station_ref = str(meta.get("thing_id", ""))
            else:
                station_ref = ""
            cache_key = (source, station_ref, start.isoformat(), end.isoformat())

            if cache_key in station_daily_cache:
                print(f"  -> cache hit: {source}:{station_ref}")
                df = station_daily_cache[cache_key].copy()
            else:
                disk_df = load_station_daily_cache_disk(args.cache_dir, cache_key)
                if disk_df is not None:
                    print(f"  -> cache disk hit: {source}:{station_ref}")
                    df = disk_df.copy()
                    station_daily_cache[cache_key] = df.copy()
                else:
                    print(f"  -> cache miss: {source}:{station_ref} (descargando)")
                    df = build_daily_meteo_for_finca(source, meta, start, end, aemet_key)
                    station_daily_cache[cache_key] = df.copy()
                    save_station_daily_cache_disk(args.cache_dir, cache_key, df)

            if df.empty:
                print("  !! Sin datos. Se guarda fichero vacío con cabecera.")
            # Añadimos metadatos como columnas fijas
            df = df.copy()
            df.insert(0, "Fca", fca)
            df.insert(1, "source", source)
            for k in ["station_id", "thing_id", "indicativo", "nombre", "provincia", "station_name", "station_lat",
                      "station_lon", "distance_km"]:
                if k in meta:
                    df[k] = meta[k]

            df.to_csv(out_path, sep="\t", index=False)
            decisions.append({"Fca": fca, "source": source, **meta, "rows": int(len(df))})
        except Exception as exc:
            print(f"  !! Error en finca {fca}: {exc}. Se continúa con la siguiente.", file=sys.stderr)
            pd.DataFrame([{"Fca": fca, "source": "error", "error": str(exc)}]).to_csv(out_path, sep="\t", index=False)
            decisions.append({"Fca": fca, "source": "error", "error": str(exc), "rows": 0})
            continue

    # Guardamos resumen
    summary_path = os.path.join(args.out, "_summary_stations.tsv")
    pd.DataFrame(decisions).to_csv(summary_path, sep="\t", index=False)
    save_sitcan_coverage_cache_disk(args.cache_dir, sitcan_coverage_cache)
    print(f"\nOK. Resumen: {summary_path}")
    print(f"TSVs por finca en: {args.out}")


if __name__ == "__main__":
    main()
