# iv_surface_dataset.py
# Memory-safe, single-file pipeline to build IV-surface dataset from SPY yearly JSON dumps (2019–2024).
# - Streams each file (no full json.load to avoid OOM)
# - Robust to nested arrays and JSON-lines
# - Estimates missing spot per date (ATM-delta heuristic), fixes IV scale if in %
# - Builds 64x64 (T x k) daily surfaces + masks, normalizes (global min-max on observed)
# - Saves one .pt bundle at the end

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterable, Union, Generator, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# -------- Config (as per your report) --------
DATA_FILES = [
    "/home/harshil/IV-Surface-Reconstruction/data/spy_options_data_19.json",
    "/home/harshil/IV-Surface-Reconstruction/data/spy_options_data_20.json",
    "/home/harshil/IV-Surface-Reconstruction/data/spy_options_data_21.json",
    "/home/harshil/IV-Surface-Reconstruction/data/spy_options_data_22.json",
    "/home/harshil/IV-Surface-Reconstruction/data/spy_options_data_23.json",
    "/home/harshil/IV-Surface-Reconstruction/data/spy_options_data_24.json",
]
OUT_PATH = "/home/harshil/IV-Surface-Reconstruction/artifacts/spy_iv_2019_2024.pt"

# Grid/domain
K_RANGE = (-0.5, 0.5)  # log-moneyness k = ln(K/S)
T_RANGE_DAYS = (7, 365)  # maturities 7d .. 365d
H, W = 64, 64  # H along T, W along k
DAY_COUNT = 365
IV_BOUNDS = (0.01, 1.00)  # sane IV range (decimals)

# Flexible field names
SPOT_KEYS = [
    "underlying_price",
    "underlying",
    "spot",
    "S",
    "stock_price",
    "adj_close",
    "close",
]
IV_KEYS = ["implied_volatility", "iv", "impliedVol"]


# -------- Small helpers --------
def _parse_date(x: Union[str, datetime]) -> Optional[datetime]:
    if isinstance(x, datetime):
        return x
    if not isinstance(x, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(x[:19], fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return None


def _pick_first(d: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return None


# -------- Streaming JSON: yield dict objects from huge array/list files --------
def iter_json_objects(path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Stream JSON 'objects' from:
      - A big JSON array (possibly nested arrays) of dicts
      - JSON-lines files (one object per line)
    Without loading the entire file into memory.
    """
    with open(path, "r") as f:
        # Peek first non-space char to decide strategy
        start = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                start = ch
                break

        if not start:
            return

        if start == "{":
            # It's a single object or JSON-lines; push back and try JSON-lines fallback
            f.seek(f.tell() - 1)
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        yield obj
                    elif isinstance(obj, list):
                        # if a line accidentally contains a list of dicts
                        for item in obj:
                            if isinstance(item, dict):
                                yield item
                    # else ignore
                except json.JSONDecodeError:
                    # Not pure JSON-lines; fall back to chunked array parsing
                    f.seek(0)
                    break
            else:
                # fully consumed as JSON-lines
                return
            # If we broke out, fall through to array streaming

        # Array (possibly nested); stream by decoding objects between braces
        decoder = json.JSONDecoder()
        buf = start + f.read(1024 * 1024)
        eof = False

        def refill():
            nonlocal buf, eof
            more = f.read(1024 * 1024)
            if more:
                buf += more
            else:
                eof = True

        i = 0
        # Skip until first '[' or '{'
        while i < len(buf) and buf[i] not in "[{":
            i += 1

        # We will try to decode dicts only; skip brackets and commas
        while True:
            # Find next '{'
            while i < len(buf) and buf[i] != "{":
                i += 1
                if i >= len(buf) and not eof:
                    refill()
            if i >= len(buf):
                break

            # Try to raw_decode from '{'
            try:
                obj, end = decoder.raw_decode(buf, i)
            except json.JSONDecodeError:
                if eof:
                    break
                # Need more data
                refill()
                continue

            i = end
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                # flatten a list of dicts at this level
                for item in obj:
                    if isinstance(item, dict):
                        yield item
            # Continue scanning for the next '{'
            if i >= len(buf) and not eof:
                refill()
            if i >= len(buf) and eof:
                break


# -------- Core transforms --------
def add_k_T_columns(df: pd.DataFrame, day_count: int = 365) -> pd.DataFrame:
    df = df.copy()
    safe_spot = df["spot"].replace(0, np.nan)
    df["k"] = np.log(df["strike"] / safe_spot)
    df["T_days"] = (df["expiration"] - df["date"]).dt.days.astype(float)
    df["T"] = df["T_days"] / float(day_count)
    return df


def make_grids(
    k_min: float = K_RANGE[0],
    k_max: float = K_RANGE[1],
    T_min_days: int = T_RANGE_DAYS[0],
    T_max_days: int = T_RANGE_DAYS[1],
    H_: int = H,
    W_: int = W,
    day_count: int = DAY_COUNT,
) -> Tuple[np.ndarray, np.ndarray]:
    T_min = T_min_days / float(day_count)
    T_max = T_max_days / float(day_count)
    T_grid = np.linspace(T_min, T_max, H_, dtype=np.float32)
    k_grid = np.linspace(k_min, k_max, W_, dtype=np.float32)
    return T_grid, k_grid


def _nearest_indices_1d(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    take_right = np.abs(values - right) < np.abs(values - left)
    idx = idx + take_right.astype(np.int64) - 1
    return idx


def bin_to_grid_for_day(
    day_df: pd.DataFrame,
    T_grid: np.ndarray,
    k_grid: np.ndarray,
    iv_bounds: Tuple[float, float] = IV_BOUNDS,
) -> Tuple[np.ndarray, np.ndarray]:
    H_, W_ = len(T_grid), len(k_grid)
    surf = np.full((H_, W_), np.nan, dtype=np.float32)
    cnt = np.zeros((H_, W_), dtype=np.int32)
    lo, hi = iv_bounds
    d = day_df[(day_df["iv"] >= lo) & (day_df["iv"] <= hi)]
    if d.empty:
        return surf, np.zeros_like(surf, dtype=bool)

    Ti = _nearest_indices_1d(T_grid, d["T"].values.astype(np.float32))
    Ki = _nearest_indices_1d(k_grid, d["k"].values.astype(np.float32))
    IV = d["iv"].values.astype(np.float32)

    for t, k, iv in zip(Ti, Ki, IV):
        if np.isnan(surf[t, k]):
            surf[t, k] = iv
            cnt[t, k] = 1
        else:
            cnt[t, k] += 1
            surf[t, k] += (iv - surf[t, k]) / cnt[t, k]

    mask = ~np.isnan(surf)
    surf[~mask] = 0.0
    return surf, mask


def normalize_global_minmax(X: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, Dict]:
    observed = X[M]
    if observed.size == 0:
        raise ValueError("No observed pixels for normalization.")
    vmin = float(np.nanmin(observed))
    vmax = float(np.nanmax(observed))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    Xn = X.copy()
    Xn[M] = (X[M] - vmin) / (vmax - vmin + 1e-12)
    meta = {"norm": "global_minmax", "vmin": vmin, "vmax": vmax}
    return Xn, meta


# -------- Spot estimation & IV scaling per day --------
def estimate_spot_for_day(day_df: pd.DataFrame) -> float:
    g = day_df.copy()
    g["delta"] = pd.to_numeric(g.get("delta", np.nan), errors="coerce")

    def target_delta(row):
        if str(row.get("option_type", "")).lower() == "put":
            return -0.5
        return 0.5

    tgt = g.apply(target_delta, axis=1)
    dist = (g["delta"] - tgt).abs()
    if dist.notna().any():
        idx = dist.idxmin()
        return float(g.loc[idx, "strike"])

    if (
        "open_interest" in g
        and pd.to_numeric(g["open_interest"], errors="coerce").notna().any()
    ):
        oi = pd.to_numeric(g["open_interest"], errors="coerce")
        return float(g.loc[oi.idxmax(), "strike"])
    if "volume" in g and pd.to_numeric(g["volume"], errors="coerce").notna().any():
        vol = pd.to_numeric(g["volume"], errors="coerce")
        return float(g.loc[vol.idxmax(), "strike"])
    return float(np.nanmedian(g["strike"]))


# -------- Main builder: stream → per-file per-date build → append → free --------
def build_and_save_from_json(
    input_paths: Iterable[str],
    out_path: str,
    H_: int = H,
    W_: int = W,
    k_range: Tuple[float, float] = K_RANGE,
    T_range_days: Tuple[int, int] = T_RANGE_DAYS,
    iv_bounds: Tuple[float, float] = IV_BOUNDS,
    day_count: int = DAY_COUNT,
) -> str:
    # Precompute grids once
    T_grid, k_grid = make_grids(
        k_min=k_range[0],
        k_max=k_range[1],
        T_min_days=T_range_days[0],
        T_max_days=T_range_days[1],
        H_=H_,
        W_=W_,
        day_count=day_count,
    )

    X_all: List[np.ndarray] = []
    M_all: List[np.ndarray] = []
    D_all: List[str] = []

    for p in input_paths:
        if not os.path.exists(p):
            print(f"[WARN] Missing file: {p}")
            continue
        print(f"[INFO] Streaming file: {p}")

        # Accumulate records per date (keeps memory bounded per file)
        by_date: Dict[datetime, List[dict]] = defaultdict(list)
        count = 0

        for row in iter_json_objects(p):
            # Minimal extraction, keep as raw dict to reduce overhead
            date = _parse_date(
                row.get("date") or row.get("trade_date") or row.get("quote_date")
            )
            expiration = _parse_date(
                row.get("expiration") or row.get("expiry") or row.get("maturity")
            )
            if date is None or expiration is None:
                continue

            strike = row.get("strike") or row.get("K") or row.get("strike_price")
            try:
                strike_f = float(strike) if strike is not None else None
            except Exception:
                strike_f = None
            if strike_f is None:
                continue

            # IV
            iv_raw = None
            for k in IV_KEYS:
                if k in row and row[k] not in (None, ""):
                    iv_raw = row[k]
                    break
            try:
                iv_f = float(iv_raw) if iv_raw is not None else None
            except Exception:
                iv_f = None
            if iv_f is None:
                continue

            spot = _pick_first(row, SPOT_KEYS)

            rec = {
                "date": date,
                "expiration": expiration,
                "strike": strike_f,
                "iv": iv_f,
                "spot": spot,  # may be None
                "option_type": (
                    row.get("type") or row.get("option_type") or ""
                ).lower(),
                "delta": row.get("delta"),
                "open_interest": row.get("open_interest"),
                "volume": row.get("volume"),
            }
            by_date[date].append(rec)
            count += 1

            # Periodic small progress (no heavy prints)
            if count % 500000 == 0:
                print(f"  parsed {count:,} option rows so far...")

        print(
            f"[INFO] Parsed ~{count:,} rows in {os.path.basename(p)}. Building daily surfaces..."
        )

        # Build per date for this file, then free by_date
        for dt, rows in sorted(by_date.items()):
            df = pd.DataFrame.from_records(rows)

            # Fix IV scaling if % (e.g., 6.45 -> 0.0645)
            med_iv = np.nanmedian(pd.to_numeric(df["iv"], errors="coerce"))
            if np.isfinite(med_iv) and med_iv > 3.0:
                df["iv"] = pd.to_numeric(df["iv"], errors="coerce") / 100.0
            else:
                df["iv"] = pd.to_numeric(df["iv"], errors="coerce")

            # Ensure numeric types
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df["delta"] = pd.to_numeric(df.get("delta", np.nan), errors="coerce")
            if "open_interest" in df:
                df["open_interest"] = pd.to_numeric(
                    df["open_interest"], errors="coerce"
                )
            if "volume" in df:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            # Spot: estimate if missing entirely for the day
            if "spot" not in df or df["spot"].isna().all():
                estS = estimate_spot_for_day(df)
                df["spot"] = float(estS)
            else:
                df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
                if df["spot"].isna().all():
                    estS = estimate_spot_for_day(df)
                    df["spot"] = float(estS)

            # Drop essentials that are still missing
            df = df.dropna(subset=["strike", "iv", "spot"])
            if df.empty:
                continue

            # Add k, T; domain filter; bin to grid
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.floor("D")
            df = add_k_T_columns(df, day_count=day_count)
            df = df[
                (df["T_days"] >= T_RANGE_DAYS[0]) & (df["T_days"] <= T_RANGE_DAYS[1])
            ]
            df = df[(df["k"] >= K_RANGE[0]) & (df["k"] <= K_RANGE[1])]
            if df.empty:
                continue

            surf, mask = bin_to_grid_for_day(df, T_grid, k_grid, iv_bounds=iv_bounds)
            if mask.sum() < 10:
                continue

            X_all.append(surf[np.newaxis, ...])  # [1,H,W]
            M_all.append(mask[np.newaxis, ...])  # [1,H,W]
            D_all.append(pd.to_datetime(dt).strftime("%Y-%m-%d"))

        # free big dict
        by_date.clear()

    if not X_all:
        raise RuntimeError("No usable daily surfaces found across all files.")

    X = np.stack(X_all, axis=0).astype(np.float32)  # [N,1,H,W]
    M = np.stack(M_all, axis=0).astype(bool)  # [N,1,H,W]

    # Normalize on observed pixels
    Xn, norm_meta = normalize_global_minmax(X, M)

    bundle = {
        "x": torch.from_numpy(Xn),
        "mask": torch.from_numpy(M.astype(np.bool_)),
        "dates": D_all,
        "T_grid": torch.from_numpy(T_grid.astype(np.float32)),
        "k_grid": torch.from_numpy(k_grid.astype(np.float32)),
        "meta": {
            "normalization": norm_meta,
            "shape": {"N": int(X.shape[0]), "H": int(X.shape[2]), "W": int(X.shape[3])},
            "k_range": list(K_RANGE),
            "T_range_days": list(T_RANGE_DAYS),
            "iv_bounds": list(IV_BOUNDS),
        },
    }

    # use the function argument `out_path` (not the global constant)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(bundle, out_path)
    return out_path


if __name__ == "__main__":
    print("[START] IV Surface dataset builder (2019–2024, SPY)")
    try:
        saved = build_and_save_from_json(DATA_FILES, OUT_PATH)
        print(f"[DONE] Saved dataset to: {saved}")
        print("[HINT] Load with:  bundle = torch.load(saved, map_location='cpu')")
        print(
            "        bundle keys: x [N,1,64,64], mask [N,1,64,64], dates, T_grid, k_grid, meta"
        )
    except Exception as e:
        print("[ERROR]", repr(e))
