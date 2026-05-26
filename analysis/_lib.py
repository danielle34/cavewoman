"""Shared utilities for the CAVEWOMAN analysis pipeline.

Every numbered script imports from this module. Centralizes config loading,
data discovery and loading, bootstrap helpers, and output path management.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


# config
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class Config:
    raw: Dict[str, Any]
    paths: Dict[str, Path]
    design: Dict[str, Any]
    columns: Dict[str, str]
    file_template: str
    augmentation_suffixes: Dict[str, str]
    thresholds: Dict[str, Any]
    stats: Dict[str, Any]
    pricing: Dict[str, Dict[str, float]]
    figures: Dict[str, Any]

    @property
    def models(self) -> List[str]:
        return list(self.design["models"])

    @property
    def conditions(self) -> List[str]:
        return list(self.design["conditions"])

    @property
    def levels(self) -> List[str]:
        return list(self.design["levels"])

    @property
    def datasets(self) -> List[str]:
        return list(self.design["datasets"].keys())

    def n_expected(self, dataset: str) -> int:
        return int(self.design["datasets"][dataset]["n_expected"])

    def task_type(self, dataset: str) -> str:
        return self.design["datasets"][dataset].get("task_type", "unknown")

    def col(self, key: str) -> str:
        return self.columns.get(key, key)


def load_config(path: Optional[str] = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    p = Path(os.path.expanduser(str(p))).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    paths = {k: Path(os.path.expanduser(v)).resolve() for k, v in raw["paths"].items()}
    return Config(
        raw=raw,
        paths=paths,
        design=raw["design"],
        columns=raw["columns"],
        file_template=raw["file_template"],
        augmentation_suffixes=raw["augmentation_suffixes"],
        thresholds=raw["thresholds"],
        stats=raw["stats"],
        pricing=raw["pricing"],
        figures=raw["figures"],
    )


# argparse boilerplate
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Path to config.yaml")
    parser.add_argument("--output-root", default=None,
                        help="Override config.paths.output_root")
    parser.add_argument("--results-root", default=None,
                        help="Override config.paths.results_root")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these model tags")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to these datasets")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=["output", "input"],
                        help="Restrict to these conditions")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    return parser


def apply_common_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if getattr(args, "output_root", None):
        cfg.paths["output_root"] = Path(os.path.expanduser(args.output_root)).resolve()
    if getattr(args, "results_root", None):
        cfg.paths["results_root"] = Path(os.path.expanduser(args.results_root)).resolve()
    if getattr(args, "models", None):
        cfg.design["models"] = args.models
    if getattr(args, "datasets", None):
        cfg.design["datasets"] = {
            d: cfg.design["datasets"][d] for d in args.datasets
            if d in cfg.design["datasets"]
        }
    if getattr(args, "conditions", None):
        cfg.design["conditions"] = args.conditions
    return cfg


# logging + output dirs
def setup_output_dirs(cfg: Config) -> Dict[str, Path]:
    root = cfg.paths["output_root"]
    dirs = {name: (root / name) for name in ("tables", "figures", "stats", "logs", "latex")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dirs["root"] = root
    return dirs


def setup_logging(cfg: Config, script_name: str, verbose: bool = False) -> logging.Logger:
    out = setup_output_dirs(cfg)
    log_path = out["logs"] / f"{script_name}.log"
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format=fmt, datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    return logging.getLogger(script_name)


# data discovery + loading
def lane_path(cfg: Config, model: str, dataset: str, condition: str, level: str,
              augmented: Optional[str] = None) -> Path:
    """Return the JSONL path for one (model, dataset, condition, level)."""
    rel = cfg.file_template.format(
        model_tag=model, dataset=dataset, condition=condition, level=level,
    )
    p = cfg.paths["results_root"] / rel
    if augmented is None:
        return p
    suffix = cfg.augmentation_suffixes[augmented]
    return p.with_name(p.stem + suffix + p.suffix)


def discover_lanes(cfg: Config, logger: Optional[logging.Logger] = None
                  ) -> List[Tuple[str, str, str]]:
    """Return [(model, dataset, condition)] tuples where ALL 5 level files exist."""
    lanes: List[Tuple[str, str, str]] = []
    for m in cfg.models:
        for c in cfg.conditions:
            for ds in cfg.datasets:
                ok = all(lane_path(cfg, m, ds, c, L).exists() for L in cfg.levels)
                if ok:
                    lanes.append((m, ds, c))
                elif logger:
                    logger.debug(f"skip incomplete lane: {m}/{ds}/{c}")
    if logger:
        logger.info(f"discovered {len(lanes)} complete lanes")
    return lanes


def read_jsonl(path: Path) -> pd.DataFrame:
    """Read one JSONL file into a DataFrame (only used internally)."""
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def load_lane(cfg: Config, model: str, dataset: str, condition: str,
              levels: Optional[Sequence[str]] = None,
              with_entailment: bool = True,
              with_embeddings: bool = True,
              logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load all levels of one lane into a long-form DataFrame.

    Prefers entailment-augmented and embedding-augmented files when they
    exist (and the user opts in). Adds model/dataset/condition columns.
    Standardizes level to one of L0..L4 (or whatever the config defines).
    Idx is preserved so paired tests can join across levels.
    """
    levels = levels or cfg.levels
    frames: List[pd.DataFrame] = []
    for L in levels:
        base = lane_path(cfg, model, dataset, condition, L)
        if not base.exists():
            if logger:
                logger.warning(f"missing level file: {base}")
            continue
        df = read_jsonl(base)
        # Try to enrich with entailment / embeddings (joined on idx where present)
        if with_entailment and L != "L0":
            ep = lane_path(cfg, model, dataset, condition, L, augmented="entailment")
            if ep.exists():
                ent = read_jsonl(ep)
                ent_cols = [c for c in ent.columns
                            if "entails" in c or c.startswith("entailment")
                            or "bidirectional" in c]
                key = cfg.col("idx")
                if key in ent.columns and ent_cols:
                    df = df.merge(ent[[key] + ent_cols], on=key, how="left",
                                  suffixes=("", "_ent"))
        if with_embeddings:
            ep = lane_path(cfg, model, dataset, condition, L, augmented="embeddings")
            if ep.exists():
                emb = read_jsonl(ep)
                key = cfg.col("idx")
                emb_cols = [c for c in emb.columns
                            if c in ("embedding_similarity", "embedding_model")]
                if key in emb.columns and emb_cols:
                    df = df.merge(emb[[key] + emb_cols], on=key, how="left",
                                  suffixes=("", "_emb"))
        # Normalize level column
        df[cfg.col("level")] = L
        df["__model"]     = model
        df["__dataset"]   = dataset
        df["__condition"] = condition
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def load_all(cfg: Config, logger: Optional[logging.Logger] = None,
             with_entailment: bool = True,
             with_embeddings: bool = True) -> pd.DataFrame:
    """Load every discovered lane into a single long-form DataFrame.

    Robust to partial sweeps: missing lanes are skipped with a warning.
    """
    lanes = discover_lanes(cfg, logger=logger)
    frames: List[pd.DataFrame] = []
    for m, ds, c in lanes:
        try:
            df = load_lane(cfg, m, ds, c, with_entailment=with_entailment,
                           with_embeddings=with_embeddings, logger=logger)
        except Exception as e:
            if logger:
                logger.warning(f"failed to load {m}/{ds}/{c}: {e}")
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# derived columns
def add_derived_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add canonical derived columns to a long-form DataFrame.

    Adds (skipping any that already exist):
    , semantic_collapse: bool, embedding_similarity < threshold
    , total_tokens:      input_tokens + output_tokens
    , cost_usd_safe:     float, NaN→0
    """
    if df.empty:
        return df
    emb_col = cfg.col("embedding_similarity")
    if emb_col in df.columns and "semantic_collapse" not in df.columns:
        df["semantic_collapse"] = (df[emb_col].astype(float)
                                    < cfg.thresholds["semantic_collapse_cosine"])
    if (cfg.col("input_tokens") in df.columns
            and cfg.col("output_tokens") in df.columns
            and "total_tokens" not in df.columns):
        df["total_tokens"] = (df[cfg.col("input_tokens")].fillna(0).astype(float)
                              + df[cfg.col("output_tokens")].fillna(0).astype(float))
    if cfg.col("cost_usd") in df.columns and "cost_usd_safe" not in df.columns:
        df["cost_usd_safe"] = df[cfg.col("cost_usd")].fillna(0).astype(float)
    return df


# bootstrap + stats helpers
def bootstrap_ci(values: Sequence[float], stat=np.mean,
                 n: int = 2000, ci: float = 0.95,
                 rng: Optional[np.random.Generator] = None
                 ) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI. Returns (point_estimate, lo, hi).

    Returns (nan, nan, nan) for empty / all-NaN inputs.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng if rng is not None else np.random.default_rng()
    boots = np.empty(n, dtype=float)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    for i in range(n):
        boots[i] = stat(arr[idx[i]])
    point = float(stat(arr))
    alpha = (1.0, ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1, alpha])
    return (point, float(lo), float(hi))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n: int = 2000, ci: float = 0.95,
                          rng: Optional[np.random.Generator] = None
                          ) -> Tuple[float, float, float]:
    """Paired bootstrap on (a, b). Inputs must align by index.

    Returns (mean_diff, lo, hi).
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    if arr_a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng if rng is not None else np.random.default_rng()
    diff = arr_a, arr_b
    boots = np.empty(n, dtype=float)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    for i in range(n):
        boots[i] = diff[idx[i]].mean()
    alpha = (1.0, ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1, alpha])
    return (float(diff.mean()), float(lo), float(hi))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for two independent samples; safe on small / NaN inputs."""
    arr_a = np.asarray(a, dtype=float); arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = np.asarray(b, dtype=float); arr_b = arr_b[~np.isnan(arr_b)]
    if arr_a.size < 2 or arr_b.size < 2:
        return float("nan")
    pooled = np.sqrt(((arr_a.var(ddof=1) * (arr_a.size, 1))
                      + (arr_b.var(ddof=1) * (arr_b.size, 1)))
                     / (arr_a.size + arr_b.size, 2))
    if pooled == 0:
        return float("nan")
    return float((arr_a.mean(), arr_b.mean()) / pooled)


def holm_bonferroni(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        v = p[idx] * (m, rank)
        running_max = max(running_max, v)
        adj[idx] = min(1.0, running_max)
    return adj


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR control)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running_min = 1.0
    for rank in range(m, 1, -1, -1):
        idx = order[rank]
        v = p[idx] * m / (rank + 1)
        running_min = min(running_min, v)
        adj[idx] = min(1.0, running_min)
    return adj


def adjust_pvalues(pvalues: Sequence[float], method: str) -> np.ndarray:
    if method in (None, "none"):
        return np.asarray(pvalues, dtype=float)
    if method == "holm-bonferroni":
        return holm_bonferroni(pvalues)
    if method == "benjamini-hochberg":
        return benjamini_hochberg(pvalues)
    raise ValueError(f"unknown multiple-comparison method: {method}")


# table writing helpers
def write_csv(df: pd.DataFrame, path: Path, logger: Optional[logging.Logger] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    if logger:
        logger.info(f"wrote {path}")
    return path


def write_json(obj: Any, path: Path, logger: Optional[logging.Logger] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    if logger:
        logger.info(f"wrote {path}")
    return path


def write_latex(df: pd.DataFrame, path: Path,
                caption: str = "", label: str = "",
                float_format: str = "%.3f",
                logger: Optional[logging.Logger] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # pandas ≥ 2.0
        body = df.to_latex(index=False, float_format=float_format,
                           caption=caption or None, label=label or None,
                           escape=True, longtable=False)
    except Exception:
        # very old pandas, fall back without caption/label
        body = df.to_latex(index=False, float_format=float_format, escape=True)
    path.write_text(body)
    if logger:
        logger.info(f"wrote {path}")
    return path
