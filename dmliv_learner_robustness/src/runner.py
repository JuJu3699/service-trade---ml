"""DML-IV learner-sensitivity experiments for US bilateral service trade.

Designed for EconML 0.16.0 DMLIV. The input file is PRIVATE and already contains
all necessary logarithmically transformed variables and constructed instruments.
No raw research data is included in this repository.

The main analysis in the manuscript used Random Forest. Parameters in config.yaml
are ILLUSTRATIVE values for additional robustness tests, not verified original settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
    "partner", "year", "commercial_export", "commercial_import",
    "transport_export", "transport_import", "travel_export", "travel_import",
    "direct_cities", "avg_seats", "internet",
    "direct_cities_iv", "avg_seats_iv", "internet_iv",
    "rta", "lsbci", "exchange_rate", "gdp_pc",
]


class ShuffledGroupKFold:
    """Repeatable group-respecting randomized folds, for sensitivity tests only.

    Unlike plain GroupKFold, changing ``random_state`` actually changes fold
    assignment. Each partner appears in one test fold and in no train rows there.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("ShuffledGroupKFold requires groups=partner identifiers")
        g = np.asarray(groups)
        labels, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
        if len(labels) < self.n_splits:
            raise ValueError(f"Need >= {self.n_splits} partner groups; got {len(labels)}")
        rng = np.random.default_rng(self.random_state)
        # Shuffle first, then assign larger groups greedily with randomized tie breaks.
        order = rng.permutation(len(labels))
        order = order[np.argsort(-counts[order], kind="stable")]
        fold_load = np.zeros(self.n_splits, dtype=int)
        fold_for_label = np.zeros(len(labels), dtype=int)
        for j in order:
            i = int(np.argmin(fold_load))
            fold_for_label[j] = i
            fold_load[i] += int(counts[j])
        row_fold = fold_for_label[inv]
        for i in range(self.n_splits):
            test = np.flatnonzero(row_fold == i)
            train = np.flatnonzero(row_fold != i)
            if not len(test) or not len(train):
                raise ValueError("Empty cross-fitting fold")
            yield train, test


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a mapping")
    for key in ("columns", "periods", "controls", "folds", "learners", "models"):
        if key not in cfg:
            raise ValueError(f"config.yaml missing {key}")
    missing = set(REQUIRED_COLUMNS) - set(cfg["columns"])
    if missing:
        raise ValueError(f"Column aliases missing: {sorted(missing)}")
    return cfg


def read_model_ready_data(input_path: Path, cfg: dict) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Private input not found: {input_path}. "
            "Prepare your own confidential model-ready CSV; do not commit it."
        )
    df = pd.read_csv(input_path)
    mapping = cfg["columns"]
    missing = [mapping[k] for k in REQUIRED_COLUMNS if mapping[k] not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    renamed = df[[mapping[k] for k in REQUIRED_COLUMNS]].copy()
    renamed.columns = REQUIRED_COLUMNS
    if renamed["partner"].isna().any():
        raise ValueError("Missing partner IDs")
    if renamed.duplicated(["partner", "year"]).any():
        dup = int(renamed.duplicated(["partner", "year"]).sum())
        raise ValueError(f"Found {dup} duplicate partner-year records")
    for c in REQUIRED_COLUMNS:
        if c != "partner":
            renamed[c] = pd.to_numeric(renamed[c], errors="coerce")
            if not np.isfinite(renamed[c].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Non-finite values in {c}. Verify log(0), missing values, "
                    "and the exact preprocessing applied to the original analysis."
                )
    renamed["year"] = renamed["year"].astype(int)
    limits = [(int(a), int(b)) for a, b in cfg["periods"].values()]
    valid_years = set().union(*(set(range(a, b + 1)) for a, b in limits))
    if not set(renamed["year"]).issubset(valid_years):
        raise ValueError("Input includes years outside configured pre/post sample")
    renamed["partner"] = renamed["partner"].astype(str)
    return renamed


def design_matrices(df: pd.DataFrame, cfg: dict, outcome: str, air: str, model: str):
    """Construct precisely aligned Y, T, Z, W and partner groups.

    Baseline: Y = theta * ln(air) + g(W) + error; IV is ln(air_iv).
    Interaction: T = [ln(air), ln(internet), ln(air)*ln(internet)],
      Z = [ln(air_iv), ln(internet_iv), ln(air_iv)*ln(internet_iv)].

    The product is formed AFTER log-transforming its components, as in Eq. (10)
    of the current model specification. It is NOT log(air_iv*internet_iv).
    """
    y = df[outcome].to_numpy(dtype=float)
    ta = df[air].to_numpy(dtype=float)
    zi = df[f"{air}_iv"].to_numpy(dtype=float)
    if model == "baseline":
        t = ta
        z = zi
        t_names = [f"ln_{air}"]
    elif model == "interaction":
        internet = df["internet"].to_numpy(dtype=float)
        ziv = df["internet_iv"].to_numpy(dtype=float)
        t = np.column_stack((ta, internet, ta * internet))
        z = np.column_stack((zi, ziv, zi * ziv))
        t_names = [f"ln_{air}", "ln_internet", f"ln_{air}__x__ln_internet"]
    else:
        raise ValueError(f"Unknown model: {model}")
    controls = df[cfg["controls"]].astype(float).reset_index(drop=True)
    ids = pd.DataFrame({
        "partner": df["partner"].astype(str).to_numpy(),
        "year": df["year"].astype(str).to_numpy(),
    })
    dummies = pd.get_dummies(ids, columns=["partner", "year"], dtype=float)
    w = pd.concat((controls, dummies), axis=1).to_numpy(dtype=float)
    groups = df["partner"].to_numpy()
    if w.shape[1] == 0:
        raise ValueError("No controls found")
    return y, t, z, w, groups, t_names


def make_regressor(learner: str, cfg: dict, seed: int):
    """Base regressor for nuisance functions, identical across Y and T models."""
    if learner == "rf":
        return RandomForestRegressor(random_state=seed, **cfg["rf"])
    if learner == "lasso":
        return make_pipeline(
            StandardScaler(),
            LassoCV(random_state=seed, **cfg["lasso"]),
        )
    if learner == "gbr":
        return GradientBoostingRegressor(random_state=seed, **cfg["gbr"])
    if learner == "ensemble":
        return VotingRegressor([
            ("rf", make_regressor("rf", cfg, seed)),
            ("lasso", make_regressor("lasso", cfg, seed)),
            ("gbr", make_regressor("gbr", cfg, seed)),
        ])
    raise ValueError(f"Unsupported learner: {learner}")


def make_nuisance_regressor(learner: str, cfg: dict, seed: int, n_targets: int):
    base = make_regressor(learner, cfg, seed)
    return MultiOutputRegressor(base) if n_targets > 1 else base


def fold_diagnostics(groups: np.ndarray, cfg: dict, seed: int, repeat_index: int) -> dict:
    cv = (GroupKFold(n_splits=int(cfg["folds"])) if repeat_index == 0
          else ShuffledGroupKFold(n_splits=int(cfg["folds"]), random_state=seed))
    splits = list(cv.split(np.ones((len(groups), 1)), groups=groups))
    for train, test in splits:
        if set(groups[train]).intersection(set(groups[test])):
            raise AssertionError("Partner group appears in both training/test")
    test_counts = [int(len(test)) for _, test in splits]
    return {"cv": cv, "n_folds": len(splits), "test_n": test_counts}


def fit_econml_dmliv(y, t, z, w, groups, cfg: dict, learner: str,
                     seed: int, repeat_index: int) -> np.ndarray:
    """Fit the same EconML DMLIV estimator with a chosen nuisance learner.

    Returns treatment-effect coefficients only. No clustered/robust SE claims.
    """
    try:
        from econml.iv.dml import DMLIV
    except ImportError as e:
        raise ImportError(
            "EconML is required for estimation. Install requirements.txt. "
            "Use --validate-only to inspect the data/CV setup without EconML."
        ) from e
    cv = fold_diagnostics(groups, cfg, seed, repeat_index)["cv"]
    n_t = t.shape[1] if t.ndim == 2 else 1
    est = DMLIV(
        model_y_xw=make_nuisance_regressor(learner, cfg, seed, 1),
        model_t_xw=make_nuisance_regressor(learner, cfg, seed, n_t),
        model_t_xwz=make_nuisance_regressor(learner, cfg, seed, n_t),
        cv=cv,
        random_state=seed,
        fit_cate_intercept=True,
    )
    # W includes partner/year indicators and economic controls; X=None makes
    # theta constant, matching a PLIV coefficient rather than a varying CATE.
    # inference=None is intentional: learner sensitivity, not new SE claims.
    est.fit(Y=y, T=t, Z=z, X=None, W=w, groups=groups, inference=None)
    effect = np.asarray(est.const_marginal_effect(X=None), dtype=float).reshape(-1)
    if effect.size != n_t:
        # intercept_ is the constant coefficient of the CATE model with X=None.
        effect = np.asarray(est.intercept_, dtype=float).reshape(-1)
    if effect.size != n_t or not np.isfinite(effect).all():
        raise RuntimeError(f"Unexpected EconML coefficient shape / values: {effect}")
    return effect


def run(cfg: dict, input_path: Path, output_dir: Path, validate_only: bool = False,
        only_outcome: str | None = None, only_model: str | None = None,
        only_learner: str | None = None):
    df = read_model_ready_data(input_path, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_outcomes = [only_outcome] if only_outcome else cfg["outcomes"]
    selected_models = [only_model] if only_model else cfg["models"]
    selected_learners = [only_learner] if only_learner else cfg["learners"]
    raw, samples = [], []
    for period, bounds in cfg["periods"].items():
        lo, hi = map(int, bounds)
        d = df.loc[df["year"].between(lo, hi)].reset_index(drop=True)
        if len(d) == 0:
            raise ValueError(f"No observations in {period}: {lo}-{hi}")
        n_groups = int(d["partner"].nunique())
        if n_groups < int(cfg["folds"]):
            raise ValueError(f"Need >= {cfg['folds']} countries for {period}; got {n_groups}")
        samples.append({"period": period, "start_year": lo, "end_year": hi,
                        "observations": len(d), "partner_countries": n_groups})
        for air in cfg["connectivity_measures"]:
            for outcome in selected_outcomes:
                for model in selected_models:
                    y, t, z, w, groups, names = design_matrices(d, cfg, outcome, air, model)
                    fd = fold_diagnostics(groups, cfg, int(cfg["seed"]), 0)
                    if validate_only:
                        print(f"OK {period:5} {air:13} {outcome:20} {model:11} "
                              f"N={len(y)}, partners={n_groups}, "
                              f"Y={y.shape}, T={t.shape}, Z={z.shape}, W={w.shape}; "
                              f"fold test sizes={fd['test_n']}")
                        continue
                    for learner in selected_learners:
                        for r in range(int(cfg["split_repetitions"])):
                            seed = int(cfg["seed"]) + r
                            eff = fit_econml_dmliv(y, t, z, w, groups, cfg, learner, seed, r)
                            for treatment, coef in zip(names, eff):
                                entry = {"period": period, "outcome": outcome,
                                         "connectivity": air, "model": model,
                                         "learner": learner, "split_repeat": r + 1,
                                         "treatment": treatment, "estimate": float(coef),
                                         "observations": len(d), "partner_countries": n_groups,
                                         "n_folds": int(cfg["folds"])}
                                raw.append(entry)
                                print(json.dumps(entry))
    pd.DataFrame(samples).to_csv(output_dir / "sample_sizes.csv", index=False)
    if validate_only:
        print("Validation succeeded; NO estimates were computed.")
        return
    if not raw:
        raise RuntimeError("No estimates computed")
    frame = pd.DataFrame(raw)
    frame.to_csv(output_dir / "learner_estimates_by_split.csv", index=False)
    group_cols = ["period", "outcome", "connectivity", "model", "learner", "treatment"]
    aggregate = frame.groupby(group_cols, as_index=False).agg(
        median_estimate=("estimate", "median"),
        min_estimate=("estimate", "min"),
        max_estimate=("estimate", "max"),
        sd_across_splits=("estimate", "std"),
        split_repetitions=("estimate", "size"),
    )
    aggregate["sd_across_splits"] = aggregate["sd_across_splits"].fillna(0.0)
    aggregate.to_csv(output_dir / "median_estimates_by_learner.csv", index=False)
    pivot = aggregate.pivot_table(
        index=["period", "outcome", "connectivity", "model", "treatment"],
        columns="learner", values="median_estimate",
    ).reset_index()
    pivot.columns.name = None
    pivot.to_csv(output_dir / "comparison_by_learner.csv", index=False)
    manifest = {"note": "SUPPLEMENTARY learner-sensitivity runs; results are not the main paper.",
                "data_included_in_repo": False,
                "random_seed": int(cfg["seed"]), "folds": int(cfg["folds"]),
                "split_repetitions": int(cfg["split_repetitions"]),
                "cross_fitting": "GroupKFold; randomized group folds on repeated runs",
                "inference": "Point estimates only; no clustered or robust standard-error claim",
                "treatment_interaction": "ln(air)*ln(internet)",
                "instrument_interaction": "ln(airIV)*ln(internetIV)",
                "original_rf_parameters_verified": False}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(frame)} estimates and comparisons to {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", help="Override input_csv in config.yaml")
    p.add_argument("--output", help="Override output_dir in config.yaml")
    p.add_argument("--validate-only", action="store_true", help="Validate input + grouped folds without fitting")
    p.add_argument("--outcome", choices=["commercial_export", "commercial_import", "transport_export", "transport_import", "travel_export", "travel_import"])
    p.add_argument("--model", choices=["baseline", "interaction"])
    p.add_argument("--learner", choices=["rf", "lasso", "gbr", "ensemble"])
    args = p.parse_args()
    cp = Path(args.config).resolve()
    cfg = load_config(cp)
    root = cp.parent
    inp = Path(args.input).resolve() if args.input else (root / cfg["input_csv"]).resolve()
    out = Path(args.output).resolve() if args.output else (root / cfg["output_dir"]).resolve()
    run(cfg, inp, out, args.validate_only, args.outcome, args.model, args.learner)


if __name__ == "__main__":
    main()
