"""
src/models/monte_carlo.py

Monte Carlo price path simulation layered on top of XGBoost point forecasts.

Method
------
1. Load out-of-sample forecasts from walk-forward evaluation.
2. Compute residuals = actual − forecast over all folds except the last.
   (The last fold is reserved as the simulation target so residual
   distribution fitting never sees the test data.)
3. Test residuals for normality (Jarque-Bera; Shapiro-Wilk is used only
   on small samples because it doesn't scale beyond ~5,000 obs).
4. If normal → sample from N(μ, σ²) of fitted residuals.
   If not normal (the typical case for power prices) → empirical bootstrap
   (np.random.choice with replacement from historical residuals).
5. Optionally model AR(1) residual autocorrelation for more realistic paths.
6. For each hour in the forecast horizon, simulate 1,000 price paths and
   collapse to p5/p25/p50/p75/p95 percentile bands.
7. Save results to data/results/monte_carlo_dk1.parquet.

Usage::

    python -m src.models.monte_carlo           # DK1 (default)
    python -m src.models.monte_carlo --zone DK2
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"

# Normality test significance level
NORMALITY_ALPHA = 0.05

# AR(1) autocorrelation threshold — if |ρ| exceeds this, apply AR(1) structure
AR1_THRESHOLD = 0.1

# Above this sample size Shapiro-Wilk loses power; switch to Jarque-Bera
SHAPIRO_MAX_SAMPLES = 5_000

# Monte Carlo simulation defaults
MC_N_SIMULATIONS = 1_000
MC_RANDOM_SEED   = 42


# ---------------------------------------------------------------------------
# Normality testing
# ---------------------------------------------------------------------------

def test_normality(residuals: np.ndarray) -> tuple[bool, str]:
    """
    Test whether residuals are normally distributed.

    Uses Jarque-Bera for samples larger than 5,000 (Shapiro-Wilk loses power
    and becomes very slow beyond that size); falls back to Shapiro-Wilk for
    smaller samples.

    Parameters
    ----------
    residuals : np.ndarray
        Array of model residuals (actual − forecast).

    Returns
    -------
    is_normal : bool
        True if the null hypothesis of normality cannot be rejected at
        ``NORMALITY_ALPHA``.
    description : str
        Human-readable test result for logging.
    """
    n = len(residuals)

    if n > SHAPIRO_MAX_SAMPLES:
        stat, p_value = stats.jarque_bera(residuals)
        test_name = "Jarque-Bera"
    else:
        stat, p_value = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"

    is_normal = p_value > NORMALITY_ALPHA
    verdict = "NORMAL" if is_normal else "NOT NORMAL"
    description = (
        f"{test_name}: stat={stat:.4f}, p={p_value:.4e} → {verdict} "
        f"(α={NORMALITY_ALPHA}, n={n})"
    )
    return is_normal, description


# ---------------------------------------------------------------------------
# AR(1) autocorrelation estimation
# ---------------------------------------------------------------------------

def estimate_ar1(residuals: np.ndarray) -> float:
    """
    Estimate the AR(1) autocorrelation coefficient ρ from residuals.

    Fits residuals[t] = ρ * residuals[t-1] + ε via OLS (lag-1 regression).

    Parameters
    ----------
    residuals : np.ndarray

    Returns
    -------
    float
        Estimated ρ.  Close to 0 → no useful autocorrelation structure.
    """
    y = residuals[1:]
    x = residuals[:-1]
    # OLS: ρ = cov(x, y) / var(x)
    rho = float(np.cov(x, y)[0, 1] / np.var(x))
    return np.clip(rho, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_price_paths(
    forecasts: np.ndarray,
    residuals: np.ndarray,
    n_simulations: int = 1_000,
    seed: int = 42,
    use_ar1: bool = True,
) -> pd.DataFrame:
    """
    Simulate Monte Carlo price paths by adding bootstrapped residuals to
    XGBoost point forecasts.

    For each hour h in the forecast horizon:
      - Draw ``n_simulations`` residuals from the fitted distribution.
      - If AR(1) structure is detected (|ρ| > ``AR1_THRESHOLD``), propagate
        the previous draw's innovation through the AR(1) filter so
        consecutive hours have correlated noise.
      - Add the drawn residual to the point forecast.

    Parameters
    ----------
    forecasts : np.ndarray
        Array of XGBoost point forecasts, shape (H,), one per hour.
    residuals : np.ndarray
        Historical residuals (actual − forecast) used to fit the noise
        distribution.  Should NOT include the hours being forecast.
    n_simulations : int
        Number of Monte Carlo paths.  Default: 1,000.
    seed : int
        Random seed for reproducibility.  Default: 42.
    use_ar1 : bool
        Whether to apply AR(1) autocorrelation structure to the noise draws.
        Default: True.

    Returns
    -------
    pd.DataFrame
        One row per forecast hour with columns:
        hour, p5, p25, p50, p75, p95.
        ``hour`` is 0-indexed (0 = first forecast hour).
    """
    rng = np.random.default_rng(seed)
    H   = len(forecasts)

    clean_residuals = residuals[np.isfinite(residuals)]
    resid_mean = float(np.mean(clean_residuals))
    resid_std  = float(np.std(clean_residuals, ddof=1))

    # Normality test — determines sampling strategy
    is_normal, norm_desc = test_normality(clean_residuals)
    log.info("Normality test on %d residuals: %s", len(clean_residuals), norm_desc)
    log.info("Residual bias (mean): %.3f EUR/MWh | std: %.3f EUR/MWh", resid_mean, resid_std)

    # Center residuals so the AR(1) filter operates around zero.
    # The mean bias is added back per-step so it doesn't compound through AR(1).
    centered_residuals = clean_residuals - resid_mean

    if is_normal:
        log.info("Using parametric Normal(0, σ) distribution for centered residual sampling.")
        def draw_innovations(n: int) -> np.ndarray:
            return rng.normal(0.0, resid_std, size=n)
    else:
        log.info("Using empirical bootstrap for residual sampling (non-normal residuals).")
        def draw_innovations(n: int) -> np.ndarray:
            return rng.choice(centered_residuals, size=n, replace=True)

    # AR(1) autocorrelation — estimated on centered residuals
    ar1_rho = 0.0
    if use_ar1:
        ar1_rho = estimate_ar1(centered_residuals)
        if abs(ar1_rho) > AR1_THRESHOLD:
            log.info("AR(1) autocorrelation detected: ρ=%.4f — applying to simulated paths.", ar1_rho)
        else:
            log.info("AR(1) autocorrelation negligible: ρ=%.4f — using i.i.d. draws.", ar1_rho)
            ar1_rho = 0.0

    # Simulation matrix: shape (H, n_simulations)
    paths = np.empty((H, n_simulations), dtype=np.float64)

    # Initialise AR(1) state at zero (centered)
    prev_noise = np.zeros(n_simulations)

    for h in range(H):
        raw_draws = draw_innovations(n_simulations)

        if ar1_rho != 0.0:
            # AR(1) on centered noise: noise[h] = ρ * noise[h-1] + ε[h] * sqrt(1 - ρ²)
            # This preserves unit variance when ε ~ N(0,1); for empirical draws the
            # scaling is approximate but prevents compounding drift.
            innovation = raw_draws * np.sqrt(max(0.0, 1.0 - ar1_rho ** 2))
            noise = ar1_rho * prev_noise + innovation
        else:
            noise = raw_draws

        # Add mean bias back to each path — bias is added once per step, not compounded
        paths[h] = forecasts[h] + resid_mean + noise
        prev_noise = noise

    # Collapse to percentile bands
    percentiles = [5, 25, 50, 75, 95]
    pct_values  = np.percentile(paths, percentiles, axis=1)  # shape (5, H)

    result = pd.DataFrame({
        "hour": np.arange(H),
        "p5":   pct_values[0],
        "p25":  pct_values[1],
        "p50":  pct_values[2],
        "p75":  pct_values[3],
        "p95":  pct_values[4],
    })
    return result


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_monte_carlo(zone: str = "DK1") -> pd.DataFrame:
    """
    Full Monte Carlo pipeline for one bidding zone.

    Loads walk-forward forecasts, fits the residual distribution on all folds
    except the last, runs simulation on the last fold, and saves results.

    Parameters
    ----------
    zone : str
        ``"DK1"`` or ``"DK2"``.

    Returns
    -------
    pd.DataFrame
        Monte Carlo percentile bands for the last test fold.
    """
    forecasts_path = RESULTS_DIR / f"forecasts_{zone.lower()}.parquet"
    if not forecasts_path.exists():
        raise FileNotFoundError(
            f"Forecasts not found: {forecasts_path}. Run forecaster.py first."
        )

    forecasts_df = pd.read_parquet(forecasts_path)
    log.info("[%s] Loaded %d forecast rows across folds: %s",
             zone, len(forecasts_df), sorted(forecasts_df["fold_id"].unique()))

    # Split: all-but-last folds → residual fitting; last fold → simulation
    all_folds  = sorted(forecasts_df["fold_id"].unique())
    last_fold  = all_folds[-1]
    train_folds = [f for f in all_folds if f != last_fold]

    if not train_folds:
        log.warning(
            "[%s] Only one fold available — using it for both fitting and simulation. "
            "Residual distribution fit and evaluation overlap.", zone
        )
        residual_df = forecasts_df
    else:
        residual_df = forecasts_df[forecasts_df["fold_id"].isin(train_folds)]

    test_df = forecasts_df[forecasts_df["fold_id"] == last_fold].copy()
    test_df = test_df.sort_values("timestamp_utc").reset_index(drop=True)

    # Compute residuals on the training folds
    residuals = (residual_df["actual"] - residual_df["forecast"]).values
    residuals = residuals[np.isfinite(residuals)]
    log.info("[%s] Residual distribution: mean=%.3f, std=%.3f, n=%d",
             zone, residuals.mean(), residuals.std(), len(residuals))

    # Summary stats for interview reference
    log.info(
        "[%s] Residual percentiles — p5=%.1f, p25=%.1f, p50=%.1f, p75=%.1f, p95=%.1f",
        zone,
        np.percentile(residuals, 5), np.percentile(residuals, 25),
        np.percentile(residuals, 50), np.percentile(residuals, 75),
        np.percentile(residuals, 95),
    )

    # Simulate on last fold
    log.info(
        "[%s] Simulating on last fold (fold %d): %s → %s (%d hours)",
        zone, last_fold,
        test_df["timestamp_utc"].min(),
        test_df["timestamp_utc"].max(),
        len(test_df),
    )

    mc_df = simulate_price_paths(
        forecasts=test_df["forecast"].values,
        residuals=residuals,
        n_simulations=MC_N_SIMULATIONS,
        seed=MC_RANDOM_SEED,
        use_ar1=True,
    )

    # Attach timestamps and actual prices for easy plotting
    mc_df.insert(0, "timestamp_utc", test_df["timestamp_utc"].values)
    mc_df["forecast"] = test_df["forecast"].values
    mc_df["actual"]   = test_df["actual"].values
    mc_df["fold_id"]  = last_fold

    # Save
    out_path = RESULTS_DIR / f"monte_carlo_{zone.lower()}.parquet"
    mc_df.to_parquet(out_path, index=False)
    log.info("[%s] Monte Carlo results saved to %s (%d rows)", zone, out_path, len(mc_df))

    # Print summary
    coverage_90 = (
        (test_df["actual"].values >= mc_df["p5"].values) &
        (test_df["actual"].values <= mc_df["p95"].values)
    ).mean() * 100
    coverage_50 = (
        (test_df["actual"].values >= mc_df["p25"].values) &
        (test_df["actual"].values <= mc_df["p75"].values)
    ).mean() * 100

    log.info(
        "[%s] Monte Carlo summary | fold=%d | hours=%d | "
        "90%% coverage=%.1f%% | 50%% coverage=%.1f%% | "
        "p50_mean=%.2f EUR/MWh | band_width=%.2f EUR/MWh",
        zone, last_fold, len(mc_df),
        coverage_90, coverage_50,
        mc_df["p50"].mean(),
        (mc_df["p95"] - mc_df["p5"]).mean(),
    )

    return mc_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo price path simulation.")
    parser.add_argument(
        "--zone",
        choices=["DK1", "DK2", "both"],
        default="DK1",
        help="Bidding zone (default: DK1)",
    )
    args = parser.parse_args()

    zones = ["DK1", "DK2"] if args.zone == "both" else [args.zone]
    for z in zones:
        run_monte_carlo(z)
