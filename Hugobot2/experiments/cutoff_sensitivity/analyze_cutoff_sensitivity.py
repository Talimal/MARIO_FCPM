"""
Analyze cutoff stability across 5-fold CV for TID3 variants.
Each row in the CSV = one (dataset, combo, variable, cutoff_idx) with 5 fold values.

Enhanced with candidate-pool-index stability analysis:
  - For every cutoff we also know *where* in the sorted candidate pool it sat.
  - Comparing pool indices across folds tells us whether each fold picks from
    a similar region of the value space, independent of the absolute value.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("plots_tid3_kdd")
OUTPUT_DIR.mkdir(exist_ok=True)

FOLD_COLS = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
FOLD_PIDX_COLS = [f'fold_{i}_pool_idx' for i in range(1, 6)]
FOLD_PSIZE_COLS = [f'fold_{i}_pool_size' for i in range(1, 6)]


# ── Loading ──────────────────────────────────────────────────────────────

def load_and_parse(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['bins'] = df['combo'].str.extract(r'abs_b=(\d+)').astype(int)
    df['abs_method'] = df['combo'].str.extract(r'-(\w+)-ig=')
    df['ig'] = df['combo'].str.extract(r'ig=(\d+)').astype(int)

    # ── Cutoff-value statistics ──
    df['fold_mean'] = df[FOLD_COLS].mean(axis=1)
    df['fold_std'] = df[FOLD_COLS].std(axis=1)
    df['fold_median'] = df[FOLD_COLS].median(axis=1)
    df['cv'] = df['fold_std'] / df['fold_mean'].abs()
    df['cv'] = df['cv'].replace([np.inf, -np.inf], np.nan)

    # ── Pool-index statistics ──
    if all(c in df.columns for c in FOLD_PIDX_COLS):
        df['pidx_mean'] = df[FOLD_PIDX_COLS].mean(axis=1)
        df['pidx_std'] = df[FOLD_PIDX_COLS].std(axis=1)
        df['pidx_median'] = df[FOLD_PIDX_COLS].median(axis=1)
        df['pidx_cv'] = df['pidx_std'] / df['pidx_mean'].abs()
        df['pidx_cv'] = df['pidx_cv'].replace([np.inf, -np.inf], np.nan)

    # ── Normalised pool position (pool_idx / pool_size) ──
    if all(c in df.columns for c in FOLD_PIDX_COLS + FOLD_PSIZE_COLS):
        for i in range(1, 6):
            idx_col = f'fold_{i}_pool_idx'
            sz_col = f'fold_{i}_pool_size'
            df[f'fold_{i}_norm_pos'] = df[idx_col] / df[sz_col].replace(0, np.nan)
        norm_cols = [f'fold_{i}_norm_pos' for i in range(1, 6)]
        df['norm_pos_mean'] = df[norm_cols].mean(axis=1)
        df['norm_pos_std'] = df[norm_cols].std(axis=1)
        df['norm_pos_cv'] = df['norm_pos_std'] / df['norm_pos_mean'].abs()
        df['norm_pos_cv'] = df['norm_pos_cv'].replace([np.inf, -np.inf], np.nan)

    return df


# ── CV reporting ─────────────────────────────────────────────────────────

def report_cv(df: pd.DataFrame, label: str, col: str = 'cv'):
    valid = df[col].dropna()
    n = len(valid)
    if n == 0:
        print(f"  {label}: no data")
        return {}
    stats = {
        'label': label,
        'n': n,
        'median_cv': valid.median(),
        'mean_cv': valid.mean(),
        'pct_lt_05': (valid < 0.05).mean() * 100,
        'pct_lt_10': (valid < 0.10).mean() * 100,
        'pct_lt_20': (valid < 0.20).mean() * 100,
        'pct_lt_30': (valid < 0.30).mean() * 100,
    }
    print(f"  {label} (n={n}): median CV={stats['median_cv']:.3f}, "
          f"CV<0.10: {stats['pct_lt_10']:.1f}%, "
          f"CV<0.20: {stats['pct_lt_20']:.1f}%, "
          f"CV<0.30: {stats['pct_lt_30']:.1f}%")
    return stats


def fold_agreement(df: pd.DataFrame, cols: list, tolerance_pct: float = 20.0) -> dict:
    """For each row, count how many folds are within tolerance_pct% of the median."""
    tol = tolerance_pct / 100.0
    med_col = cols[0].rsplit('_', 1)[0] + '_median' if '_pool_' not in cols[0] else None

    counts = []
    for _, row in df.iterrows():
        vals = [row[c] for c in cols if pd.notna(row[c])]
        if len(vals) < 2:
            continue
        med = np.median(vals)
        if med == 0 or np.isnan(med):
            continue
        n_agree = sum(abs(v - med) / abs(med) < tol for v in vals)
        counts.append(n_agree)

    counts = np.array(counts)
    if len(counts) == 0:
        return {}
    return {
        'avg_agree': np.mean(counts),
        'pct_4plus': (counts >= 4).mean() * 100,
        'pct_3plus': (counts >= 3).mean() * 100,
    }


def random_baseline_cv(df: pd.DataFrame, cols: list, n_trials: int = 200) -> float:
    """
    For each (dataset, abs_method, TemporalPropertyID), pool all observed fold
    values across bins and cutoff_idx.  Draw 5 at random and compute CV.
    """
    np.random.seed(42)
    random_cvs = []
    groups = df.groupby(['dataset', 'abs_method', 'TemporalPropertyID'])
    for _, grp in groups:
        pool = grp[cols].values.flatten()
        pool = pool[~np.isnan(pool)]
        if len(pool) < 5:
            continue
        n_rows = len(grp)
        for _ in range(n_trials):
            for _ in range(n_rows):
                sample = np.random.choice(pool, size=5, replace=True)
                m = np.mean(sample)
                if abs(m) > 0:
                    random_cvs.append(np.std(sample, ddof=1) / abs(m))
    return np.median(random_cvs) if random_cvs else np.nan


# ── Per-method analysis ──────────────────────────────────────────────────

def _section(title: str):
    print(f"\n--- {title} ---")


def analyze_method(df: pd.DataFrame, method_name: str):
    """Run all analyses for a single abs_method."""
    header = f"  TID3 Variant: {method_name}  (n={len(df)} rows)"
    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")

    has_pidx = 'pidx_cv' in df.columns
    has_norm = 'norm_pos_cv' in df.columns

    # ── 1. Overall CV (cutoff values) ──
    _section("1  Per-row CV on cutoff VALUES (std/mean across 5 folds)")
    report_cv(df, "Overall", col='cv')
    if has_pidx:
        _section("1b Per-row CV on pool INDICES")
        report_cv(df, "Overall (pool idx)", col='pidx_cv')
    if has_norm:
        _section("1c Per-row CV on normalised pool POSITION")
        report_cv(df, "Overall (norm pos)", col='norm_pos_cv')

    # ── 2. By dataset ──
    _section("2  CV by dataset")
    for ds in sorted(df['dataset'].unique()):
        report_cv(df[df['dataset'] == ds], ds)
    if has_pidx:
        _section("2b Pool-index CV by dataset")
        for ds in sorted(df['dataset'].unique()):
            report_cv(df[df['dataset'] == ds], ds, col='pidx_cv')

    # ── 3. By bins ──
    _section("3  CV by number of bins")
    for b in sorted(df['bins'].unique()):
        report_cv(df[df['bins'] == b], f"bins={b}")
    if has_pidx:
        _section("3b Pool-index CV by bins")
        for b in sorted(df['bins'].unique()):
            report_cv(df[df['bins'] == b], f"bins={b}", col='pidx_cv')

    # ── 4. By cutoff index ──
    _section("4  CV by cutoff index (sorted order)")
    for ci in sorted(df['cutoff_idx'].unique()):
        report_cv(df[df['cutoff_idx'] == ci], f"cutoff_idx={ci}")
    if has_pidx:
        _section("4b Pool-index CV by cutoff index")
        for ci in sorted(df['cutoff_idx'].unique()):
            report_cv(df[df['cutoff_idx'] == ci], f"cutoff_idx={ci}", col='pidx_cv')

    # ── 5. Fold agreement ──
    _section("5  Fold agreement on cutoff values")
    for tol in [10, 15, 20, 25]:
        ag = fold_agreement(df, FOLD_COLS, tol)
        if ag:
            print(f"  Within {tol}%: avg {ag['avg_agree']:.1f}/5 agree, "
                  f"{ag['pct_4plus']:.1f}% have 4+/5, "
                  f"{ag['pct_3plus']:.1f}% have 3+/5")
    if has_pidx:
        _section("5b Fold agreement on pool indices")
        for tol in [10, 15, 20, 25]:
            ag = fold_agreement(df, FOLD_PIDX_COLS, tol)
            if ag:
                print(f"  Within {tol}%: avg {ag['avg_agree']:.1f}/5 agree, "
                      f"{ag['pct_4plus']:.1f}% have 4+/5, "
                      f"{ag['pct_3plus']:.1f}% have 3+/5")

    # ── 6. Random baseline ──
    _section("6  Random baseline comparison (cutoff values)")
    actual = df['cv'].dropna().median()
    random_cv = random_baseline_cv(df, FOLD_COLS)
    ratio = random_cv / actual if actual > 0 else np.nan
    print(f"  Actual median CV:  {actual:.3f}")
    print(f"  Random median CV:  {random_cv:.3f}")
    print(f"  TID3 is {ratio:.1f}x more consistent than random")

    pidx_ratio = np.nan
    if has_pidx:
        _section("6b Random baseline comparison (pool indices)")
        actual_pidx = df['pidx_cv'].dropna().median()
        random_pidx = random_baseline_cv(df, FOLD_PIDX_COLS)
        pidx_ratio = random_pidx / actual_pidx if actual_pidx > 0 else np.nan
        print(f"  Actual median CV:  {actual_pidx:.3f}")
        print(f"  Random median CV:  {random_pidx:.3f}")
        print(f"  TID3 is {pidx_ratio:.1f}x more consistent than random")

    return {
        'method': method_name,
        'n_rows': len(df),
        'median_cv': actual,
        'random_cv': random_cv,
        'ratio': ratio,
        'pidx_ratio': pidx_ratio,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    csv_path = "cutoffs_sensitivity.csv"
    print(f"Loading {csv_path}...")
    df = load_and_parse(csv_path)

    print(f"Loaded {len(df)} rows")
    print(f"Datasets    : {sorted(df['dataset'].unique())}")
    print(f"Abs methods : {sorted(df['abs_method'].unique())}")
    print(f"Bins        : {sorted(df['bins'].unique())}")
    print(f"Variables per dataset:")
    for ds in sorted(df['dataset'].unique()):
        n_vars = df[df['dataset'] == ds]['TemporalPropertyID'].nunique()
        print(f"  {ds}: {n_vars} variables")

    method_summaries = []
    for method in sorted(df['abs_method'].unique()):
        sub = df[df['abs_method'] == method]
        summary = analyze_method(sub, method)
        method_summaries.append(summary)

    summary_all = analyze_method(df, "ALL VARIANTS COMBINED")
    method_summaries.append(summary_all)

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}")
    hdr = f"  {'Method':<20s} {'N':>5s} {'Med CV':>8s} {'Rand CV':>9s} {'Ratio':>6s} {'PIdx Ratio':>11s}"
    print(hdr)
    print(f"  {'-' * len(hdr)}")
    for s in method_summaries:
        pidx_str = f"{s['pidx_ratio']:>10.1f}x" if not np.isnan(s.get('pidx_ratio', np.nan)) else "       N/A"
        print(f"  {s['method']:<20s} {s['n_rows']:>5d} {s['median_cv']:>8.3f} "
              f"{s['random_cv']:>9.3f} {s['ratio']:>5.1f}x {pidx_str}")

    # ── Export detailed per-row results ──
    out_cols = ['dataset', 'abs_method', 'bins', 'TemporalPropertyID', 'cutoff_idx']
    out_cols += FOLD_COLS + ['fold_mean', 'fold_std', 'cv']
    if 'pidx_mean' in df.columns:
        out_cols += FOLD_PIDX_COLS + ['pidx_mean', 'pidx_std', 'pidx_cv']
    if 'norm_pos_mean' in df.columns:
        norm_pos_cols = [f'fold_{i}_norm_pos' for i in range(1, 6)]
        out_cols += norm_pos_cols + ['norm_pos_mean', 'norm_pos_std', 'norm_pos_cv']

    out_cols = [c for c in out_cols if c in df.columns]
    export_path = OUTPUT_DIR / "cutoff_stability_detailed.csv"
    df[out_cols].to_csv(export_path, index=False)
    print(f"\nDetailed results saved to {export_path}")


if __name__ == "__main__":
    main()
