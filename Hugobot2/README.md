# Temporal Abstraction Package

A Python package for temporal abstraction of time series data, providing various methods for symbolic representation of time series.

## Installation

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/yuval-haim/ta_package.git
```

## Usage

The package provides several methods for temporal abstraction:

```python
from temporal_abstraction import TemporalAbstraction
from temporal_abstraction.methods import equal_width, equal_frequency, sax, td4c

# Create a TemporalAbstraction instance
ta = TemporalAbstraction()

# Use different abstraction methods
# Equal Width
result = equal_width(data, n_bins=3)

# Equal Frequency
result = equal_frequency(data, n_bins=3)

# SAX (Symbolic Aggregate Approximation)
result = sax(data, n_bins=3)

# TD4C (Time Domain 4C)
result = td4c(data, n_bins=3)
```

## Methods Available

- Equal Width Binning
- Equal Frequency Binning
- SAX (Symbolic Aggregate Approximation)
- TD4C (Time Domain 4C)
- TID3 (univariate) and TID3_MV (multivariate) — see [ta_package/methods/README.md](ta_package/methods/README.md) for algorithm details.

## TID3 log-rank (censoring-aware) scoring (`tid3_logrank`)

### What TID3 optimizes

TID3 is a **supervised** discretization method: it chooses the cutoffs for a continuous
variable so that the resulting symbolic states are maximally informative about the class
label (case vs. control). Its signal is **duration** — how long each symbolic time interval
(STI) persists. A good cutoff set produces states whose *time-duration distributions* differ
between the two classes; those states then carry class information into downstream temporal
pattern mining. The default scorer (`max_t_stat_sum`) measures this difference with a
per-state Welch **t-test** on the raw STI durations and greedily keeps the cutoff that
maximizes the summed |t|.

### The motivation for the log-rank variant

The t-test silently assumes that **every STI duration is a complete, fully observed
measurement**. In real clinical time series that assumption is false for a meaningful
fraction of intervals: many STIs do not *end* — we simply **stop observing them**. Treating
such a truncated interval as if its observed length were its true length systematically
**under-measures the long-lasting states** — exactly the states that are often the most
clinically discriminative — and corrupts both the means and the variances the t-test relies
on.

This is precisely the **right-censoring** problem from survival analysis. The correct tool
for comparing two time-to-event distributions when some observations are censored is the
**log-rank test** (the Kaplan–Meier / survival-curve comparison). The `max_logrank_sum`
scorer reframes each STI duration as a *survival time*, marks truncated intervals as
right-censored, and scores each state by the log-rank statistic between the two classes'
duration survival curves (summing per-state statistics, mirroring the t-stat scorer). The
result is a duration-separability score that stays statistically valid when STI durations are
only **partially observed**.

### Which STI types are treated as *censored*, and what that means

Each STI duration is treated as a time-to-event observation:

- **Event (uncensored):** the STI ended in an **observed state change** — the discretized
  state actually changed at the next sample *and* that change was **not** coincident with an
  observation gap. We saw the state both begin and end, so its duration is fully known.
- **Right-censored** (true duration is unknown but *at least* the observed length):
  1. **Record-end STIs** — the last interval of an entity's record. The state was still
     active when observation stopped, so its true duration is ≥ what we measured. A
     single-observation entity (duration 1 by convention) falls here too.
  2. **Gap-truncated STIs** — the interval is cut by an observation gap larger than
     `max_gap`. We saw the state up to the gap but cannot know whether it continued through
     the unobserved window. A state change that lands **coincident with a gap** is also
     censored, because the true end happened somewhere inside the unobserved interval.

A right-censored sample contributes the information "this duration was **at least** X" rather
than "this duration was **exactly** X." Internally these are packed into scipy's
`CensoredData(uncensored=…, right=…)` and passed to `scipy.stats.logrank`.

### Directional preference and sign convention

Like the t-stat scorer, the log-rank scorer comes in three flavors — `tid3_logrank`
(two-sided), `tid3_logrank_c1longer`, and `tid3_logrank_c0longer` — favoring any difference,
or a one-sided "class 1 persists longer" / "class 0 persists longer" preference. Note the
sign convention is the **mirror** of the t-test's: scipy's log-rank statistic counts
(observed − expected) events in the *first* sample (class 1), so when class-1 STIs survive
longer the statistic is **negative** and the matching one-sided alternative is `'less'`.

### Edge cases

- A state must have **≥ 2 samples in each class** to be scored.
- If **every** STI for a state is censored in both classes, the log-rank test is undefined and
  the state is skipped (contributes 0).
- **Log-rank scoring is Phase-1 univariate only** — it is not combinable with the multivariate
  refinement (`tid3_logrank_mv` is rejected).

## TID3 multivariate scoring (`tid3_mv`)

Phase 2 of `tid3_mv` evaluates every joint combination of per-variable candidate cutoff sets (beam-search best + alternatives) and selects the combination that maximizes a cross-variable separability score. The score is computed over **size-2 TIRPs** (pairs of intervals from two variables, joined by an Allen relation). Because different cutoff combinations populate different subsets of the pattern space, the scoring rule below matters.

### Per-combo score formula

For a given combination, each observed pattern key `sym_earlier_sym_later_relation` contributes a per-pattern statistic (t-stat, KS, Wasserstein, KL divergence, Mann-Whitney, or mean-duration difference — selected via `scoring_method`). The combo's final score is the **average across scorable patterns**:

```
combo_score = sum(per-pattern stat) / num_scored
```

`num_scored` is the number of patterns that met the scorability rule for that combo (see below). The denominator is per-combo, so combos are compared on their *average* per-scorable-pattern separability, not on total signal volume.

### Top-K restriction for t-stat scoring (`mv_top_tirps`)

When `scoring_method="max_t_stat_sum"`, the `mv_top_tirps` parameter (default `100`) controls the Phase 2 per-combo score:

```
combo_score = sum(top-K |t|) / K          # when mv_top_tirps > 0  (default 100)
combo_score = sum(all |t|) / num_scored   # when mv_top_tirps == -1
```

With a positive K the numerator keeps only the K most class-separating patterns and the **denominator is fixed at K** — combos with fewer than K scorable TIRPs are effectively zero-padded. This forces all combos onto a common scale and prevents a combo of "3 very strong TIRPs" (mean |t|=10) from beating "80 moderately strong TIRPs" (mean |t|=5). Set `mv_top_tirps=-1` to disable the fixed denominator and recover the legacy "mean over all scorable patterns" behavior. Applies to Phase 2 and t-stat scoring only — other scoring methods and Phase 1 are unaffected.

### Scorability rule

A pattern is scorable only if it has **at least 2 duration samples in each class**, and non-zero variance within the class samples ([tid3.py:1309](ta_package/methods/tid3.py#L1309)). Patterns that fail this check contribute nothing to the numerator and are not counted in `num_scored`.

### Pattern ceiling is combo-invariant

The *theoretical maximum* number of distinct size-2 pattern keys depends only on `bins`, `num_relations` (3 or 7), and the number of variables `N`:

```
max_patterns = C(N, 2) × bins² × num_relations × 2   (× 2 for time ordering)
```

Different cutoff combinations produce the same ceiling; they differ in which subset of that ceiling is actually populated (some states may be empty, some symbol-pair co-occurrences may not happen) and in how many of the populated patterns pass the scorability rule.

### Known limitation: class-exclusive patterns are ignored

A pattern that appears many times in one class and never in the other (e.g., 999 instances in class 1, 0 in class 0) is highly discriminative for classification, but the `≥2-per-class` rule drops it because the duration-based statistics cannot be computed against an empty sample. This is an accepted trade-off for the stability and simplicity of the duration-based scorer; presence/absence reasoning over patterns is out of scope for the current implementation.

## Requirements

- Python >= 3.6
- numpy >= 1.19.0
- pandas >= 1.0.0
- scikit-learn >= 0.24.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
