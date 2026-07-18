"""
pipeline_viz.py — summary PNG visualizations for the MARIO SLURM pipeline.

Ports the *summary* plots from the explore_stageX.ipynb debug notebooks so the
sbatch pipeline saves them to disk (the notebooks only render inline). Each
function reads a stage's already-written on-disk outputs and writes a PNG into a
`viz/` subdirectory of that stage's output dir.

Design rules (so plotting never breaks the pipeline):
  * Headless 'Agg' backend — safe on SLURM compute nodes with no display.
  * Every function is fully self-contained and swallows its own errors, returning
    the PNG path on success or None on failure. Callers should still wrap the
    call in try/except as belt-and-suspenders; a failed plot must never stop a
    stage from writing its .done file.

Per-TIRP / per-entity forecast-timeline plots (Stage 1 raw-signal example is a
single sampled entity; Stage 3/4 timelines) are intentionally NOT produced here
— only aggregate summary plots, by design decision.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless: no display on SLURM nodes
import matplotlib.pyplot as plt
import pandas as pd


def _viz_dir(base_dir):
    """Create and return `<base_dir>/viz`."""
    d = os.path.join(base_dir, "viz")
    os.makedirs(d, exist_ok=True)
    return d


def save_stage1_abstraction_example(abstraction_output_dir, d_method="", num_of_bins=""):
    """
    Stage 1 summary: one example entity/variable's raw signal colored by its
    discretized StateID, with the learned bin edges as dashed lines.

    Reads Train/symbolic_time_series.csv (raw values + StateID) and Train/states.csv
    (BinLow/BinHigh edges). Writes <abstraction_output_dir>/viz/abstraction_example.png.
    """
    try:
        train_dir = os.path.join(abstraction_output_dir, "Train")
        sym_path = os.path.join(train_dir, "symbolic_time_series.csv")
        states_path = os.path.join(train_dir, "states.csv")
        if not os.path.exists(sym_path):
            print(f"[viz] Stage 1: {sym_path} missing, skipping abstraction example plot.")
            return None

        train_sym = pd.read_csv(sym_path)
        if train_sym.empty:
            print("[viz] Stage 1: symbolic_time_series.csv empty, skipping plot.")
            return None
        train_states = pd.read_csv(states_path) if os.path.exists(states_path) else pd.DataFrame()

        VAR = train_sym["TemporalPropertyID"].unique()[0]
        ent = train_sym[train_sym["TemporalPropertyID"] == VAR]["EntityID"].iloc[0]
        series = train_sym[(train_sym["EntityID"] == ent) &
                           (train_sym["TemporalPropertyID"] == VAR)].sort_values("TimeStamp")

        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(series["TimeStamp"], series["TemporalPropertyValue"],
                        c=series["StateID"], cmap="viridis", s=40, zorder=3)
        ax.plot(series["TimeStamp"], series["TemporalPropertyValue"], color="lightgray", zorder=1)

        if not train_states.empty and "TemporalPropertyID" in train_states.columns:
            for _, r in train_states[train_states["TemporalPropertyID"] == VAR].iterrows():
                for edge in (r.get("BinLow"), r.get("BinHigh")):
                    if pd.notna(edge) and abs(edge) != float("inf"):
                        ax.axhline(edge, color="crimson", ls="--", lw=0.7, alpha=0.6)

        ax.set_title(f"Entity {ent}, variable {VAR}: raw signal colored by discretized StateID\n"
                     f"(dashed red = learned bin edges, method={d_method}, bins={num_of_bins})")
        ax.set_xlabel("TimeStamp"); ax.set_ylabel("TemporalPropertyValue")
        fig.colorbar(sc, ax=ax, label="StateID (symbol)")
        fig.tight_layout()

        out_path = os.path.join(_viz_dir(abstraction_output_dir), "abstraction_example.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[viz] Stage 1 abstraction example saved: {out_path}")
        return out_path
    except Exception as e:
        print(f"[viz] Stage 1 plot failed (non-fatal): {e}")
        plt.close("all")
        return None


def save_stage2_tirp_distributions(mining_run_dir, mvs=None, vs_min=0.0, vs_max=1.0):
    """
    Stage 2 summary: TIRP size distribution + vertical-support-fraction histogram
    (with mvs floor and the vs_min/vs_max keep-range marked).

    Reads <mining_run_dir>/tirp_selection_scores.csv. Writes
    <mining_run_dir>/viz/tirp_distributions.png.
    """
    try:
        scores_path = os.path.join(mining_run_dir, "tirp_selection_scores.csv")
        if not os.path.exists(scores_path):
            print(f"[viz] Stage 2: {scores_path} missing, skipping distribution plot.")
            return None
        scores = pd.read_csv(scores_path)
        if scores.empty or "Size" not in scores.columns:
            print("[viz] Stage 2: scores empty / missing columns, skipping plot.")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        size_counts = scores["Size"].value_counts().sort_index()
        axes[0].bar(size_counts.index.astype(str), size_counts.values, color="steelblue")
        axes[0].set_title("TIRP size distribution")
        axes[0].set_xlabel("Size (number of symbols)"); axes[0].set_ylabel("count")

        if "Vertical_Support_Fraction" in scores.columns:
            axes[1].hist(scores["Vertical_Support_Fraction"], bins=30, color="seagreen")
            if mvs is not None:
                axes[1].axvline(mvs, color="crimson", ls="--", label=f"mvs floor = {mvs}")
            if vs_min is not None and vs_min > 0:
                axes[1].axvline(vs_min, color="navy", ls=":", label=f"vs_min = {vs_min}")
            if vs_max is not None and vs_max < 1:
                axes[1].axvline(vs_max, color="navy", ls=":", label=f"vs_max = {vs_max}")
            axes[1].set_title("Vertical support fraction distribution")
            axes[1].set_xlabel("Vertical support (fraction of entities)"); axes[1].set_ylabel("count")
            axes[1].legend()

        fig.tight_layout()
        out_path = os.path.join(_viz_dir(mining_run_dir), "tirp_distributions.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[viz] Stage 2 TIRP distributions saved: {out_path}")
        return out_path
    except Exception as e:
        print(f"[viz] Stage 2 plot failed (non-fatal): {e}")
        plt.close("all")
        return None


def save_tirp_train_accuracy_barplot(feature_matrix_dir, out_dir=None, title_suffix="",
                                     also_save_csv=True):
    """
    Stage 3.5 summary: one horizontal bar per TIRP model showing its TRAIN-set
    forecast accuracy, ordered by accuracy top (best) -> bottom (worst). Bars are
    colored by whether the model beats its majority-class baseline, and each TIRP's
    baseline is marked so "beats baseline" is visible at a glance.

    Scans <feature_matrix_dir>/tirp_*/train_summary_metrics.csv (written by
    run_stage3_5_validation.py). Writes <out_dir or feature_matrix_dir>/viz/
    tirp_train_accuracy.png (+ tirp_train_accuracy_per_tirp.csv when also_save_csv).

    Returns the PNG path, or None if no train metrics exist (e.g. Stage 3.5 was not
    run) or the plot fails — never raises.
    """
    try:
        import glob as _glob
        rows = []
        for d in sorted(_glob.glob(os.path.join(feature_matrix_dir, "tirp_*"))):
            smf = os.path.join(d, "train_summary_metrics.csv")
            if os.path.exists(smf):
                try:
                    rows.append(pd.read_csv(smf).iloc[0].to_dict())
                except Exception:
                    continue
        if not rows:
            print(f"[viz] Stage 3.5: no train_summary_metrics.csv under {feature_matrix_dir}; "
                  f"skipping per-TIRP train-accuracy barplot (was Stage 3.5 run?).")
            return None

        df = pd.DataFrame(rows)
        if "train_accuracy" not in df.columns:
            print("[viz] Stage 3.5: 'train_accuracy' column missing, skipping barplot.")
            return None
        # Drop TIRPs with no labelled rows (accuracy NaN) and sort best -> worst.
        df = df[pd.to_numeric(df["train_accuracy"], errors="coerce").notna()].copy()
        if df.empty:
            print("[viz] Stage 3.5: all TIRPs had NaN train accuracy, skipping barplot.")
            return None
        df["train_accuracy"] = df["train_accuracy"].astype(float)
        has_base = "majority_baseline_acc" in df.columns
        if has_base:
            df["majority_baseline_acc"] = pd.to_numeric(df["majority_baseline_acc"], errors="coerce")
            df["beats_baseline"] = df["train_accuracy"] > df["majority_baseline_acc"]
        df = df.sort_values("train_accuracy", ascending=False).reset_index(drop=True)

        out_base = out_dir if out_dir else feature_matrix_dir
        if also_save_csv:
            keep = [c for c in ["TIRP_name", "train_accuracy", "majority_baseline_acc",
                                "beats_baseline", "n_rows", "n_classes"] if c in df.columns]
            df[keep].to_csv(os.path.join(_viz_dir(out_base), "tirp_train_accuracy_per_tirp.csv"),
                            index=False)

        n = len(df)
        labels = df["TIRP_name"].astype(str).tolist() if "TIRP_name" in df.columns else \
                 [str(i) for i in range(n)]
        # Green when the model beats its baseline, muted grey otherwise.
        C_WIN, C_LOSE = "#2a8f5a", "#b3b1a8"
        colors = ([C_WIN if b else C_LOSE for b in df["beats_baseline"]]
                  if has_base else [C_WIN] * n)

        fig_h = max(3.0, min(0.32 * n + 1.0, 40.0))  # scale height with TIRP count, capped
        fig, ax = plt.subplots(figsize=(9, fig_h))
        y = range(n)
        ax.barh(list(y), df["train_accuracy"].values, color=colors, zorder=2)
        if has_base:
            # Each TIRP's own majority baseline as a short marker on its row.
            ax.scatter(df["majority_baseline_acc"].values, list(y), marker="|",
                       s=120, color="#0b0b0b", lw=1.2, zorder=3, label="majority baseline")
        ax.set_yticks(list(y))
        show_labels = n <= 60  # per-TIRP labels get unreadable past ~60 bars
        if show_labels:
            ax.set_yticklabels(labels, fontsize=max(5, min(9, int(600 / max(n, 1)))))
        else:
            ax.set_yticklabels([])
            ax.set_ylabel(f"{n} TIRP models (labels hidden; see per-TIRP CSV)")
        ax.invert_yaxis()  # best accuracy at the TOP
        ax.set_xlim(0, 1)
        ax.set_xlabel("train-set forecast accuracy")
        ax.grid(axis="x", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        title = f"Per-TIRP train-set forecast accuracy ({n} models, best at top)"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, loc="left", fontweight="bold")
        if has_base:
            from matplotlib.patches import Patch
            handles = [Patch(color=C_WIN, label="beats baseline"),
                       Patch(color=C_LOSE, label="below baseline")]
            ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
        fig.tight_layout()

        out_path = os.path.join(_viz_dir(out_base), "tirp_train_accuracy.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[viz] Stage 3.5 per-TIRP train-accuracy barplot saved: {out_path} ({n} TIRPs)")
        return out_path
    except Exception as e:
        print(f"[viz] Stage 3.5 barplot failed (non-fatal): {e}")
        plt.close("all")
        return None


def save_stage5_accuracy_vs_baseline(results_output_dir, aggregation_method="", context_window=""):
    """
    Stage 5 summary: this aggregation combo's MARIO accuracy vs the majority
    baseline, per scope (overall / seen_future / new_entity).

    Reads <results_output_dir>/stage5_metrics.csv. Writes
    <results_output_dir>/viz/accuracy_vs_baseline.png.
    """
    try:
        metrics_path = os.path.join(results_output_dir, "stage5_metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"[viz] Stage 5: {metrics_path} missing, skipping accuracy plot.")
            return None
        m = pd.read_csv(metrics_path)
        if m.empty or "scope" not in m.columns:
            print("[viz] Stage 5: metrics empty / missing columns, skipping plot.")
            return None

        scope_order = ["overall", "seen_future", "new_entity"]
        m = m.set_index("scope")
        scopes = [s for s in scope_order if s in m.index]
        if not scopes:
            print("[viz] Stage 5: no known scopes present, skipping plot.")
            return None

        import numpy as np
        # Light-surface palette from explore_stage5.ipynb (blue = MARIO, grey = baseline).
        C_MARIO, C_BASE, GRID, INK2 = "#2a78d6", "#b3b1a8", "#e1e0d9", "#52514e"
        x = np.arange(len(scopes)); w = 0.38

        fig, ax = plt.subplots(figsize=(7, 4.2))
        mario = [float(m.loc[s, "accuracy"]) for s in scopes]
        base = [float(m.loc[s, "majority_baseline_acc"]) for s in scopes]
        groups = [ax.bar(x - w / 2, mario, w, color=C_MARIO, label="MARIO (aggregated)"),
                  ax.bar(x + w / 2, base, w, color=C_BASE, label="majority baseline")]
        for group in groups:
            for r in group:
                ax.annotate(f"{r.get_height():.2f}",
                            (r.get_x() + r.get_width() / 2, r.get_height()),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(scopes)
        ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
        ax.set_title(f"Aggregated MARIO vs majority baseline "
                     f"(method={aggregation_method}, context_window={context_window})",
                     loc="left", fontweight="bold")
        fig.tight_layout()

        out_path = os.path.join(_viz_dir(results_output_dir), "accuracy_vs_baseline.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[viz] Stage 5 accuracy-vs-baseline saved: {out_path}")
        return out_path
    except Exception as e:
        print(f"[viz] Stage 5 plot failed (non-fatal): {e}")
        plt.close("all")
        return None
