import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e2s.paths import ProjPaths

# This script only reads the CSV tables written by 02_validate.py - it never
# touches the zarr store, so it stays fast even on large ensembles/rollouts.
paths = ProjPaths()
output_dir = paths.ensemble_validation_path
tables_dir = paths.ensemble_validation_tables_path

COLOR_VALID = "#3A9D5D"
COLOR_INVALID = "#D64545"
COLOR_MEMBER = "#6E7B8B"
COLOR_MEAN = "#1F5C99"

UNITS = {
    "t2m": "K", "msl": "Pa", "sp": "Pa", "tcwv": "kg/m^2", "z500": "m^2/s^2",
    "kinetic_energy": "m^2/s^2 (KE proxy)",
}


def load_tables():
    paths = {
        "standalone_variable": tables_dir / "standalone_variable.csv",
        "cross_variable_consistency": tables_dir / "cross_variable_consistency.csv",
        "cross_time_consistency": tables_dir / "cross_time_consistency.csv",
        "cross_ensemble_consistency": tables_dir / "cross_ensemble_consistency.csv",
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        print(f"Error: missing table(s) {missing} in {tables_dir}/. Run 02_validate.py first.")
        sys.exit(2)
    return {name: pd.read_csv(p) for name, p in paths.items()}


# ---------------------------------------------------------------------------
# Standalone-variable & cross-variable-consistency: per-member binary heatmap
# + aggregate "fraction of members invalid" heatmap.
# ---------------------------------------------------------------------------
def plot_member_heatmap(table, row_col, member_id, title, out_path):
    sub = table[table["ensemble"] == member_id]
    pivot = sub.pivot(index=row_col, columns="lead_time", values="valid").sort_index()
    matrix = ~pivot.to_numpy(dtype=bool)  # True = invalid

    fig_h = max(4, 0.25 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    cmap = mcolors.ListedColormap([COLOR_VALID, COLOR_INVALID])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="none")

    for i, j in zip(*np.where(matrix)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor="black", linewidth=0))

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    n_cols = matrix.shape[1]
    step = max(1, n_cols // 20)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_xticklabels(pivot.columns[::step], fontsize=7)
    ax.set_xlabel("Lead time step")
    ax.set_title(title)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_VALID, label="Valid"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_INVALID, label="Invalid"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregate_heatmap(table, row_col, title, out_path):
    frac_invalid = (
        table.assign(invalid=(~table["valid"]).astype(float))
        .groupby([row_col, "lead_time"])["invalid"].mean()
        .unstack("lead_time")
        .sort_index()
    )

    fig_h = max(4, 0.25 * len(frac_invalid))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    mesh = ax.imshow(frac_invalid.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="none")

    ax.set_yticks(range(len(frac_invalid.index)))
    ax.set_yticklabels(frac_invalid.index.tolist(), fontsize=7)
    n_cols = frac_invalid.shape[1]
    step = max(1, n_cols // 20)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_xticklabels(frac_invalid.columns[::step], fontsize=7)
    ax.set_xlabel("Lead time step")
    ax.set_title(title)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.set_label("Fraction of members invalid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-time consistency: reconstruct the per-member value series from the
# transition rows (union of each row's "from" and "to" points, deduplicated
# by (ensemble, lead_time) - the value at a given lead_time is the same
# regardless of which transition row it came from). Flag BOTH endpoints of
# any invalid transition, since the jump could be caused by either side.
# ---------------------------------------------------------------------------
def plot_cross_time_check(sub, out_path):
    check_name = sub["check"].iloc[0]
    metric = sub["metric"].iloc[0]
    ylabel = f"{metric} ({UNITS.get(metric, '?')})"

    pts_from = sub[["ensemble", "lead_time_from", "lead_time_from_hours", "value_from"]].rename(
        columns={"lead_time_from": "lead_time", "lead_time_from_hours": "lead_time_hours", "value_from": "value"}
    )
    pts_to = sub[["ensemble", "lead_time_to", "lead_time_to_hours", "value_to"]].rename(
        columns={"lead_time_to": "lead_time", "lead_time_to_hours": "lead_time_hours", "value_to": "value"}
    )
    points = pd.concat([pts_from, pts_to], ignore_index=True).drop_duplicates(["ensemble", "lead_time"])
    points = points.sort_values(["ensemble", "lead_time"])

    invalid = sub[~sub["valid"]]
    flag_from = invalid[["ensemble", "lead_time_from", "lead_time_from_hours", "value_from"]].rename(
        columns={"lead_time_from": "lead_time", "lead_time_from_hours": "lead_time_hours", "value_from": "value"}
    )
    flag_to = invalid[["ensemble", "lead_time_to", "lead_time_to_hours", "value_to"]].rename(
        columns={"lead_time_to": "lead_time", "lead_time_to_hours": "lead_time_hours", "value_to": "value"}
    )
    flagged = pd.concat([flag_from, flag_to], ignore_index=True).drop_duplicates(["ensemble", "lead_time"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, grp in points.groupby("ensemble"):
        ax.plot(grp["lead_time_hours"], grp["value"], color=COLOR_MEMBER, alpha=0.35, linewidth=0.8)
    mean_series = points.groupby("lead_time_hours")["value"].mean().sort_index()
    ax.plot(mean_series.index, mean_series.values, color=COLOR_MEAN, linewidth=2.2)

    handles = [
        plt.Line2D([0], [0], color=COLOR_MEMBER, alpha=0.6, linewidth=1.5, label="Individual member"),
        plt.Line2D([0], [0], color=COLOR_MEAN, linewidth=2.2, label="Ensemble mean"),
    ]
    if not flagged.empty:
        ax.scatter(flagged["lead_time_hours"], flagged["value"], color=COLOR_INVALID, s=18, zorder=5)
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", color=COLOR_INVALID, label="Flagged step"))

    ax.legend(handles=handles, loc="best")
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cross-time consistency: {check_name}")
    ax.grid(True, color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_ensemble_spread(sub, out_path):
    check_name = sub["check"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["lead_time_hours"], sub["spread"], color=COLOR_MEAN, linewidth=2.0)

    invalid = sub[~sub["valid"]]
    if not invalid.empty:
        ax.scatter(invalid["lead_time_hours"], invalid["spread"], color=COLOR_INVALID, s=24, zorder=5, label="Flagged")
        ax.legend(loc="best")

    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("Std dev across ensemble")
    ax.set_title(f"Cross-ensemble consistency: {check_name}")
    ax.grid(True, color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    tables = load_tables()
    output_dir.mkdir(parents=True, exist_ok=True)

    standalone = tables["standalone_variable"]
    cross_variable = tables["cross_variable_consistency"]
    cross_time = tables["cross_time_consistency"]
    cross_ensemble = tables["cross_ensemble_consistency"]

    member_ids = set()
    if not standalone.empty:
        member_ids |= set(standalone["ensemble"].unique())
    if not cross_variable.empty:
        member_ids |= set(cross_variable["ensemble"].unique())

    print("Rendering standalone-variable bounds charts...")
    if not standalone.empty:
        plot_aggregate_heatmap(
            standalone, "variable", "Standalone variable bounds - fraction of members invalid",
            output_dir / "standalone_variable_summary_heatmap.png",
        )
    else:
        print("[INFO] standalone_variable table is empty.")

    print("Rendering cross-variable consistency charts...")
    if not cross_variable.empty:
        plot_aggregate_heatmap(
            cross_variable, "check", "Cross-variable consistency - fraction of members invalid",
            output_dir / "cross_variable_consistency_summary_heatmap.png",
        )
    else:
        print("[INFO] cross_variable_consistency table is empty - no pressure-level variables to compare.")

    for member_id in sorted(member_ids):
        member_dir = output_dir / f"member_{int(member_id):02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        if not standalone.empty:
            plot_member_heatmap(
                standalone, "variable", member_id, f"Standalone variable bounds - member {member_id}",
                member_dir / "standalone_variable_heatmap.png",
            )
        if not cross_variable.empty:
            plot_member_heatmap(
                cross_variable, "check", member_id, f"Cross-variable consistency - member {member_id}",
                member_dir / "cross_variable_consistency_heatmap.png",
            )

    print("Rendering cross-time consistency charts...")
    if not cross_time.empty:
        for check_name in sorted(cross_time["check"].unique()):
            safe_name = check_name.replace(":", "_")
            plot_cross_time_check(
                cross_time[cross_time["check"] == check_name],
                output_dir / f"cross_time_{safe_name}.png",
            )
    else:
        print("[INFO] cross_time_consistency table is empty.")

    print("Rendering cross-ensemble consistency charts...")
    if not cross_ensemble.empty:
        for check_name in sorted(cross_ensemble["check"].unique()):
            safe_name = check_name.replace(":", "_")
            plot_cross_ensemble_spread(
                cross_ensemble[cross_ensemble["check"] == check_name],
                output_dir / f"cross_ensemble_{safe_name}.png",
            )
    else:
        print("[INFO] cross_ensemble_consistency table is empty - no rules implemented yet.")

    print(f"\nDone. Validation charts written to {output_dir}/")


if __name__ == "__main__":
    main()
