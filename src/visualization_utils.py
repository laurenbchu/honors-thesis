import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# Plotting and visualization utilities
# ------------------------------------------------------------------

Z = 1.96

COLOR_MAP = {
    "Black/African American": "#D55E00", # Dark orange
    "Hispanic/Latino": "#0072B2", # Dark blue
    "White": "#999999", # Gray
    "Asian": "#009E73" # Green
}

SENSITIVITY_COLOR_MAP = {
    "Baseline": "#E69F00",
    "Mixed": "#56B4E9",
    "Multiperson": "#CC79A7"
}

RACE_ORDER = ["Black/African American", "Hispanic/Latino", "White", "Asian"]



def _safe_visualization_setup(df, race_col):
    """
    Apply race ordering with minimal setup.
    """
    out = df.copy()
    if race_col in out.columns:
        out[race_col] = pd.Categorical(out[race_col], categories=RACE_ORDER, ordered=True)
    if "Year" in out.columns and race_col in out.columns:
        out = out.sort_values(["Year", race_col])
    return out



def plot_policing_by_year(policing_analysis):
    """
    Two-row summary figure:
      Top row    -- pooled 2022-2024 dot plots (one point per race, with CI for rates)
      Bottom row -- by-year line plots (with CI ribbons for rates)
    White group: dashed line, hollow circle markers throughout.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    df = policing_analysis.copy()
    race_col = "Perceived Race"
    df[race_col] = pd.Categorical(df[race_col], categories=RACE_ORDER, ordered=True)
    df = df.sort_values(["Year", race_col])

    color_map = COLOR_MAP
    years = sorted(df["Year"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Pre-compute pooled values for top row
    # ------------------------------------------------------------------
    pooled_rows = []
    for race in RACE_ORDER:
        d = df[df[race_col] == race]
        row = {"Perceived Race": race}

        for col in ["Stops per 1,000", "Searches per 1,000"]:
            row[col] = d[col].mean()

        sc  = d["Search Count"].sum()
        stc = d["Stop Count"].sum()
        sr  = sc / stc if stc > 0 else np.nan
        row["Search Rate"]    = sr
        row["Search Rate SE"] = np.sqrt(sr * (1 - sr) / stc) if stc > 0 else np.nan

        pooled_rows.append(row)

    pooled = pd.DataFrame(pooled_rows)
    pooled[race_col] = pd.Categorical(
        pooled[race_col], categories=RACE_ORDER, ordered=True
    )

    # ------------------------------------------------------------------
    # Figure: taller to give header area more room
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    panels = [
        {
            "col":       0,
            "y_col":     "Stops per 1,000",
            "top_title": "(A) Stop Rate\n(per 1,000 residents)",
            "bot_title": "(D) Stop Rate by Year",
            "ylabel":    "Stops per 1,000 Residents",
            "is_rate":   False,
        },
        {
            "col":       1,
            "y_col":     "Searches per 1,000",
            "top_title": "(B) Search Rate\n(per 1,000 residents)",
            "bot_title": "(E) Search Rate by Year",
            "ylabel":    "Searches per 1,000 Residents",
            "is_rate":   False,
        },
        {
            "col":       2,
            "y_col":     "Search Rate",
            "top_title": "(C) Conditional Search Rate\n(among those stopped)",
            "bot_title": "(F) Conditional Search Rate by Year",
            "ylabel":    "Search Rate",
            "is_rate":   True,
        },
    ]

    SHORT_LABELS = ["Black", "Hispanic", "White", "Asian"]

    def fmt_clean(y, _):
        if abs(y - round(y)) < 1e-9:
            return str(int(round(y)))
        return f"{y:.1f}"

    # ------------------------------------------------------------------
    # TOP ROW: pooled dot plots
    # ------------------------------------------------------------------
    for panel in panels:
        ax    = axes[0, panel["col"]]
        y_col = panel["y_col"]

        for race in RACE_ORDER:
            d = pooled[pooled[race_col] == race]
            if d.empty or d[y_col].isna().all():
                continue

            val   = float(d[y_col].values[0])
            xpos  = list(RACE_ORDER).index(race)
            color = color_map[race]

            # Error bars for conditional rate only, and only if CI > marker radius
            if panel["is_rate"]:
                se_val = (
                    float(d["Search Rate SE"].values[0])
                    if "Search Rate SE" in d.columns
                    else np.nan
                )
                ci = se_val * 1.96 if pd.notna(se_val) else np.nan

                if pd.notna(ci) and ci > 0:
                    ax.errorbar(
                        xpos, val,
                        yerr=ci,
                        fmt="none",
                        ecolor=color,
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        alpha=0.9,
                        zorder=4,
                    )

            # Marker on top
            ax.plot(
                xpos, val,
                marker="o", markersize=10,
                color=color,
                markerfacecolor="white" if race == "White" else color,
                markeredgecolor=color,
                markeredgewidth=2.0,
                linestyle="none",
                zorder=5,
            )

        ax.set_title(panel["top_title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(panel["ylabel"], fontsize=14)
        ax.set_xticks(range(len(RACE_ORDER)))
        ax.set_xticklabels(SHORT_LABELS, fontsize=13)
        ax.set_xlim(-0.6, len(RACE_ORDER) - 0.4)
        ax.tick_params(axis="x", length=3)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)

        if panel["is_rate"]:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        elif y_col == "Searches per 1,000":
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_clean))

        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.18 * (yhi - ylo))

    # ------------------------------------------------------------------
    # BOTTOM ROW: by-year line plots
    # White: line first (zorder=3), then markers with white fill (zorder=5)
    # so hollow circle sits cleanly above the dashes.
    # ------------------------------------------------------------------
    for panel in panels:
        ax    = axes[1, panel["col"]]
        y_col = panel["y_col"]

        for race in RACE_ORDER:
            d = df[df[race_col] == race].sort_values("Year")
            if d.empty or d[y_col].isna().all():
                continue

            color    = color_map[race]
            is_white = race == "White"
            lw       = 2.8 if is_white else 2.2
            ls       = "--" if is_white else "-"
            mew      = 2.0 if is_white else 1.2

            # CI ribbon
            if panel["is_rate"]:
                se_col = f"{y_col} SE"
                if se_col in d.columns and d[se_col].notna().any():
                    ax.fill_between(
                        d["Year"],
                        d[y_col] - d[se_col] * 1.96,
                        d[y_col] + d[se_col] * 1.96,
                        color=color, alpha=0.18, linewidth=0, zorder=2,
                    )

            if is_white:
                # Dashed line, no markers
                ax.plot(
                    d["Year"], d[y_col],
                    marker="none",
                    linewidth=lw, linestyle=ls,
                    color=color,
                    zorder=3,
                )
                # Markers only, white fill covers the line beneath
                ax.plot(
                    d["Year"], d[y_col],
                    marker="o", markersize=8,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=mew,
                    linestyle="none",
                    color=color,
                    label=race,
                    zorder=5,
                )
            else:
                ax.plot(
                    d["Year"], d[y_col],
                    marker="o", markersize=7,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=mew,
                    linewidth=lw, linestyle=ls,
                    color=color,
                    label=race,
                    zorder=3,
                )

        ax.set_title(panel["bot_title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel(panel["ylabel"], fontsize=14)
        ax.set_xticks(years)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)

        if y_col == "Stops per 1,000":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
        elif y_col == "Searches per 1,000":
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_clean))
        elif panel["is_rate"]:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.15 * (yhi - ylo))

    # ------------------------------------------------------------------
    # Spacing first -- must be called before placing fig.text so that
    # figure-fraction coordinates are stable.
    #
    # top=0.84 leaves ~16% of the figure height above the panels for:
    #   suptitle  (~y=0.98)
    #   legend    (~y=0.93)
    #   "Pooled"  (~y=0.87, just above top panel edge at 0.84)
    # ------------------------------------------------------------------
    plt.subplots_adjust(
        left=0.07, right=0.97,
        top=0.83, bottom=0.07,
        hspace=0.50, wspace=0.28,
    )

    # ------------------------------------------------------------------
    # Overall title
    # ------------------------------------------------------------------
    fig.suptitle(
        "Police Stop and Search Patterns by Perceived Race",
        fontsize=18, fontweight="bold", y=0.985,
    )

    # ------------------------------------------------------------------
    # Legend: sits between suptitle and "Pooled" row header, no overlap
    # ------------------------------------------------------------------
    legend_handles = []
    for race in RACE_ORDER:
        is_white = race == "White"
        handle = mlines.Line2D(
            [0], [0],
            color=color_map[race],
            linestyle=(0, (2.5, 2.5)) if is_white else "-",
            linewidth=3.5 if is_white else 3.0,
            marker="o",
            markersize=13,
            markerfacecolor="white" if is_white else color_map[race],
            markeredgecolor=color_map[race],
            markeredgewidth=2.0 if is_white else 1.2,
            label=race,
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=16,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.7,
        columnspacing=2.0,
    )

    # ------------------------------------------------------------------
    # Row header labels + divider
    # "Pooled" sits just above the top panel row edge (top=0.84)
    # Divider at y=0.455; "By Year" just below at y=0.433
    # ------------------------------------------------------------------
    fig.text(
        0.03, 0.89,
        "Pooled 2022-2024",
        ha="left", va="bottom",
        fontsize=16, fontweight="bold", color="0.25",
    )

    fig.add_artist(
    plt.Line2D(
        [0.03, 0.99], [0.46, 0.46],
        transform=fig.transFigure,
        color="0.75", linewidth=0.8,
    )

    )
    fig.text(
        0.03, 0.43,
        "By Year",
        ha="left", va="top",
        fontsize=16, fontweight="bold", color="0.25",
    )
    return fig



def plot_hit_rate_by_year(policing_analysis):
    """
    1x2 figure of contraband hit rates by perceived race.

    Panel A: Pooled 2022-2024 hit rate by race (dot plot with 95% CI)
    Panel B: Hit rate by year and race (line plot with 95% CI ribbons)

    Formatting matches visualize_search_and_hit_rates_by_reason.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    race_col = "Perceived Race"
    df = policing_analysis.copy()
    df[race_col] = pd.Categorical(df[race_col], categories=RACE_ORDER, ordered=True)
    df = df[df[race_col] != "Other"].sort_values(["Year", race_col])

    years = sorted(df["Year"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Pooled stats for Panel A
    # ------------------------------------------------------------------
    pooled = (
        df.groupby(race_col, as_index=False)
        .agg({"Search Count": "sum", "Hit Count": "sum"})
    )
    pooled["Hit Rate"] = np.where(
        pooled["Search Count"] > 0,
        pooled["Hit Count"] / pooled["Search Count"],
        np.nan,
    )
    pooled["Hit Rate SE"] = np.where(
        pooled["Search Count"] > 0,
        np.sqrt(
            pooled["Hit Rate"] * (1 - pooled["Hit Rate"]) / pooled["Search Count"]
        ),
        np.nan,
    )
    pooled["Hit Rate %"]  = pooled["Hit Rate"] * 100
    pooled["Hit CI %"]    = pooled["Hit Rate SE"] * 1.96 * 100
    pooled[race_col] = pd.Categorical(
        pooled[race_col], categories=RACE_ORDER, ordered=True
    )
    pooled = pooled.sort_values(race_col)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.32})

    # ── Panel A: pooled dot plot ───────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(RACE_ORDER))
    short_labels = ["Black", "Hispanic", "White", "Asian"]

    for i, race in enumerate(RACE_ORDER):
        row = pooled[pooled[race_col] == race]
        if row.empty:
            continue

        is_white = race == "White"
        mfc  = "white" if is_white else COLOR_MAP[race]
        mew  = 2.0    if is_white else 1.0

        ax.errorbar(
            x[i],
            row["Hit Rate %"].values[0],
            yerr=row["Hit CI %"].values[0],
            fmt="o",
            linestyle="none",
            markersize=11,
            markerfacecolor=mfc,
            markeredgecolor=COLOR_MAP[race],
            markeredgewidth=mew,
            capsize=4, capthick=1.5, elinewidth=1.8,
            color=COLOR_MAP[race],
            zorder=4 if not is_white else 3,
        )

    ax.set_title("(A) Hit Rate\n(Pooled 2022–2024)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Perceived Race", fontsize=13)
    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=13)
    ax.tick_params(labelsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(RACE_ORDER) - 0.5)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(max(0, ylo), yhi * 1.08)

    # ── Panel B: by-year line plot ─────────────────────────────────────
    ax = axes[1]

    for race in RACE_ORDER:
        d = df[df[race_col] == race].sort_values("Year")
        if d.empty or d["Hit Rate"].isna().all():
            continue

        color    = COLOR_MAP[race]
        is_white = race == "White"
        lw       = 2.8 if is_white else 2.2
        ls       = "--" if is_white else "-"
        mew      = 2.0 if is_white else 1.2

        se_col = "Hit Rate SE"
        if se_col in d.columns and d[se_col].notna().any():
            ax.fill_between(
                d["Year"],
                (d["Hit Rate"] - d[se_col] * 1.96) * 100,
                (d["Hit Rate"] + d[se_col] * 1.96) * 100,
                color=color, alpha=0.12, linewidth=0, zorder=2,
            )

        y_vals = d["Hit Rate"] * 100

        if is_white:
            ax.plot(d["Year"], y_vals,
                    marker="none", linewidth=lw, linestyle=ls,
                    color=color, zorder=3)
            ax.plot(d["Year"], y_vals,
                    marker="o", markersize=11,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=mew,
                    linestyle="none", color=color, zorder=5)
        else:
            ax.plot(d["Year"], y_vals,
                    marker="o", markersize=11,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=mew,
                    linewidth=lw, linestyle=ls,
                    color=color, zorder=3)

    ax.set_title("(B) Hit Rate by Year", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_xticks(years)
    ax.set_xlim(min(years) - 0.2, max(years) + 0.2)
    ax.tick_params(labelsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(max(0, ylo), yhi * 1.08)

    # ------------------------------------------------------------------
    # Shared title
    # ------------------------------------------------------------------
    fig.suptitle(
        "Contraband Hit Rate by Perceived Race (2022–2024)",
        fontsize=16, fontweight="bold", y=0.97,
    )

    # ------------------------------------------------------------------
    # Legend: top center, matches other figures
    # ------------------------------------------------------------------
    legend_handles = []
    for race in RACE_ORDER:
        is_white = race == "White"
        handle = mlines.Line2D(
            [0], [0],
            color=COLOR_MAP[race],
            linestyle=(0, (2.1, 2.1)) if is_white else "-",
            linewidth=3.0 if is_white else 2.5,
            marker="o",
            markersize=11,
            markerfacecolor="white" if is_white else COLOR_MAP[race],
            markeredgecolor=COLOR_MAP[race],
            markeredgewidth=2.0 if is_white else 1.2,
            label=race,
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=13,
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.subplots_adjust(top=0.75, bottom=0.15, left=0.08, right=0.98)

    return fig



def visualize_search_and_hit_rates_by_reason(policing_by_reason):
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    race_order   = ["Black/African American", "Hispanic/Latino", "White", "Asian"]
    reason_order = list(policing_by_reason.keys())

    def wrap_label(s, max_chars=12):
        """Insert a newline before the last word if label exceeds max_chars."""
        if len(s) <= max_chars:
            return s
        words = s.split()
        # find best split point near the middle
        mid = len(s) // 2
        best, best_diff = 0, float("inf")
        pos = 0
        for i, w in enumerate(words[:-1]):
            pos += len(w) + 1
            diff = abs(pos - mid)
            if diff < best_diff:
                best_diff, best = diff, i + 1
        return " ".join(words[:best]) + "\n" + " ".join(words[best:])

    wrapped_reason_labels = [wrap_label(r) for r in reason_order]

    # ------------------------------------------------------------------
    # Build pooled dataframe
    # ------------------------------------------------------------------
    pooled_rows = []
    for reason, rates_df in policing_by_reason.items():
        df = rates_df[rates_df["Perceived Race"] != "Other"].copy()
        pooled = (
            df.groupby("Perceived Race", as_index=False)
            .agg({"Stop Count": "sum", "Search Count": "sum", "Hit Count": "sum"})
        )
        pooled["Reason"] = reason
        pooled["Search Rate"] = np.where(
            pooled["Stop Count"] > 0,
            pooled["Search Count"] / pooled["Stop Count"], np.nan,
        )
        pooled["Search Rate SE"] = np.where(
            pooled["Stop Count"] > 0,
            np.sqrt(pooled["Search Rate"] * (1 - pooled["Search Rate"]) / pooled["Stop Count"]),
            np.nan,
        )
        pooled["Hit Rate"] = np.where(
            pooled["Search Count"] > 0,
            pooled["Hit Count"] / pooled["Search Count"], np.nan,
        )
        pooled["Hit Rate SE"] = np.where(
            pooled["Search Count"] > 0,
            np.sqrt(pooled["Hit Rate"] * (1 - pooled["Hit Rate"]) / pooled["Search Count"]),
            np.nan,
        )
        pooled_rows.append(pooled)

    pooled_all = pd.concat(pooled_rows, ignore_index=True)
    pooled_all["Perceived Race"] = pd.Categorical(
        pooled_all["Perceived Race"], categories=race_order, ordered=True
    )
    pooled_all["Reason"] = pd.Categorical(
        pooled_all["Reason"], categories=reason_order, ordered=True
    )
    pooled_all = pooled_all.sort_values(["Perceived Race", "Reason"])
    pooled_all["Search Rate %"] = pooled_all["Search Rate"] * 100
    pooled_all["Search CI %"]   = pooled_all["Search Rate SE"] * 1.96 * 100
    pooled_all["Hit Rate %"]    = pooled_all["Hit Rate"] * 100
    pooled_all["Hit CI %"]      = pooled_all["Hit Rate SE"] * 1.96 * 100

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.32})
    x = np.arange(len(reason_order))

    offsets = {
        "Black/African American": -0.18,
        "Hispanic/Latino":        -0.06,
        "White":                   0.06,
        "Asian":                   0.18,
    }

    panel_specs = [
        {"ax": axes[0], "y_col": "Search Rate %", "ci_col": "Search CI %",
         "ylabel": "Search Rate (%)", "title": "(A) Search Rate"},
        {"ax": axes[1], "y_col": "Hit Rate %",    "ci_col": "Hit CI %",
         "ylabel": "Hit Rate (%)",    "title": "(B) Hit Rate"},
    ]

    for panel in panel_specs:
        ax = panel["ax"]

        for race in race_order:
            d = pooled_all[pooled_all["Perceived Race"] == race].sort_values("Reason")
            if d.empty:
                continue

            y      = d[panel["y_col"]].to_numpy()
            yerr   = d[panel["ci_col"]].to_numpy()
            x_race = x + offsets[race]
            is_white = race == "White"

            ax.errorbar(
                x_race, y, yerr=yerr,
                fmt="o", linestyle="none",
                markersize=11,
                markerfacecolor="white" if is_white else COLOR_MAP[race],
                markeredgecolor=COLOR_MAP[race],
                markeredgewidth=2.0 if is_white else 1.0,
                capsize=4, capthick=1.5, elinewidth=1.8,
                color=COLOR_MAP[race], label=race,
                zorder=3 if is_white else 4,
            )

            if panel["title"] == "(B) Hit Rate" and race == "Asian":
                non_moving_idx = (
                    reason_order.index("Non-moving violation")
                    if "Non-moving violation" in reason_order else None
                )
                if non_moving_idx is not None:
                    row = d[d["Reason"] == "Non-moving violation"]
                    if not row.empty:
                        n_val = int(row["Search Count"].values[0])
                        xi    = non_moving_idx + offsets[race]
                        yi    = row[panel["y_col"]].values[0]
                        ax.annotate(
                            f"n={n_val:,}",
                            xy=(xi, yi), xytext=(6, 6),
                            textcoords="offset points",
                            fontsize=13,    
                            color=COLOR_MAP[race], fontstyle="italic",
                        )

        ax.set_title(panel["title"], fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Reason for Contact", fontsize=13, labelpad=12)
        ax.set_ylabel(panel["ylabel"], fontsize=13, labelpad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            wrapped_reason_labels,  
            rotation=0, ha="center",
            fontsize=12,         
        )
        ax.tick_params(labelsize=13)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.set_xlim(-0.5, len(reason_order) - 0.5 + 0.2)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(max(0, y_min), y_max * 1.08)

    # ------------------------------------------------------------------
    # Title & legend
    # ------------------------------------------------------------------
    fig.suptitle(
        "Search and Contraband Hit Rates by Reason for Contact and Race "
        "(2022\u20132024 Pooled)",
        fontsize=18, fontweight="bold", y=0.99,
    )

    legend_handles = []
    for race in race_order:
        is_white = race == "White"
        handle = mlines.Line2D(
            [0], [0], color=COLOR_MAP[race], linestyle="none", marker="o",
            markersize=12,
            markerfacecolor="white" if is_white else COLOR_MAP[race],
            markeredgecolor=COLOR_MAP[race],
            markeredgewidth=2.0 if is_white else 1.0,
            label=race,
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.95),
        ncol=4, frameon=True, fancybox=False, edgecolor="0.8",
        fontsize=14, handletextpad=0.6, columnspacing=1.5,
    )

    fig.subplots_adjust(top=0.8, bottom=0.18, left=0.08, right=0.98)

    return fig



def plot_agency_black_white_rates(agency_df):
    """
    Two-panel publication-ready scatter plot comparing White and Black
    search rates (Panel A) and hit rates (Panel B) by agency.

    Exclusion criteria applied consistently across both panels:
    - Agencies with fewer than 30 Black searches excluded
    - Agencies with Black hit rate 95% CI > ±20 percentage points excluded

    Panel A: White vs Black conditional search rate (searches/stops)
    Panel B: White vs Black contraband hit rate (hits/searches)
    Points above the 45-degree parity line indicate higher Black rates.
    X and Y axes scale independently to maximize plotting area.
    """

    plt.rcParams.update({
        "font.size":        13,
        "axes.titlesize":   15,
        "axes.labelsize":   14,
        "xtick.labelsize":  13,
        "ytick.labelsize":  13,
        "legend.fontsize":  13,
    })

    d = agency_df.copy()

    # ── Compute CIs for exclusion ─────────────────────────────────────
    d["Black_Hit_Rate_SE"] = np.where(
        d["Black_Search_Count"] > 0,
        np.sqrt(
            d["Black_Hit_Rate"] *
            (1 - d["Black_Hit_Rate"]) /
            d["Black_Search_Count"]
        ),
        np.nan,
    )
    d["Black_CI_Width"] = d["Black_Hit_Rate_SE"] * 1.96 * 100

    # ── Apply exclusion criteria ──────────────────────────────────────
    d = d[
        (d["Black_Search_Count"] >= 30) &
        (d["Black_CI_Width"] <= 20)
    ].reset_index(drop=True)

    # ── Dot sizing ────────────────────────────────────────────────────
    sizes = 80 + 6.0 * np.sqrt(d["Avg_Search_Count"])

    # ── Figure: two panels side by side ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(22, 10),
        gridspec_kw={"wspace": 0},
    )

    # ── Helper: draw one scatter panel ───────────────────────────────
    def draw_panel(ax, x_vals, y_vals, panel_title, xlabel, ylabel, panel_label, manual_offsets=None):

        colors = np.where(y_vals > x_vals, "#009E73", "#D55E00")

        ax.scatter(
            x_vals, y_vals,
            s=sizes,
            alpha=0.75,
            c=colors,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )

        # ── Independent axis limits with 8% headroom ─────────────────
        x_max = np.nanmax(x_vals) * 1.08
        y_max = np.nanmax(y_vals) * 1.08
        max_val = max(x_max, y_max)

        # Parity line drawn to whichever axis limit is smaller
        ax.plot(
            [0, max_val], [0, max_val],
            linestyle="--", linewidth=2.5,
            color="gray", alpha=0.5, zorder=2,
        )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect("equal", adjustable="box")

        # ── Agency labels ─────────────────────────────────────────────
        for _, row in d.iterrows():
            xi = x_vals[row.name]
            yi = y_vals[row.name]
            ax.annotate(
                row["agency_name"],
                xy=(xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=11,
                alpha=0.9,
                zorder=4,
            )

        # Apply manual offset corrections if provided
        if manual_offsets:
            for txt in ax.texts:
                name = txt.get_text()
                if name in manual_offsets:
                    dx, dy = manual_offsets[name]
                    x0, y0 = txt.get_position()
                    txt.set_position((x0 + dx, y0 + dy))

        # ── Above/below counts ────────────────────────────────────────
        n_above = int((y_vals > x_vals).sum())
        n_below = int((y_vals <= x_vals).sum())
        n_label_above = "agency" if n_above == 1 else "agencies"
        n_label_below = "agency" if n_below == 1 else "agencies"

        ax.text(
            0.97, 0.05,
            f"{n_below} {n_label_below}: White rate > Black\n"
            f"{n_above}  {n_label_above}: Black rate > White",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=13, color="dimgray",
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.8", alpha=0.85),
        )

        # ── Panel label ───────────────────────────────────────────────
        ax.text(
            0.03, 0.97, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=15, fontweight="bold",
        )

        # ── Dot-size subtitle ─────────────────────────────────────────
        ax.text(
            0.5, 0.997,
            "Dot size proportional to average searches (2022–2024)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, color="dimgray", style="italic",
        )

        # ── Axes formatting ───────────────────────────────────────────
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", labelpad=10)
        ax.set_title(panel_title, fontsize=15, fontweight="bold", pad=16)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # ── Panel A: Search rates ─────────────────────────────────────────
    draw_panel(
        ax1,
        x_vals      = (d["White_Search_Rate"] * 100).to_numpy(),
        y_vals      = (d["Black_Search_Rate"]  * 100).to_numpy(),
        panel_title = "Search Rate by Agency\n(Conditional on Stop)",
        xlabel      = "White Search Rate (%)",
        ylabel      = "Black/African American Search Rate (%)",
        panel_label = "(A)",
        manual_offsets = {
            "Tustin PD":    (-60, -15),
            "Cypress PD":  (5, -5),
            "Laguna Beach PD": (-80, 2),
            "Costa Mesa PD": (-80, -20),
            "Irvine PD": (0, -20),
            "Orange PD": (-70, 0),
            "Santa Ana PD": (2, -15),
            "Newport Beach PD": (4, -7),
            "Huntington Beach PD": (-120, 1),
        }
    )

    # ── Panel B: Hit rates ────────────────────────────────────────────
    draw_panel(
        ax2,
        x_vals      = (d["White_Hit_Rate"] * 100).to_numpy(),
        y_vals      = (d["Black_Hit_Rate"]  * 100).to_numpy(),
        panel_title = "Contraband Hit Rate by Agency\n(Outcome Test for Searches)",
        xlabel      = "White Hit Rate (%)",
        ylabel      = "Black/African American Hit Rate (%)",
        panel_label = "(B)",
        manual_offsets = {
            "Santa Ana PD":  (-80, 0),
            "Tustin PD":    (5, -10),
            "Buena Park PD": (-80, 0),
            "Orange County Sheriff's Office": (5, -5),
            "Newport Beach PD": (-100, -20),
        },
    )

    # ── Shared legend ─────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], linestyle="--", linewidth=2.5,
                   color="gray", alpha=0.5, label="Equal Rates"),
        ax1.scatter([], [], s=100, c="#009E73", alpha=0.75,
                    edgecolor="white", linewidth=1.5,
                    label="Black Rate > White"),
        ax1.scatter([], [], s=100, c="#D55E00", alpha=0.75,
                    edgecolor="white", linewidth=1.5,
                    label="White Rate > Black"),
        ax1.scatter([], [], s=80 + 6.0 * np.sqrt(50),  c="gray",
                    alpha=0.5, edgecolor="white", linewidth=1.5,
                    label="Avg. searches ≈ 50"),
        ax1.scatter([], [], s=80 + 6.0 * np.sqrt(200), c="gray",
                    alpha=0.5, edgecolor="white", linewidth=1.5,
                    label="Avg. searches ≈ 200"),
        ax1.scatter([], [], s=80 + 6.0 * np.sqrt(500), c="gray",
                    alpha=0.5, edgecolor="white", linewidth=1.5,
                    label="Avg. searches ≈ 500"),
    ]

    fig.suptitle(
        "Agency-Level Comparison: White vs. Black/African American "
        "Search and Hit Rates",
        fontsize=16, fontweight="bold", y=0.98,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=13,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(top=0.84, bottom=0.06, left=0.07, right=0.97)

    return fig



def create_combined_sensitivity_visualization(baseline_df, mixed_df, multiperson_df):
    """
    Two-panel sensitivity figure comparing Baseline, Mixed classification,
    and Multiperson stops using a dot plot with 95% CI error bars.

    Panel A: Conditional Search Rate
    Panel B: Contraband Hit Rate
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    latest_year = baseline_df["Year"].max()

    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian"]

    baseline    = (baseline_df[baseline_df["Year"] == latest_year]
                   .set_index("Perceived Race").reindex(race_order).reset_index())
    mixed       = (mixed_df[mixed_df["Year"] == latest_year]
                   .set_index("Perceived Race").reindex(race_order).reset_index())
    multiperson = (multiperson_df[multiperson_df["Year"] == latest_year]
                   .set_index("Perceived Race").reindex(race_order).reset_index())

    conditions = [
        ("Baseline",    baseline,    SENSITIVITY_COLOR_MAP["Baseline"],    "o",  -0.18),
        ("Mixed",       mixed,       SENSITIVITY_COLOR_MAP["Mixed"],       "s",   0.00),
        ("Multiperson", multiperson, SENSITIVITY_COLOR_MAP["Multiperson"], "^",   0.18),
    ]

    x = np.arange(len(race_order))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.32})

    panel_specs = [
        {
            "ax":       axes[0],
            "rate_col": "Search Rate",
            "se_col":   "Search Rate SE",
            "title":    "(A) Conditional Search Rate",
            "ylabel":   "Search Rate (%)",
        },
        {
            "ax":       axes[1],
            "rate_col": "Hit Rate",
            "se_col":   "Hit Rate SE",
            "title":    "(B) Contraband Hit Rate",
            "ylabel":   "Hit Rate (%)",
        },
    ]

    for panel in panel_specs:
        ax = panel["ax"]

        for label, df, color, marker, offset in conditions:
            y    = df[panel["rate_col"]] * 100
            yerr = df[panel["se_col"]] * 1.96 * 100

            ax.errorbar(
                x + offset, y,
                yerr=yerr,
                fmt=marker,
                linestyle="none",
                markersize=10,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                capsize=4, capthick=1.5, elinewidth=1.8,
                color=color,
                label=label,
                zorder=4,
            )

        ax.set_title(panel["title"], fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Perceived Race", fontsize=13, labelpad=10)
        ax.set_ylabel(panel["ylabel"], fontsize=13)
        ax.set_xticks(x)
        wrapped_labels = ["Black/African\nAmerican", "Hispanic/\nLatino", "White", "Asian"]
        ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=13)
        ax.tick_params(labelsize=13)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(race_order) - 0.5)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(max(0, ylo), yhi * 1.08)

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------
    fig.suptitle(
        f"Sensitivity Analysis: Baseline vs. Mixed Classification vs. "
        f"Multiperson Stops ({latest_year})",
        fontsize=16, fontweight="bold", y=0.99,
    )

    # ------------------------------------------------------------------
    # Legend: single horizontal legend below title
    # ------------------------------------------------------------------
    legend_handles = []
    for label, _, color, marker, _ in conditions:
        handle = mlines.Line2D(
            [0], [0],
            color=color,
            linestyle="none",
            marker=marker,
            markersize=10,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.2,
            label=label,
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=13,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.07, right=0.98)

    return fig



def plot_enhancement_rate_by_race_statute(enhancement_by_primary):
    """
    Enhancement rates by race, stratified by statute level (Felony vs Misdemeanor).
    Creates side-by-side panels with one for Misdemeanor and one for Felony, colored by race.
    """
    # Aggregate by race and statute level
    summary = (
        enhancement_by_primary
        .groupby(['race_std', 'primary_statute_level'], as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Apply race ordering
    summary['race_std'] = pd.Categorical(summary['race_std'], categories=RACE_ORDER, ordered=True)
    summary = summary.sort_values(['race_std', 'primary_statute_level'])
    
    # Separate by statute level
    mis = summary[summary['primary_statute_level'] == 'Misdemeanor'].copy()
    fel = summary[summary['primary_statute_level'] == 'Felony'].copy()
    
    # Helper function to darken a color
    def darken_color(color, factor=0.7):
        """Darken a hex color by multiplying RGB values by factor"""
        rgb = mcolors.to_rgb(color)
        darkened = tuple(c * factor for c in rgb)
        return darkened
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.3})
    
    # ---- Panel A: Misdemeanor ----
    races_mis = mis['race_std'].astype(str).to_list()
    mis_rates = mis['Enhancement Rate'].to_numpy() * 100
    mis_se = mis['SE'].to_numpy() * 100
    mis_n = mis['N'].to_numpy()
    mis_errors = mis_se * Z
    
    x_mis = np.arange(len(races_mis))
    colors_mis = [COLOR_MAP.get(r, "#7f7f7f") for r in races_mis]
    
    # Create bars without outline
    ax1.bar(
        x_mis, mis_rates,
        color=colors_mis,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )
    
    # Add error bars matching bar color
    for i, (rate, err, color) in enumerate(zip(mis_rates, mis_errors, colors_mis)):
        if np.isfinite(rate):
            ax1.errorbar(
                i, rate,
                yerr=err,
                fmt='none',
                ecolor=darken_color(color),
                alpha=0.4,
                capsize=5,
                capthick=1.5,
                linewidth=1.5
            )
    
    # Add labels for Misdemeanor panel
    ax1.set_xlabel('Canonical Race', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Enhancement Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title('(A) Misdemeanor Charges', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_mis)
    ax1.set_xticklabels(races_mis, rotation=0, ha='center', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add sample sizes as text above bars
    max_y_mis = 0
    for i, (rate, err, n) in enumerate(zip(mis_rates, mis_errors, mis_n)):
        if np.isfinite(rate):
            y_pos = rate + err + 0.3
            ax1.text(
                i, y_pos,
                f'{rate:.1f}%\n(n={int(n):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
            max_y_mis = max(max_y_mis, y_pos + 2)
    
    ax1.set_ylim(0, min(100, max_y_mis))
    
    # ---- Panel B: Felony ----
    races_fel = fel['race_std'].astype(str).to_list()
    fel_rates = fel['Enhancement Rate'].to_numpy() * 100
    fel_se = fel['SE'].to_numpy() * 100
    fel_n = fel['N'].to_numpy()
    fel_errors = fel_se * Z
    
    x_fel = np.arange(len(races_fel))
    colors_fel = [COLOR_MAP.get(r, "#7f7f7f") for r in races_fel]
    
    # Create bars without outline
    ax2.bar(
        x_fel, fel_rates,
        color=colors_fel,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )
    
    # Add error bars matching bar color
    for i, (rate, err, color) in enumerate(zip(fel_rates, fel_errors, colors_fel)):
        if np.isfinite(rate):
            ax2.errorbar(
                i, rate,
                yerr=err,
                fmt='none',
                ecolor=darken_color(color),
                alpha=0.4,
                capsize=5,
                capthick=1.5,
                linewidth=1.5
            )
    
    # Add labels for Felony panel
    ax2.set_xlabel('Canonical Race', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Felony Charges', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_fel)
    ax2.set_xticklabels(races_fel, rotation=0, ha='center', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Add sample sizes as text above bars
    max_y_fel = 0
    for i, (rate, err, n) in enumerate(zip(fel_rates, fel_errors, fel_n)):
        if np.isfinite(rate):
            y_pos = rate + err + 0.3
            ax2.text(
                i, y_pos,
                f'{rate:.1f}%\n(n={int(n):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
            max_y_fel = max(max_y_fel, y_pos + 2)
    
    ax2.set_ylim(0, min(100, max_y_fel))
    
    # Add overall title
    fig.suptitle(
        'Enhancement Charge Rate by Race and Statute Level\n(with 95% Confidence Intervals)',
        fontsize=15,
        fontweight='bold',
        y=1.03
    )

    fig.text(
    0.505, 0.52, 'Enhancement Rate (%)',
    ha='center',
    va='center',
    rotation='vertical',
    fontsize=12,
    fontweight='bold'
    )
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.10, wspace=0.3)
    return fig



def plot_enhancement_rate_by_race_statute_category(enhancement_by_primary, top_categories=None):
    """
    Returns a dict with:
      - 'assault_violence_weapons': 2x2 figure
          Row 1: Assault/Violence (Misdemeanor, Felony)
          Row 2: Weapons (Misdemeanor, Felony)
          Rows separated by a horizontal divider with section labels.
      - 'dui': 1x2 figure
          DUI (Misdemeanor, Felony)

    Dot plot with 95% CI error bars, colored by race using COLOR_MAP.
    Formatting matches other figures in the paper.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    Z = 1.96

    df = enhancement_by_primary.copy()

    # ------------------------------------------------------------------
    # Resolve category names exactly as they appear in the data
    # ------------------------------------------------------------------
    available_categories = df["primary_charge_category"].dropna().astype(str).unique().tolist()

    def resolve_category_name(target):
        target_norm = target.strip().lower()
        for cat in available_categories:
            if cat.strip().lower() == target_norm:
                return cat
        raise ValueError(
            f"Could not find category '{target}' in primary_charge_category.\n"
            f"Available categories: {sorted(available_categories)}"
        )

    assault_cat = resolve_category_name("Assault/Violence")
    weapons_cat = resolve_category_name("Weapons")
    dui_cat     = resolve_category_name("Obstruct/Resist Officer")

    if top_categories is not None:
        keep = {str(x).strip().lower() for x in top_categories}
        requested = {
            assault_cat.strip().lower(),
            weapons_cat.strip().lower(),
            dui_cat.strip().lower(),
        }
        use_requested = requested.intersection(keep)
        if use_requested:
            df = df[
                df["primary_charge_category"].str.strip().str.lower().isin(use_requested)
            ].copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def summarize_category(category_name):
        cat = df[df["primary_charge_category"] == category_name].copy()
        summary = (
            cat.groupby(["race_std", "primary_statute_level"], as_index=False)
               .agg({"Enhanced": "sum", "N": "sum"})
        )
        if summary.empty:
            return None, None

        summary["Enhancement Rate"] = np.where(
            summary["N"] > 0,
            summary["Enhanced"] / summary["N"],
            np.nan,
        )
        summary["SE"] = np.where(
            summary["N"] > 0,
            np.sqrt(
                summary["Enhancement Rate"] *
                (1 - summary["Enhancement Rate"]) /
                summary["N"]
            ),
            np.nan,
        )
        summary["race_std"] = pd.Categorical(
            summary["race_std"], categories=RACE_ORDER, ordered=True
        )
        summary = summary.sort_values(["race_std", "primary_statute_level"])
        present_races = [
            r for r in RACE_ORDER
            if r in set(summary["race_std"].dropna().astype(str))
        ]
        return summary, present_races

    def aligned_arrays(summary, races, statute_level):
        sub    = summary[summary["primary_statute_level"] == statute_level].copy()
        lookup = {str(row["race_std"]): row for _, row in sub.iterrows()}

        rates = np.array(
            [100 * lookup[r]["Enhancement Rate"] if r in lookup else np.nan
             for r in races],
            dtype=float,
        )
        ses = np.array(
            [100 * lookup[r]["SE"] if r in lookup else np.nan
             for r in races],
            dtype=float,
        )
        ns = np.array(
            [lookup[r]["N"] if r in lookup else 0 for r in races],
            dtype=float,
        )
        return rates, ses * Z, ns

    # ------------------------------------------------------------------
    # Panel drawing — dot plot
    # ------------------------------------------------------------------
    short_labels = {
        "Black/African American": "Black/\nAfrican American",
        "Hispanic/Latino":        "Hispanic/\nLatino",
        "White":                  "White",
        "Asian":                  "Asian",
    }

    def plot_panel(ax, rates, errs, ns, races, panel_title, show_ylabel=False):
        x = np.arange(len(races))

        # ------ Pass 1: draw dots and error bars ----------------------
        for i, race in enumerate(races):
            if not np.isfinite(rates[i]) or ns[i] == 0:
                continue

            is_white = race == "White"
            color    = COLOR_MAP.get(race, "#7f7f7f")
            mfc      = "white" if is_white else color
            mew      = 2.0    if is_white else 1.0

            ax.errorbar(
                x[i], rates[i],
                yerr=errs[i],
                fmt="o",
                linestyle="none",
                markersize=13,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=mew,
                capsize=4, capthick=1.5, elinewidth=1.8,
                color=color,
                zorder=4 if not is_white else 3,
            )

        # ------ Set y limits before annotations -----------------------
        finite_tops = [
            rates[i] + errs[i]
            for i in range(len(rates))
            if np.isfinite(rates[i]) and ns[i] > 0
        ]
        max_top = max(finite_tops) if finite_tops else 20

        # Cap at 100% for high-rate panels (e.g. DUI misdemeanor ~91-94%)
        # so y-axis never shows >100%; otherwise use 1.35x for breathing room
        if max_top > 85:
            y_upper = 100
        else:
            y_upper = max_top * 1.35

        ax.set_ylim(0, y_upper)

        # ------ Pass 2: annotations using actual y range --------------
        y_range = y_upper
        offset  = max(y_range * 0.03, 0.5)  # proportional, floor of 0.5pp

        for i, race in enumerate(races):
            if not np.isfinite(rates[i]) or ns[i] == 0:
                continue

            annot_y = rates[i] + errs[i] + offset

            # If annotation would exceed y_upper, place it below the dot
            if annot_y + y_range * 0.08 > y_upper:
                ax.text(
                    x[i],
                    rates[i] - errs[i] - offset,
                    f"{rates[i]:.1f}%\n(n={int(ns[i]):,})",
                    ha="center", va="top",
                    fontsize=13,
                )
            else:
                ax.text(
                    x[i],
                    annot_y,
                    f"{rates[i]:.1f}%\n(n={int(ns[i]):,})",
                    ha="center", va="bottom",
                    fontsize=13,
                )

        # ------ Axes formatting ---------------------------------------
        ax.set_title(panel_title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Canonical Race", fontsize=13, labelpad=10)
        if show_ylabel:
            ax.set_ylabel("Enhancement Rate (%)", fontsize=13)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_labels.get(r, r) for r in races],
            rotation=0, ha="center", fontsize=13,
        )
        ax.tick_params(labelsize=13)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(races) - 0.5)

        return max_top

    # ------------------------------------------------------------------
    # Legend builder
    # ------------------------------------------------------------------
    def add_legend(fig, races):
        handles = []
        for race in races:
            is_white = race == "White"
            color    = COLOR_MAP.get(race, "#7f7f7f")
            h = mlines.Line2D(
                [0], [0],
                color=color, linestyle="none", marker="o",
                markersize=11,
                markerfacecolor="white" if is_white else color,
                markeredgecolor=color,
                markeredgewidth=2.0 if is_white else 1.0,
                label=race,
            )
            handles.append(h)

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=len(races),
            frameon=True,
            fancybox=False,
            edgecolor="0.8",
            fontsize=13,
            handletextpad=0.6,
            columnspacing=1.5,
        )

    # ------------------------------------------------------------------
    # Four-panel figure (Assault/Violence + Weapons)
    # ------------------------------------------------------------------
    def make_four_panel_combined_figure(cat1, cat2, figure_title):
        summary1, races1 = summarize_category(cat1)
        summary2, races2 = summarize_category(cat2)

        if summary1 is None and summary2 is None:
            return None

        fig, axes = plt.subplots(
            2, 2, figsize=(16, 12),
            gridspec_kw={"wspace": 0.32, "hspace": 0.60},
        )
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Row 1
        if summary1 is not None and races1:
            mis_rates, mis_errs, mis_n = aligned_arrays(summary1, races1, "Misdemeanor")
            fel_rates, fel_errs, fel_n = aligned_arrays(summary1, races1, "Felony")
            plot_panel(ax1, mis_rates, mis_errs, mis_n, races1,
                       "(A) Misdemeanor", show_ylabel=True)
            plot_panel(ax2, fel_rates, fel_errs, fel_n, races1,
                       "(B) Felony",      show_ylabel=False)
        else:
            ax1.axis("off"); ax2.axis("off")

        # Row 2
        if summary2 is not None and races2:
            mis_rates, mis_errs, mis_n = aligned_arrays(summary2, races2, "Misdemeanor")
            fel_rates, fel_errs, fel_n = aligned_arrays(summary2, races2, "Felony")
            plot_panel(ax3, mis_rates, mis_errs, mis_n, races2,
                       "(C) Misdemeanor", show_ylabel=True)
            plot_panel(ax4, fel_rates, fel_errs, fel_n, races2,
                       "(D) Felony",      show_ylabel=False)
        else:
            ax3.axis("off"); ax4.axis("off")

        # Row section labels
        fig.text(0.5, 0.87, cat1, ha="center", va="center",
                 fontsize=14, fontweight="bold")
        fig.text(0.5, 0.42, cat2, ha="center", va="center",
                 fontsize=14, fontweight="bold")

        # Horizontal divider between rows
        fig.add_artist(plt.Line2D(
            [0.08, 0.97], [0.46, 0.46],
            transform=fig.transFigure,
            color="0.75", linewidth=1.0, linestyle="--",
        ))

        fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.95)
        add_legend(fig, races1 or races2)
        fig.subplots_adjust(top=0.82, bottom=0.10, left=0.08, right=0.97)
        return fig

    # ------------------------------------------------------------------
    # Two-panel figure (DUI)
    # ------------------------------------------------------------------
    def make_two_panel_category_figure(category_name, figure_title):
        summary, races = summarize_category(category_name)
        if summary is None or not races:
            return None

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={"wspace": 0.32},
        )

        mis_rates, mis_errs, mis_n = aligned_arrays(summary, races, "Misdemeanor")
        fel_rates, fel_errs, fel_n = aligned_arrays(summary, races, "Felony")

        plot_panel(ax1, mis_rates, mis_errs, mis_n, races,
                   "(A) Misdemeanor Charges", show_ylabel=True)
        plot_panel(ax2, fel_rates, fel_errs, fel_n, races,
                   "(B) Felony Charges",      show_ylabel=False)

        fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.99)
        add_legend(fig, races)
        fig.subplots_adjust(top=0.78, bottom=0.15, left=0.08, right=0.97)
        return fig

    # ------------------------------------------------------------------
    # Build and return
    # ------------------------------------------------------------------
    figs = {}

    figs["assault_violence_weapons"] = make_four_panel_combined_figure(
        assault_cat, weapons_cat,
        "Enhancement Charge Rate by Race and Statute Level",
    )

    return figs



def plot_enhancement_by_statute(statute_rate_by_race, focus_statutes, statute_descriptions,
                                total_cases_by_race):
    """
    Dot plot of enhancement rates by statute and race, matching the
    existing figure style in the paper. One panel per statute arranged
    horizontally, with 95% CI error bars. Y-axes are shared across panels.
    """

    plt.rcParams.update({
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   13,
        "xtick.labelsize":  13,
        "ytick.labelsize":  13,
        "legend.fontsize":  13,
    })

    Z = 1.96

    short_labels = {
        "Black/African American": "Black/\nAfrican American",
        "Hispanic/Latino":        "Hispanic/\nLatino",
        "White":                  "White",
        "Asian":                  "Asian",
    }

    races      = [r for r in RACE_ORDER if r != "Other"]
    n_panels   = len(focus_statutes)
    total_lookup = (
        total_cases_by_race
        .set_index("race_std")["total_cases"]
        .to_dict()
    )

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7 * n_panels, 7),
        gridspec_kw={"wspace": 0.32},
    )
    if n_panels == 1:
        axes = [axes]

    # ── First pass: draw all panels, collect y maxima ─────────────────
    panel_max_tops = []

    for ax, statute, panel_letter in zip(axes, focus_statutes, "ABCDEFG"):

        sub = (
            statute_rate_by_race[
                (statute_rate_by_race["statute_normalized"] == statute) &
                (statute_rate_by_race["race_std"].isin(races))
            ]
            .set_index("race_std")
            .reindex(races)
        )

        rates = sub["rate"].to_numpy(dtype=float) * 100
        ses   = sub["se"].to_numpy(dtype=float)  * 100
        ns    = sub["cases_with_statute"].to_numpy(dtype=float)
        errs  = ses * Z
        x     = np.arange(len(races))

        # Dots and error bars
        for i, race in enumerate(races):
            if not np.isfinite(rates[i]) or ns[i] == 0:
                continue

            is_white = race == "White"
            color    = COLOR_MAP.get(race, "#7f7f7f")

            ax.errorbar(
                x[i], rates[i],
                yerr=errs[i],
                fmt="o",
                linestyle="none",
                markersize=13,
                markerfacecolor="white" if is_white else color,
                markeredgecolor=color,
                markeredgewidth=2.0 if is_white else 1.0,
                capsize=4, capthick=1.5, elinewidth=1.8,
                color=color,
                zorder=4 if not is_white else 3,
            )

        finite_tops = [
            rates[i] + errs[i]
            for i in range(len(races))
            if np.isfinite(rates[i]) and ns[i] > 0
        ]
        panel_max_tops.append(max(finite_tops) if finite_tops else 0.1)

        # Store data on axis for annotation pass
        ax._statute_data = {
            "rates": rates, "errs": errs, "ns": ns,
            "races": races, "statute": statute,
            "panel_letter": panel_letter,
        }

        # Axes formatting
        description = statute_descriptions.get(statute, statute)
        ax.set_title(
            f"({panel_letter}) {statute}\n{description}",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Canonical Race", fontsize=13, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_labels.get(r, r) for r in races],
            rotation=0, ha="center", fontsize=13,
        )
        ax.tick_params(labelsize=13)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(races) - 0.5)

    # ── Shared y-axis across all panels ───────────────────────────────
    global_max = max(panel_max_tops)
    y_upper    = global_max * 1.6

    for ax in axes:
        ax.set_ylim(0, y_upper)

    # ── Shared y-axis label ───────────────────────────────────────────
    fig.supylabel("Enhancement Rate (%)", fontsize=13, x=0.01)

    # ── Second pass: annotations (after y limits are fixed) ───────────
    offset = max(y_upper * 0.03, 0.01)

    for ax in axes:
        d      = ax._statute_data
        rates  = d["rates"]
        errs   = d["errs"]
        ns     = d["ns"]
        races_ = d["races"]

        for i, race in enumerate(races_):
            if not np.isfinite(rates[i]) or ns[i] == 0:
                continue

            denom = int(total_lookup.get(race, 0))
            label = f"{rates[i]:.2f}%\n(n={int(ns[i]):,} of {denom:,})"

            ax.text(
                i,
                rates[i] + errs[i] + offset,
                label,
                ha="center", va="bottom", fontsize=11,
            )

    # ── Shared legend ─────────────────────────────────────────────────
    legend_handles = []
    for race in races:
        is_white = race == "White"
        color    = COLOR_MAP.get(race, "#7f7f7f")
        h = mlines.Line2D(
            [0], [0],
            color=color, linestyle="none", marker="o", markersize=11,
            markerfacecolor="white" if is_white else color,
            markeredgecolor=color,
            markeredgewidth=2.0 if is_white else 1.0,
            label=race,
        )
        legend_handles.append(h)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(races),
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=13,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.suptitle(
        "Enhancement Rate by Statute and Race",
        fontsize=18, fontweight="bold", y=1.02,
    )

    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.08, right=0.97)
    return fig



def plot_wobbler_combined(df, top_categories=None, sort_by="rate_overall"):
    """
    Two-panel wobbler figure:
    - Panel A: Overall wobbler felony filing rate by race (dot plot with 95% CI)
    - Panel B: Cleveland dot plot by charge category and race

    Formatting matches other figures in the paper.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    Z_      = 1.96
    races   = list(RACE_ORDER)
    color_map = COLOR_MAP

    short_labels = {
        "Black/African American": "Black/\nAfrican American",
        "Hispanic/Latino":        "Hispanic/\nLatino",
        "White":                  "White",
        "Asian":                  "Asian",
    }

    wobblers = df[df["is_wobbler"]].copy()

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8),
        gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1.6]},
    )

    # ==================================================================
    # PANEL A: Overall wobbler felony filing rate — dot plot
    # ==================================================================
    summary = (
        wobblers.groupby(["race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Felony", "Misdemeanor"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["Total"]         = summary["Felony"] + summary["Misdemeanor"]
    summary["Felony Rate"]   = np.where(
        summary["Total"] > 0, summary["Felony"] / summary["Total"], np.nan
    )
    summary["Felony Rate SE"] = np.where(
        summary["Total"] > 0,
        np.sqrt(summary["Felony Rate"] * (1 - summary["Felony Rate"]) / summary["Total"]),
        np.nan,
    )
    summary = summary.reindex(races)

    rates = summary["Felony Rate"].to_numpy(dtype=float) * 100
    ses   = summary["Felony Rate SE"].to_numpy(dtype=float) * 100
    ns    = summary["Total"].to_numpy(dtype=float)
    errs  = ses * Z_
    x     = np.arange(len(races))

    for i, race in enumerate(races):
        if not np.isfinite(rates[i]) or ns[i] == 0:
            continue

        is_white = race == "White"
        color    = color_map.get(race, "#7f7f7f")

        ax1.errorbar(
            x[i], rates[i],
            yerr=errs[i],
            fmt="o",
            linestyle="none",
            markersize=13,
            markerfacecolor="white" if is_white else color,
            markeredgecolor=color,
            markeredgewidth=2.0 if is_white else 1.0,
            capsize=4, capthick=1.5, elinewidth=1.8,
            color=color,
            zorder=4 if not is_white else 3,
        )

    # Annotations — set y limits first
    finite_tops = [rates[i] + errs[i] for i in range(len(rates))
                   if np.isfinite(rates[i]) and ns[i] > 0]
    max_top  = max(finite_tops) if finite_tops else 60
    y_upper  = max_top * 1.35
    ax1.set_ylim(0, y_upper)
    offset   = max(y_upper * 0.03, 0.5)

    for i, race in enumerate(races):
        if not np.isfinite(rates[i]) or ns[i] == 0:
            continue
        ax1.text(
            x[i], rates[i] + errs[i] + offset,
            f"{rates[i]:.1f}%\n(n={int(ns[i]):,})",
            ha="center", va="bottom", fontsize=13,
        )

    ax1.set_title("(A) Overall Felony Filing Rate", fontsize=14,
                  fontweight="bold", pad=12)
    ax1.set_xlabel("Canonical Race", fontsize=13, labelpad=10)
    ax1.set_ylabel("Wobbler Charged as Felony (%)", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([short_labels.get(r, r) for r in races],
                        rotation=0, ha="center", fontsize=13)
    ax1.tick_params(labelsize=13)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax1.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlim(-0.5, len(races) - 0.5)

    # ==================================================================
    # PANEL B: Cleveland dot plot by charge category
    # ==================================================================
    g = (
        wobblers.groupby(["charge_category", "race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Felony", "Misdemeanor"]:
        if col not in g.columns:
            g[col] = 0

    g["Total"]      = g["Felony"] + g["Misdemeanor"]
    g["Felony Rate"] = np.where(g["Total"] > 0, g["Felony"] / g["Total"], np.nan)
    g["SE"]         = np.where(
        g["Total"] > 0,
        np.sqrt(g["Felony Rate"] * (1 - g["Felony Rate"]) / g["Total"]),
        np.nan,
    )
    g["lo"] = (g["Felony Rate"] - Z_ * g["SE"]) * 100
    g["hi"] = (g["Felony Rate"] + Z_ * g["SE"]) * 100
    g["x"]  = g["Felony Rate"] * 100
    g = g[g["race_std"].isin(races)].copy()
    g["race_std"] = pd.Categorical(g["race_std"], categories=races, ordered=True)

    # Determine categories
    if top_categories is not None:
        existing   = set(g["charge_category"].unique())
        categories = [c for c in top_categories if c in existing]
        g          = g[g["charge_category"].isin(categories)].copy()
    else:
        overall_rate = g.groupby("charge_category")[["Felony", "Misdemeanor"]].sum()
        overall_rate["Total"] = overall_rate["Felony"] + overall_rate["Misdemeanor"]
        overall_rate["Rate"]  = np.where(
            overall_rate["Total"] > 0,
            overall_rate["Felony"] / overall_rate["Total"],
            np.nan,
        )
        if sort_by == "rate_overall":
            categories = overall_rate["Rate"].sort_values(ascending=True).index.tolist()
        elif sort_by == "total":
            categories = g.groupby("charge_category")["Total"].sum() \
                          .sort_values(ascending=True).index.tolist()
        elif sort_by == "rate_black":
            blk = g[g["race_std"] == races[0]].set_index("charge_category")["Felony Rate"]
            categories = blk.sort_values(ascending=True).index.tolist()
        else:
            categories = sorted(overall_rate.index.tolist())

    g = g[g["charge_category"].isin(categories)].copy()

    y_base  = np.arange(len(categories))
    offsets = np.linspace(-0.28, 0.28, num=len(races))

    # Alternating row shading
    for i, y0 in enumerate(y_base):
        if i % 2 == 0:
            ax2.axhspan(y0 - 0.5, y0 + 0.5, color="0.95", alpha=1.0, zorder=0)

    ax2.grid(axis="x", alpha=0.35, linestyle=":", linewidth=0.8, zorder=1)
    ax2.set_axisbelow(True)

    for r_i, race in enumerate(races):
        sub    = g[g["race_std"] == race].set_index("charge_category").reindex(categories)
        y      = y_base + offsets[r_i]
        x_vals = sub["x"].to_numpy(dtype=float)
        lo     = sub["lo"].to_numpy(dtype=float)
        hi     = sub["hi"].to_numpy(dtype=float)
        n      = sub["Total"].to_numpy(dtype=float)
        color  = color_map.get(race, "#7f7f7f")
        is_white = race == "White"

        # CI lines
        for yi, l, h, nn in zip(y, lo, hi, n):
            if np.isfinite(l) and np.isfinite(h) and nn > 0:
                ax2.hlines(yi, l, h, linewidth=1.5, alpha=0.45,
                           color=color, zorder=2)

        # Dots
        ax2.scatter(
            x_vals, y,
            s=80,
            facecolors="white" if is_white else color,
            edgecolors=color,
            linewidths=2.0 if is_white else 1.0,
            alpha=0.95,
            zorder=3,
            label=race,
        )

    ax2.set_yticks(y_base)
    ax2.set_yticklabels(categories, fontsize=13)
    ax2.invert_yaxis()
    ax2.set_xlabel("Wobbler Charged as Felony (%)", fontsize=13, labelpad=10)
    ax2.set_ylabel("", fontsize=13)
    ax2.set_title("(B) Felony Filing Rate by Charge Category",
                  fontsize=14, fontweight="bold", pad=12)
    ax2.tick_params(labelsize=13)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    xmin = np.nanmin(g["lo"].to_numpy(dtype=float))
    xmax = np.nanmax(g["hi"].to_numpy(dtype=float))
    if np.isfinite(xmin) and np.isfinite(xmax):
        ax2.set_xlim(max(0, xmin - 2), min(100, xmax + 5))
    else:
        ax2.set_xlim(0, 100)

    # ------------------------------------------------------------------
    # Shared legend — top center, matches other figures
    # ------------------------------------------------------------------
    legend_handles = []
    for race in races:
        is_white = race == "White"
        color    = color_map.get(race, "#7f7f7f")
        h = mlines.Line2D(
            [0], [0],
            color=color, linestyle="none", marker="o",
            markersize=11,
            markerfacecolor="white" if is_white else color,
            markeredgecolor=color,
            markeredgewidth=2.0 if is_white else 1.0,
            label=race,
        )
        legend_handles.append(h)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=13,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------
    fig.suptitle(
        "Wobbler Charges: Felony Filing Rates by Race and Charge Category",
        fontsize=16, fontweight="bold", y=1.01,
    )

    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.98)

    return fig



def export_figure_to_pdf(fig, fig_name):
    """
    Export a single figure to PDF for Overleaf/LaTeX.
    """
    os.makedirs('../output/figures', exist_ok=True)
    output_path = f"../output/figures/{fig_name}.pdf"
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_name}.pdf")



def plot_policing_pooled(policing_analysis):
    """
    1x4 pooled summary figure for poster.
    Panels: (A) Stop Rate, (B) Search Rate, (C) Conditional Search Rate, (D) Hit Rate
    All pooled 2022-2024. No by-year row.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    df = policing_analysis.copy()
    race_col = "Perceived Race"
    df[race_col] = pd.Categorical(df[race_col], categories=RACE_ORDER, ordered=True)
    df = df[df[race_col] != "Other"].sort_values(["Year", race_col])

    # ------------------------------------------------------------------
    # Pooled values across all years
    # ------------------------------------------------------------------
    pooled_rows = []
    for race in RACE_ORDER:
        d = df[df[race_col] == race]
        row = {"Perceived Race": race}

        # Per-capita rates: average across years
        for col in ["Stops per 1,000", "Searches per 1,000"]:
            row[col] = d[col].mean()

        # Conditional search rate
        sc  = d["Search Count"].sum()
        stc = d["Stop Count"].sum()
        sr  = sc / stc if stc > 0 else np.nan
        row["Search Rate"]    = sr
        row["Search Rate SE"] = np.sqrt(sr * (1 - sr) / stc) if stc > 0 else np.nan

        # Hit rate
        hc = d["Hit Count"].sum()
        hr = hc / sc if sc > 0 else np.nan
        row["Hit Rate"]    = hr
        row["Hit Rate SE"] = np.sqrt(hr * (1 - hr) / sc) if sc > 0 else np.nan

        pooled_rows.append(row)

    pooled = pd.DataFrame(pooled_rows)
    pooled[race_col] = pd.Categorical(
        pooled[race_col], categories=RACE_ORDER, ordered=True
    )

    # ------------------------------------------------------------------
    # Panel definitions
    # ------------------------------------------------------------------
    panels = [
        {
            "y_col":   "Stops per 1,000",
            "title":   "(A) Stop Rate\n(per 1,000 residents)",
            "ylabel":  "Stops per 1,000 Residents",
            "is_rate": False,
            "se_col":  None,
            "pct":     False,
        },
        {
            "y_col":   "Searches per 1,000",
            "title":   "(B) Search Rate\n(per 1,000 residents)",
            "ylabel":  "Searches per 1,000 Residents",
            "is_rate": False,
            "se_col":  None,
            "pct":     False,
        },
        {
            "y_col":   "Search Rate",
            "title":   "(C) Conditional Search Rate\n(among those stopped)",
            "ylabel":  "Search Rate",
            "is_rate": True,
            "se_col":  "Search Rate SE",
            "pct":     True,
        },
        {
            "y_col":   "Hit Rate",
            "title":   "(D) Contraband Hit Rate\n(outcome test)",
            "ylabel":  "Hit Rate",
            "is_rate": True,
            "se_col":  "Hit Rate SE",
            "pct":     True,
        },
    ]

    SHORT_LABELS = ["Black", "Hispanic", "White", "Asian"]

    # ------------------------------------------------------------------
    # Figure: wide and compact for poster
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, panel in zip(axes, panels):
        y_col  = panel["y_col"]
        se_col = panel["se_col"]

        for i, race in enumerate(RACE_ORDER):
            row = pooled[pooled[race_col] == race]
            if row.empty or row[y_col].isna().all():
                continue

            val      = float(row[y_col].values[0])
            color    = COLOR_MAP[race]
            is_white = race == "White"
            mfc      = "white" if is_white else color
            mew      = 2.0    if is_white else 1.2

            # Error bars for rate panels
            if se_col and se_col in row.columns:
                se_val = float(row[se_col].values[0])
                ci     = se_val * 1.96 if pd.notna(se_val) else np.nan
                if pd.notna(ci) and ci > 0:
                    ax.errorbar(
                        i, val,
                        yerr=ci,
                        fmt="none",
                        ecolor=color,
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        alpha=0.9,
                        zorder=4,
                    )

            ax.plot(
                i, val,
                marker="o", markersize=11,
                color=color,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=mew,
                linestyle="none",
                zorder=5,
            )

        ax.set_title(panel["title"], fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(panel["ylabel"], fontsize=12)
        ax.set_xticks(range(len(RACE_ORDER)))
        ax.set_xticklabels(SHORT_LABELS, fontsize=12)
        ax.set_xlim(-0.6, len(RACE_ORDER) - 0.4)
        ax.tick_params(axis="x", length=3)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)

        if panel["pct"]:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

        # Modest top padding so the highest point doesn't crowd the title
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.14 * (yhi - ylo))

    # ------------------------------------------------------------------
    # Shared title
    # ------------------------------------------------------------------
    fig.suptitle(
        "Policing Disparities (Pooled 2022–2024)",
        fontsize=15, fontweight="bold", y=0.96,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return fig


def plot_wobbler_overall(df):
    """
    Single-panel wobbler figure for poster.
    Overall wobbler felony filing rate by race (dot plot with 95% CI).
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    Z_        = 1.96
    races     = list(RACE_ORDER)
    color_map = COLOR_MAP

    short_labels = {
        "Black/African American": "Black/\nAfrican American",
        "Hispanic/Latino":        "Hispanic/\nLatino",
        "White":                  "White",
        "Asian":                  "Asian",
    }

    wobblers = df[df["is_wobbler"]].copy()

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    summary = (
        wobblers.groupby(["race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Felony", "Misdemeanor"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["Total"]          = summary["Felony"] + summary["Misdemeanor"]
    summary["Felony Rate"]    = np.where(
        summary["Total"] > 0, summary["Felony"] / summary["Total"], np.nan
    )
    summary["Felony Rate SE"] = np.where(
        summary["Total"] > 0,
        np.sqrt(summary["Felony Rate"] * (1 - summary["Felony Rate"]) / summary["Total"]),
        np.nan,
    )
    summary = summary.reindex(races)

    rates = summary["Felony Rate"].to_numpy(dtype=float) * 100
    ses   = summary["Felony Rate SE"].to_numpy(dtype=float) * 100
    ns    = summary["Total"].to_numpy(dtype=float)
    errs  = ses * Z_
    x     = np.arange(len(races))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, race in enumerate(races):
        if not np.isfinite(rates[i]) or ns[i] == 0:
            continue

        is_white = race == "White"
        color    = color_map.get(race, "#7f7f7f")

        ax.errorbar(
            x[i], rates[i],
            yerr=errs[i],
            fmt="o",
            linestyle="none",
            markersize=13,
            markerfacecolor="white" if is_white else color,
            markeredgecolor=color,
            markeredgewidth=2.0 if is_white else 1.0,
            capsize=4, capthick=1.5, elinewidth=1.8,
            color=color,
            zorder=4 if not is_white else 3,
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    finite_tops = [rates[i] + errs[i] for i in range(len(rates))
                   if np.isfinite(rates[i]) and ns[i] > 0]
    max_top = max(finite_tops) if finite_tops else 60
    y_upper = max_top * 1.18
    ax.set_ylim(20, y_upper)
    offset  = max(y_upper * 0.03, 0.5)

    for i, race in enumerate(races):
        if not np.isfinite(rates[i]) or ns[i] == 0:
            continue
        ax.text(
            x[i], rates[i] + errs[i] + offset,
            f"{rates[i]:.1f}%\n(n={int(ns[i]):,})",
            ha="center", va="bottom", fontsize=13,
        )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax.set_title(
        "Wobbler Felony Filing Rates",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Canonical Race", fontsize=13, labelpad=8)
    ax.set_ylabel("Wobbler Charged as Felony (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_labels.get(r, r) for r in races],
        rotation=0, ha="center", fontsize=13,
    )
    ax.tick_params(labelsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(races) - 0.5)

    plt.tight_layout()

    return fig

def plot_weapons_felony_enhancement(enhancement_by_primary):
    """
    Single-panel figure for poster.
    Felony weapons enhancement rate by race (dot plot with 95% CI).
    Uses same data structure as plot_enhancement_rate_by_race_statute_category.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    Z  = 1.96
    df = enhancement_by_primary.copy()

    # ------------------------------------------------------------------
    # Resolve weapons category name
    # ------------------------------------------------------------------
    available_categories = df["primary_charge_category"].dropna().astype(str).unique().tolist()
    weapons_cat = next(
        (c for c in available_categories if c.strip().lower() == "weapons"), None
    )
    if weapons_cat is None:
        raise ValueError(
            f"Could not find 'Weapons' in primary_charge_category.\n"
            f"Available: {sorted(available_categories)}"
        )

    # ------------------------------------------------------------------
    # Summary stats — felony only
    # ------------------------------------------------------------------
    sub = df[
        (df["primary_charge_category"] == weapons_cat) &
        (df["primary_statute_level"] == "Felony")
    ].copy()

    summary = (
        sub.groupby("race_std", as_index=False)
           .agg({"Enhanced": "sum", "N": "sum"})
    )
    summary["Enhancement Rate"] = np.where(
        summary["N"] > 0,
        summary["Enhanced"] / summary["N"],
        np.nan,
    )
    summary["SE"] = np.where(
        summary["N"] > 0,
        np.sqrt(
            summary["Enhancement Rate"] *
            (1 - summary["Enhancement Rate"]) /
            summary["N"]
        ),
        np.nan,
    )
    summary["race_std"] = pd.Categorical(
        summary["race_std"], categories=RACE_ORDER, ordered=True
    )
    summary = summary.sort_values("race_std")

    races  = [r for r in RACE_ORDER if r in set(summary["race_std"].dropna().astype(str))]
    lookup = {str(row["race_std"]): row for _, row in summary.iterrows()}

    rates = np.array([100 * lookup[r]["Enhancement Rate"] if r in lookup else np.nan
                      for r in races], dtype=float)
    errs  = np.array([100 * lookup[r]["SE"] * Z if r in lookup else np.nan
                      for r in races], dtype=float)
    ns    = np.array([lookup[r]["N"] if r in lookup else 0
                      for r in races], dtype=float)

    x = np.arange(len(races))

    short_labels = {
        "Black/African American": "Black/\nAfrican American",
        "Hispanic/Latino":        "Hispanic/\nLatino",
        "White":                  "White",
        "Asian":                  "Asian",
    }

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, race in enumerate(races):
        if not np.isfinite(rates[i]) or ns[i] == 0:
            continue
        is_white = race == "White"
        color    = COLOR_MAP.get(race, "#7f7f7f")

        ax.errorbar(
            x[i], rates[i],
            yerr=errs[i],
            fmt="o",
            linestyle="none",
            markersize=13,
            markerfacecolor="white" if is_white else color,
            markeredgecolor=color,
            markeredgewidth=2.0 if is_white else 1.0,
            capsize=4, capthick=1.5, elinewidth=1.8,
            color=color,
            zorder=4 if not is_white else 3,
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    finite_tops = [rates[i] + errs[i] for i in range(len(rates))
                   if np.isfinite(rates[i]) and ns[i] > 0]
    max_top = max(finite_tops) if finite_tops else 20
    y_upper = max_top * 1.45
    ax.set_ylim(0, y_upper)
    offset  = max(y_upper * 0.03, 0.5)

    for i, race in enumerate(races):
        if not np.isfinite(rates[i]) or ns[i] == 0:
            continue
        ax.text(
            x[i], rates[i] + errs[i] + offset,
            f"{rates[i]:.1f}%\n(n={int(ns[i]):,})",
            ha="center", va="bottom", fontsize=13,
        )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax.set_title(
        "Felony Weapons Enhancement Rate",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Canonical Race", fontsize=13, labelpad=8)
    ax.set_ylabel("Enhancement Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_labels.get(r, r) for r in races],
        rotation=0, ha="center", fontsize=13,
    )
    ax.tick_params(labelsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(races) - 0.5)

    plt.tight_layout()

    return fig



def plot_agency_black_white_rates_poster(agency_df):

    d = agency_df.copy()

    d["Black_Hit_Rate_SE"] = np.where(
        d["Black_Search_Count"] > 0,
        np.sqrt(d["Black_Hit_Rate"] * (1 - d["Black_Hit_Rate"]) / d["Black_Search_Count"]),
        np.nan,
    )
    d["Black_CI_Width"] = d["Black_Hit_Rate_SE"] * 1.96 * 100
    d = d[(d["Black_Search_Count"] >= 30) & (d["Black_CI_Width"] <= 20)].reset_index(drop=True)

    sizes = 60 + 4.0 * np.sqrt(d["Avg_Search_Count"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.3})

    def draw_panel(ax, x_vals, y_vals, panel_title, xlabel, ylabel, outliers):

        colors = np.where(y_vals > x_vals, "#009E73", "#D55E00")

        ax.scatter(
            x_vals, y_vals,
            s=sizes, alpha=0.75, c=colors,
            edgecolor="white", linewidth=1.0, zorder=3,
        )

        max_val = max(np.nanmax(x_vals), np.nanmax(y_vals)) * 1.08
        ax.plot([0, max_val], [0, max_val],
                linestyle="--", linewidth=1.5, color="gray", alpha=0.5, zorder=2)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect("equal", adjustable="box")

        # Label only the White > Black outliers
        for _, row in d.iterrows():
            name = row["agency_name"]
            if name in outliers:
                label, (dx, dy) = outliers[name]
                ax.annotate(
                    label,
                    xy=(x_vals[row.name], y_vals[row.name]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, alpha=0.9, zorder=4,
                    ha="right" if dx < 0 else "left",
                )

        n_above = int((y_vals > x_vals).sum())
        n_below = int((y_vals <= x_vals).sum())
        n_label_below = "agency" if n_below == 1 else "agencies"
        n_label_above = "agency" if n_above == 1 else "agencies"
        ax.text(
            0.97, 0.05,
            f"{n_below} {n_label_below}: White > Black\n{n_above} {n_label_above}: Black > White",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="black", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.4", alpha=0.95),
        )

        ax.set_xlabel(xlabel, fontsize=9, fontweight="bold", labelpad=6)
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold", labelpad=6)
        ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    draw_panel(
        ax1,
        x_vals      = (d["White_Search_Rate"] * 100).to_numpy(),
        y_vals      = (d["Black_Search_Rate"]  * 100).to_numpy(),
        panel_title = "Conditional Search Rate by Agency",
        xlabel      = "White Search Rate (%)",
        ylabel      = "Black Search Rate (%)",
        outliers    = {
            "La Palma PD": ("La Palma PD", (5, 5)),
        },
    )

    draw_panel(
        ax2,
        x_vals      = (d["White_Hit_Rate"] * 100).to_numpy(),
        y_vals      = (d["Black_Hit_Rate"]  * 100).to_numpy(),
        panel_title = "Contraband Hit Rate by Agency",
        xlabel      = "White Hit Rate (%)",
        ylabel      = "Black Hit Rate (%)",
        outliers    = {
            "Laguna Beach PD": ("Laguna Beach PD", (-5, 5)),
            "Garden Grove PD": ("Garden Grove PD", (-5, 5)),
        },
    )

    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.97)

    return fig


def plot_risk_adjusted(risk_adjusted):
    """
    Dot plot comparing actual vs. predicted hit rates by race
    after controlling for observable stop characteristics.
    Hollow circle = predicted (risk-adjusted), filled = actual.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    # Match ordering and exclude Other to stay consistent with other figures
    races = [r for r in RACE_ORDER if r != "Other"]
    df = risk_adjusted.loc[races].copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(races))

    for i, race in enumerate(races):
        color    = COLOR_MAP[race]
        is_white = race == "White"
        actual   = df.loc[race, "actual_hit_rate"] * 100
        pred     = df.loc[race, "predicted_hit_rate"] * 100

        # Line connecting predicted to actual
        ax.plot([i, i], [pred, actual],
                color=color, linewidth=1.5, zorder=2)

        # Predicted: hollow circle
        ax.plot(i, pred,
                marker="o", markersize=11,
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.0,
                linestyle="none",
                zorder=4)

        # Actual: always filled
        ax.plot(i, actual,
                marker="o", markersize=11,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                linestyle="none",
                zorder=5)

        # Annotation — Asian on left, everyone else on right
        residual = df.loc[race, "residual"] * 100
        sign     = "+" if residual > 0 else ""
        ha       = "right" if race == "Asian" else "left"
        x_offset = -0.15 if race == "Asian" else 0.15
        ax.text(i + x_offset, actual,
                f"{sign}{residual:.1f}pp",
                va="center", ha=ha, fontsize=12, color=color)

    # Axes formatting
    ax.set_title(
        "Actual vs. Risk-Adjusted Hit Rate by Race (2024)",
        fontsize=13, fontweight="bold", pad=8
    )
    ax.set_ylabel("Hit Rate", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Black/\nAfrican American", "Hispanic/\nLatino", "White", "Asian"],
        fontsize=12
    )
    ax.set_xlim(-0.6, len(races) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)

    # Top padding
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.14 * (yhi - ylo))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=2.0,
               markersize=9, label="Predicted (risk-adjusted)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=9, label="Actual"),
    ]
    ax.legend(handles=legend_elements, fontsize=12,
              loc="upper left", borderpad=0.5)

    plt.tight_layout()
    return fig


def plot_kitchen_sink(race_results):
    """
    Vertical dot plot of odds ratios from kitchen sink logistic regression.
    Reference line at 1.0 = no difference from White.
    Points below 1.0 = lower odds of contraband recovery than White.
    """

    plt.rcParams.update({
        "font.size":       13,
        "axes.titlesize":  14,
        "axes.labelsize":  13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    race_order_ks = ["Black/African American", "Hispanic/Latino", "Asian"]
    df = race_results.reindex(race_order_ks).copy()

    ors   = np.exp(df["Coef."].values)
    lower = np.exp(df["[0.025"].values)
    upper = np.exp(df["0.975]"].values)
    x     = np.arange(len(race_order_ks))

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, race in enumerate(race_order_ks):
        color    = COLOR_MAP[race]
        is_white = race == "White"

        ax.errorbar(
            x[i], ors[i],
            yerr=[[ors[i] - lower[i]], [upper[i] - ors[i]]],
            fmt="o",
            linestyle="none",
            markersize=13,
            markerfacecolor="white" if is_white else color,
            markeredgecolor=color,
            markeredgewidth=2.0 if is_white else 1.0,
            capsize=4, capthick=1.5, elinewidth=1.8,
            color=color,
            zorder=4,
        )

    # Reference line at 1.0
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, zorder=1)

    # Set ylim so reference line is always visible
    ylo = min(lower) - 0.05
    yhi = 1.05
    ax.set_ylim(ylo, yhi)

    # Annotations above each point
    offset = (yhi - ylo) * 0.03
    for i, race in enumerate(race_order_ks):
        ax.text(
            x[i], upper[i] + offset,
            f"{ors[i]:.3f}\n({lower[i]:.3f}, {upper[i]:.3f})",
            ha="center", va="bottom", fontsize=12,
            color=COLOR_MAP[race],
        )

    # Reference label
    ax.text(0.02, 1.015, "White (reference)",
            fontsize=10, color="gray", ha="left",
            transform=ax.get_yaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(["Black/\nAfrican American", "Hispanic/\nLatino", "Asian"],
                       fontsize=13)
    ax.set_ylabel("Odds Ratio (relative to White)", fontsize=13)
    ax.set_title(
        "Odds of Contraband Recovery Relative to White\n(Controlling for Observable Stop Characteristics)",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.set_xlim(-0.5, len(race_order_ks) - 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()