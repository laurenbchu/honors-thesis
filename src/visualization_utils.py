import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

from matplotlib.ticker import FuncFormatter

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
    import matplotlib.lines as mlines

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
    import matplotlib.lines as mlines

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
    import matplotlib.lines as mlines

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


def plot_agency_black_white_hit_rates(agency_hit_df):
    """
    Publication-ready scatter plot comparing White and Black hit rates by agency.

    Exclusion criteria (Option 3):
    - Agencies with fewer than 30 Black searches excluded
    - Agencies with Black hit rate 95% CI > ±20 percentage points excluded

    Improvements:
    - Legend uses scatter handles (circles) instead of patches (squares)
    - Dot size scaled by Avg_Search_Count
    - Above/below diagonal counts annotated in corner
    - Subtitle noting dot size meaning
    - All font sizes >= 13
    - UC Irvine PD excluded via criteria above
    """
    from adjustText import adjust_text

    d = agency_hit_df.copy()

    # ------------------------------------------------------------------
    # Compute Black CI width for exclusion
    # ------------------------------------------------------------------
    d["Black_Hit_Rate_SE"] = np.where(
        d["Black_Search_Count"] > 0,
        np.sqrt(
            d["Black_Hit_Rate"] * (1 - d["Black_Hit_Rate"]) / d["Black_Search_Count"]
        ),
        np.nan,
    )
    d["Black_CI_Width"] = d["Black_Hit_Rate_SE"] * 1.96 * 100  # in percentage points

    # ------------------------------------------------------------------
    # Apply exclusion criteria (Option 3): n >= 30 AND CI <= ±20pp
    # ------------------------------------------------------------------
    n_total = len(d)
    d = d[
        (d["Black_Search_Count"] >= 30) &
        (d["Black_CI_Width"] <= 20)
    ].reset_index(drop=True)
    n_excluded = n_total - len(d)

    x = d["White_Hit_Rate"] * 100
    y = d["Black_Hit_Rate"] * 100

    # ------------------------------------------------------------------
    # Dot size scaled by average search count
    # ------------------------------------------------------------------
    sizes = 80 + 6.0 * np.sqrt(d["Avg_Search_Count"])

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    colors = np.where(y > x, "#009E73", "#D55E00")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(
        x, y,
        s=sizes,
        alpha=0.75,
        c=colors,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # Parity line
    max_val = max(5, np.nanmax(np.concatenate([x.to_numpy(), y.to_numpy()])) + 5)
    min_val = max(0, np.nanmin(np.concatenate([x.to_numpy(), y.to_numpy()])) - 2)

    ax.plot(
        [min_val, max_val], [min_val, max_val],
        linestyle="--", linewidth=2.5,
        color="gray", alpha=0.5,
        zorder=2,
    )

    # ------------------------------------------------------------------
    # Labels with adjustText
    # ------------------------------------------------------------------
    texts = []
    for _, row in d.iterrows():
        xi = row["White_Hit_Rate"] * 100
        yi = row["Black_Hit_Rate"] * 100
        texts.append(
            ax.text(xi, yi, row["agency_name"],
                    fontsize=13, alpha=0.9, zorder=4,
                    ha="center", va="center")
        )

    adjust_text(
        texts,
        x=x.values,
        y=y.values,
        ax=ax,
        expand_points=(1.8, 1.8),
        expand_text=(1.5, 1.5),
        force_points=(0.4, 0.4),
        force_text=(0.8, 0.8),
        lim=500,
    )

    # ------------------------------------------------------------------
    # Above / below diagonal counts
    # ------------------------------------------------------------------
    n_above = int((y > x).sum())
    n_below = int((y <= x).sum())

    ax.text(
        0.97, 0.05,
        f"{n_below} agencies: White hit rate > Black\n"
        f"{n_above}  agencies: Black hit rate > White",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=13, color="dimgray",
        linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="0.8", alpha=0.85),
    )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax.set_xlabel("White Hit Rate (%)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Black/African American Hit Rate (%)", fontsize=14,
                  fontweight="bold", labelpad=10)
    ax.set_title(
        "Agency-Level Comparison: White vs. Black Hit Rates\n"
        "(Outcome Test for Searches)",
        fontsize=16, fontweight="bold", pad=20,
    )

    # Subtitle noting dot size
    ax.text(
        0.5, 0.995,
        "Dot size proportional to average annual searches",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=13, color="dimgray", style="italic",
    )

    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    # ------------------------------------------------------------------
    # Legend: scatter handles (circles) instead of patches (squares)
    # ------------------------------------------------------------------
    legend_handles = [
        plt.Line2D([0], [0], linestyle="--", linewidth=2.5,
                   color="gray", alpha=0.5, label="Equal Hit Rates"),
        ax.scatter([], [], s=100, c="#009E73", alpha=0.75,
                   edgecolor="white", linewidth=1.5,
                   label="Black Hit Rate > White"),
        ax.scatter([], [], s=100, c="#D55E00", alpha=0.75,
                   edgecolor="white", linewidth=1.5,
                   label="White Hit Rate > Black"),
        # Size reference
        ax.scatter([], [], s=80 + 6.0 * np.sqrt(50),   c="gray", alpha=0.5,
                   edgecolor="white", linewidth=1.5, label="Avg. searches ≈ 50"),
        ax.scatter([], [], s=80 + 6.0 * np.sqrt(200),  c="gray", alpha=0.5,
                   edgecolor="white", linewidth=1.5, label="Avg. searches ≈ 200"),
        ax.scatter([], [], s=80 + 6.0 * np.sqrt(500),  c="gray", alpha=0.5,
                   edgecolor="white", linewidth=1.5, label="Avg. searches ≈ 500"),
    ]

    ax.legend(
        handles=legend_handles,
        fontsize=13,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        loc="upper left",
        labelspacing=0.8,
    )

    plt.tight_layout()
    return fig



def create_combined_sensitivity_visualization(baseline_df, mixed_df, multiperson_df):
    """
    Create a publication-ready two-panel sensitivity figure comparing:
    - Baseline
    - Mixed classification
    - Multiperson stops

    Panel A: Conditional Search Rate
    Panel B: Contraband Hit Rate
    """

    latest_year = baseline_df["Year"].max()

    baseline = baseline_df[baseline_df["Year"] == latest_year].copy()
    mixed = mixed_df[mixed_df["Year"] == latest_year].copy()
    multiperson = multiperson_df[multiperson_df["Year"] == latest_year].copy()

    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White",
        "Asian"
    ]

    baseline = baseline.set_index("Perceived Race").reindex(race_order).reset_index()
    mixed = mixed.set_index("Perceived Race").reindex(race_order).reset_index()
    multiperson = multiperson.set_index("Perceived Race").reindex(race_order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(
        f"Sensitivity Analysis: Baseline vs. Mixed Classification vs. Multiperson Stops ({latest_year})",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    x = np.arange(len(race_order))
    width = 0.24

    baseline_color = SENSITIVITY_COLOR_MAP["Baseline"]
    mixed_color = SENSITIVITY_COLOR_MAP["Mixed"]
    multiperson_color = SENSITIVITY_COLOR_MAP["Multiperson"]

    def darken_color(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

    # --------------------
    # Panel A: Search Rate
    # --------------------
    base_search = baseline["Search Rate"] * 100
    mixed_search = mixed["Search Rate"] * 100
    multi_search = multiperson["Search Rate"] * 100

    base_search_se = baseline["Search Rate SE"] * 100 * 1.96
    mixed_search_se = mixed["Search Rate SE"] * 100 * 1.96
    multi_search_se = multiperson["Search Rate SE"] * 100 * 1.96

    x_base = x - width
    x_mixed = x
    x_multi = x + width

    axes[0].bar(
        x_base,
        base_search,
        width,
        label="Baseline",
        color=baseline_color,
        alpha=0.8,
        yerr=base_search_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(baseline_color)}
    )

    axes[0].bar(
        x_mixed,
        mixed_search,
        width,
        label="Mixed",
        color=mixed_color,
        alpha=0.8,
        yerr=mixed_search_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(mixed_color)}
    )

    axes[0].bar(
        x_multi,
        multi_search,
        width,
        label="Multiperson",
        color=multiperson_color,
        alpha=0.8,
        yerr=multi_search_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(multiperson_color)}
    )

    max_y_search = 0
    for i, xi in enumerate(x):
        vals = [
            (x_base[i], base_search.iloc[i], base_search_se.iloc[i], baseline.iloc[i]["Search Count"]),
            (x_mixed[i], mixed_search.iloc[i], mixed_search_se.iloc[i], mixed.iloc[i]["Search Count"]),
            (x_multi[i], multi_search.iloc[i], multi_search_se.iloc[i], multiperson.iloc[i]["Search Count"]),
        ]

        for xpos, val, err, n in vals:
            if np.isfinite(val):
                axes[0].text(
                    xpos, val + err + 0.3,
                    f"{val:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=7, fontweight="bold"
                )
                max_y_search = max(max_y_search, val + err + 3)

    axes[0].set_title("(A) Conditional Search Rate", fontsize=13, fontweight="bold", pad=15)
    axes[0].set_ylabel("Search Rate (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Perceived Race", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(race_order, rotation=0, ha="center", fontsize=10)
    axes[0].legend(fontsize=10, frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle=":", axis="y")
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0, max_y_search * 1.02)

    # -----------------
    # Panel B: Hit Rate
    # -----------------
    base_hit = baseline["Hit Rate"] * 100
    mixed_hit = mixed["Hit Rate"] * 100
    multi_hit = multiperson["Hit Rate"] * 100

    base_hit_se = baseline["Hit Rate SE"] * 100 * 1.96
    mixed_hit_se = mixed["Hit Rate SE"] * 100 * 1.96
    multi_hit_se = multiperson["Hit Rate SE"] * 100 * 1.96

    axes[1].bar(
        x_base,
        base_hit,
        width,
        label="Baseline",
        color=baseline_color,
        alpha=0.8,
        yerr=base_hit_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(baseline_color)}
    )

    axes[1].bar(
        x_mixed,
        mixed_hit,
        width,
        label="Mixed",
        color=mixed_color,
        alpha=0.8,
        yerr=mixed_hit_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(mixed_color)}
    )

    axes[1].bar(
        x_multi,
        multi_hit,
        width,
        label="Multiperson",
        color=multiperson_color,
        alpha=0.8,
        yerr=multi_hit_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(multiperson_color)}
    )

    max_y_hit = 0
    for i, xi in enumerate(x):
        vals = [
            (x_base[i], base_hit.iloc[i], base_hit_se.iloc[i], baseline.iloc[i]["Hit Count"]),
            (x_mixed[i], mixed_hit.iloc[i], mixed_hit_se.iloc[i], mixed.iloc[i]["Hit Count"]),
            (x_multi[i], multi_hit.iloc[i], multi_hit_se.iloc[i], multiperson.iloc[i]["Hit Count"]),
        ]

        for xpos, val, err, n in vals:
            if np.isfinite(val):
                axes[1].text(
                    xpos, val + err + 0.5,
                    f"{val:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=7, fontweight="bold"
                )
                max_y_hit = max(max_y_hit, val + err + 5)

    axes[1].set_title("(B) Contraband Hit Rate", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_ylabel("Hit Rate (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Perceived Race", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(race_order, rotation=0, ha="center", fontsize=10)
    axes[1].legend(fontsize=10, frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle=":", axis="y")
    axes[1].set_axisbelow(True)
    axes[1].set_ylim(0, max_y_hit * 1.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
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
      - 'dui': 1x2 figure
          DUI (Misdemeanor, Felony)

    Bars are colored by race using COLOR_MAP and include 95% CI error bars.
    """

    df = enhancement_by_primary.copy()

    # --- resolve category names exactly as they appear in the data ---
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
    dui_cat = resolve_category_name("DUI")

    # Optional filtering if user passes a subset
    if top_categories is not None:
        keep = {str(x).strip().lower() for x in top_categories}
        requested = {
            assault_cat.strip().lower(),
            weapons_cat.strip().lower(),
            dui_cat.strip().lower()
        }
        use_requested = requested.intersection(keep)
        if use_requested:
            df = df[df["primary_charge_category"].str.strip().str.lower().isin(use_requested)].copy()

    def darken_color(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

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
            np.nan
        )
        summary["SE"] = np.where(
            summary["N"] > 0,
            np.sqrt(summary["Enhancement Rate"] * (1 - summary["Enhancement Rate"]) / summary["N"]),
            np.nan
        )

        summary["race_std"] = pd.Categorical(
            summary["race_std"],
            categories=RACE_ORDER,
            ordered=True
        )
        summary = summary.sort_values(["race_std", "primary_statute_level"])

        present_races = [
            r for r in RACE_ORDER
            if r in set(summary["race_std"].dropna().astype(str))
        ]

        return summary, present_races

    def aligned_arrays(summary, races, statute_level):
        sub = summary[summary["primary_statute_level"] == statute_level].copy()
        lookup = {str(row["race_std"]): row for _, row in sub.iterrows()}

        rates = np.array([
            100 * lookup[r]["Enhancement Rate"] if r in lookup else np.nan
            for r in races
        ], dtype=float)

        ses = np.array([
            100 * lookup[r]["SE"] if r in lookup else np.nan
            for r in races
        ], dtype=float)

        ns = np.array([
            lookup[r]["N"] if r in lookup else 0
            for r in races
        ], dtype=float)

        errs = ses * Z
        return rates, errs, ns

    def plot_panel(ax, rates, errs, ns, races, panel_title, show_ylabel=False):
        x = np.arange(len(races))
        colors = [COLOR_MAP.get(r, "#7f7f7f") for r in races]

        heights = np.where(np.isfinite(rates), rates, 0.0)
        bars = ax.bar(
            x,
            heights,
            color=colors,
            alpha=0.85,
            edgecolor="none",
            linewidth=0
        )

        # Hide truly missing bars rather than showing a visible zero bar
        for bar, rate in zip(bars, rates):
            if not np.isfinite(rate):
                bar.set_alpha(0.0)

        for i, (rate, err, col, n) in enumerate(zip(rates, errs, colors, ns)):
            if np.isfinite(rate) and n > 0:
                ax.errorbar(
                    i, rate,
                    yerr=err,
                    fmt="none",
                    ecolor=darken_color(col),
                    alpha=0.4,
                    capsize=5,
                    capthick=1.5,
                    linewidth=1.5
                )

        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Canonical Race", fontsize=12, fontweight="bold")
        if show_ylabel:
            ax.set_ylabel("Enhancement Rate (%)", fontsize=12, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(races, rotation=0, ha="center", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)

        max_y = 0
        for i, (rate, err, n) in enumerate(zip(rates, errs, ns)):
            if np.isfinite(rate) and n > 0:
                y_pos = rate + err + 0.3
                ax.text(
                    i, y_pos,
                    f"{rate:.1f}%\n(n={int(n):,})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )
                max_y = max(max_y, y_pos + 2)

        return max_y

    def make_two_panel_category_figure(category_name, figure_title):
        summary, races = summarize_category(category_name)
        if summary is None or not races:
            return None

        mis_rates, mis_errs, mis_n = aligned_arrays(summary, races, "Misdemeanor")
        fel_rates, fel_errs, fel_n = aligned_arrays(summary, races, "Felony")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"wspace": 0.3})

        max_y1 = plot_panel(
            ax1, mis_rates, mis_errs, mis_n, races,
            "(A) Misdemeanor Charges",
            show_ylabel=True
        )
        max_y2 = plot_panel(
            ax2, fel_rates, fel_errs, fel_n, races,
            "(B) Felony Charges",
            show_ylabel=False
        )

        ymax = min(100, max(max_y1, max_y2) + 2 if max(max_y1, max_y2) > 0 else 100)
        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)

        fig.suptitle(
            figure_title,
            fontsize=15,
            fontweight="bold",
            y=0.98
        )
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.17, wspace=0.3)

        return fig

    def make_four_panel_combined_figure(cat1, cat2, figure_title):
        summary1, races1 = summarize_category(cat1)
        summary2, races2 = summarize_category(cat2)

        if summary1 is None and summary2 is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(18, 13), gridspec_kw={"wspace": 0.3, "hspace": 0.4})
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Row 1: Assault/Violence
        if summary1 is not None and races1:
            mis_rates, mis_errs, mis_n = aligned_arrays(summary1, races1, "Misdemeanor")
            fel_rates, fel_errs, fel_n = aligned_arrays(summary1, races1, "Felony")

            max_y1 = plot_panel(
                ax1, mis_rates, mis_errs, mis_n, races1,
                f"(A) {cat1}: Misdemeanor",
                show_ylabel=True
            )
            max_y2 = plot_panel(
                ax2, fel_rates, fel_errs, fel_n, races1,
                f"(B) {cat1}: Felony",
                show_ylabel=False
            )

            ymax_row1 = min(100, max(max_y1, max_y2) + 2 if max(max_y1, max_y2) > 0 else 100)
            ax1.set_ylim(0, ymax_row1)
            ax2.set_ylim(0, ymax_row1)
        else:
            ax1.axis("off")
            ax2.axis("off")

        # Row 2: Weapons
        if summary2 is not None and races2:
            mis_rates, mis_errs, mis_n = aligned_arrays(summary2, races2, "Misdemeanor")
            fel_rates, fel_errs, fel_n = aligned_arrays(summary2, races2, "Felony")

            max_y3 = plot_panel(
                ax3, mis_rates, mis_errs, mis_n, races2,
                f"(C) {cat2}: Misdemeanor",
                show_ylabel=True
            )
            max_y4 = plot_panel(
                ax4, fel_rates, fel_errs, fel_n, races2,
                f"(D) {cat2}: Felony",
                show_ylabel=False
            )

            ymax_row2 = min(100, max(max_y3, max_y4) + 2 if max(max_y3, max_y4) > 0 else 100)
            ax3.set_ylim(0, ymax_row2)
            ax4.set_ylim(0, ymax_row2)
        else:
            ax3.axis("off")
            ax4.axis("off")

        fig.suptitle(
            figure_title,
            fontsize=16,
            fontweight="bold",
            y=0.98
        )
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.3, hspace=0.4)

        return fig

    figs = {}

    figs["assault_violence_weapons"] = make_four_panel_combined_figure(
        assault_cat,
        weapons_cat,
        "Enhancement Charge Rate by Race and Statute Level:\nAssault/Violence and Weapons"
    )

    figs["dui"] = make_two_panel_category_figure(
        dui_cat,
        "Enhancement Charge Rate by Race and Statute Level:\nDUI"
    )

    return figs



def plot_wobbler_combined(df, top_categories=None, sort_by="rate_overall"):
    """
    Create a combined two-panel figure for wobbler charges:
    - Panel A: Overall wobbler felony filing rate by race
    - Panel B: Cleveland dot plot by charge category and race
    """
    
    Z_ = globals().get("Z", 1.96)
    races = list(globals().get("RACE_ORDER", ["Black/African American", "Hispanic/Latino", "White", "Asian"]))
    color_map = globals().get("COLOR_MAP", {})
    
    # Helper: darken a color for error bars
    def darken(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)
    
    # Calculate figure height based on number of categories for Panel B
    wobblers = df[df["is_wobbler"]].copy()
    
    # Determine categories for Panel B
    if top_categories is not None:
        g = wobblers.groupby(["charge_category", "race_std", "statute_level"]).size().unstack(fill_value=0).reset_index()
        for col in ["Felony", "Misdemeanor"]:
            if col not in g.columns:
                g[col] = 0
        g["Total"] = g["Felony"] + g["Misdemeanor"]
        existing = set(g["charge_category"].unique())
        categories = [c for c in top_categories if c in existing]
        n_categories = len(categories)
    else:
        g = wobblers.groupby(["charge_category", "race_std", "statute_level"]).size().unstack(fill_value=0).reset_index()
        for col in ["Felony", "Misdemeanor"]:
            if col not in g.columns:
                g[col] = 0
        g["Total"] = g["Felony"] + g["Misdemeanor"]
        n_categories = len(g["charge_category"].unique())
    
    # Create figure with side-by-side panels
    fig_height = max(8, 0.35 * n_categories + 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, fig_height), gridspec_kw={'wspace': 0.30, 'width_ratios': [1, 1.5]})
    
    # ==========================================
    # PANEL A: Overall Wobbler Felony Rate
    # ==========================================
    summary = (
        wobblers.groupby(["race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Felony", "Misdemeanor"]:
        if col not in summary.columns:
            summary[col] = 0
    
    summary["Total"] = summary["Felony"] + summary["Misdemeanor"]
    summary["Felony Rate"] = np.where(summary["Total"] > 0, summary["Felony"] / summary["Total"], np.nan)
    summary["Felony Rate SE"] = np.where(
        summary["Total"] > 0,
        np.sqrt(summary["Felony Rate"] * (1 - summary["Felony Rate"]) / summary["Total"]),
        np.nan,
    )
    
    summary = summary.reindex(races)
    
    races_a = summary.index.astype(str).tolist()
    rates = (summary["Felony Rate"].to_numpy(dtype=float) * 100)
    ses = (summary["Felony Rate SE"].to_numpy(dtype=float) * 100)
    ns = summary["Total"].to_numpy(dtype=float)
    errs = ses * Z_
    
    colors = [color_map.get(r, "#7f7f7f") for r in races_a]
    x = np.arange(len(races_a))
    
    ax1.bar(x, np.nan_to_num(rates, nan=0.0), color=colors, alpha=0.85, edgecolor="none", linewidth=0)
    
    for i, (rate, err, col, n) in enumerate(zip(rates, errs, colors, ns)):
        if np.isfinite(rate) and n > 0:
            ax1.errorbar(i, rate, yerr=err, fmt="none", ecolor=darken(col),
                        alpha=0.4, capsize=5, capthick=1.5, linewidth=1.5)
    
    ax1.set_title("(A) Overall Wobbler Felony Filing Rate by Race",
                 fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Canonical Race", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Wobbler Charged as Felony (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(races_a, fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax1.set_axisbelow(True)
    
    max_y = 0
    for i, (rate, err, n) in enumerate(zip(rates, errs, ns)):
        if np.isfinite(rate) and n > 0:
            y_pos = rate + err + 0.7
            ax1.text(i, y_pos, f"{rate:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            max_y = max(max_y, y_pos + 2)
    
    pad = 2
    ymax = max_y if max_y > 0 else 100
    ax1.set_ylim(0, min(100, ymax + pad))
    
    # ==========================================
    # PANEL B: Cleveland Dot Plot by Category
    # ==========================================
    g = (
        wobblers.groupby(["charge_category", "race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    
    for col in ["Felony", "Misdemeanor"]:
        if col not in g.columns:
            g[col] = 0
    
    g["Total"] = g["Felony"] + g["Misdemeanor"]
    g["Felony Rate"] = np.where(g["Total"] > 0, g["Felony"] / g["Total"], np.nan)
    g["SE"] = np.where(
        g["Total"] > 0,
        np.sqrt(g["Felony Rate"] * (1 - g["Felony Rate"]) / g["Total"]),
        np.nan,
    )
    g["lo"] = (g["Felony Rate"] - Z_ * g["SE"]) * 100
    g["hi"] = (g["Felony Rate"] + Z_ * g["SE"]) * 100
    g["x"] = g["Felony Rate"] * 100
    
    g = g[g["race_std"].isin(races)].copy()
    g["race_std"] = pd.Categorical(g["race_std"], categories=races, ordered=True)
    
    # Determine and sort categories
    if top_categories is not None:
        categories = list(top_categories)
        existing = set(g["charge_category"].unique())
        categories = [c for c in categories if c in existing]
        g = g[g["charge_category"].isin(categories)].copy()
    else:
        cat_totals = g.groupby("charge_category")["Total"].sum().sort_values(ascending=False)
        
        overall_rate = g.groupby("charge_category")[["Felony", "Misdemeanor"]].sum()
        overall_rate["Total"] = overall_rate["Felony"] + overall_rate["Misdemeanor"]
        overall_rate["Rate"] = np.where(
            overall_rate["Total"] > 0,
            overall_rate["Felony"] / overall_rate["Total"],
            np.nan
        )
        
        if sort_by == "total":
            categories = cat_totals.index.tolist()
        elif sort_by == "rate_overall":
            categories = overall_rate["Rate"].sort_values(ascending=False).index.tolist()
        elif sort_by == "rate_black":
            blk = g[g["race_std"] == races[0]].set_index("charge_category")["Felony Rate"]
            categories = blk.sort_values(ascending=False).index.tolist()
        else:
            categories = sorted(cat_totals.index.tolist())
    
    g = g[g["charge_category"].isin(categories)].copy()
    categories = list(categories)
    
    y_base = np.arange(len(categories))
    offsets = np.linspace(-0.32, 0.32, num=len(races))
    
    ax2.grid(axis="x", alpha=0.15)
    ax2.set_axisbelow(True)
    
    for i, y0 in enumerate(y_base):
        if i % 2 == 0:
            ax2.axhspan(y0 - 0.5, y0 + 0.5, color="0.95", alpha=1.0, zorder=0)
    
    for r_i, race in enumerate(races):
        sub = g[g["race_std"] == race].copy()
        sub = sub.set_index("charge_category").reindex(categories)
        y = y_base + offsets[r_i]
        
        x_vals = sub["x"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        n = sub["Total"].to_numpy(dtype=float)
        
        for yi, xi, l, h, nn in zip(y, x_vals, lo, hi, n):
            if np.isfinite(xi) and nn > 0 and np.isfinite(l) and np.isfinite(h):
                ax2.hlines(
                    yi, l, h,
                    linewidth=1.2,
                    alpha=0.18,
                    color=color_map.get(race, "#7f7f7f"),
                    zorder=2
                )
        
        if race == "White":
            ax2.scatter(
                x_vals, y,
                s=95,
                label=race,
                facecolors="none",
                edgecolors=color_map.get(race, "#7f7f7f"),
                linewidths=1.6,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax2.scatter(
                x_vals, y,
                s=95,
                label=race,
                color=color_map.get(race, "#7f7f7f"),
                edgecolors="none",
                alpha=0.95,
                zorder=3,
            )
    
    for y in np.arange(-0.5, len(categories), 1):
        ax2.axhline(y, color="gray", linewidth=0.8, alpha=0.35, zorder=1)
    
    ax2.set_yticks(y_base)
    ax2.set_yticklabels(categories, fontsize=11)
    ax2.invert_yaxis()
    
    ax2.set_xlabel("Wobbler Charged as Felony (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Charge Category", fontsize=11, fontweight="bold")
    ax2.set_title(
        "(B) Felony Charging Rate by Charge Category and Race",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    
    xmin = np.nanmin(g["lo"].to_numpy(dtype=float))
    xmax = np.nanmax(g["hi"].to_numpy(dtype=float))
    if np.isfinite(xmin) and np.isfinite(xmax):
        ax2.set_xlim(max(0, xmin - 2), min(100, xmax + 2))
    else:
        ax2.set_xlim(0, 100)
    
    ax2.legend(
        title="Canonical Race",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        loc="upper left"
    )
    
    # Overall title
    fig.suptitle(
        "Wobbler Charges: Felony Filing Rates by Race and Charge Category\n(with 95% Confidence Intervals)",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])

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