# TODO: add row of plots for the frame 2 frame geodesic error over time?

# --- Imports ---------------------------------------------------------------
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu, spearmanr  # spearmanr unused but kept
from matplotlib.patches import Patch
from scipy.stats import norm  # for z from two-sided p


# --- Matplotlib style & colors --------------------------------------------
plt.style.use(['tableau-colorblind10', 'ediheadytrack.mplstyle'])
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def _series_color(i: int):
    """Match original color choices for series 0/1/2."""
    if i == 0:
        return _colors[0]
    if i == 1:
        return _colors[1]
    return 'tab:green'

# --- Load data -------------------------------------------------------------
with open('../../../Data/predictions.jsonl', 'r') as f:
    df = pd.DataFrame(json.loads(line) for line in f)

# Find the lowest values of geodesic error and their corresponding impact ID and frame number
df['frame_idx'] = df['geodesic_error'].apply(lambda lst: list(range(len(lst))) if isinstance(lst, list) else [])
df_exploded = df.explode(['geodesic_error', 'frame_idx'], ignore_index=True)
df_exploded['geodesic_error'] = df_exploded['geodesic_error'].astype(float)

# Filter rows with finite geodesic error values
df_filtered = df_exploded[np.isfinite(df_exploded['geodesic_error'])]

# Sort by geodesic error to find the lowest values
lowest_values = df_filtered.nsmallest(10, 'geodesic_error')[['Impact ID', 'frame_idx', 'geodesic_error', 'Data source']]

# Print the results
print(lowest_values)

# --- Constants -------------------------------------------------------------
sources = ['6DRepNet360', 'VGGHeads', 'SAM-3D']

# --- Utilities -------------------------------------------------------------
def explode_series_for_camera(sub_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Explode a list-valued radians column `value_col` to per-frame rows with degrees.
    sub_df should include optional PAIR_KEY + ['Camera'] or any grouping cols used by caller.
    Returns a DataFrame containing ['frame_idx', 'val_deg'] among the original columns.
    """
    out = sub_df.copy()
    out['frame_idx'] = out[value_col].apply(lambda lst: list(range(len(lst))) if isinstance(lst, list) else [])
    out = out.explode([value_col, 'frame_idx'], ignore_index=True)
    out[value_col] = pd.to_numeric(out[value_col], errors='coerce')
    out['val_deg'] = np.degrees(out[value_col].astype(float))
    return out

def _annotate_pairwise_single_star(ax, positions, idx_pairs, p_values, data_min, data_max, alpha):
    """Draw single-star significance bars above paired groups when p<alpha."""
    for (i1, i2), p in zip(idx_pairs, p_values):
        if p < alpha:
            x1, x2 = positions[i1], positions[i2]
            y = data_max * 1.15
            h = data_max * 0.05
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='k')
            ax.text((x1 + x2) * 0.5, y + h, '*', ha='center', va='bottom', color='k')

def print_wilcoxon_summary(a_vals, b_vals, left_label='F', right_label='S', metric_name='GE'):
    """Print summary of paired Wilcoxon result with direction, median diff, and effect size."""
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    m = np.isfinite(a_vals) & np.isfinite(b_vals)
    a_vals, b_vals = a_vals[m], b_vals[m]
    if a_vals.size == 0:
        print(f"No valid pairs for {metric_name} ({left_label} vs {right_label})")
        return

    diff = b_vals - a_vals
    med_diff = np.nanmedian(diff)
    res = wilcoxon(a_vals, b_vals, zero_method='pratt', alternative='two-sided')
    p = res.pvalue

    # derive effect size r = z/sqrt(N)
    nz = np.count_nonzero(diff != 0)
    sign = np.sign(med_diff) if med_diff != 0 else 1
    z = norm.isf(p / 2.0) * sign
    r = z / np.sqrt(nz) if nz > 0 else np.nan

    if med_diff > 0:
        direction = f"{right_label}>{left_label}"
    elif med_diff < 0:
        direction = f"{left_label}>{right_label}"
    else:
        direction = f"{left_label}≈{right_label}"

    print(f"{metric_name}: {direction}, median Δ={med_diff:.2f}°, r={r:.2f}, p={p:.4f}")


from scipy.stats import mannwhitneyu

def print_mwu_summary(a_vals, b_vals, a_label='L', b_label='R', metric_name='GE', unit='°'):
    """
    Print a concise Mann–Whitney U summary with direction, median diff, and effect sizes.
    a_vals = group A values (e.g., L), b_vals = group B values (e.g., R)
    Direction is based on median difference and sign of Cliff's delta.
    """
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    m_a = np.isfinite(a_vals)
    m_b = np.isfinite(b_vals)
    a_vals, b_vals = a_vals[m_a], b_vals[m_b]

    if a_vals.size == 0 or b_vals.size == 0:
        print(f"{metric_name}: No valid data for {a_label} vs {b_label}")
        return

    # Mann–Whitney U (two-sided)
    res = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
    p = res.pvalue

    # Effect sizes
    n1, n2 = len(a_vals), len(b_vals)
    auc = res.statistic / (n1 * n2)              # P(A > B)
    cliffs_delta = 2 * auc - 1                   # δ in [-1, 1]; >0 => a_vals tend larger

    # Direction & magnitude (robust)
    med_diff = np.nanmedian(b_vals) - np.nanmedian(a_vals)  # >0 => b_label larger
    if med_diff > 0:
        direction = f"{b_label}>{a_label}"
    elif med_diff < 0:
        direction = f"{a_label}>{b_label}"
    else:
        # if medians tie, fall back to δ sign for direction
        if cliffs_delta > 0:
            direction = f"{a_label}>{b_label}"
        elif cliffs_delta < 0:
            direction = f"{b_label}>{a_label}"
        else:
            direction = f"{a_label}≈{b_label}"

    # Optional: common-language effect percentages
    p_a_gt_b = auc              # P(A > B)
    p_b_gt_a = 1.0 - auc        # P(B > A)

    print(
        f"{metric_name}: {direction}, median Δ={med_diff:.2f}{unit}, "
        f"δ={cliffs_delta:.2f} (AUC={auc:.2f}; P[{a_label}>{b_label}]={p_a_gt_b:.2f}), p={p:.4f}"
    )

# ===========================================================================
# 1) Prepare series (identity GE and incremental GE)
# ===========================================================================
geodesic_error_series = {}
ge_ident_pred_series = {}
ge_ident_gt_series   = {}

# NEW: incremental series
inc_pred_series = {}
inc_gt_series   = {}
inc_dev_series  = {}

for i, s in enumerate(sources):
    
    mask = (df['Data source'] == s) & (df['filtered'])

    # --- Identity geodesic error (Prediction vs Ground Truth)
    ge_raw = df.loc[mask, 'geodesic_error'].dropna().explode()
    ge_raw = pd.to_numeric(ge_raw, errors='coerce').astype(float)
    ge_deg = np.degrees(ge_raw.to_numpy())
    geodesic_error_series[s] = ge_deg.tolist()

    pred_raw = df.loc[mask, 'geodesic_to_identity_prediction'].dropna().explode()
    pred_raw = pd.to_numeric(pred_raw, errors='coerce').astype(float)
    ge_ident_pred_series[s] = np.degrees(pred_raw.to_numpy()).tolist()

    gt_raw = df.loc[mask, 'geodesic_to_identity_ground_truth'].dropna().explode()
    gt_raw = pd.to_numeric(gt_raw, errors='coerce').astype(float)
    ge_ident_gt_series[s] = np.degrees(gt_raw.to_numpy()).tolist()

    # --- Incremental (frame-to-frame)
    incp_raw = df.loc[mask, 'frame_to_frame_prediction'].dropna().explode()
    incp_raw = pd.to_numeric(incp_raw, errors='coerce').astype(float)
    inc_pred_series[s] = np.degrees(incp_raw.to_numpy()).tolist()

    incg_raw = df.loc[mask, 'frame_to_frame_ground_truth'].dropna().explode()
    incg_raw = pd.to_numeric(incg_raw, errors='coerce').astype(float)
    inc_gt_series[s] = np.degrees(incg_raw.to_numpy()).tolist()

    incd_raw = df.loc[mask, 'frame_to_frame_deviation'].dropna().explode()
    incd_raw = pd.to_numeric(incd_raw, errors='coerce').astype(float)
    inc_dev_series[s] = np.degrees(incd_raw.to_numpy()).tolist()

    # Quick summary for geodesic error
    median_val = np.nanmedian(geodesic_error_series[s]) if len(geodesic_error_series[s]) else np.nan
    mean_val   = np.nanmean(geodesic_error_series[s]) if len(geodesic_error_series[s]) else np.nan
    print(f"Mean geodesic error for {s}: {mean_val}")
    print(f"Median geodesic error for {s}: {median_val}")
    print("")
    # Quick summary for incremental values
    median_inc = np.nanmedian(inc_dev_series[s]) if len(inc_dev_series[s]) else np.nan
    mean_inc   = np.nanmean(inc_dev_series[s]) if len(inc_dev_series[s]) else np.nan
    print(f"Mean incremental geodesic error for {s}: {mean_inc}")
    print(f"Median incremental geodesic error for {s}: {median_inc}")


from matplotlib.colors import to_rgba

from matplotlib.colors import to_rgba

# ===========================================================================
# 2) Boxplots: Cameras (paired Wilcoxon F vs S)
#     Row 1: Identity geodesic error
#     Row 2: Incremental frame-to-frame deviation
# ===========================================================================

df = df.copy()
df['pair_id'] = df['Impact ID'].str.rsplit('_', n=1).str[0]
PAIR_KEY = ['pair_id']

def get_paired_values(df_in, source, pair_key, value_col: str, camera_col='Camera'):
    
    base = df_in.loc[
        (df_in['Data source'] == source) & (df_in['filtered'] == True),
        pair_key + [camera_col, value_col]
    ]

    print(len(base), "rows for source", source)
    
    long_df = explode_series_for_camera(base, value_col=value_col)
    f_df = long_df.loc[long_df[camera_col] == 'F', pair_key + ['frame_idx', 'val_deg']].rename(columns={'val_deg': 'val_F'})
    s_df = long_df.loc[long_df[camera_col] == 'S', pair_key + ['frame_idx', 'val_deg']].rename(columns={'val_deg': 'val_S'})
    paired = f_df.merge(s_df, on=pair_key + ['frame_idx'], how='inner')
    F_vals = paired['val_F'].to_numpy(dtype=float)
    S_vals = paired['val_S'].to_numpy(dtype=float)
    m = np.isfinite(F_vals) & np.isfinite(S_vals)
    return F_vals[m], S_vals[m]

# Camera-specific alpha for line drawings only
ALPHA_F = 1
ALPHA_S = 0.4

fig, axs = plt.subplots(2, 3, figsize=(8, 5), sharey=True)
for i, s in enumerate(sources):
    base_color = _series_color(i)

    # -------- Row 1: identity geodesic error --------
    ge_f, ge_s = get_paired_values(df, s, PAIR_KEY, value_col='geodesic_error')
    data = [ge_f, ge_s]
    labels = ['F', 'S']
    positions = [1, 2]

    bp = axs[0, i].boxplot(
        data, positions=positions, widths=0.3,
        patch_artist=True, tick_labels=labels, showfliers=False
    )

    # Style: both F and S use the same base color, but different alpha for line elements
    for k, alpha in enumerate([ALPHA_F, ALPHA_S]):  # F then S
        # Boxes
        bp['boxes'][k].set_facecolor("white")
        bp['boxes'][k].set_edgecolor(base_color)
        bp['boxes'][k].set_linewidth(1.2)
        bp['boxes'][k].set_alpha(alpha)
        # Medians
        bp['medians'][k].set_color(base_color)
        bp['medians'][k].set_linewidth(1.2)
        bp['medians'][k].set_alpha(alpha)
        # Whiskers & caps
        for idx in range(2 * k, 2 * k + 2):
            bp['whiskers'][idx].set_color(base_color)
            bp['whiskers'][idx].set_linewidth(1.2)
            bp['whiskers'][idx].set_alpha(alpha)
            bp['caps'][idx].set_color(base_color)
            bp['caps'][idx].set_linewidth(1.2)
            bp['caps'][idx].set_alpha(alpha)

        # Scatter (keep constant alpha)
        x = np.random.normal(loc=positions[k], scale=0.06, size=len(data[k]))
        axs[0, i].scatter(x, data[k], s=10, alpha=0.1, color=base_color, zorder=3)

    # Wilcoxon (paired)
    if len(data[0]) > 0 and len(data[1]) > 0:
        p_val = wilcoxon(data[0], data[1], zero_method='pratt', alternative='two-sided').pvalue
        print_wilcoxon_summary(data[0], data[1], left_label='F', right_label='S',
                       metric_name=f"{s} - GE")
        data_min = min(d.min() for d in data if len(d) > 0)
        data_max = max(d.max() for d in data if len(d) > 0)
        _annotate_pairwise_single_star(axs[0, i], positions, [(0, 1)], [p_val], data_min, data_max, alpha=0.05)

    axs[0, i].set_title(s)
    axs[0, i].set_xticks(positions)
    axs[0, i].set_xticklabels(labels)
    axs[0, i].tick_params(axis='both', which='major', labelsize=10, direction='in')
    axs[0, i].set_yscale('log')
    if i == 0:
        axs[0, i].set_ylabel(r'$\mathrm{GE}$ (deg)')

    # -------- Row 2: incremental frame-to-frame deviation --------
    inc_f, inc_s = get_paired_values(df, s, PAIR_KEY, value_col='frame_to_frame_deviation')
    data2 = [inc_f, inc_s]

    bp2 = axs[1, i].boxplot(
        data2, positions=positions, widths=0.3,
        patch_artist=True, tick_labels=labels, showfliers=False
    )

    for k, alpha in enumerate([ALPHA_F, ALPHA_S]):
        bp2['boxes'][k].set_facecolor("white")
        bp2['boxes'][k].set_edgecolor(base_color)
        bp2['boxes'][k].set_linewidth(1.2)
        bp2['boxes'][k].set_alpha(alpha)
        bp2['medians'][k].set_color(base_color)
        bp2['medians'][k].set_linewidth(1.2)
        bp2['medians'][k].set_alpha(alpha)
        for idx in range(2 * k, 2 * k + 2):
            bp2['whiskers'][idx].set_color(base_color)
            bp2['whiskers'][idx].set_linewidth(1.2)
            bp2['whiskers'][idx].set_alpha(alpha)
            bp2['caps'][idx].set_color(base_color)
            bp2['caps'][idx].set_linewidth(1.2)
            bp2['caps'][idx].set_alpha(alpha)

        # Scatter (constant alpha)
        if len(data2[k]) > 0:
            x = np.random.normal(loc=positions[k], scale=0.06, size=len(data2[k]))
            axs[1, i].scatter(x, data2[k], s=10, alpha=0.05, color=base_color, zorder=3)

    if len(data2[0]) > 0 and len(data2[1]) > 0:
        p_val2 = wilcoxon(data2[0], data2[1], zero_method='pratt', alternative='two-sided').pvalue
        print_wilcoxon_summary(data2[0], data2[1], left_label='F', right_label='S',
                       metric_name=f"{s} - IGE")
        data_min2 = min(d.min() for d in data2 if len(d) > 0)
        data_max2 = max(d.max() for d in data2 if len(d) > 0)
        _annotate_pairwise_single_star(axs[1, i], positions, [(0, 1)], [p_val2], data_min2, data_max2, alpha=0.05)

    axs[1, i].set_xticks(positions)
    axs[1, i].set_xticklabels(labels)
    axs[1, i].tick_params(axis='both', which='major', labelsize=10, direction='in')
    axs[1, i].set_yscale('log')
    if i == 0:
        axs[1, i].set_ylabel(r'$\mathrm{IGE}$ (deg)')
    if i == 1:
        axs[1, i].set_xlabel('Camera')

plt.ylim([0.001, 500])
plt.tight_layout()
plt.savefig('../FigureDev/outputs/boxplots_cameras.pdf')
plt.show()


# Add summary print statements for each subplot
for i, s in enumerate(sources):
    # Identity geodesic error
    ge_f, ge_s = get_paired_values(df, s, PAIR_KEY, value_col='geodesic_error')
    mean_ge_f, median_ge_f = np.mean(ge_f), np.median(ge_f)
    mean_ge_s, median_ge_s = np.mean(ge_s), np.median(ge_s)
    print(f"{s} - Identity GE (F): Mean = {mean_ge_f:.2f}, Median = {median_ge_f:.2f}")
    print(f"{s} - Identity GE (S): Mean = {mean_ge_s:.2f}, Median = {median_ge_s:.2f}")

    # Incremental frame-to-frame deviation
    inc_f, inc_s = get_paired_values(df, s, PAIR_KEY, value_col='frame_to_frame_deviation')
    mean_inc_f, median_inc_f = np.mean(inc_f), np.median(inc_f)
    mean_inc_s, median_inc_s = np.mean(inc_s), np.median(inc_s)
    print(f"{s} - Incremental GE (F): Mean = {mean_inc_f:.2f}, Median = {median_inc_f:.2f}")
    print(f"{s} - Incremental GE (S): Mean = {mean_inc_s:.2f}, Median = {median_inc_s:.2f}")


# ===========================================================================
# 3) Boxplots: L vs R (unpaired Mann–Whitney U)
#     Row 1: Identity geodesic error
#     Row 2: Incremental frame-to-frame deviation
# ===========================================================================

def get_unpaired_values_by_group(df_in, source, group_col, a_val, b_val, value_col: str):
    base = df_in.loc[
        (df_in['Data source'] == source) & (df_in['filtered']),
        [group_col, value_col]
    ]
    long_df = explode_series_for_camera(base, value_col=value_col)
    A_all = long_df.loc[long_df[group_col] == a_val, 'val_deg'].to_numpy(dtype=float)
    B_all = long_df.loc[long_df[group_col] == b_val, 'val_deg'].to_numpy(dtype=float)
    return A_all[np.isfinite(A_all)], B_all[np.isfinite(B_all)]

# Trial-type specific alpha for line drawings only (mirror camera plot style)
ALPHA_L = 1.0
ALPHA_R = 0.4

fig2, axs2 = plt.subplots(2, 3, figsize=(8, 5), sharey=True)
for i, s in enumerate(sources):
    base_color = _series_color(i)
    labels = ['L', 'R']
    positions = [1, 2]

    # ----- Row 1: identity geodesic error -----
    L_all, R_all = get_unpaired_values_by_group(df, s, group_col='Type', a_val='L', b_val='R', value_col='geodesic_error')
    data = [L_all, R_all]

    if len(L_all) == 0 and len(R_all) == 0:
        axs2[0, i].set_title(f"{s} (no L or R data)")
        axs2[0, i].set_yscale('log')
        if i == 0:
            axs2[0, i].set_ylabel(r'$\mathrm{GE}$ (deg)')
    else:
        bp = axs2[0, i].boxplot(
            data, positions=positions, widths=0.3,
            patch_artist=True, tick_labels=labels, showfliers=False
        )

        # Style like plot 2: same hue per column, alpha differs by group (L/R)
        for k, alpha in enumerate([ALPHA_L, ALPHA_R]):  # L then R
            # Boxes
            bp['boxes'][k].set_facecolor("white")
            bp['boxes'][k].set_edgecolor(base_color)
            bp['boxes'][k].set_linewidth(1.2)
            bp['boxes'][k].set_alpha(alpha)
            # Medians
            bp['medians'][k].set_color(base_color)
            bp['medians'][k].set_linewidth(1.2)
            bp['medians'][k].set_alpha(alpha)
            # Whiskers & caps
            for idx in range(2 * k, 2 * k + 2):
                bp['whiskers'][idx].set_color(base_color)
                bp['whiskers'][idx].set_linewidth(1.2)
                bp['whiskers'][idx].set_alpha(alpha)
                bp['caps'][idx].set_color(base_color)
                bp['caps'][idx].set_linewidth(1.2)
                bp['caps'][idx].set_alpha(alpha)
            # Scatter points (constant alpha)
            if len(data[k]) > 0:
                x = np.random.normal(loc=positions[k], scale=0.06, size=len(data[k]))
                axs2[0, i].scatter(x, data[k], s=10, alpha=0.1, color=base_color, zorder=3)

        if len(L_all) > 0 and len(R_all) > 0:
            p_val = mannwhitneyu(L_all, R_all, alternative='two-sided').pvalue
            data_min = min(d.min() for d in data if len(d) > 0)
            data_max = max(d.max() for d in data if len(d) > 0)
            _annotate_pairwise_single_star(axs2[0, i], positions, [(0, 1)], [p_val], data_min, data_max, alpha=0.05)
            print_mwu_summary(L_all, R_all, a_label='L', b_label='R',
                            metric_name=f"{s} - GE", unit='°')

        axs2[0, i].set_title(f"{s}")
        axs2[0, i].set_xticks(positions)
        axs2[0, i].set_xticklabels(labels)
        axs2[0, i].tick_params(axis='both', which='major', labelsize=10, direction='in')
        axs2[0, i].set_yscale('log')
        if i == 0:
            axs2[0, i].set_ylabel(r'$\mathrm{GE}$ (deg)')

    # ----- Row 2: incremental frame-to-frame deviation -----
    L_inc, R_inc = get_unpaired_values_by_group(df, s, group_col='Type', a_val='L', b_val='R', value_col='frame_to_frame_deviation')
    data2 = [L_inc, R_inc]

    if len(L_inc) == 0 and len(R_inc) == 0:
        axs2[1, i].set_title(f"{s} (no L or R data)")
        axs2[1, i].set_yscale('log')
        if i == 0:
            axs2[1, i].set_ylabel(r'$\mathrm{IGE}$ (deg)')
    else:
        bp2 = axs2[1, i].boxplot(
            data2, positions=positions, widths=0.3,
            patch_artist=True, tick_labels=labels, showfliers=False
        )

        for k, alpha in enumerate([ALPHA_L, ALPHA_R]):  # L then R
            # Boxes
            bp2['boxes'][k].set_facecolor("white")
            bp2['boxes'][k].set_edgecolor(base_color)
            bp2['boxes'][k].set_linewidth(1.2)
            bp2['boxes'][k].set_alpha(alpha)
            # Medians
            bp2['medians'][k].set_color(base_color)
            bp2['medians'][k].set_linewidth(1.2)
            bp2['medians'][k].set_alpha(alpha)
            # Whiskers & caps
            for idx in range(2 * k, 2 * k + 2):
                bp2['whiskers'][idx].set_color(base_color)
                bp2['whiskers'][idx].set_linewidth(1.2)
                bp2['whiskers'][idx].set_alpha(alpha)
                bp2['caps'][idx].set_color(base_color)
                bp2['caps'][idx].set_linewidth(1.2)
                bp2['caps'][idx].set_alpha(alpha)
            # Scatter points (constant alpha)
            if len(data2[k]) > 0:
                x = np.random.normal(loc=positions[k], scale=0.06, size=len(data2[k]))
                axs2[1, i].scatter(x, data2[k], s=10, alpha=0.1, color=base_color, zorder=3)

        if len(L_inc) > 0 and len(R_inc) > 0:
            p_val2 = mannwhitneyu(L_inc, R_inc, alternative='two-sided').pvalue
            data_min2 = min(d.min() for d in data2 if len(d) > 0)
            data_max2 = max(d.max() for d in data2 if len(d) > 0)
            _annotate_pairwise_single_star(axs2[1, i], positions, [(0, 1)], [p_val2], data_min2, data_max2, alpha=0.05)
            print_mwu_summary(L_inc, R_inc, a_label='L', b_label='R',
                      metric_name=f"{s} - IGE", unit='°')

        axs2[1, i].set_xticks(positions)
        axs2[1, i].set_xticklabels(labels)
        axs2[1, i].tick_params(axis='both', which='major', labelsize=10, direction='in')
        axs2[1, i].set_yscale('log')
        if i == 0:
            axs2[1, i].set_ylabel(r'$\mathrm{IGE}$ (deg)')
        if i == 1:
            axs2[1, i].set_xlabel('Header Type')

plt.ylim([0.001, 1000])
plt.tight_layout()
plt.savefig('../FigureDev/outputs/boxplots_trialtype.pdf')
plt.show()


# Add summary print statements for trial type plots
for i, s in enumerate(sources):
    # Identity geodesic error
    L_all, R_all = get_unpaired_values_by_group(df, s, group_col='Type', a_val='L', b_val='R', value_col='geodesic_error')
    mean_L_ge, median_L_ge = np.mean(L_all), np.median(L_all)
    mean_R_ge, median_R_ge = np.mean(R_all), np.median(R_all)
    print(f"{s} - Identity GE (L): Mean = {mean_L_ge:.2f}, Median = {median_L_ge:.2f}")
    print(f"{s} - Identity GE (R): Mean = {mean_R_ge:.2f}, Median = {median_R_ge:.2f}")

    # Incremental frame-to-frame deviation
    L_inc, R_inc = get_unpaired_values_by_group(df, s, group_col='Type', a_val='L', b_val='R', value_col='frame_to_frame_deviation')
    mean_L_inc, median_L_inc = np.mean(L_inc), np.median(L_inc)
    mean_R_inc, median_R_inc = np.mean(R_inc), np.median(R_inc)
    print(f"{s} - Incremental GE (L): Mean = {mean_L_inc:.2f}, Median = {median_L_inc:.2f}")
    print(f"{s} - Incremental GE (R): Mean = {mean_R_inc:.2f}, Median = {median_R_inc:.2f}")


# --- Nonparametric Bland–Altman (median + empirical centile LoA, with labels) -----------
percentile_bounds = (2.5, 97.5)
tolerances_deg = [5, 10, 15]  # for console summary

fig, axs = plt.subplots(2, 3, figsize=(8, 5), sharex='row', sharey='row')
for i, s in enumerate(sources):
    color = _series_color(i)

    # =========================
    # Row 1: Identity
    # =========================
    pred_vals = np.array(ge_ident_pred_series[s], dtype=float)
    gt_vals   = np.array(ge_ident_gt_series[s], dtype=float)
    m = np.isfinite(pred_vals) & np.isfinite(gt_vals)
    pred_vals, gt_vals = pred_vals[m], gt_vals[m]

    mean_vals = (pred_vals + gt_vals) / 2.0
    diff_vals = pred_vals - gt_vals

    if diff_vals.size > 0:
        median_diff = np.median(diff_vals)
        loa_lower, loa_upper = np.percentile(diff_vals, percentile_bounds)

        axs[0, i].scatter(mean_vals, diff_vals, alpha=0.2, s=20, color=color)
        axs[0, i].axhline(median_diff, color='k', alpha=0.7, linestyle='--', linewidth=1)
        axs[0, i].axhspan(loa_lower, loa_upper, edgecolor='none', facecolor='k', alpha=0.1, zorder=0)

        # --- Add text labels ---
        x_text = 80
        print(f"x_text for {s}: {x_text}")
        dy = (loa_upper - loa_lower) * 0.03  # vertical offset

        axs[0, i].text(x_text, median_diff + dy,
                       f"Median = {median_diff:.2f}",
                       ha='left', va='bottom', fontsize=7, color='k')

        axs[0, i].text(
            x_text, loa_upper + dy,
            fr"$\mathrm{{U}}_{{95}}$ = {loa_upper:.2f}",
            ha='left', va='bottom', fontsize=7, color='k'
        )


        axs[0, i].text(x_text, loa_lower - dy,
                       fr"$\mathrm{{L}}_{{95}}$ = {loa_lower:.2f}",
                       ha='left', va='top', fontsize=7, color='k')

        axs[0, i].set_xlim([0, 130])

        # --- Console summary ---
        print(f"\n{s} — Identity (nonparametric BA)")
        for tol in tolerances_deg:
            pct = np.mean(np.abs(diff_vals) <= tol) * 100.0
            print(f"  {pct:5.1f}% within ±{tol}")
    else:
        median_diff = loa_lower = loa_upper = np.nan

    if i == 0:
        axs[0, i].set_ylabel(r'$d(\hat{R},I) - d(R,I)$ (deg)')
    if i == 1:
        axs[0, i].set_xlabel(r'Mean($d(\hat{R},I), d(R,I)$) (deg)')
    axs[0, i].set_title(f"{s}")

    # =========================
    # Row 2: Incremental
    # =========================
    # print(inc_pred_series)
    incp = np.array(inc_pred_series[s], dtype=float)
    incg = np.array(inc_gt_series[s], dtype=float)
    m2 = np.isfinite(incp) & np.isfinite(incg)
    incp, incg = incp[m2], incg[m2]

    mean_vals2 = (incp + incg) / 2.0
    diff_vals2 = incp - incg

    if diff_vals2.size > 0:
        median_diff2 = np.median(diff_vals2)
        loa_lower2, loa_upper2 = np.percentile(diff_vals2, percentile_bounds)

        axs[1, i].scatter(mean_vals2, diff_vals2, alpha=0.2, s=20, color=color)
        axs[1, i].axhline(median_diff2, color='k', alpha=0.7, linestyle='--', linewidth=1)
        axs[1, i].axhspan(loa_lower2, loa_upper2, edgecolor='none', facecolor='k', alpha=0.1, zorder=0)

        # --- Add text labels ---
        x_text2 = 30
        dy2 = (loa_upper2 - loa_lower2) * 0.03

        axs[1, i].text(x_text2, median_diff2 + dy2,
                       f"Median = {median_diff2:.2f}",
                       ha='left', va='bottom', fontsize=7, color='k')

        axs[1, i].text(x_text2, loa_upper2 + dy2,
                       fr"$\mathrm{{U}}_{{95}}$ = {loa_upper2:.2f}",
                       ha='left', va='bottom', fontsize=7, color='k')

        axs[1, i].text(x_text2, loa_lower2 - dy2,
                       fr"$\mathrm{{L}}_{{95}}$ = {loa_lower2:.2f}",
                       ha='left', va='top', fontsize=7, color='k')

        axs[1, i].set_xlim([0, 50])
        # --- Console summary ---
        print(f"\n{s} — Incremental (nonparametric BA)")
        for tol in tolerances_deg:
            pct = np.mean(np.abs(diff_vals2) <= tol) * 100.0
            print(f"  {pct:5.1f}% within ±{tol}°")

    if i == 1:
        axs[1, i].set_xlabel(r'Mean($d(\hat{R},\hat{R}_{\mathrm{prev}}), d(R,{R}_{\mathrm{prev}})$) (deg)')
    if i == 0:
        axs[1, i].set_ylabel(r'$d(\hat{R},\hat{R}_{\mathrm{prev}}) - d(R,{R}_{\mathrm{prev}})$ (deg)')

plt.tight_layout()
plt.savefig('../FigureDev/outputs/bland_altman.pdf')
plt.show()



# ===========================================================================
# 5) Scatter & correlation
#     Row 1: Identity (Prediction vs Ground Truth)
#     Row 2: Incremental (Prediction vs Ground Truth)
# ===========================================================================

fig, axs = plt.subplots(2, 3, figsize=(8, 5), sharex='row', sharey='row')

for i, s in enumerate(sources):
    color = _series_color(i)

    # ----- Row 1: Identity -----
    pred_vals = np.array(ge_ident_pred_series[s], dtype=float)
    gt_vals   = np.array(ge_ident_gt_series[s], dtype=float)
    m = np.isfinite(pred_vals) & np.isfinite(gt_vals)
    pred_vals, gt_vals = pred_vals[m], gt_vals[m]

    if len(gt_vals) > 1 and len(pred_vals) > 1:
        model = LinearRegression()
        model.fit(gt_vals.reshape(-1, 1), pred_vals.reshape(-1, 1))
        corr, _ = pearsonr(gt_vals, pred_vals)
    else:
        corr = np.nan

    axs[0, i].set_xlim(-10, 100)
    axs[0, i].set_ylim(-20, 195)

    # ±10° tolerance band
    xline = np.linspace(axs[0, i].get_xlim()[0], axs[0, i].get_xlim()[1], 200)
    axs[0, i].fill_between(
        xline, xline - 10, xline + 10,
        facecolor='grey', edgecolor='None', alpha=0.1, label=r"$y=x \pm 10$"
    )

    # Scatter points
    axs[0, i].scatter(gt_vals, pred_vals, alpha=0.1, s=20, color=color)

    # Legend (only y=x band)
    legend_elems = [Patch(facecolor='grey', alpha=0.1, label=r"$y=x \pm 10$")]
    axs[0, i].legend(handles=legend_elems, fontsize=7, loc='upper left', frameon=False)

    # Correlation coefficient annotation (top-right corner)
    axs[0, i].text(
        0.97, 0.89, fr"$r$ = {corr:.2f}",
        ha='right', va='bottom', transform=axs[0, i].transAxes,
        fontsize=7, color='black'
    )

    axs[0, i].set_title(s)
    if i == 0:
        axs[0, i].set_ylabel(r'$d(\hat{R},I)$ (deg)')
    if i == 1:
        axs[0, i].set_xlabel(r'$d(R,I)$ (deg)')

    # ----- Row 2: Incremental -----
    incp = np.array(inc_pred_series[s], dtype=float)
    incg = np.array(inc_gt_series[s], dtype=float)
    m2 = np.isfinite(incp) & np.isfinite(incg)
    incp, incg = incp[m2], incg[m2]

    if len(incg) > 1 and len(incp) > 1:
        model2 = LinearRegression()
        model2.fit(incg.reshape(-1, 1), incp.reshape(-1, 1))
        corr2, _ = pearsonr(incg, incp)
    else:
        corr2 = np.nan

    axs[1, i].set_xlim(-5, 25)
    axs[1, i].set_ylim(-5, 75)

    # ±3° tolerance band
    xline2 = np.linspace(axs[1, i].get_xlim()[0], axs[1, i].get_xlim()[1], 200)
    axs[1, i].fill_between(
        xline2, xline2 - 3, xline2 + 3,
        facecolor='grey', edgecolor='None', alpha=0.1, label=r"$y=x \pm 3$"
    )

    # Scatter points
    axs[1, i].scatter(incg, incp, alpha=0.1, s=20, color=color)

    # Legend (only tolerance band)
    legend_elems2 = [Patch(facecolor='grey', alpha=0.1, label=r"$y=x \pm 3$")]
    axs[1, i].legend(handles=legend_elems2, fontsize=7, loc='upper left', frameon=False)

    # Correlation coefficient annotation (top-right corner)
    axs[1, i].text(
        0.97, 0.89, fr"$r$ = {corr2:.2f}",
        ha='right', va='bottom', transform=axs[1, i].transAxes,
        fontsize=7, color='black'
    )

    if i == 1:
        axs[1, i].set_xlabel(r'$d(R, R_{\mathrm{prev}})$ (deg)')
    if i == 0:
        axs[1, i].set_ylabel(r'$d(\hat{R},\hat{R}_{\mathrm{prev}})$ (deg)')

plt.tight_layout()
plt.savefig('../FigureDev/outputs/scatter.pdf')
plt.show()

# ===========================================================================
# Binned visibility vs error (cleaner than scatter)
#     Row 1: GE to Identity vs Face Visibility
#     Row 2: Incremental GE Deviation vs Face Visibility
# ===========================================================================

N_BINS = 5         # number of visibility bins in [0, 1]
MIN_PER_BIN = 10   # require at least this many samples to show a bin
USE_95_CI = False  # if True: show 1.96*SEM; else: show SEM

# ---------------------------------------------------------------------------
# Compute global face visibility normalizers (F/S) across entire dataset
# ---------------------------------------------------------------------------
face_visibility_f, face_visibility_s = [], []

for _, rec in df.iterrows():
    cam = rec.get("Camera")
    fv = rec.get("face_visibility")
    if isinstance(fv, list):
        if cam == "F":
            face_visibility_f.extend(fv)
        elif cam == "S":
            face_visibility_s.extend(fv)

face_visibility_f = np.array(face_visibility_f, dtype=float)
face_visibility_s = np.array(face_visibility_s, dtype=float)

fv_max = {
    "F": np.nanmax(face_visibility_f) if len(face_visibility_f) else np.nan,
    "S": np.nanmax(face_visibility_s) if len(face_visibility_s) else np.nan,
}


def _binned_stats(df_long: pd.DataFrame, n_bins: int = N_BINS,
                  min_per_bin: int = MIN_PER_BIN) -> dict:
    """Compute binned mean ± SEM of val_deg vs face_visibility per camera."""
    out = {}
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for cam in ['F', 'S']:
        sub = df_long.loc[df_long['Camera'] == cam].copy()
        if sub.empty:
            out[cam] = dict(x=np.array([]), y=np.array([]), err=np.array([]), n=np.array([]))
            continue

        vis = sub['face_visibility'].to_numpy(dtype=float)
        val = sub['val_deg'].to_numpy(dtype=float)

        vmax = fv_max.get(cam, np.nan)
        if np.isfinite(vmax) and vmax > 0:
            vis = vis / vmax
        else:
            vis[:] = np.nan

        m = np.isfinite(vis) & np.isfinite(val) & (vis >= 0)
        vis, val = vis[m], val[m]

        inds = np.digitize(vis, bin_edges, right=True) - 1
        valid = (inds >= 0) & (inds < n_bins)
        inds = inds[valid]
        val = val[valid]

        y_mean = np.full(n_bins, np.nan)
        y_sem = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            yb = val[inds == b]
            if len(yb) >= min_per_bin:
                counts[b] = len(yb)
                y_mean[b] = np.nanmean(yb)
                sd = np.nanstd(yb, ddof=1) if len(yb) > 1 else np.nan
                sem = sd / np.sqrt(len(yb)) if np.isfinite(sd) else np.nan
                y_sem[b] = (1.96 * sem) if USE_95_CI else sem

        ok = ~np.isnan(y_mean)
        out[cam] = dict(
            x=bin_centers[ok],
            y=y_mean[ok],
            err=y_sem[ok],
            n=counts[ok]
        )
    return out


def explode_metric_and_visibility(df_in: pd.DataFrame, source: str, value_col: str) -> pd.DataFrame:
    """Return per-frame rows for (metric, face_visibility, Camera)."""
    is_incremental = value_col.startswith('frame_to_frame')

    sub = df_in.loc[
        (df_in['Data source'] == source) & (df_in['filtered']),
        [value_col, 'face_visibility', 'Camera']
    ].copy()

    def _to_list(x): return x if isinstance(x, list) else [x]

    rows = []
    for _, r in sub.iterrows():
        vals = _to_list(r[value_col])
        vis = _to_list(r['face_visibility'])
        if is_incremental and len(vis) == len(vals) + 1:
            vis = vis[1:]
        L = min(len(vals), len(vis))
        if L == 0:
            continue
        vals, vis = vals[:L], vis[:L]
        cam = r['Camera']
        for v, fv in zip(vals, vis):
            rows.append({
                'face_visibility': fv,
                'val_deg': np.degrees(float(v)) if pd.notna(v) else np.nan,
                'Camera': cam
            })

    out = pd.DataFrame(rows)
    if len(out):
        out['face_visibility'] = pd.to_numeric(out['face_visibility'], errors='coerce')
        out['val_deg'] = pd.to_numeric(out['val_deg'], errors='coerce')
        m = np.isfinite(out['face_visibility']) & np.isfinite(out['val_deg'])
        out = out.loc[m]
    return out


# ------- Plot (two rows, binned means with error bars) -------
ALPHA_F = 1.0
ALPHA_S = 0.4

fig, axs = plt.subplots(2, 3, figsize=(8, 5), sharex='col', sharey='row')

for i, s in enumerate(sources):
    base_color = _series_color(i)

    # -------- Row 1: Identity GE binned vs visibility --------
    df_top = explode_metric_and_visibility(df, s, value_col='geodesic_error')
    if len(df_top) == 0:
        axs[0, i].set_title(f"{s} (no data)")
    else:
        stats_top = _binned_stats(df_top, n_bins=N_BINS, min_per_bin=MIN_PER_BIN)
        for cam, alpha in zip(['F', 'S'], [ALPHA_F, ALPHA_S]):
            x = stats_top[cam]['x']; y = stats_top[cam]['y']; e = stats_top[cam]['err']
            if len(x):
                axs[0, i].errorbar(
                    x, y, yerr=e, fmt='-o', ms=3, lw=1.2,
                    elinewidth=0.9, capsize=2,
                    color=base_color, alpha=alpha, label=cam
                )
        axs[0, i].tick_params(axis='both', which='major', labelsize=10, direction='in')
        axs[0, i].set_title(s, fontsize=10)
        if i == 0:
            axs[0, i].set_ylabel('GE (deg)')

    # -------- Row 2: Incremental GE binned vs visibility -----
    df_bot = explode_metric_and_visibility(df, s, value_col='frame_to_frame_deviation')
    if len(df_bot) == 0:
        axs[1, i].set_title(f"{s} (no data)")
    else:
        stats_bot = _binned_stats(df_bot, n_bins=N_BINS, min_per_bin=MIN_PER_BIN)
        for cam, alpha in zip(['F', 'S'], [ALPHA_F, ALPHA_S]):
            x2 = stats_bot[cam]['x']; y2 = stats_bot[cam]['y']; e2 = stats_bot[cam]['err']
            if len(x2):
                axs[1, i].errorbar(
                    x2, y2, yerr=e2, fmt='-o', ms=3, lw=1.2,
                    elinewidth=0.9, capsize=2,
                    color=base_color, alpha=alpha, label=cam
                )
        axs[1, i].tick_params(axis='both', which='major', labelsize=10, direction='in')
        # axs[1, i].set_title(s, fontsize=10)
        if i == 0:
            axs[1, i].set_ylabel('IGE (deg)')

    # Legend per column (bottom axis)
    h, l = axs[0, i].get_legend_handles_labels()
    if h:
        axs[0, i].legend(h, l, fontsize=7, loc='upper right')
        axs[1, i].legend(h, l, fontsize=7, loc='upper right')

axs[1, 1].set_xlabel('Visible face area (normalised)')

plt.tight_layout()
plt.savefig('../FigureDev/outputs/error_visibility.pdf')
plt.show()
