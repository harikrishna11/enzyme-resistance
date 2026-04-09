#!/usr/bin/env python3
"""
Generate Figure 1 for the Application Note.

Multi-panel figure:
  (A) Impressive electronic-circuit analogy of the protein
  (B) Bar chart: Pearson r comparison with published methods on S2648
  (C) Top 20 feature importances
  (D) Scatter plot: predicted vs experimental ΔΔG on S2648 (r = 0.656)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc, Circle
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "manuscript")
RESULTS = os.path.join(BASE, "benchmark_results_s2648")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})


# ─── Helper: draw a zigzag resistor between two points ───────────
def draw_resistor(ax, p1, p2, n_zigzag=5, width=0.25, color='#555555',
                  lw=1.5, alpha=1.0, zorder=2):
    """Draw a zigzag resistor symbol between p1 and p2."""
    p1, p2 = np.array(p1), np.array(p2)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-6:
        return
    u = d / length                    # unit along
    v = np.array([-u[1], u[0]])       # unit perp

    # Lead-in / lead-out fraction
    lead = 0.2
    pts = [p1, p1 + lead * d]

    zigzag_start = lead
    zigzag_end = 1.0 - lead
    seg = (zigzag_end - zigzag_start) / (2 * n_zigzag)

    for i in range(2 * n_zigzag):
        t = zigzag_start + (i + 1) * seg
        sign = 1 if i % 2 == 0 else -1
        pt = p1 + t * d + sign * width * length * 0.08 * v
        pts.append(pt)

    pts.append(p1 + (1 - lead) * d)
    pts.append(p2)

    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], '-', color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=zorder)


def draw_node(ax, pos, label='', color='#3498DB', size=280, fontsize=9,
              edgecolor='white', textcolor='white', zorder=5, glow=False):
    """Draw a circuit node (residue)."""
    if glow:
        ax.scatter(*pos, s=size * 2.0, c=color, alpha=0.15, edgecolors='none', zorder=zorder - 1)
        ax.scatter(*pos, s=size * 1.4, c=color, alpha=0.25, edgecolors='none', zorder=zorder - 1)
    ax.scatter(*pos, s=size, c=color, edgecolors=edgecolor, linewidths=1.5,
              zorder=zorder)
    if label:
        ax.text(pos[0], pos[1], label, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold', zorder=zorder + 1)


def draw_current_arrow(ax, p1, p2, color='#E74C3C', lw=1.8, alpha=0.7):
    """Draw a current-flow arrow between two points (offset from edge)."""
    p1, p2 = np.array(p1), np.array(p2)
    mid = (p1 + p2) / 2
    d = p2 - p1
    u = d / np.linalg.norm(d)
    v = np.array([-u[1], u[0]])  # perpendicular

    # Offset arrow slightly from the edge
    offset = v * 0.18
    start = p1 + 0.35 * d + offset
    end = p1 + 0.65 * d + offset

    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=12),
                zorder=4)


# ═══════════════════════════════════════════════════════════════════
# Create figure
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30,
              left=0.06, right=0.97, top=0.97, bottom=0.04)

# ═══════════════════════════════════════════════════════════════════
# Panel A: Impressive Electronic Circuit Analogy
# ═══════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(-0.5, 11.5)
ax_a.set_ylim(-0.5, 11.5)
ax_a.set_aspect('equal')
ax_a.axis('off')
ax_a.set_title('(A)', fontweight='bold', loc='left', fontsize=15, pad=10)

# Faint background: protein backbone ribbon
theta = np.linspace(0, 2 * np.pi, 300)
ribbon_x = 5.5 + 3.8 * np.cos(theta)
ribbon_y = 5.5 + 3.8 * np.sin(theta)
# Draw a thicker faded ribbon
for w, a in [(18, 0.04), (12, 0.06), (6, 0.08)]:
    ax_a.plot(ribbon_x, ribbon_y, '-', color='#B0C4DE', lw=w, alpha=a, zorder=0)
ax_a.plot(ribbon_x, ribbon_y, '-', color='#B0C4DE', lw=2.0, alpha=0.25, zorder=0)

# Also draw a secondary structure hint (helical curve)
t_helix = np.linspace(0, 4 * np.pi, 200)
hx = 5.5 + (3.8 + 0.3 * np.sin(8 * t_helix)) * np.cos(t_helix / (4*np.pi) * 2*np.pi)
hy = 5.5 + (3.8 + 0.3 * np.sin(8 * t_helix)) * np.sin(t_helix / (4*np.pi) * 2*np.pi)
ax_a.plot(hx, hy, '-', color='#87CEEB', lw=0.5, alpha=0.3, zorder=0)

# ── Define residue node positions (layout like a circuit board) ──
nodes = {
    # Residue label: (x, y)
    'R1': (2.0, 9.5),
    'R2': (5.5, 10.2),
    'R3': (9.0, 9.5),
    'R4': (1.0, 6.8),    # mutation site
    'R5': (3.8, 7.5),
    'R6': (7.2, 7.8),
    'R7': (10.0, 6.8),
    'R8': (1.5, 4.0),
    'R9': (4.5, 4.8),
    'R10': (6.8, 4.2),   # active site
    'R11': (9.5, 4.0),   # active site
    'R12': (3.0, 1.8),
    'R13': (5.5, 1.2),
    'R14': (8.0, 1.8),
}

# Node types
mutation_nodes = {'R4'}
active_nodes = {'R10', 'R11'}
regular_nodes = set(nodes.keys()) - mutation_nodes - active_nodes

# ── Define edges (contacts/resistors) ──
edges = [
    ('R1', 'R2'), ('R2', 'R3'), ('R1', 'R4'), ('R1', 'R5'),
    ('R2', 'R5'), ('R2', 'R6'), ('R3', 'R6'), ('R3', 'R7'),
    ('R4', 'R5'), ('R4', 'R8'), ('R5', 'R6'), ('R5', 'R9'),
    ('R6', 'R7'), ('R6', 'R10'), ('R7', 'R11'),
    ('R8', 'R9'), ('R8', 'R12'), ('R9', 'R10'), ('R9', 'R12'),
    ('R9', 'R13'), ('R10', 'R11'), ('R10', 'R13'), ('R10', 'R14'),
    ('R11', 'R14'), ('R12', 'R13'), ('R13', 'R14'),
]

# Draw edges as resistor symbols
for n1, n2 in edges:
    p1, p2 = np.array(nodes[n1]), np.array(nodes[n2])
    d = np.linalg.norm(p2 - p1)

    # Color & style by involvement with mutation/active site
    if n1 in mutation_nodes or n2 in mutation_nodes:
        color = '#E74C3C'
        lw = 2.0
        alpha = 0.85
    elif n1 in active_nodes or n2 in active_nodes:
        color = '#F39C12'
        lw = 1.8
        alpha = 0.75
    else:
        color = '#7F8C8D'
        lw = 1.5
        alpha = 0.65

    draw_resistor(ax_a, p1, p2, n_zigzag=4, width=0.3, color=color,
                  lw=lw, alpha=alpha)

# Draw current flow arrows along a path from mutation → active site
current_path = [('R4', 'R5'), ('R5', 'R9'), ('R9', 'R10')]
for n1, n2 in current_path:
    draw_current_arrow(ax_a, nodes[n1], nodes[n2],
                       color='#E74C3C', lw=2.2, alpha=0.8)

# Draw current flow label
mid_path = (np.array(nodes['R5']) + np.array(nodes['R9'])) / 2
ax_a.text(mid_path[0] - 0.9, mid_path[1] + 0.5, 'I(ω)',
          fontsize=12, color='#E74C3C', fontstyle='italic',
          fontweight='bold', alpha=0.85, zorder=6)

# ── Draw nodes ──
for name, pos in nodes.items():
    if name in mutation_nodes:
        draw_node(ax_a, pos, name[1:], color='#E74C3C', size=380,
                  fontsize=11, edgecolor='#C0392B', glow=True)
    elif name in active_nodes:
        draw_node(ax_a, pos, name[1:], color='#F39C12', size=350,
                  fontsize=11, edgecolor='#E67E22', glow=True)
    else:
        draw_node(ax_a, pos, name[1:], color='#2C3E50', size=300,
                  fontsize=10, edgecolor='#1A252F')

# ── Annotations ──
# Mutation site annotation with a styled box
ax_a.annotate('Mutation site\n(ΔR perturbation)',
              xy=nodes['R4'], xytext=(-0.8, 9.0),
              fontsize=10, color='#E74C3C', fontweight='bold',
              ha='center', va='bottom',
              arrowprops=dict(arrowstyle='->', color='#E74C3C',
                              lw=2, connectionstyle='arc3,rad=-0.2'),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC',
                        edgecolor='#E74C3C', alpha=0.9),
              zorder=7)

# Active site annotation
ax_a.annotate('Active site\n(functional hub)',
              xy=((nodes['R10'][0] + nodes['R11'][0])/2,
                  (nodes['R10'][1] + nodes['R11'][1])/2),
              xytext=(10.5, 2.5),
              fontsize=10, color='#E67E22', fontweight='bold',
              ha='center', va='top',
              arrowprops=dict(arrowstyle='->', color='#E67E22',
                              lw=2, connectionstyle='arc3,rad=0.2'),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF5E7',
                        edgecolor='#E67E22', alpha=0.9),
              zorder=7)

# Voltage/ground symbols
# "V+" at mutation site
ax_a.text(nodes['R4'][0] - 1.8, nodes['R4'][1] - 0.8, '$V^{+}$',
          fontsize=16, color='#C0392B', fontweight='bold',
          zorder=6)
# Ground symbol near bottom
gnd_x, gnd_y = 5.5, -0.1
for w in [0.8, 0.5, 0.2]:
    ax_a.plot([gnd_x - w, gnd_x + w], [gnd_y - (0.8 - w) * 0.8, gnd_y - (0.8 - w) * 0.8],
              '-', color='#2C3E50', lw=2, zorder=5)
ax_a.plot([gnd_x, gnd_x], [gnd_y, gnd_y + 0.6], '-', color='#2C3E50', lw=2, zorder=5)
ax_a.text(gnd_x, gnd_y - 0.6, 'GND', fontsize=9, ha='center', color='#2C3E50',
          fontweight='bold')

# ── Legend (circuit notation) ──
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50',
               markersize=10, label='Residue (node)', markeredgecolor='#1A252F'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='Mutation site', markeredgecolor='#C0392B'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12',
               markersize=10, label='Active site (hub)', markeredgecolor='#E67E22'),
    plt.Line2D([0], [0], color='#7F8C8D', lw=2, label='Contact (resistor)', linestyle='-'),
    mlines.Line2D([], [], color='#E74C3C', marker='>', markersize=7,
                  lw=2, label='Current flow I(ω)'),
]
ax_a.legend(handles=legend_elements, loc='lower left', fontsize=9,
            framealpha=0.95, edgecolor='#BDC3C7', fancybox=True,
            borderpad=0.8)

# Title subtitle
ax_a.text(5.5, 11.2, 'Protein → Electrical Circuit', fontsize=13,
          ha='center', va='bottom', fontstyle='italic', color='#2C3E50',
          fontweight='bold')


# ═══════════════════════════════════════════════════════════════════
# Panel B: Pearson r comparison with published methods (S2648)
# ═══════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])

methods = [
    ('DynaMut2', 0.720, 'NMA'),
    ('ThermoNet', 0.690, 'DL'),
    ('PoPMuSiC', 0.670, 'Stat'),
    ('DynaMut', 0.670, 'NMA'),
    ('RaSP', 0.670, 'DL'),
    ('ACDC-NN', 0.660, 'DL'),
    ('Circuit (ours)', 0.656, 'Ours'),
    ('DUET', 0.640, 'ML'),
    ('MAESTRO', 0.630, 'ML'),
    ('mCSM', 0.610, 'ML'),
    ('Rosetta', 0.580, 'Phys'),
    ('DDGun3D', 0.570, 'ML'),
    ('INPS3D', 0.580, 'ML'),
    ('I-Mutant', 0.540, 'ML'),
    ('SDM', 0.520, 'Stat'),
    ('FoldX', 0.480, 'Phys'),
]

# Sort by pearson_r
methods.sort(key=lambda x: x[1], reverse=True)
names = [m[0] for m in methods]
pearson_vals = [m[1] for m in methods]
cats = [m[2] for m in methods]

cat_colors = {
    'NMA': '#3498DB', 'DL': '#9B59B6', 'Stat': '#1ABC9C',
    'ML': '#95A5A6', 'Phys': '#E67E22', 'Ours': '#E74C3C',
}

bar_colors = [cat_colors.get(c, '#95A5A6') for c in cats]
y_pos = np.arange(len(names))

bars = ax_b.barh(y_pos, pearson_vals, color=bar_colors, edgecolor='white',
                 height=0.72, zorder=3)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(names, fontsize=10)
ax_b.set_xlabel('Pearson r', fontsize=12, fontweight='bold')
ax_b.set_title('(B)', fontweight='bold', loc='left', fontsize=15, pad=10)
ax_b.text(0.5, 1.02, 'S2648 Benchmark: Pearson r', fontsize=13,
          fontweight='bold', ha='center', va='bottom',
          transform=ax_b.transAxes, color='#2C3E50')
ax_b.set_xlim(0.35, 0.79)
ax_b.invert_yaxis()

# Add light grid
ax_b.xaxis.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax_b.set_axisbelow(True)

# Highlight our bar with special styling
for i, (name, val, cat) in enumerate(methods):
    if cat == 'Ours':
        ax_b.barh(i, val, color='#E74C3C', edgecolor='#C0392B',
                  height=0.72, linewidth=2.5, zorder=4)
        ax_b.text(val + 0.006, i, f'{val:.3f}', va='center', fontsize=11,
                  fontweight='bold', color='#E74C3C')
        # Add a star marker
        ax_b.plot(val + 0.06, i, '*', color='#E74C3C', markersize=14,
                  markeredgecolor='#C0392B', markeredgewidth=0.5, zorder=5)
    else:
        ax_b.text(val + 0.006, i, f'{val:.3f}', va='center', fontsize=9,
                  color='#555555')

# Category legend
cat_legend = [mpatches.Patch(facecolor=c, edgecolor='white', label=l) for l, c in
              [('Physics-based', '#E67E22'), ('Statistical', '#1ABC9C'),
               ('ML / Graph', '#95A5A6'), ('NMA + ML', '#3498DB'),
               ('Deep Learning', '#9B59B6'), ('Circuit (ours)', '#E74C3C')]]
ax_b.legend(handles=cat_legend, loc='lower right', fontsize=9,
            title='Category', title_fontsize=10, framealpha=0.95,
            edgecolor='#BDC3C7', fancybox=True)


# ═══════════════════════════════════════════════════════════════════
# Panel C: Top 20 feature importances
# ═══════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 0])

# Load feature importances
fi = pd.read_csv(os.path.join(RESULTS, 'final_feature_importances.csv'))
fi_top = fi.head(20).iloc[::-1]  # reverse for horizontal bar (top on top)

# Color by source category
def feat_color(fname):
    if fname.startswith('cc_'):
        return '#E67E22'   # AA circuit
    elif fname.startswith('scale_'):
        return '#9B59B6'   # cross-scale
    elif fname.endswith('_6A'):
        return '#1ABC9C'   # 6Å cutoff
    else:
        return '#3498DB'   # 8Å cutoff

fc = [feat_color(f) for f in fi_top['feature'].values]

ax_c.barh(range(len(fi_top)), fi_top['imp'].values, color=fc,
          edgecolor='white', height=0.72, zorder=3)

# Clean up feature names for display
clean_names = []
for f in fi_top['feature'].values:
    f = f.replace('_wt', '').replace('delta_', 'Δ').replace('_6A', ' (6Å)')
    f = f.replace('cc_', 'Δ').replace('_', ' ')
    if len(f) > 32:
        f = f[:30] + '..'
    clean_names.append(f)

ax_c.set_yticks(range(len(fi_top)))
ax_c.set_yticklabels(clean_names, fontsize=9)
ax_c.set_xlabel('Feature importance (GBR)', fontsize=12, fontweight='bold')
ax_c.set_title('(C)', fontweight='bold', loc='left', fontsize=15, pad=10)
ax_c.text(0.5, 1.02, f'Top 20 Features (56 selected)', fontsize=13,
          fontweight='bold', ha='center', va='bottom',
          transform=ax_c.transAxes, color='#2C3E50')

# Light grid
ax_c.xaxis.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax_c.set_axisbelow(True)

# Source legend
src_legend = [
    mpatches.Patch(facecolor='#3498DB', edgecolor='white', label='8 Å circuit'),
    mpatches.Patch(facecolor='#1ABC9C', edgecolor='white', label='6 Å circuit'),
    mpatches.Patch(facecolor='#E67E22', edgecolor='white', label='AA circuit'),
    mpatches.Patch(facecolor='#9B59B6', edgecolor='white', label='Cross-scale'),
]
ax_c.legend(handles=src_legend, loc='lower right', fontsize=9,
            title='Feature source', title_fontsize=10, framealpha=0.95,
            edgecolor='#BDC3C7', fancybox=True)


# ═══════════════════════════════════════════════════════════════════
# Panel D: Predicted vs Experimental (S2648)
# ═══════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1])

# Load best predictions
pred_file = os.path.join(RESULTS, 'best_predictions.csv')
if os.path.exists(pred_file):
    preds = pd.read_csv(pred_file)
    y = preds['y_true'].values
    y_pred = preds['y_pred'].values
else:
    # Fallback: load feature matrix and recompute
    fm = pd.read_csv(os.path.join(RESULTS, 'feature_matrix.csv'))
    tgt = pd.read_csv(os.path.join(RESULTS, 'target_values.csv'))
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    X = fm.values
    y = tgt['ddG_experimental'].values
    model = GradientBoostingRegressor(n_estimators=500, max_depth=6,
                                       learning_rate=0.02, subsample=0.7,
                                       min_samples_leaf=5, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)

from scipy.stats import pearsonr, spearmanr
r_val, _ = pearsonr(y, y_pred)
rho_val, _ = spearmanr(y, y_pred)
rmse_val = np.sqrt(np.mean((y - y_pred)**2))

# Scatter with density coloring
from scipy.stats import gaussian_kde

try:
    xy = np.vstack([y, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_s, yp_s, z_s = y[idx], y_pred[idx], z[idx]
    sc = ax_d.scatter(y_s, yp_s, c=z_s, s=18, alpha=0.7, edgecolors='none',
                      cmap='plasma', rasterized=True, zorder=3)
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax_d, shrink=0.6, pad=0.02, aspect=25)
    cbar.set_label('Density', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
except Exception:
    ax_d.scatter(y, y_pred, c='#3498DB', s=12, alpha=0.4, edgecolors='none',
                 rasterized=True, zorder=3)

# Fit line
z_fit = np.polyfit(y, y_pred, 1)
x_line = np.linspace(y.min(), y.max(), 100)
ax_d.plot(x_line, np.polyval(z_fit, x_line), '--', color='#E74C3C', lw=2.5,
          label='Fit', zorder=4)

# Identity line
lims = [min(y.min(), y_pred.min()) - 1.0, max(y.max(), y_pred.max()) + 1.0]
ax_d.plot(lims, lims, '-', color='#95A5A6', lw=1.2, alpha=0.6, label='y = x',
          zorder=2)
ax_d.set_xlim(lims)
ax_d.set_ylim(lims)

ax_d.set_xlabel('Experimental ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
ax_d.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
ax_d.set_title('(D)', fontweight='bold', loc='left', fontsize=15, pad=10)
ax_d.text(0.5, 1.02, 'S2648: Predicted vs. Experimental', fontsize=13,
          fontweight='bold', ha='center', va='bottom',
          transform=ax_d.transAxes, color='#2C3E50')

# Light grid
ax_d.xaxis.grid(True, alpha=0.2, linestyle='--', zorder=0)
ax_d.yaxis.grid(True, alpha=0.2, linestyle='--', zorder=0)
ax_d.set_axisbelow(True)

# Stats box (prominent)
stats_text = (f'n = {len(y):,}\n'
              f'r  = {r_val:.3f}\n'
              f'ρ  = {rho_val:.3f}\n'
              f'RMSE = {rmse_val:.2f}')
props = dict(boxstyle='round,pad=0.5', facecolor='#FDFEFE',
             edgecolor='#2C3E50', alpha=0.95, lw=1.5)
ax_d.text(0.04, 0.96, stats_text, transform=ax_d.transAxes, fontsize=11,
          verticalalignment='top', fontfamily='monospace', bbox=props,
          zorder=6)

ax_d.legend(loc='lower right', fontsize=10, framealpha=0.95,
            edgecolor='#BDC3C7', fancybox=True)


# ─── Save ─────────────────────────────────────────────────────────
fig_path = os.path.join(OUT_DIR, 'Figure1.png')
fig_tiff = os.path.join(OUT_DIR, 'Figure1.tiff')
plt.savefig(fig_path, dpi=300, facecolor='white')
plt.savefig(fig_tiff, dpi=300, facecolor='white')
plt.close()
print(f"Figure saved: {fig_path}")
print(f"Figure saved: {fig_tiff}")
