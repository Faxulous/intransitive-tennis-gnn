#!/usr/bin/env python3
"""Render the two Section 6 example cycles to outputs/ (copied into the paper's figures/jpegs/).

classic_int_example.jpg (fig:non-transitive_triangle): career head-to-head
match-win proportions, walkovers excluded, from data/examples/fig5_h2h.csv (ATP H2H).
women_int_example.jpg (fig:womens_nontransitive): dominance edges from
eval/intr/triadviz.py (gender=women, window 1202, Hard, T170 config), the graph
state before Sakkari-Sabalenka, Abu Dhabi 2021-01-12.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT, DPI, R, FONT = 'outputs', 330, 45, 'STIXGeneral'  # serif bundled with matplotlib
H2H = 'data/examples/fig5_h2h.csv'
# Women: (src, dst, weight, label xy) from triadviz w1202 output
WOMEN = [('Cornet', 'Sabalenka', 0.71, (338, 95)), ('Sabalenka', 'Sakkari', 0.62, (285, 240)),
         ('Sakkari', 'Cornet', 0.57, (75, 250)), ('Sakkari', 'Anisimova', 0.75, (350, 370)),
         ('Anisimova', 'Sabalenka', 0.69, (580, 245))]
# Layouts in pixels of the published images: name -> (x, y, colour, name xy, name ha)
CLASSIC_NODES = {'Davydenko': (420, 125, '#c5e3f4', (420, 48), 'center'),
                 'Federer':   (262, 400, '#d9f2d3', (205, 400), 'right'),
                 'Nadal':     (578, 400, '#fbe1d6', (640, 400), 'left')}
WOMEN_NODES = {'Cornet':    (140, 125, '#f4665f', (140, 50), 'center'),
               'Sabalenka': (520, 125, '#c5e3f4', (520, 50), 'center'),
               'Sakkari':   (140, 400, '#c4ecc4', (140, 475), 'center'),
               'Anisimova': (520, 400, '#efd2ef', (520, 475), 'center')}


def career_edges():  # (dominant, dominated, match-win share) per pair, walkovers excluded
    d = pd.read_csv(H2H)
    d = d[~d.walkover]
    out = []
    for a, b, xy in [('Federer', 'Davydenko', (255, 255)), ('Davydenko', 'Nadal', (580, 255)),
                     ('Nadal', 'Federer', (435, 460))]:
        x = d[d.winner.isin([a, b]) & d.loser.isin([a, b])]
        w = (x.winner == a).sum()
        if w * 2 <= len(x):
            raise ValueError(f'{a} does not lead {b}: {w}/{len(x)}')
        print(f'{a} -> {b}: {w}/{len(x)} = {w / len(x):.3f}')
        out.append((a, b, w / len(x), xy))
    return out


def draw(nodes, edges, size, fn):  # pixel-space render at the published image size
    fig = plt.figure(figsize=(size[0] / DPI, size[1] / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    for name, (x, y, c, (nx, ny), ha) in nodes.items():
        ax.add_patch(Circle((x, y), R, facecolor=c, edgecolor='black', lw=2, zorder=3))
        ax.text(nx, ny, name, ha=ha, va='center', fontsize=11, fontweight='bold', fontfamily=FONT)
    for src, dst, w, (lx, ly) in edges:
        (x1, y1), (x2, y2) = nodes[src][:2], nodes[dst][:2]
        L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        ux, uy, g = (x2 - x1) / L, (y2 - y1) / L, R + 4  # trim so heads sit outside the circles
        ax.add_patch(FancyArrowPatch((x1 + ux * g, y1 + uy * g), (x2 - ux * g, y2 - uy * g),
                                     arrowstyle='-|>', color='black', lw=2, mutation_scale=15, zorder=2))
        ax.text(lx, ly, f'{w:.2f}', ha='center', va='center', fontsize=11, fontfamily=FONT)
    ax.set_xlim(0, size[0]), ax.set_ylim(size[1], 0), ax.axis('off')
    fig.savefig(f'{OUT}/{fn}', dpi=DPI, pil_kwargs={'quality': 95})
    print(f'saved -> {OUT}/{fn}')


draw(CLASSIC_NODES, career_edges(), (1002, 518), 'classic_int_example.jpg')
draw(WOMEN_NODES, WOMEN, (685, 534), 'women_int_example.jpg')
