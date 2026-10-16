#!/usr/bin/env python3
"""Dominance-matrix cells and edges for the graph-construction figure (fig:graph_construction).

Feeds the figure's matches through the model's own PlayerInteractionMatrix under the
final config (T170), one window per snapshot with one row per set played, as
windowstacks.py does. Graphs start empty at the 2024 Wimbledon semi-finals.
Matches (public results) are in data/examples/fig1_matches.csv. The Paris Olympics
is not in the model's dataset; it is fed here as in the figure.
"""
import os, sys, json
from datetime import datetime, timezone
from types import SimpleNamespace
import torch
import pandas as pd
from rich.console import Console
from rich.table import Table

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.getcwd())
from model.graph.player_interaction import PlayerInteractionMatrix

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TUNE = 'tunes/tune_degree_postab_best.json'  # final T170 config
MATCHES = 'data/examples/fig1_matches.csv'
NAMES = ['Musetti', 'Djokovic', 'Alcaraz', 'Medvedev']  # figure's node indices 0-3
SURFS = ['Hard', 'Grass', 'Clay']

p = json.load(open(TUNE))['params']
pim = PlayerInteractionMatrix(
    gender='men', granularity=p['granularity'], lambda_param=p['lambda_param'],
    surfaces={'Hard':  {'Hard': 1.0, 'Clay': p['HC'], 'Grass': p['HG']},
              'Clay':  {'Hard': p['CH'], 'Clay': 1.0, 'Grass': p['CG']},
              'Grass': {'Hard': p['GH'], 'Clay': p['GC'], 'Grass': 1.0}},
    tournament_quality={'GS': 1.0, 'Finals': p['beta_Finals'], '1000': p['beta_1000'], '500': p['beta_500']},
    max_edge_age=p['max_edge_age'], min_dom_pctile=p['min_dom_pctile'],
    node_feature_matrix=torch.zeros(len(NAMES), 1))

def parse(r):  # (winner idx, loser idx, winner games, loser games, sets) from a score like '6-4 7-6'
    sets = [tuple(map(int, s.split('-'))) for s in r.score.split()]
    return NAMES.index(r.winner), NAMES.index(r.loser), sum(w for w, _ in sets), sum(l for _, l in sets), len(sets)

df = pd.read_csv(MATCHES, keep_default_na=False)
con = Console(width=130)
t = Table(title='Figure 1 matches: games won by the winner (caption: Djokovic wins 54% in the Olympic final)')
for c in ['Date', 'Match', 'Surface', 'Winner', 'Loser', 'Score', 'Winner games', 'Snapshot']: t.add_column(c, justify='right')
for r in df.itertuples():
    _, _, wg, lg, _ = parse(r)
    t.add_row(r.date, r.tournament, r.surface, r.winner, r.loser, r.score, f'{wg}/{wg + lg} = {wg / (wg + lg):.0%}', r.snapshot or '-')
con.print(t)
for (day, surf, level, snap), g in df.groupby(['date', 'surface', 'level', 'snapshot'], sort=False):  # one window per day
    ms = [parse(r) for r in g.itertuples()]
    rows = [m for m in ms for _ in range(m[4])]  # one row per set, match-level game share
    pim.update_interaction_matrix(surf, SimpleNamespace(
        results=torch.ones(len(rows)), matches=torch.tensor([[w, l] for w, l, *_ in rows]),
        game_pct=torch.tensor([wg / (wg + lg) for _, _, wg, lg, _ in rows]),
        timestamp=datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp(),
        tourney_level=level, a_serve_pct=None, a_return_pct=None, a_bp_pct=None))
    if not snap:  # window not shown in the figure
        continue
    # D(u,v) = 0.5 + stored |posterior| on the dominant direction (row u beats column v)
    t = Table(title=f'{snap} ({day}): D(row, col)')
    t.add_column('surface'), t.add_column('row'), [t.add_column(f'{i} {n}') for i, n in enumerate(NAMES)]
    for s in SURFS:
        im = pim.interaction_matrices[s]
        for u, nu in enumerate(NAMES):
            cells = ['-' if u == v else '0' if im[u, v] == im[v, u] == 0
                     else f'{0.5 + im[u, v] - im[v, u]:.3f}' for v in range(len(NAMES))]
            t.add_row(s if u == 0 else '', f'{u} {nu}', *cells, end_section=u == len(NAMES) - 1)
    con.print(t)
