"""Differentiable point->game->set->match probability model (Klaassen & Magnus 2003).
All functions are batch-vectorised and support PyTorch autograd.
"""
import torch

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERVE_MEANS = {
    'men':   {'Hard': 0.641, 'Clay': 0.620, 'Grass': 0.664},
    'women': {'Hard': 0.567, 'Clay': 0.560, 'Grass': 0.587},
}
def raw_to_serve(raw, gender, surface, scale=0.20):
    """Map raw model output to bounded serve win probability: mu ± scale."""
    mu = SERVE_MEANS[gender][surface]
    return mu + scale * torch.tanh(raw)


def game_prob(p):
    """P(server wins game | point win prob p). Exact under iid (K&M 2003 eq.1)."""
    q = 1 - p
    pre = p**4 * (1 + 4*q + 10*q**2)       # Win 4-0, 4-1, or 4-2
    deuce = 20 * p**3 * q**3 * p**2 / (p**2 + q**2)  # Reach 3-3, then win
    return pre + deuce


def tiebreak_prob(pa, pb):
    """P(A wins tiebreak). Constant-prob approx: p = (pa + 1-pb)/2."""
    p = (pa + 1 - pb) / 2  # A's avg point win prob across serve/return
    q = 1 - p
    coeffs = [1, 7, 28, 84, 210, 462]  # C(6+k, k) for k=0..5
    pre = p**7 * sum(c * q**k for k, c in enumerate(coeffs))  # Win 7-k, k<=5
    deuce = 924 * p**6 * q**6 * p**2 / (p**2 + q**2)  # Reach 6-6, then win
    return pre + deuce


def set_prob(pa, pb):
    """P(A wins set) via DP over game scores. A serves first, TB at 6-6."""
    ga = game_prob(pa)      # P(A holds serve)
    br = 1 - game_prob(pb)  # P(A breaks B)

    dp = [[None] * 8 for _ in range(8)]
    # Terminals: A won
    for b in range(8): dp[7][b] = torch.ones_like(pa)
    for b in range(5): dp[6][b] = torch.ones_like(pa)
    # Terminals: B won
    for a in range(7): dp[a][7] = torch.zeros_like(pa)
    for a in range(5): dp[a][6] = torch.zeros_like(pa)
    dp[6][6] = tiebreak_prob(pa, pb)

    # Fill backwards by total games (11 down to 0)
    for total in range(11, -1, -1):
        for a in range(min(total, 6), max(total - 6, 0) - 1, -1):
            b = total - a
            if dp[a][b] is not None:
                continue
            pw = ga if (a + b) % 2 == 0 else br  # A serves even-numbered games
            dp[a][b] = pw * dp[a + 1][b] + (1 - pw) * dp[a][b + 1]
    return dp[0][0]


def match_prob(pa, pb, best_of):
    """P(A wins match). Handles mixed bo3/bo5 batches via masking."""
    s = set_prob(pa, pb)
    bo3 = s**2 * (3 - 2 * s)
    bo5 = s**3 * (10 - 15 * s + 6 * s**2)
    mask = (best_of == 5).float()
    return bo3 + mask * (bo5 - bo3)
