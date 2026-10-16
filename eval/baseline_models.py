import choix
import numpy as np
import pandas as pd


def calculate_elo_predictions(df, initial_elo=1500):
    """Vectorised Elo: surface-specific, all-surface, and game-weighted variants.
    Uses numpy arrays instead of iterrows for ~10-20x speedup."""
    df = df.sort_index().copy()
    n = len(df)

    # Pre-extract columns as numpy arrays (avoids pandas overhead per row)
    pa = df['Aname'].values
    pb = df['Bname'].values
    won = df['Awin'].values.astype(float)
    surf = df['Surface'].values
    ag = df['A_games'].values if 'A_games' in df.columns else np.ones(n)
    bg = df['B_games'].values if 'B_games' in df.columns else np.ones(n)

    # Output arrays
    esp, eap, ewp = np.empty(n), np.empty(n), np.empty(n)
    aes, bes, aea, bea, aew, bew = [np.empty(n) for _ in range(6)]

    # State dicts
    r_surf, r_all, r_wt, counts = {}, {}, {}, {}

    for i in range(n):
        a, b, w, s = pa[i], pb[i], won[i], surf[i]

        # Dynamic K factors: K_i = 250 / (N_i + 5)^0.4
        ca = counts.get(a, 0); counts[a] = ca + 1
        cb = counts.get(b, 0); counts[b] = cb + 1
        ka, kb = 250 / ((ca + 5) ** 0.4), 250 / ((cb + 5) ** 0.4)

        # Surface Elo
        ra = r_surf.get(a, {}).get(s, initial_elo)
        rb = r_surf.get(b, {}).get(s, initial_elo)
        aes[i], bes[i] = ra, rb
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        esp[i] = p
        r_surf.setdefault(a, {})[s] = ra + ka * (w - p)
        r_surf.setdefault(b, {})[s] = rb + kb * (p - w)

        # All-surface Elo
        ra = r_all.get(a, initial_elo); rb = r_all.get(b, initial_elo)
        aea[i], bea[i] = ra, rb
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        eap[i] = p
        r_all[a] = ra + ka * (w - p); r_all[b] = rb + kb * (p - w)

        # Weighted Elo (game-pct scaled K)
        ra = r_wt.get(a, initial_elo); rb = r_wt.get(b, initial_elo)
        aew[i], bew[i] = ra, rb
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        ewp[i] = p
        total = ag[i] + bg[i]
        gp = (ag[i] / total if w else bg[i] / total) if total > 0 else 1.0
        r_wt[a] = ra + ka * gp * (w - p); r_wt[b] = rb + kb * gp * (p - w)

    df['elo_surf_prob'] = esp; df['elo_all_prob'] = eap; df['elo_weighted_prob'] = ewp
    df['A_elo_surf'] = aes; df['B_elo_surf'] = bes
    df['A_elo_all'] = aea; df['B_elo_all'] = bea
    df['A_elo_weighted'] = aew; df['B_elo_weighted'] = bew
    return df


def calc_bradley_terry_predictions(df):
    """Bradley-Terry with rolling 2-year window, fitted once per unique training
    window via choix ILSR (regularised ML, alpha=0.01). The alpha regularisation
    couples every player pair, bounding ratings of short perfect records; a plain
    per-player-regularised MM fixed-point lets them blow up to extreme probs."""
    import time
    df = df.sort_values('Date').reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    n = len(df)
    bt_probs = np.full(n, 0.5)

    # Pre-extract as numpy arrays
    dates = df['Date'].values                    # datetime64[ns]
    names_a = df['Aname'].values
    names_b = df['Bname'].values
    wins = df['Awin'].values
    widxs = df['window_idx'].values
    two_years = np.timedelta64(2 * 365, 'D')

    # Pre-compute group indices per window (avoids O(n) np.where per window)
    order = np.argsort(widxs, kind='stable')
    sorted_w = widxs[order]
    split_pts = np.flatnonzero(np.diff(sorted_w)) + 1
    groups = np.split(order, split_pts)          # groups[i] = indices for window i
    n_windows = len(groups)

    # Cache: windows starting on same date share identical training data
    last_lo, last_hi, params, p2i = -1, -1, None, {}
    n_fits, t0 = 0, time.time()

    for wi, group_idx in enumerate(groups):
        if wi % 200 == 0:
            elapsed = time.time() - t0
            print(f"    BT window {wi}/{n_windows} ({elapsed:.0f}s, {n_fits} fits)")

        window_start = dates[group_idx[0]]

        # Training data: [window_start - 2yr, window_start) via searchsorted
        train_start = window_start - two_years
        lo = int(np.searchsorted(dates, train_start, side='left'))
        hi = int(np.searchsorted(dates, window_start, side='left'))
        if lo >= hi:
            continue

        # Only refit if training window changed
        if lo != last_lo or hi != last_hi:
            last_lo, last_hi = lo, hi
            n_fits += 1
            t_a, t_b, t_w = names_a[lo:hi], names_b[lo:hi], wins[lo:hi]
            all_players = list(set(t_a) | set(t_b))
            if len(all_players) < 2:
                params, p2i = None, {}
                continue
            p2i = {p: i for i, p in enumerate(all_players)}
            n_p = len(all_players)

            # Build winner/loser index arrays (vectorised)
            a_idx = np.array([p2i[x] for x in t_a])
            b_idx = np.array([p2i[x] for x in t_b])
            w_arr = np.where(t_w == 1, a_idx, b_idx)
            l_arr = np.where(t_w == 1, b_idx, a_idx)

            # Regularised ML via choix ILSR; alpha couples all pairs (no blowups)
            params = choix.ilsr_pairwise(n_p, list(zip(w_arr.tolist(), l_arr.tolist())), alpha=0.01)

        if params is None:
            continue

        # Vectorised probability: P(A>B) = sigmoid(params[A] - params[B])
        ga, gb = names_a[group_idx], names_b[group_idx]
        ai = np.array([p2i.get(a, -1) for a in ga])
        bi = np.array([p2i.get(b, -1) for b in gb])
        valid = (ai >= 0) & (bi >= 0)
        if valid.any():
            diff = params[ai[valid]] - params[bi[valid]]
            bt_probs[group_idx[valid]] = 1.0 / (1.0 + np.exp(-diff))

    print(f"    BT done: {n_fits} unique fits in {time.time()-t0:.0f}s")
    df['bt_prob'] = bt_probs
    return df
