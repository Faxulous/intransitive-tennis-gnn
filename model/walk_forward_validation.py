from model.graph.windowstacks import *
import torch
import pandas as pd
import time
from model.seed import set_seed



def _collect_surface_data(meta_stacks, data_type, mode='classification'):
    """Collect training/validation data from all surfaces.

    Regression mode uses precomputed game_error column (A_game_pct residual from PSA_ip).
    Pointwise mode deduplicates to match level (data is expanded per set).
    """
    surface_data = {}

    for surf_name, surf_meta_stack in meta_stacks.items():
        stack_list = getattr(surf_meta_stack, data_type)
        if stack_list:
            merged_stack = merge_window_stacks(stack_list)
            if merged_stack and merged_stack.matches.size(0) > 0:
                if mode == 'pointwise':
                    # Deduplicate to match level (one row per match, not per set)
                    import numpy as np
                    _, idx = np.unique(merged_stack.match_ids.cpu().numpy(), return_index=True)
                    idx = torch.tensor(sorted(idx), device=merged_stack.match_ids.device)
                    surface_data[surf_name] = {
                        'query_edges': merged_stack.matches[idx],
                        'y': merged_stack.results[idx],
                        'best_of': merged_stack.best_of[idx],
                        'market_probs': merged_stack.market_probs[idx],
                    }
                else:
                    y = merged_stack.game_error if mode == 'regression' else merged_stack.set_results
                    surface_data[surf_name] = {
                        'query_edges': merged_stack.matches,
                        'y': y,
                        'market_probs': merged_stack.market_probs
                    }

    return surface_data


def precompute_walk_forward_data(df, gender, g_pct=0.5, t_pct=0.1, v_pct=0.1):
    """Precompute data splits + tensor conversion (identical across tuning trials)."""
    surface_splits = split_df_by_surface(df, g_pct, t_pct, v_pct)
    meta_stacks = create_meta_stacks(surface_splits, gender)
    test_dfs = [sd['test'] for sd in surface_splits.values() if len(sd['test']) > 0]
    pred_df = pd.concat(test_dfs, ignore_index=True).sort_values(['window_idx', 'match_id']).reset_index(drop=True)
    # Store WindowStack lists; cascade_left mutates lists, not WindowStacks themselves
    stacks = {
        surf: {k: list(getattr(sms, k)) for k in ('expired', 'graph', 'train', 'val', 'test')}
        for surf, sms in meta_stacks.items()
    }
    return {'stacks': stacks, 'pred_df': pred_df}


def walk_forward_validation(graph_engine, df, gender, g_pct=0.5, t_pct=0.1, v_pct=0.1,
                           profit_mode=False, theta=0.0, seed=8,
                           mode='classification', verbose=True, compute_intransitivity=False,
                           initial_epochs=150, subsequent_epochs=30, retrain_every=38,
                           cached_data=None):
    """Walk forward validation with multi-surface training and global chronological splits.

    Args:
        mode: 'classification' (predict match winner) or 'regression' (predict game_pct residual).
              Regression uses precomputed 'game_error' column in the data.
        verbose: Print progress info (disable for tuning loops).
        compute_intransitivity: Compute per-match I(A) values (expensive, not needed for tuning).
        initial_epochs: Epochs for first training pass.
        subsequent_epochs: Epochs for retraining passes.
    """
    _print = print if verbose else lambda *a, **k: None
    # Set seed for reproducibility
    set_seed(seed)

    _print("Starting walk forward validation setup...")
    setup_start_time = time.time()

    if cached_data:
        # Reconstruct fresh meta_stacks from cached WindowStacks (skip split + tensor conversion)
        meta_stacks = {
            surf: SurfaceMetaStack(
                list(s['expired']), list(s['graph']), list(s['train']),
                list(s['val']), list(s['test'])
            )
            for surf, s in cached_data['stacks'].items()
        }
        pred_df = cached_data['pred_df'].copy()
    else:
        surface_split_start = time.time()
        surface_splits = split_df_by_surface(df, g_pct, t_pct, v_pct)

        _print("Creating meta stacks...")
        meta_stack_start = time.time()
        meta_stacks = create_meta_stacks(surface_splits, gender)
        _print(f"Meta stacks created in {time.time() - meta_stack_start:.2f} seconds")

        test_dfs = []
        for surface_data in surface_splits.values():
            if len(surface_data['test']) > 0:
                test_dfs.append(surface_data['test'])
        pred_df = pd.concat(test_dfs, ignore_index=True).sort_values(['window_idx', 'match_id']).reset_index(drop=True)
    
    # Create interaction matrices: build graph+train first, copy for val, then add val to pred
    # (avoids double-counting graph+train data in val_pim)
    pim_start = time.time()
    pred_pim = graph_engine.pim
    pred_pim.update_pim_from_stacks(meta_stacks, ['graph', 'train'])
    print(f"      graph+train PIM: {time.time() - pim_start:.1f}s", flush=True)
    pim2 = time.time()
    val_pim = pred_pim.create_matrix_copy()
    gt_time = max(
        (w.timestamp for s in meta_stacks.values() for stage in ('graph', 'train')
         for w in getattr(s, stage) if w is not None), default=None
    )
    val_pim.init_surface_graphs(gt_time)
    pred_pim.update_pim_from_stacks(meta_stacks, ['val'])  # pred_pim = graph+train+val
    pim_total = time.time() - pim_start
    graph_engine._timing_pim = pim_total
    print(f"      copy+val PIM: {time.time() - pim2:.1f}s (total PIM: {pim_total:.1f}s)", flush=True)

    # Initialize for main loop
    all_pred_data = []
    all_window_match_ids = []
    active_stacks = {surf for surf, stk in meta_stacks.items() if stk.test}
    window_counter = 0
    first_training = True
    loop_start = time.time()

    print(f"      starting loop...", flush=True)

    max_win = int(pred_df['window_idx'].max())
    n_test_windows = sum(len(stk.test) for stk in meta_stacks.values())
    print(f"      {n_test_windows} test windows (max_idx={max_win})", flush=True)

    while active_stacks:
        # Pick surface whose next test window has the smallest window_idx
        next_surf, surf_stack_of_interest = min(
            ((surf, meta_stacks[surf]) for surf in active_stacks),
            key=lambda item: item[1].min_window()
        )

        window_idx = torch.unique(surf_stack_of_interest.test[0].windows).item()
        _print(f"Predicting window {int(window_idx)}/{max_win}")

        # Training schedule: full training initially, then every 10th window
        should_train = first_training or (window_counter % retrain_every == 0)

        if should_train:
            epochs = initial_epochs if first_training else subsequent_epochs
            train_type = "Init" if first_training else "Retrain"

            # Collect multi-surface training data
            t_collect = time.time()
            surface_train_data = _collect_surface_data(meta_stacks, 'val', mode=mode)
            t_train = time.time()

            # Multi-surface training loop
            graph_engine.pim = val_pim
            for epoch in range(epochs):
                if surface_train_data:
                    _ = graph_engine.train_multi_surface(surface_train_data)

            print(f"      w{int(window_idx)}: {train_type} {epochs}ep "
                  f"collect={t_train-t_collect:.1f}s train={time.time()-t_train:.1f}s", flush=True)
            first_training = False
        # Prediction phase
        graph_engine.pim = pred_pim
        test_stack = surf_stack_of_interest.test[0]

        if mode == 'pointwise':
            preds, pa, pb = graph_engine.predict_window(
                test_stack.matches, surface=next_surf,
                best_of=test_stack.best_of
            )
        else:
            preds = graph_engine.predict_window(
                test_stack.matches, surface=next_surf
            )

        # Note: We'll calculate intransitivity AFTER updating PIM with current window data
        intransitivity_results = None

        if mode == 'regression':
            pred_data = pd.DataFrame({
                'match_id': test_stack.match_ids.cpu().numpy(),
                'best_of': test_stack.best_of.cpu().numpy(),
                'predicted_residual': preds.cpu().numpy(),
                'window_idx': int(window_idx),
                'surface': next_surf,
            })
        elif mode == 'pointwise':
            pred_data = pd.DataFrame({
                'match_id': test_stack.match_ids.cpu().numpy(),
                'best_of': test_stack.best_of.cpu().numpy(),
                'model_prob': preds[:, 1].cpu().numpy(),
                'pa_serve': pa.cpu().numpy(),  # A's on-serve point win prob
                'pb_serve': pb.cpu().numpy(),  # B's on-serve point win prob
                'window_idx': int(window_idx),
                'surface': next_surf,
            })
        else:
            pred_data = pd.DataFrame({
                'match_id': test_stack.match_ids.cpu().numpy(),
                'best_of': test_stack.best_of.cpu().numpy(),
                'p_win_set': preds[:, 1].cpu().numpy(),
                'window_idx': int(window_idx),
                'surface': next_surf,
            })
        
        # Add essential intransitivity columns (will be populated before PIM update)
        pred_data['intransitivity_measure'] = None
        pred_data['matchup_confidence'] = None
        all_pred_data.append(pred_data)
        
        # Store window match IDs for later probability extraction
        window_match_ids = test_stack.match_ids.unique().cpu().numpy()
        all_window_match_ids.append(window_match_ids)

        # Calculate MATCH-SPECIFIC local intransitivity measures BEFORE updating PIM
        # Skip in lite mode (tuning) — only needed for full evaluation
        n_matches = len(test_stack.matches)
        if not compute_intransitivity:
            local_ia_values = [0.0] * n_matches
            extended_ia_results = {
                'intransitivity_measure': local_ia_values,
                'density': [0.0] * n_matches,
                'total_norm': [0.0] * n_matches,
                'matchup_confidence': [0.0] * n_matches,
            }
        else:
            try:
                extended_ia_results = pred_pim.calculate_match_local_intransitivity_extended(
                    surface=next_surf,
                    query_edges=test_stack.matches,
                    k_hops=1, min_common=1
                )
                local_ia_values = extended_ia_results['intransitivity_measure']
                _print(f"DEBUG: Window {int(window_idx)} calculated {len(local_ia_values)} match-specific I(A) values")
                if local_ia_values:
                    ia_range = f"[{min(local_ia_values):.3f}, {max(local_ia_values):.3f}]"
                    ia_mean = sum(local_ia_values) / len(local_ia_values)
                    _print(f"DEBUG: Local I(A) range: {ia_range}, mean: {ia_mean:.3f}")

                try:
                    confidence_values = pred_pim.get_match_confidence_values(
                        surface=next_surf, query_edges=test_stack.matches
                    )
                    extended_ia_results['matchup_confidence'] = confidence_values
                except Exception as e:
                    _print(f"DEBUG: Confidence value extraction failed for window {int(window_idx)}: {e}")
                    extended_ia_results['matchup_confidence'] = [0.0] * n_matches
            except Exception as e:
                _print(f"ERROR: Local intransitivity calculation failed for window {int(window_idx)}: {e}")
                local_ia_values = [0.0] * n_matches
                extended_ia_results = {
                    'intransitivity_measure': local_ia_values,
                    'density': [0.0] * n_matches,
                    'total_norm': [0.0] * n_matches,
                    'matchup_confidence': [0.0] * n_matches,
                }

        # NOW update PIMs with new data (AFTER calculating intransitivity)
        pred_pim.update_interaction_matrix(next_surf, test_stack)
        pred_pim.create_surface_graph(next_surf, test_stack.timestamp)

        if surf_stack_of_interest.val:
            val_pim.update_interaction_matrix(next_surf, surf_stack_of_interest.val[0])
            val_pim.create_surface_graph(next_surf, surf_stack_of_interest.val[0].timestamp)
        
        # Update the prediction dataframe with essential intransitivity values only
        if len(all_pred_data) > 0 and len(local_ia_values) == len(test_stack.matches):
            current_pred_data = all_pred_data[-1]  # Get the most recent prediction data
            current_pred_data['intransitivity_measure'] = extended_ia_results['intransitivity_measure']
            current_pred_data['matchup_confidence'] = extended_ia_results['matchup_confidence']

        # Cascade to next window
        surf_stack_of_interest.cascade_left()
        if not surf_stack_of_interest.test:
            active_stacks.remove(next_surf)
        window_counter += 1
        if window_counter % 50 == 0:
            print(f"      w{int(window_idx)}: {window_counter} windows done, "
                  f"{time.time()-loop_start:.1f}s elapsed", flush=True)
    # Process predictions
    non_empty_pred_data = [df for df in all_pred_data if not df.empty]
    if non_empty_pred_data:
        all_pred_data_df = pd.concat(non_empty_pred_data, ignore_index=True)
    else:
        col = 'predicted_residual' if mode == 'regression' else ('model_prob' if mode == 'pointwise' else 'p_win_set')
        all_pred_data_df = pd.DataFrame(columns=['match_id', 'best_of', col])

    if not all_pred_data_df.empty:
        if mode == 'regression':
            # Average predicted residuals per match (expanded by sets)
            pred_residuals = all_pred_data_df.groupby('match_id')['predicted_residual'].mean()
            predicted_ids = set(pred_residuals.index)
            pred_df = pred_df[pred_df['match_id'].isin(predicted_ids)].reset_index(drop=True)
            pred_df['predicted_residual'] = pred_df['match_id'].map(pred_residuals)
        elif mode == 'pointwise':
            # Predictions are already match-level; deduplicate by match_id
            dedup = all_pred_data_df.drop_duplicates('match_id').set_index('match_id')
            predicted_ids = set(dedup.index)
            pred_df = pred_df[pred_df['match_id'].isin(predicted_ids)].reset_index(drop=True)
            for col in ('model_prob', 'pa_serve', 'pb_serve'):
                pred_df[col] = pred_df['match_id'].map(dedup[col])
        else:
            match_probs_df = all_pred_data_df.groupby('match_id', group_keys=False).apply(
                lambda x: calculate_match_prob(x['p_win_set'].mean(), x['best_of'].iloc[0]),
                include_groups=False,
            )
            predicted_ids = set(match_probs_df.index)
            pred_df = pred_df[pred_df['match_id'].isin(predicted_ids)].reset_index(drop=True)
            pred_df['model_prob'] = pred_df['match_id'].map(match_probs_df)
    else:
        if mode == 'regression':
            pred_df['predicted_residual'] = 0.0
        else:
            pred_df['model_prob'] = 0.5
    
    # Add intransitivity information to the main prediction dataframe
    if not all_pred_data_df.empty:
        # Handle potential duplicate match_ids by taking the first occurrence of each match
        extended_data_unique = all_pred_data_df.drop_duplicates(subset=['match_id'], keep='first')

        # Add intransitivity columns - only the ones we're actually populating
        intransitivity_cols = ['intransitivity_measure',
                              'matchup_confidence']

        # Check which columns exist and add them
        existing_cols = [col for col in intransitivity_cols if col in extended_data_unique.columns]

        if existing_cols:
            intransitivity_lookup = extended_data_unique.set_index('match_id')[existing_cols].to_dict('index')
            for col in existing_cols:
                if 'intransitivity_measure' in col:
                    default_val = 1.0
                elif col == 'matchup_confidence':
                    default_val = 0.0
                else:
                    default_val = 0.0
                pred_df[col] = pred_df['match_id'].map(lambda x: intransitivity_lookup.get(x, {}).get(col, default_val))
        else:
            # Add default intransitivity columns if not available
            pred_df['intransitivity_measure'] = 1.0
            pred_df['matchup_confidence'] = 0.0
        
        # Print intransitivity summary (only if actually computed)
        if compute_intransitivity and 'intransitivity_measure' in pred_df.columns:
            avg_intransitivity = pred_df['intransitivity_measure'].mean()
            _print(f"\nIntransitivity analysis:")
            _print(f"Average I(A) measure: {avg_intransitivity:.3f}")
            if avg_intransitivity < 1.0:
                _print("Overall behavior: Predominantly transitive")
            elif avg_intransitivity > 1.0:
                _print("Overall behavior: Effectively intransitive")
            else:
                _print("Overall behavior: Balanced")
    else:
        # Add default columns if no prediction data available
        pred_df['intransitivity_measure'] = 0.0
        pred_df['matchup_confidence'] = 0.0
    
    graph_engine._timing_loop = time.time() - loop_start  # Expose for callers
    return pred_df


def calculate_match_prob(p, best_of):
    if best_of == 3:
        return 3 * p**2 - 2 * p**3
    elif best_of == 5:
        return 10 * p**3 - 15 * p**4 + 6 * p**5
    else:
        return p # Should not happen for tennis matches

