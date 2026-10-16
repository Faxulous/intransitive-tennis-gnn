import csv
import math
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_permutation_variance(query_edges):
    # Example: ensure both (u, v) and (v, u) exist
    reversed_edges = query_edges.flip(0)
    return torch.cat((query_edges, reversed_edges), dim=1)


def node_feature_matrix(surf_mode, gender="men", size_limit=None):
    """Create node feature matrix for all players"""
    # Load player data
    player_data = load_player_data(gender, surf_mode)
    
    # Get maximum index to determine matrix size
    max_idx = max(data['index'] for data in player_data.values())
    n_nodes = min(size_limit, max_idx + 1) if size_limit else max_idx + 1
    
    # Initialize feature matrix (4 player features)
    feature_matrix = torch.zeros((n_nodes, 4))
    
    # Fill in features for each player
    for player_info in player_data.values():
        idx = player_info['index']
        if idx < n_nodes:  # Only include players within size limit
            # Set player features
            feature_matrix[idx] = torch.tensor(player_info['feature_vector'])
    print(feature_matrix)
    return feature_matrix


surface_map = {
    0: "All", 1: "Hard", 2: "Clay", 3: "Grass",
    "All": 0, "Hard": 1, "Clay": 2, "Grass": 3
}


def load_player_data(gender, surf_mode="one-gnn-three-graphs"):
    """Load player data from CSV and create feature vectors"""
    # Load player data from CSV
    file_path = f"data/player_data/{gender}.csv" if not "test" in gender else f"data/player_data/men_test.csv"
    df = pd.read_csv(file_path)

    # Define features
    numerical_to_scale = ['height', 'weight', 'bmi']
    binary_feature = ['righthanded']
    attrs = numerical_to_scale + binary_feature
    present = [c for c in attrs if c in df.columns]
    if not present:  # public package: tennisexplorer.com terms forbid publishing them -> zeros (final model zeroes them anyway)
        print(f"Note: {file_path} has no player attributes; using zeros")
        processed_features = np.zeros((len(df), len(attrs)))
    elif len(present) < len(attrs):
        raise KeyError(f"{file_path} has only some player attributes: {present}")
    else:
        # Fill NaNs with medians first
        df[attrs] = df[attrs].fillna(df[attrs].median())
        scaled_features = StandardScaler().fit_transform(df[numerical_to_scale])
        # Combine scaled features with binary feature (not scaled)
        processed_features = np.hstack([scaled_features, df[binary_feature].values.reshape(-1, 1)])
    
    # Create dictionary with player data
    player_data = {}
    for _, row in df.iterrows():
        idx = int(row['index'])
        name = row['player_name']
        
        # Get processed features for this player
        feature_vector = processed_features[idx].tolist()
        
        player_data[name] = {
            'index': idx,
            'name': name,
            'feature_vector': feature_vector
        }
        
        # If using surface-specific names in one-gnn-one-graph-see-surface mode
        if surf_mode == "one-gnn-one-graph-see-surface":
            for surface in ['Hard', 'Clay', 'Grass']:
                surface_name = f"{name}_{surface}"
                player_data[surface_name] = {
                    'index': idx,
                    'name': name,
                    'feature_vector': feature_vector
                }
    
    return player_data


def add_player_match_counts(df):  # Time: O(N), Space: O(P)
    """Add A/B match counts (previous matches played) and experience categories."""
    result_df = df.copy()
    counts = {}
    a_counts, b_counts = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
    for i, (a, b) in enumerate(zip(df['Aname'].values, df['Bname'].values)):
        a_counts[i], b_counts[i] = counts.get(a, 0), counts.get(b, 0)  # pre-match counts
        counts[a], counts[b] = a_counts[i] + 1, b_counts[i] + 1
    result_df['A_match_count'], result_df['B_match_count'] = a_counts, b_counts
    result_df['match_count_diff'] = np.abs(a_counts - b_counts)
    result_df['A_experience'] = result_df['A_match_count'].apply(_categorize_experience)
    result_df['B_experience'] = result_df['B_match_count'].apply(_categorize_experience)
    result_df['match_experience_diff'] = abs(
        result_df['A_experience'].astype(int) - result_df['B_experience'].astype(int)
    )
    return result_df


def _categorize_experience(count):
    """Bucket player experience: 0=newcomer (<5), 1=low (<15), 2=mod (<30), 3=exp (<50), 4=veteran."""
    return 0 if count < 5 else 1 if count < 15 else 2 if count < 30 else 3 if count < 50 else 4
