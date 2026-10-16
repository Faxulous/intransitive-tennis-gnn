"""
Intransitivity measurement implementation based on:
"Elo Ratings in the Presence of Intransitivity" by Hamilton, Kalenkova, and Roughan

This module implements the I(A) intransitivity measure that quantifies the degree
of intransitivity in pairwise comparison data using combinatorial Hodge theory.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Union, Optional, List
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import pandas as pd


def combinatorial_gradient(v: np.ndarray) -> np.ndarray:
    """
    Combinatorial gradient operator: grad(v)_ij = v_i - v_j
    
    Args:
        v: Vector of length m (player ratings/scores)
        
    Returns:
        Matrix M where M_ij = v_i - v_j
    """
    v = v.reshape(-1, 1)  # Column vector
    return v - v.T  # Broadcasting creates the difference matrix


def combinatorial_divergence(A: np.ndarray) -> np.ndarray:
    """
    Combinatorial divergence operator: div(A) = row_means(A)
    
    Args:
        A: Matrix of shape (m, m)
        
    Returns:
        Vector of row means: div(A)_i = mean(A[i, :])
    """
    return np.mean(A, axis=1)


def interaction_to_advantage_matrix(interaction_matrix: Union[dok_matrix, csr_matrix, np.ndarray],
                                   clip_epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert interaction probability matrix to advantage matrix using logit transformation.
    A_ij = log(P_ij / (1 - P_ij)) = logit(P_ij)
    
    Args:
        interaction_matrix: Sparse or dense matrix of probabilities
        clip_epsilon: Small value to avoid log(0) or log(1)
        
    Returns:
        Dense advantage matrix A
    """
    # Convert to dense array if sparse
    if hasattr(interaction_matrix, 'toarray'):
        P = interaction_matrix.toarray()
    else:
        P = np.array(interaction_matrix)
    
    # Edge weights are |posterior| in (0, 0.5]; shift to probability space for logit
    A = np.zeros_like(P)
    meaningful_mask = P > clip_epsilon
    if np.any(meaningful_mask):
        prob = np.minimum(P[meaningful_mask] + 0.5, 1.0 - clip_epsilon)  # Dominance magnitude → probability, cap at 1-eps
        A[meaningful_mask] = np.log(prob / (1.0 - prob))
    
    return A


def calculate_intransitivity_measure(A: np.ndarray) -> Dict[str, float]:
    """
    Calculate the I(A) intransitivity measure:
    I(A) = (1 + ||A - grad∘div(A)||_F) / (1 + ||grad∘div(A)||_F)
    Exactly Eq. 8 of Hamilton, Kalenkova & Roughan (PLOS ONE 2025): equals 1
    when cyclic and transitive components have equal magnitude.
    
    Args:
        A: Advantage matrix (m x m)
        
    Returns:
        Dictionary containing:
        - intransitivity_measure: I(A) value
        - transitive_component: ||grad∘div(A)||_F
        - cyclic_component: ||A - grad∘div(A)||_F
        - interpretation: String describing predominant behavior
    """
    # Check for and handle NaN values in input
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply combinatorial divergence then gradient
    div_A = combinatorial_divergence(A)
    
    # Handle potential NaN/inf in divergence
    if np.any(np.isnan(div_A)) or np.any(np.isinf(div_A)):
        div_A = np.nan_to_num(div_A, nan=0.0, posinf=0.0, neginf=0.0)
    
    grad_div_A = combinatorial_gradient(div_A)
    
    # Handle potential NaN/inf in gradient
    if np.any(np.isnan(grad_div_A)) or np.any(np.isinf(grad_div_A)):
        grad_div_A = np.nan_to_num(grad_div_A, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate the cyclic component: A - grad∘div(A)
    cyclic_component_matrix = A - grad_div_A
    
    # Handle potential NaN/inf in cyclic component
    if np.any(np.isnan(cyclic_component_matrix)) or np.any(np.isinf(cyclic_component_matrix)):
        cyclic_component_matrix = np.nan_to_num(cyclic_component_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate Frobenius norms
    transitive_norm = np.linalg.norm(grad_div_A, 'fro')
    cyclic_norm = np.linalg.norm(cyclic_component_matrix, 'fro')
    
    # Handle potential NaN in norms
    if np.isnan(transitive_norm) or np.isinf(transitive_norm):
        transitive_norm = 0.0
    if np.isnan(cyclic_norm) or np.isinf(cyclic_norm):
        cyclic_norm = 0.0
    
    # Calculate I(A) measure with safeguard against division by zero
    denominator = 1 + transitive_norm
    if denominator <= 0 or np.isnan(denominator) or np.isinf(denominator):
        denominator = 1.0
    
    intransitivity_measure = (1 + cyclic_norm) / denominator
    
    # Final safeguard for the result
    if np.isnan(intransitivity_measure) or np.isinf(intransitivity_measure):
        intransitivity_measure = 1.0
    
    # Interpretation
    if intransitivity_measure < 1.0:
        interpretation = "predominantly_transitive"
    elif intransitivity_measure > 1.0:
        interpretation = "effectively_intransitive"  
    else:
        interpretation = "balanced"
    
    return {
        'intransitivity_measure': float(intransitivity_measure),
        'transitive_component': float(transitive_norm),
        'cyclic_component': float(cyclic_norm), 
        'interpretation': interpretation
    }


def calculate_intransitivity_measure_extended(A: np.ndarray, interaction_mask: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate extended intransitivity measures including density and proportional normalizations.
    
    Args:
        A: Advantage matrix (m x m)
        interaction_mask: Binary mask indicating which entries have actual interactions
        
    Returns:
        Dictionary containing original plus normalized measures:
        - All fields from calculate_intransitivity_measure()
        - intransitivity_measure_proportional: Proportional cyclic/total measure
        - density: Graph density used for normalization
    """
    # Get basic measures
    basic_results = calculate_intransitivity_measure(A)
    
    transitive_norm = basic_results['transitive_component']
    cyclic_norm = basic_results['cyclic_component']
    
    # Calculate density
    if interaction_mask is not None:
        density = np.count_nonzero(interaction_mask) / interaction_mask.size
    else:
        # Estimate density from non-zero entries in A
        density = np.count_nonzero(A) / A.size
    
    density = max(density, 1e-6)  # Avoid division by zero
    
    # Density scale for reference (no longer used for density_normalized)
    density_scale = np.sqrt(density)
    
    # Proportional intransitivity (cyclic as proportion of total)
    total_norm = np.linalg.norm(A, 'fro')
    if total_norm < 1e-6:
        intransitivity_proportional = 0.0
    else:
        intransitivity_proportional = cyclic_norm / total_norm
    
    # Combine results
    extended_results = basic_results.copy()
    extended_results.update({
        'intransitivity_measure_proportional': float(intransitivity_proportional),
        'density': float(density),
        'total_norm': float(total_norm)
    })
    
    return extended_results


def extract_submatrix(matrix: Union[dok_matrix, csr_matrix], indices: list) -> np.ndarray:
    """
    Extract a submatrix from a sparse matrix using the given row/column indices.
    
    Args:
        matrix: Sparse matrix (dok_matrix or csr_matrix)
        indices: List of row/column indices to extract
        
    Returns:
        Dense numpy array of the extracted submatrix
    """
    if len(indices) == 0:
        return np.array([])
    
    # Convert to CSR for efficient slicing
    if not hasattr(matrix, 'tocsr'):
        matrix_csr = matrix
    else:
        matrix_csr = matrix.tocsr()
    
    # Extract the submatrix
    submatrix = matrix_csr[np.ix_(indices, indices)]
    return submatrix.toarray()


def calculate_match_local_intransitivity_vectorized_extended(interaction_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                                          surface: str,
                                                          query_edges: np.ndarray,
                                                          k_hops: int = 1,
                                                          min_common: int = 1) -> Dict[str, List[float]]:
    """
    Extended version that calculates local intransitivity with normalized measures.
    Returns all measures (basic, density normalized, proportional) for each match.
    
    Args:
        interaction_matrices: Dict mapping surface names to interaction matrices
        surface: Surface to analyze
        query_edges: Array of shape (n_matches, 2) containing match pairs
        k_hops: Number of hops for neighborhood (1 for immediate triads) 
        min_common: Minimum number of common opponents needed for calculation
        
    Returns:
        Dictionary containing lists of measures:
        - 'intransitivity_measure': Basic I(A) values
        - 'intransitivity_measure_proportional': Proportional values
        - 'density': Graph density values
        - 'total_norm': Total norm values
    """
    if surface not in interaction_matrices or len(query_edges) == 0:
        empty_result = {
            'intransitivity_measure': [0.0] * len(query_edges),
            'intransitivity_measure_proportional': [0.0] * len(query_edges),
            'density': [0.0] * len(query_edges),
            'total_norm': [0.0] * len(query_edges)
        }
        return empty_result
    
    matrix_csr = interaction_matrices[surface].tocsr()
    results = {
        'intransitivity_measure': [],
        'intransitivity_measure_proportional': [],
        'density': [],
        'total_norm': []
    }
    
    def get_k_hop_neighbors(player_id, k):
        """Get all players within k hops of the given player."""
        if k <= 0:
            return set()
        
        # Start with direct connections (k=1)
        current_level = set()
        outgoing = set(matrix_csr[player_id].nonzero()[1])
        incoming = set(matrix_csr[:, player_id].nonzero()[0])
        current_level = outgoing | incoming
        
        all_neighbors = current_level.copy()
        
        # Expand to k hops
        for hop in range(2, k + 1):
            next_level = set()
            for neighbor in current_level:
                if neighbor < matrix_csr.shape[0]:
                    neighbor_outgoing = set(matrix_csr[neighbor].nonzero()[1])
                    neighbor_incoming = set(matrix_csr[:, neighbor].nonzero()[0])
                    next_level.update(neighbor_outgoing | neighbor_incoming)
            
            all_neighbors.update(next_level)
            current_level = next_level
        
        # Remove self-connections
        all_neighbors.discard(player_id)
        return all_neighbors
    
    for a, b in query_edges:
        try:
            # Get k-hop neighborhoods
            a_connected = get_k_hop_neighbors(int(a), k_hops)
            b_connected = get_k_hop_neighbors(int(b), k_hops)
            
            # Find common connections
            common = list(a_connected & b_connected)
            
            if len(common) >= min_common:
                # Create local subgraph with A, B, and common opponents
                submatrix_indices = [a, b] + common
                submatrix = extract_submatrix(matrix_csr, submatrix_indices)
                
                if submatrix.size > 0 and submatrix.shape[0] >= 3:
                    # Calculate local I(A) on the subgraph with extended measures
                    advantage_matrix = interaction_to_advantage_matrix(submatrix)
                    # Create interaction mask for density calculation
                    interaction_mask = (submatrix != 0).toarray() if hasattr(submatrix, 'toarray') else (submatrix != 0)
                    local_result = calculate_intransitivity_measure_extended(advantage_matrix, interaction_mask)
                    
                    results['intransitivity_measure'].append(local_result['intransitivity_measure'])
                    results['intransitivity_measure_proportional'].append(local_result['intransitivity_measure_proportional'])
                    results['density'].append(local_result['density'])
                    results['total_norm'].append(local_result['total_norm'])
                else:
                    results['intransitivity_measure'].append(0.0)
                    results['intransitivity_measure_proportional'].append(0.0)
                    results['density'].append(0.0)
                    results['total_norm'].append(0.0)
            else:
                results['intransitivity_measure'].append(0.0)
                results['intransitivity_measure_proportional'].append(0.0)
                results['density'].append(0.0)
                results['total_norm'].append(0.0)
                
        except Exception as e:
            print(f"Warning: Extended local intransitivity calculation failed for edge ({a}, {b}): {e}")
            results['intransitivity_measure'].append(0.0)
            results['intransitivity_measure_proportional'].append(0.0)
            results['density'].append(0.0)
            results['total_norm'].append(0.0)
    
    return results


def calculate_match_local_intransitivity_vectorized(interaction_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                                  surface: str,
                                                  query_edges: np.ndarray,
                                                  k_hops: int = 1,
                                                  min_common: int = 1) -> List[float]:
    """
    Optimized version that calculates local intransitivity for matches on a specific surface.
    Pre-computes neighbor sets for efficiency when processing multiple matches.
    
    Args:
        interaction_matrices: Dict mapping surface names to interaction matrices
        surface: Surface to analyze
        query_edges: Array of shape (n_matches, 2) containing match pairs
        k_hops: Number of hops for neighborhood (1 for immediate triads) 
        min_common: Minimum number of common opponents needed for calculation
        
    Returns:
        List of local intransitivity measures, one per match
    """
    if surface not in interaction_matrices or len(query_edges) == 0:
        return [0.0] * len(query_edges)
    
    matrix_csr = interaction_matrices[surface].tocsr()
    results = []
    
    # Pre-compute player connections based on k_hops
    player_connections = {}
    max_player_id = max(query_edges.max(), matrix_csr.shape[0] - 1) if len(query_edges) > 0 else 0
    
    def get_k_hop_neighbors(player_id, k):
        """Get all players within k hops of the given player."""
        if k <= 0:
            return set()
        
        # Start with direct connections (k=1)
        current_level = set()
        outgoing = set(matrix_csr[player_id].nonzero()[1])
        incoming = set(matrix_csr[:, player_id].nonzero()[0])
        current_level = outgoing | incoming
        
        all_neighbors = current_level.copy()
        
        # Expand to k hops
        for hop in range(2, k + 1):
            next_level = set()
            for neighbor in current_level:
                if neighbor < matrix_csr.shape[0]:
                    neighbor_outgoing = set(matrix_csr[neighbor].nonzero()[1])
                    neighbor_incoming = set(matrix_csr[:, neighbor].nonzero()[0])
                    next_level.update(neighbor_outgoing | neighbor_incoming)
            
            # Remove already visited nodes to avoid cycles
            next_level -= all_neighbors
            next_level.discard(player_id)  # Remove self
            
            all_neighbors.update(next_level)
            current_level = next_level
            
            if not current_level:  # No more neighbors to expand
                break
        
        return all_neighbors
    
    for player_id in range(min(max_player_id + 1, matrix_csr.shape[0])):
        player_connections[player_id] = get_k_hop_neighbors(player_id, k_hops)
    
    for a, b in query_edges:
        try:
            # Get pre-computed connections
            a_connected = player_connections.get(a, set())
            b_connected = player_connections.get(b, set())
            
            # Find common connections
            common = list(a_connected & b_connected)
            
            if len(common) >= min_common:
                # Create local subgraph with A, B, and common opponents
                submatrix_indices = [a, b] + common
                submatrix = extract_submatrix(matrix_csr, submatrix_indices)
                
                if submatrix.size > 0 and submatrix.shape[0] >= 3:
                    # Calculate local I(A) on the subgraph
                    advantage_matrix = interaction_to_advantage_matrix(submatrix)
                    # Create interaction mask for density calculation
                    interaction_mask = (submatrix != 0).toarray() if hasattr(submatrix, 'toarray') else (submatrix != 0)
                    local_result = calculate_intransitivity_measure_extended(advantage_matrix, interaction_mask)
                    results.append(local_result['intransitivity_measure'])
                else:
                    results.append(0.0)
            else:
                results.append(0.0)
                
        except Exception as e:
            print(f"Warning: Local intransitivity calculation failed for edge ({a}, {b}): {e}")
            results.append(0.0)
    
    return results


def calculate_match_local_intransitivity(interaction_matrix: Union[dok_matrix, csr_matrix], 
                                       query_edges: np.ndarray,
                                       k_hops: int = 1,
                                       min_common: int = 1) -> List[float]:
    """
    Calculate local intransitivity measures for each match based on common opponents.
    For each match (A, B), finds common opponents and calculates I(A) on the local subgraph.
    
    Args:
        interaction_matrix: Sparse interaction matrix
        query_edges: Array of shape (n_matches, 2) containing match pairs
        k_hops: Number of hops for neighborhood (1 for immediate triads)
        min_common: Minimum number of common opponents needed for calculation
        
    Returns:
        List of local intransitivity measures, one per match
    """
    if len(query_edges) == 0:
        return []
    
    # Convert to CSR for efficient operations
    if hasattr(interaction_matrix, 'tocsr'):
        matrix_csr = interaction_matrix.tocsr()
    else:
        matrix_csr = interaction_matrix
    
    results = []
    
    for a, b in query_edges:
        try:
            # Find neighbors efficiently using sparse matrix operations
            a_neighbors = set(matrix_csr[a].nonzero()[1]) if a < matrix_csr.shape[0] else set()
            b_neighbors = set(matrix_csr[b].nonzero()[1]) if b < matrix_csr.shape[0] else set()
            
            # Also consider who plays against A and B (incoming edges)
            a_opponents = set(matrix_csr[:, a].nonzero()[0]) if a < matrix_csr.shape[1] else set()
            b_opponents = set(matrix_csr[:, b].nonzero()[0]) if b < matrix_csr.shape[1] else set()
            
            # Combine outgoing and incoming relationships for fuller neighborhood
            a_connected = a_neighbors | a_opponents
            b_connected = b_neighbors | b_opponents
            
            # Find common connections (players connected to both A and B)
            common = list(a_connected & b_connected)
            
            if len(common) >= min_common:
                # Create submatrix with A, B, and all common opponents
                submatrix_indices = [a, b] + common
                submatrix = extract_submatrix(matrix_csr, submatrix_indices)
                
                if submatrix.size > 0 and submatrix.shape[0] >= 3:
                    # Convert to advantage matrix and calculate intransitivity
                    advantage_matrix = interaction_to_advantage_matrix(submatrix)
                    local_result = calculate_intransitivity_measure(advantage_matrix)
                    results.append(local_result['intransitivity_measure'])
                else:
                    results.append(0.0)  # Flag value for insufficient data
            else:
                results.append(0.0)  # Flag value when no common opponents
                
        except Exception as e:
            print(f"Warning: Local intransitivity calculation failed for edge ({a}, {b}): {e}")
            results.append(0.0)  # Flag value for failed calculation
    
    return results


def calculate_surface_intransitivity(interaction_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                   surface_weights: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, Dict]]:
    """
    Calculate intransitivity measures for multiple surfaces and compute weighted average.
    
    Args:
        interaction_matrices: Dict mapping surface names to interaction matrices
        surface_weights: Optional weights for each surface (defaults to equal weighting)
        
    Returns:
        Dictionary containing:
        - overall_intransitivity: Weighted average I(A) across surfaces
        - surface_measures: Per-surface intransitivity breakdowns
        - weighted_components: Weighted transitive/cyclic components
    """
    if not interaction_matrices:
        return {
            'overall_intransitivity': 0.0,  # Default flag value
            'surface_measures': {},
            'weighted_components': {'transitive': 0.0, 'cyclic': 0.0}
        }
    
    surface_measures = {}
    
    # Default to equal weights if not provided
    if surface_weights is None:
        surface_weights = {surface: 1.0 / len(interaction_matrices) 
                          for surface in interaction_matrices}
    
    # Normalize weights
    total_weight = sum(surface_weights.values())
    if total_weight > 0:
        surface_weights = {k: v / total_weight for k, v in surface_weights.items()}
    
    weighted_intransitivity = 0.0
    weighted_transitive = 0.0
    weighted_cyclic = 0.0
    weighted_density_normalized = 0.0
    weighted_proportional = 0.0
    weighted_density = 0.0
    weighted_total_norm = 0.0
    
    for surface, matrix in interaction_matrices.items():
        # Check matrix state more thoroughly
        matrix_size = 0
        nonzero_count = 0
        
        if hasattr(matrix, 'nnz'):
            nonzero_count = matrix.nnz
            if hasattr(matrix, 'shape'):
                matrix_size = matrix.shape[0] * matrix.shape[1]
        
        print(f"DEBUG: Surface {surface} - nnz: {nonzero_count}, total size: {matrix_size}")
        
        # Skip truly empty matrices, but be more lenient
        if nonzero_count == 0:
            print(f"DEBUG: Skipping {surface} - no non-zero entries")
            continue
        
        try:
            # Convert to advantage matrix
            A = interaction_to_advantage_matrix(matrix)
            
            # Check if advantage matrix has any meaningful values
            if A is None or A.size == 0:
                print(f"DEBUG: Advantage matrix for {surface} is empty")
                continue
                
            nonzero_advantages = np.count_nonzero(A)
            print(f"DEBUG: Surface {surface} advantage matrix - shape: {A.shape}, nonzero: {nonzero_advantages}")
            
            if nonzero_advantages == 0:
                print(f"DEBUG: Advantage matrix for {surface} has no nonzero entries")
                continue
            
            # Calculate extended intransitivity measures including normalized versions
            # Create interaction mask for density calculation
            interaction_mask = (matrix != 0).toarray() if hasattr(matrix, 'toarray') else (matrix != 0)
            measures = calculate_intransitivity_measure_extended(A, interaction_mask)
            surface_measures[surface] = measures
            
            print(f"DEBUG: Surface {surface} I(A): {measures['intransitivity_measure']:.4f}, "
                  f"proportional: {measures['intransitivity_measure_proportional']:.4f}")
            
            # Add to weighted averages
            weight = surface_weights.get(surface, 0.0)
            weighted_intransitivity += weight * measures['intransitivity_measure']
            weighted_transitive += weight * measures['transitive_component']
            weighted_cyclic += weight * measures['cyclic_component']
            # Removed density_normalized weighted calculation
            weighted_proportional += weight * measures['intransitivity_measure_proportional']
            weighted_density += weight * measures['density']
            weighted_total_norm += weight * measures['total_norm']
            
        except Exception as e:
            print(f"ERROR: Failed to calculate intransitivity for surface {surface}: {e}")
            continue
    
    return {
        'overall_intransitivity': weighted_intransitivity,
        'overall_intransitivity_density_normalized': weighted_density_normalized,
        'overall_intransitivity_proportional': weighted_proportional,
        'overall_density': weighted_density,
        'overall_total_norm': weighted_total_norm,
        'surface_measures': surface_measures,
        'weighted_components': {
            'transitive': weighted_transitive,
            'cyclic': weighted_cyclic
        }
    }


def validate_intransitivity_implementation():
    """
    Validate the implementation using simple examples from the paper.
    Tests both transitive and intransitive scenarios.
    """
    print("Validating intransitivity implementation...")
    
    # Test 1: Perfectly transitive case
    # A beats B with prob 0.7, B beats C with prob 0.6
    # Expected: A should beat C with prob determined by transitivity
    P_transitive = np.array([
        [0.5, 0.7, 0.75],  # A vs [A, B, C]
        [0.3, 0.5, 0.6 ],  # B vs [A, B, C] 
        [0.25, 0.4, 0.5]   # C vs [A, B, C]
    ])
    
    A_trans = interaction_to_advantage_matrix(P_transitive)
    measures_trans = calculate_intransitivity_measure(A_trans)
    print(f"Transitive case - I(A): {measures_trans['intransitivity_measure']:.3f} "
          f"({measures_trans['interpretation']})")
    
    # Test 2: Rock-Paper-Scissors intransitive case
    # A beats B, B beats C, C beats A with equal probability 0.6
    P_intrans = np.array([
        [0.5, 0.6, 0.4],  # A vs [A, B, C]
        [0.4, 0.5, 0.6],  # B vs [A, B, C]
        [0.6, 0.4, 0.5]   # C vs [A, B, C]
    ])
    
    A_intrans = interaction_to_advantage_matrix(P_intrans)
    measures_intrans = calculate_intransitivity_measure(A_intrans)
    print(f"Intransitive case - I(A): {measures_intrans['intransitivity_measure']:.3f} "
          f"({measures_intrans['interpretation']})")
    
    # Test 3: Neutral/balanced case (all 0.5 probabilities)
    P_neutral = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ])
    
    A_neutral = interaction_to_advantage_matrix(P_neutral)
    measures_neutral = calculate_intransitivity_measure(A_neutral)
    print(f"Neutral case - I(A): {measures_neutral['intransitivity_measure']:.3f} "
          f"({measures_neutral['interpretation']})")
    
    print("Validation complete!\n")


# ==========================================
# NORMALIZATION METHODS BASED ON HAMILTON ET AL.
# ==========================================

def calculate_selection_matrix_baselines(validation_data: Dict[str, Dict[str, Union[dok_matrix, csr_matrix]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate expected intransitivity baselines for each gender/surface combination
    using selection matrix Q structure from validation data.
    
    Based on Hamilton et al. Theorem 4.1: Different selection matrices Q lead to
    different final Elo scores when intransitivity is present.
    
    Args:
        validation_data: Dict[gender][surface] -> interaction_matrix
        
    Returns:
        Dict[gender][surface] -> expected_intransitivity_baseline
    """
    baselines = {}
    
    for gender in validation_data:
        baselines[gender] = {}
        for surface, matrix in validation_data[gender].items():
            try:
                # Calculate intransitivity measure for this gender/surface combination
                A = interaction_to_advantage_matrix(matrix)
                if A is not None and A.size > 0:
                    interaction_mask = (matrix != 0).toarray() if hasattr(matrix, 'toarray') else (matrix != 0)
                    measures = calculate_intransitivity_measure_extended(A, interaction_mask)
                    baselines[gender][surface] = measures['intransitivity_measure']
                else:
                    baselines[gender][surface] = 1.0  # Default neutral baseline
            except Exception as e:
                print(f"Warning: Could not calculate baseline for {gender}-{surface}: {e}")
                baselines[gender][surface] = 1.0
                
    return baselines


def calculate_hodge_surface_baselines(validation_data: Dict[str, Dict[str, Union[dok_matrix, csr_matrix]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate surface-specific standard deviations of cyclic and transitive components
    from validation data for Hodge decomposition normalization.
    
    Args:
        validation_data: Dict[gender][surface] -> interaction_matrix
        
    Returns:
        Dict[gender][surface] -> {'cyclic_std': float, 'transitive_std': float}
    """
    baselines = {}
    
    for gender in validation_data:
        baselines[gender] = {}
        for surface, matrix in validation_data[gender].items():
            try:
                A = interaction_to_advantage_matrix(matrix)
                if A is not None and A.size > 0:
                    interaction_mask = (matrix != 0).toarray() if hasattr(matrix, 'toarray') else (matrix != 0)
                    measures = calculate_intransitivity_measure_extended(A, interaction_mask)
                    
                    # Use the components as estimates of standard deviations
                    # In practice, you'd calculate this across multiple tournaments
                    baselines[gender][surface] = {
                        'cyclic_std': max(measures['cyclic_component'], 1.0),
                        'transitive_std': max(measures['transitive_component'], 1.0)
                    }
                else:
                    baselines[gender][surface] = {'cyclic_std': 1.0, 'transitive_std': 1.0}
            except Exception as e:
                print(f"Warning: Could not calculate Hodge baseline for {gender}-{surface}: {e}")
                baselines[gender][surface] = {'cyclic_std': 1.0, 'transitive_std': 1.0}
                
    return baselines


def calculate_connectivity_baselines(validation_data: Dict[str, Dict[str, Union[dok_matrix, csr_matrix]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate spectral properties (second eigenvalue of Laplacian) for gender/surface networks.
    
    Based on Hamilton et al. discussion of graph connectivity affecting final ratings.
    
    Args:
        validation_data: Dict[gender][surface] -> interaction_matrix
        
    Returns:
        Dict[gender][surface] -> lambda_2_baseline (second eigenvalue)
    """
    baselines = {}
    
    for gender in validation_data:
        baselines[gender] = {}
        for surface, matrix in validation_data[gender].items():
            try:
                # Convert to symmetric adjacency matrix
                if hasattr(matrix, 'toarray'):
                    adj = matrix.toarray()
                else:
                    adj = np.array(matrix)
                
                # Make symmetric (undirected graph approximation)
                adj_sym = (adj + adj.T) > 0
                adj_sym = adj_sym.astype(float)
                
                if adj_sym.shape[0] > 2 and np.sum(adj_sym) > 0:
                    # Calculate degree matrix
                    degrees = np.sum(adj_sym, axis=1)
                    D = np.diag(degrees)
                    
                    # Laplacian matrix L = D - A
                    L = D - adj_sym
                    
                    # Get second smallest eigenvalue (connectivity measure)
                    try:
                        eigenvals = eigsh(L, k=min(3, L.shape[0]-1), which='SM', return_eigenvectors=False)
                        lambda_2 = sorted(eigenvals)[1] if len(eigenvals) > 1 else 0.1
                        baselines[gender][surface] = max(lambda_2, 0.01)  # Avoid zero
                    except:
                        baselines[gender][surface] = 0.1  # Default connectivity
                else:
                    baselines[gender][surface] = 0.1
                    
            except Exception as e:
                print(f"Warning: Could not calculate connectivity baseline for {gender}-{surface}: {e}")
                baselines[gender][surface] = 0.1
                
    return baselines


def calculate_elotope_baselines(validation_data: Dict[str, Dict[str, Union[dok_matrix, csr_matrix]]]) -> Dict[str, Dict[str, float]]:
    """
    Estimate Elotope diameter baselines from validation tournaments.
    
    Based on Hamilton et al. concept of Elotope - the set of possible final Elo scores.
    
    Args:
        validation_data: Dict[gender][surface] -> interaction_matrix
        
    Returns:
        Dict[gender][surface] -> elotope_diameter_baseline
    """
    baselines = {}
    
    for gender in validation_data:
        baselines[gender] = {}
        for surface, matrix in validation_data[gender].items():
            try:
                A = interaction_to_advantage_matrix(matrix)
                if A is not None and A.size > 0:
                    # Estimate Elotope diameter as the range of advantage values
                    # This approximates the range of possible final Elo differences
                    finite_mask = np.isfinite(A)
                    if np.any(finite_mask):
                        diameter = np.max(A[finite_mask]) - np.min(A[finite_mask])
                        baselines[gender][surface] = max(diameter, 1.0)
                    else:
                        baselines[gender][surface] = 1.0
                else:
                    baselines[gender][surface] = 1.0
                    
            except Exception as e:
                print(f"Warning: Could not calculate Elotope baseline for {gender}-{surface}: {e}")
                baselines[gender][surface] = 1.0
                
    return baselines


def calculate_selection_bias_baselines(validation_data: Dict[str, Dict[str, Union[dok_matrix, csr_matrix]]]) -> Dict[str, Dict[str, float]]:
    """
    Measure selection bias (stronger players over-representation) by context.
    
    Args:
        validation_data: Dict[gender][surface] -> interaction_matrix
        
    Returns:
        Dict[gender][surface] -> selection_bias_baseline
    """
    baselines = {}
    
    for gender in validation_data:
        baselines[gender] = {}
        for surface, matrix in validation_data[gender].items():
            try:
                if hasattr(matrix, 'toarray'):
                    adj = matrix.toarray()
                else:
                    adj = np.array(matrix)
                
                if adj.size > 0:
                    # Calculate degree distribution to measure selection bias
                    degrees = np.sum(adj > 0, axis=1) + np.sum(adj > 0, axis=0)
                    if len(degrees) > 0 and np.std(degrees) > 0:
                        # Coefficient of variation as bias measure
                        bias = np.std(degrees) / (np.mean(degrees) + 1e-6)
                        baselines[gender][surface] = bias
                    else:
                        baselines[gender][surface] = 0.1
                else:
                    baselines[gender][surface] = 0.1
                    
            except Exception as e:
                print(f"Warning: Could not calculate selection bias baseline for {gender}-{surface}: {e}")
                baselines[gender][surface] = 0.1
                
    return baselines


def apply_selection_matrix_normalization(intransitivity_measure: float, 
                                       gender: str, 
                                       surface: str,
                                       baselines: Dict[str, Dict[str, float]]) -> float:
    """
    Apply Selection Matrix Q-Based Normalization.
    
    I_selection_normalized = I(A) / I_expected(Q_gender_surface)
    
    Args:
        intransitivity_measure: Raw I(A) value
        gender: 'men' or 'women' 
        surface: 'Hard', 'Clay', or 'Grass'
        baselines: Baseline values from validation set
        
    Returns:
        Normalized intransitivity measure
    """
    try:
        baseline = baselines.get(gender, {}).get(surface, 1.0)
        if baseline <= 0:
            baseline = 1.0
        return intransitivity_measure / baseline
    except:
        return intransitivity_measure


def apply_hodge_surface_normalization(cyclic_component: float,
                                    transitive_component: float,
                                    gender: str,
                                    surface: str, 
                                    baselines: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    """
    Apply Surface-Hodge Decomposition Normalization.
    
    I_hodge_surface_norm = (1 + ||A_cyclic||/σ_cyclic_surface) / (1 + ||A_transitive||/σ_transitive_surface)
    
    Args:
        cyclic_component: Cyclic component norm
        transitive_component: Transitive component norm
        gender: 'men' or 'women'
        surface: 'Hard', 'Clay', or 'Grass'
        baselines: Baseline standard deviations from validation set
        
    Returns:
        Normalized intransitivity measure
    """
    try:
        surface_baselines = baselines.get(gender, {}).get(surface, {'cyclic_std': 1.0, 'transitive_std': 1.0})
        cyclic_std = surface_baselines.get('cyclic_std', 1.0)
        transitive_std = surface_baselines.get('transitive_std', 1.0)
        
        numerator = 1 + cyclic_component / cyclic_std
        denominator = 1 + transitive_component / transitive_std
        
        if denominator <= 0:
            denominator = 1.0
            
        return numerator / denominator
    except:
        return (1 + cyclic_component) / (1 + transitive_component)


def apply_connectivity_normalization(intransitivity_measure: float,
                                   current_connectivity: float,
                                   gender: str,
                                   surface: str,
                                   baselines: Dict[str, Dict[str, float]]) -> float:
    """
    Apply Graph Connectivity Normalization.
    
    I_connectivity_norm = I(A) × (λ_2_baseline / λ_2_actual)
    
    Args:
        intransitivity_measure: Raw I(A) value
        current_connectivity: Current λ_2 value
        gender: 'men' or 'women'
        surface: 'Hard', 'Clay', or 'Grass' 
        baselines: Baseline connectivity values
        
    Returns:
        Normalized intransitivity measure
    """
    try:
        baseline_connectivity = baselines.get(gender, {}).get(surface, 0.1)
        if current_connectivity <= 0:
            current_connectivity = 0.1
            
        correction_factor = baseline_connectivity / current_connectivity
        return intransitivity_measure * correction_factor
    except:
        return intransitivity_measure


def apply_elotope_normalization(intransitivity_measure: float,
                               current_diameter: float,
                               gender: str, 
                               surface: str,
                               baselines: Dict[str, Dict[str, float]]) -> float:
    """
    Apply Elotope Size Normalization.
    
    I_elotope_norm = I(A) × (diameter_baseline / diameter_actual)
    
    Args:
        intransitivity_measure: Raw I(A) value
        current_diameter: Current Elotope diameter
        gender: 'men' or 'women'
        surface: 'Hard', 'Clay', or 'Grass'
        baselines: Baseline diameter values
        
    Returns:
        Normalized intransitivity measure  
    """
    try:
        baseline_diameter = baselines.get(gender, {}).get(surface, 1.0)
        if current_diameter <= 0:
            current_diameter = 1.0
            
        correction_factor = baseline_diameter / current_diameter
        return intransitivity_measure * correction_factor
    except:
        return intransitivity_measure


def apply_temporal_bias_correction(intransitivity_measure: float,
                                 current_bias: float,
                                 gender: str,
                                 surface: str, 
                                 baselines: Dict[str, Dict[str, float]],
                                 alpha: float = 0.1) -> float:
    """
    Apply Temporal Selection Bias Correction.
    
    I_temporal_norm = I(A) / (1 + α × selection_bias(Q,t))
    
    Args:
        intransitivity_measure: Raw I(A) value
        current_bias: Current selection bias measure
        gender: 'men' or 'women'
        surface: 'Hard', 'Clay', or 'Grass'
        baselines: Baseline bias values
        alpha: Strength of bias correction
        
    Returns:
        Normalized intransitivity measure
    """
    try:
        baseline_bias = baselines.get(gender, {}).get(surface, 0.1)
        excess_bias = max(0, current_bias - baseline_bias)
        correction = 1 + alpha * excess_bias
        return intransitivity_measure / correction
    except:
        return intransitivity_measure


def calculate_intransitivity_measure_normalized(A: np.ndarray, 
                                              gender: str,
                                              surface: str,
                                              baselines: Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculate all normalized intransitivity measures using the baselines.
    
    Args:
        A: Advantage matrix (m x m)
        gender: 'men' or 'women'
        surface: 'Hard', 'Clay', or 'Grass' 
        baselines: Dict containing all baseline calculations from validation set
        
    Returns:
        Dictionary containing all normalized measures
    """
    # Get basic measures
    interaction_mask = (A != 0)
    basic_results = calculate_intransitivity_measure_extended(A, interaction_mask)
    
    # Calculate additional properties needed for normalization
    try:
        # Connectivity (second eigenvalue approximation)
        adj_sym = (A != 0).astype(float)
        degrees = np.sum(adj_sym, axis=1)
        if len(degrees) > 2 and np.sum(degrees) > 0:
            D = np.diag(degrees)
            L = D - adj_sym
            try:
                eigenvals = eigsh(L, k=min(3, L.shape[0]-1), which='SM', return_eigenvectors=False)
                current_connectivity = sorted(eigenvals)[1] if len(eigenvals) > 1 else 0.1
            except:
                current_connectivity = 0.1
        else:
            current_connectivity = 0.1
            
        # Elotope diameter
        finite_mask = np.isfinite(A)
        if np.any(finite_mask):
            current_diameter = np.max(A[finite_mask]) - np.min(A[finite_mask])
        else:
            current_diameter = 1.0
            
        # Selection bias
        degrees_total = np.sum(A != 0, axis=1) + np.sum(A != 0, axis=0)
        if len(degrees_total) > 0 and np.std(degrees_total) > 0:
            current_bias = np.std(degrees_total) / (np.mean(degrees_total) + 1e-6)
        else:
            current_bias = 0.1
    except:
        current_connectivity = 0.1
        current_diameter = 1.0
        current_bias = 0.1
    
    # Apply all normalizations
    results = basic_results.copy()
    
    # Selection matrix normalization
    if 'selection_matrix' in baselines:
        results['intransitivity_measure_selection_normalized'] = apply_selection_matrix_normalization(
            basic_results['intransitivity_measure'], gender, surface, baselines['selection_matrix'])
    
    # Hodge surface normalization  
    if 'hodge_surface' in baselines:
        results['intransitivity_measure_hodge_surface_normalized'] = apply_hodge_surface_normalization(
            basic_results['cyclic_component'], basic_results['transitive_component'], 
            gender, surface, baselines['hodge_surface'])
    
    # Connectivity normalization
    if 'connectivity' in baselines:
        results['intransitivity_measure_connectivity_normalized'] = apply_connectivity_normalization(
            basic_results['intransitivity_measure'], current_connectivity,
            gender, surface, baselines['connectivity'])
    
    # Elotope normalization
    if 'elotope' in baselines:
        results['intransitivity_measure_elotope_normalized'] = apply_elotope_normalization(
            basic_results['intransitivity_measure'], current_diameter,
            gender, surface, baselines['elotope'])
    
    # Temporal bias correction
    if 'temporal_bias' in baselines:
        results['intransitivity_measure_temporal_corrected'] = apply_temporal_bias_correction(
            basic_results['intransitivity_measure'], current_bias,
            gender, surface, baselines['temporal_bias'])
    
    return results


def compute_all_validation_baselines(validation_data: Dict[str, Dict[str, Union[dok_matrix, csr_matrix]]]) -> Dict[str, Dict]:
    """
    Compute all normalization baselines from validation data.
    
    Args:
        validation_data: Dict[gender][surface] -> interaction_matrix
        
    Returns:
        Dict containing all baseline calculations:
        - 'selection_matrix': Selection matrix baselines
        - 'hodge_surface': Hodge surface baselines  
        - 'connectivity': Connectivity baselines
        - 'elotope': Elotope baselines
        - 'temporal_bias': Temporal bias baselines
    """
    print("Computing all validation baselines...")
    
    baselines = {}
    
    # Calculate all baseline types
    print("- Selection matrix baselines...")
    baselines['selection_matrix'] = calculate_selection_matrix_baselines(validation_data)
    
    print("- Hodge surface baselines...")
    baselines['hodge_surface'] = calculate_hodge_surface_baselines(validation_data)
    
    print("- Connectivity baselines...")
    baselines['connectivity'] = calculate_connectivity_baselines(validation_data)
    
    print("- Elotope baselines...")
    baselines['elotope'] = calculate_elotope_baselines(validation_data)
    
    print("- Temporal bias baselines...")
    baselines['temporal_bias'] = calculate_selection_bias_baselines(validation_data)
    
    # Print summary
    print("\nBaseline Summary:")
    for baseline_type, data in baselines.items():
        print(f"\n{baseline_type}:")
        for gender in data:
            for surface in data[gender]:
                value = data[gender][surface]
                if isinstance(value, dict):
                    print(f"  {gender}-{surface}: {value}")
                else:
                    print(f"  {gender}-{surface}: {value:.4f}")
    
    return baselines


def save_baselines_to_file(baselines: Dict[str, Dict], filepath: str):
    """Save baselines to a file for later use."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(baselines, f)
    print(f"Baselines saved to {filepath}")


def load_baselines_from_file(filepath: str) -> Dict[str, Dict]:
    """Load baselines from a file."""
    import pickle
    with open(filepath, 'rb') as f:
        baselines = pickle.load(f)
    print(f"Baselines loaded from {filepath}")
    return baselines


def apply_all_normalizations_to_dataframe(df: pd.DataFrame, 
                                        baselines: Dict[str, Dict],
                                        intransitivity_col: str = 'intransitivity_measure_isolated',
                                        cyclic_col: str = 'cyclic_component', 
                                        transitive_col: str = 'transitive_component',
                                        gender_col: str = 'gender_source',
                                        surface_col: str = 'Surface') -> pd.DataFrame:
    """
    Apply all normalization methods to a dataframe with tennis match data.
    
    Args:
        df: DataFrame with tennis match data
        baselines: Baselines dictionary from validation set
        intransitivity_col: Column name for raw intransitivity measure
        cyclic_col: Column name for cyclic component (optional)
        transitive_col: Column name for transitive component (optional) 
        gender_col: Column name for gender ('men'/'women')
        surface_col: Column name for surface ('Hard'/'Clay'/'Grass')
        
    Returns:
        DataFrame with additional normalized columns
    """
    df_copy = df.copy()
    
    # Map gender values to standard format
    gender_map = {'men': 'men', 'women': 'women', 0: 'men', 1: 'women'}
    
    # Initialize new columns
    df_copy['intransitivity_measure_selection_normalized'] = df_copy[intransitivity_col]
    df_copy['intransitivity_measure_hodge_surface_normalized'] = df_copy[intransitivity_col] 
    df_copy['intransitivity_measure_connectivity_normalized'] = df_copy[intransitivity_col]
    df_copy['intransitivity_measure_elotope_normalized'] = df_copy[intransitivity_col]
    df_copy['intransitivity_measure_temporal_corrected'] = df_copy[intransitivity_col]
    
    for idx, row in df_copy.iterrows():
        try:
            # Get gender and surface
            gender_raw = row[gender_col] 
            gender = gender_map.get(gender_raw, str(gender_raw).lower())
            surface = str(row[surface_col])
            
            intransitivity_val = row[intransitivity_col]
            
            # Skip invalid values
            if pd.isna(intransitivity_val) or intransitivity_val == 0:
                continue
            
            # Apply selection matrix normalization
            if 'selection_matrix' in baselines:
                normalized_val = apply_selection_matrix_normalization(
                    intransitivity_val, gender, surface, baselines['selection_matrix'])
                df_copy.at[idx, 'intransitivity_measure_selection_normalized'] = normalized_val
            
            # Apply Hodge surface normalization (if components available)
            if 'hodge_surface' in baselines and cyclic_col in df.columns and transitive_col in df.columns:
                cyclic_val = row.get(cyclic_col, 1.0)
                transitive_val = row.get(transitive_col, 1.0)
                if not pd.isna(cyclic_val) and not pd.isna(transitive_val):
                    normalized_val = apply_hodge_surface_normalization(
                        cyclic_val, transitive_val, gender, surface, baselines['hodge_surface'])
                    df_copy.at[idx, 'intransitivity_measure_hodge_surface_normalized'] = normalized_val
            
            # Apply connectivity normalization (using intransitivity as proxy)
            if 'connectivity' in baselines:
                # Use a simplified connectivity estimate
                current_connectivity = min(max(intransitivity_val * 0.1, 0.01), 1.0)
                normalized_val = apply_connectivity_normalization(
                    intransitivity_val, current_connectivity, gender, surface, baselines['connectivity'])
                df_copy.at[idx, 'intransitivity_measure_connectivity_normalized'] = normalized_val
            
            # Apply elotope normalization
            if 'elotope' in baselines:
                # Use intransitivity as proxy for elotope diameter
                current_diameter = max(intransitivity_val, 1.0)
                normalized_val = apply_elotope_normalization(
                    intransitivity_val, current_diameter, gender, surface, baselines['elotope'])
                df_copy.at[idx, 'intransitivity_measure_elotope_normalized'] = normalized_val
            
            # Apply temporal bias correction
            if 'temporal_bias' in baselines:
                # Use a simple bias estimate based on intransitivity
                current_bias = intransitivity_val * 0.05  # Simple proxy
                normalized_val = apply_temporal_bias_correction(
                    intransitivity_val, current_bias, gender, surface, baselines['temporal_bias'])
                df_copy.at[idx, 'intransitivity_measure_temporal_corrected'] = normalized_val
                
        except Exception as e:
            print(f"Warning: Could not normalize row {idx}: {e}")
            continue
    
    return df_copy


def calculate_time_weighted_intransitivity_measures(interaction_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                                   last_update_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                                   surface: str,
                                                   query_edges: np.ndarray,
                                                   current_time: float,
                                                   k_hops: int = 1,
                                                   min_common: int = 1,
                                                   decay_days: float = 30.0) -> List[float]:
    """
    Calculate time-weighted local intransitivity measures.

    Args:
        interaction_matrices: Dict mapping surface names to interaction matrices
        last_update_matrices: Dict mapping surface names to last update time matrices
        surface: Surface to analyze
        query_edges: Array of shape (n_matches, 2) containing match pairs
        current_time: Current timestamp for calculating edge recency
        k_hops: Number of hops for neighborhood
        min_common: Minimum number of common opponents needed
        decay_days: Number of days for edge weight to decay by factor of e

    Returns:
        List of time-weighted intransitivity measures
    """
    if surface not in interaction_matrices:
        return [0.0] * len(query_edges)

    # Create time-weighted interaction matrix
    time_weighted_matrix = interaction_matrices[surface].copy()

    # Apply time decay to edge weights
    for (i, j), prob in time_weighted_matrix.items():
        if prob > 0:
            last_update = last_update_matrices[surface].get((i, j), current_time)
            days_old = (current_time - last_update) / (24 * 60 * 60)

            # Exponential decay: weight = exp(-days_old / decay_days)
            time_weight = np.exp(-max(0, days_old) / decay_days)
            time_weighted_matrix[(i, j)] = prob * time_weight

    # Calculate intransitivity using time-weighted matrix
    extended_results = calculate_match_local_intransitivity_vectorized_extended(
        {surface: time_weighted_matrix},
        surface,
        query_edges,
        k_hops=k_hops,
        min_common=min_common
    )

    return extended_results.get('intransitivity_measure', [0.0] * len(query_edges))


def calculate_confidence_weighted_intransitivity_measures(interaction_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                                        cumulative_weight_matrices: Dict[str, Union[dok_matrix, csr_matrix]],
                                                        surface: str,
                                                        query_edges: np.ndarray,
                                                        k_hops: int = 1,
                                                        min_common: int = 1) -> List[float]:
    """
    Calculate confidence-weighted local intransitivity measures.

    Args:
        interaction_matrices: Dict mapping surface names to interaction matrices
        cumulative_weight_matrices: Dict mapping surface names to cumulative weight matrices
        surface: Surface to analyze
        query_edges: Array of shape (n_matches, 2) containing match pairs
        k_hops: Number of hops for neighborhood
        min_common: Minimum number of common opponents needed

    Returns:
        List of confidence-weighted intransitivity measures
    """
    if surface not in interaction_matrices:
        return [0.0] * len(query_edges)

    # Create confidence-weighted matrix using cumulative weights
    confidence_weighted_matrix = interaction_matrices[surface].copy()

    # Scale edge probabilities by confidence (cumulative weight)
    for (i, j), prob in confidence_weighted_matrix.items():
        if prob > 0:
            confidence = cumulative_weight_matrices[surface].get((i, j), 1e-8)
            # Normalize confidence to [0, 1] range - higher confidence = more weight
            confidence_factor = min(1.0, confidence / 10.0)  # Scale factor
            confidence_weighted_matrix[(i, j)] = prob * (0.5 + 0.5 * confidence_factor)

    extended_results = calculate_match_local_intransitivity_vectorized_extended(
        {surface: confidence_weighted_matrix},
        surface,
        query_edges,
        k_hops=k_hops,
        min_common=min_common
    )

    return extended_results.get('intransitivity_measure', [0.0] * len(query_edges))


if __name__ == "__main__":
    validate_intransitivity_implementation()