"""
Visualization tool for Player Interaction Matrix (PIM)
Shows common opponent graphs with edge weights to understand match scoring
"""

import sys
import os
import json
# Add parent directory to path FIRST before any local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Tuple, Optional
import torch
from dataclasses import dataclass

con = Console(width=120)

from model.graph.player_interaction import PlayerInteractionMatrix
from model.graph.windowstacks import split_df_by_surface, create_meta_stacks
from utils.helpers import load_player_data
from model.graph.ia_compute import (
    interaction_to_advantage_matrix,
    calculate_intransitivity_measure,
)


@dataclass
class PIMVisualizer:
    """Visualizer for Player Interaction Matrix graphs"""
    gender: str = "men"
    surfaces: Dict[str, Dict[str, float]] = None
    lambda_param: float = 0.43331168995512426
    tournament_quality: Dict[str, float] = None
    granularity: str = "games"
    
    def __post_init__(self):      
        # Create player name mappings
        self.player_data = load_player_data(self.gender)
        self.index_to_name = {data['index']: name for name, data in self.player_data.items()}
        self.name_to_index = {name: data['index'] for name, data in self.player_data.items()}
    
    def build_pim_to_window(self, df: pd.DataFrame, target_window_id: int, 
                           target_surface: str) -> PlayerInteractionMatrix:
        """
        Build PIM up to a specific window ID for visualization
        
        Args:
            df: Full dataframe with match data
            target_window_id: Window ID to build PIM up to
            target_surface: Surface to analyze
        
        Returns:
            PlayerInteractionMatrix with data up to target window
        """
        # Filter data up to target window
        df_filtered = df[df['window_idx'] <= target_window_id].copy()
        
        # Create PIM
        pim = PlayerInteractionMatrix(
            gender=self.gender,
            granularity=self.granularity,
            surfaces=self.surfaces,
            lambda_param=self.lambda_param,
            tournament_quality=self.tournament_quality
        )
        
        # Split by surface and create meta stacks
        g_pct = 0.8  # Use most data for building PIM
        t_pct = 0.1
        v_pct = 0.1
        surface_splits = split_df_by_surface(df_filtered, g_pct, t_pct, v_pct)
        meta_stacks = create_meta_stacks(surface_splits, self.gender)
        
        # Update PIM with all available data
        pim.update_pim_from_stacks(meta_stacks, ['graph', 'train', 'val', 'test'])
        
        return pim
    
    def find_most_intransitive_subgraph(self, pim: PlayerInteractionMatrix,
                                       surface: str,
                                       player_a_name: str,
                                       player_b_name: str,
                                       max_nodes: int = 10) -> nx.DiGraph:
        """
        Find the subgraph that contributes most to intransitivity around two players.
        
        This identifies cycles and strongly connected components that create intransitivity.
        
        Args:
            pim: PlayerInteractionMatrix instance
            surface: Surface to analyze
            player_a_name: Name of first player
            player_b_name: Name of second player
            max_nodes: Maximum number of nodes to include in visualization
        
        Returns:
            NetworkX directed graph showing key intransitive structure
        """
        # Get player indices
        player_a_idx = self.name_to_index.get(player_a_name, None)
        player_b_idx = self.name_to_index.get(player_b_name, None)
        
        if player_a_idx is None or player_b_idx is None:
            raise ValueError(f"Player not found: {player_a_name if player_a_idx is None else player_b_name}")
        
        # Get interaction matrix for surface
        if surface not in pim.interaction_matrices:
            raise ValueError(f"Surface {surface} not found in PIM")
        
        # Convert to dense array
        matrix = pim.interaction_matrices[surface].toarray()
        
        # Find all players connected to both A and B
        a_connected = set()
        b_connected = set()
        
        for i in range(matrix.shape[0]):
            if matrix[player_a_idx, i] != 0 or matrix[i, player_a_idx] != 0:
                a_connected.add(i)
            if matrix[player_b_idx, i] != 0 or matrix[i, player_b_idx] != 0:
                b_connected.add(i)
        
        common_opponents = list(a_connected.intersection(b_connected))
        
        # Calculate contribution to intransitivity for each common opponent
        intransitivity_scores = []
        
        for opp_idx in common_opponents:
            # Create mini subgraph with A, B, and this opponent
            mini_indices = [player_a_idx, player_b_idx, opp_idx]
            mini_matrix = matrix[np.ix_(mini_indices, mini_indices)]
            
            # Calculate intransitivity of this 3-player subgraph
            if np.any(mini_matrix > 0):
                A_matrix = interaction_to_advantage_matrix(mini_matrix)
                measures = calculate_intransitivity_measure(A_matrix)
                
                # Also check for cycles (most important for intransitivity)
                has_cycle = self._detect_3_cycle(matrix, player_a_idx, player_b_idx, opp_idx)
                cycle_bonus = 2.0 if has_cycle else 1.0
                
                score = measures['intransitivity_measure'] * cycle_bonus
                intransitivity_scores.append((opp_idx, score, has_cycle))
            else:
                intransitivity_scores.append((opp_idx, 1.0, False))
        
        # Sort by intransitivity contribution and take top nodes
        intransitivity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate TOTAL cycle statistics (N=infinite) before filtering
        total_cycles_all = 0
        total_cycles_with_main_players = 0
        all_cycle_combinations = []
        
        # Build full graph with ALL common opponents to count total cycles
        all_relevant_nodes = {player_a_idx, player_b_idx}
        all_relevant_nodes.update([opp_idx for opp_idx, _, _ in intransitivity_scores])
        
        # Check all possible 3-cycles in the full common opponent set
        all_nodes_list = list(all_relevant_nodes)
        for i in range(len(all_nodes_list)):
            for j in range(i+1, len(all_nodes_list)):
                for k in range(j+1, len(all_nodes_list)):
                    n1, n2, n3 = all_nodes_list[i], all_nodes_list[j], all_nodes_list[k]
                    
                    # Check if these 3 nodes form a cycle
                    if self._detect_3_cycle(matrix, n1, n2, n3):
                        total_cycles_all += 1
                        all_cycle_combinations.append([n1, n2, n3])
                        
                        # Check if main players are involved
                        if player_a_idx in [n1, n2, n3] or player_b_idx in [n1, n2, n3]:
                            total_cycles_with_main_players += 1
        
        # Now filter for visualization (top N nodes)
        top_opponents = intransitivity_scores[:max(0, max_nodes - 2)]  # -2 for A and B
        
        # Build the graph with selected nodes (limited by max_nodes)
        G = nx.DiGraph()
        
        # Always include A and B
        selected_nodes = {player_a_idx, player_b_idx}
        selected_nodes.update([opp_idx for opp_idx, _, _ in top_opponents])
        
        # Add nodes with labels and metadata
        for node_idx in selected_nodes:
            node_name = self.index_to_name.get(node_idx, f"Player_{node_idx}")
            is_main = node_idx in {player_a_idx, player_b_idx}
            
            # Find if this node is part of a cycle
            in_cycle = False
            for opp_idx, score, has_cycle in top_opponents:
                if opp_idx == node_idx and has_cycle:
                    in_cycle = True
                    break
            
            G.add_node(node_idx, label=node_name, is_main=is_main, in_cycle=in_cycle)
        
        # Add all edges between selected nodes
        for i in selected_nodes:
            for j in selected_nodes:
                if i != j and matrix[i, j] > 0:
                    weight = matrix[i, j]
                    G.add_edge(i, j, weight=weight)
        
        # Calculate match-level intransitivity exactly like generate_intransitivity_data.py
        # Use the same method as calculate_match_local_intransitivity_vectorized
        query_edges = np.array([[player_a_idx, player_b_idx]])
        match_ia_values = pim.calculate_match_local_intransitivity(surface, torch.tensor(query_edges))
        
        if match_ia_values and len(match_ia_values) > 0:
            overall_measures = {
                'intransitivity_measure': match_ia_values[0],
                'interpretation': 'effectively_intransitive' if match_ia_values[0] > 1.0 else 'predominantly_transitive' if match_ia_values[0] < 1.0 else 'balanced'
            }
        else:
            overall_measures = {'intransitivity_measure': 1.0, 'interpretation': 'insufficient_data'}
        
        # Find all 3-cycles in the subgraph
        cycles = self._find_all_3_cycles(G)
        
        # Store metadata in graph
        G.graph['player_a'] = player_a_name
        G.graph['player_b'] = player_b_name
        G.graph['player_a_idx'] = player_a_idx
        G.graph['player_b_idx'] = player_b_idx
        G.graph['intransitivity_measures'] = overall_measures
        G.graph['cycles'] = cycles
        G.graph['num_common_opponents'] = len(common_opponents)
        # Store total cycle statistics (N=infinite)
        G.graph['total_cycles_all_opponents'] = total_cycles_all
        G.graph['total_cycles_with_main_players'] = total_cycles_with_main_players
        G.graph['all_cycle_combinations'] = all_cycle_combinations
        
        return G
    
    def _detect_3_cycle(self, matrix: np.ndarray, a: int, b: int, c: int) -> bool:
        """
        Detect if three players form any kind of cycle (rock-paper-scissors pattern).
        
        Returns True if there's a cycle like A→B→C→A or any permutation.
        """
        # Check all possible 3-cycles
        players = [a, b, c]
        
        # Check each possible cycle direction
        for i in range(3):
            p1, p2, p3 = players[i], players[(i+1)%3], players[(i+2)%3]
            # Check if p1→p2→p3→p1 exists
            if matrix[p1, p2] > 0.5 and matrix[p2, p3] > 0.5 and matrix[p3, p1] > 0.5:
                return True
        
        return False
    
    def _find_all_3_cycles(self, G: nx.DiGraph) -> List[List[int]]:
        """Find all 3-cycles in the graph."""
        cycles = []
        nodes = list(G.nodes())
        
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                for k in range(j+1, len(nodes)):
                    n1, n2, n3 = nodes[i], nodes[j], nodes[k]
                    
                    # Check all possible cycle orderings
                    perms = [(n1,n2,n3), (n1,n3,n2), (n2,n1,n3), (n2,n3,n1), (n3,n1,n2), (n3,n2,n1)]
                    
                    for p1, p2, p3 in perms:
                        if G.has_edge(p1, p2) and G.has_edge(p2, p3) and G.has_edge(p3, p1):
                            # Found a cycle
                            cycle = [p1, p2, p3]
                            # Normalize to avoid duplicates (sort by first element)
                            min_idx = cycle.index(min(cycle))
                            normalized = cycle[min_idx:] + cycle[:min_idx]
                            if normalized not in cycles:
                                cycles.append(normalized)
                            break
        
        return cycles
    
    def extract_common_opponent_subgraph(self, pim: PlayerInteractionMatrix, 
                                        surface: str,
                                        player_a_name: str, 
                                        player_b_name: str) -> nx.DiGraph:
        """
        Extract subgraph showing common opponents between two players
        
        Args:
            pim: PlayerInteractionMatrix instance
            surface: Surface to analyze
            player_a_name: Name of first player
            player_b_name: Name of second player
        
        Returns:
            NetworkX directed graph with common opponents
        """
        # Get player indices
        player_a_idx = self.name_to_index.get(player_a_name, None)
        player_b_idx = self.name_to_index.get(player_b_name, None)
        
        if player_a_idx is None or player_b_idx is None:
            raise ValueError(f"Player not found: {player_a_name if player_a_idx is None else player_b_name}")
        
        # Get interaction matrix for surface
        if surface not in pim.interaction_matrices:
            raise ValueError(f"Surface {surface} not found in PIM")
        
        # Convert to dense array for easier manipulation
        matrix = pim.interaction_matrices[surface].toarray()
        
        # Find common opponents (players who have edges with both A and B)
        # Check both directions for each player - any non-zero edge counts
        a_opponents = set()
        b_opponents = set()
        
        for i in range(matrix.shape[0]):
            # Players A beat or lost to (any non-zero edge)
            if matrix[player_a_idx, i] != 0 or matrix[i, player_a_idx] != 0:
                a_opponents.add(i)
            # Players B beat or lost to (any non-zero edge)
            if matrix[player_b_idx, i] != 0 or matrix[i, player_b_idx] != 0:
                b_opponents.add(i)
        
        # Common opponents
        common_opponents = a_opponents.intersection(b_opponents)
        
        # Build subgraph with players A, B and common opponents
        G = nx.DiGraph()
        
        # Add all relevant nodes
        relevant_nodes = {player_a_idx, player_b_idx} | common_opponents
        
        # Add nodes with labels
        for node_idx in relevant_nodes:
            node_name = self.index_to_name.get(node_idx, f"Player_{node_idx}")
            is_main = node_idx in {player_a_idx, player_b_idx}
            G.add_node(node_idx, label=node_name, is_main=is_main)
        
        # Add all non-zero edges between relevant nodes
        for i in relevant_nodes:
            for j in relevant_nodes:
                if i != j and matrix[i, j] > 0:
                    weight = matrix[i, j]
                    G.add_edge(i, j, weight=weight)
        
        # Calculate potential match score components
        match_score_info = self._calculate_match_score_components(
            matrix, player_a_idx, player_b_idx, common_opponents
        )
        
        # Store match info in graph attributes
        G.graph['player_a'] = player_a_name
        G.graph['player_b'] = player_b_name
        G.graph['player_a_idx'] = player_a_idx
        G.graph['player_b_idx'] = player_b_idx
        G.graph['match_score_info'] = match_score_info
        
        return G
    
    def _calculate_match_score_components(self, matrix: np.ndarray, 
                                         player_a_idx: int, 
                                         player_b_idx: int,
                                         common_opponents: set) -> Dict:
        """
        Calculate how the match score would be computed from common opponents
        """
        info = {
            'direct_a_to_b': matrix[player_a_idx, player_b_idx] if matrix[player_a_idx, player_b_idx] > 0 else None,
            'direct_b_to_a': matrix[player_b_idx, player_a_idx] if matrix[player_b_idx, player_a_idx] > 0 else None,
            'common_opponent_paths': []
        }
        
        # Find transitive paths through common opponents
        for opp_idx in common_opponents:
            # A -> Opponent -> B path
            a_to_opp = matrix[player_a_idx, opp_idx]
            opp_to_b = matrix[opp_idx, player_b_idx]
            if a_to_opp > 0 and opp_to_b > 0:
                info['common_opponent_paths'].append({
                    'type': 'A_beats_C_beats_B',
                    'opponent': self.index_to_name.get(opp_idx, f"Player_{opp_idx}"),
                    'a_to_opp': a_to_opp,
                    'opp_to_b': opp_to_b,
                    'path_strength': min(a_to_opp, opp_to_b)
                })
            
            # B -> Opponent -> A path
            b_to_opp = matrix[player_b_idx, opp_idx]
            opp_to_a = matrix[opp_idx, player_a_idx]
            if b_to_opp > 0 and opp_to_a > 0:
                info['common_opponent_paths'].append({
                    'type': 'B_beats_C_beats_A',
                    'opponent': self.index_to_name.get(opp_idx, f"Player_{opp_idx}"),
                    'b_to_opp': b_to_opp,
                    'opp_to_a': opp_to_a,
                    'path_strength': min(b_to_opp, opp_to_a)
                })
        
        return info
    
    def visualize_intransitive_graph(self, G: nx.DiGraph, 
                                   title: str = "Intransitivity Graph",
                                   figsize: Tuple[int, int] = (15, 12),
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize the graph with intransitivity highlighting
        
        Args:
            G: NetworkX directed graph
            title: Title for the plot
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Set up layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Separate different types of nodes
        main_nodes = [n for n in G.nodes() if G.nodes[n].get('is_main', False)]
        cycle_nodes = [n for n in G.nodes() if G.nodes[n].get('in_cycle', False) and not G.nodes[n].get('is_main', False)]
        other_nodes = [n for n in G.nodes() if not G.nodes[n].get('is_main', False) and not G.nodes[n].get('in_cycle', False)]
        
        # Draw nodes with different colors
        if main_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, 
                                  node_color='red', node_size=1500, alpha=0.9,
                                  label='Main Players')
        if cycle_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cycle_nodes,
                                  node_color='orange', node_size=1000, alpha=0.8,
                                  label='Cycle Contributors')
        if other_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                                  node_color='lightblue', node_size=800, alpha=0.7,
                                  label='Other Opponents')
        
        # Draw node labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        # Draw edges with highlighting for cycles
        edges = list(G.edges())
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Identify edges that are part of 3-cycles
        cycle_edges = set()
        if 'cycles' in G.graph:
            for cycle in G.graph['cycles']:
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1) % len(cycle)]
                    if G.has_edge(u, v):
                        cycle_edges.add((u, v))
        
        if weights:
            # Separate cycle edges from regular edges
            regular_edges = [(u, v) for u, v in edges if (u, v) not in cycle_edges]
            cycle_edge_list = [(u, v) for u, v in edges if (u, v) in cycle_edges]
            
            # Draw regular edges
            if regular_edges:
                regular_weights = [G[u][v]['weight'] for u, v in regular_edges]
                min_weight = min(weights)
                max_weight = max(weights)
                
                if max_weight > min_weight:
                    edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) for w in regular_weights]
                else:
                    edge_widths = [2.0] * len(regular_weights)
                
                edge_colors = plt.cm.Blues([(w - 0.5) * 0.8 + 0.4 for w in regular_weights])
                
                nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=edge_widths,
                                      edge_color=edge_colors, alpha=0.6,
                                      arrows=True, arrowsize=15, arrowstyle='->')
            
            # Draw cycle edges with special highlighting
            if cycle_edge_list:
                cycle_weights = [G[u][v]['weight'] for u, v in cycle_edge_list]
                min_weight = min(weights)
                max_weight = max(weights)
                
                if max_weight > min_weight:
                    cycle_widths = [2 + 4 * (w - min_weight) / (max_weight - min_weight) for w in cycle_weights]
                else:
                    cycle_widths = [3.0] * len(cycle_weights)
                
                nx.draw_networkx_edges(G, pos, edgelist=cycle_edge_list, width=cycle_widths,
                                      edge_color='red', alpha=0.8,
                                      arrows=True, arrowsize=20, arrowstyle='->')
            
            # Draw edge labels
            edge_labels = {(u, v): f'{G[u][v]["weight"] + 0.5:.2f}' for u, v in edges}  # +0.5: display on paper's (0.5,1] dominance-score scale
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        # Add title and legend
        plt.title(f"{title}\n{G.graph['player_a']} vs {G.graph['player_b']}", fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.axis('off')
        
        # Add intransitivity information
        self._add_intransitivity_info(G, plt.gca())

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"saved -> {save_path}")
    
    def _add_intransitivity_info(self, G: nx.DiGraph, ax) -> None:
        """Add text box with intransitivity information"""
        text_lines = ["Intransitivity Analysis:"]
        text_lines.append("-" * 30)
        
        # Overall intransitivity measure
        if 'intransitivity_measures' in G.graph:
            measures = G.graph['intransitivity_measures']
            text_lines.append(f"Match I(A) = {measures['intransitivity_measure']:.3f}")
            text_lines.append(f"Interpretation: {measures.get('interpretation', 'unknown')}")
        
        # Cycle information
        if 'cycles' in G.graph and G.graph['cycles']:
            text_lines.append(f"\nDetected {len(G.graph['cycles'])} cycles:")
            for i, cycle in enumerate(G.graph['cycles'][:3]):  # Show first 3 cycles
                cycle_names = [G.nodes[node]['label'] for node in cycle]
                text_lines.append(f"  {' -> '.join(cycle_names)} -> {cycle_names[0]}")
        else:
            text_lines.append("\nNo cycles detected")
        
        # Node count info
        text_lines.append(f"\nNodes: {len(G.nodes())}")
        text_lines.append(f"Total opponents: {G.graph.get('num_common_opponents', 'unknown')}")
        
        text = '\n'.join(text_lines)
        
        # Add text box on the right side
        props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
        ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace')
    
    def _add_match_score_text(self, G: nx.DiGraph, ax) -> None:
        """Add text box with match score information"""
        score_info = G.graph.get('match_score_info', {})
        
        text_lines = ["Match Score Components:"]
        text_lines.append("-" * 30)
        
        # Direct edges
        if score_info.get('direct_a_to_b'):
            text_lines.append(f"Direct: {G.graph['player_a']} → {G.graph['player_b']}: {score_info['direct_a_to_b']:.3f}")
        if score_info.get('direct_b_to_a'):
            text_lines.append(f"Direct: {G.graph['player_b']} → {G.graph['player_a']}: {score_info['direct_b_to_a']:.3f}")
        
        # Common opponent paths
        if score_info.get('common_opponent_paths'):
            text_lines.append("\nTransitive Paths:")
            for path in score_info['common_opponent_paths'][:5]:  # Show top 5
                if path['type'] == 'A_beats_C_beats_B':
                    text_lines.append(f"  {G.graph['player_a']} → {path['opponent']} → {G.graph['player_b']}")
                    text_lines.append(f"    Strengths: {path['a_to_opp']:.2f} × {path['opp_to_b']:.2f} = {path['path_strength']:.2f}")
                else:
                    text_lines.append(f"  {G.graph['player_b']} → {path['opponent']} → {G.graph['player_a']}")
                    text_lines.append(f"    Strengths: {path['b_to_opp']:.2f} × {path['opp_to_a']:.2f} = {path['path_strength']:.2f}")
        
        text = '\n'.join(text_lines)
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')


def main():
    """Example usage of the PIM visualizer"""
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONFIG
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    gender         = "women"
    target_window  = 1202                          # last window before Sakkari-Sabalenka, 2021-01-12 Abu Dhabi (fig:womens_nontransitive)
    target_surface = "Hard"
    player_a       = "Sakkari M."
    player_b       = "Sabalenka A."
    max_nodes      = 4                             # A + B + 2 cycle opponents
    save_graph     = True
    tune_path      = "tunes/tune_degree_postab_best.json"  # final T170 config

    # Load tune params (so surfaces/TQ/lambda/granularity stay in sync with the GNN)
    with open(tune_path) as f:
        p = json.load(f)["params"]
    surfaces = {
        "Hard":  {"Hard": 1.0,     "Clay": p["HC"], "Grass": p["HG"]},
        "Clay":  {"Hard": p["CH"], "Clay": 1.0,     "Grass": p["CG"]},
        "Grass": {"Hard": p["GH"], "Clay": p["GC"], "Grass": 1.0},
    }
    tournament_quality = {
        "GS":     1.0,
        "Finals": p["beta_Finals"],
        "1000":   p["beta_1000"],
        "500":    p["beta_500"],
    }

    # Load data (no more _w500 suffix)
    print("Loading data...")
    df = pd.read_csv(f"data/{'atp' if gender == 'men' else 'wta'}_cleaned.csv")

    visualizer = PIMVisualizer(
        gender=gender,
        surfaces=surfaces,
        tournament_quality=tournament_quality,
        lambda_param=p["lambda_param"],
        granularity=p["granularity"],
    )
    
    print(f"Building PIM up to window {target_window}...")
    pim = visualizer.build_pim_to_window(df, target_window, target_surface)
    
    print(f"Finding most intransitive subgraph for {player_a} vs {player_b}...")
    G = visualizer.find_most_intransitive_subgraph(
        pim, target_surface, player_a, player_b, max_nodes
    )
    
    print(f"Found {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Print intransitivity info
    if 'intransitivity_measures' in G.graph:
        measures = G.graph['intransitivity_measures']
        print(f"Match I(A) = {measures['intransitivity_measure']:.3f} ({measures.get('interpretation', 'unknown')})")
    
    # Cycle statistics: full common-opponent network vs the selected subgraph
    cyc = G.graph.get('cycles') or []
    main = [c for c in cyc if G.graph['player_a_idx'] in c or G.graph['player_b_idx'] in c]
    tab = Table(title=f"Cycle counts: {player_a} vs {player_b}, window {target_window}")
    for c in ['Graph', 'Players', 'Cycles', f'Cycles with {player_a} or {player_b}']:
        tab.add_column(c, justify='right')
    tab.add_row('Full network', f"{G.graph['num_common_opponents']} common opponents",
                str(G.graph['total_cycles_all_opponents']), str(G.graph['total_cycles_with_main_players']))
    tab.add_row(f'Selected subgraph (N={max_nodes})', str(len(G.nodes())), str(len(cyc)), str(len(main)))
    con.print(tab)

    # Figure 6 edge weights (example_figs.py WOMEN), on the paper's (0.5, 1] dominance scale
    lab = {n: G.nodes[n]['label'] for n in G.nodes()}
    cyc_edges = {(c[k], c[(k + 1) % len(c)]) for c in cyc for k in range(len(c))}
    tab = Table(title="Subgraph edges: dominant -> dominated player")
    for c in ['From', 'To', 'Dominance', 'In cycle']:
        tab.add_column(c, justify='right')
    for u, v in sorted(G.edges(), key=lambda e: (lab[e[0]], lab[e[1]])):
        tab.add_row(lab[u], lab[v], f"{G[u][v]['weight'] + 0.5:.2f}", 'yes' if (u, v) in cyc_edges else 'no')
    con.print(tab)
    for c in main[:5]:  # show first 5
        names = [visualizer.index_to_name.get(n, f"Player_{n}") for n in c]
        print(f"cycle: {' -> '.join(names)} -> {names[0]}")

    # Generate save path if needed
    save_path = None
    if save_graph:
        save_path = f"outputs/intransitivity_graph_{player_a.replace(' ', '_')}_vs_{player_b.replace(' ', '_')}_w{target_window}.png"
    
    # Visualize
    visualizer.visualize_intransitive_graph(
        G, 
        title=f"Intransitivity Graph - Window {target_window}, {target_surface} Surface",
        save_path=save_path
    )


if __name__ == "__main__":
    main()