import types
import torch
import torch.nn.functional as F
from model.graph.player_interaction import PlayerInteractionMatrix

from torch_geometric_signed_directed import MagNet_link_prediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _custom_forward(self, real, imag, edge_index, query_edges, edge_weight=None, edge_features=None):
    """Unified forward: optional edge feature concat, mode-controlled softmax."""
    for cheb in self.Chebs:
        real, imag = cheb(real, imag, edge_index, edge_weight)
        if self.activation:
            real, imag = self.complex_relu(real, imag)
    x = torch.cat((real[query_edges[:, 0]], real[query_edges[:, 1]],
                    imag[query_edges[:, 0]], imag[query_edges[:, 1]]), dim=-1)
    if edge_features is not None:
        x = torch.cat([x, edge_features], dim=-1)
    if self.dropout > 0:
        x = F.dropout(x, self.dropout, training=self.training)
    x = self.linear(x)
    return F.log_softmax(x, dim=1) if self._classify else x


class GraphEngine:

    VALID_GRANULARITY = {"games", "sets", "points", "serve", "serve_surprise", "point_surprise", "break", "break_surprise", "binary"}

    def __init__(self, lr=0.01, label_smoothing=0.0, granularity="games",  # ls=0 optimal (sweep_ls.py): smoothing did not help
                 K=2, hidden=32, layers=1, activation=True, dropout=0.3, gender="men", surfaces=None,
                 lambda_param=0.001, tournament_quality=None, q=0.25, trainable_q=False, node_feature_matrix=None, gamma=0.0, theta=0.0, kappa=0.0, max_edge_age=0.0, min_dom_pctile=0.0, min_edge_prob=0.0, mode='classification', mlp_head=False, n_edge_features=0, scale=0.20):
        """Initialize graph engine with parameters for both NN and interaction matrix"""
        # Node features: 4 static + 2 degree + 8 dynamic (srv/ret/bp/1st_in/1st_win/2nd_win/ace/df) = 14
        num_features = 14

        self.mode = mode  # 'classification', 'regression', or 'pointwise'
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.granularity = granularity
        self.K = K
        self.q = q
        self.hidden = hidden
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.gender = gender
        self.gamma = gamma  # Decorrelation parameter from Hubacek 2023
        self.theta = theta  # Market surprise weighting parameter for profit optimization
        self.scale = scale  # Pointwise: tanh output range for serve prob (mu ± scale)
        
        # Set default surfaces if not provided
        if surfaces is None:
            surfaces = {
                "Hard": {"idx": 1, "Grass": 0.4, "Clay": 0.95, "Hard": 1.0},
                "Clay": {"idx": 2, "Grass": 0.55, "Clay": 1.0, "Hard": 0.08}, 
                "Grass": {"idx": 3, "Grass": 1.0, "Clay": 0.375, "Hard": 0.8},
            }
                    
        # Default tier weights: neutral (1.0) across all 4 tiers unless tuned.
        # Tiers: GS / Finals / 1000 / 500; see model/graph/tq_buckets.py.
        if tournament_quality is None:
            tournament_quality = {'GS': 1.0, 'Finals': 1.0, '1000': 1.0, '500': 1.0}
        
        # Create player interaction matrix
        self.pim = PlayerInteractionMatrix(
            gender=gender,
            granularity=granularity,
            surfaces=surfaces,
            lambda_param=lambda_param,
            tournament_quality=tournament_quality,
            kappa=kappa,
            max_edge_age=max_edge_age,
            min_dom_pctile=min_dom_pctile,
            min_edge_prob=min_edge_prob,
            node_feature_matrix=node_feature_matrix
        )
        
        self.model_probs = None
        self.n_edge_features = n_edge_features
        # How each edge feature transforms when query edge (A,B) → (B,A)
        # posterior=negate, confidence=keep, staleness=keep, serve/return/bp=swap A↔B
        self._full_feature_types = ['signed', 'symmetric', 'symmetric',
                                     'swap', 'swap', 'swap', 'swap', 'swap', 'swap']
        self.edge_feature_types = self._full_feature_types[:n_edge_features]
        self.swap_pairs = [(i, i+1) for i in range(3, n_edge_features, 2)]  # (3,4), (5,6), (7,8)

        label_dim = 1 if mode == 'regression' else 2  # pointwise also uses 2 (raw_pa, raw_pb)
        linear_in = hidden * 4 + n_edge_features

        self.model = MagNet_link_prediction(
                num_features=num_features,
                normalization="sym",
                q=q,
                trainable_q=trainable_q,
                label_dim=label_dim,
                cached=False,
                K=K,
                hidden=hidden,
                layer=layers,
                activation=activation,
                dropout=dropout,
        ).to(device)

        # Replace prediction head if edge features or MLP requested
        if mlp_head:
            import torch.nn as nn
            self.model.linear = nn.Sequential(
                nn.Linear(linear_in, hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, label_dim),
            ).to(device)
        elif n_edge_features > 0:
            self.model.linear = torch.nn.Linear(linear_in, label_dim).to(device)

        # Unified forward: handles edge features + regression/classification via flag
        self.model._classify = (mode == 'classification')  # Only softmax for classification
        self.model.forward = types.MethodType(_custom_forward, self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        if mode == 'regression':
            self.criterion = torch.nn.MSELoss()
        elif mode == 'pointwise':
            self.criterion = None  # Loss computed inline via hierarchical model
        else:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.surface_graphs = {surface: {} for surface in self.pim.surface_graphs}

    def decorrelation_loss(self, model_probs, market_probs, labels):
        """
        Compute decorrelation loss based on Hubacek 2023 equation (61):
        MSE*(R, M, T) = E[(T - R)² - γ * (T - M)²]
        
        Args:
            model_probs: Model probability predictions for player A winning (T)
            market_probs: Market (bookmaker) probability estimates (M) 
            labels: True binary outcomes (R)
            
        Returns:
            Combined loss tensor
        """
        if self.gamma == 0.0:
            # Standard cross-entropy loss only
            return self.criterion(torch.stack([1 - model_probs, model_probs], dim=1), labels)
        
        # Convert labels to probabilities (0 or 1)
        labels_float = labels.float()
        
        # Standard prediction loss: (T - R)²
        prediction_loss = torch.mean((model_probs - labels_float) ** 2)
        
        # Decorrelation term: γ * (T - M)²
        decorrelation_term = self.gamma * torch.mean((model_probs - market_probs) ** 2)
        
        # Combined loss: prediction loss - decorrelation term
        # We subtract because we want to PENALIZE correlation with market
        total_loss = prediction_loss - decorrelation_term
        
        return total_loss

    def train_multi_surface(self, surface_data_dict, surface_weights=None):
        """
        Multi-task training: Train on multiple surfaces simultaneously.
        
        Args:
            surface_data_dict: {surface: {'query_edges': tensor, 'y': tensor, 'market_probs': tensor (optional)}}
            surface_weights: Optional weights per surface (defaults to equal)
        Returns:
            Weighted average loss across all surfaces
        """
        self.model.train()
        
        # Default to equal weights if not provided
        if surface_weights is None:
            surface_weights = {surf: 1.0 for surf in surface_data_dict.keys()}
        
        total_loss = 0.0
        total_weight = 0.0
        
        # Train on each surface and aggregate losses
        for surface, data in surface_data_dict.items():
            if data['query_edges'].size(0) == 0:  # Skip empty surfaces
                continue

            ef = self._build_edge_features(surface, data['query_edges'])
            query_edges_doubled, y_doubled = self.handle_permutation_variance(
                data['query_edges'], data['y']
            )
            ef_doubled = self._double_edge_features(ef) if ef is not None else None

            surface_graph = self.pim.surface_graphs[surface]
            out = self.model(
                *[surface_graph[k] for k in ["X_real", "X_img", "edge_index"]],
                query_edges_doubled,
                surface_graph["edge_weight"],
                edge_features=ef_doubled
            )
            
            if self.mode == 'pointwise':
                from model.graph.hierarchical import raw_to_serve, match_prob as hier_match
                pa = raw_to_serve(out[:, 0], self.gender, surface, self.scale)
                pb = raw_to_serve(out[:, 1], self.gender, surface, self.scale)
                bo = torch.cat([data['best_of'], data['best_of']])
                pm = hier_match(pa, pb, bo)
                pm = torch.nan_to_num(pm, nan=0.5).clamp(1e-7, 1-1e-7)
                loss = F.binary_cross_entropy(pm, y_doubled.float())
            elif self.mode == 'regression':
                loss = self.criterion(out.squeeze(-1), y_doubled.float())
            elif 'market_probs' in data and data['market_probs'] is not None and self.gamma > 0.0:
                market_probs_doubled = torch.cat([
                    data['market_probs'],
                    1 - data['market_probs']
                ], dim=0)
                model_probs = torch.softmax(out, dim=1)[:, 1]
                loss = self.decorrelation_loss(model_probs, market_probs_doubled, y_doubled)
            else:
                loss = self.criterion(out, y_doubled)

            # Weight and accumulate
            weight = surface_weights[surface]
            total_loss += loss * weight
            total_weight += weight
        
        if total_weight > 0:
            # Backpropagate averaged loss
            avg_loss = total_loss / total_weight
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return avg_loss.item()
        
        return 0.0

    def train(self, query_edges, y, surface, market_probs=None, best_of=None):
        ef = self._build_edge_features(surface, query_edges)
        query_edges_doubled, y_doubled = self.handle_permutation_variance(query_edges, y)
        ef_doubled = self._double_edge_features(ef) if ef is not None else None
        self.model.train()
        surface_graph = self.pim.surface_graphs[surface]

        out = self.model(
            *[surface_graph[k] for k in ["X_real", "X_img", "edge_index"]],
            query_edges_doubled,
            surface_graph["edge_weight"],
            edge_features=ef_doubled
        )

        if self.mode == 'pointwise':
            from model.graph.hierarchical import raw_to_serve, match_prob as hier_match
            pa = raw_to_serve(out[:, 0], self.gender, surface, self.scale)
            pb = raw_to_serve(out[:, 1], self.gender, surface, self.scale)
            bo = torch.cat([best_of, best_of]) if best_of is not None else torch.full((out.size(0),), 3, device=device)
            pm = hier_match(pa, pb, bo)
            pm = torch.nan_to_num(pm, nan=0.5).clamp(1e-7, 1-1e-7)
            logloss = F.binary_cross_entropy(pm, y_doubled.float())
        elif self.mode == 'regression':
            logloss = self.criterion(out.squeeze(-1), y_doubled.float())
        elif market_probs is not None and self.gamma > 0.0:
            market_probs_doubled = torch.cat([
                market_probs,
                1 - market_probs
            ], dim=0)
            model_probs = torch.softmax(out, dim=1)[:, 1]
            logloss = self.decorrelation_loss(model_probs, market_probs_doubled, y_doubled)
        else:
            logloss = self.criterion(out, y_doubled)
        
        self.optimizer.zero_grad()
        logloss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return logloss.detach().item()

    def predict_window(self, query_edges, surface, best_of=None):
        self.model.eval()
        with torch.no_grad():
            ef = self._build_edge_features(surface, query_edges)
            surface_graph = self.pim.surface_graphs[surface]
            doubled_edges = self.handle_permutation_variance(query_edges)
            ef_doubled = self._double_edge_features(ef) if ef is not None else None

            out = self.model(
                *[surface_graph[k] for k in ["X_real", "X_img", "edge_index"]],
                doubled_edges,
                surface_graph["edge_weight"],
                edge_features=ef_doubled
            )

            n = query_edges.size(0)

            if self.mode == 'pointwise':
                from model.graph.hierarchical import raw_to_serve, match_prob as hier_match
                # Avg serve probs from both edge orientations
                pa = (raw_to_serve(out[:n, 0], self.gender, surface, self.scale)
                      + raw_to_serve(out[n:, 1], self.gender, surface, self.scale)) / 2
                pb = (raw_to_serve(out[:n, 1], self.gender, surface, self.scale)
                      + raw_to_serve(out[n:, 0], self.gender, surface, self.scale)) / 2
                if best_of is None:
                    best_of = torch.full((n,), 3, device=device)
                pm = hier_match(pa, pb, best_of)
                return torch.stack([1 - pm, pm], dim=1), pa, pb

            if self.mode == 'regression':
                preds = out.squeeze(-1)
                # Symmetric avg: orig should be ≈ -rev, so (orig - rev)/2
                return (preds[:n] - preds[n:]) / 2

            probs = torch.softmax(out, dim=1)
            orig = probs[:n]
            rev_flipped = torch.stack([probs[n:][:, 1], probs[n:][:, 0]], dim=1)
            return (orig + rev_flipped) / 2.0

    def _build_edge_features(self, surface, query_edges):
        """Extract edge features from PIM for query edges. Returns None if n_edge_features=0."""
        if self.n_edge_features == 0:
            return None
        return self.pim.get_edge_features(surface, query_edges)[:, :self.n_edge_features]

    def _double_edge_features(self, ef):
        """Reverse edge features for permutation variance doubling."""
        rev = ef.clone()
        for i, t in enumerate(self.edge_feature_types):
            if t == 'signed': rev[:, i] *= -1
            elif t == 'prob': rev[:, i] = 1.0 - rev[:, i]
        for a, b in self.swap_pairs:  # Swap A↔B stats
            rev[:, a], rev[:, b] = ef[:, b].clone(), ef[:, a].clone()
        return torch.cat((ef, rev), dim=0)

    def handle_permutation_variance(self, query_edges, y=None):
        reversed_edges = query_edges.flip(1)
        doubled_edges = torch.cat((query_edges, reversed_edges), dim=0)

        if y is not None:
            reversed_y = -y if self.mode == 'regression' else 1 - y
            doubled_y = torch.cat((y, reversed_y), dim=0)
            return doubled_edges, doubled_y

        return doubled_edges
