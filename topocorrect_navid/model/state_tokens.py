"""TopoCorrect state tokens: a relation-aware RGB-only topological state token."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHistoryEncoder(nn.Module):
    def __init__(self, num_actions, action_dim, state_dim):
        super().__init__()
        self.num_actions = num_actions
        self.embedding = nn.Embedding(num_actions, action_dim)
        self.gru = nn.GRU(action_dim, state_dim, batch_first=True)
        self.empty_action = nn.Parameter(torch.zeros(state_dim))

    def forward(self, action_history, action_history_mask, batch_size, device):
        if action_history is None:
            return self.empty_action.unsqueeze(0).expand(batch_size, -1)
        if action_history.ndim != 2 or action_history.shape[0] != batch_size:
            raise ValueError("action_history must have shape [B, T].")
        if action_history_mask is None:
            action_history_mask = torch.ones_like(action_history, dtype=torch.bool)
        if action_history_mask.shape != action_history.shape:
            raise ValueError("action_history_mask must have the same shape as action_history.")
        if action_history_mask.dtype != torch.bool:
            raise ValueError("action_history_mask must have dtype torch.bool.")
        if action_history.numel() and (action_history.min() < 0 or action_history.max() >= self.num_actions):
            raise ValueError("action_history contains an action ID outside configured num_actions.")
        summaries = []
        for actions, valid in zip(action_history, action_history_mask):
            actions = actions[valid]
            if actions.numel() == 0:
                summaries.append(self.empty_action)
                continue
            embedded = self.embedding(actions).unsqueeze(0)
            _, hidden = self.gru(embedded)
            summaries.append(hidden[-1, 0])
        return torch.stack(summaries, dim=0).to(device=device)


class ObservationNodePooler(nn.Module):
    """Learned attention pooling, never a mean-only group reduction."""
    def __init__(self, hidden_size, state_dim):
        super().__init__()
        self.project = nn.Linear(hidden_size, state_dim)
        self.score = nn.Sequential(nn.Linear(state_dim, state_dim), nn.Tanh(), nn.Linear(state_dim, 1))

    def forward(self, group_tokens):
        projected = self.project(group_tokens)
        weights = torch.softmax(self.score(projected).squeeze(-1), dim=0)
        return torch.sum(weights.unsqueeze(-1) * projected, dim=0)


class RelationAwareTransformerLayer(nn.Module):
    def __init__(self, state_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(state_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(state_dim)
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(state_dim * 4, state_dim)
        )
        self.norm2 = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states, relation_bias, padding_mask):
        batch_size, node_count, _ = node_states.shape
        attn_mask = relation_bias.reshape(batch_size * self.num_heads, node_count, node_count)
        attended, _ = self.attention(
            node_states, node_states, node_states, attn_mask=attn_mask, key_padding_mask=padding_mask,
            need_weights=False,
        )
        node_states = self.norm1(node_states + self.dropout(attended))
        return self.norm2(node_states + self.dropout(self.ffn(node_states)))


class TopologicalStateEncoder(nn.Module):
    """Builds one global state from dynamic observation/topological nodes."""
    def __init__(self, hidden_size, state_dim, num_heads, num_layers, num_actions, action_dim, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.node_pooler = ObservationNodePooler(hidden_size, state_dim)
        self.current_pooler = ObservationNodePooler(hidden_size, state_dim)
        self.action_encoder = ActionHistoryEncoder(num_actions, action_dim, state_dim)
        self.node_fusion = nn.Sequential(
            nn.Linear(state_dim * 3 + 3, state_dim), nn.GELU(), nn.LayerNorm(state_dim)
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(4, state_dim), nn.GELU(), nn.Linear(state_dim, num_heads)
        )
        self.layers = nn.ModuleList(
            [RelationAwareTransformerLayer(state_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.topo_query = nn.Parameter(torch.randn(1, 1, state_dim) / math.sqrt(state_dim))
        self.query_attention = nn.MultiheadAttention(state_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(state_dim)
        self.output_projection = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size))
        self.revisit_head = nn.Linear(state_dim, 1)
        self.novelty_head = nn.Linear(state_dim, 1)
        self.stagnation_head = nn.Linear(state_dim, 1)

    def _validate_inputs(self, histories, group_lengths, current_visual_tokens):
        if not isinstance(histories, (list, tuple)) or not isinstance(group_lengths, (list, tuple)):
            raise ValueError("history_visual_tokens and history_group_lengths must be lists by batch.")
        batch_size = len(histories)
        if batch_size == 0 or len(group_lengths) != batch_size:
            raise ValueError("history inputs must contain one non-empty entry per batch sample.")
        if current_visual_tokens.ndim != 3 or current_visual_tokens.shape[0] != batch_size:
            raise ValueError("current_visual_tokens must have shape [B, 64, hidden_size].")
        if current_visual_tokens.shape[1] != 64 or current_visual_tokens.shape[2] != self.hidden_size:
            raise ValueError("current_visual_tokens must have shape [B, 64, hidden_size].")
        for history, lengths in zip(histories, group_lengths):
            if history.ndim != 2 or history.shape[1] != self.hidden_size:
                raise ValueError("Each history visual tensor must have shape [N, hidden_size].")
            if not lengths or any(length <= 0 for length in lengths) or sum(lengths) != history.shape[0]:
                raise ValueError("history_group_lengths must be positive and sum to history token count.")

    def forward(self, history_visual_tokens, history_group_lengths, current_visual_tokens,
                action_history=None, action_history_mask=None):
        self._validate_inputs(history_visual_tokens, history_group_lengths, current_visual_tokens)
        batch_size = len(history_visual_tokens)
        device = current_visual_tokens.device
        current_summary = torch.stack(
            [self.current_pooler(tokens) for tokens in current_visual_tokens], dim=0
        )
        action_summary = self.action_encoder(action_history, action_history_mask, batch_size, device)
        node_lists, compression_lists = [], []
        for history, lengths in zip(history_visual_tokens, history_group_lengths):
            nodes, compressed = [], []
            offset = 0
            total = len(lengths)
            for index, length in enumerate(lengths):
                visual_node = self.node_pooler(history[offset:offset + length])
                temporal = 0.0 if total == 1 else index / (total - 1)
                extras = visual_node.new_tensor([temporal, math.log1p(length), float(length == 1)])
                nodes.append((visual_node, extras))
                compressed.append(length == 1)
                offset += length
            node_lists.append(nodes)
            compression_lists.append(compressed)
        max_nodes = max(len(nodes) for nodes in node_lists)
        padded_nodes = current_summary.new_zeros(batch_size, max_nodes, self.state_dim)
        raw_nodes = current_summary.new_zeros(batch_size, max_nodes, self.state_dim)
        padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
        compressed_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        for batch_index, nodes in enumerate(node_lists):
            for node_index, (visual_node, extras) in enumerate(nodes):
                raw_nodes[batch_index, node_index] = visual_node
                padded_nodes[batch_index, node_index] = self.node_fusion(torch.cat([
                    visual_node, current_summary[batch_index], action_summary[batch_index], extras
                ]))
                padding_mask[batch_index, node_index] = False
                compressed_mask[batch_index, node_index] = compression_lists[batch_index][node_index]
        normalized_nodes = F.normalize(raw_nodes, dim=-1)
        visual_similarity = torch.matmul(normalized_nodes, normalized_nodes.transpose(1, 2))
        temporal_distance = raw_nodes.new_zeros(batch_size, max_nodes, max_nodes)
        adjacency = raw_nodes.new_zeros(batch_size, max_nodes, max_nodes)
        for batch_index, nodes in enumerate(node_lists):
            node_count = len(nodes)
            positions = torch.arange(node_count, device=device, dtype=raw_nodes.dtype)
            distances = (positions[:, None] - positions[None, :]).abs()
            temporal_distance[batch_index, :node_count, :node_count] = distances / max(node_count - 1, 1)
            adjacency[batch_index, :node_count, :node_count] = (distances == 1).to(raw_nodes.dtype)
        same_compression = (compressed_mask[:, :, None] == compressed_mask[:, None, :]).to(raw_nodes.dtype)
        relation_features = torch.stack(
            [visual_similarity, temporal_distance, adjacency, same_compression], dim=-1
        )
        relation_bias = self.relation_mlp(relation_features).permute(0, 3, 1, 2)
        for layer in self.layers:
            padded_nodes = layer(padded_nodes, relation_bias, padding_mask)
        query = self.topo_query.expand(batch_size, -1, -1)
        topo_state, _ = self.query_attention(query, padded_nodes, padded_nodes, key_padding_mask=padding_mask)
        topo_state = self.query_norm(topo_state + query)
        delta_topo = self.output_projection(topo_state)
        revisit_logits = self.revisit_head(padded_nodes).squeeze(-1).masked_fill(padding_mask, 0.0)
        outputs = {
            "node_states": padded_nodes,
            "topo_state": topo_state,
            "relation_bias": relation_bias,
            "revisit_logits": revisit_logits,
            "novelty_logit": self.novelty_head(topo_state.squeeze(1)),
            "stagnation_logit": self.stagnation_head(topo_state.squeeze(1)),
            "padding_mask": padding_mask,
        }
        return delta_topo, outputs


class TopoCorrectStateModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = getattr(config, "hidden_size", 4096)
        state_dim = getattr(config, "topocorrect_state_dim", 512)
        self.topo_gate = nn.Parameter(torch.zeros(()))
        self.nav_gate = nn.Parameter(torch.zeros(()))
        self.topological_state_encoder = TopologicalStateEncoder(
            hidden_size, state_dim, getattr(config, "topocorrect_num_heads", 8),
            getattr(config, "topocorrect_num_layers", 2), getattr(config, "topocorrect_num_actions", 4),
            getattr(config, "topocorrect_action_dim", 128), getattr(config, "topocorrect_dropout", 0.0),
        )
        self.last_topo_outputs = None

    def build_zero_topo_delta(self, base_video_end_embedding):
        return torch.zeros_like(base_video_end_embedding)

    def build_zero_nav_delta(self, base_navigation_embedding):
        return torch.zeros_like(base_navigation_embedding)

    def build_tst_embedding(self, base_video_end_embedding, history_visual_tokens=None,
                            history_group_lengths=None, current_visual_tokens=None,
                            action_history=None, action_history_mask=None, use_encoder=False):
        if not use_encoder:
            return base_video_end_embedding - torch.tanh(self.topo_gate) * self.build_zero_topo_delta(base_video_end_embedding)
        original_shape = base_video_end_embedding.shape
        base = base_video_end_embedding.unsqueeze(1) if base_video_end_embedding.ndim == 2 else base_video_end_embedding
        delta_topo, outputs = self.topological_state_encoder(
            history_visual_tokens, history_group_lengths, current_visual_tokens, action_history, action_history_mask
        )
        self.last_topo_outputs = {name: value.detach() for name, value in outputs.items()}
        embedding = base + torch.tanh(self.topo_gate) * delta_topo
        return embedding.squeeze(1) if len(original_shape) == 2 else embedding

    def build_nst_embedding(self, base_navigation_embedding):
        return base_navigation_embedding - torch.tanh(self.nav_gate) * self.build_zero_nav_delta(base_navigation_embedding)

    @staticmethod
    def validate_tokenizer_ids(tokenizer, video_end_token_id, navigation_token_id):
        for token, expected_id in (("</video_special>", video_end_token_id), ("[Navigation]", navigation_token_id)):
            actual_id = tokenizer.convert_tokens_to_ids(token)
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if actual_id != expected_id or encoded != [expected_id]:
                raise RuntimeError(
                    "TopoCorrect token validation failed for {!r}: expected single token ID {}, got "
                    "convert_tokens_to_ids={} and encode={}".format(token, expected_id, actual_id, encoded)
                )
