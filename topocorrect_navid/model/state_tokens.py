"""TopoCorrect state tokens: a relation-aware RGB-only topological state token."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHistoryEncoder(nn.Module):
    def __init__(self, num_actions, action_dim, state_dim, max_action_history):
        super().__init__()
        self.num_actions = num_actions
        self.max_action_history = max_action_history
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
            if self.max_action_history is not None and actions.numel() > self.max_action_history:
                actions = actions[-self.max_action_history:]
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
    def __init__(self, hidden_size, state_dim, num_heads, num_layers, num_actions, action_dim, dropout, max_topo_nodes, max_action_history):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.max_topo_nodes = max_topo_nodes
        self.node_pooler = ObservationNodePooler(hidden_size, state_dim)
        self.current_pooler = ObservationNodePooler(hidden_size, state_dim)
        self.action_encoder = ActionHistoryEncoder(num_actions, action_dim, state_dim, max_action_history)
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
            original_indices = list(range(len(lengths)))
            if self.max_topo_nodes is not None and len(lengths) > self.max_topo_nodes:
                # Deterministically retain the first node and most recent nodes.
                keep_indices = [0] + list(range(len(lengths) - (self.max_topo_nodes - 1), len(lengths)))
                offsets, start = [], 0
                for length in lengths:
                    offsets.append((start, start + length))
                    start += length
                history = torch.cat([history[left:right] for left, right in (offsets[i] for i in keep_indices)], dim=0)
                lengths = [lengths[i] for i in keep_indices]
                original_indices = [original_indices[i] for i in keep_indices]
            nodes, compressed = [], []
            offset = 0
            total = len(lengths)
            for index, length in enumerate(lengths):
                visual_node = self.node_pooler(history[offset:offset + length])
                temporal = 0.0 if len(original_indices) == 1 else original_indices[index] / max(original_indices[-1], 1)
                extras = visual_node.new_tensor([temporal, math.log1p(length), float(length == 1)])
                nodes.append((visual_node, extras, original_indices[index]))
                compressed.append(length == 1)
                offset += length
            node_lists.append(nodes)
            compression_lists.append(compressed)
        max_nodes = max(len(nodes) for nodes in node_lists)
        padded_nodes = current_summary.new_zeros(batch_size, max_nodes, self.state_dim)
        raw_nodes = current_summary.new_zeros(batch_size, max_nodes, self.state_dim)
        padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
        compressed_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        original_node_indices = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=device)
        for batch_index, nodes in enumerate(node_lists):
            for node_index, (visual_node, extras, original_index) in enumerate(nodes):
                raw_nodes[batch_index, node_index] = visual_node
                padded_nodes[batch_index, node_index] = self.node_fusion(torch.cat([
                    visual_node, current_summary[batch_index], action_summary[batch_index], extras
                ]))
                padding_mask[batch_index, node_index] = False
                compressed_mask[batch_index, node_index] = compression_lists[batch_index][node_index]
                original_node_indices[batch_index, node_index] = original_index
        normalized_nodes = F.normalize(raw_nodes, dim=-1)
        visual_similarity = torch.matmul(normalized_nodes, normalized_nodes.transpose(1, 2))
        temporal_distance = raw_nodes.new_zeros(batch_size, max_nodes, max_nodes)
        adjacency = raw_nodes.new_zeros(batch_size, max_nodes, max_nodes)
        for batch_index, nodes in enumerate(node_lists):
            node_count = len(nodes)
            positions = original_node_indices[batch_index, :node_count].to(raw_nodes.dtype)
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
            "relation_features": relation_features,
            "revisit_logits": revisit_logits,
            "novelty_logit": self.novelty_head(topo_state.squeeze(1)),
            "stagnation_logit": self.stagnation_head(topo_state.squeeze(1)),
            "padding_mask": padding_mask,
            "action_summary": action_summary,
            "current_visual_summary": current_summary,
            "original_node_indices": original_node_indices,
            "original_time_indices": original_node_indices.clone(),
        }
        return delta_topo, outputs


class NavigationStateEncoder(nn.Module):
    """Landmark-aware instruction grounding conditioned on the TST state."""
    def __init__(self, hidden_size, state_dim, num_heads, dropout, num_failure_types, num_recovery_modes):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.instruction_norm = nn.LayerNorm(hidden_size)
        self.instruction_projector = nn.Linear(hidden_size, state_dim)
        self.landmark_head = nn.Sequential(nn.Linear(state_dim, state_dim // 2), nn.GELU(), nn.Linear(state_dim // 2, 1))
        self.current_pooler = ObservationNodePooler(hidden_size, state_dim)
        self.nav_query_mlp = nn.Sequential(nn.Linear(state_dim * 3, state_dim), nn.GELU(), nn.LayerNorm(state_dim))
        self.cross_attention = nn.MultiheadAttention(state_dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(state_dim * 5, state_dim), nn.GELU(), nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim), nn.GELU(), nn.LayerNorm(state_dim),
        )
        self.delta_projection = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size))
        self.pointer_key = nn.Linear(state_dim, state_dim, bias=False)
        self.progress_confidence_head = nn.Linear(state_dim, 1)
        self.unknown_progress = nn.Parameter(torch.zeros(()))
        self.stop_readiness_head = nn.Linear(state_dim, 1)
        self.failure_head = nn.Linear(state_dim, num_failure_types)
        self.correction_head = nn.Linear(state_dim, 1)
        self.recovery_mode_head = nn.Linear(state_dim, num_recovery_modes)
        self.recovery_query = nn.Linear(state_dim, state_dim, bias=False)
        self.recovery_key = nn.Linear(state_dim, state_dim, bias=False)

    def _validate(self, instruction_ids, instruction_attention_mask, instruction_embeddings,
                  current_visual_tokens, topo_state, topo_node_states, topo_node_mask, action_summary):
        batch_size = current_visual_tokens.shape[0]
        TopoCorrectStateModule.validate_instruction_inputs(instruction_ids, instruction_attention_mask, batch_size)
        if current_visual_tokens.ndim != 3 or current_visual_tokens.shape[1:] != (64, self.hidden_size):
            raise ValueError("current_visual_tokens must have shape [B, 64, hidden_size].")
        if instruction_embeddings.ndim != 3 or instruction_embeddings.shape[:2] != instruction_ids.shape or instruction_embeddings.shape[2] != self.hidden_size:
            raise ValueError("instruction_embeddings must have shape [B, L, hidden_size].")
        if topo_state.shape != (batch_size, 1, self.state_dim):
            raise ValueError("topo_state must have shape [B, 1, state_dim].")
        if topo_node_states.ndim != 3 or topo_node_states.shape[0] != batch_size or topo_node_states.shape[2] != self.state_dim:
            raise ValueError("topo_node_states must have shape [B, N, state_dim].")
        if topo_node_mask.shape != topo_node_states.shape[:2] or topo_node_mask.dtype != torch.bool:
            raise ValueError("topo_node_mask must be bool with shape [B, N].")
        if topo_node_mask.all(dim=1).any():
            raise ValueError("Each sample must provide at least one unmasked topology node.")
        if action_summary.shape != (batch_size, self.state_dim):
            raise ValueError("action_summary must have shape [B, state_dim].")

    def forward(self, instruction_ids, instruction_attention_mask, instruction_embeddings,
                current_visual_tokens, topo_state, topo_node_states, topo_node_mask, action_summary):
        self._validate(instruction_ids, instruction_attention_mask, instruction_embeddings,
                       current_visual_tokens, topo_state, topo_node_states, topo_node_mask, action_summary)
        batch_size, instruction_length = instruction_ids.shape
        current_summary = torch.stack([self.current_pooler(tokens) for tokens in current_visual_tokens], dim=0)
        nav_query = self.nav_query_mlp(torch.cat([current_summary, topo_state.squeeze(1), action_summary], dim=-1)).unsqueeze(1)
        if instruction_length == 0:
            instruction_context = nav_query.new_zeros(batch_size, 1, self.state_dim)
            landmark_logits = nav_query.new_zeros(batch_size, 0)
            pointer_logits = nav_query.new_zeros(batch_size, 0)
            pointer_probs = nav_query.new_zeros(batch_size, 0)
            progress = torch.sigmoid(self.unknown_progress).expand(batch_size, 1)
        else:
            instruction_states = self.instruction_projector(self.instruction_norm(instruction_embeddings))
            landmark_logits = self.landmark_head(instruction_states).squeeze(-1)
            valid = instruction_attention_mask
            safe_states = instruction_states.clone()
            safe_mask = ~valid.clone()
            empty_rows = ~valid.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = False
                safe_states[empty_rows, 0] = 0
            landmark_bias = torch.log(torch.sigmoid(landmark_logits).clamp_min(1e-6))
            attn_mask = landmark_bias[:, None, :].expand(-1, self.cross_attention.num_heads, -1)
            attn_mask = attn_mask.reshape(batch_size * self.cross_attention.num_heads, 1, instruction_length)
            instruction_context, instruction_attention = self.cross_attention(
                nav_query, safe_states, safe_states, key_padding_mask=safe_mask,
                attn_mask=attn_mask, need_weights=True, average_attn_weights=True,
            )
            instruction_context = torch.where(valid.any(dim=1)[:, None, None], instruction_context,
                                              torch.zeros_like(instruction_context))
            pointer_logits = torch.bmm(nav_query, self.pointer_key(instruction_states).transpose(1, 2)).squeeze(1)
            pointer_logits = pointer_logits + landmark_logits
            pointer_logits = pointer_logits.masked_fill(~valid, float("-inf"))
            pointer_probs = torch.zeros_like(pointer_logits)
            valid_rows = valid.any(dim=1)
            if valid_rows.any():
                pointer_probs[valid_rows] = torch.softmax(pointer_logits[valid_rows], dim=-1)
            ranks = valid.to(nav_query.dtype).cumsum(dim=-1) - 1
            valid_counts = valid.sum(dim=-1, keepdim=True).to(nav_query.dtype)
            positions = torch.where(valid, ranks / (valid_counts - 1).clamp_min(1), torch.zeros_like(ranks))
            progress = (pointer_probs * positions).sum(dim=-1, keepdim=True)
            progress = torch.where(valid_rows[:, None], progress,
                                   torch.sigmoid(self.unknown_progress).expand(batch_size, 1))
        pointer_context = torch.bmm(pointer_probs.unsqueeze(1), instruction_states).squeeze(1) if instruction_length else current_summary.new_zeros(batch_size, self.state_dim)
        nav_state = self.fusion(torch.cat([
            instruction_context.squeeze(1), pointer_context, current_summary, topo_state.squeeze(1), action_summary
        ], dim=-1)).unsqueeze(1)
        delta_nav = self.delta_projection(nav_state)
        recovery_target_logits = torch.bmm(
            self.recovery_query(nav_state), self.recovery_key(topo_node_states).transpose(1, 2)
        ).squeeze(1) / math.sqrt(self.state_dim)
        recovery_target_logits = recovery_target_logits.masked_fill(topo_node_mask, float("-inf"))
        return delta_nav, {
            "delta_nav": delta_nav,
            "landmark_logits": landmark_logits,
            "instruction_pointer_logits": pointer_logits,
            "instruction_pointer_probs": pointer_probs,
            "instruction_mask": instruction_attention_mask,
            "progress": progress.clamp(0.0, 1.0),
            "progress_confidence": torch.sigmoid(self.progress_confidence_head(nav_state.squeeze(1))),
            "stop_readiness_logit": self.stop_readiness_head(nav_state.squeeze(1)),
            "failure_logits": self.failure_head(nav_state.squeeze(1)),
            "correction_logit": self.correction_head(nav_state.squeeze(1)),
            "recovery_mode_logits": self.recovery_mode_head(nav_state.squeeze(1)),
            "recovery_target_logits": recovery_target_logits,
            "nav_state": nav_state,
            "instruction_cross_attention": instruction_attention.squeeze(1) if instruction_length else pointer_probs,
            "instruction_pointer_probs": pointer_probs,
        }


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
            getattr(config, "topocorrect_max_topo_nodes", 64), getattr(config, "topocorrect_max_action_history", 32),
        )
        self.navigation_state_encoder = NavigationStateEncoder(
            hidden_size, state_dim, getattr(config, "topocorrect_nav_num_heads", 8),
            getattr(config, "topocorrect_nav_dropout", 0.0),
            getattr(config, "topocorrect_num_failure_types", 8),
            getattr(config, "topocorrect_num_recovery_modes", 4),
        )
        self.last_topo_outputs = None
        self.last_nav_outputs = None
        self.current_batch_outputs = []

    def clear_debug_outputs(self):
        self.last_topo_outputs = None
        self.last_nav_outputs = None
        self.current_batch_outputs = []

    def build_zero_topo_delta(self, base_video_end_embedding):
        return torch.zeros_like(base_video_end_embedding)

    def build_zero_nav_delta(self, base_navigation_embedding):
        return torch.zeros_like(base_navigation_embedding)

    def build_tst_embedding(self, base_video_end_embedding, history_visual_tokens=None,
                            history_group_lengths=None, current_visual_tokens=None,
                            action_history=None, action_history_mask=None, use_encoder=False, return_outputs=False):
        if not use_encoder:
            result = base_video_end_embedding - torch.tanh(self.topo_gate) * self.build_zero_topo_delta(base_video_end_embedding)
            return (result, None) if return_outputs else result
        original_shape = base_video_end_embedding.shape
        base = base_video_end_embedding.unsqueeze(1) if base_video_end_embedding.ndim == 2 else base_video_end_embedding
        delta_topo, outputs = self.topological_state_encoder(
            history_visual_tokens, history_group_lengths, current_visual_tokens, action_history, action_history_mask
        )
        self.last_topo_outputs = {name: value.detach() for name, value in outputs.items()}
        embedding = base + torch.tanh(self.topo_gate) * delta_topo
        outputs["delta_topo"] = delta_topo
        embedding = embedding.squeeze(1) if len(original_shape) == 2 else embedding
        return (embedding, outputs) if return_outputs else embedding

    def build_nst_embedding(self, base_navigation_embedding):
        return base_navigation_embedding - torch.tanh(self.nav_gate) * self.build_zero_nav_delta(base_navigation_embedding)

    def build_nst_embedding_with_encoder(self, base_navigation_embedding, instruction_ids,
                                         instruction_attention_mask, instruction_embeddings,
                                         current_visual_tokens, topo_outputs, use_encoder=False):
        if not use_encoder:
            return self.build_nst_embedding(base_navigation_embedding), None
        delta_nav, outputs = self.navigation_state_encoder(
            instruction_ids, instruction_attention_mask, instruction_embeddings,
            current_visual_tokens, topo_outputs["topo_state"], topo_outputs["node_states"],
            topo_outputs["padding_mask"], topo_outputs["action_summary"],
        )
        original_shape = base_navigation_embedding.shape
        base = base_navigation_embedding.unsqueeze(1) if base_navigation_embedding.ndim == 2 else base_navigation_embedding
        embedding = base + torch.tanh(self.nav_gate) * delta_nav
        embedding = embedding.squeeze(1) if len(original_shape) == 2 else embedding
        self.last_nav_outputs = {name: value.detach() for name, value in outputs.items()}
        return embedding, outputs

    @staticmethod
    def compute_topocorrect_aux_losses(outputs, labels, ignore_index=-100):
        """Optional supervised losses; labels are never consumed by inference paths."""
        if not labels:
            return None, {}
        losses = {}
        def bce(name, output_name):
            target = labels.get(name)
            if target is not None:
                losses[name] = F.binary_cross_entropy_with_logits(outputs[output_name], target.to(outputs[output_name]))
        def ce(name, output_name):
            target = labels.get(name)
            if target is not None:
                losses[name] = F.cross_entropy(outputs[output_name], target.to(outputs[output_name].device), ignore_index=ignore_index)
        if labels.get("pointer_labels") is not None:
            losses["pointer"] = F.cross_entropy(outputs["instruction_pointer_logits"], labels["pointer_labels"].to(outputs["instruction_pointer_logits"].device), ignore_index=ignore_index)
        if labels.get("progress_target") is not None:
            losses["progress"] = F.mse_loss(outputs["progress"], labels["progress_target"].to(outputs["progress"]))
        bce("correction_target", "correction_logit")
        bce("landmark_labels", "landmark_logits")
        ce("failure_target", "failure_logits")
        ce("recovery_mode_target", "recovery_mode_logits")
        ce("recovery_target_index", "recovery_target_logits")
        total = sum(losses.values()) if losses else None
        return total, losses

    @staticmethod
    def map_recovery_target_to_original(recovery_target_index, original_node_indices,
                                        original_time_indices, node_mask):
        """Map padded recovery indices to stable original node/time identities."""
        if recovery_target_index.ndim != 1:
            raise ValueError("recovery_target_index must have shape [B].")
        if original_node_indices.shape != original_time_indices.shape or node_mask.shape != original_node_indices.shape:
            raise ValueError("node identity tensors and node_mask must have identical [B, N] shape.")
        rows = torch.arange(recovery_target_index.shape[0], device=recovery_target_index.device)
        if (recovery_target_index < 0).any() or (recovery_target_index >= node_mask.shape[1]).any():
            raise ValueError("recovery target index is outside padded node range.")
        valid = node_mask[rows, recovery_target_index]
        if not valid.all():
            raise ValueError("recovery target points to a padded topology node.")
        return {
            "original_node_indices": original_node_indices[rows, recovery_target_index],
            "original_time_indices": original_time_indices[rows, recovery_target_index],
        }

    @staticmethod
    def collate_batch_outputs(sample_outputs):
        """Pad variable-node and variable-instruction outputs into a batch structure."""
        if not sample_outputs or not any(item.get("topo") is not None for item in sample_outputs):
            return None
        topo_template = next(item["topo"] for item in sample_outputs if item.get("topo") is not None)
        nav_template = next((item["nav"] for item in sample_outputs if item.get("nav") is not None), None)
        batch_size = len(sample_outputs)
        node_max = max((item["topo"]["node_states"].shape[1] if item.get("topo") is not None else 0) for item in sample_outputs)
        instruction_max = max((item["nav"]["landmark_logits"].shape[1] if item.get("nav") is not None else 0) for item in sample_outputs)
        device, dtype = topo_template["topo_state"].device, topo_template["topo_state"].dtype
        state_dim = topo_template["node_states"].shape[-1]
        hidden = topo_template["delta_topo"].shape[-1]
        topo = {
            "delta_topo": torch.zeros(batch_size, 1, hidden, device=device, dtype=dtype),
            "topo_state": torch.zeros(batch_size, 1, state_dim, device=device, dtype=dtype),
            "node_states": torch.zeros(batch_size, node_max, state_dim, device=device, dtype=dtype),
            "node_mask": torch.zeros(batch_size, node_max, device=device, dtype=torch.bool),
            "original_node_indices": torch.full((batch_size, node_max), -1, device=device, dtype=torch.long),
            "original_time_indices": torch.full((batch_size, node_max), -1, device=device, dtype=torch.long),
            "revisit_logits": torch.zeros(batch_size, node_max, device=device, dtype=dtype),
            "novelty_logit": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "stagnation_logit": torch.zeros(batch_size, 1, device=device, dtype=dtype),
        }
        nav = {
            "delta_nav": torch.zeros(batch_size, 1, hidden, device=device, dtype=dtype),
            "landmark_logits": torch.zeros(batch_size, instruction_max, device=device, dtype=dtype),
            "instruction_cross_attention": torch.zeros(batch_size, instruction_max, device=device, dtype=dtype),
            "instruction_pointer_logits": torch.zeros(batch_size, instruction_max, device=device, dtype=dtype),
            "instruction_pointer_probs": torch.zeros(batch_size, instruction_max, device=device, dtype=dtype),
            "instruction_mask": torch.zeros(batch_size, instruction_max, device=device, dtype=torch.bool),
            "progress": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "progress_confidence": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "stop_readiness_logit": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "failure_logits": torch.zeros(batch_size, (nav_template["failure_logits"].shape[-1] if nav_template is not None else 8), device=device, dtype=dtype),
            "correction_logit": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "recovery_mode_logits": torch.zeros(batch_size, (nav_template["recovery_mode_logits"].shape[-1] if nav_template is not None else 4), device=device, dtype=dtype),
            "recovery_target_logits": torch.full((batch_size, node_max), float("-inf"), device=device, dtype=dtype),
        }
        for batch_index, item in enumerate(sample_outputs):
            source_topo, source_nav = item.get("topo"), item.get("nav")
            if source_topo is None:
                continue
            count = source_topo["node_states"].shape[1]
            for key in ("delta_topo", "topo_state", "novelty_logit", "stagnation_logit"):
                topo[key][batch_index] = source_topo[key].squeeze(0)
            topo["node_states"][batch_index, :count] = source_topo["node_states"].squeeze(0)
            topo["node_mask"][batch_index, :count] = ~source_topo["padding_mask"].squeeze(0)
            topo["original_node_indices"][batch_index, :count] = source_topo["original_node_indices"].squeeze(0)
            topo["original_time_indices"][batch_index, :count] = source_topo["original_time_indices"].squeeze(0)
            topo["revisit_logits"][batch_index, :count] = source_topo["revisit_logits"].squeeze(0)
            if source_nav is None:
                continue
            length = source_nav["landmark_logits"].shape[1]
            for key in ("delta_nav", "progress", "progress_confidence", "stop_readiness_logit", "failure_logits", "correction_logit", "recovery_mode_logits"):
                nav[key][batch_index] = source_nav[key].squeeze(0)
            for key in ("landmark_logits", "instruction_cross_attention", "instruction_pointer_logits", "instruction_pointer_probs"):
                nav[key][batch_index, :length] = source_nav[key].squeeze(0)
            nav["instruction_mask"][batch_index, :length] = source_nav["instruction_mask"].squeeze(0)
            nav["recovery_target_logits"][batch_index, :count] = source_nav["recovery_target_logits"].squeeze(0)
        return {"topo": topo, "nav": nav}

    @staticmethod
    def validate_instruction_inputs(instruction_ids, instruction_attention_mask, batch_size):
        """Validate explicit raw-instruction tokens without reading their values."""
        if instruction_ids is None and instruction_attention_mask is None:
            return
        if instruction_ids is None or instruction_attention_mask is None:
            raise ValueError("instruction_ids and instruction_attention_mask must be provided together.")
        if instruction_ids.ndim != 2 or instruction_attention_mask.ndim != 2:
            raise ValueError("instruction_ids and instruction_attention_mask must have rank 2 [B, L].")
        if instruction_ids.dtype != torch.long:
            raise ValueError("instruction_ids must have dtype torch.long.")
        if instruction_attention_mask.dtype != torch.bool:
            raise ValueError("instruction_attention_mask must have dtype torch.bool.")
        if instruction_ids.shape != instruction_attention_mask.shape:
            raise ValueError("instruction_ids and instruction_attention_mask must have identical shapes.")
        if instruction_ids.shape[0] != batch_size:
            raise ValueError("instruction token batch size must match input_ids batch size.")

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
