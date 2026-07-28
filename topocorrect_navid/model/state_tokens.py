"""Zero-gated placeholder state tokens for the first TopoCorrect integration."""

import torch
import torch.nn as nn


class TopoCorrectStateModule(nn.Module):
    """Owns the two gates and currently produces only zero residuals.

    Full TST/NST encoders intentionally do not live here yet.  Keeping the
    module under ``model.topocorrect_state`` makes ordinary model state_dicts
    retain its parameters without changing the tokenizer or sequence length.
    """

    def __init__(self, config):
        super().__init__()
        self.topo_gate = nn.Parameter(torch.zeros(()))
        self.nav_gate = nn.Parameter(torch.zeros(()))

    def build_zero_topo_delta(self, base_video_end_embedding):
        return torch.zeros_like(base_video_end_embedding)

    def build_zero_nav_delta(self, base_navigation_embedding):
        return torch.zeros_like(base_navigation_embedding)

    def build_tst_embedding(self, base_video_end_embedding):
        delta_topo = self.build_zero_topo_delta(base_video_end_embedding)
        return base_video_end_embedding - torch.tanh(self.topo_gate) * delta_topo

    def build_nst_embedding(self, base_navigation_embedding):
        delta_nav = self.build_zero_nav_delta(base_navigation_embedding)
        return base_navigation_embedding - torch.tanh(self.nav_gate) * delta_nav

    @staticmethod
    def validate_tokenizer_ids(tokenizer, video_end_token_id, navigation_token_id):
        """Require the configured IDs to be exactly the two single tokens."""
        expected = (
            ("</video_special>", video_end_token_id),
            ("[Navigation]", navigation_token_id),
        )
        for token, expected_id in expected:
            actual_id = tokenizer.convert_tokens_to_ids(token)
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if actual_id != expected_id or encoded != [expected_id]:
                raise RuntimeError(
                    "TopoCorrect token validation failed for {!r}: expected single "
                    "token ID {}, got convert_tokens_to_ids={} and encode={}".format(
                        token, expected_id, actual_id, encoded
                    )
                )
