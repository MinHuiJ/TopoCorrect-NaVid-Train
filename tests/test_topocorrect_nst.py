import types
import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule


def _config():
    return types.SimpleNamespace(
        hidden_size=4096, topocorrect_state_dim=512, topocorrect_num_heads=8,
        topocorrect_num_layers=2, topocorrect_num_actions=4, topocorrect_action_dim=128,
        topocorrect_dropout=0.0, topocorrect_nav_num_heads=8, topocorrect_nav_dropout=0.0,
        topocorrect_num_failure_types=8, topocorrect_num_recovery_modes=4,
    )


class NavigationStateEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.module = TopoCorrectStateModule(_config())
        self.histories = [torch.randn(5, 4096), torch.randn(5, 4096)]
        self.lengths = [[4, 1], [1, 4]]
        self.current = torch.randn(2, 64, 4096)
        _, self.topo = self.module.topological_state_encoder(
            self.histories, self.lengths, self.current,
            torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            torch.tensor([[True, True], [True, False]], dtype=torch.bool),
        )
        self.ids = torch.tensor([[101, 102, 103], [201, 202, 203]], dtype=torch.long)
        self.mask = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)
        self.embeddings = torch.randn(2, 3, 4096, requires_grad=True)

    def _run(self, ids=None, mask=None, embeddings=None, topo=None):
        return self.module.navigation_state_encoder(
            self.ids if ids is None else ids, self.mask if mask is None else mask,
            self.embeddings if embeddings is None else embeddings, self.current,
            (self.topo if topo is None else topo)["topo_state"],
            (self.topo if topo is None else topo)["node_states"],
            (self.topo if topo is None else topo)["padding_mask"],
            (self.topo if topo is None else topo)["action_summary"],
        )

    def test_shapes_progress_recovery_and_gradients(self):
        delta, outputs = self._run()
        self.assertEqual(delta.shape, (2, 1, 4096))
        self.assertEqual(outputs["landmark_logits"].shape, (2, 3))
        self.assertEqual(outputs["instruction_pointer_logits"].shape, (2, 3))
        self.assertEqual(outputs["progress"].shape, (2, 1))
        self.assertTrue(((outputs["progress"] >= 0) & (outputs["progress"] <= 1)).all())
        self.assertEqual(outputs["failure_logits"].shape, (2, 8))
        self.assertEqual(outputs["recovery_mode_logits"].shape, (2, 4))
        self.assertEqual(outputs["recovery_target_logits"].shape, (2, 2))
        self.assertTrue(torch.isfinite(delta).all())
        (delta.square().mean() + torch.nan_to_num(outputs["recovery_target_logits"], neginf=0.0).square().mean()).backward()
        encoder = self.module.navigation_state_encoder
        for parameter in (
            encoder.instruction_projector.weight, encoder.landmark_head[0].weight,
            encoder.nav_query_mlp[0].weight, encoder.cross_attention.in_proj_weight,
            encoder.fusion[0].weight, encoder.delta_projection[0].weight,
            encoder.recovery_query.weight,
        ):
            self.assertIsNotNone(parameter.grad)

    def test_empty_false_mask_single_node_and_sensitivity(self):
        empty_ids = torch.empty(2, 0, dtype=torch.long)
        empty_mask = torch.empty(2, 0, dtype=torch.bool)
        empty_embeddings = torch.empty(2, 0, 4096)
        delta_empty, outputs_empty = self._run(empty_ids, empty_mask, empty_embeddings)
        self.assertTrue(torch.isfinite(delta_empty).all())
        self.assertEqual(outputs_empty["progress"].shape, (2, 1))
        false_mask = torch.zeros_like(self.mask)
        delta_false, _ = self._run(mask=false_mask)
        self.assertTrue(torch.isfinite(delta_false).all())
        _, single_topo = self.module.topological_state_encoder(
            [torch.randn(1, 4096)], [[1]], torch.randn(1, 64, 4096)
        )
        single_delta, single_outputs = self.module.navigation_state_encoder(
            self.ids[:1], self.mask[:1], self.embeddings[:1].detach(), torch.randn(1, 64, 4096),
            single_topo["topo_state"], single_topo["node_states"], single_topo["padding_mask"],
            single_topo["action_summary"],
        )
        self.assertEqual(single_outputs["recovery_target_logits"].shape, (1, 1))
        self.assertTrue(torch.isfinite(single_delta).all())
        alt_delta, _ = self._run(embeddings=self.embeddings.detach() + 1.0)
        self.assertFalse(torch.equal(delta_false, alt_delta))
        altered_topo = dict(self.topo)
        altered_topo["topo_state"] = altered_topo["topo_state"] + 0.5
        topo_delta, _ = self._run(topo=altered_topo)
        self.assertFalse(torch.equal(alt_delta, topo_delta))


if __name__ == "__main__":
    unittest.main()
