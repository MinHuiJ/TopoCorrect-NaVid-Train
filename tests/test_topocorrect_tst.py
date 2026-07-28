import types
import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule


def _config():
    return types.SimpleNamespace(
        hidden_size=4096, topocorrect_state_dim=512, topocorrect_num_heads=8,
        topocorrect_num_layers=2, topocorrect_num_actions=4, topocorrect_action_dim=128,
        topocorrect_dropout=0.0,
    )


class TopologicalStateEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = TopoCorrectStateModule(_config())
        self.histories = [torch.randn(5, 4096), torch.randn(5, 4096)]
        self.lengths = [[4, 1], [1, 4]]
        self.current = torch.randn(2, 64, 4096)

    def test_shapes_relations_auxiliary_outputs_and_gradients(self):
        delta, outputs = self.module.topological_state_encoder(
            self.histories, self.lengths, self.current,
            torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long),
            torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool),
        )
        self.assertEqual(delta.shape, (2, 1, 4096))
        self.assertEqual(outputs["node_states"].shape, (2, 2, 512))
        self.assertEqual(outputs["relation_bias"].shape, (2, 8, 2, 2))
        self.assertEqual(outputs["revisit_logits"].shape, (2, 2))
        self.assertEqual(outputs["novelty_logit"].shape, (2, 1))
        self.assertEqual(outputs["stagnation_logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(delta).all())
        delta.square().mean().backward()
        for parameter in (
            self.module.topological_state_encoder.node_pooler.project.weight,
            self.module.topological_state_encoder.relation_mlp[0].weight,
            self.module.topological_state_encoder.layers[0].attention.in_proj_weight,
            self.module.topological_state_encoder.topo_query,
            self.module.topological_state_encoder.output_projection[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)

    def test_none_empty_and_masked_actions_are_finite(self):
        for actions, mask in (
            (None, None),
            (torch.empty(2, 0, dtype=torch.long), torch.empty(2, 0, dtype=torch.bool)),
            (torch.tensor([[0, 1], [2, 3]]), torch.zeros(2, 2, dtype=torch.bool)),
        ):
            delta, _ = self.module.topological_state_encoder(self.histories, self.lengths, self.current, actions, mask)
            self.assertTrue(torch.isfinite(delta).all())

    def test_invalid_group_metadata_raises(self):
        with self.assertRaisesRegex(ValueError, "sum to history token count"):
            self.module.topological_state_encoder([self.histories[0]], [[4]], self.current[:1])
        with self.assertRaisesRegex(ValueError, "positive"):
            self.module.topological_state_encoder([self.histories[0]], [[5, 0]], self.current[:1])


if __name__ == "__main__":
    unittest.main()
