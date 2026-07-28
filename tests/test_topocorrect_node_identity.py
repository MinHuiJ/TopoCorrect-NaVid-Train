import types
import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule


def _config():
    return types.SimpleNamespace(
        hidden_size=8, topocorrect_state_dim=8, topocorrect_num_heads=2,
        topocorrect_num_layers=1, topocorrect_num_actions=4, topocorrect_action_dim=4,
        topocorrect_dropout=0.0, topocorrect_nav_num_heads=2,
        topocorrect_nav_dropout=0.0, topocorrect_max_topo_nodes=3,
        topocorrect_max_action_history=8,
    )


class TopoCorrectNodeIdentityTest(unittest.TestCase):
    def test_truncated_nodes_keep_original_identity_and_time_adjacency(self):
        torch.manual_seed(43)
        module = TopoCorrectStateModule(_config())
        _, outputs = module.topological_state_encoder(
            [torch.randn(5, 8)], [[1, 1, 1, 1, 1]], torch.randn(1, 64, 8),
        )
        self.assertTrue(torch.equal(outputs["original_node_indices"], torch.tensor([[0, 3, 4]])))
        self.assertTrue(torch.equal(outputs["original_time_indices"], torch.tensor([[0, 3, 4]])))
        adjacency = outputs["relation_features"][0, :, :, 2]
        self.assertEqual(adjacency[0, 1].item(), 0.0)  # original 0 and 3 are not adjacent.
        self.assertEqual(adjacency[1, 2].item(), 1.0)  # original 3 and 4 remain adjacent.
        mapped = module.map_recovery_target_to_original(
            torch.tensor([1]), outputs["original_node_indices"], outputs["original_time_indices"],
            ~outputs["padding_mask"],
        )
        self.assertEqual(mapped["original_node_indices"].item(), 3)
        self.assertEqual(mapped["original_time_indices"].item(), 3)

    def test_padded_recovery_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "padded topology node"):
            TopoCorrectStateModule.map_recovery_target_to_original(
                torch.tensor([1]), torch.tensor([[3, -1]]), torch.tensor([[3, -1]]),
                torch.tensor([[True, False]]),
            )


if __name__ == "__main__":
    unittest.main()
