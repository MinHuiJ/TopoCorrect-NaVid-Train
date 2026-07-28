import types
import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule


def _config():
    return types.SimpleNamespace(
        hidden_size=8, topocorrect_state_dim=8, topocorrect_num_heads=2,
        topocorrect_num_layers=1, topocorrect_num_actions=4, topocorrect_action_dim=4,
        topocorrect_dropout=0.0, topocorrect_nav_num_heads=2,
        topocorrect_nav_dropout=0.0, topocorrect_num_failure_types=8,
        topocorrect_num_recovery_modes=4, topocorrect_max_topo_nodes=8,
        topocorrect_max_action_history=8,
    )


class TopoCorrectBatchOutputsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.module = TopoCorrectStateModule(_config())

    def _sample(self, node_count, instruction_length, valid_instruction_count):
        history = torch.randn(node_count, 8)
        current = torch.randn(1, 64, 8)
        _, topo = self.module.topological_state_encoder([history], [[1] * node_count], current)
        instruction_ids = torch.arange(instruction_length, dtype=torch.long).unsqueeze(0)
        instruction_mask = torch.zeros(1, instruction_length, dtype=torch.bool)
        instruction_mask[:, :valid_instruction_count] = True
        instruction_embeddings = torch.randn(1, instruction_length, 8)
        _, nav = self.module.navigation_state_encoder(
            instruction_ids, instruction_mask, instruction_embeddings, current,
            topo["topo_state"], topo["node_states"], topo["padding_mask"], topo["action_summary"],
        )
        topo["delta_topo"] = torch.randn(1, 1, 8)
        return {"topo": topo, "nav": nav}

    def test_variable_nodes_and_instructions_are_padded_as_a_current_batch(self):
        outputs = self.module.collate_batch_outputs([
            self._sample(node_count=3, instruction_length=5, valid_instruction_count=5),
            self._sample(node_count=6, instruction_length=9, valid_instruction_count=9),
        ])
        topo, nav = outputs["topo"], outputs["nav"]
        self.assertEqual(topo["delta_topo"].shape, (2, 1, 8))
        self.assertEqual(topo["topo_state"].shape, (2, 1, 8))
        self.assertEqual(topo["node_states"].shape, (2, 6, 8))
        self.assertEqual(nav["delta_nav"].shape, (2, 1, 8))
        self.assertEqual(nav["landmark_logits"].shape, (2, 9))
        self.assertTrue(topo["node_mask"][0, :3].all())
        self.assertFalse(topo["node_mask"][0, 3:].any())
        self.assertTrue(topo["node_mask"][1].all())
        self.assertTrue(nav["instruction_mask"][0, :5].all())
        self.assertTrue(nav["instruction_mask"][1, :9].all())
        self.assertTrue(torch.equal(topo["original_node_indices"][0, :3], torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(topo["original_time_indices"][1, :6], torch.arange(6)))
        self.assertTrue((topo["original_node_indices"][0, 3:] == -1).all())
        self.assertTrue((topo["original_time_indices"][0, 3:] == -1).all())
        self.assertTrue((topo["node_states"][0, 3:] == 0).all())
        self.assertTrue((topo["revisit_logits"][0, 3:] == 0).all())
        self.assertTrue((nav["recovery_target_logits"][0, 3:] == float("-inf")).all())


if __name__ == "__main__":
    unittest.main()
