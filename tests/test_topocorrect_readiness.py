import types
import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule
from uninavid.model.language_model.llava_llama_vid import LlavaConfig


def config(hidden=16, dim=8):
    return types.SimpleNamespace(
        hidden_size=hidden, topocorrect_state_dim=dim, topocorrect_num_heads=2,
        topocorrect_num_layers=2, topocorrect_num_actions=4, topocorrect_action_dim=4,
        topocorrect_dropout=0.0, topocorrect_nav_num_heads=2, topocorrect_nav_dropout=0.0,
        topocorrect_num_failure_types=8, topocorrect_num_recovery_modes=4,
        topocorrect_max_topo_nodes=3, topocorrect_max_action_history=2,
    )


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.module = TopoCorrectStateModule(config())
        self.current = torch.randn(2, 64, 16)
        _, self.topo = self.module.topological_state_encoder(
            [torch.randn(5, 16), torch.randn(4, 16)], [[1, 1, 1, 1, 1], [1, 1, 1, 1]], self.current,
            torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]), torch.ones(2, 4, dtype=torch.bool),
        )

    def _nav(self, ids, mask, embeddings=None, topo=None):
        if embeddings is None:
            embeddings = torch.randn(ids.shape[0], ids.shape[1], 16)
        topo = self.topo if topo is None else topo
        return self.module.navigation_state_encoder(
            ids, mask, embeddings, self.current[:ids.shape[0]], topo["topo_state"][:ids.shape[0]],
            topo["node_states"][:ids.shape[0]], topo["padding_mask"][:ids.shape[0]], topo["action_summary"][:ids.shape[0]],
        )

    def test_config_progress_pointer_and_auxiliary_gradients(self):
        self.assertEqual(LlavaConfig().topocorrect_dropout, 0.0)
        ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]], dtype=torch.long)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        embeddings = torch.randn(2, 4, 16, requires_grad=True)
        delta, out = self._nav(ids, mask, embeddings)
        # First sample's valid positions are normalized as [0,1], independent of padding length.
        self.assertTrue(torch.isfinite(out["progress"]).all())
        expected_first = out["instruction_pointer_probs"][0, 1]
        self.assertAlmostEqual(out["progress"][0, 0].item(), expected_first.item(), places=6)
        altered = embeddings.detach().clone(); altered[:, 0] += 3.0
        changed, _ = self._nav(ids, mask, altered)
        self.assertFalse(torch.equal(delta, changed))
        labels = {
            "pointer_labels": torch.tensor([0, 1]), "progress_target": torch.zeros(2, 1),
            "failure_target": torch.tensor([0, 1]), "correction_target": torch.zeros(2, 1),
            "recovery_mode_target": torch.tensor([0, 1]), "recovery_target_index": torch.tensor([0, 0]),
        }
        loss, pieces = self.module.compute_topocorrect_aux_losses(out, labels)
        self.assertIsNotNone(loss); self.assertEqual(set(pieces), {"pointer", "progress", "failure_target", "correction_target", "recovery_mode_target", "recovery_target_index"})
        loss.backward()
        for parameter in (self.module.navigation_state_encoder.pointer_key.weight,
                          self.module.navigation_state_encoder.failure_head.weight,
                          self.module.navigation_state_encoder.correction_head.weight,
                          self.module.navigation_state_encoder.recovery_mode_head.weight,
                          self.module.navigation_state_encoder.recovery_query.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0.0)

    def test_caps_all_mask_and_debug_clear(self):
        self.assertLessEqual(self.topo["node_states"].shape[1], 3)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self._nav(torch.tensor([[1]], dtype=torch.long), torch.tensor([[True]]), topo={
                **self.topo, "padding_mask": torch.ones_like(self.topo["padding_mask"][:1])
            })
        self.module.last_topo_outputs = {"old": torch.ones(1)}
        self.module.last_nav_outputs = {"old": torch.ones(1)}
        self.module.clear_debug_outputs()
        self.assertIsNone(self.module.last_topo_outputs)
        self.assertIsNone(self.module.last_nav_outputs)


if __name__ == "__main__":
    unittest.main()
