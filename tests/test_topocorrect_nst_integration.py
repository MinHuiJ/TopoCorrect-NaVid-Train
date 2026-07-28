import unittest

import torch


class NavigationStateIntegrationTest(unittest.TestCase):
    def setUp(self):
        from test_topocorrect_navigation_integration import TopoCorrectNavigationIntegrationTest

        torch.manual_seed(17)
        self.case = TopoCorrectNavigationIntegrationTest()
        self.case.setUp()
        self.case.harness.config.use_topological_state_encoder = True
        self.case.harness.config.use_navigation_state_encoder = True
        self.ids = torch.tensor([[101, 102, 103]], dtype=torch.long)
        self.mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    def _prepare(self, enabled, ids=None):
        return self.case.harness.prepare_inputs_labels_for_multimodal(
            self.case.input_ids, self.case.mask.clone(), None, self.case.labels.clone(), self.case.images,
            prompts=self.case.prompts, use_topocorrect_tokens=enabled,
            instruction_ids=self.ids if ids is None else ids,
            instruction_attention_mask=self.mask,
            action_history=torch.tensor([[0, 1]], dtype=torch.long),
            action_history_mask=torch.tensor([[True, True]], dtype=torch.bool),
        )

    def test_zero_gates_are_exact_baseline_noop_and_cache_isolated(self):
        baseline = self._prepare(False)
        zero = self._prepare(True)
        torch.testing.assert_close(baseline[3], zero[3])
        torch.testing.assert_close(baseline[1], zero[1])
        torch.testing.assert_close(baseline[4], zero[4])
        self.assertEqual(baseline[3].shape[1], zero[3].shape[1])
        state = self.case.harness.get_model().topocorrect_state
        self.assertFalse(hasattr(self.case.harness.get_model(), "feat_cache"))
        self.assertFalse(hasattr(self.case.harness.get_model(), "long_feat_cache"))
        self.assertIsNotNone(state.last_nav_outputs)

    def test_nonzero_nav_gate_changes_only_navigation_position(self):
        state = self.case.harness.get_model().topocorrect_state
        with torch.no_grad():
            state.nav_gate.fill_(1.0)
        changed = self._prepare(True)[3]
        with torch.no_grad():
            state.nav_gate.zero_()
        baseline = self._prepare(True)[3]
        differing = torch.where((changed - baseline).abs().amax(dim=-1)[0] > 0)[0].tolist()
        self.assertEqual(differing, [74])
        torch.testing.assert_close(changed[0, 7], baseline[0, 7])


if __name__ == "__main__":
    unittest.main()
