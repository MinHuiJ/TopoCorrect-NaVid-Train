import unittest

import torch


class TopologicalStateIntegrationTest(unittest.TestCase):
    def setUp(self):
        from test_topocorrect_navigation_integration import TopoCorrectNavigationIntegrationTest

        torch.manual_seed(11)
        self.navigation_case = TopoCorrectNavigationIntegrationTest()
        self.navigation_case.setUp()
        self.harness = self.navigation_case.harness
        self.harness.config.use_topological_state_encoder = True

    def _prepare(self, enabled=True):
        return self.harness.prepare_inputs_labels_for_multimodal(
            self.navigation_case.input_ids, self.navigation_case.mask.clone(), None,
            self.navigation_case.labels.clone(), self.navigation_case.images,
            prompts=self.navigation_case.prompts, use_topocorrect_tokens=enabled,
            action_history=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            action_history_mask=torch.tensor([[True, True, True, True]], dtype=torch.bool),
        )

    def test_zero_gate_is_exact_noop_and_cache_is_untouched(self):
        baseline = self._prepare(False)
        zero_gate = self._prepare(True)
        _, baseline_mask, _, baseline_embeds, baseline_labels = baseline
        _, zero_mask, _, zero_embeds, zero_labels = zero_gate
        state = self.harness.get_model().topocorrect_state
        self.assertEqual(state.topo_gate.item(), 0.0)
        self.assertIsNotNone(state.last_topo_outputs)
        torch.testing.assert_close(baseline_embeds, zero_embeds)
        torch.testing.assert_close(baseline_mask, zero_mask)
        torch.testing.assert_close(baseline_labels, zero_labels)
        self.assertFalse(hasattr(self.harness.get_model(), "feat_cache"))
        self.assertFalse(hasattr(self.harness.get_model(), "long_feat_cache"))

    def test_nonzero_gate_changes_only_video_end_position(self):
        state = self.harness.get_model().topocorrect_state
        with torch.no_grad():
            state.topo_gate.fill_(1.0)
        changed = self._prepare()[3]
        with torch.no_grad():
            state.topo_gate.zero_()
        baseline = self._prepare()[3]
        differing = torch.where((changed - baseline).abs().amax(dim=-1)[0] > 0)[0].tolist()
        self.assertEqual(differing, [7])
        self.assertFalse(torch.equal(changed[0, 7], baseline[0, 7]))
        torch.testing.assert_close(changed[0, 74], baseline[0, 74])


if __name__ == "__main__":
    unittest.main()
