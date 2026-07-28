import inspect
import unittest

import torch

from uninavid.model.language_model.llava_llama_vid import LlavaLlamaAttForCausalLM
from uninavid.model.uninavid_arch import UniNaVIDMetaForCausalLM


class TopoCorrectActionInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.action_history = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        self.action_history_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    def test_signatures_accept_optional_action_history_fields(self):
        forward_parameters = inspect.signature(LlavaLlamaAttForCausalLM.forward).parameters
        multimodal_parameters = inspect.signature(
            UniNaVIDMetaForCausalLM.prepare_inputs_labels_for_multimodal
        ).parameters
        self.assertIn("action_history", forward_parameters)
        self.assertIn("action_history_mask", forward_parameters)
        self.assertIn("action_history", multimodal_parameters)
        self.assertIn("action_history_mask", multimodal_parameters)
        self.assertIsNone(forward_parameters["action_history"].default)
        self.assertIsNone(forward_parameters["action_history_mask"].default)

    def test_generation_preserves_fields_on_first_and_cached_steps(self):
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        images = [torch.zeros(1, 3, 2, 2)]
        first = LlavaLlamaAttForCausalLM.prepare_inputs_for_generation(
            object(), input_ids, attention_mask=torch.ones_like(input_ids), images=images,
            use_cache=True, action_history=self.action_history,
            action_history_mask=self.action_history_mask,
        )
        cached = LlavaLlamaAttForCausalLM.prepare_inputs_for_generation(
            object(), input_ids, past_key_values=(object(),), attention_mask=torch.ones_like(input_ids),
            images=images, use_cache=True, action_history=self.action_history,
            action_history_mask=self.action_history_mask,
        )
        for model_inputs in (first, cached):
            self.assertIs(model_inputs["action_history"], self.action_history)
            self.assertIs(model_inputs["action_history_mask"], self.action_history_mask)
        self.assertTrue(torch.equal(cached["input_ids"], input_ids[:, -1:]))

    def test_generation_omits_none_action_fields(self):
        input_ids = torch.tensor([[5, 6]], dtype=torch.long)
        model_inputs = LlavaLlamaAttForCausalLM.prepare_inputs_for_generation(
            object(), input_ids, attention_mask=torch.ones_like(input_ids), images=None, use_cache=True,
        )
        self.assertNotIn("action_history", model_inputs)
        self.assertNotIn("action_history_mask", model_inputs)
        self.assertEqual(set(model_inputs), {
            "input_ids", "past_key_values", "use_cache", "attention_mask", "images"
        })

    def test_action_fields_are_multimodal_noops_at_zero_gate(self):
        from test_topocorrect_navigation_integration import TopoCorrectNavigationIntegrationTest

        navigation_case = TopoCorrectNavigationIntegrationTest()
        navigation_case.setUp()
        harness = navigation_case.harness
        baseline = harness.prepare_inputs_labels_for_multimodal(
            navigation_case.input_ids, navigation_case.mask.clone(), None,
            navigation_case.labels.clone(), navigation_case.images, prompts=navigation_case.prompts,
            use_topocorrect_tokens=True,
        )
        with_actions = harness.prepare_inputs_labels_for_multimodal(
            navigation_case.input_ids, navigation_case.mask.clone(), None,
            navigation_case.labels.clone(), navigation_case.images, prompts=navigation_case.prompts,
            use_topocorrect_tokens=True, action_history=self.action_history,
            action_history_mask=self.action_history_mask,
        )
        _, baseline_mask, _, baseline_embeds, baseline_labels = baseline
        _, actions_mask, _, actions_embeds, actions_labels = with_actions
        torch.testing.assert_close(baseline_embeds, actions_embeds)
        torch.testing.assert_close(baseline_labels, actions_labels)
        torch.testing.assert_close(baseline_mask, actions_mask)
        self.assertEqual(baseline_embeds.shape, actions_embeds.shape)
        self.assertEqual(baseline_embeds.shape[1], 77)


if __name__ == "__main__":
    unittest.main()
