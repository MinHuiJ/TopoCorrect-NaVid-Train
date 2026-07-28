import inspect
import unittest

import torch

from uninavid.model.language_model.llava_llama_vid import LlavaLlamaAttForCausalLM
from uninavid.model.uninavid_arch import UniNaVIDMetaForCausalLM


class InstructionInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.instruction_ids = torch.tensor([[101, 102, 103, 104]], dtype=torch.long)
        self.instruction_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)
        self.instruction_ids_alt = torch.tensor([[201, 202, 203, 204]], dtype=torch.long)

    def _harness(self):
        from test_topocorrect_navigation_integration import TopoCorrectNavigationIntegrationTest

        case = TopoCorrectNavigationIntegrationTest()
        case.setUp()
        case.harness.config.use_topological_state_encoder = True
        return case

    def _prepare(self, case, instruction_ids=None, instruction_mask=None):
        return case.harness.prepare_inputs_labels_for_multimodal(
            case.input_ids, case.mask.clone(), None, case.labels.clone(), case.images,
            prompts=case.prompts, use_topocorrect_tokens=True,
            instruction_ids=instruction_ids, instruction_attention_mask=instruction_mask,
        )

    def test_signatures_and_generation_forwarding(self):
        for method in (LlavaLlamaAttForCausalLM.forward,
                       UniNaVIDMetaForCausalLM.prepare_inputs_labels_for_multimodal):
            parameters = inspect.signature(method).parameters
            self.assertIn("instruction_ids", parameters)
            self.assertIn("instruction_attention_mask", parameters)
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        first = LlavaLlamaAttForCausalLM.prepare_inputs_for_generation(
            object(), input_ids, instruction_ids=self.instruction_ids,
            instruction_attention_mask=self.instruction_mask,
        )
        cached = LlavaLlamaAttForCausalLM.prepare_inputs_for_generation(
            object(), input_ids, past_key_values=(object(),), instruction_ids=self.instruction_ids,
            instruction_attention_mask=self.instruction_mask,
        )
        for result in (first, cached):
            self.assertIs(result["instruction_ids"], self.instruction_ids)
            self.assertIs(result["instruction_attention_mask"], self.instruction_mask)
        self.assertTrue(torch.equal(cached["input_ids"], input_ids[:, -1:]))

    def test_none_empty_and_false_mask_are_noops(self):
        case = self._harness()
        baseline = self._prepare(case)
        provided = self._prepare(case, self.instruction_ids, self.instruction_mask)
        empty = self._prepare(case, torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.bool))
        false_mask = self._prepare(case, self.instruction_ids, torch.zeros_like(self.instruction_mask))
        for result in (provided, empty, false_mask):
            torch.testing.assert_close(baseline[3], result[3])
            torch.testing.assert_close(baseline[1], result[1])
            torch.testing.assert_close(baseline[4], result[4])
            self.assertEqual(baseline[3].shape, result[3].shape)
            self.assertEqual(baseline[3].shape[1], 77)

    def test_different_instruction_values_do_not_affect_tst(self):
        case = self._harness()
        self._prepare(case, self.instruction_ids, self.instruction_mask)
        topo_first = case.harness.get_model().topocorrect_state.last_topo_outputs["topo_state"].clone()
        self._prepare(case, self.instruction_ids_alt, self.instruction_mask)
        topo_second = case.harness.get_model().topocorrect_state.last_topo_outputs["topo_state"]
        torch.testing.assert_close(topo_first, topo_second, rtol=0.0, atol=0.0)

    def test_invalid_instruction_inputs_raise_value_error(self):
        case = self._harness()
        invalid_cases = (
            (self.instruction_ids, None),
            (None, self.instruction_mask),
            (self.instruction_ids.float(), self.instruction_mask),
            (self.instruction_ids, self.instruction_mask.long()),
            (self.instruction_ids[:, :3], self.instruction_mask),
            (self.instruction_ids.unsqueeze(0), self.instruction_mask.unsqueeze(0)),
            (self.instruction_ids.repeat(2, 1), self.instruction_mask.repeat(2, 1)),
        )
        for ids, mask in invalid_cases:
            with self.assertRaises(ValueError):
                self._prepare(case, ids, mask)


if __name__ == "__main__":
    unittest.main()
