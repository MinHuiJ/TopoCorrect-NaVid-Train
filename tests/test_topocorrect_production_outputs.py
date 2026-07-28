"""Synthetic production-path coverage for formal, per-forward batch outputs."""
import types
import unittest

import torch
import torch.nn as nn

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    transformers = None


@unittest.skipIf(transformers is None, "requires the project transformers dependency")
class TopoCorrectProductionOutputsTest(unittest.TestCase):
    def setUp(self):
        from topocorrect_navid.model import TopoCorrectStateModule
        from uninavid.constants import IMAGE_TOKEN_INDEX
        from uninavid.model.uninavid_arch import UniNaVIDMetaForCausalLM

        class _Vision(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError("Synthetic features must bypass the vision tower.")

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(256, 8)
                self.vision_tower = _Vision()
                self.topocorrect_state = TopoCorrectStateModule(types.SimpleNamespace(
                    hidden_size=8, topocorrect_state_dim=8, topocorrect_num_heads=2,
                    topocorrect_num_layers=1, topocorrect_num_actions=4,
                    topocorrect_action_dim=4, topocorrect_dropout=0.0,
                    topocorrect_nav_num_heads=2, topocorrect_nav_dropout=0.0,
                    topocorrect_num_failure_types=8, topocorrect_num_recovery_modes=4,
                    topocorrect_max_topo_nodes=8, topocorrect_max_action_history=8,
                ))

            def get_vision_tower(self):
                return self.vision_tower

        class _Harness(UniNaVIDMetaForCausalLM):
            def __init__(self):
                self.model = _Backbone()
                self.config = types.SimpleNamespace(
                    compress_type="grid:2", tune_mm_mlp_adapter=False, mm_use_im_start_end=False,
                    use_topocorrect_tokens=True, use_topological_state_encoder=True,
                    use_navigation_state_encoder=True, topocorrect_video_end_token_id=21,
                    topocorrect_image_start_token_id=22, topocorrect_image_end_token_id=23,
                    topocorrect_navigation_token_id=24,
                )
                self.histories = [torch.randn(3, 8), torch.randn(6, 8)]
                self.currents = [torch.randn(1, 64, 8), torch.randn(1, 64, 8)]

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def dtype(self):
                return torch.float32

            def get_model(self):
                return self.model

            def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
                return [[history] for history in self.histories], [True, True], self.currents, [[1, 1, 1], [1] * 6]

        self.harness = _Harness()
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.input_ids = torch.tensor([
            [10, 20, 25, IMAGE_TOKEN_INDEX, 21, 22, 23, 24, 30, 31],
            [11, 20, 25, IMAGE_TOKEN_INDEX, 21, 22, 23, 24, 30, 31],
        ])
        self.labels = self.input_ids.clone()
        self.attention_mask = torch.ones_like(self.input_ids)
        self.images = [torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)]
        self.instruction_ids = torch.arange(9, dtype=torch.long).repeat(2, 1)
        self.instruction_mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 9])

    def test_prepare_collects_both_samples_and_resets_diagnostic_outputs(self):
        baseline = self.harness.prepare_inputs_labels_for_multimodal(
            self.input_ids, self.attention_mask, None, self.labels, self.images,
            prompts=[["a"], ["b"]], use_topocorrect_tokens=False,
            instruction_ids=self.instruction_ids, instruction_attention_mask=self.instruction_mask,
        )
        result = self.harness.prepare_inputs_labels_for_multimodal(
            self.input_ids, self.attention_mask, None, self.labels, self.images,
            prompts=[["a"], ["b"]], use_topocorrect_tokens=True,
            instruction_ids=self.instruction_ids, instruction_attention_mask=self.instruction_mask,
        )
        _, returned_mask, _, embeds, returned_labels = result
        outputs = self.harness.get_model().topocorrect_state.current_batch_outputs
        self.assertEqual(outputs["topo"]["node_states"].shape, (2, 6, 8))
        self.assertEqual(outputs["nav"]["instruction_mask"].shape, (2, 9))
        self.assertTrue(outputs["topo"]["node_mask"][0, :3].all())
        self.assertFalse(outputs["topo"]["node_mask"][0, 3:].any())
        self.assertTrue(torch.equal(outputs["nav"]["instruction_mask"], self.instruction_mask))
        _, baseline_mask, _, baseline_embeds, baseline_labels = baseline
        self.assertEqual(embeds.shape, baseline_embeds.shape)
        self.assertEqual(returned_labels.shape, baseline_labels.shape)
        self.assertEqual(returned_mask.shape, baseline_mask.shape)
        torch.testing.assert_close(embeds, baseline_embeds)
        torch.testing.assert_close(returned_labels, baseline_labels)
        torch.testing.assert_close(returned_mask, baseline_mask)
        self.assertIsNotNone(self.harness.get_model().topocorrect_state.last_topo_outputs)
        self.assertFalse(self.harness.get_model().topocorrect_state.last_topo_outputs["node_states"].requires_grad)
        # A new no-op preparation must clear both formal and diagnostic state.
        self.harness.prepare_inputs_labels_for_multimodal(
            self.input_ids[:, :1], self.attention_mask[:, :1], None, None, self.images,
        )
        self.assertIsNone(self.harness.get_model().topocorrect_state.last_topo_outputs)
        self.assertEqual(self.harness.get_model().topocorrect_state.current_batch_outputs, [])

    def test_forward_adds_outputs_only_when_requested(self):
        from uninavid.model.language_model.llava_llama_vid import LlavaConfig, LlavaLlamaAttForCausalLM

        config = LlavaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, use_cache=True,
            compress_type="grid:2",
            topocorrect_state_dim=8, topocorrect_num_heads=2, topocorrect_num_layers=1,
            topocorrect_action_dim=4, topocorrect_nav_num_heads=2,
        )
        model = LlavaLlamaAttForCausalLM(config)
        model.train()  # avoids evaluation-only image device handling; no image is supplied.
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ordinary = model(input_ids=input_ids, return_topocorrect_outputs=False)
        requested = model(input_ids=input_ids, return_topocorrect_outputs=True)
        self.assertFalse(hasattr(ordinary, "topocorrect_outputs"))
        self.assertEqual(requested.topocorrect_outputs, [])
        self.assertEqual(ordinary.logits.shape, requested.logits.shape)
        self.assertEqual(ordinary.past_key_values[0][0].shape, requested.past_key_values[0][0].shape)


if __name__ == "__main__":
    unittest.main()
