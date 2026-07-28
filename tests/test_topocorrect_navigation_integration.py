"""Synthetic integration tests for the actual multimodal navigation branch.

They intentionally require the project's transformers dependency, but never load
weights, a vision tower, or Habitat.
"""

import types
import unittest

import torch
import torch.nn as nn

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    transformers = None


@unittest.skipIf(transformers is None, "requires the project transformers dependency")
class TopoCorrectNavigationIntegrationTest(unittest.TestCase):
    def setUp(self):
        from topocorrect_navid.model import TopoCorrectStateModule
        from uninavid.constants import IMAGE_TOKEN_INDEX
        from uninavid.model.uninavid_arch import UniNaVIDMetaForCausalLM

        class _DummyVisionTower(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError(
                    "Dummy vision tower should not be called because the test "
                    "harness supplies synthetic encoded image features."
                )

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(128, 8)
                self.topocorrect_state = TopoCorrectStateModule(types.SimpleNamespace(
                    hidden_size=8, topocorrect_state_dim=8, topocorrect_num_heads=2,
                    topocorrect_num_layers=2, topocorrect_num_actions=4,
                    topocorrect_action_dim=4, topocorrect_dropout=0.0,
                ))
                self.vision_tower = _DummyVisionTower()

            def get_vision_tower(self):
                return self.vision_tower

        class _Harness(UniNaVIDMetaForCausalLM):
            def __init__(self):
                self.model = _Backbone()
                self.config = types.SimpleNamespace(
                    compress_type="grid:2", tune_mm_mlp_adapter=False,
                    mm_use_im_start_end=False, use_topocorrect_tokens=False,
                    topocorrect_video_end_token_id=21,
                    topocorrect_image_start_token_id=22,
                    topocorrect_image_end_token_id=23,
                    topocorrect_navigation_token_id=24,
                )
                self.history = torch.randn(1, 4, 8)
                self.current = torch.randn(1, 64, 8)

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def dtype(self):
                return torch.float32

            def get_model(self):
                return self.model

            def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
                return [self.history], [True], [self.current], [[4]]

        self.harness = _Harness()
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.input_ids = torch.tensor([[10, 20, 25, IMAGE_TOKEN_INDEX, 21, 22, 23, 24, 30, 31]])
        self.labels = self.input_ids.clone()
        self.mask = torch.ones_like(self.input_ids)
        self.images = [torch.zeros(1, 3, 2, 2)]
        self.prompts = [["a video of historical observations and an image of the current observation"]]

    def _prepare(self, enabled, input_ids=None):
        return self.harness.prepare_inputs_labels_for_multimodal(
            self.input_ids if input_ids is None else input_ids, self.mask.clone(), None, self.labels.clone(), self.images,
            prompts=self.prompts, use_topocorrect_tokens=enabled,
        )

    def test_baseline_and_zero_gated_paths_are_exact_noops(self):
        baseline = self._prepare(False)
        replaced = self._prepare(True)
        _, baseline_mask, _, baseline_embeds, baseline_labels = baseline
        _, replaced_mask, _, replaced_embeds, replaced_labels = replaced
        self.assertEqual(self.harness.get_model().topocorrect_state.topo_gate.item(), 0.0)
        self.assertEqual(self.harness.get_model().topocorrect_state.nav_gate.item(), 0.0)
        self.assertEqual(baseline_embeds.shape, replaced_embeds.shape)
        self.assertEqual(baseline_embeds.shape[1], 77)  # raw 10, sentinel replaced by 68 embeddings
        torch.testing.assert_close(baseline_embeds, replaced_embeds)
        torch.testing.assert_close(baseline_mask, replaced_mask)
        torch.testing.assert_close(baseline_labels, replaced_labels)

    def test_bad_navigation_order_raises_runtime_error(self):
        bad = self.input_ids.clone()
        bad[0, 7] = 99
        with self.assertRaisesRegex(RuntimeError, "TopoCorrect navigation token order"):
            self._prepare(True, input_ids=bad)


if __name__ == "__main__":
    unittest.main()
