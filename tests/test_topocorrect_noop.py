import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule


class _Config:
    pass


class _Tokenizer:
    def __init__(self, ids, encoded=None):
        self.ids = ids
        self.encoded = encoded or {token: [token_id] for token, token_id in ids.items()}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, token, add_special_tokens=False):
        return self.encoded[token]


class TopoCorrectNoOpTest(unittest.TestCase):
    def setUp(self):
        self.module = TopoCorrectStateModule(_Config())
        self.video_end = torch.randn(1, 8)
        self.navigation = torch.randn(1, 8)

    def test_zero_gates_are_exact_and_noop(self):
        self.assertEqual(self.module.topo_gate.item(), 0.0)
        self.assertEqual(self.module.nav_gate.item(), 0.0)
        torch.testing.assert_close(self.module.build_zero_topo_delta(self.video_end), torch.zeros_like(self.video_end))
        torch.testing.assert_close(self.module.build_zero_nav_delta(self.navigation), torch.zeros_like(self.navigation))
        torch.testing.assert_close(self.module.build_tst_embedding(self.video_end), self.video_end)
        torch.testing.assert_close(self.module.build_nst_embedding(self.navigation), self.navigation)

    def test_two_replacements_keep_synthetic_sequence_length(self):
        prefix = torch.randn(3, 8)
        history = torch.randn(4, 8)
        image_start = torch.randn(1, 8)
        current = torch.randn(64, 8)
        image_end = torch.randn(1, 8)
        instruction = torch.randn(2, 8)
        baseline = torch.cat(
            [prefix, history, self.video_end, image_start, current, image_end, self.navigation, instruction]
        )
        replacement = torch.cat(
            [prefix, history, self.module.build_tst_embedding(self.video_end), image_start,
             current, image_end, self.module.build_nst_embedding(self.navigation), instruction]
        )
        self.assertEqual(baseline.shape, replacement.shape)
        torch.testing.assert_close(baseline, replacement)
        self.assertEqual(baseline.shape[0], 77)

    def test_tokenizer_validation_and_bad_token_error(self):
        ids = {"</video_special>": 101, "[Navigation]": 104}
        self.module.validate_tokenizer_ids(_Tokenizer(ids), 101, 104)
        with self.assertRaisesRegex(RuntimeError, "TopoCorrect token validation failed"):
            self.module.validate_tokenizer_ids(_Tokenizer(ids), 102, 104)
        with self.assertRaisesRegex(RuntimeError, "TopoCorrect token validation failed"):
            self.module.validate_tokenizer_ids(
                _Tokenizer(ids, {"</video_special>": [101, 9], "[Navigation]": [104]}), 101, 104
            )


if __name__ == "__main__":
    unittest.main()
