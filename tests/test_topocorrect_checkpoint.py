import io
import unittest

import torch

from topocorrect_navid.model import TopoCorrectStateModule


class _Config:
    use_topocorrect_tokens = True


class TopoCorrectCheckpointTest(unittest.TestCase):
    def test_state_dict_roundtrip_preserves_gates_and_config_value(self):
        source = TopoCorrectStateModule(_Config())
        with torch.no_grad():
            source.topo_gate.fill_(0.25)
            source.nav_gate.fill_(-0.5)
        payload = {"config": {"use_topocorrect_tokens": _Config.use_topocorrect_tokens},
                   "state_dict": source.state_dict()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        target = TopoCorrectStateModule(_Config())
        target.load_state_dict(restored["state_dict"])
        self.assertTrue(restored["config"]["use_topocorrect_tokens"])
        torch.testing.assert_close(source.topo_gate, target.topo_gate)
        torch.testing.assert_close(source.nav_gate, target.nav_gate)


if __name__ == "__main__":
    unittest.main()
