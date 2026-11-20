import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVTransferDestination
from sglang.srt.disaggregation.mooncake.conn import (
    OWNER_MLA_V1,
    TransferInfo,
    _owned_mla_ptrs,
    _owner_for_peer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOwnerMLA(unittest.TestCase):
    def test_both_room_capability_and_local_switch_required(self):
        manager = SimpleNamespace(attn_tp_rank=0, enable_owner_mla_send=False)
        peer = SimpleNamespace(dst_tp_rank=0, dst_attn_tp_size=4)
        legacy = [
            b"42",
            b"127.0.0.1",
            b"17",
            b"test",
            np.array([5], dtype=np.int32).tobytes(),
            b"1",
            b"",
            b"1",
            b"0",
            b"",
        ]
        for suffix, enabled, expected in [
            ([], True, None),
            ([b"host"], True, None),
            ([b"host", OWNER_MLA_V1], False, None),
            ([b"host", b"unknown"], True, None),
            ([b"host", OWNER_MLA_V1], True, 0),
        ]:
            manager.enable_owner_mla_send = enabled
            req = TransferInfo.from_zmq(legacy + suffix)
            self.assertEqual(_owner_for_peer(manager, req, peer), expected)
            self.assertEqual(
                req.destination,
                KVTransferDestination.HOST if suffix else KVTransferDestination.DEVICE,
            )

    def test_only_target_is_shared(self):
        ptrs = list(range(1001, 1080))
        dst = list(range(2101, 2180))
        stride = 64 * 576
        kv = SimpleNamespace(
            kv_data_ptrs=ptrs,
            kv_item_lens=[stride] * 79,
            page_size=64,
            num_draft_entries=1,
            prefill_start_layer=0,
            mla_compression_ratios=None,
            kv_layer_ids=[],
        )
        union = []
        for rank in range(4):
            manager = SimpleNamespace(
                kv_args=kv,
                pp_size=1,
                is_mla_backend=True,
                attn_tp_rank=rank,
                attn_tp_size=4,
            )
            keep = _owned_mla_ptrs(manager, rank, dst, 4, stride, [])
            self.assertIn(ptrs[-1], keep)
            union.extend(keep - {ptrs[-1]})
            self.assertEqual(keep - {ptrs[-1]}, set(ptrs[rank:78:4]))
        self.assertCountEqual(union, ptrs[:78])


if __name__ == "__main__":
    unittest.main()
