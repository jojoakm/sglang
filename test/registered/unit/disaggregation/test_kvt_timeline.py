import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
    kvt_timeline,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def logged_payload(call):
    assert call.args[0] == "KVT_TIMELINE %s"
    return json.loads(call.args[1])


class TestKVTTimeline(unittest.TestCase):
    def test_disabled_by_default(self):
        with envs.SGLANG_DEBUG_KVT_TIMELINE.override(False):
            with patch("sglang.srt.disaggregation.common.utils.logger.info") as log:
                kvt_timeline("test", "event", room=1)
        log.assert_not_called()

    def test_emits_json_and_samples_by_room(self):
        with (
            envs.SGLANG_DEBUG_KVT_TIMELINE.override(True),
            envs.SGLANG_DEBUG_KVT_TIMELINE_SAMPLE_EVERY.override(2),
            patch("sglang.srt.disaggregation.common.utils.logger.info") as log,
        ):
            kvt_timeline("test", "skip", room=1)
            kvt_timeline("test", "event", room=2, duration_ms=3.5)
        payload = logged_payload(log.call_args)
        self.assertEqual(payload["event"], "event")
        self.assertEqual(payload["component"], "test")
        self.assertEqual(payload["room"], 2)
        self.assertEqual(payload["duration_ms"], 3.5)
        self.assertIn("ts", payload)

    def test_native_transfer_logs_begin_and_end(self):
        manager = SimpleNamespace(
            engine=MagicMock(batch_transfer_sync=MagicMock(return_value=0))
        )
        with (
            envs.SGLANG_DEBUG_KVT_TIMELINE.override(True),
            patch("sglang.srt.disaggregation.common.utils.logger.info") as log,
        ):
            ret = MooncakeKVManager._transfer_data(
                manager,
                "session",
                [(1000, 2000, 16)],
                timeline={"room": 7, "worker": 1},
            )
        self.assertEqual(ret, 0)
        events = [logged_payload(call) for call in log.call_args_list]
        self.assertEqual(
            [row["event"] for row in events], ["engine_begin", "engine_end"]
        )
        self.assertEqual(events[0]["room"], 7)
        self.assertEqual(events[0]["bytes"], 16)
        self.assertEqual(events[1]["ret"], 0)
        self.assertGreaterEqual(events[1]["duration_ms"], 0)

    def test_enqueue_assigns_timestamp_and_shard_metadata(self):
        manager = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            request_status={7: 2},
            transfer_infos={7: {"127.0.0.1:9000": object()}},
            transfer_queues=[FastQueue()],
            check_status=lambda room: 2,
        )
        with (
            envs.SGLANG_DEBUG_KVT_TIMELINE.override(True),
            patch("sglang.srt.disaggregation.common.utils.logger.info") as log,
        ):
            MooncakeKVManager.add_transfer_request(
                manager,
                7,
                np.array([4, 5], dtype=np.int32),
                slice(0, 2),
                False,
                num_kv_tokens=128,
            )
        self.assertEqual(manager.transfer_queues[0].qsize(), 1)
        chunk = manager.transfer_queues[0].get()
        self.assertIsInstance(chunk, TransferKVChunk)
        self.assertGreater(chunk.enqueued_at, 0)
        payload = logged_payload(log.call_args)
        self.assertEqual(payload["event"], "enqueue")
        self.assertEqual(payload["room"], 7)
        self.assertEqual(payload["worker"], 0)
        self.assertEqual(payload["session"], "127.0.0.1:9000")
        self.assertEqual(payload["num_pages"], 2)
        self.assertEqual(payload["num_kv_tokens"], 128)
        self.assertEqual(payload["queue_depth"], 1)


if __name__ == "__main__":
    unittest.main()
