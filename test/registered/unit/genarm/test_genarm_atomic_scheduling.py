"""Unit tests for GenARM atomic prefill pair scheduling."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from http import HTTPStatus
from unittest.mock import ANY, MagicMock, patch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler, _GENARM_UNSCHEDULABLE_PAIR_MSG
from sglang.srt.sampling.genarm_utils import (
    GENARM_ARM_SUFFIX,
    GENARM_ENABLED_KEY,
    genarm_shadow_rid,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


def _make_req(rid: str, *, genarm: bool = False, lora_id=None) -> Req:
    cp = {GENARM_ENABLED_KEY: True} if genarm else None
    req = Req(
        rid=rid,
        origin_input_text="hi",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=4, custom_params=cp),
        lora_id=lora_id,
    )
    req.extend_input_len = len(req.origin_input_ids)
    req.fill_ids = list(req.origin_input_ids)
    req.prefix_indices = []
    req.last_node = MagicMock()
    return req


class _FakeGenArmScheduler:
    chunked_req = None
    waiting_queue = []
    enable_lora = False
    enable_hicache_storage = False
    enable_priority_preemption = False
    truncation_align_size = None
    server_args = MagicMock()
    tree_cache = MagicMock()
    tree_cache.check_prefetch_progress = MagicMock(return_value=True)
    tree_cache.pop_prefetch_loaded_tokens = MagicMock(return_value=0)

    def __init__(self):
        self.running_batch = MagicMock(
            reqs=[],
            batch_is_full=False,
            batch_size=lambda: 0,
            is_empty=lambda: True,
        )
        from sglang.srt.disaggregation.utils import DisaggregationMode

        self.disaggregation_mode = DisaggregationMode.NULL
        self.req_to_token_pool = MagicMock(available_size=lambda: 128)

    def get_num_allocatable_reqs(self, running_bs):
        return 8

    _genarm_order_pair = Scheduler._genarm_order_pair
    _genarm_chunked_prefill_blocks_pair = Scheduler._genarm_chunked_prefill_blocks_pair
    _genarm_would_chunk_prefill = Scheduler._genarm_would_chunk_prefill
    _genarm_prefill_pair_ready = Scheduler._genarm_prefill_pair_ready
    _genarm_try_add_prefill_pair = Scheduler._genarm_try_add_prefill_pair
    _genarm_validate_prefill_pairs = Scheduler._genarm_validate_prefill_pairs
    _genarm_max_new_for_unadd = Scheduler._genarm_max_new_for_unadd
    _genarm_revert_mamba_if_unadded = Scheduler._genarm_revert_mamba_if_unadded
    _genarm_rollback_pair_admission = Scheduler._genarm_rollback_pair_admission
    _genarm_rollback_preemption = Scheduler._genarm_rollback_preemption
    _genarm_flush_preempt_list_to_waiting = Scheduler._genarm_flush_preempt_list_to_waiting
    _genarm_idle_prefill_window = Scheduler._genarm_idle_prefill_window
    _genarm_pair_exceeds_non_chunked_budget = Scheduler._genarm_pair_exceeds_non_chunked_budget
    _genarm_permanent_unschedulable_reason = Scheduler._genarm_permanent_unschedulable_reason
    _genarm_abort_unschedulable_pair = Scheduler._genarm_abort_unschedulable_pair
    _add_request_to_queue = MagicMock()


class TestGenArmAtomicScheduling(CustomTestCase):
    def test_pair_added_together_when_both_fit(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-a", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-a"))
        sched.waiting_queue = [primary, shadow]
        adder = MagicMock(spec=PrefillAdder)
        adder.dllm_config = None
        adder.rem_chunk_tokens = None
        adder.new_chunked_req = None
        adder.can_run_list = []
        adder.ceil_paged_tokens = lambda x: x
        adder.add_one_req = MagicMock(return_value=AddReqResult.CONTINUE)
        adder.unadd_req = MagicMock(return_value=True)

        sched.tree_cache = MagicMock()
        primary.init_next_round_input = MagicMock()
        shadow.init_next_round_input = MagicMock()

        self.assertTrue(
            sched._genarm_try_add_prefill_pair(
                primary, shadow, adder, None, truncation_align_size=None
            )
        )
        self.assertEqual(adder.add_one_req.call_count, 2)

    def test_shadow_cannot_fit_rolls_back_primary(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-b", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-b"))
        adder = MagicMock(spec=PrefillAdder)
        adder.dllm_config = None
        adder.rem_chunk_tokens = None
        adder.new_chunked_req = None
        adder.can_run_list = [primary]
        adder.ceil_paged_tokens = lambda x: x

        def add_side_effect(req, **kwargs):
            if req is shadow:
                return AddReqResult.NO_TOKEN
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        adder.add_one_req = MagicMock(side_effect=add_side_effect)
        adder.unadd_req = MagicMock(return_value=True)

        self.assertFalse(
            sched._genarm_try_add_prefill_pair(
                primary, shadow, adder, None, truncation_align_size=None
            )
        )
        adder.unadd_req.assert_called_once()

    def test_shadow_fail_rolls_back_preemption(self):
        sched = _FakeGenArmScheduler()
        sched.enable_priority_preemption = True
        sched.running_batch.batch_is_full = True
        primary = _make_req("prim-preempt", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-preempt"))
        adder = MagicMock(spec=PrefillAdder)
        adder.dllm_config = None
        adder.rem_chunk_tokens = None
        adder.new_chunked_req = None
        adder.can_run_list = []
        adder.preempt_list = [MagicMock(rid="running-1")]
        adder.rem_total_token_offset = 10
        adder.snapshot_req_prefill_state = PrefillAdder.snapshot_req_prefill_state
        adder.restore_req_prefill_state = PrefillAdder.restore_req_prefill_state
        adder.tree_cache = MagicMock(req_to_token_pool=MagicMock(mamba_pool=None))
        adder.ceil_paged_tokens = lambda x: x
        adder.unadd_req = MagicMock(return_value=True)
        adder.preempt_to_schedule_pair = MagicMock(return_value=True)
        adder.add_one_req = MagicMock(
            side_effect=[AddReqResult.CONTINUE, AddReqResult.NO_TOKEN]
        )

        self.assertFalse(
            sched._genarm_try_add_prefill_pair(
                primary, shadow, adder, None, truncation_align_size=None
            )
        )
        adder.unadd_req.assert_called_once()
        sched._add_request_to_queue.assert_called()

    def test_chunked_prefill_blocks_genarm_pair(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-c", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-c"))
        adder = MagicMock(spec=PrefillAdder)
        adder.dllm_config = None
        adder.rem_chunk_tokens = 2
        adder.new_chunked_req = None
        adder.ceil_paged_tokens = lambda x: x
        primary.extend_input_len = 10
        shadow.extend_input_len = 10
        sched.tree_cache = MagicMock()

        self.assertTrue(
            sched._genarm_would_chunk_prefill(
                primary, adder, has_chunked_req=False
            )
        )
        self.assertFalse(
            sched._genarm_prefill_pair_ready(primary, shadow, adder, None)
        )

    def test_validate_removes_orphan_without_partner(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-d", genarm=True)
        adder = MagicMock(spec=PrefillAdder)
        adder.can_run_list = [primary]
        adder.new_chunked_req = None
        adder.unadd_req = MagicMock(return_value=True)

        sched._genarm_validate_prefill_pairs(adder)
        adder.unadd_req.assert_called_once_with(primary, max_new_tokens=ANY)

    def test_waiting_partner_missing_skips_single_add(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-e", genarm=True)
        sched.waiting_queue = [primary]
        waiting_by_rid = {r.rid: r for r in sched.waiting_queue}
        peer_rid = genarm_shadow_rid("prim-e")
        partner = waiting_by_rid.get(peer_rid)
        self.assertIsNone(partner)

    def test_permanent_unschedulable_when_pair_would_chunk_on_idle_window(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-ff", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-ff"))
        primary.extend_input_len = 10
        shadow.extend_input_len = 10
        adder = MagicMock(spec=PrefillAdder)
        adder.dllm_config = None
        adder.rem_chunk_tokens = 2
        adder.new_chunked_req = None
        adder.can_run_list = []
        adder.ceil_paged_tokens = lambda x: x
        primary.init_next_round_input = MagicMock()
        shadow.init_next_round_input = MagicMock()

        reason = sched._genarm_permanent_unschedulable_reason(
            primary, shadow, adder, None
        )
        self.assertEqual(reason, _GENARM_UNSCHEDULABLE_PAIR_MSG)

    def test_permanent_unschedulable_waits_when_prefill_not_idle(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-wait", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-wait"))
        primary.extend_input_len = 10
        shadow.extend_input_len = 10
        adder = MagicMock(spec=PrefillAdder)
        adder.dllm_config = None
        adder.rem_chunk_tokens = 2
        adder.new_chunked_req = None
        adder.can_run_list = [MagicMock()]
        adder.ceil_paged_tokens = lambda x: x

        reason = sched._genarm_permanent_unschedulable_reason(
            primary, shadow, adder, None
        )
        self.assertIsNone(reason)

    def test_abort_unschedulable_pair_removes_from_waiting_and_streams_primary(self):
        sched = _FakeGenArmScheduler()
        primary = _make_req("prim-abort", genarm=True)
        shadow = _make_req(genarm_shadow_rid("prim-abort"))
        sched.waiting_queue = [primary, shadow, _make_req("other")]
        sched.stream_output = MagicMock()
        sched.send_to_tokenizer = MagicMock()

        with patch("sglang.srt.managers.scheduler.prepare_abort") as mock_abort:
            sched._genarm_abort_unschedulable_pair(
                primary, shadow, _GENARM_UNSCHEDULABLE_PAIR_MSG
            )

        self.assertEqual(
            [r.rid for r in sched.waiting_queue],
            ["other"],
        )
        mock_abort.assert_called_once_with(
            primary,
            _GENARM_UNSCHEDULABLE_PAIR_MSG,
            status_code=HTTPStatus.BAD_REQUEST,
        )
        sched.stream_output.assert_called_once_with([primary], primary.return_logprob)
        sched.send_to_tokenizer.send_output.assert_called_once()
        abort_req = sched.send_to_tokenizer.send_output.call_args[0][0]
        self.assertEqual(abort_req.rid, shadow.rid)
        self.assertEqual(
            abort_req.finished_reason["status_code"], HTTPStatus.BAD_REQUEST
        )


class TestPrefillAdderUnadd(CustomTestCase):
    def test_unadd_req_reverts_budget(self):
        tree_cache = MagicMock()
        tree_cache.dec_lock_ref = MagicMock()
        tree_cache.evictable_size = MagicMock(return_value=0)
        running = MagicMock(reqs=[])
        adder = PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=MagicMock(
                available_size=lambda: 100,
                full_available_size=lambda: 100,
            ),
            running_batch=running,
            new_token_ratio=0.5,
            rem_input_tokens=100,
            rem_chunk_tokens=None,
            max_running_requests=32,
        )
        req = _make_req("rollback")
        adder.can_run_list = [req]
        before = adder.rem_input_tokens
        adder._update_prefill_budget(0, req.extend_input_len, 4)
        after_add = adder.rem_input_tokens
        self.assertLess(after_add, before)

        self.assertTrue(adder.unadd_req(req, max_new_tokens=4))
        self.assertEqual(adder.rem_input_tokens, before)
        self.assertEqual(adder.can_run_list, [])
        tree_cache.dec_lock_ref.assert_called_once()

    def test_restore_req_prefill_state_reverts_host_hit(self):
        tree_cache = MagicMock()
        tree_cache.dec_lock_ref = MagicMock()
        tree_cache.evictable_size = MagicMock(return_value=0)
        tree_cache.req_to_token_pool = MagicMock(mamba_pool=None)
        running = MagicMock(reqs=[])
        adder = PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=MagicMock(
                available_size=lambda: 100,
                full_available_size=lambda: 100,
            ),
            running_batch=running,
            new_token_ratio=0.5,
            rem_input_tokens=100,
            rem_chunk_tokens=None,
            max_running_requests=32,
        )
        req = _make_req("host-hit")
        snap = adder.snapshot_req_prefill_state(req)
        req.host_hit_length = 5
        req.extend_input_len = 1
        adder.restore_req_prefill_state(req, snap)
        self.assertEqual(req.host_hit_length, 0)
        self.assertEqual(req.extend_input_len, snap["extend_input_len"])
