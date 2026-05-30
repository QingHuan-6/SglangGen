"""Unit tests for GenARM pairing-failure fallbacks (HTTP defer, KV sweep, sampling)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import asyncio
import unittest
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.genarm_utils import (
    GENARM_ARM_SUFFIX,
    GENARM_ARM_LORA_PATH_KEY,
    GENARM_ALPHA_KEY,
    GENARM_ENABLED_KEY,
    GenArmMissingPrimaryInBatchError,
    combine_genarm_logits_rows,
    genarm_shadow_rid,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


def _make_req(rid: str, *, req_pool_idx: int = 7) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="hi",
        origin_input_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    req.req_pool_idx = req_pool_idx
    return req


class _FakeScheduler(SchedulerOutputProcessorMixin):
    def __init__(self):
        self.waiting_queue = []
        self.running_batch = None
        self.chunked_req = None
        self.last_batch = None
        self.cur_batch = None
        self.disagg_prefill_inflight_queue = []
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None
        self.released_rids = []
        self.server_args = MagicMock(disaggregation_decode_enable_offload_kvcache=False)
        self.enable_hisparse = False
        self.tree_cache = MagicMock()

    def maybe_collect_routed_experts(self, req):
        return None

    def maybe_collect_indexer_topk(self, req):
        return None

    def _release_req_kv_if_allocated(self, req):
        if req.req_pool_idx is None:
            return
        self.released_rids.append(req.rid)
        req.req_pool_idx = None


class _FakeTokenizer:
    """Minimal host for TokenizerManager GenARM HTTP helpers."""

    def __init__(self):
        self.rid_to_state = {}
        self.genarm_http_settled_prim_rids = set()
        self.server_args = MagicMock(
            weight_version="test",
            enable_lora=False,
            speculative_algorithm=None,
        )
        self.genarm_shadow_meta = {}
        self.enable_metrics = False
        self.dump_requests_folder = None
        self.crash_dump_folder = None
        self.dump_requests_exclude_meta_keys = None
        self.dump_requests_threshold = 1
        self.dump_request_list = []
        self.crash_dump_request_list = []
        self.crash_dump_performed = False
        self.enable_lora = False

    _genarm_try_finish_primary_http = TokenizerManager._genarm_try_finish_primary_http
    _genarm_apply_primary_http_terminal = (
        TokenizerManager._genarm_apply_primary_http_terminal
    )
    _mark_genarm_primary_http_settled = TokenizerManager._mark_genarm_primary_http_settled
    _maybe_build_genarm_shadow_tokenized = (
        TokenizerManager._maybe_build_genarm_shadow_tokenized
    )
    _dispatch_genarm_tokenized = TokenizerManager._dispatch_genarm_tokenized


def _primary_state(*, await_shadow: bool = True) -> ReqState:
    obj = MagicMock(stream=False)
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=MagicMock(),
        genarm_await_shadow_http=await_shadow,
    )


def _deferred_out(prim_rid: str) -> dict:
    return {
        "text": "hello",
        "output_ids": [9],
        "meta_info": {
            "id": prim_rid,
            "finish_reason": {"type": "stop"},
            "prompt_tokens": 1,
            "reasoning_tokens": 0,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "weight_version": "test",
            "num_retractions": 0,
        },
    }


class TestGenArmTokenizerShadowFirst(CustomTestCase):
    def test_shadow_first_does_not_wake_http_waiter(self):
        tok = _FakeTokenizer()
        prim_rid = "req-1"
        state = _primary_state()
        tok.rid_to_state[prim_rid] = state
        shadow_fr = {"type": "stop", "matched": None}

        finished = tok._genarm_try_finish_primary_http(
            prim_rid=prim_rid,
            prim_state=state,
            pending_notify={},
            shadow_fr=shadow_fr,
        )

        self.assertFalse(finished)
        self.assertEqual(state.genarm_pending_shadow_finish_reason, shadow_fr)
        self.assertIsNone(state.genarm_deferred_terminal_out)
        self.assertEqual(state.out_list, [])
        self.assertFalse(state.finished)
        self.assertFalse(state.event.is_set())

    def test_shadow_first_then_primary_emits_single_http_result(self):
        tok = _FakeTokenizer()
        prim_rid = "req-2"
        state = _primary_state()
        tok.rid_to_state[prim_rid] = state
        shadow_fr = {"type": "stop", "matched": None}

        tok._genarm_try_finish_primary_http(
            prim_rid=prim_rid,
            prim_state=state,
            pending_notify={},
            shadow_fr=shadow_fr,
        )
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)
        pending_notify = {}
        finished = tok._genarm_try_finish_primary_http(
            prim_rid=prim_rid,
            prim_state=state,
            pending_notify=pending_notify,
        )

        self.assertTrue(finished)
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(state.out_list[0]["text"], "hello")
        self.assertNotIn(prim_rid, tok.rid_to_state)
        self.assertIn(prim_rid, pending_notify)

    def test_primary_first_does_not_finish_before_shadow(self):
        tok = _FakeTokenizer()
        prim_rid = "req-primary-first"
        state = _primary_state()
        tok.rid_to_state[prim_rid] = state
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)

        finished = tok._genarm_try_finish_primary_http(
            prim_rid=prim_rid,
            prim_state=state,
            pending_notify={},
        )

        self.assertFalse(finished)
        self.assertIsNotNone(state.genarm_deferred_terminal_out)
        self.assertEqual(state.out_list, [])
        self.assertFalse(state.finished)

    def test_primary_first_then_shadow_emits_single_http_result(self):
        tok = _FakeTokenizer()
        prim_rid = "req-3"
        state = _primary_state()
        tok.rid_to_state[prim_rid] = state
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)

        self.assertFalse(
            tok._genarm_try_finish_primary_http(
                prim_rid=prim_rid,
                prim_state=state,
                pending_notify={},
            )
        )
        pending_notify = {}
        self.assertTrue(
            tok._genarm_try_finish_primary_http(
                prim_rid=prim_rid,
                prim_state=state,
                pending_notify=pending_notify,
                shadow_fr={"type": "stop", "matched": None},
            )
        )
        self.assertEqual(len(state.out_list), 1)

    def test_shadow_abort_overrides_deferred_success(self):
        tok = _FakeTokenizer()
        prim_rid = "req-4"
        state = _primary_state()
        tok.rid_to_state[prim_rid] = state
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)
        abort_fr = {
            "type": "abort",
            "message": "GenARM pairing failed",
            "status_code": HTTPStatus.BAD_REQUEST,
            "err_type": "BadRequestError",
        }

        tok._genarm_try_finish_primary_http(
            prim_rid=prim_rid,
            prim_state=state,
            pending_notify={},
            shadow_fr=abort_fr,
        )

        self.assertEqual(len(state.out_list), 1)
        out = state.out_list[0]
        self.assertEqual(out["text"], "")
        self.assertEqual(out["output_ids"], [])
        self.assertEqual(out["meta_info"]["finish_reason"], abort_fr)


    def test_both_ready_does_not_double_finalize(self):
        tok = _FakeTokenizer()
        prim_rid = "req-no-double"
        state = _primary_state()
        tok.rid_to_state[prim_rid] = state
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)
        pending_notify = {}

        self.assertTrue(
            tok._genarm_try_finish_primary_http(
                prim_rid=prim_rid,
                prim_state=state,
                pending_notify=pending_notify,
                shadow_fr={"type": "stop", "matched": None},
            )
        )
        self.assertNotIn(prim_rid, tok.rid_to_state)
        self.assertEqual(len(state.out_list), 1)

        # Simulate outer loop guard: must not treat as still live in rid_to_state.
        self.assertNotIn(prim_rid, tok.rid_to_state)


class TestGenArmSchedulerAbortSweep(CustomTestCase):
    def test_primary_in_waiting_queue_is_released(self):
        sched = _FakeScheduler()
        prim_rid = "prim-wait"
        shadow_rid = prim_rid + GENARM_ARM_SUFFIX
        primary = _make_req(prim_rid, req_pool_idx=11)
        shadow = _make_req(shadow_rid, req_pool_idx=12)
        sched.waiting_queue = [primary]
        batch = ScheduleBatch(reqs=[shadow])

        exc = GenArmMissingPrimaryInBatchError(
            "missing primary",
            prim_rid=prim_rid,
            shadow_rid=shadow_rid,
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            sched._promote_genarm_pairing_abort_requests(batch, exc)

        self.assertTrue(primary.finished())
        self.assertTrue(shadow.finished())
        # Primary is off-batch; shadow stays in batch for normal finished post-processing.
        self.assertEqual(sched.released_rids, [prim_rid])
        self.assertIsNone(primary.req_pool_idx)
        self.assertEqual(shadow.req_pool_idx, 12)
        self.assertEqual(sched.waiting_queue, [])

    def test_shadow_in_batch_defers_kv_release_to_finished_path(self):
        sched = _FakeScheduler()
        prim_rid = "prim-batch"
        shadow_rid = prim_rid + GENARM_ARM_SUFFIX
        shadow = _make_req(shadow_rid, req_pool_idx=12)
        batch = ScheduleBatch(reqs=[shadow])

        exc = GenArmMissingPrimaryInBatchError(
            "missing primary",
            prim_rid=prim_rid,
            shadow_rid=shadow_rid,
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            sched._promote_genarm_pairing_abort_requests(batch, exc)

        self.assertTrue(shadow.finished())
        self.assertEqual(sched.released_rids, [])
        self.assertEqual(shadow.req_pool_idx, 12)

    def test_handle_finished_req_without_kv_is_safe(self):
        sched = _FakeScheduler()
        req = _make_req("no-kv-abort", req_pool_idx=None)
        with patch(
            "sglang.srt.managers.schedule_batch.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            req.set_finish_with_abort("pairing failed")
        req.check_finished()

        sched._handle_finished_req(req, 0, MagicMock(customized_info=None))

        self.assertEqual(sched.released_rids, [])
        self.assertIsNone(req.req_pool_idx)

    def test_primary_in_chunked_req_is_released(self):
        sched = _FakeScheduler()
        prim_rid = "prim-chunk"
        shadow_rid = prim_rid + GENARM_ARM_SUFFIX
        primary = _make_req(prim_rid, req_pool_idx=21)
        shadow = _make_req(shadow_rid, req_pool_idx=22)
        sched.chunked_req = primary
        batch = ScheduleBatch(reqs=[shadow])

        exc = GenArmMissingPrimaryInBatchError(
            "missing primary",
            prim_rid=prim_rid,
            shadow_rid=shadow_rid,
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            sched._promote_genarm_pairing_abort_requests(batch, exc)

        self.assertEqual(set(sched.released_rids), {prim_rid})
        self.assertIsNone(primary.req_pool_idx)
        self.assertEqual(shadow.req_pool_idx, 22)


class TestGenArmTpWorkerPairingFailure(CustomTestCase):
    def test_unrelated_samples_are_not_zeroed(self):
        worker = TpModelWorker.__new__(TpModelWorker)
        prim_rid = "prim-batch"
        shadow_rid = prim_rid + GENARM_ARM_SUFFIX
        other = _make_req("other-req")
        shadow = _make_req(shadow_rid)
        model_worker_batch = MagicMock(
            reqs=[other, shadow],
            seq_lens=[1, 1],
        )
        exc = GenArmMissingPrimaryInBatchError(
            "missing primary",
            prim_rid=prim_rid,
            shadow_rid=shadow_rid,
        )
        forward_batch = MagicMock(genarm_pairing_exc=exc)
        batch_result = GenerationBatchResult(
            next_token_ids=torch.tensor([42, 99], dtype=torch.long)
        )

        with patch(
            "sglang.srt.managers.schedule_batch.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            worker._note_genarm_pairing_failure(
                batch_result, model_worker_batch, forward_batch
            )

        self.assertTrue(torch.equal(batch_result.next_token_ids, torch.tensor([42, 99])))
        self.assertIs(batch_result.genarm_pairing_exc, exc)
        self.assertTrue(shadow.finished())
        self.assertFalse(other.finished())


class TestGenArmFusionMissingPrimary(CustomTestCase):
    def test_fusion_returns_exc_and_continues_sampling(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = ModelRunner.__new__(ModelRunner)
        runner.sampler = MagicMock(return_value=torch.tensor([5], dtype=torch.long))
        runner._preprocess_logits = MagicMock()
        runner.maybe_update_ngram_token_table = MagicMock()

        prim_rid = "prim-fuse"
        shadow_rid = prim_rid + GENARM_ARM_SUFFIX
        logits = torch.zeros(1, 4)
        logits_output = LogitsProcessorOutput(next_token_logits=logits)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.zeros(1, dtype=torch.long),
            seq_lens=torch.ones(1, dtype=torch.long),
            out_cache_loc=torch.zeros(1, dtype=torch.long),
            seq_lens_sum=1,
            rids=[shadow_rid],
            sampling_info=MagicMock(
                custom_params=[{GENARM_ENABLED_KEY: True}],
            ),
        )

        next_ids = runner.sample(logits_output, forward_batch)

        self.assertEqual(next_ids.item(), 5)
        self.assertIsInstance(forward_batch.genarm_pairing_exc, GenArmMissingPrimaryInBatchError)
        self.assertEqual(forward_batch.genarm_pairing_exc.prim_rid, prim_rid)
        runner.sampler.assert_called_once()


class TestGenArmFusionMath(CustomTestCase):
    def test_fusion_matches_formula_and_shifts_toward_arm(self):
        logits_base = torch.tensor([6.0, 1.0, 0.5], dtype=torch.float32)
        logits_arm = torch.tensor([0.5, 5.5, 1.0], dtype=torch.float32)
        alpha = 1.5

        fused = combine_genarm_logits_rows(logits_base, logits_arm, alpha)

        log_p_b = torch.log_softmax(logits_base.unsqueeze(0), dim=-1)
        log_p_a = torch.log_softmax(logits_arm.unsqueeze(0), dim=-1)
        expected = (
            (log_p_b + alpha * log_p_a) / (1.0 + alpha)
        )
        expected = expected - torch.logsumexp(expected, dim=-1, keepdim=True)
        expected = expected.squeeze(0)

        self.assertTrue(torch.allclose(fused, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(int(fused.argmax().item()), 1)
        self.assertEqual(int(logits_base.argmax().item()), 0)
        self.assertEqual(int(logits_arm.argmax().item()), 1)
        self.assertAlmostEqual(float(torch.logsumexp(fused, dim=-1).item()), 0.0, places=6)


class TestGenArmAbortRace(CustomTestCase):
    def test_genarm_tokenization_creates_shadow_request(self):
        mgr = _FakeTokenizer()
        prim_rid = "prim-tokenized"
        mgr.lora_registry = MagicMock()
        mgr.lora_registry.acquire = AsyncMock(return_value="arm-lora-0")

        request = GenerateReqInput(
            rid=prim_rid,
            text="hello",
            sampling_params={
                "custom_params": {
                    GENARM_ENABLED_KEY: True,
                    GENARM_ALPHA_KEY: 0.7,
                    GENARM_ARM_LORA_PATH_KEY: "/tmp/arm.lora",
                    "foo": "bar",
                }
            },
        )
        primary = TokenizedGenerateReqInput(
            rid=prim_rid,
            input_text="hello",
            input_ids=[1, 2, 3],
            mm_inputs=None,
            sampling_params=SamplingParams(
                max_new_tokens=4,
                custom_params={
                    GENARM_ENABLED_KEY: True,
                    GENARM_ALPHA_KEY: 0.7,
                    GENARM_ARM_LORA_PATH_KEY: "/tmp/arm.lora",
                    "foo": "bar",
                },
            ),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
        )

        shadow = asyncio.run(
            mgr._maybe_build_genarm_shadow_tokenized(request, primary)
        )

        self.assertIsNotNone(shadow)
        self.assertEqual(shadow.rid, genarm_shadow_rid(prim_rid))
        self.assertEqual(shadow.lora_id, "arm-lora-0")
        self.assertEqual(shadow.sampling_params.custom_params, {"foo": "bar"})
        mgr.lora_registry.acquire.assert_called_once_with("/tmp/arm.lora")

    def test_genarm_dispatch_registers_shadow_and_marks_primary(self):
        mgr = _FakeTokenizer()
        prim_rid = "prim-dispatch"
        mgr.lora_registry = MagicMock()
        mgr.lora_registry.acquire = AsyncMock(return_value="arm-lora-1")
        mgr._send_batch_request = MagicMock()
        mgr.rid_to_state = {prim_rid: _primary_state()}

        request = GenerateReqInput(
            rid=prim_rid,
            text="hello",
            sampling_params={
                "custom_params": {
                    GENARM_ENABLED_KEY: True,
                    GENARM_ARM_LORA_PATH_KEY: "/tmp/arm.lora",
                }
            },
        )
        primary = TokenizedGenerateReqInput(
            rid=prim_rid,
            input_text="hello",
            input_ids=[1, 2, 3],
            mm_inputs=None,
            sampling_params=SamplingParams(
                max_new_tokens=4,
                custom_params={
                    GENARM_ENABLED_KEY: True,
                    GENARM_ARM_LORA_PATH_KEY: "/tmp/arm.lora",
                },
            ),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
        )

        asyncio.run(mgr._dispatch_genarm_tokenized(request, primary))

        shadow_rid = genarm_shadow_rid(prim_rid)
        self.assertIn(shadow_rid, mgr.genarm_shadow_meta)
        self.assertEqual(mgr.genarm_shadow_meta[shadow_rid], (prim_rid, "arm-lora-1"))
        self.assertTrue(mgr.rid_to_state[prim_rid].genarm_await_shadow_http)
        mgr._send_batch_request.assert_called_once()
        sent = mgr._send_batch_request.call_args.args[0]
        self.assertEqual([req.rid for req in sent], [prim_rid, shadow_rid])

    def test_shadow_abort_finalizes_primary_with_deferred_terminal(self):
        mgr = TokenizerManager.__new__(TokenizerManager)
        prim_rid = "prim-unsched"
        shadow_rid = genarm_shadow_rid(prim_rid)
        state = _primary_state()
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)
        mgr.rid_to_state = {prim_rid: state}
        mgr.genarm_shadow_meta = {shadow_rid: (prim_rid, None)}
        mgr.genarm_http_settled_prim_rids = set()
        mgr.server_args = MagicMock(weight_version="test", enable_lora=False)
        mgr.dump_requests_folder = None
        mgr.crash_dump_folder = None
        mgr.dump_requests_exclude_meta_keys = None
        mgr.dump_request_list = []
        mgr.crash_dump_request_list = []
        mgr.crash_dump_performed = False
        mgr._genarm_try_finish_primary_http = (
            TokenizerManager._genarm_try_finish_primary_http.__get__(mgr)
        )
        mgr._genarm_apply_primary_http_terminal = (
            TokenizerManager._genarm_apply_primary_http_terminal.__get__(mgr)
        )
        mgr._mark_genarm_primary_http_settled = (
            TokenizerManager._mark_genarm_primary_http_settled.__get__(mgr)
        )
        mgr._genarm_take_shadow_sidecar = (
            TokenizerManager._genarm_take_shadow_sidecar.__get__(mgr)
        )
        mgr._genarm_release_shadow_lora = MagicMock()

        abort_fr = {
            "type": "abort",
            "message": "GenARM request cannot be scheduled atomically",
            "status_code": HTTPStatus.BAD_REQUEST,
            "err_type": "BadRequestError",
        }
        mgr._handle_abort_req(
            AbortReq(rid=shadow_rid, finished_reason=abort_fr)
        )

        self.assertNotIn(prim_rid, mgr.rid_to_state)
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(state.out_list[0]["meta_info"]["finish_reason"], abort_fr)
        self.assertTrue(state.event.is_set())

    def test_shadow_abort_before_primary_terminal_stores_pending_finish(self):
        mgr = TokenizerManager.__new__(TokenizerManager)
        prim_rid = "prim-pending"
        shadow_rid = genarm_shadow_rid(prim_rid)
        state = _primary_state()
        mgr.rid_to_state = {prim_rid: state}
        mgr.genarm_shadow_meta = {shadow_rid: (prim_rid, None)}
        mgr.genarm_http_settled_prim_rids = set()
        mgr.server_args = MagicMock(weight_version="test", enable_lora=False)
        mgr.dump_requests_folder = None
        mgr.crash_dump_folder = None
        mgr.dump_requests_exclude_meta_keys = None
        mgr.dump_request_list = []
        mgr.crash_dump_request_list = []
        mgr.crash_dump_performed = False
        mgr._genarm_try_finish_primary_http = (
            TokenizerManager._genarm_try_finish_primary_http.__get__(mgr)
        )
        mgr._genarm_take_shadow_sidecar = (
            TokenizerManager._genarm_take_shadow_sidecar.__get__(mgr)
        )
        mgr._genarm_release_shadow_lora = MagicMock()

        abort_fr = {
            "type": "abort",
            "message": "GenARM request cannot be scheduled atomically",
            "status_code": HTTPStatus.BAD_REQUEST,
        }
        mgr._handle_abort_req(
            AbortReq(rid=shadow_rid, finished_reason=abort_fr)
        )

        self.assertIn(prim_rid, mgr.rid_to_state)
        self.assertEqual(state.genarm_pending_shadow_finish_reason, abort_fr)
        self.assertFalse(state.event.is_set())

        pending_notify = {}
        mgr._mark_genarm_primary_http_settled = (
            TokenizerManager._mark_genarm_primary_http_settled.__get__(mgr)
        )
        mgr._genarm_apply_primary_http_terminal = (
            TokenizerManager._genarm_apply_primary_http_terminal.__get__(mgr)
        )
        state.genarm_deferred_terminal_out = _deferred_out(prim_rid)
        self.assertTrue(
            mgr._genarm_try_finish_primary_http(
                prim_rid=prim_rid,
                prim_state=state,
                pending_notify=pending_notify,
                shadow_fr=state.genarm_pending_shadow_finish_reason,
            )
        )
        self.assertNotIn(prim_rid, mgr.rid_to_state)

    def test_shadow_abort_without_rid_to_state_releases_lora(self):
        mgr = TokenizerManager.__new__(TokenizerManager)
        mgr.rid_to_state = {}
        mgr.genarm_shadow_meta = {}
        mgr.server_args = MagicMock(weight_version="test", enable_lora=True)

        shadow_rid = genarm_shadow_rid("prim-abort")
        mgr.genarm_shadow_meta[shadow_rid] = ("prim-abort", "lora-0")

        with patch.object(mgr, "_genarm_release_shadow_lora") as release_lora:
            mgr._handle_abort_req(AbortReq(rid=shadow_rid))

        self.assertNotIn(shadow_rid, mgr.genarm_shadow_meta)
        self.assertEqual(mgr.rid_to_state, {})
        release_lora.assert_called_once_with("lora-0")

    def test_shadow_abort_idempotent_when_sidecar_already_taken(self):
        mgr = TokenizerManager.__new__(TokenizerManager)
        mgr.rid_to_state = {}
        mgr.genarm_shadow_meta = {}
        mgr.server_args = MagicMock(weight_version="test", enable_lora=True)

        shadow_rid = genarm_shadow_rid("prim-abort-2")
        with patch.object(mgr, "_genarm_release_shadow_lora") as release_lora:
            mgr._handle_abort_req(AbortReq(rid=shadow_rid))

        release_lora.assert_not_called()

    def test_primary_abort_after_state_cleared_is_noop(self):
        mgr = TokenizerManager.__new__(TokenizerManager)
        mgr.rid_to_state = {}
        mgr.genarm_shadow_meta = {}
        mgr.server_args = MagicMock(weight_version="test")

        mgr._handle_abort_req(AbortReq(rid="prim-gone"))
        self.assertEqual(mgr.rid_to_state, {})
