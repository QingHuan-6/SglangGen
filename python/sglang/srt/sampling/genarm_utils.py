# GenARM (log-space base + alpha * arm) helpers for SGLang fork integration.
# Mirrors OpenRLHF/examples/genarm_guided_decode.py combine_logprobs_genarm and
# vLLM fork vllm/v1/sample/genarm_utils.combine_genarm_logits_rows.
#
# Pairing convention: shadow request_id == primary_id + GENARM_ARM_SUFFIX

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

# Internal shadow request id suffix (must not collide with user ids).
GENARM_ARM_SUFFIX = "__sgl_genarm_arm__"


class GenArmMissingPrimaryInBatchError(RuntimeError):
    """Shadow GenARM rid is present in the forward batch without its paired primary row."""

    def __init__(self, message: str, *, prim_rid: str, shadow_rid: str):
        super().__init__(message)
        self.prim_rid = prim_rid
        self.shadow_rid = shadow_rid


# sampling_params.custom_params keys (API-stable for users)
GENARM_ENABLED_KEY = "genarm_enabled"
GENARM_ALPHA_KEY = "genarm_alpha"
GENARM_TEMPERATURE_SCALE_KEY = "genarm_temperature_scale"
GENARM_ARM_LORA_PATH_KEY = "genarm_arm_lora_path"

# End-to-end fusion validation: set ``SGLANG_GENARM_E2E_DUMP_DIR=/path`` to write compressed
# NumPy archives with base/arm raw logits + fused log-prob row (fp32) on each fusion step.
GENARM_E2E_DUMP_DIR_ENV = "SGLANG_GENARM_E2E_DUMP_DIR"
GENARM_E2E_DUMP_MAX_ENV = "SGLANG_GENARM_E2E_DUMP_MAX"

_GENARM_E2E_DUMP_LOCK = threading.Lock()
_GENARM_E2E_DUMP_TOTAL = [0]


def _sanitize_rid_for_filename(prim_rid: str) -> str:
    out = []
    for c in prim_rid:
        if c.isalnum() or c in "._-":
            out.append(c)
        else:
            out.append("_")
    s = "".join(out)
    return s[:220] if len(s) > 220 else s


def maybe_dump_genarm_e2e(
    prim_rid: str,
    alpha: float,
    logits_base_row: torch.Tensor,
    logits_arm_row: torch.Tensor,
    fused_logp_row: torch.Tensor,
) -> None:
    """If ``SGLANG_GENARM_E2E_DUMP_DIR`` is set, save one ``.npz`` per fusion (bounded by ``SGLANG_GENARM_E2E_DUMP_MAX``)."""
    dump_root = os.environ.get(GENARM_E2E_DUMP_DIR_ENV, "").strip()
    if not dump_root:
        return
    max_raw = os.environ.get(GENARM_E2E_DUMP_MAX_ENV, "").strip()
    max_n: Optional[int] = None
    if max_raw:
        try:
            max_n = int(max_raw)
            if max_n <= 0:
                return
        except ValueError:
            max_n = None

    with _GENARM_E2E_DUMP_LOCK:
        if max_n is not None and _GENARM_E2E_DUMP_TOTAL[0] >= max_n:
            return
        _GENARM_E2E_DUMP_TOTAL[0] += 1
        seq = _GENARM_E2E_DUMP_TOTAL[0]

    root = Path(dump_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    import numpy as np

    fname = f"{_sanitize_rid_for_filename(prim_rid)}_{seq:06d}.npz"
    path = root / fname
    lb = logits_base_row.detach().float().cpu().numpy()
    la = logits_arm_row.detach().float().cpu().numpy()
    lf = fused_logp_row.detach().float().cpu().numpy()
    np.savez_compressed(
        path,
        logits_base=lb,
        logits_arm=la,
        fused_logp=lf,
        alpha=np.float32(float(alpha)),
        prim_rid=np.array(prim_rid, dtype=object),
        seq=np.int32(seq),
    )


def is_genarm_shadow_request_id(request_id: str) -> bool:
    return request_id.endswith(GENARM_ARM_SUFFIX)


def primary_id_from_genarm_shadow(request_id: str) -> str:
    assert is_genarm_shadow_request_id(request_id)
    return request_id[: -len(GENARM_ARM_SUFFIX)]


def genarm_shadow_rid(primary_rid: str) -> str:
    return f"{primary_rid}{GENARM_ARM_SUFFIX}"


def genarm_peer_rid(*, rid: str, custom_params: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return paired GenARM rid (primary <-> shadow), or None."""
    if is_genarm_shadow_request_id(rid):
        return primary_id_from_genarm_shadow(rid)
    if isinstance(custom_params, dict) and custom_params.get(GENARM_ENABLED_KEY):
        return genarm_shadow_rid(rid)
    return None


def strip_genarm_custom_params(custom_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove genarm_* keys for shadow-request SamplingParams."""
    if not isinstance(custom_params, dict):
        return custom_params
    return {
        k: v
        for k, v in custom_params.items()
        if not str(k).startswith("genarm_")
    }


def genarm_alpha_from_custom_params(custom_params: Optional[Dict[str, Any]]) -> float:
    if not isinstance(custom_params, dict):
        return 1.0
    try:
        return float(custom_params.get(GENARM_ALPHA_KEY, 1.0))
    except (TypeError, ValueError):
        return 1.0


def combine_genarm_logits_rows(
    logits_base: torch.Tensor,
    logits_arm: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Fuse last-step logits (same shape [vocab]) in log-softmax space.

    logits_* : 1D [vocab] or same trailing dim as from lm_head.
    Returns **log-prob** row (same convention as HF guided_decode / vLLM GenARM fork).

    Formula:
      s = log_softmax(Lb) + alpha * log_softmax(La)
      s /= (1 + alpha)
      out = s - logsumexp(s)
    """
    lb = logits_base.to(dtype=torch.float32).unsqueeze(0)
    la = logits_arm.to(dtype=torch.float32).unsqueeze(0)
    log_p_b = F.log_softmax(lb, dim=-1)
    log_p_a = F.log_softmax(la, dim=-1)
    s = log_p_b + float(alpha) * log_p_a
    s = s / (1.0 + float(alpha))
    out = s - torch.logsumexp(s, dim=-1, keepdim=True)
    return out.squeeze(0)
