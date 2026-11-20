"""Opt-in experiment: separate model and H2D workqueue configurations.

CUDA's balanced policy is a scheduling hint, not a concurrency guarantee.
Contexts share all SMs unless SGLANG_EXPERIMENTAL_H2D_SM_COUNT is positive.
In that case model and H2D use complementary resources from one SM split.
Requires CUDA driver support for WORKQUEUE_CONFIG (13.1+) and cuda.bindings.
"""

import logging
import os
from functools import cache

import torch

logger = logging.getLogger(__name__)
_SWITCH = "SGLANG_EXPERIMENTAL_WORKQUEUE_ISOLATION"
_H2D_SM_COUNT = "SGLANG_EXPERIMENTAL_H2D_SM_COUNT"


def enabled() -> bool:
    return os.environ.get(_SWITCH) == "1"


def _checked(call):
    error, *values = call
    if int(error):
        raise RuntimeError(f"CUDA workqueue experiment failed: {error}")
    return values[0] if len(values) == 1 else tuple(values)


@cache
def _sm_resources(pid: int, device_index: int, requested: int):
    from cuda.bindings import driver as drv

    device = _checked(drv.cuDeviceGet(device_index))
    whole = _checked(
        drv.cuDeviceGetDevResource(
            device, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    if requested <= 0 or requested >= whole.sm.smCount:
        raise ValueError(f"Invalid H2D SM count: {requested}/{whole.sm.smCount}")
    # Use default co-scheduling alignment so model thread-block clusters remain
    # supported. The driver may round the requested H2D partition up.
    groups, count, remainder = _checked(
        drv.cuDevSmResourceSplitByCount(1, whole, 0, requested)
    )
    if count != 1 or remainder.sm.smCount <= 0:
        raise RuntimeError("Failed to split model and H2D SM resources")
    copy = groups[0]
    if copy.sm.smCount + remainder.sm.smCount != whole.sm.smCount:
        raise RuntimeError("SM split does not cover the complete device")
    return {"h2d": copy, "model": remainder}


def model_sm_count(device_index: int) -> int | None:
    """Return the SM budget of the same model Green Context used for forward."""
    if not enabled():
        return None
    return _context(os.getpid(), "model", device_index).sm_count


class _Context:
    """Keep one Green Context per device and role for the worker's lifetime."""

    def __init__(self, role: str, device_index: int):
        from cuda.bindings import driver as drv

        self.drv = drv
        self.device_index = device_index
        _checked(drv.cuInit(0))
        device = _checked(drv.cuDeviceGet(device_index))
        query = drv.cuDeviceGetDevResource(
            device, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
        )
        if int(query[0]):
            raise RuntimeError(
                f"{_SWITCH}=1 requires CUDA WORKQUEUE_CONFIG support; "
                f"device {device_index} returned {query[0]}. "
                "Use a compatible CUDA 13.1+ driver. Isolation was not enabled."
            )
        resource = query[1]
        available = resource.wqConfig.wqConcurrencyLimit
        if available < 2:
            raise RuntimeError("Workqueue experiment requires at least two connections")
        resource.wqConfig.sharingScope = (
            drv.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
        )
        resource.wqConfig.wqConcurrencyLimit = 1 if role == "h2d" else available - 1
        requested_sms = int(os.environ.get(_H2D_SM_COUNT, "0"))
        resources = [resource]
        if requested_sms:
            resources.insert(
                0, _sm_resources(os.getpid(), device_index, requested_sms)[role]
            )
        descriptor = _checked(drv.cuDevResourceGenerateDesc(resources, len(resources)))
        self.handle = _checked(
            drv.cuGreenCtxCreate(
                descriptor,
                device,
                drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
        )
        try:
            _checked(drv.cuCtxFromGreenCtx(self.handle))
            assigned = _checked(
                drv.cuGreenCtxGetDevResource(
                    self.handle, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
                )
            )
            self.sm_count = int(assigned.sm.smCount)
            if self.sm_count <= 0:
                raise RuntimeError("CUDA returned an empty Green Context SM budget")
        except Exception:
            drv.cuGreenCtxDestroy(self.handle)
            raise
        logger.info(
            "CUDA workqueue context: role=%s device=%s workqueues=%s sm_count=%s",
            role,
            device_index,
            resource.wqConfig.wqConcurrencyLimit,
            self.sm_count,
        )

    def stream(self):
        handle = _checked(
            self.drv.cuGreenCtxStreamCreate(
                self.handle, self.drv.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
            )
        )
        return torch.cuda.ExternalStream(int(handle), self.device_index)


@cache
def _context(pid: int, role: str, device_index: int):
    return _Context(role, device_index)


def new_stream(device_module, role: str):
    if not enabled():
        return device_module.Stream()
    if device_module is not torch.cuda:
        raise RuntimeError(f"{_SWITCH} requires NVIDIA CUDA")
    return _context(os.getpid(), role, torch.cuda.current_device()).stream()
