"""Read the 78 replicated main MLA layers from TP4 host owners, with ordinary local sidecars."""

import mmap
import os
from contextlib import ExitStack, contextmanager
from copy import copy


@contextmanager
def map_owner_layers(local_pool, rank, peer_fds, register, unregister):
    import torch

    local = local_pool.kv_buffer
    if (
        local_pool.layout != "layer_first"
        or not local.is_contiguous()
        or local.storage_offset()
    ):
        raise ValueError("Expected the original contiguous layer-first pool")
    if (
        tuple(local.shape)[0] != 79
        or local.shape[-2:] != (1, 576)
        or local.element_size() != 1
    ):
        raise ValueError(
            "Expected 78 target layers and one packed draft with 576-byte rows"
        )
    layer_bytes = local.stride(0) * local.element_size()
    if layer_bytes % mmap.ALLOCATIONGRANULARITY:
        raise ValueError("Layer offsets must be mmap-aligned")

    with ExitStack() as resources:
        views = []
        for owner in range(4):
            if owner == rank:
                views.append(local_pool)
                continue
            fd = peer_fds[(owner, 0)]
            if os.fstat(fd).st_size < 79 * layer_bytes:
                raise ValueError("Peer file is smaller than the original host pool")
            layers = {}
            for layer in range(owner, 78, 4):
                mm = resources.enter_context(
                    mmap.mmap(
                        fd,
                        layer_bytes,
                        offset=layer * layer_bytes,
                        flags=mmap.MAP_SHARED,
                    )
                )
                tensor = torch.frombuffer(mm, dtype=local.dtype).view(local.shape[1:])
                register(tensor)
                resources.callback(unregister, tensor)
                layers[layer] = tensor
            # Q's layer_first AOT method takes the source from kv_buffer[layer]
            # and the destination from this rank's original device_kv_buffers.
            view = copy(local_pool)
            view.kv_buffer = layers
            views.append(view)
        yield views


@contextmanager
def mapped_tp4_owner_layers(local_pool, group, rank):
    import torch.distributed as dist
    from sglang.srt.mem_cache.pool_host.common import (
        _cuda_host_register,
        _cuda_host_unregister,
    )
    from sglang.srt.utils.cuda_vmm_utils import exchange_posix_fds

    if local_pool.fd is None:
        raise ValueError("Main KV host pool has no shareable backing FD")
    peer_fds = exchange_posix_fds(group, rank, 4, [local_pool.fd], [1, 1, 1, 1])
    with ExitStack() as resources:
        for fd in peer_fds.values():
            resources.callback(os.close, fd)
        error = None
        try:
            views = resources.enter_context(
                map_owner_layers(
                    local_pool,
                    rank,
                    peer_fds,
                    _cuda_host_register,
                    _cuda_host_unregister,
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
        statuses = [None] * 4
        dist.all_gather_object(statuses, error, group=group)
        if any(status is not None for status in statuses):
            raise RuntimeError(f"TP4 owner host registration failed: {statuses}")
        yield views


def for_owner(layer_mapper, owner):
    def owned_layer(controller_layer):
        host_layer = layer_mapper(controller_layer)
        if host_layer is not None and host_layer % 4 == owner:
            return host_layer
        return None

    return owned_layer


class OwnerPlanRejectedBeforeRead(ValueError):
    """All participating ranks rejected the plan before any peer GPU read."""


def gather_owner_rows(staged, group, device, page_size=64):
    """Gather real page IDs once for an agreed restore batch; keep token order.

    staged is an ordered sequence of (bootstrap_room, rid, padded CPU host
    indices). Divergent room order is checked before indexing any peer's slab.
    """
    import numpy as np
    import torch
    import torch.distributed as dist

    from sglang.srt.mem_cache.common import kv_to_page_indices

    keys = []
    local_pages = []
    error = None
    for room, rid, rows in staged:
        if rows.device.type != "cpu" or rows.ndim != 1 or len(rows) % page_size:
            error = "Expected the original host allocator's padded CPU rows"
            break
        keys.append((int(room), rid, len(rows) // page_size))
        local_pages.append(kv_to_page_indices(rows, page_size).astype(np.int32))
    packed_pages = None
    if local_pages and error is None:
        packed_pages = (
            local_pages[0] if len(local_pages) == 1 else np.concatenate(local_pages)
        )
    packet = (keys, packed_pages, error)
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, packet, group=group)
    if len(peers) != 4 or any(
        peer_error or peer_keys != keys or pages is None
        for peer_keys, pages, peer_error in peers
    ):
        raise OwnerPlanRejectedBeforeRead(
            "Different TP4 restore batch identity or row counts"
        )
    on_cuda = torch.device(device).type == "cuda"
    packed = torch.empty((4, packed_pages.size), dtype=torch.int32, pin_memory=on_cuda)
    for rank, (_, pages, _) in enumerate(peers):
        packed[rank].numpy()[:] = pages
    pages = packed.to(device=device, dtype=torch.int64, non_blocking=on_cuda)
    offsets = torch.arange(page_size, device=device)
    return (pages.unsqueeze(-1) * page_size + offsets).reshape(4, -1).unbind()


def main_restore_plan(
    transfers,
    owner_views,
    owner_host_rows,
    staged_device_rows,
    private_host_rows,
    private_device_rows,
):
    """Replace only the target main MLA transfer of a merged Q load.

    Inputs are prepared once per merged operation. Indexer, packed draft and
    any other sidecars already have the original row order in `transfers`.
    """
    anchor, *sidecars = transfers
    if anchor.is_draft or anchor.layer_mapper is None:
        raise ValueError("Expected a TP4 main MLA transfer")
    if (
        len(private_host_rows) != len(private_device_rows)
        or any(len(rows) != len(staged_device_rows) for rows in owner_host_rows)
        or len(private_device_rows) + len(staged_device_rows)
        != len(anchor.device_indices)
    ):
        raise ValueError("Restore rows differ from the merged main transfer")
    result = []
    if len(private_device_rows):
        result.append(
            anchor._replace(
                host_indices=private_host_rows, device_indices=private_device_rows
            )
        )
    for owner, view in enumerate(owner_views):
        result.append(
            anchor._replace(
                host_pool=view,
                host_indices=owner_host_rows[owner],
                device_indices=staged_device_rows,
                layer_mapper=for_owner(anchor.layer_mapper, owner),
            )
        )
    result.extend(sidecars)
    return result


def prepare_owner_read_plan(views, group, operations, requests_by_operation):
    """Prepare the shared sub-batch before Q records its producer CUDA event."""
    import torch

    def combine(parts):
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    staged = []
    device_rows = []
    local_host = []
    local_device = []
    for op in operations:
        req = requests_by_operation.get(op)
        if req is None:
            local_host.append(op.host_indices)
            local_device.append(op.device_indices)
        else:
            staged.append((req.bootstrap_room, req.rid, op.host_indices))
            device_rows.append(op.device_indices)
    staged_device = combine(device_rows)
    by_owner = gather_owner_rows(staged, group, staged_device.device)
    empty = staged_device[:0]
    private_host = (
        combine(local_host).to(staged_device.device, non_blocking=True)
        if local_host
        else empty
    )
    return (
        views,
        by_owner,
        staged_device,
        private_host,
        combine(local_device) if local_device else empty,
    )
