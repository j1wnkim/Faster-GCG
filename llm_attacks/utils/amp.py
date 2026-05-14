from collections import defaultdict
from typing import Dict, List

import torch
from torch import Tensor, amp


@torch.no_grad()
def amp_unscale_grad(scaler: amp.GradScaler, *params: Tensor):
    """Unscale gradients of tensors.

    The code is basically from `GradScaler.unscale_`.

    Args:
        scaler (torch.amp.GradScaler): gradient scaler.
        params (Tensor...): tensors to unscale grad.

    Returns:
        per_device_found_inf (Dict[device, Tensor]): number of infs found per device.

    Raises:
        ValueError: If any tensor has FP16 gradient.
    """
    # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
    assert scaler._scale is not None
    inv_scale = scaler._scale.double().reciprocal().float()
    found_inf = torch.full((), 0.0, dtype=torch.float32, device=scaler._scale.device)
    per_device_inv_scale = amp.grad_scaler._MultiDeviceReplicator(inv_scale)
    per_device_found_inf = amp.grad_scaler._MultiDeviceReplicator(found_inf)
    per_device_and_dtype_grads: Dict[
        torch.device, Dict[torch.dtype, List[torch.Tensor]]
    ] = defaultdict(lambda: defaultdict(list))
    for param in params:
        assert isinstance(param, torch.Tensor)
        if param.grad is None:
            continue
        if param.grad.dtype == torch.float16:
            raise ValueError("Attempting to unscale FP16 gradients.")
        if param.grad.is_sparse:
            # is_coalesced() == False means the sparse grad has values with duplicate indices.
            # coalesce() deduplicates indices and adds all values that have the same index.
            # For scaled fp16 values, there's a good chance coalescing will cause overflow,
            # so we should check the coalesced _values().
            if param.grad.dtype is torch.float16:
                param.grad = param.grad.coalesce()
            to_unscale = param.grad._values()
        else:
            to_unscale = param.grad

        per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(
            to_unscale
        )

    for device, per_dtype_grads in per_device_and_dtype_grads.items():
        for grads in per_dtype_grads.values():
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads,
                per_device_found_inf.get(device),
                per_device_inv_scale.get(device),
            )
    return per_device_found_inf._per_device_tensors
