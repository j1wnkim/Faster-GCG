import torch
import torch.distributed as dist

from .average_meter import AverageMeter


def reduce_average_meters(
    meters: list[AverageMeter], dst=-1, inplace=False, device=None
):
    """Reduce average meters across processes.

    Args:
        meters (list[AverageMeter]):
            List of meters to be reduced.
        dst (int, optional):
            Destination process rank. -1 for all reduce. Default: -1.
        inplace (bool, optional):
            True if modify source meters. Default: False.
        device (torch.device | None, optional):
            Device to create the target tensor. Default: None.

    Returns:
        meters (list[AverageMeter]):
            New meters that have reduced result.
    """
    sum_values = []
    count_values = []
    result_meters: list[AverageMeter] = []
    for m in meters:
        sum_values.append(m.sum)
        count_values.append(m.count)
        if inplace:
            result_meters.append(m)
        else:
            new_m = AverageMeter(m.name, m.fmt, m.summary_type)
            new_m.val = m.val
            result_meters.append(new_m)
    sum_values = torch.tensor(sum_values, device=device)
    count_values = torch.tensor(count_values, dtype=torch.int32, device=device)
    if dst < 0:
        handle1 = dist.all_reduce(sum_values, dist.ReduceOp.SUM, async_op=True)
        handle2 = dist.all_reduce(count_values, dist.ReduceOp.SUM, async_op=True)
    else:
        handle1 = dist.reduce(sum_values, dst, dist.ReduceOp.SUM, async_op=True)
        handle2 = dist.reduce(count_values, dst, dist.ReduceOp.SUM, async_op=True)
    handle1.wait()
    handle2.wait()
    sum_values = sum_values.tolist()
    count_values = count_values.tolist()
    for m, sum, count in zip(result_meters, sum_values, count_values):
        m.sum = sum
        m.count = count
        m.avg = sum / count
    return result_meters
