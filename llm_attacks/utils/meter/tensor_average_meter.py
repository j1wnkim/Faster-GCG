import torch

from .average_meter import AverageMeter, Summary


class TensorAverageMeter(AverageMeter):
    """Similar to AverageMeter, but keep everything as tensor.

    This class could help to eliminate gpu synchronization across batches.
    """

    def __init__(
        self,
        name="",
        fmt="{}",
        summary_type=Summary.AVERAGE,
        device=torch.device("cpu"),
        dtype: torch.dtype = None,
    ):
        super().__init__(name, fmt, summary_type)
        self._sum = torch.zeros((1,), dtype=dtype, device=device)

    def update(self, val: torch.Tensor, n=1):
        self.val = val.detach()
        self._sum += self.val * n
        self.count += n

    def collect(self):
        self.sum = self._sum.item()
        self.avg = self.sum / self.count

    def result(self):
        self.collect()
        return super().result()

    def __str__(self):
        self.collect()
        return super().__str__()

    def summary(self):
        self.collect()
        return super().summary()
