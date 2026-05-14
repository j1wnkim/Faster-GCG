import torch
from torch import Tensor


class BucketHistoryTracker(object):
    """Track average values for multiple buckets.

    This class can be used to track the significance sampling with multiple buckets.

    Args:
        decay_factor (float, optional): decay factor. Default: 0.5.
        device (optional): device to put all tensors.

    Attributes:
        values ((N,) Tensor): sum of values.
        counts ((N,) Tensor): count of tracked instances.
        avg ((N,) Tensor, property): average values.
    """

    def __init__(self, num_buckets: int, decay_factor=0.5, device=None) -> None:
        self.values = torch.ones(num_buckets, device=device)
        self.counts = torch.ones(num_buckets, device=device)
        self.decay_factor = decay_factor
        self.device = device
        self._EPS = torch.finfo(self.values.dtype).eps

    def update(self, indices: Tensor, values: Tensor):
        """Update with a list of values.

        indices ((M,) int Tensor): indices of the buckets that the values belong to.
        values ((M,) Tensor): values to accumulate.
        """
        self.values.index_put_((indices,), values, accumulate=True)
        self.counts.index_put_((indices,), torch.ones_like(values), accumulate=True)
        return self

    def decay(self):
        """Decay the values for one step.

        t->inf gets avg->1.
        """
        self.values = self.values * self.decay_factor + self._EPS
        self.counts = self.counts * self.decay_factor + self._EPS
        return self

    @property
    def avg(self):
        """Average values."""
        return self.values / self.counts
