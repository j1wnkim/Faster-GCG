from typing import Iterable

from .meter import Meter


class ProgressMeter(object):
    """A 'meter of meters', used to track progress in one epoch."""

    def __init__(self, num_batches: int, meters: Iterable[Meter], prefix="", sep="\t"):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.sep = sep

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return self.sep.join(entries)

    def display_summary(self):
        entries = [self.prefix + " *"]
        entries += [meter.summary() for meter in self.meters]
        return self.sep.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
