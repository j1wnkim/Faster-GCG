from enum import Enum

from .meter import Meter


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(Meter):
    """Compute and store the average and current value."""

    def __init__(self, name="", fmt="{}", summary_type=Summary.AVERAGE):
        super().__init__(name, fmt)
        self.summary_type = summary_type
        self.val = None
        self.avg = None
        self.sum = None
        self.count: int = 0

    def update(self, val, n=1):
        self.val = val
        if self.sum is None:
            self.sum = val * n
        else:
            self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self

    def result(self):
        if self.summary_type is Summary.NONE:
            val = self.val
        elif self.summary_type is Summary.AVERAGE:
            val = self.avg
        elif self.summary_type is Summary.SUM:
            val = self.sum
        elif self.summary_type is Summary.COUNT:
            val = self.count
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return val

    def __str__(self):
        fmtstr_name = "{name}=" if self.name else ""
        fmtstr = (
            fmtstr_name
            + self.fmt.replace("{", "{val")
            + " ("
            + self.fmt.replace("{", "{avg")
            + ")"
        )
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name}=" + self.fmt.replace("{", "{avg")
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name}=" + self.fmt.replace("{", "{sum")
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name}=" + self.fmt.replace("{", "{count")
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)
