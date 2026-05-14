from .meter import Meter


class SimpleMeter(Meter):
    """Always store the current value."""

    def __init__(self, name="", fmt="{}"):
        super().__init__(name, fmt)
        self.val = None

    def update(self, val, n=1):
        self.val = val
        return self

    def result(self):
        return self.val

    def __str__(self):
        fmtstr_name = "{name}=" if self.name else ""
        fmtstr = fmtstr_name + self.fmt.replace("{", "{val")
        return fmtstr.format(**self.__dict__)
