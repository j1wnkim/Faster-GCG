class Meter(object):
    """Base clss of meters."""

    def __init__(self, name="", fmt="{}"):
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def summary(self) -> str:
        return self.__str__()
