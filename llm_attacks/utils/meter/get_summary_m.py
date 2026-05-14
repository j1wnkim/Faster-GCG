from typing import Iterable

from .meter import Meter


def get_results(meters: Iterable[Meter]):
    """Get result values of the meters as a dictionary."""
    return dict((m.name, m.result()) for m in meters)


def get_summary(meters: Iterable[Meter], format="{name}={val}", delim="  "):
    r"""Get summary string from meters.

    Suggested formats:
      - format="{name}={val}", delim="  "
      - format="{val}", delim="\t" (note: use format="{name}", delim="\t" to get header)
      - format="csv" (note: delim is ignored and set to "," in csv format)
            (not recommended, this doesn't use standard csv writer and doesn't support content with ',')
    """
    if format == "csv":
        return ",".join(str(m.result()) for m in meters)
    l = []
    for m in meters:
        l.append(
            format.replace("{name}", m.name).replace("{val}", m.fmt.format(m.result()))
        )
    return delim.join(l)
