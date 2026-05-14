import csv

try:
    import wandb
except ImportError:
    pass
from collections import OrderedDict
from typing import Any, Callable

from .meter import Meter


def write_meters_summary(
    meters: list[Meter],
    log_wandb=False,
    csv_filename: str | None = None,
    csv_write_header=False,
    stream=None,
    log_fn: Callable | None = None,
    delim="  ",
):
    """All-in-one writer for meters.

    meters (list[Meter]):
        Meters to summarize.
    log_wandb (bool):
        Set to True if you want to log to wandb.
    csv_filename (str | None):
        Set to write to csv file.
    csv_write_header (bool):
        Set to True to write header.
    stream (Any):
        Set to write to stream.
    log_fn (Callable):
        Log function to write to log.
    delim (str):
        Delimitator between meters. Default: two spaces.
    """
    row: OrderedDict[str, Any] = OrderedDict()
    for m in meters:
        # gather result
        row[m.name] = m.result()
    if log_wandb:
        # log wandb
        wandb.log(row)
    if csv_filename:
        # log csv
        with open(csv_filename, "a") as fp:
            writer = csv.DictWriter(fp, fieldnames=row.keys())
            if csv_write_header:
                writer.writeheader()
            writer.writerow(row)

    # Formatted row
    row_f: OrderedDict[str, Any] = OrderedDict()
    for m in meters:
        # gather result with format
        row_f[m.name] = m.summary()
    row_f = row_f.values()
    row_f = delim.join(row_f)
    if stream is not None:
        print(row_f, file=stream)
    if log_fn is not None:
        log_fn(row_f)
    return row
