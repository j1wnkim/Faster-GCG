import os.path as osp
from dataclasses import dataclass
from typing import Optional

import datasets as hfds
import torch.utils.data as thds
import torchvision.datasets as tvds

## hfds.load_dataset: datasetname, subsetname, splitname


@dataclass
class HfDatasetInfo:
    """Information of a huggingface dataset."""

    path: str
    """Path on hf.co, in the format of 'provider/dataset_name'."""
    default_name: str | None = None
    """Default subset name of dataset."""
    default_split: str | None = None
    """Default split of dataset."""


HF_DATASETS_INFO = {
    "jbb-behaviors": HfDatasetInfo(
        path="JailbreakBench/JBB-Behaviors",
        default_name="behaviors",
        default_split="harmful",
    ),
    "advbench": HfDatasetInfo(
        path="walledai/AdvBench",
        default_split="train",
    ),
}


def load_dataset(
    name="walledai/AdvBench",
    root="./dataset",
    train=False,
    split: Optional[str] = None,
    **kwargs,
) -> hfds.Dataset:
    """Load dataset from various sources.

    Some arguments are inherited and not used currently.
    """
    ds_base_name = name.rsplit("/", 1)[-1].lower()
    if ds_base_name in HF_DATASETS_INFO:
        if "/" not in name:
            # Then path and name are specified separately
            name = osp.join(root, name)
        ds_info = HF_DATASETS_INFO[ds_base_name]
        return hfds.load_dataset(
            name,
            name=ds_info.default_name,
            split=split or ds_info.default_split,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset {name}")
