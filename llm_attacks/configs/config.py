from argparse import ArgumentParser, Namespace
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Optional, TypeVar

import yaml
from transformers import HfArgumentParser

from llm_attacks.utils.yaml import safe_load_with_env

T = TypeVar("T")


class BaseCLIConfig(Namespace):
    """This config specifies config files only."""

    config: list[str]


def get_base_parser():
    """Base parser to specify YAML config files.

    Support `-c file1 -c file2 file3`."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        "--cfg",
        "-c",
        action="extend",
        metavar="FILE",
        nargs="+",
        default=[],
        help="YAML config file(s) specifying default arguments",
    )
    return parser


def init_nested_dataclasses(obj):
    """Try to initialize nested dataclass objects. Return the object itself."""
    for f in fields(obj):
        attr = getattr(obj, f.name)
        if is_dataclass(f.type) and isinstance(attr, dict):
            setattr(obj, f.name, f.type(**attr))
    return obj


def parse_args(cls: type[T], args: Optional[list[str]] = None) -> T:
    """Parse arguments.

    Overriding hierarchy is (priority high to low):
        - Command line argument
        - Yaml files (latter overrides former)
        - Dataclass default

    Args:
        cls (type[T: DataClass]):
            Dataclass with specifications. Refer to `transformers.HfArgumentParser` for
            usage.
        args (Sequence[str], optional):
            Arguments. If left blank, will parse args from command line. Default: None.

    Returns:
        dataclass (T):
            Dataclass object containing parsed arguments.
    """
    # Base parser is fixed
    base_parser = get_base_parser()
    base_args, remaining = base_parser.parse_known_args(args, namespace=BaseCLIConfig())
    # Load yaml files
    parser = HfArgumentParser(cls)
    for cfg in base_args.config:
        parser.set_defaults(**load_yaml_args(cfg))
    objects = parser.parse_args_into_dataclasses(args=remaining)
    if not is_dataclass(objects[-1]):
        # That's remaining args from yaml file
        raise ValueError(
            "Some args in config file(s) are not used by the HfArgumentParser: "
            + ", ".join(objects[-1].__dict__.keys())
        )
    for obj in objects:
        # A bonus: we init nested dataclasses
        init_nested_dataclasses(obj)
    return objects[0]


def load_yaml_args(fn: str):
    """Load args in yaml file recursively.

    A 'base_config' field is reserved, which support either a string for filename,
    or a list of filenames, which are the parent configs that are automatically loaded.
    However, using a list of configs is recommended, since it better supports combination
    of configs of different parts.

    Args:
        fn (str): filename.
        parser (ArgumentParser): parser to store arguments.

    Returns:
        args (dict): the args loaded.
    """
    path = Path(fn)
    yaml_args: dict = safe_load_with_env(path.read_text())

    # Base config is still supported, but in some cases, file list is more convenient
    base_files: list[str] | str = yaml_args.pop("base_config", [])
    if isinstance(base_files, str):
        base_files = [base_files]
    all_args = {}
    for base_file in base_files:
        # Recursive load
        all_args.update(load_yaml_args(path.parent / base_file))

    all_args.update(yaml_args)
    return all_args


def dump_args(fn: str, args):
    """Dump all arguments into yaml file.

    Args:
        fn (str): filename.
        args (DataClass): arguments to be dumped.
    """
    with open(fn, "w") as f:
        yaml.safe_dump(asdict(args), f)
