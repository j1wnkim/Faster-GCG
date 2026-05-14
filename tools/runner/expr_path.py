import os.path as osp

from llm_attacks import utils

from .cli_config import CLIConfig


def get_expr_path(args: CLIConfig, train=True):
    if args.expr_path:
        return args.expr_path
    expr_path = args.output_dir
    suffix = utils.get_expr_suffix(expr_path, args.expr_name, train)
    return osp.join(expr_path, suffix)
