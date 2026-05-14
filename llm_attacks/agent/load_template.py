import json

import yaml


def load_template(path: str):
    """Load template from file.

    Returns:
        object: The object loaded.

    Raises:
        RuntimeError: when the format is not recognized.
    """
    path = str(path)
    if path.endswith(".json"):
        with open(path, "r") as fp:
            obj = json.load(fp)
    elif path.endswith((".yaml", ".yml")):
        with open(path, "r") as fp:
            obj = yaml.safe_load(fp)
    else:
        # load txt
        with open(path, "r") as fp:
            text = fp.read(1 << 31)  # 2GiB at most
            obj = {"prompts": [{"role": "user", "content": text}]}
    return obj
