"""General representation paradigm."""


def get_general_repr(
    obj: object,
    field_cfgs: list[dict | str],
    extra_fields: dict | None = None,
    delim=", ",
):
    for i, cfg in enumerate(field_cfgs):
        if isinstance(cfg, str):
            field_cfgs[i] = dict(field=cfg)
    fmt = get_fmt_string(field_cfgs, delim=delim)
    cls_name = obj.__class__.__name__
    extra_fields = extra_fields or {}
    kwargs = obj.__dict__.copy()
    kwargs.update(extra_fields)
    return "{}({})".format(cls_name, fmt).format_map(kwargs)


def get_fmt_string(field_cfgs: list[dict], delim=", "):
    """Get format string for multiple fields."""
    fmt_strings = []
    for field_cfg in field_cfgs:
        fmt = get_single_fmt_string(**field_cfg)
        if fmt:
            fmt_strings.append(fmt)
    return delim.join(fmt_strings)


def get_single_fmt_string(
    field: str, var_name=None, fmt=None, prefix="", suffix="", cond=True
):
    """Get format string for a single field.

    Args:
        field (str):
            Field name.
        var_name (str, optional):
            Variable name override. For example, when you want `a={a_display}`.
        fmt (str, optional):
            Format string like '2d'.
        prefix (str, optional):
            Prefix on field value.
        suffix (str, optional):
            Suffix on field value.
        cond (bool, optional):
            If False, this function returns empty string.
    """
    if not cond:
        return ""
    between_brackets = var_name or field
    if fmt:
        between_brackets = f"{between_brackets}:{fmt}"
    return f"{field}={prefix}{{{between_brackets}}}{suffix}"
