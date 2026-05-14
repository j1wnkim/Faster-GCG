import re


def render_template(template: str, values: dict) -> str:
    """Render a template."""

    def repl(match: re.Match):
        key = match.group(1)
        return str(values.get(key, match.group(0)))  # keep unchanged if missing

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", repl, template)
