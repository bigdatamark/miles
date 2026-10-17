# [MODULE DOC] HELPER:
# Small naming utilities for the Megatron backend.
def strip_param_name_prefix(name: str | None) -> str | None:
    # [FUNCTION DOC] HELPER: Returns none for none. Otherwise repeatedly removes a leading
    # "module." from the parameter name until it no longer starts with it (handles nested
    # DDP/Float16 wrapper prefixes) and returns the result.
    if name is None:
        return None
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name
