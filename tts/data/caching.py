import os

_HF_CACHE_DIR = "hf_cache"


def get_hf_cache_dir() -> str:
    """Returns the cache directory for the HF assets."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(project_root, _HF_CACHE_DIR)
