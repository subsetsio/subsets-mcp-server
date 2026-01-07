import json
from pathlib import Path
from typing import Optional


def get_api_key() -> Optional[str]:
    """Get API key from auth file."""
    auth_path = Path.home() / "subsets" / "auth.json"
    if auth_path.exists():
        try:
            with open(auth_path, 'r') as f:
                auth_data = json.load(f)
                return auth_data.get('api_key')
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_api_key(api_key: str) -> None:
    """Save API key to auth file."""
    auth_path = Path.home() / "subsets" / "auth.json"
    auth_path.parent.mkdir(exist_ok=True)
    
    auth_data = {'api_key': api_key}
    with open(auth_path, 'w') as f:
        json.dump(auth_data, f, indent=2)


def clear_api_key() -> None:
    """Clear API key from auth file."""
    auth_path = Path.home() / "subsets" / "auth.json"
    if auth_path.exists():
        auth_path.unlink()


def get_api_url() -> str:
    """Get API URL from environment variable.

    Defaults to https://api.subsets.io if SUBSETS_API_URL is not set.
    """
    import os
    return os.environ.get('SUBSETS_API_URL', 'https://api.subsets.io')