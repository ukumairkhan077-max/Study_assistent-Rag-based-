"""Usage (from backend/, venv active):  python reindex.py
Re-embeds every uploaded document with the model currently set in .env."""
import json

from app.core.config import get_settings
from app.services.reindex import reindex_all

if __name__ == "__main__":
    print(json.dumps(reindex_all(get_settings()), indent=2))
