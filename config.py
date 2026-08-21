"""
Central config: reads API keys from environment variables.
Locally: put them in a .env file (gitignored) and this loads it automatically.
In GitHub Actions: they're injected as repo secrets, no .env needed.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def _clean_key(value):
    """Strip whitespace/newlines that sometimes sneak in when a key is pasted
    into a GitHub secret or .env file — a trailing '\\n' breaks HTTP headers
    with a cryptic 'Invalid leading whitespace' error otherwise."""
    return value.strip() if value else value


CFBD_API_KEY = _clean_key(os.environ.get("CFBD_API_KEY"))
ODDS_API_KEY = _clean_key(os.environ.get("ODDS_API_KEY"))
ODDSPAPI_API_KEY = _clean_key(os.environ.get("ODDSPAPI_API_KEY"))

CFBD_BASE_URL = "https://api.collegefootballdata.com"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDSPAPI_BASE_URL = "https://api.oddspapi.io/v4"

# The Odds API sport key for college football
ODDS_API_NCAAF_KEY = "americanfootball_ncaaf"

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_CURRENT_DIR = "data/current"
DATA_PREDICTIONS_DIR = "data/predictions"
MODELS_DIR = "models"

# CLV (closing-line value) line-snapshot log. Deliberately NOT under
# data/raw or data/processed -- those are gitignored/regenerated-at-runtime
# (see .gitignore), but this file has to ACCUMULATE across every workflow
# run to be useful (it's the only record of what line the model's pick was
# actually captured at, before the closing line comes in) -- so it's
# committed to the repo like docs/data/*.json, not cache-only. See
# src/analysis/clv.py for how it's written/read.
DATA_CLV_DIR = "data/clv"
CLV_SNAPSHOTS_PATH = f"{DATA_CLV_DIR}/line_snapshots.csv"


def require_keys(*names):
    """Fail loudly and early if a required key is missing, instead of a confusing
    downstream 401 from the API."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise EnvironmentError(
            f"Missing required API key(s): {', '.join(missing)}. "
            f"Set them in a .env file locally, or as GitHub Actions secrets in CI."
        )
