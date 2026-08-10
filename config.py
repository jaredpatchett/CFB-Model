"""
Central config: reads API keys from environment variables.
Locally: put them in a .env file (gitignored) and this loads it automatically.
In GitHub Actions: they're injected as repo secrets, no .env needed.
"""
import os
from dotenv import load_dotenv

load_dotenv()

CFBD_API_KEY = os.environ.get("CFBD_API_KEY")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
ODDSPAPI_API_KEY = os.environ.get("ODDSPAPI_API_KEY")

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


def require_keys(*names):
    """Fail loudly and early if a required key is missing, instead of a confusing
    downstream 401 from the API."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise EnvironmentError(
            f"Missing required API key(s): {', '.join(missing)}. "
            f"Set them in a .env file locally, or as GitHub Actions secrets in CI."
        )
