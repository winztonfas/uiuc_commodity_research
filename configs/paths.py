from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" 

# states future data path 
BENTO_DATABASE_DIR  = DATA_DIR / "us_continuous.duckdb"

# china future data path 
TUSHARE_DATABASE_PATH = DATA_DIR / "futures_data.ddb"

