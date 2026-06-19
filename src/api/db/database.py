import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# override=False ensures Docker-injected env vars (via --env-file) are NOT overwritten.
# We also mount the .env as a file inside the container at /app/.env for load_dotenv to find.
load_dotenv("/app/.env", override=False)

# Local dev fallback: walk up the directory tree looking for a .env file
for candidate in [
    Path(__file__).resolve().parents[3] / ".env",
    Path(__file__).resolve().parents[2] / ".env",
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / ".env",
]:
    if candidate.exists():
        load_dotenv(candidate, override=False)
        break

# Configuration
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME",     "TPRE612"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
}
print(f"DB_CONFIG: {DB_CONFIG}")

SCHEMA = os.getenv("API_DB_SCHEMA", "tpre612_data_warehouse")

DATABASE_URL = (
    f"postgresql://{quote_plus(str(DB_CONFIG['user']))}:{quote_plus(str(DB_CONFIG['password']))}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"options": f"-csearch_path={SCHEMA}"},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
Base.metadata.schema = SCHEMA

def get_db():
    """Dépendance FastAPI : fournit une session DB et la ferme après usage."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()