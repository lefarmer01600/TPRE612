# src/api/auth.py

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

# Load valid tokens from environment variable
# Format: "token1,token2,token3"
VALID_TOKENS = set(
    t.strip()
    for t in os.getenv("API_TOKENS", "").split(",")
    if t.strip()
)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validate the Bearer token.
    Raises 401 if invalid or missing.
    """
    token = credentials.credentials

    if not VALID_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No API tokens configured on the server",
        )

    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
