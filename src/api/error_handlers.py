import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logger = logging.getLogger("api.errors")


def _error_response(
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    error: dict[str, object] = {
        "code": code,
        "message": message,
    }

    if details:
        error["details"] = details

    return {"error": error}


def _validation_details(errors: list[dict[str, object]]) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []

    for error in errors:
        location = error.get("loc", ())
        if isinstance(location, tuple):
            field_parts = [str(part) for part in location if part != "body"]
        elif isinstance(location, list):
            field_parts = [str(part) for part in location if part != "body"]
        else:
            field_parts = []

        details.append(
            {
                "field": ".".join(field_parts) if field_parts else "request",
                "message": str(error.get("msg", "Valeur invalide")),
                "type": str(error.get("type", "validation_error")),
            }
        )

    return details


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = exc.status_code
    detail = exc.detail if isinstance(exc.detail, str) else None

    if status_code == status.HTTP_401_UNAUTHORIZED:
        code = "unauthorized"
        message = "Authentification requise ou invalide."
    elif status_code == status.HTTP_403_FORBIDDEN:
        code = "forbidden"
        message = "Accès refusé à cette ressource."
    elif status_code == status.HTTP_404_NOT_FOUND:
        code = "not_found"
        message = detail or "La ressource demandée est introuvable."
    elif status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        code = "validation_error"
        message = "La requête est invalide. Vérifiez les champs fournis."
    elif status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        code = "service_unavailable"
        message = "Le service est temporairement indisponible. Réessayez plus tard."
    elif status_code >= 500:
        code = "internal_error"
        message = "Une erreur interne est survenue. Réessayez plus tard."
    else:
        code = f"http_{status_code}"
        message = detail or "Une erreur est survenue."

    logger.info(
        "http_exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "error_code": code,
        },
    )

    return JSONResponse(
        status_code=status_code,
        content=_error_response(code=code, message=message),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.info(
        "request_validation_failed",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_response(
            code="validation_error",
            message="La requête est invalide. Vérifiez les champs fournis.",
            details=_validation_details(exc.errors()),
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_exception",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_response(
            code="internal_error",
            message="Une erreur interne est survenue. Réessayez plus tard.",
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)