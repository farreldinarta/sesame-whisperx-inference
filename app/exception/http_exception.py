from typing import Any
from fastapi import HTTPException as FastAPIHTTPException
from starlette.websockets import WebSocketException as StarletteWebSocketException
from app.dto.base import APIResponse


class BaseAPIException(Exception):
    def __init__(self, status: int, message: str, error: Any):
        self.status = status
        self.api_response = APIResponse(
            message=message,
            data=None,
            error=error
        )


class HTTPException(BaseAPIException, FastAPIHTTPException):
    def __init__(self, status: int, message: str, error: Any):
        BaseAPIException.__init__(self, status, message, error)
        FastAPIHTTPException.__init__(self, status_code=status, detail=self.api_response.dict())


class WebSocketAPIException(BaseAPIException, StarletteWebSocketException):
    def __init__(self, status: int, message: str, error: Any):
        BaseAPIException.__init__(self, status, message, error)
        StarletteWebSocketException.__init__(self, code=status, reason=self.api_response.message)
