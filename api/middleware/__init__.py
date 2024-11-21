import base64
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class PassThroughMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        return response

class SimpleAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, username: str, password: str):
        super().__init__(app)
        self.username = username
        self.password = password

    async def dispatch(self, request: Request, call_next):
        
        # Extract the Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(status_code=401, content={"detail": "Authorization header missing"})
        
        # Basic validation assuming the format: "Basic username:password"
        try:
            auth_type, encoded_credentials = auth_header.split()
            if auth_type != "Basic":
                return JSONResponse(status_code=401, content={"detail": "Invalid authentication type"})
            
            # Decode the Base64 encoded credentials
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            provided_username, provided_password = decoded_credentials.split(":")
            
            if provided_username != self.username or provided_password != self.password:
                return JSONResponse(status_code=401, content={"detail":"Invalid credentials"})
        
        except ValueError:
            return JSONResponse(status_code=401, content={"detail": "Invalid Authorization header format"})

        # If validation passes, continue to the next middleware or route
        response = await call_next(request)
        return response
    
class ValueErrorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except ValueError as e:
            return JSONResponse(
                status_code=400, 
                content={
                    "type": "SCHEMA_VALIDATION_ERROR",
                    "message": str(e)
                }
            )
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "type": "UNKNOWN_ERROR",
                    "message": "Internal Server Error"
                }
            )