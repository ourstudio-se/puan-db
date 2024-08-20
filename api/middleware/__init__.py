import base64
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

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