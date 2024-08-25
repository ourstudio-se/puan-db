import logging
import uvicorn

from fastapi import FastAPI
from starlette.config import Config

from api.routers.database_router import router as database_router
from api.middleware import SimpleAuthMiddleware

config = Config(".env")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    SimpleAuthMiddleware, 
    username=config('username', str, 'root'), 
    password=config('password', str, '')
)

# Include the router with the dependency passed
app.include_router(
    database_router, 
    prefix="/api/v1",
)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=config('PORT', int, 8000))