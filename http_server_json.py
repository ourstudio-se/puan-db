import logging
import uvicorn

from fastapi import FastAPI
from redis import Redis
from starlette.config import Config

from api.routers.db import router as db_router
from api.middleware import SimpleAuthMiddleware
from api.settings import PuanDBSettings
from api.services.database_service import DatabaseService, RedisHandler

config = Config(".env")
logger = logging.getLogger(__name__)
PuanDBSettings(
    logger = logging.getLogger(__name__),
    service= DatabaseService(
        logger=logger,
        handler=RedisHandler(
            logger=logger,
            client=Redis(
                host=config('REDIS_HOST', str, 'localhost'),
                port=config('REDIS_PORT', int, 6379),
                db=config('REDIS_DB', int, 0),
                password=config('REDIS_PASSWORD', str, None),
            )
        )
    )
)

app = FastAPI()
app.add_middleware(
    SimpleAuthMiddleware, 
    username=config('username', str, 'root'), 
    password=config('password', str, '')
)

# Include the router with the dependency passed
app.include_router(
    db_router, 
    prefix="/api/v1/db",
)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=config('PORT', int, 8000))