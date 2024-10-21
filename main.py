import logging
import uvicorn

from fastapi import FastAPI
from api.settings import EnvironmentVariables
from api.routers.database_router import router as database_router
from api.routers.tools_router import router as tools_router
from api.middleware import SimpleAuthMiddleware, PassThroughMiddleware

env = EnvironmentVariables()
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    PassThroughMiddleware, 
    # username=env.USERNAME, 
    # password=env.PASSWORD,
)

# Include the router with the dependency passed
app.include_router(
    database_router, 
    prefix="/api/v1",
)

app.include_router(
    tools_router, 
    prefix="/api/v1",
)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=env.PORT)