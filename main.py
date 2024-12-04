import uvicorn
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.settings import EnvironmentVariables
from api.routers.database_router import router as database_router
from api.routers.tools_router import router as tools_router
from api.middleware import SimpleAuthMiddleware, ValueErrorMiddleware

env = EnvironmentVariables()

# Configure logging
logging.disable(getattr(logging, env.LOG_LEVEL, logging.INFO))

# Print all environment variables on initialization
for key, value in env.model_dump().items():
    logger.info(f"{key}={value}")

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

if env.USERNAME and env.PASSWORD:
    app.add_middleware(
        SimpleAuthMiddleware, 
        username=env.USERNAME, 
        password=env.PASSWORD,
    )

app.add_middleware(ValueErrorMiddleware)
app.include_router(database_router, prefix="/api/v1")
app.include_router(tools_router, prefix="/api/v1")

@app.get("/")
async def health_check_0():
    return {"status": "ok"}

@app.get("/health", tags=["Monitoring"])
async def health_check_1():
    """
    Health endpoint to check if the application is running.
    This is a simple endpoint that just returns a 200 status.
    """
    return {"status": "ok"}

@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    """
    Ready endpoint to check if the application is fully initialized and ready to handle requests.
    It checks dependencies like database connectivity.
    """
    try:
        result = await env.solver.health()
        if not result:
            raise Exception("no response from solver")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

    return {"status": "ready"}

class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1 and record.getMessage().find("/ready") == -1

# Filter out healthcheck and readiness check logs
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=env.PORT)