import logging
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pldag_solver_sdk import Solver as PLDAGSolver, ConnectionError as SolverConnectionError

from api.settings import EnvironmentVariables
from api.routers.database_router import router as database_router
from api.routers.tools_router import router as tools_router
from api.middleware import SimpleAuthMiddleware

env = EnvironmentVariables()
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

if env.USERNAME and env.PASSWORD:
    app.add_middleware(
        SimpleAuthMiddleware, 
        username=env.USERNAME, 
        password=env.PASSWORD,
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

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Health endpoint to check if the application is running.
    This is a simple endpoint that just returns a 200 status.
    """
    return {"status": "healthy"}

@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    """
    Ready endpoint to check if the application is fully initialized and ready to handle requests.
    It checks dependencies like database connectivity.
    """
    try:
        env.solver.health()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

    return {"status": "ready"}

@app.exception_handler(ValueError)
async def model_validation_exception_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"errors": [e.get('msg') for e in exc.errors()]}
    )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=env.PORT)