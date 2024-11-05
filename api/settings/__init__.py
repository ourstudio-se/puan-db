from pydantic_settings import BaseSettings
from typing import Optional
from pldag_solver_sdk import Solver as PLDAGSolver, ConnectionError as SolverConnectionError
from api.storage.cache import PolyhedronBuilderCache

import redis

class EnvironmentVariables(BaseSettings):
    
    # Required settings
    DATABASE_URL:               str
    SOLVER_API_URL:             str

    # Optional settings
    CACHE_URL:  Optional[str]           = None
    USERNAME:   Optional[str]           = None
    PASSWORD:   Optional[str]           = None
    VERSION:    str                     = "0.1.0"
    PORT:       int                     = 8000
    LOG_LEVEL:  str                     = "INFO"
    APP_NAME:   str                     = "Puan DB"

    # Internal single-instance placeholders
    _solver_instance: Optional[PLDAGSolver] = None
    _polyhedron_cache_instance: Optional[PolyhedronBuilderCache] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def solver(self):
        # Initialize solver instance only once
        if not self._solver_instance:
            try:
                self._solver_instance = PLDAGSolver(url=self.SOLVER_API_URL, cache_builder=self.polyhedron_builder_cache)
            except SolverConnectionError as e:
                raise Exception(status_code=500, detail=f"No solver cluster available: {e}")
        return self._solver_instance

    @property
    def polyhedron_builder_cache(self):
        # Initialize cache instance only once
        if not self._polyhedron_cache_instance:
            self._polyhedron_cache_instance = PolyhedronBuilderCache(
                client=redis.Redis.from_url(self.CACHE_URL) if self.CACHE_URL else None
            )
        return self._polyhedron_cache_instance