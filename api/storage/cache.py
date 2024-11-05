import redis
from typing import Any, Optional
from pldag_solver_sdk import ICache

class PolyhedronBuilderCache(ICache):

    CACHE_PREFIX = "cache-polyhedron-build:"

    def __init__(self, client: redis.Redis = None):
        self.client = client

    def get(self, key: str) -> Optional[Any]:
        if self.client:
            data = self.client.get(key)
            if data is not None:
                return data
        return None
    
    def set(self, key: str, value: Any):
        if self.client:
            self.client.set(PolyhedronBuilderCache.CACHE_PREFIX+key, value)

    def clear(self):
        if self.client:
            for key in self.client.scan_iter(f"{PolyhedronBuilderCache.CACHE_PREFIX}:*"):
                self.client.delete(key)
