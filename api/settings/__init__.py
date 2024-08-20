from logging import Logger
from dataclasses import dataclass
from api.services.database_service import DatabaseService
from typing import Optional

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def clear(cls):
        cls._instances = {}

@dataclass
class PuanDBSettings(metaclass=SingletonMeta):

    service: Optional[DatabaseService] = None
    logger: Optional['Logger'] = None

    @classmethod
    def instance(cls) -> "PuanDBSettings":
        return cls._instances.get(cls, None)