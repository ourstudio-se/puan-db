
import logging
import pickle

from dataclasses import dataclass
from redis import Redis
from typing import Type, TypeVar, Generic, Optional, Dict, List
from pydantic import BaseModel
from fastapi import HTTPException

logger = logging.getLogger(__name__)

@dataclass
class RedisStorage:

    url: str

    def __post_init__(self):
        self.client = Redis.from_url(self.url)
        self.client.ping()

    def store_pickle(self, key: str, value):
        self.client.set(key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def load_pickle(self, key: str):
        data = self.client.get(key)
        if data is None:
            return None
        return pickle.loads(data)
    
# Define a type variable that must be a subclass of BaseModel
T = TypeVar("T", bound=BaseModel)

@dataclass
class NaiveStorage(Generic[T], RedisStorage):

    __mname__: str
    model_class: Type[T]

    def get(self, id: str) -> Optional[T]:
        data = self.load_pickle(f"{self.__mname__}:{id}")
        if data is None:
            return None
        
        try:
            return self.model_class.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to validate data for {id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load data for '{id}': data is corrupt. If possible, try to delete and then recreate.")
    
    def get_keys(self) -> List[str]:
        return list(
            map(
                lambda x: x.decode("utf-8").replace(f"{self.__mname__}:", ""),
                self.client.keys(f"{self.__mname__}:*")
            )
        )
    
    def set(self, id: str, obj: T):
        # We can now safely call model_dump, knowing T is a BaseModel
        self.store_pickle(f"{self.__mname__}:{id}", obj.model_dump())

    def delete(self, id: str):
        self.client.delete(f"{self.__mname__}:{id}")

    def exists(self, id: str) -> bool:
        return self.client.exists(f"{self.__mname__}:{id}") == 1