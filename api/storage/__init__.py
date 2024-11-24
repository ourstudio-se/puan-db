
import logging
import pickle
import re

from dataclasses import dataclass, field
from redis import Redis
from typing import Type, TypeVar, Generic, Optional, Dict, List, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class LocalStorage:

    _data: Dict[str, Any] = field(default_factory=dict)

    def ping(self) -> bool:
        return True

    def set(self, key: str, value) -> None:
        self._data[key] = value

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)
    
    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._data
    
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        return [self._data.get(key) for key in keys]
    
    def keys(self, pattern: str) -> List[str]:
        return [
            key
            for key in self._data.keys()
            if re.match(pattern.replace("*", ".*"), key)
        ]

@dataclass
class RedisStorage:

    url: str

    def __post_init__(self):
        if self.url is None:
            self.client = LocalStorage()
        else:
            self.client = Redis.from_url(self.url)
        self.client.ping()

    def store_pickle(self, key: str, value):
        self.client.set(key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def load_pickle(self, key: str):
        data = self.client.get(key)
        if data is None:
            return None
        return pickle.loads(data)
    
    def load_pickles(self, keys: List[str]):
        datas = self.client.mget(keys)
        return [
            pickle.loads(data)
            for data in datas
            if data is not None
        ]
    
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
        
        return data
    
    def get_keys(self) -> List[str]:
        return list(
            map(
                lambda x: x.decode("utf-8").replace(f"{self.__mname__}:", ""),
                self.client.keys(f"{self.__mname__}:*")
            )
        )
    
    def get_all(self) -> Dict[str, T]:
        return self.load_pickles(self.client.keys(f"{self.__mname__}:*"))
    
    def set(self, id: str, obj: T):
        # We can now safely call model_dump, knowing T is a BaseModel
        self.store_pickle(f"{self.__mname__}:{id}", obj.model_dump())

    def delete(self, id: str):
        self.client.delete(f"{self.__mname__}:{id}")

    def exists(self, id: str) -> bool:
        return self.client.exists(f"{self.__mname__}:{id}") == 1