from api.storage import NaiveStorage
from api.models.schema import Database
from api.models.typed_model import DatabaseModel
from pydantic import BaseModel  # Assuming you're using Pydantic's BaseModel
    
class SchemaStorage(NaiveStorage[BaseModel]):

    def __init__(self, url: str):
        super().__init__(url, "schema", model_class=Database)

class ModelStorage(NaiveStorage[BaseModel]):

    def __init__(self, url: str):
        super().__init__(url, "model", model_class=DatabaseModel)
