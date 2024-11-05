from api.storage import NaiveStorage
from api.models.schema import DatabaseSchema
from api.models.typed_model import Model
from pydantic import BaseModel  # Assuming you're using Pydantic's BaseModel
    
class SchemaStorage(NaiveStorage[BaseModel]):

    def __init__(self, url: str):
        super().__init__(url, "schema", model_class=DatabaseSchema)

class ModelStorage(NaiveStorage[BaseModel]):

    def __init__(self, url: str):
        super().__init__(url, "model", model_class=Model)
