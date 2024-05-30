import hashlib
import pickle
import gzip
import os
import logging

from dataclasses import dataclass, field
from pldag import Puan
from typing import Optional

@dataclass
class ModelHandler:

    salt: str

    def create_token(self, id: str, password: str) -> str:
        return hashlib.sha256(f"{id}-{password}-{self.salt}".encode()).hexdigest()
    
    def create_token_from_model(self, model) -> str:
        return hashlib.sha256(f"{model.sha1()}{self.salt}".encode()).hexdigest()

    def create_model(self, id: str, password: str) -> str:
        token = self.create_token(id, password)
        self.save_model(Puan(), token)
        return token

    def save_model(self, model, token):
        raise NotImplementedError("Method not implemented")

    def load_model(self, token: str) -> Optional[Puan]:
        raise NotImplementedError("Method not implemented")
        
    def verify_token(self, token: str) -> bool:
        raise NotImplementedError("Method not implemented")

    
@dataclass
class LocalModelHandler(ModelHandler):

    def save_model(self, model, token):
        with open(token, "wb") as f:
            f.write(gzip.compress(pickle.dumps(model)))

    def load_model(self, token: str) -> Optional[Puan]:
        try:
            with open(token, "rb") as f:
                return pickle.loads(gzip.decompress(f.read()))
        except Exception as e:
            logging.log(logging.ERROR, e)
            return None
        
    def verify_token(self, token: str) -> bool:
        return os.path.exists(token)
    

from io import BytesIO
from azure.storage.blob import BlobServiceClient

@dataclass
class AzureBlobModelHandler(ModelHandler):
    
    connection_string: str = field(repr=False)
    container: str

    def blob(self, token: str):
        return BlobServiceClient.from_connection_string(self.connection_string).get_blob_client(self.container, token)
    
    def save_model(self, model, token):
        client = self.blob(token)
        client.upload_blob(gzip.compress(pickle.dumps(model)), overwrite=True)
    
    def load_model(self, token: str) -> Optional[Puan]:

        # Download the blob's content into a BytesIO object
        stream = BytesIO()
        self.blob(token).download_blob().download_to_stream(stream)

        # Reset the stream position to the beginning
        stream.seek(0)
        model = pickle.loads(gzip.decompress(stream.read()))
        return model
    
    def verify_token(self, token: str) -> bool:
        return self.blob(token).exists()
    
    def list_blobs(self):
        container_client = BlobServiceClient.from_connection_string(self.connection_string).get_container_client(self.container)
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs]
        
@dataclass
class ComputingDevice:

    token: str
    handler: LocalModelHandler

    def get(self):
        return self.handler.load_model(self.token)
    
    def save(self, model):
        self.handler.save_model(model, self.token)

    def modify(self, f):
        """
            Compute and save model.
        """
        model = self.get()
        result = f(model)
        self.handler.save_model(model, self.token)
        return result
    
    def compute(self, f):
        """
            Just computation, no saving.
        """
        return f(self.get())