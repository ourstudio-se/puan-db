
from os import path, listdir
from gzip import compress, decompress
from time import time, strftime, localtime
from uuid import uuid5, NAMESPACE_DNS
from pickle import loads, dumps
from pldag import PLDAG, CompilationSetting, Solver
from itertools import chain, starmap

from dataclasses import dataclass, field
from api.models.system import (
    Database, 
    DatabaseMeta,
    Primitive, 
    LinearInequalityComposite, 
    Coefficient, 
    Bounds,
    PropositionStringUnionType,
    BinaryComposite,
    LogicalComposite, 
    ValueComposite,
    PropositionType,
)
from api.models.search import (
    SearchDatabaseRequest,
    SearchSolution,
    SearchSolutionVariable,
    SolverType,
    SearchDirection,
    EvaluateDatabaseRequestResponse
)
from typing import Optional, List, Dict, Any
from logging import Logger

class BytesNotFoundException(Exception):
    pass

class DatabaseAlreadyExistsException(Exception):
    pass

class SaveBytesException(Exception):
    pass

class DeleteBytesException(Exception):
    pass

@dataclass
class Handler:

    logger: Logger

    def load(self, ids: List[str]) -> Dict[str, bytes]:
        raise NotImplementedError("Method not implemented")

    def save(self, id: str, data: bytes):
        raise NotImplementedError("Method not implemented")
    
    def delete(self, id: str):
        raise NotImplementedError("Method not implemented")

    def exists(self, id: str) -> bool:
        raise NotImplementedError("Method not implemented")
    
    def blobs(self) -> List[str]:
        raise NotImplementedError("Method not implemented")

@dataclass
class LocalHandler(Handler):

    folder: str

    def load(self, ids: List[str]) -> Dict[str, bytes]:
        try:
            return dict(
                map(
                    lambda id: (id, open(self.folder + "/" + id, "rb").read()),
                    ids
                )
            )
        except FileNotFoundError as e:
            raise BytesNotFoundException()

    def save(self, id: str, data: bytes):
        try:
            with open(self.folder + "/" + id, "wb") as f:
                f.write(data)
        except Exception as e:
            raise SaveBytesException(str(e))
        
    def delete(self, id: str):
        try:
            path.unlink(self.folder + "/" + id)
        except Exception as e:
            raise DeleteBytesException(str(e))

    def exists(self, id: str) -> bool:
        return path.exists(self.folder + "/" + id)
    
    def blobs(self) -> List[str]:
        return [f for f in listdir(self.folder) if path.isfile(f)]

import asyncio
from io import BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
from io import BytesIO

@dataclass
class AzureHandler(Handler):

    connection_string: str = field(repr=False)
    container: str

    async def _download_blob_async(self, blob_client: BlobClient):
        stream = BytesIO()
        await blob_client.download_blob().download_to_stream(stream)
        stream.seek(0)
        return stream.read()

    async def _download_blobs_async(self, blob_names: List[str]) -> Dict[str, bytes]:
        container_client = self._blob_container_client()
        results = await asyncio.gather(
            *map(
                lambda x: self._download_blob_async(
                    container_client.get_blob_client(x),
                ),
                blob_names
            )
        )
        return dict(zip(blob_names, results))
    
    def _blob_service_client(self):
        return BlobServiceClient.from_connection_string(self.connection_string)

    def _blob_client(self, id: str):
        return self._blob_service_client().get_blob_client(self.container, id)    
    
    def _blob_container_client(self):
        return self._blob_service_client().get_container_client(self.container)
    
    def load(self, ids: List[str]) -> Dict[str, Optional[bytes]]:

        try:
            return asyncio.run(self._download_blobs_async(ids))
        except ResourceNotFoundError:
            raise BytesNotFoundException()    
        
    def delete(self, id: str):
        try:
            self._blob_client(id).delete_blob()
        except Exception as e:
            raise DeleteBytesException(str(e))

    def save(self, id: str, data: bytes):
        try:
            client = self._blob_client(id)
            client.upload_blob(data, overwrite=True)
        except Exception as e:
            raise SaveBytesException(str(e))

    def exists(self, id: str) -> bool:
        return self._blob_client(id).exists()
    
    def blobs(self):
        return [blob.name for blob in self._blob_container_client.list_blobs()]
    
from redis import Redis
    
@dataclass
class RedisHandler(Handler):

    client: Redis

    def load(self, ids: List[str]) -> Dict[str, bytes]:
        return dict(zip(ids, self.client.mget(ids)))
    
    def save(self, id: str, data: bytes):
        try:
            self.client.set(id, data)
        except Exception as e:
            raise SaveBytesException(str(e))
    
    def exists(self, id: str) -> bool:
        return bool(self.client.exists(id))
    
    def blobs(self) -> List[str]:
        return list(map(lambda x: x.decode(), self.client.keys("*-meta")))
    
    def delete(self, id: str):
        try:
            self.client.delete(id)
        except Exception as e:
            raise DeleteBytesException(str(e))

class SetCompositeException(Exception):
    
    def __init__(self, message: str, composite: PropositionStringUnionType):
        self.message = message
        self.composite = composite

class DatabaseNotFoundException(Exception):
    
    def __init__(self, id: str):
        super().__init__(f"Database not found: '{id}'")
        self.id = id

@dataclass
class DatabaseService:

    handler:    Handler
    logger:     Logger

    _meta_postfix:  str = "-meta"
    _model_postfix: str = "-model"

    def _name_meta(self, id: str) -> str:
        return id + self._meta_postfix
    
    def _name_model(self, id: str) -> str:
        return id + self._model_postfix

    @staticmethod
    def _from_bytes(b: Optional[bytes]) -> Any:
        if b is None:
            raise BytesNotFoundException()
        
        return loads(decompress(b))
    
    @staticmethod
    def _to_bytes(o: Any) -> bytes:
        return compress(dumps(o))
    
    def _rawload(self, id: str):
        try:
            return list(
                map(
                    self._from_bytes, 
                    self.handler.load([
                        self._name_meta(id),
                        self._name_model(id),
                    ]).values()
                )
            )
        except BytesNotFoundException:
            raise DatabaseNotFoundException(id)
        
    def _rawsave(self, meta: DatabaseMeta, model: PLDAG):
        for k, data in zip([self._name_meta(meta.id), self._name_model(meta.id)], map(self._to_bytes, [meta, model])):
            self.handler.save(k, data)

    @staticmethod
    def meta_repr(meta: DatabaseMeta) -> DatabaseMeta:
        return DatabaseMeta(
            id=meta.id,
            name=meta.name,
            description=meta.description,
            createdAt=strftime('%Y-%m-%d %H:%M:%S', localtime(float(meta.createdAt))),
            updatedAt=strftime('%Y-%m-%d %H:%M:%S', localtime(float(meta.updatedAt))),
            compilation=meta.compilation,
            nComposites=meta.nComposites,
            nPrimitives=meta.nPrimitives,
            nCombinations=meta.nCombinations,
        )

    def load(self, id: str) -> Database:        
        meta, model = self._rawload(id)
        try:
            return Database(
                propositions=self.from_pldag(model),
                meta=self.meta_repr(meta),
            )
        except BytesNotFoundException:
            raise DatabaseNotFoundException(id)
        except Exception as e:
            self.logger.error(f"Error loading database: {e}")
            raise Exception("Something went wrong")
        
    def load_model(self, id: str) -> PLDAG:
        try:
            return self._from_bytes(list(self.handler.load([self._name_model(id)]).values())[0])
        except BytesNotFoundException:
            raise DatabaseNotFoundException(id)
        
    def save(self, database: Database):
        
        try:
            current_model = self._from_bytes(self.handler.load(self._name_model(database.meta.id))[0])
            updated_model = self.merge(database, current_model)
            database.meta.updatedAt = str(time())
            self._rawsave(database.meta.id, database.meta, updated_model)

        except BytesNotFoundException:
            raise DatabaseNotFoundException(database.meta.id)
    
    def create(self, name: str, description: Optional[str] = None) -> DatabaseMeta:
        
        _id = str(uuid5(NAMESPACE_DNS, name))
        if self.handler.exists(self._name_meta(_id)):
            raise DatabaseAlreadyExistsException("Database already exists")
        
        meta = DatabaseMeta(
            id=_id,
            name=name, 
            description=description, 
            createdAt=str(time()), 
            updatedAt=str(time()),
        )
        
        try:    

            self._rawsave(meta, PLDAG(compilation_setting=CompilationSetting.ON_DEMAND))
            return self.meta_repr(meta)

        except Exception as e:

            # Rollback and delete if something goes wrong
            try:
                for k in [self._name_meta(meta.id), self._name_model(meta.id)]:
                    self.handler.delete(k)
            except Exception as _e:
                self.logger.critical(f"Rollback and deletion failed: {_id}, {_e}")

            self.logger.error(f"Error saving model: {e}")
            raise Exception("Somthing went wrong")
    
    def exists(self, id: str) -> bool:
        return self.handler.exists(self._name_meta(id)) and self.handler.exists(self._name_model(id))
    
    def databases(self) -> List[DatabaseMeta]:
        blobs_ = self.handler.blobs()
        return list(
            map(
                lambda x: self.meta_repr(self._from_bytes),
                self.handler.load(
                    list(
                        filter(
                            lambda x: x.endswith(self._meta_postfix),
                            blobs_
                        )
                    )
                ).values()
            )
        )
    
    def evaluate(self, id: str, request: EvaluateDatabaseRequestResponse) -> EvaluateDatabaseRequestResponse:

        model = self.load_model(id)
        return EvaluateDatabaseRequestResponse(
            interpretations=list(
                map(
                    lambda solution: SearchSolution(
                        variables=list(
                            starmap(
                                lambda k,v: SearchSolutionVariable(
                                    id=k, 
                                    bounds=Bounds(
                                        lower=int(v.real), 
                                        upper=int(v.imag)
                                    ),
                                    alias=model.id_to_alias(k)
                                ),
                                solution.items()
                            )
                        )
                    ),
                    map(
                        model.propagate,
                        map(
                            lambda interpretation: dict(
                                map(
                                    lambda variables: (
                                        variables.id, 
                                        complex(
                                            variables.bounds.lower, 
                                            variables.bounds.upper,
                                        )
                                    ),
                                    interpretation.variables
                                )
                            ),
                            request.interpretations
                        )
                    )
                )
            )
        )

    def search(self, id: str, request: SearchDatabaseRequest) -> List[SearchSolution]:
        
        model = self.load_model(id)
        if isinstance(request.suchthat.composite, str):
            suchthat_id = request.suchthat.composite
        else:
            suchthat_id = DatabaseService.set_composite(
                request.suchthat.composite,
                model
            )
            model.compile()
        
        return list(
                map(
                    lambda solution: SearchSolution(
                        variables=list(
                            starmap(
                                lambda k,v: SearchSolutionVariable(
                                    id=k, 
                                    bounds=Bounds(
                                        lower=int(v.real), 
                                        upper=int(v.imag)
                                    ),
                                    alias=model.id_to_alias(k)
                                ),
                                solution.items()
                            )
                        )
                    ),
                    model.solve(
                        objectives=list(
                            map(
                                lambda obj: dict(
                                    starmap(
                                        lambda k, v: (
                                            k,
                                            v * (-1 if obj.direction == SearchDirection.minimize else 1)
                                        ),
                                        filter(
                                            lambda x: x[0] in model.ids,
                                            obj.objective.items()
                                        )
                                    )
                                ),
                                request.objectives, 
                            )
                        ),
                        assume={suchthat_id: complex(request.suchthat.equals.lower, request.suchthat.equals.upper)}, 
                        solver=Solver[SolverType.default.value],
                    )
                )
            )
    
    def delete(self, id: str):

        meta, model = self._rawload(id)
        try:
            for k in [self._name_meta(id), self._name_model(id)]:
                self.handler.delete(k)
        except Exception as e:

            # Rollback deletion if something goes wrong
            try:
                for k, data in zip([self._name_meta(meta.id), self._name_model(meta.id)], map(self._to_bytes, [meta, model])):
                    self.handler.save(k, data)
            except Exception as _e:
                self.logger.critical(f"Rollback failed: ({id}) {_e}")

            self.logger.error(f"Error deleting database: ({id}) {e}")
            raise Exception("Something went wrong")
    
    def update(self, id: str, propositions: List[PropositionStringUnionType]) -> Database:
        

        try:

            meta, model = self._rawload(id)
            
            # This returns a updated PLDAG model
            updated_model = self.merge(propositions, model)

            # Compile before saving
            updated_model.compile()

            # Save the updated model
            self._rawsave(meta, updated_model)

            return Database(
                meta=self.meta_repr(meta, model),
                propositions=self.from_pldag(updated_model),
            )

        except BytesNotFoundException:
            raise DatabaseNotFoundException(id)
        
    def remove_propositions(self, id: str, proposition_ids: List[str]) -> Database:
        
        try:

            meta, model = self._rawload(id)

            # Delete the propositions
            for proposition in proposition_ids:
                model.delete(proposition)

            # Compile before saving
            model.compile()

            # Save the updated model
            self.handler.save(self._name_model(id), self._to_bytes(model))

            return Database(
                meta=self.meta_repr(meta, model),
                propositions=self.from_pldag(model),
            )

        except BytesNotFoundException:
            raise DatabaseNotFoundException(id)
        
    @staticmethod
    def from_pldag(model: PLDAG) -> List[PropositionStringUnionType]:
        return list(
            chain(
                map(
                    lambda x: Primitive(
                        id=x, 
                        bounds=Bounds(
                            lower=int(model.get(x)[0].real),
                            upper=int(model.get(x)[0].imag)
                        ),
                        alias=model.id_to_alias(x),
                    ),
                    model.primitives
                ),
                map(
                    lambda x: LinearInequalityComposite(
                        id=x,
                        alias=model.id_to_alias(x),
                        coefficients=list(
                            map(
                                lambda y: Coefficient(
                                    id=y, 
                                    coef=model._amat[
                                        model._row(x), 
                                        model._col(y)
                                    ],
                                ),
                                sorted(model.dependencies(x)),
                            )
                        ),
                        bias=int(model._bvec[model._row(x)].real),
                    ),
                    model.composites
                )
            )
        )
    
    @staticmethod
    def is_composite(x) -> bool:
        return isinstance(x, (LinearInequalityComposite, BinaryComposite, LogicalComposite, ValueComposite))

    @staticmethod
    def set_composite(composite: PropositionStringUnionType, current_model: PLDAG) -> str:

        _id = None
        if composite.rtype in ['and', 'or', 'xor', 'not'] or composite.rtype in ['atleast', 'atmost', 'equal']:
            children = list(
                chain(
                    map(
                        lambda x: DatabaseService.set_composite(x, current_model=current_model), 
                        filter(
                            DatabaseService.is_composite, 
                            composite.inputs
                        )
                    ),
                    filter(lambda x: issubclass(x.__class__, str), composite.inputs)
                )
            )
            rtype = composite.rtype.value
            alias = composite.alias
            if rtype == 'and':
                _id = current_model.set_and(children, alias=alias)
            elif rtype == 'or':
                _id = current_model.set_or(children, alias=alias)
            elif rtype == 'xor':
                _id = current_model.set_xor(children, alias=alias)
            elif rtype == 'not':
                _id = current_model.set_not(children, alias=alias)

            if rtype in ['atleast', 'atmost', 'equal']:
                value = composite.value
                if value is None:
                    raise SetCompositeException("Composite `value` missing", composite)
                
                if rtype == 'atleast':
                    _id = current_model.set_atleast(children, value, alias=alias)
                elif rtype == 'atmost':
                    _id = current_model.set_atmost(children, value, alias=alias)
                elif rtype == 'equal':
                    _id = current_model.set_equal(children, value, alias=alias)

        elif rtype in ['equiv', 'imply']:
            lhs = DatabaseService.set_composite(composite.lhsInput, current_model=current_model) if DatabaseService.is_composite(composite.lhsInput) else composite.lhsInput
            rhs = DatabaseService.set_composite(composite.rhsInput, current_model=current_model) if DatabaseService.is_composite(composite.rhsInput) else composite.rhsInput
            if not lhs or not rhs:
                raise SetCompositeException("Composite `lhs` or `rhs` missing", composite)
            
            alias = composite.alias
            if rtype == 'equiv':
                _id = current_model.set_equiv(lhs, rhs, alias=alias)
            elif rtype == 'imply':
                _id = current_model.set_imply(lhs, rhs, alias=alias)

        elif rtype == 'lineq':
            try:
                coefficients = list(
                    map(
                        lambda x: (
                            DatabaseService.set_composite(x.id, current_model=current_model) if DatabaseService.is_composite(x.id) else x.id,
                            x.coef,
                        ),
                        composite.coefficients
                    )
                )
            except Exception:
                raise SetCompositeException("Invalid `coefficients`. Should be a list of {'coef': int, 'id': str}", composite)
            
            bias = composite.bias
            alias = composite.alias

            if not coefficients:
                raise SetCompositeException("Composite `coefficients` missing", composite)
            
            if bias is None or not isinstance(bias, int):
                raise SetCompositeException("Composite `bias` missing or invalid (should be int)", composite)
            
            _id = current_model.set_gelineq(coefficients, bias, alias=alias)

        return _id

    def merge(self, propositions: List[PropositionStringUnionType], current_model: PLDAG) -> PLDAG:
                
        # Update the model
        for primitive in filter(lambda x: x.ptype == PropositionType.primitive, propositions):
            current_model.set_primitive(
                id=primitive.id, 
                bound=complex(
                    primitive.bounds.lower, 
                    primitive.bounds.upper,
                )
            )
        
        for composite in filter(lambda x: x.ptype == PropositionType.composite, propositions):
            self.set_composite(composite, current_model)
        
        return current_model
