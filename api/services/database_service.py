from dataclasses import dataclass
from hashlib import sha1
from gzip import compress, decompress
from itertools import starmap, chain
from pickle import loads, dumps, HIGHEST_PROTOCOL
from pydantic import BaseModel, Field, computed_field
from pldag import (
    PLDAG, 
    Solver, 
    CompilationSetting, 
    MissingVariableException,
)
from redis import Redis
from typing import List, Union, Optional, Dict
from time import time, strftime, localtime
from starlette.config import Config

from api.models.system import (
    Primitive,
    BinaryComposite,
    LogicalComposite,
    ValueComposite,
    LinearInequalityComposite,
    Coefficient,
)
from api.models.search import (
    SearchDatabaseRequest, 
    SearchDatabaseResponse,
    SearchSolution,
    SearchSolutionVariable,
    Bounds,
    SearchDirection,
    SolverType,
    EvaluateDatabaseRequest
)
from api.services.tools import set_composite, merge_composites

Composite = Union[Primitive, BinaryComposite, LogicalComposite, ValueComposite, LinearInequalityComposite]
compress_dump   = lambda x: compress(dumps(x, protocol=HIGHEST_PROTOCOL), mtime=0)
decompress_load = lambda x: loads(decompress(x))
timestamp = lambda: strftime('%Y-%m-%d %H:%M:%S', localtime(time()))
config = Config(".env")

class Commit(BaseModel):
    
    database_name:      str
    date:               str
    parents:            List[str]

    # Raw data is stored but not exposed
    data_bytes:         bytes = Field(None, exclude=True)

    @computed_field
    @property
    def id(self) -> str:
        return sha1(self.data_bytes + self.database_name.encode()).hexdigest()
    
    @computed_field
    @property
    def data(self) -> List[Composite]:
        return decompress_load(self.data_bytes)
    
    def empty(database_name: str) -> "Commit":
        return Commit(
            database_name=database_name,
            date=timestamp(),
            parents=[],
            data_bytes=compress_dump([]),
        )
    
    def grandulate(self) -> "Commit":
        """
            Runs through PLDAG to verify and set the composite ID's
        """
        return Commit(
            database_name=self.database_name,
            date=self.date,
            parents=self.parents,
            data_bytes=compress_dump(
                Commit.from_pldag(
                    self.to_pldag()
                )
            ),
        )
    
    def to_pldag(self) -> PLDAG:
        model = PLDAG(CompilationSetting.ON_DEMAND)
        merge_composites(self.data, model)
        try:
            model.compile()
        except MissingVariableException as e:
            raise VariableMissingException(e)
        return model
    
    @staticmethod
    def from_pldag(model: PLDAG) -> List[Composite]:
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

class Branch(BaseModel):
    name:       str
    commit_id:  str

    def key(self, database: str) -> str:
        return sha1((database + self.name).encode()).hexdigest()

class Database(BaseModel):
    name:       str
    branches:   List[Branch]

    @staticmethod
    def key(name: str) -> str:
        return sha1(name.encode()).hexdigest() + "-database"
    
class DatabaseExistsException(Exception):
    pass
    
class BranchExistsException(Exception):
    pass
    
class DatabaseDoesNotExistsException(Exception):
    pass
    
class BranchDoesNotExistsException(Exception):
    pass
    
class CommitDoesNotExistsException(Exception):
    pass

class VariableMissingException(Exception):
    pass

@dataclass
class DatabaseService:

    client: Redis = Redis(
        host=config('REDIS_HOST', str, 'localhost'),
        port=config('REDIS_PORT', int, 6379),
        db=config('REDIS_DB', int, 0),
        password=config('REDIS_PASSWORD', str, None),
    )
    
    def get_database(self, database_name: str) -> Database:
        if not self.client.exists(Database.key(database_name)):
            raise DatabaseDoesNotExistsException()
        return decompress_load(self.client.get(Database.key(database_name)))
    
    def get_databases(self) -> List[Database]:
        return [
            decompress_load(self.client.get(key))
            for key in self.client.keys('*-database')
        ]
    
    def get_commit(self, commit_id: str) -> Commit:
        if not self.client.exists(commit_id):
            raise CommitDoesNotExistsException()
        return decompress_load(self.client.get(commit_id))

    def create_database(self, database_name: str):
        if self.client.exists(Database.key(database_name)):
            raise DatabaseExistsException()
        commit = Commit.empty(database_name)
        database = Database(
            name=database_name, 
            branches=[
                Branch(
                    name='main', 
                    commit_id=commit.id,
                )
            ]
        )
        self.client.set(commit.id, compress_dump(commit))
        self.client.set(Database.key(database_name), compress_dump(database))

    def create_branch(self, database_name: str, branch_name: str, from_branch: str = 'main'):
        if not self.client.exists(Database.key(database_name)):
            raise DatabaseDoesNotExistsException()
        
        database = self.get_database(database_name)
        if next(filter(lambda x: x.name == branch_name, database.branches), None) is not None:
            raise BranchExistsException()

        database.branches.append(
            Branch(
                name=branch_name, 
                commit_id=next(
                    filter(
                        lambda x: x.name == from_branch,
                        database.branches,
                    )
                ).commit_id
            )
        )
        self.client.set(Database.key(database_name), compress_dump(database))

    def create_branch_from_commit(self, database_name: str, branch_name: str, from_commit_id: str):
        if not self.client.exists(Database.key(database_name)):
            raise DatabaseDoesNotExistsException()
        
        database = self.get_database(database_name)
        if next(filter(lambda x: x.name == branch_name, database.branches), None) is not None:
            raise BranchExistsException()

        database.branches.append(
            Branch(
                name=branch_name, 
                commit_id=from_commit_id
            )
        )
        self.client.set(Database.key(database_name), compress_dump(database))

    def branch_latest_commit(self, database_name: str, branch_name: str) -> str:
        database = self.get_database(database_name)
        branch_index = next((i for i, branch in enumerate(database.branches) if branch.name == branch_name), None)
        if branch_index is None:
            raise BranchDoesNotExistsException()
        return database.branches[branch_index].commit_id

    def commit(self, database_name: str, branch_name: str, propositions: List[str]) -> str:
        database = self.get_database(database_name)
        branch_index = next((i for i, branch in enumerate(database.branches) if branch.name == branch_name), None)
        if branch_index is None:
            raise BranchDoesNotExistsException()
        
        commit = Commit(
            database_name=database_name,
            date=timestamp(),
            parents=[
                database.branches[branch_index].commit_id
            ],
            data_bytes=compress_dump(propositions)
        )        
        if self.client.get(commit.id) is not None:
            return commit.id
        
        database.branches[branch_index].commit_id = commit.id
        self.client.set(commit.id, compress_dump(commit.grandulate()))
        self.client.set(Database.key(database_name), compress_dump(database))
        return commit.id
    
    def search(self, commit_id: str, search_request: SearchDatabaseRequest) -> SearchDatabaseResponse:

        # Load model from commit
        # First try from cache, otherwise build from propositions in commit
        model_bytes: Optional[bytes] = self.client.get(sha1(commit_id.encode()).hexdigest())
        if model_bytes is None:

            commit_bytes = self.client.get(commit_id)
            if commit_bytes is None:
                raise CommitDoesNotExistsException()
            
            commit: Commit = decompress_load(commit_bytes)
            model = commit.to_pldag()

            # Store model in cache
            self.client.set(sha1(commit_id.encode()).hexdigest(), compress_dump(model))
        else:
            model: PLDAG = decompress_load(model_bytes)

        # Add temporary constraints to model
        if isinstance(search_request.suchthat.composite, str):
            suchthat_id = search_request.suchthat.composite
        else:
            suchthat_id = set_composite(
                search_request.suchthat.composite,
                model
            )
            try:
                model.compile()
            except MissingVariableException as e:
                raise VariableMissingException(e)

        # Check if exact same search has been done before
        search_request_bytes = compress_dump(search_request) + commit_id.encode()
        search_request_id = sha1(search_request_bytes).hexdigest()
        if self.client.exists(search_request_id):
            return decompress_load(self.client.get(search_request_id))

        try:
            result = list(
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
                        # Note we're filtering the objectives to only include variables that are in the model
                        objectives=list(map(lambda objective: dict(filter(lambda kv: kv[0] in model.ids, objective.items())), search_request.objectives)),
                        assume={suchthat_id: complex(search_request.suchthat.equals.lower, search_request.suchthat.equals.upper)}, 
                        solver=Solver[SolverType.default.value.upper()],
                        minimize=search_request.direction == SearchDirection.minimize,
                    )
                )
            )
        except MissingVariableException as e:
            raise VariableMissingException(e)

        response = SearchDatabaseResponse(solutions=result)
        self.client.set(search_request_id, compress_dump(response))
        return response

    def evaluate(self, commit_id: str, evaluate_request: EvaluateDatabaseRequest) -> EvaluateDatabaseRequest:

        # Load model from commit
        # First try from cache, otherwise build from propositions in commit
        model_bytes: Optional[bytes] = self.client.get(sha1(commit_id.encode()).hexdigest())
        if model_bytes is None:

            commit_bytes = self.client.get(commit_id)
            if commit_bytes is None:
                raise CommitDoesNotExistsException()
            
            commit: Commit = decompress_load(commit_bytes)
            model: PLDAG = commit.to_pldag()

            # Store model in cache
            self.client.set(sha1(commit_id.encode()).hexdigest(), compress_dump(model))
        else:
            model: PLDAG = decompress_load(model_bytes)

        return EvaluateDatabaseRequest(
            interpretations=list(
                map(
                    lambda evaluated: SearchSolution(
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
                                evaluated.items()
                            )
                        )
                    ),
                    map(
                        lambda solution: model.propagate(
                            dict(
                                map(
                                    lambda variable: (
                                        variable.id, 
                                        complex(
                                            variable.bounds.lower, 
                                            variable.bounds.upper,
                                        )
                                    ),
                                    filter(
                                        lambda variable: variable.id in model.ids,
                                        solution.variables
                                    )
                                )
                            )
                        ),
                        evaluate_request.interpretations
                    )
                )
            )
        )
    
    def sub_database(self, database: str, old_branch: str, new_branch: str, roots: List[str]) -> str:

        # Get latest commit
        commit_id = self.branch_latest_commit(database, old_branch)
        
        # Load model from commit
        # First try from cache, otherwise build from propositions in commit
        model_bytes: Optional[bytes] = self.client.get(sha1(commit_id.encode()).hexdigest())
        if model_bytes is None:

            commit_bytes = self.client.get(commit_id)
            if commit_bytes is None:
                raise CommitDoesNotExistsException()
            
            commit: Commit = decompress_load(commit_bytes)
            model: PLDAG = commit.to_pldag()

            # Store model in cache
            self.client.set(sha1(commit_id.encode()).hexdigest(), compress_dump(model))
        else:
            model: PLDAG = decompress_load(model_bytes)

        try:
            commit = Commit(
                database_name=database,
                date=timestamp(),
                parents=[commit_id],
                data_bytes=compress_dump(
                    Commit.from_pldag(model.sub(roots))
                )
            )        
            if self.client.get(commit.id) is None:
                self.client.set(commit.id, compress_dump(commit.grandulate()))

            # Create new branch - this will point new_branch to the commit of old_branch
            self.create_branch_from_commit(database, new_branch, commit.id)
            
            return commit.id
        
        except MissingVariableException as e:
            raise VariableMissingException(e)
        
    def cut_database(self, database: str, old_branch: str, new_branch: str, leafs: Dict[str, str]) -> str:

        # Get latest commit
        commit_id = self.branch_latest_commit(database, old_branch)
        
        # Load model from commit
        # First try from cache, otherwise build from propositions in commit
        model_bytes: Optional[bytes] = self.client.get(sha1(commit_id.encode()).hexdigest())
        if model_bytes is None:

            commit_bytes = self.client.get(commit_id)
            if commit_bytes is None:
                raise CommitDoesNotExistsException()
            
            commit: Commit = decompress_load(commit_bytes)
            model: PLDAG = commit.to_pldag()

            # Store model in cache
            self.client.set(sha1(commit_id.encode()).hexdigest(), compress_dump(model))
        else:
            model: PLDAG = decompress_load(model_bytes)

        try:
            commit = Commit(
                database_name=database,
                date=timestamp(),
                parents=[commit_id],
                data_bytes=compress_dump(
                    Commit.from_pldag(model.cut(leafs))
                )
            )        
            if self.client.get(commit.id) is None:
                self.client.set(commit.id, compress_dump(commit.grandulate()))

            # Create new branch - this will point new_branch to the commit of old_branch
            self.create_branch_from_commit(database, new_branch, commit.id)
            
            return commit.id
        
        except MissingVariableException as e:
            raise VariableMissingException(e)
