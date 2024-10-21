# schemas.py

from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from api.models.system import Proposition, Bounds
from pldag import PLDAG
from itertools import chain
from enum import Enum

class Version(BaseModel):
    
    hash: str
    database_id: int
    data: bytes

    parent_hash: Optional[str] = None
    message: Optional[str] = None
    created_at: datetime = datetime.now()

    class Config:
        orm_mode = True

class VersionCreate(BaseModel):
    propositions: List[Proposition]
    parent_hash: Optional[str] = None
    message: Optional[str] = None

class VersionCreateFromBytes(BaseModel):
    data: str # Base64 encoded
    parent_hash: Optional[str] = None
    message: Optional[str] = None

class DatabaseBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatabaseCreate(DatabaseBase):
    pass

class Database(DatabaseBase):
    id: int
    created_at: datetime
    versions: List[Version] = []

    class Config:
        orm_mode = True

class VersionResponse(BaseModel):
    hash: str
    database_id: int
    parent_hash: Optional[str] = None
    message: Optional[str] = None
    created_at: datetime

    class Config:
        orm_mode = True

class DatabaseResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    versions: List[VersionResponse] = []

    class Config:
        orm_mode = True

class Edge(BaseModel):
    source: str
    target: str

class Node(BaseModel):
    
    id: str
    type: str
    label: str
    alias: Optional[str] = None
    bound: Dict[str, int]
    properties: Dict[str, str]
    bias: Optional[int] = None
    children: Optional[List[str]] = None
    coefficients: Optional[List[int]] = None

class Graph(BaseModel):

    edges: List[Edge]
    nodes: List[Node]
    version: VersionResponse 

    @staticmethod
    def from_model(version: VersionResponse, model: PLDAG, solution: Dict[str, complex] = {}) -> Dict[str, List]:
        return Graph(
            version=version,
            edges=Graph.to_edges(model),
            nodes=Graph.to_nodes(model, solution)
        )
    
    @staticmethod
    def to_edges(model: PLDAG) -> List[Edge]:
        return list(
            chain(
                *map(
                    lambda composite: map(
                        lambda dependency: Edge(
                            source=dependency,
                            target=composite,
                        ),
                        model.dependencies(composite)
                    ),
                    filter(
                        lambda x: not model._svec[model._col(x)],
                        model.composites
                    )
                )
            )
        )

    @staticmethod
    def to_nodes(model: PLDAG, solution: Dict[str, complex] = {}) -> List[Node]:
        return list(
            map(
                lambda x: Node(**{
                    "id": x, 
                    "type": "primitive" if x in model.primitives else "composite",
                    "label": x if x in model.primitives else model._tvec[model._col(x)], 
                    "alias": model.id_to_alias(x),
                    "bound": {
                        "lower": int(solution[x].real if x in solution else model._dvec[model._col(x)].real),
                        "upper": int(solution[x].imag if x in solution else model._dvec[model._col(x)].imag),
                    },
                    "properties": {},
                    "bias": int(model._bvec[model._row(x)].real) if x in model.composites else None,
                    "children": model.dependencies(x) if x in model.composites else None,
                    "coefficients": model._amat[model._row(x)][model._amat[model._row(x)] != 0].tolist() if x in model.composites else None,
                }), 
                filter(lambda x: not model._svec[model._col(x)], model._imap)
            )
        )
    
class SolverAssume(BaseModel):

    proposition: Proposition
    bounds: Bounds

class SolverDirection(str, Enum):
    
    maximize = "maximize"
    minimize = "minimize"
    
class SolverProblemRequest(BaseModel):

    objectives: List[Dict[str, int]]
    assume: Optional[SolverAssume] = None
    direction: SolverDirection = SolverDirection.maximize

class SolutionResponse(BaseModel):

    solution:   Optional[Dict[str, int]]    = None
    error:      Optional[str]               = None

class SolverProblemResponse(BaseModel):

    solution_responses: List[SolutionResponse]

class ToolsSearchModel(BaseModel):

    model: List[Proposition]
    problem: SolverProblemRequest