# schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from api.models.system import Proposition, Bounds
from pldag import PLDAG
from itertools import chain, starmap
from enum import Enum

class DatabaseBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatabaseCreate(DatabaseBase):
    propositions: List[Proposition]

class Database(DatabaseBase):
    created_at: datetime
    data_hash: str

class DatabaseResponse(BaseModel):
    name: str
    description: Optional[str] = None
    created_at: datetime

class DatabaseCreateFromBytes(DatabaseBase):
    data: str

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
    database: Database 

    @staticmethod
    def from_model(database: Database, model: PLDAG, solution: Dict[str, complex] = {}) -> Dict[str, List]:
        return Graph(
            database=database,
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

class EvaluateRequest(BaseModel):

    variables: Dict[str, Bounds]

    def to_complex(self) -> dict:
        return dict(
            starmap(
                lambda k,v: (
                    k,
                    complex(v.lower, v.upper)
                ),
                self.variables.items()
            )
        )

class EvaluateResponse(BaseModel):

    variables: Dict[str, Bounds]

    @staticmethod
    def from_complex(complex_values: Dict[str, complex]) -> "EvaluateResponse":
        return EvaluateResponse(
            variables=dict(
                starmap(
                    lambda k,v: (
                        k,
                        Bounds(lower=v.real, upper=v.imag)
                    ),
                    complex_values.items()
                )
            )
        )