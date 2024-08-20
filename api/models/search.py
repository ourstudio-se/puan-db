import enum
from pydantic import BaseModel
from typing import List, Optional, Dict

from api.models.system import (
    Bounds, 
    PropositionStringUnionType, 
    
    # Let the following import be uncommented, even though they're "not used"
    # They are indirectly used and un-commenting them will cause issues
    BinaryComposite, 
    LogicalComposite, 
    ValueComposite, 
    LinearInequalityComposite
)

class SolverType(str, enum.Enum):
    default = "GLPK"
    brute = "brute"
    cplex = "cplex"
    gurobi = "gurobi"

class SearchDirection(str, enum.Enum):
    minimize = "minimize"
    maximize = "maximize"

class SearchObjectiveField(BaseModel):
    direction: SearchDirection
    objective: Dict[str, int]

class SearchSuchThatField(BaseModel):
    composite: PropositionStringUnionType
    equals: Bounds

class SearchDatabaseRequest(BaseModel):
    objectives: List[SearchObjectiveField]
    suchthat: SearchSuchThatField
    solver: Optional[SolverType] = "default"

class SearchSolutionVariable(BaseModel):
    id: str
    bounds: Bounds
    alias: Optional[str] = None

class SearchSolution(BaseModel):
    variables: List[SearchSolutionVariable]

class SearchDatabaseResponse(BaseModel):
    solutions: List[SearchSolution]

class EvaluateDatabaseRequestResponse(BaseModel):
    interpretations: List[SearchSolution]