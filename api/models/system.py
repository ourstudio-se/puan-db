import enum

from pydantic import BaseModel
from typing import List, Optional, Union, Dict
from hashlib import sha256

class CreateDatabaseRequest(BaseModel):
    name: str
    description: Optional[str] = None

class CreateDatabaseResponse(BaseModel):
    id:     str

class DeleteDatabaseRequest(BaseModel):
    id: str

class PropositionType(str, enum.Enum):
    primitive = "primitive"
    composite = "composite"

class Bounds(BaseModel):
    lower: int = 0
    upper: int = 1

class Primitive(BaseModel):
    id: str
    ptype: PropositionType = PropositionType.primitive
    bounds: Bounds = Bounds()

PropositionStringUnionType = Union[str, Primitive, "BinaryComposite", "LogicalComposite", "ValueComposite", "LinearInequalityComposite"]

class Coefficient(BaseModel):
    id: PropositionStringUnionType
    coef: int

class Composite(BaseModel):
    id: Optional[str] = None
    ptype: PropositionType = PropositionType.composite

class LinearInequalityComposite(Composite):
    coefficients: List[Coefficient]
    bias: int
    alias: Optional[str] = None

class LogicalCompositeType(str, enum.Enum):
    and_ = "and"
    or_ = "or"
    xor_ = "xor"
    not_ = "not"

class LogicalComposite(Composite):
    rtype: LogicalCompositeType
    inputs: List[PropositionStringUnionType]
    alias: Optional[str] = None

class ValueCompositeType(str, enum.Enum):
    atleast = "atleast"
    atmost = "atmost"
    equal = "equal"

class ValueComposite(Composite):
    rtype: ValueCompositeType
    inputs: List[PropositionStringUnionType]
    value: int
    alias: Optional[str] = None

class BinaryCompositeType(str, enum.Enum):
    equiv = "equiv"
    imply = "imply"

class BinaryComposite(Composite):
    rtype: BinaryCompositeType
    lhsInput: PropositionStringUnionType
    rhsInput: PropositionStringUnionType
    alias: Optional[str] = None

class DatabaseMeta(BaseModel):

    id:             str
    name:           str
    createdAt:      str
    
    parentId:       Optional[str] = None
    description:    Optional[str] = None

class Database(BaseModel):

    meta:           DatabaseMeta
    propositions:   List[PropositionStringUnionType]

class DatabaseBranchPointer(BaseModel):

    id: str
    name: str

    def sha256(self) -> str:
        return sha256(self.name.encode()).hexdigest()

class RemovePropositionsRequest(BaseModel):
    propositions: List[str]

class RemovePropositionsResponse(BaseModel):
    result: Dict[str, bool]