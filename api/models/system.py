import enum

from pydantic import BaseModel
from typing import List, Optional, Union
from pldag import PLDAG

class PropositionType(str, enum.Enum):
    primitive = "primitive"
    composite = "composite"

class Bounds(BaseModel):
    lower: int = 0
    upper: int = 1

    def to_complex(self) -> complex:
        return complex(self.lower, self.upper)

class Primitive(BaseModel):
    id: str
    ptype: PropositionType = PropositionType.primitive
    bounds: Bounds = Bounds()

    def set_model(self, model: PLDAG) -> str:
        return model.set_primitive(self.id, complex(self.bounds.lower, self.bounds.upper))

class Coefficient(BaseModel):
    id: "PropositionStringUnionType"
    coef: int

class LinearInequalityComposite(BaseModel):
    coefficients: List[Coefficient]
    bias: int
    alias: Optional[str] = None

    def set_model(self, model: PLDAG) -> str:
        references = [coef.id if isinstance(coef.id, str) else coef.id.set_model(model) for coef in self.coefficients]
        values = [coef.coef for coef in self.coefficients]
        return model.set_gelineq(dict(zip(references, values), self.bias, self.alias))

class LogicalCompositeType(str, enum.Enum):
    or_     = "or"
    nor_    = "nor"
    and_    = "and"
    nand_   = "nand"
    xor_    = "xor"
    nxor_   = "nxor"
    not_    = "not"

class LogicalComposite(BaseModel):
    rtype: LogicalCompositeType
    inputs: List["PropositionStringUnionType"]
    alias: Optional[str] = None

    def set_model(self, model: PLDAG) -> str:
        if self.rtype == LogicalCompositeType.or_:
            fn = model.set_or
        elif self.rtype == LogicalCompositeType.nor_:
            fn = lambda x: model.set_not(model.set_or(x))
        elif self.rtype == LogicalCompositeType.and_:
            fn = model.set_and
        elif self.rtype == LogicalCompositeType.nand_:
            fn = model.set_nand
        elif self.rtype == LogicalCompositeType.xor_:
            fn = model.set_xor
        elif self.rtype == LogicalCompositeType.nxor_:
            fn = model.set_xnor
        else:
            raise ValueError(f"Unknown value composite type: {self}")
        
        return fn([inp if isinstance(inp, str) else inp.set_model(model) for inp in self.inputs], self.alias)

class ValueCompositeType(str, enum.Enum):
    atleast = "atleast"
    atmost = "atmost"
    equal = "equal"

class ValueComposite(BaseModel):
    rtype: ValueCompositeType
    inputs: List["PropositionStringUnionType"]
    value: int
    alias: Optional[str] = None

    def set_model(self, model: PLDAG) -> str:
        if self.rtype == ValueCompositeType.atleast:
            fn = model.set_atleast
        elif self.rtype == ValueCompositeType.atmost:
            fn = model.set_atmost
        elif self.rtype == ValueCompositeType.equal:
            fn = model.set_equal
        else:
            raise ValueError(f"Unknown value composite type: {self.rtype}")

        return fn([inp if isinstance(inp, str) else inp.set_model(model) for inp in self.inputs], self.value, self.alias)

class BinaryCompositeType(str, enum.Enum):
    equiv = "equiv"
    imply = "imply"

class BinaryComposite(BaseModel):
    rtype: BinaryCompositeType
    lhsInput: "PropositionStringUnionType"
    rhsInput: "PropositionStringUnionType"
    alias: Optional[str] = None

    def set_model(self, model: PLDAG) -> str:

        lhs = self.lhsInput if isinstance(self.lhsInput, str) else self.lhsInput.set_model(model)
        rhs = self.rhsInput if isinstance(self.rhsInput, str) else self.rhsInput.set_model(model)
        
        if self.rtype == BinaryCompositeType.equiv:
            fn = model.set_equiv
        elif self.rtype == BinaryCompositeType.imply:
            fn = model.set_imply
        else:
            raise ValueError(f"Unknown binary composite type: {self.rtype}")

        return fn(lhs, rhs)

Proposition = Union[Primitive, BinaryComposite, LogicalComposite, ValueComposite, LinearInequalityComposite]
PropositionStringUnionType = Union[str, Proposition]