from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Solver(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GLPK: _ClassVar[Solver]
GLPK: Solver

class Bound(_message.Message):
    __slots__ = ("lower", "upper")
    LOWER_FIELD_NUMBER: _ClassVar[int]
    UPPER_FIELD_NUMBER: _ClassVar[int]
    lower: int
    upper: int
    def __init__(self, lower: _Optional[int] = ..., upper: _Optional[int] = ...) -> None: ...

class Primitive(_message.Message):
    __slots__ = ("id", "bound")
    ID_FIELD_NUMBER: _ClassVar[int]
    BOUND_FIELD_NUMBER: _ClassVar[int]
    id: str
    bound: Bound
    def __init__(self, id: _Optional[str] = ..., bound: _Optional[_Union[Bound, _Mapping]] = ...) -> None: ...

class Composite(_message.Message):
    __slots__ = ("id", "references", "bias", "negated", "alias")
    ID_FIELD_NUMBER: _ClassVar[int]
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    BIAS_FIELD_NUMBER: _ClassVar[int]
    NEGATED_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    id: str
    references: _containers.RepeatedScalarFieldContainer[str]
    bias: int
    negated: bool
    alias: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., references: _Optional[_Iterable[str]] = ..., bias: _Optional[int] = ..., negated: bool = ..., alias: _Optional[_Iterable[str]] = ...) -> None: ...

class Proposition(_message.Message):
    __slots__ = ("AT_LEAST", "AT_MOST", "AND", "OR", "XOR", "NOT", "IMPLY")
    AT_LEAST_FIELD_NUMBER: _ClassVar[int]
    AT_MOST_FIELD_NUMBER: _ClassVar[int]
    AND_FIELD_NUMBER: _ClassVar[int]
    OR_FIELD_NUMBER: _ClassVar[int]
    XOR_FIELD_NUMBER: _ClassVar[int]
    NOT_FIELD_NUMBER: _ClassVar[int]
    IMPLY_FIELD_NUMBER: _ClassVar[int]
    AT_LEAST: AtLeast
    AT_MOST: AtMost
    AND: And
    OR: Or
    XOR: Xor
    NOT: Not
    IMPLY: Imply
    def __init__(self, AT_LEAST: _Optional[_Union[AtLeast, _Mapping]] = ..., AT_MOST: _Optional[_Union[AtMost, _Mapping]] = ..., AND: _Optional[_Union[And, _Mapping]] = ..., OR: _Optional[_Union[Or, _Mapping]] = ..., XOR: _Optional[_Union[Xor, _Mapping]] = ..., NOT: _Optional[_Union[Not, _Mapping]] = ..., IMPLY: _Optional[_Union[Imply, _Mapping]] = ...) -> None: ...

class SetPrimitivesRequest(_message.Message):
    __slots__ = ("ids", "bound")
    IDS_FIELD_NUMBER: _ClassVar[int]
    BOUND_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    bound: Bound
    def __init__(self, ids: _Optional[_Iterable[str]] = ..., bound: _Optional[_Union[Bound, _Mapping]] = ...) -> None: ...

class Reference(_message.Message):
    __slots__ = ("id", "proposition")
    ID_FIELD_NUMBER: _ClassVar[int]
    PROPOSITION_FIELD_NUMBER: _ClassVar[int]
    id: str
    proposition: Proposition
    def __init__(self, id: _Optional[str] = ..., proposition: _Optional[_Union[Proposition, _Mapping]] = ...) -> None: ...

class References(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ids: _Optional[_Iterable[str]] = ...) -> None: ...

class AtLeast(_message.Message):
    __slots__ = ("references", "value", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedCompositeFieldContainer[Reference]
    value: int
    alias: str
    def __init__(self, references: _Optional[_Iterable[_Union[Reference, _Mapping]]] = ..., value: _Optional[int] = ..., alias: _Optional[str] = ...) -> None: ...

class AtMost(_message.Message):
    __slots__ = ("references", "value", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedCompositeFieldContainer[Reference]
    value: int
    alias: str
    def __init__(self, references: _Optional[_Iterable[_Union[Reference, _Mapping]]] = ..., value: _Optional[int] = ..., alias: _Optional[str] = ...) -> None: ...

class And(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedCompositeFieldContainer[Reference]
    alias: str
    def __init__(self, references: _Optional[_Iterable[_Union[Reference, _Mapping]]] = ..., alias: _Optional[str] = ...) -> None: ...

class Or(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedCompositeFieldContainer[Reference]
    alias: str
    def __init__(self, references: _Optional[_Iterable[_Union[Reference, _Mapping]]] = ..., alias: _Optional[str] = ...) -> None: ...

class Xor(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedCompositeFieldContainer[Reference]
    alias: str
    def __init__(self, references: _Optional[_Iterable[_Union[Reference, _Mapping]]] = ..., alias: _Optional[str] = ...) -> None: ...

class Not(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedCompositeFieldContainer[Reference]
    alias: str
    def __init__(self, references: _Optional[_Iterable[_Union[Reference, _Mapping]]] = ..., alias: _Optional[str] = ...) -> None: ...

class Imply(_message.Message):
    __slots__ = ("condition", "consequence", "alias")
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    CONSEQUENCE_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    condition: Reference
    consequence: Reference
    alias: str
    def __init__(self, condition: _Optional[_Union[Reference, _Mapping]] = ..., consequence: _Optional[_Union[Reference, _Mapping]] = ..., alias: _Optional[str] = ...) -> None: ...

class Variable(_message.Message):
    __slots__ = ("id", "bound")
    ID_FIELD_NUMBER: _ClassVar[int]
    BOUND_FIELD_NUMBER: _ClassVar[int]
    id: str
    bound: Bound
    def __init__(self, id: _Optional[str] = ..., bound: _Optional[_Union[Bound, _Mapping]] = ...) -> None: ...

class FixedVariable(_message.Message):
    __slots__ = ("id", "value")
    ID_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    id: str
    value: int
    def __init__(self, id: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...

class Interpretation(_message.Message):
    __slots__ = ("variables",)
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    variables: _containers.RepeatedCompositeFieldContainer[Variable]
    def __init__(self, variables: _Optional[_Iterable[_Union[Variable, _Mapping]]] = ...) -> None: ...

class Objective(_message.Message):
    __slots__ = ("fixed_variables",)
    FIXED_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    fixed_variables: _containers.RepeatedCompositeFieldContainer[FixedVariable]
    def __init__(self, fixed_variables: _Optional[_Iterable[_Union[FixedVariable, _Mapping]]] = ...) -> None: ...

class Solution(_message.Message):
    __slots__ = ("fixed_variables",)
    FIXED_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    fixed_variables: _containers.RepeatedCompositeFieldContainer[FixedVariable]
    def __init__(self, fixed_variables: _Optional[_Iterable[_Union[FixedVariable, _Mapping]]] = ...) -> None: ...

class SolveRequest(_message.Message):
    __slots__ = ("objectives", "fix", "solver")
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    FIX_FIELD_NUMBER: _ClassVar[int]
    SOLVER_FIELD_NUMBER: _ClassVar[int]
    objectives: _containers.RepeatedCompositeFieldContainer[Objective]
    fix: _containers.RepeatedCompositeFieldContainer[FixedVariable]
    solver: Solver
    def __init__(self, objectives: _Optional[_Iterable[_Union[Objective, _Mapping]]] = ..., fix: _Optional[_Iterable[_Union[FixedVariable, _Mapping]]] = ..., solver: _Optional[_Union[Solver, str]] = ...) -> None: ...

class SolveResponse(_message.Message):
    __slots__ = ("solutions", "error")
    SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    solutions: _containers.RepeatedCompositeFieldContainer[Solution]
    error: str
    def __init__(self, solutions: _Optional[_Iterable[_Union[Solution, _Mapping]]] = ..., error: _Optional[str] = ...) -> None: ...

class SetResponse(_message.Message):
    __slots__ = ("id", "error")
    ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    id: str
    error: str
    def __init__(self, id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class IDRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class IDResponse(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class IDsResponse(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ids: _Optional[_Iterable[str]] = ...) -> None: ...

class AliasRequest(_message.Message):
    __slots__ = ("alias",)
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    alias: str
    def __init__(self, alias: _Optional[str] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("primitives", "composites")
    PRIMITIVES_FIELD_NUMBER: _ClassVar[int]
    COMPOSITES_FIELD_NUMBER: _ClassVar[int]
    primitives: _containers.RepeatedCompositeFieldContainer[Primitive]
    composites: _containers.RepeatedCompositeFieldContainer[Composite]
    def __init__(self, primitives: _Optional[_Iterable[_Union[Primitive, _Mapping]]] = ..., composites: _Optional[_Iterable[_Union[Composite, _Mapping]]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Primitives(_message.Message):
    __slots__ = ("primitives",)
    PRIMITIVES_FIELD_NUMBER: _ClassVar[int]
    primitives: _containers.RepeatedCompositeFieldContainer[Primitive]
    def __init__(self, primitives: _Optional[_Iterable[_Union[Primitive, _Mapping]]] = ...) -> None: ...

class Composites(_message.Message):
    __slots__ = ("composites",)
    COMPOSITES_FIELD_NUMBER: _ClassVar[int]
    composites: _containers.RepeatedCompositeFieldContainer[Composite]
    def __init__(self, composites: _Optional[_Iterable[_Union[Composite, _Mapping]]] = ...) -> None: ...

class MetaInformationResponse(_message.Message):
    __slots__ = ("nrows", "ncols", "ncombs_lb", "ncombs_ub")
    NROWS_FIELD_NUMBER: _ClassVar[int]
    NCOLS_FIELD_NUMBER: _ClassVar[int]
    NCOMBS_LB_FIELD_NUMBER: _ClassVar[int]
    NCOMBS_UB_FIELD_NUMBER: _ClassVar[int]
    nrows: int
    ncols: int
    ncombs_lb: int
    ncombs_ub: int
    def __init__(self, nrows: _Optional[int] = ..., ncols: _Optional[int] = ..., ncombs_lb: _Optional[int] = ..., ncombs_ub: _Optional[int] = ...) -> None: ...
