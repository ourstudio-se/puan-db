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

class SetPrimitivesRequest(_message.Message):
    __slots__ = ("ids", "bound")
    IDS_FIELD_NUMBER: _ClassVar[int]
    BOUND_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    bound: Bound
    def __init__(self, ids: _Optional[_Iterable[str]] = ..., bound: _Optional[_Union[Bound, _Mapping]] = ...) -> None: ...

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
    references: _containers.RepeatedScalarFieldContainer[str]
    value: int
    alias: str
    def __init__(self, references: _Optional[_Iterable[str]] = ..., value: _Optional[int] = ..., alias: _Optional[str] = ...) -> None: ...

class AtMost(_message.Message):
    __slots__ = ("references", "value", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedScalarFieldContainer[str]
    value: int
    alias: str
    def __init__(self, references: _Optional[_Iterable[str]] = ..., value: _Optional[int] = ..., alias: _Optional[str] = ...) -> None: ...

class And(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedScalarFieldContainer[str]
    alias: str
    def __init__(self, references: _Optional[_Iterable[str]] = ..., alias: _Optional[str] = ...) -> None: ...

class Or(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedScalarFieldContainer[str]
    alias: str
    def __init__(self, references: _Optional[_Iterable[str]] = ..., alias: _Optional[str] = ...) -> None: ...

class Xor(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedScalarFieldContainer[str]
    alias: str
    def __init__(self, references: _Optional[_Iterable[str]] = ..., alias: _Optional[str] = ...) -> None: ...

class Not(_message.Message):
    __slots__ = ("references", "alias")
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    references: _containers.RepeatedScalarFieldContainer[str]
    alias: str
    def __init__(self, references: _Optional[_Iterable[str]] = ..., alias: _Optional[str] = ...) -> None: ...

class Imply(_message.Message):
    __slots__ = ("condition", "consequence", "alias")
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    CONSEQUENCE_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    condition: str
    consequence: str
    alias: str
    def __init__(self, condition: _Optional[str] = ..., consequence: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class Equivalent(_message.Message):
    __slots__ = ("lhs", "rhs", "alias")
    LHS_FIELD_NUMBER: _ClassVar[int]
    RHS_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    lhs: str
    rhs: str
    alias: str
    def __init__(self, lhs: _Optional[str] = ..., rhs: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("variables",)
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    variables: _containers.RepeatedCompositeFieldContainer[FixedVariable]
    def __init__(self, variables: _Optional[_Iterable[_Union[FixedVariable, _Mapping]]] = ...) -> None: ...

class SolveRequest(_message.Message):
    __slots__ = ("objectives", "assume", "solver")
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    ASSUME_FIELD_NUMBER: _ClassVar[int]
    SOLVER_FIELD_NUMBER: _ClassVar[int]
    objectives: _containers.RepeatedCompositeFieldContainer[Objective]
    assume: Interpretation
    solver: Solver
    def __init__(self, objectives: _Optional[_Iterable[_Union[Objective, _Mapping]]] = ..., assume: _Optional[_Union[Interpretation, _Mapping]] = ..., solver: _Optional[_Union[Solver, str]] = ...) -> None: ...

class SolveResponse(_message.Message):
    __slots__ = ("solutions", "error")
    SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    solutions: _containers.RepeatedCompositeFieldContainer[Interpretation]
    error: str
    def __init__(self, solutions: _Optional[_Iterable[_Union[Interpretation, _Mapping]]] = ..., error: _Optional[str] = ...) -> None: ...

class SetResponse(_message.Message):
    __slots__ = ("id", "error")
    ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    id: str
    error: str
    def __init__(self, id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class BooleanSetResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

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

class SelectModelRequest(_message.Message):
    __slots__ = ("id", "password")
    ID_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    id: str
    password: str
    def __init__(self, id: _Optional[str] = ..., password: _Optional[str] = ...) -> None: ...

class CreateModelRequest(_message.Message):
    __slots__ = ("id", "password", "name")
    ID_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    id: str
    password: str
    name: str
    def __init__(self, id: _Optional[str] = ..., password: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ModelResponse(_message.Message):
    __slots__ = ("success", "token", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    token: str
    error: str
    def __init__(self, success: bool = ..., token: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class CutVariableMapping(_message.Message):
    __slots__ = ("from_id", "to_id")
    FROM_ID_FIELD_NUMBER: _ClassVar[int]
    TO_ID_FIELD_NUMBER: _ClassVar[int]
    from_id: str
    to_id: str
    def __init__(self, from_id: _Optional[str] = ..., to_id: _Optional[str] = ...) -> None: ...

class CutRequest(_message.Message):
    __slots__ = ("cut_ids",)
    CUT_IDS_FIELD_NUMBER: _ClassVar[int]
    cut_ids: _containers.RepeatedCompositeFieldContainer[CutVariableMapping]
    def __init__(self, cut_ids: _Optional[_Iterable[_Union[CutVariableMapping, _Mapping]]] = ...) -> None: ...

class SubRequest(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ids: _Optional[_Iterable[str]] = ...) -> None: ...

class CutSubRequest(_message.Message):
    __slots__ = ("cut", "sub")
    CUT_FIELD_NUMBER: _ClassVar[int]
    SUB_FIELD_NUMBER: _ClassVar[int]
    cut: CutRequest
    sub: SubRequest
    def __init__(self, cut: _Optional[_Union[CutRequest, _Mapping]] = ..., sub: _Optional[_Union[SubRequest, _Mapping]] = ...) -> None: ...
