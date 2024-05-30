from enum import Enum
from typing import Union, Any
from dataclasses import dataclass, field

# An action that will eventually return a string
class STRING_ACTION:
    pass

class ACTION_TYPE(str, Enum):
    SET = "SET"
    GET = "GET"
    DEL = "DEL"
    SUB = "SUB"
    CUT = "CUT"
    ASSUME = "ASSUME"
    REDUCE = "REDUCE"
    PROPAGATE = "PROPAGATE"
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"

class SUB_ACTION_TYPE(str, Enum):
    ATLEAST = "ATLEAST"
    ATMOST = "ATMOST"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"
    EQUAL = "EQUAL"
    IMPLY = "IMPLY"
    EQUIV = "EQUIV"

class KEYWORD(str, Enum):
    SUCHTHAT = "SUCHTHAT"

class DATATYPE(str, Enum):
    PRIMITIVE = "PRIMITIVE"
    COMPOSITE = "COMPOSITE"

@dataclass
class PROPERTIES:
    properties: dict = field(default_factory=dict)

@dataclass
class LIST:
    items: list

class OPERATION(str, Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    AND = "&&"
    OR = "||"

@dataclass
class BOUND:
    lower: int
    upper: int

@dataclass
class VARIABLE:
    id: str

@dataclass
class VALUE:
    value: Union[int, float, str]

@dataclass
class INT_VALUE:
    value: int

@dataclass
class FLOAT_VALUE:
    value: float

@dataclass
class ASSIGNMENT:
    lhs: STRING_ACTION
    rhs: VALUE

@dataclass
class INT_ASSIGNMENT:
    lhs: STRING_ACTION
    rhs: int

@dataclass
class FUNCTION:
    name: str
    arg: Any

@dataclass
class PROPOSITION:
    operation: OPERATION
    lh: Union[VARIABLE, VALUE, "PROPOSITION"]
    rh: Union[VARIABLE, VALUE, "PROPOSITION"]

@dataclass
class PREDICATE:
    id: str
    proposition: PROPOSITION

LIST_PREDICATE = Union[PREDICATE, LIST]

@dataclass
class ACTION_SET_PRIMITIVE:
    argument: VARIABLE
    properties: PROPERTIES = field(default_factory=PROPERTIES)
    bound: BOUND = BOUND(0, 1)

@dataclass
class ACTION_SET_PRIMITIVES:
    arguments: LIST
    properties: PROPERTIES = field(default_factory=PROPERTIES)
    bound: BOUND = BOUND(0, 1)

@dataclass
class ACTION_SET_LIST_COMPOSITE:
    sub_action: SUB_ACTION_TYPE
    arguments: LIST_PREDICATE
    properties: PROPERTIES = field(default_factory=PROPERTIES)

@dataclass
class ACTION_SET_VALUE_COMPOSITE:
    sub_action: SUB_ACTION_TYPE
    arguments: LIST_PREDICATE
    value: VALUE
    properties: PROPERTIES = field(default_factory=PROPERTIES)

@dataclass
class ACTION_GET:
    argument: Union[VARIABLE, LIST_PREDICATE]

@dataclass
class ACTION_DEL:
    argument: Union[VARIABLE, LIST_PREDICATE]

@dataclass
class ACTION_SUB:
    argument: Union[VARIABLE, LIST_PREDICATE]

@dataclass
class ACTION_CUT:
    argument: Union[VARIABLE, PROPERTIES]

@dataclass
class ACTION_ASSUME:
    argument: PROPERTIES

@dataclass
class ACTION_REDUCE:
    pass

@dataclass
class ACTION_PROPAGATE:
    argument: PROPERTIES

@dataclass
class ACTION_MAXIMIZE:
    argument: PROPERTIES
    suchthat: INT_ASSIGNMENT

@dataclass
class ACTION_MINIMIZE:
    argument: PROPERTIES
    suchthat: INT_ASSIGNMENT