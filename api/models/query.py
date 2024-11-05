
# from typing import List, Dict, Union
# from enum import Enum

# class LogicalOperator(str, Enum):
#     and_ = "and"
#     or_ = "or"
#     not_ = "not"

# class ComparisionOperator(str, Enum):
#     eq_ = "equal"
#     ge_ = "greater"
#     le_ = "lesser"
#     gte_ = "greater_equal"
#     lte_ = "lesser_equal"

# SearchField = Dict[ComparisionOperator, Union[int, float, bool, str]]
# SearchImplicitConjunction = Dict[str, SearchField]
# SearchLogicalOperatorQuery = Dict[Union[LogicalOperator, str], List[SearchImplicitConjunction]]
# SearchQuery = Union[SearchLogicalOperatorQuery, SearchImplicitConjunction]

from typing import List, Union, Dict
from pydantic import BaseModel, Field
from enum import Enum

class LogicalOperator(str, Enum):
    and_ = "and"
    or_ = "or"
    not_ = "not"

class ComparisonOperator(str, Enum):
    eq = "equal"
    ge = "greater"
    le = "lesser"
    gte = "greater_equal"
    lte = "lesser_equal"
    contains = "contains"

class Condition(BaseModel):
    field: str
    operator: ComparisonOperator
    value: Union[int, float, bool, str]

class LogicalCondition(BaseModel):
    operator: LogicalOperator
    conditions: List[Union["LogicalCondition", Condition]]

# Allow recursive references
LogicalCondition.model_rebuild()

class SearchQuery(BaseModel):
    conditions: Union[LogicalCondition, Condition, List[Condition]]