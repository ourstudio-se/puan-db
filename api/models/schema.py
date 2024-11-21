from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional, Dict, Union
from enum import Enum
from graphlib import TopologicalSorter, CycleError

class SchemaRangeType(BaseModel):
    lower: int
    upper: int

class SchemaPrimitiveDtype(str, Enum):
    integer = "integer"
    boolean = "boolean"

class SchemaPropertyDType(str, Enum):
    integer = "integer"
    float   = "float"
    boolean = "boolean"
    string  = "string"

class SchemaQuantifier(Enum):
    zero_or_more = "*"
    zero_or_one = "?"
    one_or_more = "+"

class SchemaQuantifiedVariable(BaseModel):
    variable: str
    quantifier: Union[int, SchemaQuantifier]

    @field_validator("quantifier")
    def check_quantifier(cls, value):
        # If integer, then value must be greater than 0
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("Quantifier must be greater than 0 if integer")
        return value

class SchemaLogicOperator(str, Enum):
    and_ = "and"
    or_ = "or"
    not_ = "not"
    xor_ = "xor"
    nand_ = "nand"
    nor_ = "nor"
    nxor_ = "nxor"
    imply = "imply"
    equiv = "equiv"
    geq = "geq"
    leq = "leq"

class SchemaValuedOperator(str, Enum):
    atleast = "atleast"
    atmost = "atmost"
    equal = "equal"

class SchemaLogicRelation(BaseModel):
    operator: SchemaLogicOperator
    inputs: List[SchemaQuantifiedVariable]

    @field_validator("inputs")
    def check_items(cls, value, values):
        operator = values.data.get('operator')
        binary_operators = {SchemaLogicOperator.imply, SchemaLogicOperator.equiv, SchemaLogicOperator.geq, SchemaLogicOperator.leq}
        if operator in binary_operators:
            if len(value) != 2:
                raise ValueError(f"Binary relation must have exactly 2 inputs when operator is any of {binary_operators}")
        return value

class SchemaValuedRelation(BaseModel):
    operator: SchemaValuedOperator
    inputs: List[SchemaQuantifiedVariable]
    value: int

class SchemaProperty(BaseModel):
    dtype: SchemaPropertyDType
    description: Optional[str] = None

class SchemaPropertyReference(BaseModel):
    property: str
    default: Optional[Union[int, float, bool, str]] = None

class SchemaProposition(BaseModel):
    properties: List[SchemaPropertyReference] = []
    quantifier: Union[int, SchemaQuantifier] = SchemaQuantifier.zero_or_more
    description: Optional[str] = None

class SchemaPrimitive(SchemaProposition):
    ptype: Union[SchemaRangeType, SchemaPrimitiveDtype, str]

    @field_validator("ptype")
    def validate_dtype(cls, value):
        # Check if value matches a SchemaPrimitiveDtype value
        if value in SchemaPrimitiveDtype.__members__.values():
            return SchemaPrimitiveDtype(value)
        return value

class SchemaAggregateType(str, Enum):
    sum_ = "sum"
    mean_ = "mean"

class SchemaAggregate(BaseModel):
    type: SchemaAggregateType
    source: str

class SchemaComposite(SchemaProposition):
    relation: Union[SchemaValuedRelation, SchemaLogicRelation]
    aggregates: Dict[str, SchemaAggregate] = {}

    @field_validator("relation")
    def validate_relation(cls, value):
        """
            If relation operator is imply, equic, geq, leq or any other
            binary operator, then relation must have exactly 2 items with <exactly one> quantifier set.
        """
        operator = value.operator
        binary_operators = {SchemaLogicOperator.imply, SchemaLogicOperator.equiv, SchemaLogicOperator.geq, SchemaLogicOperator.leq}
        if operator in binary_operators:
            if len(value.inputs) != 2:
                raise ValueError(f"Binary relation must have exactly 2 items when operator is any of {binary_operators}")
            for item in value.inputs:
                if not type(item.quantifier) == int or item.quantifier != 1:
                    raise ValueError(f"Binary relation ({binary_operators}) must have <exactly one> quantifier set for each item")
        return value
    
    @field_validator("aggregates")
    def validate_aggregates(cls, value, values):
        """Aggregate key must not be in properties"""
        for key in value:
            if key in map(lambda property: property.property, values.data.get('properties', {})):
                raise ValueError(f"Aggregate key '{key}' will be resolved into a property, therefore it must not already be defined in properties. Remove/rename key '{key}' in properties or aggregates.")
        return value

class DatabaseSchema(BaseModel):
    primitives: Dict[str, SchemaPrimitive]
    properties: Dict[str, SchemaProperty] = {}
    composites: Dict[str, SchemaComposite] = {}

    @model_validator(mode='after')
    def typecheck(self):

        # Check that there's no circular dependencies from composite relations
        try:
            graph = TopologicalSorter(
                dict(
                    (key, [item.variable for item in comp.relation.inputs])
                    for key, comp in self.composites.items()
                )
            )
            graph.prepare()
        except CycleError as e:
            raise ValueError(f"Circular dependency detected in schema: '{e.args[1]}'")
        
        # Check that all types are defined before use
        # Definitions are done in the order of
        # 1. Properties (no dependencies)
        # 2. Primitives (depends on properties or other primitives (or basic types))
        # 3. Composites (depends on properties, primitives or other composites)
        for property_key, property in self.properties.items():
            if property.dtype not in SchemaPropertyDType.__members__:
                raise ValueError(f"Undefined property type '{property.dtype}' used in property '{property_key}'. Must be one of {', '.join(SchemaPropertyDType.__members__)}")

        for prim_key, prim in self.primitives.items():
            for property in prim.properties:
                if property.property not in self.properties:
                    raise ValueError(f"Undefined property '{property.property}' used in primitive '{prim_key}'")
            if type(prim.ptype) == str:
                if prim.ptype not in SchemaPrimitiveDtype.__members__:
                    raise ValueError(f"Undefined primitive variable type '{prim.ptype}' used in primitive '{prim_key}'. Must be one of {', '.join(SchemaPrimitiveDtype.__members__)}")
            elif type(prim.ptype) == SchemaRangeType:
                if prim.ptype.lower > prim.ptype.upper:
                    raise ValueError(f"Invalid range for primitive '{prim_key}'")

        for comp_key, comp in self.composites.items():
            for property in comp.properties:
                if property.property not in self.properties:
                    raise ValueError(f"Undefined property '{property.property}' used in composite '{comp_key}'")
            for item in comp.relation.inputs:
                if item.variable not in self.primitives and item.variable not in self.composites:
                    raise ValueError(f"Undefined input variable '{item.variable}' used in composite '{comp_key}'")

        # Validate aggregates:
        # All relation's types must have the same dtype as the source of the aggregate
        for composite_key, composite in self.composites.items():
            missing_items = []
            for aggregate_key, aggregate in composite.aggregates.items():
                for item in composite.relation.inputs:
                    schema_item = self.composites.get(item.variable, self.primitives.get(item.variable))
                    if not any(map(lambda property: property.property == aggregate.source, schema_item.properties)):
                        missing_items.append(item.variable)

                # It also includes having the property defined in the properties
                if not aggregate.source in map(lambda x: x.property, composite.properties):
                    missing_items.append(composite_key)
            
            if missing_items:
                raise ValueError(f"Aggregate '{aggregate_key}' requires all items to have a property of type '{aggregate.source}': {', '.join(missing_items)}")
            
        # Check that default values are valid
        for pkey, prim in self.primitives.items():
            for prop in prim.properties:
                if prop.default is not None:
                    if type(prop.default).__name__ != self.properties[prop.property].dtype:
                        raise ValueError(f"Invalid default value for property '{pkey}': expected {self.properties[prop.property].dtype.value}, got {type(prop.default).__name__}")
        
        return self

## Specific request models
class Database(BaseModel):
    id: str

class RequestOk(BaseModel):
    message: str