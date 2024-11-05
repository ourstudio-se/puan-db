from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional, Dict, Union
from enum import Enum

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
    dtype: str
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
    items: List[SchemaQuantifiedVariable]

    @field_validator("items")
    def check_items(cls, value, values):
        operator = values.data.get('operator')
        binary_operators = {SchemaLogicOperator.imply, SchemaLogicOperator.equiv, SchemaLogicOperator.geq, SchemaLogicOperator.leq}
        if operator in binary_operators:
            if len(value) != 2:
                raise ValueError(f"Binary relation must have exactly 2 items when operator is any of {binary_operators}")
        return value

class SchemaValuedRelation(BaseModel):
    operator: SchemaValuedOperator
    items: List[SchemaQuantifiedVariable]
    value: int

class SchemaProperty(BaseModel):
    dtype: SchemaPropertyDType
    description: Optional[str] = None

class SchemaPropertyReference(BaseModel):
    dtype: str
    default: Optional[Union[int, float, bool, str]] = None

class SchemaProposition(BaseModel):
    properties: List[SchemaPropertyReference] = []
    quantifier: Union[int, SchemaQuantifier] = SchemaQuantifier.zero_or_more
    description: Optional[str] = None

class SchemaPrimitive(SchemaProposition):
    dtype: Union[SchemaRangeType, SchemaPrimitiveDtype, str]

    @field_validator("dtype")
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
            if len(value.items) != 2:
                raise ValueError(f"Binary relation must have exactly 2 items when operator is any of {binary_operators}")
            for item in value.items:
                if not type(item.quantifier) == int or item.quantifier != 1:
                    raise ValueError(f"Binary relation ({binary_operators}) must have <exactly one> quantifier set for each item")
        return value
    
    @field_validator("aggregates")
    def validate_aggregates(cls, value, values):
        """Aggregate key must not be in properties"""
        for key in value:
            if key in map(lambda property: property.dtype, values.data.get('properties', {})):
                raise ValueError(f"Aggregate key '{key}' will be resolved into a property, therefore it must not already be defined in properties. Remove/rename key '{key}' in properties or aggregates.")
        return value

class Schema(BaseModel):
    primitives: Dict[str, SchemaPrimitive]
    properties: Dict[str, SchemaProperty] = {}
    composites: Dict[str, SchemaComposite] = {}

    @model_validator(mode='after')
    def typecheck(self):
        
        # Check that all types are defined before use
        # Definitions are done in the order of
        # 1. Properties (no dependencies)
        # 2. Primitives (depends on properties or other primitives (or basic types))
        # 3. Composites (depends on properties, primitives or other composites)
        defined_types = set()
        for prop_key in self.properties:
            defined_types.add(prop_key)

        for prim_key, prim in self.primitives.items():
            for property in prim.properties:
                if property.dtype not in defined_types:
                    raise ValueError(f"Undefined property type '{property.dtype}' used in primitive '{prim_key}'")
            if type(prim.dtype) == str:
                if prim.dtype not in defined_types:
                    raise ValueError(f"Undefined proposition type '{prim.dtype}' used in primitive '{prim_key}'")
            elif type(prim.dtype) == SchemaRangeType:
                if prim.dtype.lower > prim.dtype.upper:
                    raise ValueError(f"Invalid range for primitive '{prim_key}'")
            defined_types.add(prim_key)

        for comp_key, comp in self.composites.items():
            for property in comp.properties:
                if property.dtype not in defined_types:
                    raise ValueError(f"Undefined property type '{property.dtype}' used in primitive '{prim_key}'")
            for item in comp.relation.items:
                if item.dtype not in defined_types:
                    raise ValueError(f"Undefined proposition type '{item.dtype}' used in composite '{comp_key}'")
                defined_types.add(comp_key)

        # Validate aggregates:
        # All relation's types must have the same dtype as the source of the aggregate
        for composite_key, composite in self.composites.items():
            missing_items = []
            for aggregate_key, aggregate in composite.aggregates.items():
                for item in composite.relation.items:
                    schema_item = self.composites.get(item.dtype, self.primitives.get(item.dtype))
                    if not any(map(lambda property: property.dtype == aggregate.source, schema_item.properties)):
                        missing_items.append(item.dtype)

                # It also includes having the property defined in the properties
                if not aggregate.source in map(lambda x: x.dtype, composite.properties):
                    missing_items.append(composite_key)
            
            if missing_items:
                raise ValueError(f"Aggregate '{aggregate_key}' requires all items to have a property of type '{aggregate.source}': {', '.join(missing_items)}")
            
        # Check that default values are valid
        for pkey, prim in self.primitives.items():
            for prop in prim.properties:
                if prop.default is not None:
                    if type(prop.default).__name__ != self.properties[prop.dtype].dtype:
                        raise ValueError(f"Invalid default value for property '{pkey}': expected {self.properties[prop.dtype].dtype.value}, got {type(prop.default).__name__}")
        
        return self

## Specific request models
class DatabaseSchema(BaseModel):
    name: str
    schema: Schema = Schema(primitives={}, properties={}, composites={})
    description: Optional[str] = None

class RequestOk(BaseModel):
    message: str