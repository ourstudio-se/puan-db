from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Union, List, Optional, Tuple
from pldag import PLDAG, CompilationSetting
from graphlib import TopologicalSorter

import api.models.schema as schema_models
import api.models.query as query_models

class DynamicValue(BaseModel):
    min_: Optional[Union[float, bool, str]] = Field(None, alias="min")
    max_: Optional[Union[float, bool, str]] = Field(None, alias="max")

    def __hash__(self):
        return hash((self.min_, self.max_))

class Definition(str, Enum):
    primitive = "primitive"
    composite = "composite"

class Primitive(BaseModel):
    definition: Definition = Definition.primitive
    ptype: str
    properties: Dict[str, Optional[Union[int, float, bool, str, DynamicValue]]] = {}

    def validate_properties(self, id: str, schema: schema_models.DatabaseSchema) -> List[str]:
        type_map = {
            "integer": int,
            "float": float,
            "boolean": bool,
            "string": str
        }

        errors: List[str] = []

        # Get the specific schema properties of the node
        schema_primitive_properties = schema.primitives.get(self.ptype, schema.composites.get(self.ptype))
        if schema_primitive_properties is None:
            if self.properties:
                errors.append(f"Dtype '{self.ptype}' has no properties in schema but has properties in data ({id})")
        
        for model_property in self.properties:
            if model_property not in map(lambda p: p.property, schema_primitive_properties.properties):
                errors.append(f"Property '{model_property}' set for {id} not found in schema")
        
        for schema_property in schema_primitive_properties.properties:
            if not schema_property.property in self.properties and schema_property.default is None:
                errors.append(f"{id} missing property '{schema_property.property}' and no default value set")
            else:
                schema_property_base = schema.properties[schema_property.property]
                # It is ok to be null, or not set here, so if schema property key is not in node properties just continue
                if schema_property.property not in self.properties:
                    continue
                
                node_property = self.properties[schema_property.property]
                # Check that the property is either dynamic or of the correct type
                if not (isinstance(node_property, type_map[schema_property_base.dtype]) or schema_property.dynamic is not None):
                    errors.append(f"Property '{schema_property.property}' for '{id}' is not of type {schema_property_base.dtype.value}")

        return errors

    def validate_schema(self, id: str, model: Dict[str, "CompPrimitive"], schema: schema_models.DatabaseSchema) -> List[str]:
        """Returns a list of validation errors"""
        
        errors: List[str] = []
        
        # Check that the model has all the required propositions
        if not self.ptype in schema.primitives and not self.ptype in schema.composites:
            errors.append(f"Data type '{self.ptype}' set for '{id}' not found in schema")
        else:
            if self.definition == Definition.primitive:
                if self.ptype not in schema.primitives:
                    errors.append(f"Primitive data type '{self.ptype}' set for {id} not found in schema")
                else:
                    errors += self.validate_properties(id, schema)

            elif self.definition == Definition.composite:
                if self.ptype not in schema.composites:
                    errors.append(f"Composite data type '{self.ptype}' set for {id} not found in schema")
                else:
                    errors += self.validate_properties(id, schema)
                
                    # Includes checking argument one by one 
                    schema_composite = schema.composites[self.ptype]
                    argument_dtypes_count = {}
                    for argument in self.inputs:
                        if not argument in model:
                            errors.append(f"Argument '{argument}' set for {id} not defined in data")
                        else:
                            argument_node = model[argument]
                            if not argument_node.ptype in map(lambda x: x.variable, schema_composite.relation.inputs):
                                expecting_types = ", ".join(map(lambda item: item.variable, schema_composite.relation.inputs))
                                errors.append(f"Expected argument '{argument}' to be of type [{expecting_types}], but got <{argument_node.ptype}>")
                            else:
                                argument_dtypes_count.setdefault(argument_node.ptype, 0)
                                argument_dtypes_count[argument_node.ptype] += 1

                    schema_dtypes_count = {}
                    for relation_item in schema_composite.relation.inputs:
                        if type(relation_item.quantifier) == int:
                            schema_dtypes_count.setdefault(relation_item.variable, 0)
                            schema_dtypes_count[relation_item.variable] += relation_item.quantifier
                        else:
                            schema_dtypes_count[relation_item.variable] = relation_item.quantifier

                    for schema_dtype, schema_quantifier in schema_dtypes_count.items():
                        count = argument_dtypes_count.get(schema_dtype, 0)
                        if type(schema_quantifier) == int:
                            if count != schema_quantifier:
                                errors.append(f"Requires <exactly {schema_quantifier}> proposition(s) of type '{schema_dtype}' ({id})")
                        else:
                            if schema_quantifier == schema_models.SchemaQuantifier.one_or_more:
                                if count < 1:
                                    errors.append(f"Requires <atleast 1> propositions of type '{relation_item.variable}' ({id})")
                            elif schema_quantifier == schema_models.SchemaQuantifier.zero_or_one:
                                if count > 1:
                                    errors.append(f"Requires <atmost 1> propositions of type '{relation_item.variable}' ({id})")

        return errors

class Composite(Primitive):

    # Composite specific properties
    definition: Definition = Definition.composite
    inputs: List[str]

CompPrimitive = Union[Composite, Primitive]

class SchemaData(BaseModel):
    primitives: Dict[str, Primitive]
    composites: Dict[str, Composite]

    def get(self, id: str) -> Optional[CompPrimitive]:
        if id in self.primitives:
            return self.primitives[id]
        elif id in self.composites:
            return self.composites[id]
        return None
    
    def exists(self, id: str) -> bool:
        return id in self.primitives or id in self.composites
    
    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        # Sort the dictionaries by key
        data["primitives"] = dict(sorted(data["primitives"].items()))
        data["composites"] = dict(sorted(data["composites"].items()))
        return data
    
    def sorted(self) -> "SchemaData":
        return SchemaData(
            primitives=dict(sorted(self.primitives.items())),
            composites=dict(sorted(self.composites.items()))
        )

class DatabaseModel(BaseModel):
    
    database_schema: schema_models.DatabaseSchema
    data: SchemaData

    def sorted(self) -> "DatabaseModel":
        return DatabaseModel(
            database_schema=self.database_schema.sorted(),
            data=self.data.sorted()
        )

    @staticmethod
    def create_empty() -> "DatabaseModel":
        return DatabaseModel(
            database_schema=schema_models.DatabaseSchema(
                primitives={}, 
                properties={}, 
                composites={},
            ),
            data=SchemaData(
                primitives={}, 
                composites={},
            )
        )

    @property
    def propositions(self) -> Dict[str, CompPrimitive]:
        return {**self.data.primitives, **self.data.composites}

    def merge_data(self, other: SchemaData) -> "DatabaseModel":
        return DatabaseModel(
            database_schema=self.database_schema,
            data=SchemaData(
                primitives={**self.data.primitives, **other.primitives},
                composites={**self.data.composites, **other.composites}
            )
        )
    
    def copy(self) -> "DatabaseModel":
        return DatabaseModel(
            database_schema=self.database_schema,
            data=SchemaData(
                primitives={**self.data.primitives},
                composites={**self.data.composites}
            )
        )
    
    def validate_all(self) -> "SchemaValidationResponse":
        
        errors = SchemaValidationResponse()
        if self.database_schema is None:
            raise ValueError("Model schema is not defined")
        
        # # Validate schema first by its own
        # self.database_schema.typecheck()

        # Create a dictionary of propositions for easy access
        model_data = self.propositions
        
        # Check if all arguments are defined
        for _id, composite in self.data.composites.items():
            for argument in composite.inputs:
                if not argument in model_data:
                    errors.add_error(_id, f"Input '{argument}' is not defined")
        
        # Check each proposition against the schema
        for id, proposition in model_data.items():
            for error in proposition.validate_schema(id, model_data, self.database_schema):
                errors.add_error(id, error)

        # Check the general schema
        for schema_proposition_key, schema_proposition in {**self.database_schema.primitives, **self.database_schema.composites}.items():
            count = sum(map(lambda x: x.ptype == schema_proposition_key, self.propositions.values()))
            if type(schema_proposition.quantifier) == int:
                if count != schema_proposition.quantifier:
                    errors.add_error(
                        schema_proposition_key,
                        f"Schema requires <exactly {schema_proposition.quantifier}> proposition(s) of type '{schema_proposition_key}', but got <{count}>"
                    )
            else:
                if schema_proposition.quantifier == schema_models.SchemaQuantifier.one_or_more:
                    if count < 1:
                        errors.add_error(
                            schema_proposition_key,
                            f"Schema requires <atleast 1> proposition(s) of type '{schema_proposition_key}', but got {count}"
                        )
                elif schema_proposition.quantifier == schema_models.SchemaQuantifier.zero_or_one:
                    if count > 1:
                        errors.add_error(
                            schema_proposition_key,
                            f"Schema requires <atmost 1> proposition of type '{schema_proposition_key}', but got {count}"
                        )

        return errors
    
    def update(self, propositions: Dict[str, CompPrimitive]) -> "DatabaseModel":
        """Updates propositions by id and returns a Model with the updated proposition"""
        return DatabaseModel(
            database_schema=self.database_schema,
            data=SchemaData(
                primitives={**self.data.primitives, **{p_id: p for p_id, p in propositions.items() if p.definition == Definition.primitive}},
                composites={**self.data.composites, **{p_id: p for p_id, p in propositions.items() if p.definition == Definition.composite}}
            )
        )
    
    def delete(self, ids: List[str]) -> "DatabaseModel":
        """Deletes propositions by id and returns a new Model without the proposition"""
        return DatabaseModel(
            database_schema=self.database_schema,
            data=SchemaData(
                primitives={p_id: p for p_id, p in self.data.primitives.items() if p_id not in ids},
                composites={p_id: p for p_id, p in self.data.composites.items() if p_id not in ids}
            )
        )
    
    def delete_by_schema_id(self, schema_ids: List[str]) -> "DatabaseModel":
        """Deletes propositions by schema id and returns a new Model without the proposition"""
        return DatabaseModel(
            database_schema=self.database_schema,
            data=SchemaData(
                primitives={p_id: p for p_id, p in self.data.primitives.items() if p.ptype not in schema_ids},
                composites={p_id: p for p_id, p in self.data.composites.items() if p.ptype not in schema_ids}
            )
        )
    
    def delete_by_schema_property(self, schema_property: str) -> "DatabaseModel":
        """Deletes properties from propositions and returns a new Model without the property"""
        return DatabaseModel(
            database_schema=schema_models.DatabaseSchema(
                primitives={
                    p_id: schema_models.SchemaPrimitive(
                        ptype=p.ptype,
                        quantifier=p.quantifier,
                        properties=[prop for prop in p.properties if prop.property != schema_property],
                        description=p.description
                    )
                    for p_id, p in self.database_schema.primitives.items()
                },
                composites={
                    p_id: schema_models.SchemaComposite(
                        relation=p.relation,
                        properties=[prop for prop in p.properties if prop.property != schema_property],
                        quantifier=p.quantifier,
                        description=p.description
                    )
                    for p_id, p in self.database_schema.composites.items()
                },
                properties=self.database_schema.properties,
            ),
            data=SchemaData(
                primitives={
                    p_id: Primitive(
                        ptype=p.ptype,
                        properties={prop: value for prop, value in p.properties.items() if prop != schema_property}
                    )
                    for p_id, p in self.data.primitives.items()
                },
                composites={
                    p_id: Composite(
                        ptype=p.ptype,
                        properties={prop: value for prop, value in p.properties.items() if prop != schema_property},
                        inputs=p.inputs
                    )
                    for p_id, p in self.data.composites.items()
                }
            )
        )
    
    def search_items(self, query: query_models.SearchQuery) -> List[CompPrimitive]:
        def match_condition(item, condition: query_models.Condition) -> bool:
            field_path = condition.field.split(".")
            # Navigate the field path in the item to get the target value
            target_value = item
            for field in field_path:
                if isinstance(target_value, dict):
                    target_value = target_value.get(field, None)
                else:
                    target_value = getattr(target_value, field, None)
                if target_value is None:
                    return False
                
            # Check for compatibility: same type or both numeric (int and float)
            if condition.operator != "contains" and not (
                isinstance(target_value, type(condition.value)) or
                (isinstance(target_value, (int, float)) and isinstance(condition.value, (int, float)))
            ):
                return False

            # Evaluate the comparison based on the operator
            if condition.operator == "equal":
                return target_value == condition.value
            elif condition.operator == "greater":
                return target_value > condition.value
            elif condition.operator == "lesser":
                return target_value < condition.value
            elif condition.operator == "greater_equal":
                return target_value >= condition.value
            elif condition.operator == "lesser_equal":
                return target_value <= condition.value
            elif condition.operator == "contains":
                return isinstance(target_value, list) and condition.value in target_value
            return False

        def evaluate_logical_condition(item, logical_condition: query_models.LogicalCondition) -> bool:
            if logical_condition.operator == query_models.LogicalOperator.and_:
                return all(
                    evaluate_query(item, cond) for cond in logical_condition.conditions
                )
            elif logical_condition.operator == query_models.LogicalOperator.or_:
                return any(
                    evaluate_query(item, cond) for cond in logical_condition.conditions
                )
            elif logical_condition.operator == query_models.LogicalOperator.not_:
                return not evaluate_query(item, logical_condition.conditions[0])
            return False

        def evaluate_query(item, query) -> bool:
            if isinstance(query, query_models.Condition):
                return match_condition(item, query)
            elif isinstance(query, query_models.LogicalCondition):
                return evaluate_logical_condition(item, query)
            return False

        # Filter the propositions based on the query conditions
        return {id: item for id, item in self.propositions.items() if evaluate_query(item, query.conditions)}
    
    def to_pldag(self) -> PLDAG:
        """
            Returns the PLDAG model along with a root ID that needs to be assumed true.
        """

        # We currently have max and min integer set here
        MAX_INT = 2**31
        MIN_INT = -2**31

        model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
        for primitive_id, primitive in self.data.primitives.items():
            primitive_schema = self.database_schema.primitives.get(primitive.ptype)
            if type(primitive_schema.ptype) == schema_models.SchemaPrimitiveDtype:
                if primitive_schema.ptype == schema_models.SchemaPrimitiveDtype.integer:
                    model.set_primitive(primitive_id, complex(MIN_INT, MAX_INT))
                elif primitive_schema.ptype == schema_models.SchemaPrimitiveDtype.boolean:
                    model.set_primitive(primitive_id)
                else:
                    raise ValueError(f"Unsupported primitive type: {primitive_schema.ptype}")
            elif type(primitive_schema.ptype) == schema_models.SchemaRangeType:
                model.set_primitive(primitive_id, complex(primitive_schema.ptype.lower, primitive_schema.ptype.upper))

        # Keep track of the arguments and their ID mapping
        argument_id_map = {}
        def map_arguments_helper(arguments: List[str]) -> List[str]:
            # Map the argument ID's to pldag ID's
            return list(
                map(
                    lambda arg: argument_id_map.get(arg, arg),
                    arguments
                )
            )
        
        # To loop in correct order, create a topological sorter for the composites
        graph = TopologicalSorter(
            dict(
                (key, [item for item in comp.inputs])
                for key, comp in self.data.composites.items()
            )
        )

        for composite_id in filter(lambda p: p in self.data.composites, graph.static_order()):
            composite = self.data.composites[composite_id]
            composite_schema = self.database_schema.composites.get(composite.ptype)

            # Switch case for every schema relation operator
            if type(composite_schema.relation.operator) == schema_models.SchemaLogicOperator:
                if composite_schema.relation.operator == schema_models.SchemaLogicOperator.and_:
                    argument_id_map[composite_id] = model.set_and(map_arguments_helper(composite.inputs), alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.or_:
                    argument_id_map[composite_id] = model.set_or(map_arguments_helper(composite.inputs), alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.not_:
                    argument_id_map[composite_id] = model.set_not(composite.inputs, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.imply:
                    lhs, rhs = map_arguments_helper(composite.inputs)
                    argument_id_map[composite_id] = model.set_imply(lhs, rhs, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.equiv:
                    lhs, rhs = map_arguments_helper(composite.inputs)
                    argument_id_map[composite_id] = model.set_equiv(lhs, rhs, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.geq:
                    lhs, rhs = map_arguments_helper(composite.inputs)
                    argument_id_map[composite_id] = model.set_gelineq({lhs: 1, rhs: -1}, 0, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.leq:
                    lhs, rhs = map_arguments_helper(composite.inputs)
                    argument_id_map[composite_id] = model.set_gelineq({lhs: -1, rhs: 1}, 0, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.nand_:
                    argument_id_map[composite_id] = model.set_nand(map_arguments_helper(composite.inputs), alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.nor_:
                    argument_id_map[composite_id] = model.set_nor(map_arguments_helper(composite.inputs), alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaLogicOperator.xor_:
                    argument_id_map[composite_id] = model.set_xor(map_arguments_helper(composite.inputs), alias=composite_id)
                else:
                    raise ValueError(f"Unsupported composite operator: {composite_schema.relation.operator}")
            elif type(composite_schema.relation.operator) == schema_models.SchemaValuedOperator:
                if composite_schema.relation.operator == schema_models.SchemaValuedOperator.equal:
                    argument_id_map[composite_id] = model.set_equal(map_arguments_helper(composite.inputs), composite_schema.relation.value, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaValuedOperator.atleast:
                    argument_id_map[composite_id] = model.set_atleast(map_arguments_helper(composite.inputs), composite_schema.relation.value, alias=composite_id)
                elif composite_schema.relation.operator == schema_models.SchemaValuedOperator.atmost:
                    argument_id_map[composite_id] = model.set_atmost(map_arguments_helper(composite.inputs), composite_schema.relation.value, alias=composite_id)
                else:
                    raise ValueError(f"Unsupported composite operator: {composite_schema.relation.operator}")
            else:
                raise ValueError(f"Unsupported composite operator: {composite_schema.relation.operator}")

        # This may raise an exception if the model is not valid
        model.compile()

        return model
    
    def update_properties(self, proposition: CompPrimitive) -> CompPrimitive:
        for property_id in proposition.properties:
            schema_property = self.database_schema.properties.get(property_id)
            try:
                proposition.properties[property_id] = schema_property.dtype.to_python()(proposition.properties[property_id])
            except:
                continue
        return proposition
    
    def update_data_from_schema(self) -> "DatabaseModel":
        """
            Updates the data properties based on the schema properties.
            If property is not found in the schema, it is removed from data.
        """
        return DatabaseModel(
            database_schema=self.database_schema,
            data=SchemaData(
                primitives={
                    p_id: Primitive(
                        ptype=p.ptype,
                        properties={prop: value for prop, value in p.properties.items() if prop in map(lambda sp: sp.property, self.database_schema.primitives[p.ptype].properties)}
                    )
                    for p_id, p in self.data.primitives.items()
                },
                composites={
                    p_id: Composite(
                        ptype=p.ptype,
                        properties={prop: value for prop, value in p.properties.items() if prop in map(lambda sp: sp.property, self.database_schema.composites[p.ptype].properties)},
                        inputs=p.inputs,
                    )
                    for p_id, p in self.data.composites.items()
                }
            )
        )
    
class SchemaValidationResponse(BaseModel):

    errors: Dict[str, Dict[str, List[str]]] = {}

    def add_error(self, key: str, error: str):
        self.errors.setdefault(key, [])
        self.errors[key].append(error)