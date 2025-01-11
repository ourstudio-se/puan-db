import api.models.untyped_model as untyped_model
import api.models.typed_model as typed_model
import api.models.schema as schema_model
import api.models.query as query_model

from pydantic import BaseModel, Field
from typing import List, Union, Dict, Optional, Set
from itertools import chain, starmap

# Searches for items and uses the property value provided
class DatabaseSearchItemsStrategy(BaseModel):
    items: query_model.SearchQuery
    property: str
    default: int = 0

# Applying directly what ID's and their values
DatabaseSearchPlainStrategy = Dict[str, int]
DatabaseSearchObjective = Union[DatabaseSearchPlainStrategy, DatabaseSearchItemsStrategy]

class DatabaseSearchData(BaseModel):
    data: Optional[typed_model.SchemaData] = None
    return_: str = Field(..., alias="return")

class DatabaseSearchSuchThat(BaseModel):
    assume: DatabaseSearchData
    to_equal: untyped_model.Bounds

class DatabaseSearchRequest(BaseModel):
    objectives: List[DatabaseSearchObjective]
    suchthat: Optional[DatabaseSearchSuchThat] = None
    direction: str = "maximize"

    def objectives_dict(self, model: typed_model.DatabaseModel) -> List[Dict[str, int]]:
        
        converted_objectives = []
        for objective in self.objectives:
            if isinstance(objective, dict):
                converted_objectives.append(objective)
            elif isinstance(objective, DatabaseSearchItemsStrategy):
                propositions: Dict[str, typed_model.CompPrimitive] = model.search_items(objective.items)
                converted_objectives.append(
                    {
                        id: proposition.properties.get(
                            objective.property, 
                            None
                        ) or objective.default
                        for id, proposition in propositions.items()
                    }
                )
            else:
                raise ValueError(f"Not implemented objective type: {type(objective)}")

        return converted_objectives
    
class DatabaseSearchPrimitiveSolutionVariable(BaseModel):
    variable: typed_model.Primitive
    value: int
    
class DatabaseSearchCompositeSolutionVariable(BaseModel):
    variable: typed_model.Composite
    value: int
    
class DatabaseSearchSolution(BaseModel): 
    solution: Optional[Dict[str, Union[DatabaseSearchPrimitiveSolutionVariable, DatabaseSearchCompositeSolutionVariable]]]
    error: Optional[str] = None
    
class DatabaseSearchResponse(BaseModel):
    solutions: List[DatabaseSearchSolution]

    @staticmethod
    def transitive_dependencies(dependencies: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        # Helper function for DFS to find all reachable nodes
        def dfs(node: str, visited: Set[str]) -> Set[str]:
            if node not in dependencies:
                return set()
            
            for dep in dependencies[node]:
                if dep not in visited:
                    visited.add(dep)
                    visited.update(dfs(dep, visited))
            return visited

        # Calculate transitive dependencies for each node
        transitive_deps = {}
        for node in dependencies:
            transitive_deps[node] = dfs(node, set())
        
        return transitive_deps

    @staticmethod
    def from_untyped_solutions(
        solutions: List[untyped_model.SolutionResponse], 
        model: typed_model.DatabaseModel,
        exclude_zero: bool = True
    ) -> "DatabaseSearchResponse":
        
        """
            Converts a list of untyped solutions to a DatabaseSearchResponse object.
            This will include calculated properties and values for each proposition in the model.
        """

        # Get the model primitives and composites as id -> object mappings
        model_primitives: Dict[str, typed_model.Primitive] = model.data.primitives
        model_composites: Dict[str, typed_model.Composite] = model.data.composites

        # Since this needs only to be calculated once, we do it here
        # and don't care if no composite have aggregates (where it would be needed)
        transitive_deps = DatabaseSearchResponse.transitive_dependencies(
            dict(
                chain(
                    map(
                        lambda k: (k, {}),
                        model_primitives
                    ),
                    starmap(
                        lambda k, v: (k, set(v.inputs)),
                        model_composites.items()
                    )
                )
            )
        )

        new_solutions = []
        for solution in solutions:

            new_solution = {}

            # Check if the solution is an error
            if solution.error is not None:
                new_solutions.append(DatabaseSearchSolution(error=solution.error))
                continue

            # Convert the solution to a typed solution
            for sol_key, sol_val in solution.solution.items():
                
                # We exclude solution variables with value 0
                # if exclude_zero is set to True
                if sol_val == 0 and exclude_zero:
                    continue
                
                if sol_key in model_primitives:
                    model_primitive = model_primitives.get(sol_key)
                    if model_primitive is None:
                        raise ValueError(f"Primitive '{sol_key}' not found in model")
                    
                    new_solution[sol_key] = DatabaseSearchPrimitiveSolutionVariable(
                        variable=model_primitive,
                        value=sol_val
                    )

                elif sol_key in model_composites:
                    composite_model = model_composites.get(sol_key)
                    if composite_model is None:
                        raise ValueError(f"Composite '{sol_key}' not found in model")
                    
                    composite_schema = model.database_schema.composites.get(composite_model.ptype)
                    if composite_schema is None:
                        raise ValueError(f"Composite '{sol_key}' not found in schema")
                    
                    # new_aggregate_properties = {}
                    # if composite_schema.aggregates:
                        
                    #     for aggregate_key, aggregate in composite_schema.aggregates.items():
                            
                    #         # We start by creating a dictionery with the values used for the aggregate
                    #         aggregate_values = dict(
                    #             starmap(
                    #                 lambda proposition_id, proposition: (
                    #                     proposition_id, 
                    #                     proposition.properties.get(
                    #                         aggregate.source, 
                    #                         0 # Default value should come from the schema in the future
                    #                     )
                    #                 ),
                    #                 model.propositions.items()
                    #             )
                    #         )

                    #         # Just sum each dependency bound multiplied with value supplied in data
                    #         if aggregate.type == schema_model.SchemaAggregateType.sum_:
                    #             new_aggregate_properties[aggregate_key] = sum(
                    #                 map(
                    #                     aggregate_values.get,
                    #                     transitive_deps[sol_key]
                    #                 )
                    #             ) + solution.solution[sol_key] * aggregate_values.get(sol_key, 0.0)
                    #         else:
                    #             raise ValueError(f"Aggregate type '{aggregate.type}' not yet implemented")
                    
                    # composite_model.properties.update(new_aggregate_properties)
                    new_solution[sol_key] = DatabaseSearchCompositeSolutionVariable(
                        variable=composite_model,
                        value=sol_val,
                    )
                
            new_solutions.append(
                DatabaseSearchSolution(
                    solution=dict(sorted(new_solution.items(), key=lambda x: x[1].variable.ptype))
                )
            )

        return DatabaseSearchResponse(solutions=new_solutions)