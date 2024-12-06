import logging
import api.models.schema as schema_models
import api.models.typed_model as typed_models
import api.models.query as query_models
import api.models.database_search as database_search
import api.models.untyped_model as untyped_models

from api.settings import EnvironmentVariables
from api.storage.databases import *

from pldag import PLDAG
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from itertools import starmap
from typing import List, Dict, Optional

router = APIRouter()
env = EnvironmentVariables()
logger = logging.getLogger(__name__)

################################################################################################
# --------------------------------- DATABASE OPERATIONS [BEGIN] ---------------------------------#
################################################################################################

@router.get("/databases", response_model=List[schema_models.Database])
async def get_databases(storage: DatabaseStorage = Depends(env.get_database_storage)):
    return storage.get_all()

@router.post("/databases", response_model=schema_models.Database)
async def create_database(
    schema_request: schema_models.Database, 
    database_storage: DatabaseStorage = Depends(env.get_database_storage),
    model_storage: ModelStorage = Depends(env.get_model_storage),
):
    
    if database_storage.exists(schema_request.id):
        raise HTTPException(status_code=409, detail="Database already exists")
    
    try:
        database_storage.set(schema_request.id, schema_request)
        model_storage.set(
            schema_request.id,
            typed_models.DatabaseModel.create_empty()
        )
        return schema_request
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating database: {str(e)}")
    
@router.delete("/databases/{database}", response_model=schema_models.RequestOk)
async def delete_database(
    database: str, 
    database_storage: DatabaseStorage = Depends(env.get_database_storage),
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    try:
        model_storage.delete(database)
        database_storage.delete(database)
        return schema_models.RequestOk(message=f"Database deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting database: {str(e)}")
    
@router.get("/databases/{database}", response_model=typed_models.DatabaseModel, description="Get schema and data from the database {database}")
async def get_database(
    database: str, 
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # Get data if it exists
    data = model_storage.get(database)
    if data is None:
        raise HTTPException(status_code=404, detail="Database has no data")
    
    # Get data
    return data

# ################################################################################################
# # --------------------------------- DATABASE OPERATIONS [END] -----------------------------------#
# ################################################################################################

# ################################################################################################
# # --------------------------------- SCHEMA OPERATIONS [BEGIN] ---------------------------------#
# ################################################################################################

@router.get("/databases/{database}/schema", response_model=schema_models.DatabaseSchema)
async def get_database_schema(database: str, model_storage: ModelStorage = Depends(env.get_model_storage)):
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if database_model is None:
        raise HTTPException(status_code=404, detail="Database not found")
    return database_model.database_schema

@router.patch("/databases/{database}/schema", response_model=schema_models.DatabaseSchema)
async def update_database_schema(
    database: str,
    schema: schema_models.DatabaseSchema,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.database_schema = schema
    model_storage.set(database, database_model)
    return schema

@router.get("/databases/{database}/schema/properties", response_model=Dict[str, schema_models.SchemaProperty])
async def get_schema_properties(
    database: str, 
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    return database_model.database_schema.properties

@router.get("/databases/{database}/schema/primitives", response_model=Dict[str, schema_models.SchemaPrimitive])
async def get_schema_primitives(
    database: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    return database_model.database_schema.primitives

@router.get("/databases/{database}/schema/composites", response_model=Dict[str, schema_models.SchemaComposite])
async def get_schema_composites(
    database: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    return database_model.database_schema.composites

@router.get("/databases/{database}/schema/properties/{id}", response_model=schema_models.SchemaProperty)
async def get_schema_property(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.database_schema.properties:
        raise HTTPException(status_code=404, detail="Property not found")
    
    return database_model.database_schema.properties[id]

@router.get("/databases/{database}/schema/primitives/{id}", response_model=schema_models.SchemaPrimitive)
async def get_schema_primitive(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.database_schema.primitives:
        raise HTTPException(status_code=404, detail="Primitive not found")
    
    return database_model.database_schema.primitives[id]

@router.get("/databases/{database}/schema/composites/{id}", response_model=schema_models.SchemaComposite)
async def get_schema_composite(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.database_schema.composites:
        raise HTTPException(status_code=404, detail="Composite not found")
    
    return database_model.database_schema.composites[id]

@router.patch("/databases/{database}/schema/properties/{id}", response_model=schema_models.SchemaProperty)
async def update_schema_property(
    database: str,
    id: str,
    property: schema_models.SchemaProperty,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.database_schema.properties[id] = property
    model_storage.set(database, database_model)
    return property

@router.patch("/databases/{database}/schema/primitives/{id}", response_model=schema_models.SchemaPrimitive)
async def update_schema_primitive(
    database: str,
    id: str,
    primitive: schema_models.SchemaPrimitive,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.database_schema.primitives[id] = primitive
    model_storage.set(database, database_model)
    return primitive

@router.patch("/databases/{database}/schema/composites/{id}", response_model=schema_models.SchemaComposite)
async def update_schema_composite(
    database: str,
    id: str,
    composite: schema_models.SchemaComposite,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.database_schema.composites[id] = composite
    model_storage.set(database, database_model)
    return composite

@router.delete("/databases/{database}/schema/properties/{id}", response_model=schema_models.RequestOk)
async def delete_schema_property(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.database_schema.properties:
        raise HTTPException(status_code=404, detail="Property not found")
    
    del database_model.database_schema.properties[id]
    model_storage.set(database, database_model)
    return schema_models.RequestOk(message=f"Property deleted")

@router.delete("/databases/{database}/schema/primitives/{id}", response_model=schema_models.RequestOk)
async def delete_schema_primitive(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.database_schema.primitives:
        raise HTTPException(status_code=404, detail="Primitive not found")
    
    del database_model.database_schema.primitives[id]
    model_storage.set(database, database_model)
    return schema_models.RequestOk(message=f"Primitive deleted")

@router.delete("/databases/{database}/schema/composites/{id}", response_model=schema_models.RequestOk)
async def delete_schema_composite(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.database_schema.composites:
        raise HTTPException(status_code=404, detail="Composite not found")
    
    del database_model.database_schema.composites[id]
    model_storage.set(database, database_model)
    return schema_models.RequestOk(message=f"Composite deleted")

# ################################################################################################
# # --------------------------------- SCHEMA OPERATIONS [END] -----------------------------------#
# ################################################################################################

# ################################################################################################
# # --------------------------------- DATA OPERATIONS [BEGIN] -----------------------------------#
# ################################################################################################

@router.get("/databases/{database}/data", response_model=typed_models.SchemaData)
async def get_data(database: str, model_storage: ModelStorage = Depends(env.get_model_storage)):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    return database_model.data

@router.patch("/databases/{database}/data", response_model=typed_models.SchemaData)
async def update_data(
    database: str,
    data: typed_models.SchemaData,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.data = data
    model_storage.set(database, database_model)
    return data

@router.get("/databases/{database}/data/primitives", response_model=Dict[str, typed_models.Primitive])
async def get_data_primitives(
    database: str,
    ptype: Optional[str] = Query(None),
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    return dict(
        filter(
            lambda p: (ptype is None) or (p[1].ptype == ptype),
            database_model.data.primitives.items()
        )
    )

@router.get("/databases/{database}/data/composites", response_model=Dict[str, typed_models.Composite])
async def get_data_composites(
    database: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    return database_model.data.composites

@router.get("/databases/{database}/data/primitives/{id}", response_model=typed_models.Primitive)
async def get_data_primitive(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.data.primitives:
        raise HTTPException(status_code=404, detail="Primitive not found")
    
    return database_model.data.primitives[id]

@router.get("/databases/{database}/data/composites/{id}", response_model=typed_models.Composite)
async def get_data_composite(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.data.composites:
        raise HTTPException(status_code=404, detail="Composite not found")
    
    return database_model.data.composites[id]

@router.patch("/databases/{database}/data/primitives/{id}", response_model=typed_models.Primitive)
async def update_data_primitive(
    database: str,
    id: str,
    primitive: typed_models.Primitive,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.data.primitives[id] = database_model.update_properties(primitive)
    model_storage.set(database, database_model)
    return primitive

@router.patch("/databases/{database}/data/composites/{id}", response_model=typed_models.Composite)
async def update_data_composite(
    database: str,
    id: str,
    composite: typed_models.Composite,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    database_model.data.composites[id] = database_model.update_properties(composite)
    model_storage.set(database, database_model)
    return composite

@router.delete("/databases/{database}/data/primitives/{id}", response_model=schema_models.RequestOk)
async def delete_data_primitive(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.data.primitives:
        raise HTTPException(status_code=404, detail="Primitive not found")
    
    del database_model.data.primitives[id]
    model_storage.set(database, database_model)
    return schema_models.RequestOk(message=f"Primitive deleted")

@router.delete("/databases/{database}/data/composites/{id}", response_model=schema_models.RequestOk)
async def delete_data_composite(
    database: str,
    id: str,
    model_storage: ModelStorage = Depends(env.get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    database_model: typed_models.DatabaseModel = model_storage.get(database)
    if not id in database_model.data.composites:
        raise HTTPException(status_code=404, detail="Composite not found")
    
    del database_model.data.composites[id]
    model_storage.set(database, database_model)
    return schema_models.RequestOk(message=f"Composite deleted")


################################################################################################
# ---------------------------------- CALCULATIONS [START] -------------------------------------#
################################################################################################

@router.post(
    "/databases/{database}/evaluate",
    description="Evaluate a combination in database {database}",
    response_model=database_search.DatabaseSearchResponse,
)
async def evaluate(
    database: str,
    combination: untyped_models.EvaluateRequest,
    model_storage: ModelStorage = Depends(env.get_model_storage),
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # Get data from database and convert it to a PLDAG model
    model: typed_models.DatabaseModel = model_storage.get(database)
    pldag_model = model.to_pldag()

    # Evaluate the combination
    evaluated = pldag_model.propagate(combination.to_complex())

# @router.post(
#     "/databases/{database}/search",
#     description="Search for a combination in database {database}",
#     response_model=database_search.DatabaseSearchResponse,
# )
# async def search(
#     database: str, 
#     query: database_search.DatabaseSearchRequest = Body(...), 
#     model_storage: ModelStorage = Depends(env.get_model_storage),
#     schema_storage: SchemaStorage = Depends(get_schema_storage)
# ):
#     if not schema_storage.exists(database):
#         raise HTTPException(status_code=404, detail="Database not found")

#     database_schema: schema_models.Database = schema_storage.get(database)

#     if not model_storage.exists(database):
#         raise HTTPException(status_code=404, detail="Database has no data")
    
#     # Get data from database
#     model: typed_models.DatabaseModel = model_storage.get(database)

#     # Append input data propositions to the model
#     if suchthat := query.suchthat:
#         if suchthat.assume.data is not None:
#             model = model.merge_data(suchthat.assume.data)

#     # Construct a pldag model from the data model
#     pldag_model: PLDAG = model.to_pldag()

#     # Construct the assume ID from suchthat data
#     assume = {}
#     if suchthat := query.suchthat:
#         assume_id = pldag_model.id_from_alias(suchthat.assume.return_)
#         if assume_id is None:
#             raise HTTPException(status_code=400, detail=f"Return proposition '{suchthat.assume.return_}' not found in model")
#         assume[assume_id] = complex(suchthat.to_equal.lower, suchthat.to_equal.upper)

#     try:
#         solutions = env.solver.solve(
#             model=pldag_model,
#             assume=assume,
#             objectives=query.objectives_dict(model), 
#             maximize=query.direction == "maximize",
#         )
#     except Exception as e:
#         logger.error(f"Solver error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Solver error. Please check logs.")

#     return database_search.DatabaseSearchResponse.from_untyped_solutions(
#         solutions=[
#             untyped_models.SolutionResponse(
#                 solution=dict(
#                     starmap(
#                         lambda k,v: (
#                             pldag_model.id_to_alias(k) or k,
#                             v
#                         ),
#                         filter(
#                             lambda x: not pldag_model._svec[pldag_model._imap[x[0]]],
#                             solution.solution.items()
#                         ),
#                     )
#                 ),
#                 error=solution.error
#             ) for solution in solutions
#         ],
#         model=model,
#         schema=database_schema.database_schema
#     )

################################################################################################
# ----------------------------------- CALCULATIONS [END] --------------------------------------#
################################################################################################