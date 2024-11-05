import logging
import api.models.schema as schema_models
import api.models.typed_model as typed_models
import api.models.query as query_models
import api.models.database_search as database_search
import api.models.untyped_model as untyped_models

from api.settings import EnvironmentVariables
from api.storage.databases import SchemaStorage, ModelStorage

from pldag import PLDAG
from fastapi import APIRouter, HTTPException, Depends, Body
from itertools import starmap
from typing import List, Optional

router = APIRouter()
env = EnvironmentVariables()
logger = logging.getLogger(__name__)

def get_schema_storage():
    yield SchemaStorage(env.DATABASE_URL)

def get_model_storage():
    yield ModelStorage(env.DATABASE_URL)

@router.get("/databases", response_model=List[str])
async def get_databases(storage: SchemaStorage = Depends(get_schema_storage)):
    return storage.get_keys()

@router.post("/databases", response_model=schema_models.RequestOk)
async def create_database(schema_request: schema_models.DatabaseSchema, storage: SchemaStorage = Depends(get_schema_storage)):
    
    if storage.exists(schema_request.name):
        raise HTTPException(status_code=409, detail="Database already exists")
    
    try:
        storage.set(schema_request.name, schema_request)
        return schema_models.RequestOk(message=f"Database created")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating database: {str(e)}")
    
@router.delete("/databases/{database}", response_model=schema_models.RequestOk)
async def delete_database(
    database: str, 
    schema_storage: SchemaStorage = Depends(get_schema_storage),
    model_storage: ModelStorage = Depends(get_model_storage)
):
    try:
        model_storage.delete(database)
        schema_storage.delete(database)
        return schema_models.RequestOk(message=f"Database deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting database: {str(e)}")
    
@router.patch("/databases/{database}", response_model=schema_models.RequestOk)
async def update_database(
    database: str, 
    schema: schema_models.Schema = Body(...), 
    schema_storage: SchemaStorage = Depends(get_schema_storage),
    model_storage: ModelStorage = Depends(get_model_storage),
):

    current_database_schema: schema_models.DatabaseSchema = schema_storage.get(database)
    if current_database_schema is None:
        raise HTTPException(status_code=404, detail="Database not found")
    
    current_model: typed_models.Model = model_storage.get(database)
    if current_model is not None:
        # Validate data before updating schema
        updated_model = typed_models.Model(
            model_schema=schema,
            data=current_model.data,
        )
        # Set model with updated schema
        model_storage.set(database, updated_model)
    
    current_database_schema.schema = schema
    schema_storage.set(database, current_database_schema)
    return schema_models.RequestOk(message=f"Database updated")
    
@router.get("/databases/{database}", response_model=typed_models.Model, description="Get schema and data from the database {database}")
async def get_database(database: str, schema_storage: SchemaStorage = Depends(get_schema_storage), model_storage: ModelStorage = Depends(get_model_storage)):
    
    if not schema_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # Get data if it exists
    data = model_storage.get(database)
    if data is None:
        raise HTTPException(status_code=404, detail="Database has no data")
    
    # Get data
    return data

@router.get("/databases/{database}/schema", response_model=schema_models.DatabaseSchema)
async def get_database(database: str, storage: SchemaStorage = Depends(get_schema_storage)):
    schema: schema_models.DatabaseSchema = storage.get(database)
    if schema is None:
        raise HTTPException(status_code=404, detail="Database not found")
    return schema

@router.get("/databases/{database}/data", response_model=typed_models.Model)
async def get_data(database: str, model_storage: ModelStorage = Depends(get_model_storage)):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database data not found")
    
    # Get data from database
    return model_storage.get(database)
    
@router.post(
    "/databases/{database}/data/insert",
    description="Insert data into the database. Validation is done against the schema before insertion.",
)
async def insert(
    database: str, 
    data: typed_models.SchemaData, 
    schema_storage: SchemaStorage = Depends(get_schema_storage),
    model_storage: ModelStorage = Depends(get_model_storage)
):
    
    if not schema_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # We need to fetch schema to validate data
    database_schema: schema_models.DatabaseSchema = schema_storage.get(database)

    # Need to fetch current data to merge with new data
    current_model: typed_models.Model = model_storage.get(database)
    if current_model is not None:
        # Merge data with existing data
        # Will raise an exception if data is invalid
        model = current_model.merge_data(data)
    else:
        # Attach data and schema into a new model
        # Will raise an exception if data is invalid
        model = typed_models.Model(
            model_schema=database_schema.schema, 
            data=data,
        )

    # Insert data into database
    model_storage.set(database, model)
    return schema_models.RequestOk(message=f"Data inserted")

@router.post(
    "/databases/{database}/data/overwrite",
    description="Overwrite data in the database. Validation is done against the schema before insertion."
)
async def overwrite(
    database: str, 
    data: typed_models.SchemaData, 
    schema_storage: SchemaStorage = Depends(get_schema_storage),
    model_storage: ModelStorage = Depends(get_model_storage)
):
    
    if not schema_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # Validate each data element against the schema
    database_schema: schema_models.DatabaseSchema = schema_storage.get(database)

    # Create model from data
    # Will raise an exception if data is invalid
    model = typed_models.Model(
        model_schema=database_schema.schema, 
        data=data,
    )

    # Insert data into database
    model_storage.set(database, model)
    return schema_models.RequestOk(message=f"Data inserted")

@router.post(
    "/databases/{database}/data/validate",
    description="Validate data before inserting into database. Merges with existing data if it exists before validation."
)
async def validate(
    database: str, 
    data: typed_models.SchemaData, 
    schema_storage: SchemaStorage = Depends(get_schema_storage),
):
    
    if not schema_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # Validate each data element against the schema
    database_schema: schema_models.DatabaseSchema = schema_storage.get(database)

    # Test to construct a Model from data and schema
    # Will raise an exception if data is invalid
    typed_models.Model(
        model_schema=database_schema.schema, 
        data=data,
    )

    return schema_models.RequestOk(message=f"Data is valid")

@router.get(
    "/databases/{database}/data/items/{id}", 
    response_model=typed_models.CompPrimitive,
    description="Get element {id} from the database {database}"
)
async def get_data_item(
    database: str, 
    id: str,
    model_storage: ModelStorage = Depends(get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database has no data")
    
    # Get data from database
    model: typed_models.Model = model_storage.get(database)
    
    # Get item, None if not found
    item = model.data.get(id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Return data
    return item

@router.patch(
    "/databases/{database}/data/items/{id}", 
    response_model=schema_models.RequestOk,
    description="Update element {id} from the database {database}"
)
async def update_data_item(
    database: str,
    id: str,
    proposition: typed_models.CompPrimitive, 
    model_storage: ModelStorage = Depends(get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database has no data")
    
    # Get data from database
    model: typed_models.Model = model_storage.get(database)
    
    # Try to find item
    item = model.data.get(id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {proposition.id} not found")
    
    # Update data
    # Will raise an exception if data is invalid
    updated_model: typed_models.Model = model.update({id: proposition})
    
    # No errors, save data
    model_storage.set(database, updated_model)
    
    return schema_models.RequestOk(message=f"Data updated")

@router.delete(
    "/databases/{database}/data/items/{id}", 
    response_model=schema_models.RequestOk,
    description="Deletes element {id} from the database {database}"
)
async def delete_data_item(
    database: str, 
    id: str, 
    model_storage: ModelStorage = Depends(get_model_storage)
):
    
    # Get data from database
    model: typed_models.Model = model_storage.get(database)
    
    # Check if data exists
    if not model.data.exists(id):
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Delete data
    updated_model = model.delete([id])
    
    # No errors, save data
    model_storage.set(database, updated_model)
    
    return schema_models.RequestOk(message=f"Data deleted")

@router.get(
    "/databases/{database}/data/items",
    description="Search for data items in the database {database}"
)
async def search_data_items(
    database: str, 
    query: Optional[query_models.SearchQuery] = Body(None), 
    model_storage: ModelStorage = Depends(get_model_storage)
):
    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database has no data")
    
    # Get data from database
    model: typed_models.Model = model_storage.get(database)
    
    # Search data
    if query is None:
        return model.data
    
    return model.search_items(query)

@router.post(
    "/databases/{database}/search",
    description="Search for a combination in database {database}",
    response_model=database_search.DatabaseSearchResponse,
)
async def search(
    database: str, 
    query: database_search.DatabaseSearchRequest = Body(...), 
    model_storage: ModelStorage = Depends(get_model_storage),
    schema_storage: SchemaStorage = Depends(get_schema_storage)
):
    if not schema_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database not found")

    database_schema: schema_models.DatabaseSchema = schema_storage.get(database)

    if not model_storage.exists(database):
        raise HTTPException(status_code=404, detail="Database has no data")
    
    # Get data from database
    model: typed_models.Model = model_storage.get(database)

    # Append input data propositions to the model
    if suchthat := query.suchthat:
        if suchthat.assume.data is not None:
            model = model.merge_data(suchthat.assume.data)

    # Construct a pldag model from the data model
    pldag_model: PLDAG = model.to_pldag()

    # Construct the assume ID from suchthat data
    assume = {}
    if suchthat := query.suchthat:
        assume_id = pldag_model.id_from_alias(suchthat.assume.return_)
        if assume_id is None:
            raise HTTPException(status_code=400, detail=f"Return proposition '{suchthat.assume.return_}' not found in model")
        assume[assume_id] = complex(suchthat.to_equal.lower, suchthat.to_equal.upper)

    try:
        solutions = env.solver.solve(
            model=pldag_model,
            assume=assume,
            objectives=query.objectives_dict(model), 
            maximize=query.direction == "maximize",
        )
    except Exception as e:
        logger.error(f"Solver error: {str(e)}")
        raise HTTPException(status_code=500, detail="Solver error. Please check logs.")

    return database_search.DatabaseSearchResponse.from_untyped_solutions(
        solutions=[
            untyped_models.SolutionResponse(
                solution=dict(
                    starmap(
                        lambda k,v: (
                            pldag_model.id_to_alias(k) or k,
                            v
                        ),
                        filter(
                            lambda x: not pldag_model._svec[pldag_model._imap[x[0]]],
                            solution.solution.items()
                        ),
                    )
                ),
                error=solution.error
            ) for solution in solutions
        ],
        model=model,
        schema=database_schema.schema
    )