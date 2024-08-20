from fastapi import APIRouter, HTTPException, Body
from pldag import MissingVariableException, FailedToCompileException, IsReferencedException
from typing import List, Union
from api.models.system import (
    CreateDatabaseRequest,
    CreateDatabaseResponse,
    Database,
    RemovePropositionsRequest,
    Primitive,
    BinaryComposite,
    LogicalComposite,
    ValueComposite,
    LinearInequalityComposite,
    DeleteDatabaseRequest
)
from api.models.search import (
    SearchDatabaseRequest,
    SearchDatabaseResponse,
    EvaluateDatabaseRequestResponse
)
from api.services.database_service import (
    SetCompositeException, 
    DatabaseAlreadyExistsException,
    DatabaseNotFoundException
)
from api.settings import PuanDBSettings

# Create an instance of APIRouter
router = APIRouter()

@router.get("/")
async def list_databases():
    return PuanDBSettings.instance().service.databases()

@router.post("/")
async def create_database(request: CreateDatabaseRequest):
    try:
        return CreateDatabaseResponse(
            id=PuanDBSettings.instance().service.create(
                name=request.name,
                description=request.description
            )
        )
    except DatabaseAlreadyExistsException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail="Somthing went wrong")
    
@router.delete("/")
async def delete_database(request: DeleteDatabaseRequest) -> bool:
    try:
        PuanDBSettings.instance().service.delete(request.id)
        return True
    except DatabaseNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error deleting all databases: {e}")
        raise HTTPException(status_code=500, detail="Error deleting all databases")

@router.get("/{id}")
async def get_database(id: str):
    try:
        return PuanDBSettings.instance().service.load(id)
    except DatabaseNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error loading database: {e}")
        raise HTTPException(status_code=500, detail="Error loading database")
    
@router.post("/{id}/search")
async def search_database(id: str, request: SearchDatabaseRequest) -> SearchDatabaseResponse:
    
    try:
        return SearchDatabaseResponse(
            solutions=PuanDBSettings.instance().service.search(
                id=id,
                request=request
            )
        )
    except MissingVariableException as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error search: {e}")
        raise HTTPException(status_code=500, detail="Error searching database")
    
@router.post("/{id}/evaluate")
async def evaluate_database(id: str, request: EvaluateDatabaseRequestResponse) -> EvaluateDatabaseRequestResponse:
    try:
        return PuanDBSettings.instance().service.evaluate(
            id=id,
            request=request
        )
    except DatabaseNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))

    except MissingVariableException as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error evaluating: {e}")
        raise HTTPException(status_code=500, detail="Error evaluating database")

@router.post("/{id}")
async def set_propositions(id: str, propositions: List[Union[str, Primitive, BinaryComposite, LogicalComposite, ValueComposite, LinearInequalityComposite]] = Body(...)):

    # Update with new data
    try:
        updated_model: Database = PuanDBSettings.instance().service.update(id, propositions)

        # Compile model for effect
        return updated_model
    
    except DatabaseNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))

    except MissingVariableException as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except FailedToCompileException as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except SetCompositeException as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error updating on commit: {e}")
        raise HTTPException(status_code=500, detail="Error updating database")
    
@router.delete("/{id}")
async def delete_propositions(id: str, request: RemovePropositionsRequest) -> Database:

    """
        Deleting propositions from a database.
    """
    
    try:
        return PuanDBSettings.instance().service.remove_propositions(id, request.propositions)
        
    except IsReferencedException as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        PuanDBSettings.instance().logger.error(f"Error removing propositions: {e}")
        raise HTTPException(status_code=500, detail="Error removing propositions")
