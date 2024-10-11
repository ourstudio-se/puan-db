import logging
from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict

from api.models.search import (
    SearchDatabaseRequest,
    EvaluateDatabaseRequest,
    SearchDatabaseResponse,
)
from api.services.database_service import (
    DatabaseService,
    Composite,
    DatabaseExistsException,
    BranchExistsException,
    DatabaseDoesNotExistsException,
    BranchDoesNotExistsException,
    CommitDoesNotExistsException,
    VariableMissingException,
)

logger = logging.getLogger(__name__)
service = DatabaseService()

# Create an instance of APIRouter
router = APIRouter()

# Common creating and modifying operations
@router.get("/database")
async def get_databases():
    try:
        return service.get_databases()
    except Exception as e:
        logger.error(f"Unexpected error getting databases: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error getting databases")

@router.get("/database/{database}")
async def get_database_info(database: str):
    """Get info about {database}"""
    try:
        return service.get_database(database)
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=404, detail=f"Database '{database}' does not exist")
    except Exception as e:
        logger.error(f"Unexpected error getting database {database}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error getting database '{database}'")

@router.get("/database/{database}/branch/{branch}")
async def get_database_branch_propositions(database: str, branch: str):
    """Get propositions on latest commit for {database}'s database branch"""
    try:
        latest_commit = service.branch_latest_commit(database, branch)
        return service.get_commit(latest_commit)
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error getting branch '{branch}' for database '{database}': {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error getting latest commit for branch '{branch}' and database '{database}'")

@router.get("/commit/{commit}")
async def get_databases_commit_propositions(commit: str):
    """Get propositions on commit for {database}'s database branch"""
    try:
        return service.get_commit(commit)
    except CommitDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Commit '{commit}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error getting commit '{commit}': {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error getting commit '{commit} ")


@router.post("/database/{database}")
async def create_database(database: str):
    """Create a new database databased {database}. "main" branch is automatically created"""
    try:
        return service.create_database(database)
    except DatabaseExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' already exists")
    except Exception as e:
        logger.error(f"Unexpected error creating database {database}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error creating database '{database}'")

@router.post("/database/{database}/branch/{branch}/fromBranch/{from_branch}")
async def create_branch(database: str, branch: str, from_branch: str):
    """Create a new branch databased {branch} on database {database}"""
    try:
        return service.create_branch(database, branch, from_branch)
    except BranchExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' already exists")
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error creating database {database}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error creating branch '{branch}' on database '{database}'")

@router.post("/database/{database}/branch/{branch}/commit")
async def commit_to_branch(database: str, branch: str, propositions: List[Composite] = Body(...)) -> str:
    """Commit propositions to {database}'s branch {branch}."""
    try:
        return service.commit(database, branch, propositions)
    except VariableMissingException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error committing to branch {branch} and database {database}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error committing to branch '{branch}' and database '{database}'")
    

# @router.post("/database/{database}/branch/{branch}/rebase")
# async def rebase(database: str, branch: str, from_branch: str):
#     """ Rebases given branch (given in body) onto {branch}."""
#     raise NotImplementedError("This endpoint is not implemented yet")


@router.delete("/database")
async def delete_all():
    """ Delete all databases """
    try:
        return service.delete_all()
    except Exception as e:
        logger.error(f"Unexpected error deleting all databases: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error deleting all databases")

@router.delete("/database/{database}")
async def delete_database(database: str):
    """ Delete database {database} """
    try:
        return service.delete_database(database)
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error deleting database '{database}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error deleting database")

@router.delete("/database/{database}/branch/{branch}")
async def delete_branch(database: str, branch: str):
    """ Delete branch {branch} except 'main' branch (use delete_database instead) """
    try:
        return service.delete_branch(database, branch)
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error deleting branch '{branch}' for database '{database}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error deleting branch")

@router.delete("/commit/{commit}")
async def delete_commit(commit: str):
    """ Delete commit {commit} except root commit """
    try:
        return service.delete_commit(commit)
    except CommitDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Commit '{commit}' does not exists")
    except Exception as e:
        logger.error(f"Unexpected error deleting commit '{commit}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error deleting commit")


# Calculation and specific modification operations
@router.post("/database/{database}/branch/{branch}/search")
async def search(database: str, branch: str, search_request: SearchDatabaseRequest = Body(...)) -> SearchDatabaseResponse:
    """ Finds a combination that satisfies db's constraints, on branch {branch} latest commit. """
    try:
        return service.search(
            service.branch_latest_commit(database, branch), 
            search_request
        )
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except VariableMissingException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching database '{database}' branch '{branch}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error searching database")

@router.post("/database/{database}/branch/{branch}/evaluate")
async def evaluate(database: str, branch: str, evaluate_request: EvaluateDatabaseRequest) -> EvaluateDatabaseRequest:
    """  Evaluates a combination on db's constraints, on commit {commit}. Default is latest commit. """
    try:
        return service.evaluate(
            service.branch_latest_commit(database, branch),
            evaluate_request
        )
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except VariableMissingException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching database '{database}' branch '{branch}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error evaluating database")

@router.post("/database/{database}/branch/{branch}/subTo/{newBranch}")
async def sub(database: str, branch: str, newBranch: str, sub_request: List[str] = Body(...)):
    """ Keeps only sub graph(s) under given nodes. This will be stored on a new branch given in body. """
    try:
        return service.sub_database(
            database, 
            branch,
            newBranch,
            sub_request,
        )
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except BranchExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{newBranch}' does not exists")
    except VariableMissingException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching database '{database}' branch '{branch}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error subing database")

@router.post("/database/{database}/branch/{branch}/cutTo/{newBranch}")
async def cut(database: str, branch: str, newBranch: str, cut_request: Dict[str, str] = Body(...)):
    """ Removes, or cuts, sub graph(s) under given nodes. This will be stored on a new branch given in body. """
    try:
        return service.cut_database(
            database, 
            branch,
            newBranch,
            cut_request,
        )
    except DatabaseDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Database '{database}' does not exists")
    except BranchDoesNotExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{branch}' does not exists")
    except BranchExistsException:
        raise HTTPException(status_code=400, detail=f"Branch '{newBranch}' does not exists")
    except VariableMissingException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching database '{database}' branch '{branch}': {e}")
        raise HTTPException(status_code=500, detail="Unexpected error cutting database")
