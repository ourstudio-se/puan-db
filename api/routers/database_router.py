import base64
import logging
import api.models.database as database_models
import api.models.schemas as schema_models

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List
from pldag import PLDAG, CompilationSetting
from pldag_solver_sdk import Solver as PLDAGSolver 
from api.settings import EnvironmentVariables

router = APIRouter()
env = EnvironmentVariables()
logger = logging.getLogger(__name__)

# Dependency to get DB session
def get_db():
    db = database_models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/databases", response_model=schema_models.Database)
def create_database(database: schema_models.DatabaseCreate, db: Session = Depends(get_db)):
    db_database = database_models.Database(name=database.name, description=database.description)
    db.add(db_database)
    db.commit()
    db.refresh(db_database)
    return db_database

@router.get("/databases", response_model=List[schema_models.DatabaseResponse])
def get_databases(db: Session = Depends(get_db)):
    db_databases = db.query(database_models.Database).all()
    if not db_databases:
        raise HTTPException(status_code=404, detail="No databases found")
    return db_databases

@router.get("/databases/{database_id}", response_model=schema_models.DatabaseResponse)
def read_database(database_id: int, db: Session = Depends(get_db)):
    db_database = db.query(database_models.Database).filter(database_models.Database.id == database_id).first()
    if db_database is None:
        raise HTTPException(status_code=404, detail="Database not found")
    return db_database

@router.post("/databases/{database_id}/versions", response_model=schema_models.VersionResponse)
def create_version(database_id: str, version: schema_models.VersionCreate, db: Session = Depends(get_db)):
    model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
    for proposition in version.propositions:
        proposition.set_model(model)

    model.compile()
    version_hash = model.sha1()

    # Create or update the Version object
    db_version = database_models.Version(
        hash=version_hash,
        database_id=int(database_id),
        parent_hash=version.parent_hash,
        message=version.message,
        data=model.dump()
    )

    try:
        # Use merge to insert or update based on primary key (hash)
        merged_version = db.merge(db_version)
        db.commit()
        db.refresh(merged_version)
        return merged_version

    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="A database integrity error occurred.")

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    
@router.post("/databases/{database_id}/versions/from-bytes", response_model=schema_models.VersionResponse)
def create_version_from_bytes(database_id: str, version: schema_models.VersionCreateFromBytes, db: Session = Depends(get_db)):
    try:
        # Load the model from the received bytes data
        model = PLDAG.load(base64.b64decode(version.data))  # Assuming PLDAG has a method to load from bytes

        # Compile the model (if necessary)
        model.compile()
        version_hash = model.sha1()

        # Create or update the Version object
        db_version = database_models.Version(
            hash=version_hash,
            database_id=int(database_id),
            parent_hash=version.parent_hash,
            message=version.message,
            data=model.dump()  # Store the received bytes directly
        )

        merged_version = db.merge(db_version)
        db.commit()
        db.refresh(merged_version)
        return merged_version

    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="A database integrity error occurred.")

    except Exception as e:
        db.rollback()
        print("Unexpected error: ", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/databases/{database_id}/versions/{hash}", response_model=schema_models.Graph)
def read_version(hash: str, db: Session = Depends(get_db)):
    db_version = db.query(database_models.Version).filter(database_models.Version.hash == hash).first()
    if db_version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return schema_models.Graph.from_model(
        version=schema_models.VersionResponse(
            hash=db_version.hash,
            database_id=db_version.database_id,
            parent_hash=db_version.parent_hash,
            message=db_version.message,
            created_at=db_version.created_at
        ),
        model=PLDAG.load(db_version.data),
    )

@router.post("/databases/{database_id}/versions/{hash}/search", response_model=schema_models.SolverProblemResponse)
def solve(hash: str, problem: schema_models.SolverProblemRequest, db: Session = Depends(get_db)):
    db_version = db.query(database_models.Version).filter(database_models.Version.hash == hash).first()
    if db_version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    
    try:
        solver = PLDAGSolver(url=env.SOLVER_API_URL)
        model = PLDAG.load(db_version.data)
        try:
            solutions = solver.solve(
                model, 
                problem.objectives, 
                {problem.assume.proposition.set_model(model): problem.assume.bounds.to_complex()} if problem.assume else {},
                maximize=problem.direction.value == "maximize"
            )
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            raise HTTPException(status_code=500, detail="No solver cluster available")

        return schema_models.SolverProblemResponse(
            solution_responses=[
                schema_models.SolutionResponse(
                    solution=solution.solution,
                    error=solution.error
                ) for solution in solutions
            ]
        )
    except Exception as e:
        logger.error(f"Solver error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))