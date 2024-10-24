import logging
import api.models.schemas as schema_models

from fastapi import APIRouter, HTTPException
from pldag_solver_sdk import Solver as PLDAGSolver 
from pldag import PLDAG, CompilationSetting
from itertools import starmap

from api.settings import EnvironmentVariables

router = APIRouter()
env = EnvironmentVariables()
logger = logging.getLogger(__name__)

@router.post("/tools/search", response_model=schema_models.SolverProblemResponse)
def solve(model_problem: schema_models.ToolsSearchModel):
    try:
        solver = PLDAGSolver(url=env.SOLVER_API_URL)
        model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
        for proposition in model_problem.model:
            proposition.set_model(model)

        assume_id = None
        if model_problem.problem.assume:
            assume_id = model_problem.problem.assume.proposition.set_model(model)

        model.compile()
        try:
            solutions = solver.solve(
                model, 
                model_problem.problem.objectives, 
                {assume_id: model_problem.problem.assume.bounds.to_complex()} if assume_id else {},
                maximize=model_problem.problem.direction.value == "maximize"
            )
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            raise HTTPException(status_code=500, detail="No solver cluster available / could not solve")

        return schema_models.SolverProblemResponse(
            solution_responses=[
                schema_models.SolutionResponse(
                    solution=dict(
                        starmap(
                            lambda k,v: (
                                model.id_to_alias(k) or k,
                                v
                            ),
                            solution.solution.items(),
                        )
                    ),
                    error=solution.error
                ) for solution in solutions
            ]
        )
    except Exception as e:
        logger.error(f"Solver error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))