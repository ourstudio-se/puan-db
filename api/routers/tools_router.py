import logging
import api.models.untyped_model as schema_models

from fastapi import APIRouter, HTTPException, Query
from pldag import PLDAG, CompilationSetting
from itertools import starmap

from api.settings import EnvironmentVariables
from api.tools import timer

router = APIRouter()
env = EnvironmentVariables()
logger = logging.getLogger(__name__)

@router.post("/tools/search", response_model=schema_models.SolverProblemResponse)
def solve(model_problem: schema_models.ToolsSearchModel, only_primitives: bool = Query(False, alias="onlyPrimitives")):
    try:
        model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
        for proposition in model_problem.model:
            proposition.set_model(model)

        assume_id = None
        if model_problem.problem.assume:
            assume_id = model_problem.problem.assume.proposition.set_model(model)

        model.compile()
        assume = {assume_id: model_problem.problem.assume.bounds.to_complex()} if assume_id else {}

        try:
            solutions = env.solver.solve(
                model=model,
                assume=assume,
                objectives=model_problem.problem.objectives,
                maximize=model_problem.problem.direction.value == "maximize",
            )
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            raise HTTPException(status_code=500, detail="Solver error. Please check logs.")

        primitive_variables = model.primitives.tolist()
        return schema_models.SolverProblemResponse(
            solution_responses=[
                schema_models.SolutionResponse(
                    solution=dict(
                        starmap(
                            lambda k,v: (
                                model.id_to_alias(k) or k,
                                v
                            ),
                            filter(
                                lambda x: (not model._svec[model._imap[x[0]]]) and (x[1] != 0) and (not only_primitives or x[0] in primitive_variables),
                                solution.solution.items()
                            ),
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