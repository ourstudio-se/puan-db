import os
import grpc
import logging
import puan_db_pb2_grpc
import puan_db_pb2
import pickle
import gzip

from concurrent import futures
from dataclasses import dataclass
from itertools import chain, starmap
from pldag import PLDAG, Solver, NoSolutionsException
from typing import Optional, List, Dict

@dataclass
class LocalModelHandler:

    model_path: str

    def load_model(self) -> PLDAG:
        try:
            with open(self.model_path, "rb") as f:
                return pickle.loads(gzip.decompress(f.read()))
        except:
            new_model = PLDAG()
            self.save_model(new_model)
            return new_model
        
    def save_model(self, model: PLDAG):
        with open(self.model_path, "wb") as f:
            f.write(gzip.compress(pickle.dumps(model)))

    def modify(self, f):
        model = self.load_model()
        result = f(model)
        self.save_model(model)
        return result
    
    def compute(self, f):
        return f(self.load_model())

class PuanDB(puan_db_pb2_grpc.ModelingService):

    def __init__(self, model_handler: LocalModelHandler):
        self.model_handler = model_handler

    @staticmethod
    def bound_complex(bound: Optional[puan_db_pb2.Bound]) -> Optional[complex]:
        return complex(bound.lower, bound.upper) if bound else None
    
    @staticmethod
    def complex_bound(cmplx: complex) -> puan_db_pb2.Bound:
        return puan_db_pb2.Bound(
            lower=int(cmplx.real),
            upper=int(cmplx.imag),
        )
    
    @staticmethod
    def dict_interpretation(d: Dict[str, complex]) -> puan_db_pb2.Interpretation:
        return puan_db_pb2.Interpretation(
            variables=list(
                starmap(
                    lambda k,v: puan_db_pb2.Variable(
                        id=k,
                        bound=PuanDB.complex_bound(v),
                    ),
                    d.items()
                )
            )
        )
    
    @staticmethod
    def interpretation_dict(interpretation: puan_db_pb2.Interpretation) -> Dict[str, complex]:
        return dict(
            map(
                lambda x: (x.id, PuanDB.bound_complex(x.bound)),
                interpretation.variables
            )
        )
    
    @staticmethod
    def dict_objective(d: Dict[str, int]) -> puan_db_pb2.Objective:
        return puan_db_pb2.Objective(
            variables=list(
                starmap(
                    lambda k,v: puan_db_pb2.FixedVariable(
                        id=k,
                        value=v,
                    ),
                    d.items()
                )
            )
        )
    
    @staticmethod
    def objective_dict(objective: puan_db_pb2.Objective) -> Dict[str, int]:
        return dict(
            map(
                lambda x: (x.id, x.value),
                objective.variables
            )
        )
    
    def SetPrimitive(self, request: puan_db_pb2.Primitive, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_primitive(request.id, PuanDB.bound_complex(request.bound))
            )
        )

    def SetPrimitives(self, request: puan_db_pb2.Primitive, context):
        return puan_db_pb2.IDsResponse(
            ids=self.model_handler.modify(
                lambda model: model.set_primitives(request.ids, PuanDB.bound_complex(request.bound))
            )
        )
    
    def SetAtLeast(self, request: puan_db_pb2.AtLeast, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_atleast(
                    request.references, 
                    request.value, 
                    request.alias,
                )
            )
        )
    
    def SetAtMost(self, request: puan_db_pb2.AtMost, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_atmost(
                    request.references, 
                    request.value, 
                    request.alias,
                )
            )
        )
    
    def SetAnd(self, request: puan_db_pb2.And, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_and(
                    request.references, 
                    request.alias,
                )
            )
        )
    
    def SetOr(self, request: puan_db_pb2.Or, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_or(
                    request.references, 
                    request.alias,
                )
            )
        )
    
    def SetXor(self, request: puan_db_pb2.Xor, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_xor(
                    request.references, 
                    request.alias,
                )
            )
        )
    
    def SetNot(self, request: puan_db_pb2.Not, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_not(
                    request.references, 
                    request.alias,
                )
            )
        )
    
    def SetImply(self, request: puan_db_pb2.Imply, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_imply(
                    request.condition, 
                    request.consequence, 
                    request.alias,
                )
            )
        )
    
    def SetEqual(self, request: puan_db_pb2.Equivalent, context):
        return puan_db_pb2.SetResponse(
            id=self.model_handler.modify(
                lambda model: model.set_equal(
                    [
                        request.lhs,
                        request.rhs,
                    ], 
                    request.alias,
                )
            )
        )
    
    def PropagateUpstream(self, request: puan_db_pb2.Interpretation, context):
        return PuanDB.dict_interpretation(
            self.model_handler.compute(
                lambda model: model.propagate_upstream(PuanDB.interpretation_dict(request))
            )
        )

    def Propagate(self, request: puan_db_pb2.Interpretation, context):
        return PuanDB.dict_interpretation(
            self.model_handler.compute(
                lambda model: model.propagate(PuanDB.interpretation_dict(request))
            )
        )
    
    def PropagateBidirectional(self, request: puan_db_pb2.Interpretation, context):
        return PuanDB.dict_interpretation(
            self.model_handler.compute(
                lambda model: model.propagate_bistream(PuanDB.interpretation_dict(request))
            )
        )
    
    def Solve(self, request: puan_db_pb2.SolveRequest, context):
        try:
            return puan_db_pb2.SolveResponse(
                solutions=list(
                    map(
                        PuanDB.dict_interpretation,
                        self.model_handler.compute(
                            lambda model: model.solve(
                                list(map(PuanDB.objective_dict, request.objectives)), 
                                PuanDB.interpretation_dict(request.assume), 
                                Solver[puan_db_pb2.Solver.Name(request.solver)],
                            )
                        )
                    )
                )
            )
        except NoSolutionsException as nse:
            return puan_db_pb2.SolveResponse(
                error=str(nse)
            )
        except ValueError as ve:
            return puan_db_pb2.SolveResponse(
                error=str(ve)
            )
        except Exception as e:
            logging.log(logging.ERROR, e)
            return puan_db_pb2.SolveResponse(
                error="Something went wrong"
            )

    def Delete(self, request, context):
        return puan_db_pb2.BooleanSetResponse(
            success=self.model_handler.modify(
                lambda model: model.delete(request.id)
            )
        )
    
    def Get(self, request, context):
        try:
            return PuanDB.complex_bound(
                self.model_handler.compute(
                    lambda model: model.get(request.id)
                )[0]
            )
        except:
            raise Exception(f"ID `{request.id}` not found")
    
    def GetMetaInformation(self, request, context):
        return puan_db_pb2.MetaInformationResponse(
            nrows=self.model_handler.compute(lambda model: model._amat.shape[0]),
            ncols=self.model_handler.compute(lambda model: model._amat.shape[1]),
        )
    
    def GetDependencies(self, request, context):
        return puan_db_pb2.IDsResponse(
            ids=self.model_handler.compute(lambda model: model.dependencies(request.id))
        )
    
    def GetIDFromAlias(self, request, context):
        return puan_db_pb2.IDResponse(
            id=self.model_handler.compute(lambda model: model.id_from_alias(request.alias))
        )
    
    def GetPrimitive(self, request, context):
        bound_complex = self.model_handler.compute(lambda model: model.get(request.id))
        return puan_db_pb2.Primitive(
            id=request.id,
            bound=puan_db_pb2.Bound(
                lower=int(bound_complex.real),
                upper=int(bound_complex.imag),
            )
        )
    
    def GetPrimitives(self, request, context):
        return puan_db_pb2.Primitives(
            primitives=list(
                map(
                    lambda x: puan_db_pb2.Primitive(
                        id=x[0],
                        bound=puan_db_pb2.Bound(
                            lower=int(x[1].real),
                            upper=int(x[1].imag),
                        )
                    ),
                    self.model_handler.compute(
                        lambda model: zip(
                            model.primitives, 
                            map(model.get, model.primitives)
                        )
                    )
                )
            )
        )
    
    def GetPrimitiveIds(self, request, context):
        return puan_db_pb2.IDsResponse(
            ids=self.model_handler.compute(lambda model: model.primitives.tolist())
        )
    
    def GetComposite(self, request, context):
        return puan_db_pb2.Composite(
            **self.model_handler.compute(
                lambda model: {
                    "id": request.id,
                    "references": model.dependencies(request.id),
                    "bias": int(model._bvec[model._col(request.id)]),
                    "negated": bool(model._nvec[request.id]),
                    "alias": model.id_to_alias(request.id),
                }
            )
        )
    
    def GetComposites(self, request, context):
        model = self.model_handler.load_model()
        return puan_db_pb2.Composites(
            composites=list(
                map(
                    lambda composite_id: puan_db_pb2.Composite(
                        **{
                            "id": composite_id,
                            "references": sorted(model.dependencies(composite_id)),
                            "bias": int(model._bvec[model._row(composite_id)].real),
                            "negated": bool(model._nvec[model._row(composite_id)]),
                            "alias": model.id_to_alias(composite_id),
                        }
                    ),
                    model.composites.tolist(),
                )
            )
        )

    def GetCompositeIds(self, request, context):
        return puan_db_pb2.IDsResponse(
            ids=self.model_handler.compute(lambda model: model.composites.tolist())
        )

def serve():
    port = os.getenv('APP_PORT', '50051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    puan_db_pb2_grpc.add_ModelingServiceServicer_to_server(
        PuanDB(LocalModelHandler("_default.puan")),
        server
    )
    server.add_insecure_port("[::]:" + port)
    server.start()
    logging.log(logging.INFO, "Server started, listening on " + port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()