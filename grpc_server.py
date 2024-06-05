import os
import grpc
import logging
import puan_db_pb2_grpc
import puan_db_pb2

from concurrent import futures
from itertools import starmap
from pldag import Puan, Solver, NoSolutionsException, Solution
from typing import Optional, Dict, List, Union
from storage import LocalModelHandler, AzureBlobModelHandler, ComputingDevice

        
def _unary_unary_rpc_terminator(code, details):
    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(terminate)
    
class DataFetchingInterceptor(grpc.ServerInterceptor):

    def __init__(self, model_handler: LocalModelHandler):
        self.model_handler = model_handler

    def intercept_service(self, continuation, handler_call_details):
        # Exclude certain methods from requiring model-id
        if handler_call_details.method not in ["/ModelingService/ModelCreate", "/ModelingService/ModelSelect"]:
            metadata_dict = dict(handler_call_details.invocation_metadata)
            token = metadata_dict.get('token')
            if not token:
                return _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, "Token is missing")
            
            # Fetch data based on token
            if not self.model_handler.verify_token(token):
                return _unary_unary_rpc_terminator(grpc.StatusCode.INTERNAL, "Invalid token")

            return continuation(handler_call_details)
        
        # Continue processing the request
        return continuation(handler_call_details)

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
    def solution_interpretation(solution: Solution) -> puan_db_pb2.Interpretation:
        return puan_db_pb2.Interpretation(
            variables=list(
                map(
                    lambda variable: puan_db_pb2.Variable(
                        id=variable.id,
                        bound=PuanDB.complex_bound(variable.bound),
                    ),
                    solution.variables,
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
    
    @staticmethod
    def properties_dict(properties: List[puan_db_pb2.Property]) -> Dict[str, str]:

        def unravel_value(value):
            if value.HasField('string_value'):
                return value.string_value
            if value.HasField('int_value'):
                return value.int_value
            
            return None

        return dict(
            map(
                lambda x: (x.key, unravel_value(x.value)),
                properties,
            )
        )
    
    @staticmethod
    def dict_properties(d: Dict[str, str]) -> List[puan_db_pb2.Property]:
        return list(
            starmap(
                lambda k,v: puan_db_pb2.Property(
                    key=k,
                    value=puan_db_pb2.StringOrIntPropertyValue(
                        string_value=v if isinstance(v, str) else None,
                        int_value=v if isinstance(v, int) else None,
                    ),
                ),
                d.items()
            )
        )
    
    @staticmethod
    def find_from_predicate(model: Puan, predicate: puan_db_pb2.Predicate):

        def evaluate(key: str, pred: puan_db_pb2.Predicate.Operand):

            if pred.HasField('predicate'):
                return validate(key, pred.predicate)
            
            elif pred.HasField('value_of'):
                return model.data.get(key, {}).get(pred.value_of.key, None)
            
            elif pred.HasField('number'):
                return pred.number
            
            elif pred.HasField('text'):
                return pred.text
            
            return None
            

        def validate(key: str, predicate: puan_db_pb2.Predicate):

            val_map = {
                puan_db_pb2.Predicate.BinaryOperator.EQ: lambda lhs_evd, rhs_evd: lhs_evd == rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.NEQ: lambda lhs_evd, rhs_evd: lhs_evd != rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.GT: lambda lhs_evd, rhs_evd: lhs_evd > rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.GEQ: lambda lhs_evd, rhs_evd: lhs_evd >= rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.LT: lambda lhs_evd, rhs_evd: lhs_evd < rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.LEQ: lambda lhs_evd, rhs_evd: lhs_evd <= rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.AND: lambda lhs_evd, rhs_evd: lhs_evd and rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.OR: lambda lhs_evd, rhs_evd: lhs_evd or rhs_evd,
                puan_db_pb2.Predicate.BinaryOperator.IN: lambda lhs_evd, rhs_evd: lhs_evd in rhs_evd,
            }

            lhs_evaluated = evaluate(key, predicate.lhs)
            rhs_evaluated = evaluate(key, predicate.rhs)
            
            if lhs_evaluated is None or rhs_evaluated is None:
                return False
            
            return val_map[predicate.operator](lhs_evaluated, rhs_evaluated)
            
        return list(
            filter(
                lambda k: validate(k, predicate),
                model.data.keys()
            )
        )
    
    def device_from_context(self, context):
        token = dict(context.invocation_metadata()).get('token', None)
        if not token:
            return None
        return ComputingDevice(token, self.model_handler)
    
    def ModelCreate(self, request, context):
        try:
            token = self.model_handler.create_model(request.id, request.password)
            return puan_db_pb2.ModelResponse(
                error=None,
                token=token,
                success=True,
            )
        except Exception as e:
            logging.log(logging.ERROR, e)
            return puan_db_pb2.ModelResponse(
                error="Could not create model",
                success=False,
            )
    
    def ModelSelect(self, request, context):
        try:
            # token = self.model_handler.create_token(request.id, request.password)
            token = request.id
            if not self.model_handler.verify_token(token):
                return puan_db_pb2.ModelResponse(
                    error="Invalid credentials",
                    success=False,
                )
            return puan_db_pb2.ModelResponse(
                error=None,
                token=token,
                success=True,
            )
        except Exception as e:
            logging.log(logging.ERROR, e)
            return puan_db_pb2.ModelResponse(
                error="Invalid credentials",
                success=False,
            )
    
    def SetPrimitive(self, request: puan_db_pb2.Primitive, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_primitive(request.id, PuanDB.properties_dict(request.properties), PuanDB.bound_complex(request.bound))
            )
        )

    def SetPrimitives(self, request: puan_db_pb2.SetPrimitivesRequest, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.IDsResponse(
            ids=computing_device.modify(
                lambda model: model.set_primitives(request.ids, PuanDB.properties_dict(request.properties), PuanDB.bound_complex(request.bound))
            )
        )
    
    def SetAtLeast(self, request: puan_db_pb2.AtLeast, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_atleast(
                    children=request.references, 
                    value=request.value, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetAtMost(self, request: puan_db_pb2.AtMost, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_atmost(
                    children=request.references, 
                    value=request.value, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetAnd(self, request: puan_db_pb2.And, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_and(
                    children=request.references, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetOr(self, request: puan_db_pb2.Or, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_or(
                    children=request.references, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetXor(self, request: puan_db_pb2.Xor, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_xor(
                    children=request.references, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetNot(self, request: puan_db_pb2.Not, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_not(
                    children=request.references, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetImply(self, request: puan_db_pb2.Imply, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_imply(
                    antecedent=request.condition, 
                    consequent=request.consequence, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetEqual(self, request: puan_db_pb2.Equal, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_equal(
                    references=request.references, 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def SetEquivalent(self, request: puan_db_pb2.Equivalent, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.SetResponse(
            id=computing_device.modify(
                lambda model: model.set_equal(
                    references=[
                        request.lhs,
                        request.rhs,
                    ], 
                    alias=request.alias,
                    properties=PuanDB.properties_dict(request.properties),
                )
            )
        )
    
    def PropagateUpstream(self, request: puan_db_pb2.Interpretation, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return PuanDB.solution_interpretation(
            computing_device.compute(
                lambda model: model.propagate_upstream(PuanDB.interpretation_dict(request))
            )
        )

    def Propagate(self, request: puan_db_pb2.Interpretation, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return PuanDB.solution_interpretation(
            computing_device.compute(
                lambda model: model.propagate(PuanDB.interpretation_dict(request))
            )
        )
    
    def PropagateBidirectional(self, request: puan_db_pb2.Interpretation, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return PuanDB.solution_interpretation(
            computing_device.compute(
                lambda model: model.propagate_bistream(PuanDB.interpretation_dict(request))
            )
        )
    
    def Solve(self, request: puan_db_pb2.SolveRequest, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        try:
            return puan_db_pb2.SolveResponse(
                solutions=list(
                    map(
                        PuanDB.solution_interpretation,
                        computing_device.compute(
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
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.BooleanSetResponse(
            success=computing_device.modify(
                lambda model: model.delete(request.id)
            )
        )
    
    def Get(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        bound, props = computing_device.compute(
            lambda model: (
                model.get(request.id),
                model.data.get(request.id, {})
            )
        )

        try:
            return puan_db_pb2.VariableResponse(
                id=request.id,
                properties=PuanDB.dict_properties(props),
                bound=PuanDB.complex_bound(bound),
            )
        except:
            raise Exception(f"ID `{request.id}` not found")
    
    def GetMetaInformation(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.MetaInformationResponse(
            nrows=computing_device.compute(lambda model: model._amat.shape[0]),
            ncols=computing_device.compute(lambda model: model._amat.shape[1]),
        )
    
    def GetDependencies(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.IDsResponse(
            ids=computing_device.compute(lambda model: model.dependencies(request.id))
        )
    
    def GetIDFromAlias(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.IDResponse(
            id=computing_device.compute(lambda model: model.id_from_alias(request.alias))
        )
    
    def GetPrimitive(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        bound_complex = computing_device.compute(lambda model: model.get(request.id))
        return puan_db_pb2.Primitive(
            id=request.id,
            bound=puan_db_pb2.Bound(
                lower=int(bound_complex.real),
                upper=int(bound_complex.imag),
            )
        )
    
    def GetPrimitives(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

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
                    computing_device.compute(
                        lambda model: zip(
                            model.primitives, 
                            map(model.get, model.primitives)
                        )
                    )
                )
            )
        )
    
    def GetPrimitiveIds(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.IDsResponse(
            ids=computing_device.compute(lambda model: model.primitives.tolist())
        )
    
    def GetComposite(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.Composite(
            **computing_device.compute(
                lambda model: {
                    "id": request.id,
                    "references": model.dependencies(request.id),
                    "bias": int(model._bvec[model._row(request.id)]),
                    "negated": bool(model._nvec[model._row(request.id)]),
                    "alias": model.id_to_alias(request.id),
                    "properties": PuanDB.dict_properties(model.data.get(request.id, {})),
                }
            )
        )
    
    def GetComposites(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        model = computing_device.get()
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
                            "properties": PuanDB.dict_properties(model.data.get(composite_id, {})),
                        }
                    ),
                    model.composites.tolist(),
                )
            )
        )

    def GetCompositeIds(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')

        return puan_db_pb2.IDsResponse(
            ids=computing_device.compute(lambda model: model.composites.tolist())
        )
    
    def Find(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')
        return puan_db_pb2.IDsResponse(
            ids=computing_device.compute(
                lambda model: PuanDB.find_from_predicate(model, request)
            )
        )
    
    def Cut(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')
        try:
            new_model = computing_device.compute(
                lambda model: model.cut(
                    dict(
                        map(
                            lambda x: (x.from_id, x.to_id),
                            request.cut_ids
                        )
                    )
                )
            )
            new_token = self.model_handler.create_token_from_model(new_model)
            self.model_handler.save_model(new_model, new_token)
            return puan_db_pb2.ModelResponse(
                error=None,
                token=new_token,
                success=True,
            )
        except Exception as e:
            logging.log(logging.ERROR, e)
            return puan_db_pb2.ModelResponse(
                error="Could not cut model",
                success=False,
            )
        
    def Sub(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')
        try:
            new_model = computing_device.compute(
                lambda model: model.sub(request.ids)
            )
            new_token = self.model_handler.create_token_from_model(new_model)
            self.model_handler.save_model(new_model, new_token)
            return puan_db_pb2.ModelResponse(
                error=None,
                token=new_token,
                success=True,
            )
        except Exception as e:
            logging.log(logging.ERROR, e)
            return puan_db_pb2.ModelResponse(
                error="Could not sub model",
                success=False,
            )
        
    def CutSub(self, request, context):
        computing_device = self.device_from_context(context)
        if not computing_device:
            context.abort(grpc.StatusCode.INTERNAL, 'Model data is not available')
        try:
            new_model = computing_device.compute(
                lambda model: model.cut_sub(
                    dict(
                        map(
                            lambda x: (x.from_id, x.to_id),
                            request.cut_ids
                        )
                    ),
                    request.ids
                )
            )
            new_token = self.model_handler.create_token_from_model(new_model)
            self.model_handler.save_model(new_model, new_token)
            return puan_db_pb2.ModelResponse(
                error=None,
                token=new_token,
                success=True,
            )
        except Exception as e:
            logging.log(logging.ERROR, e)
            return puan_db_pb2.ModelResponse(
                error="Could not cut sub model",
                success=False,
            )

def serve():
    handler = AzureBlobModelHandler(
        os.getenv('SALT'),
        os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
        os.getenv('AZURE_STORAGE_CONTAINER')
    )
    port = os.getenv('APP_PORT', '50051')
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[DataFetchingInterceptor(handler)]
    )
    puan_db_pb2_grpc.add_ModelingServiceServicer_to_server(
        PuanDB(handler),
        server
    )
    server.add_insecure_port("[::]:" + port)
    server.start()
    logging.log(logging.INFO, "Server started, listening on " + port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()