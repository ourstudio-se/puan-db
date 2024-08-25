
from itertools import chain
from typing import List, Union
from pldag import PLDAG
from api.models.system import (
    LinearInequalityComposite,
    BinaryComposite,
    LogicalComposite,
    ValueComposite,
    PropositionType,
    Primitive
)

PropositionStringUnionType = Union[str, Primitive, "BinaryComposite", "LogicalComposite", "ValueComposite", "LinearInequalityComposite"]

class SetCompositeException(Exception):
    
    def __init__(self, message: str, composite: PropositionStringUnionType):
        self.message = message
        self.composite = composite


def is_composite(x) -> bool:
    return isinstance(x, (LinearInequalityComposite, BinaryComposite, LogicalComposite, ValueComposite))

def set_composite(composite: PropositionStringUnionType, current_model: PLDAG) -> str:
    _id = None
    if hasattr(composite, 'rtype'):
        rtype = composite.rtype.value
        if rtype in ['and', 'or', 'xor', 'not'] or rtype in ['atleast', 'atmost', 'equal']:
            children = list(
                chain(
                    map(
                        lambda x: set_composite(x, current_model=current_model), 
                        filter(
                            is_composite, 
                            composite.inputs
                        )
                    ),
                    filter(lambda x: issubclass(x.__class__, str), composite.inputs)
                )
            )
            alias = composite.alias
            if rtype == 'and':
                _id = current_model.set_and(children, alias=alias)
            elif rtype == 'or':
                _id = current_model.set_or(children, alias=alias)
            elif rtype == 'xor':
                _id = current_model.set_xor(children, alias=alias)
            elif rtype == 'not':
                _id = current_model.set_not(children, alias=alias)

            if rtype in ['atleast', 'atmost', 'equal']:
                value = composite.value
                if value is None:
                    raise SetCompositeException("Composite `value` missing", composite)
                
                if rtype == 'atleast':
                    _id = current_model.set_atleast(children, value, alias=alias)
                elif rtype == 'atmost':
                    _id = current_model.set_atmost(children, value, alias=alias)
                elif rtype == 'equal':
                    _id = current_model.set_equal(children, value, alias=alias)

        elif rtype in ['equiv', 'imply']:
            lhs = set_composite(composite.lhsInput, current_model=current_model) if is_composite(composite.lhsInput) else composite.lhsInput
            rhs = set_composite(composite.rhsInput, current_model=current_model) if is_composite(composite.rhsInput) else composite.rhsInput
            if not lhs or not rhs:
                raise SetCompositeException("Composite `lhs` or `rhs` missing", composite)
            
            alias = composite.alias
            if rtype == 'equiv':
                _id = current_model.set_equiv(lhs, rhs, alias=alias)
            elif rtype == 'imply':
                _id = current_model.set_imply(lhs, rhs, alias=alias)

    elif isinstance(composite, LinearInequalityComposite):
        try:
            coefficients = dict(
                map(
                    lambda x: (
                        set_composite(x.id, current_model=current_model) if is_composite(x.id) else x.id,
                        x.coef,
                    ),
                    composite.coefficients
                )
            )
        except Exception:
            raise SetCompositeException("Invalid `coefficients`. Should be a list of {'coef': int, 'id': str}", composite)
        
        bias = composite.bias
        alias = composite.alias

        if not coefficients:
            raise SetCompositeException("Composite `coefficients` missing", composite)
        
        if bias is None or not isinstance(bias, int):
            raise SetCompositeException("Composite `bias` missing or invalid (should be int)", composite)
        
        _id = current_model.set_gelineq(coefficients, bias, alias=alias)

    return _id

def merge_composites(propositions: List[PropositionStringUnionType], current_model: PLDAG) -> List[str]:
    
    """
        Returns a list of ids of the propositions that were set.
        Propositions are added to the current model.
    """

    return list(
        chain(
            map(
                lambda primitive: current_model.set_primitive(
                    id=primitive.id, 
                    bound=complex(
                        primitive.bounds.lower, 
                        primitive.bounds.upper,
                    )
                ),
                filter(lambda x: x.ptype == PropositionType.primitive, propositions)
            ),
            map(
                lambda composite: set_composite(composite, current_model),
                filter(lambda x: x.ptype == PropositionType.composite, propositions)
            )
        )
    )   