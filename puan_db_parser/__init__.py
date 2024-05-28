from dataclasses import dataclass, field
from datatypes import *
from typing import List, Dict, Any, Callable

@dataclass
class Node:
    func: Callable
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, n):
        if isinstance(n, list):
            return list(
                map(
                    self.evaluate,
                    n
                )
            )
        elif isinstance(n, dict):
            return {self.evaluate(k): self.evaluate(v) for k,v in n.items()}
        elif isinstance(n, Node):
            return n()
        else:
            return n
    def __call__(self):
        return self.func(*self.evaluate(self.args), **self.evaluate(self.kwargs))
    def __hash__(self) -> int:
        return hash(self())

class Parser:
    def __init__(self, model):
        self.model = model

    def parse(self, tokens):
        if isinstance(tokens, PROPERTIES):
            return self.arguments(tokens)[0]
        elif isinstance(tokens, LIST):
            return self.arguments(tokens)
        elif isinstance(tokens, VARIABLE):
            return tokens.id
        elif isinstance(tokens, (VALUE, INT_VALUE, FLOAT_VALUE)):
            return tokens.value
        elif isinstance(tokens, PREDICATE):
            return self.action(tokens)
        elif isinstance(tokens, ASSIGNMENT):
            return self.arguments(tokens)
        elif isinstance(tokens, list):
            return Node(
                    lambda a, k: (a, k),
                    list(
                        map(
                            self.action,
                            tokens
                        )
                    )
                )
        elif isinstance(tokens, dict):
            return {self.parse(k): self.parse(v) for k, v in tokens.items()}
        elif isinstance(tokens, (str, int, float, bool)):
            return tokens
        else:
            return self.action(tokens)
 
    def action(self, token):
        args, kwargs = self.arguments(token)
        return Node(
            self.function(token),
            args=args,
            kwargs = kwargs
        )
    def function(self, token):
        if isinstance(token, (ACTION_ASSUME, ACTION_REDUCE, ACTION_PROPAGATE)):
            return self.model.propagate
        elif isinstance(token, ACTION_CUT):
            return self.model.cut
        elif isinstance(token, ACTION_SUB):
            return self.model.sub
        elif isinstance(token, ACTION_DEL):
            return self.model.delete
        elif isinstance(token, ACTION_GET):
            return self.model.get
        elif isinstance(token, ACTION_SET_PRIMITIVE):
            return self.model.set_primitive
        elif isinstance(token, ACTION_SET_PRIMITIVES):
            return self.model.set_primitives
        elif isinstance(token, (BOUND, ASSIGNMENT)):
            return complex
        elif isinstance(token, (VARIABLE, INT_VALUE, FLOAT_VALUE, VALUE)):
            return lambda x: x
        elif isinstance(token, ACTION_SET_VALUE_COMPOSITE):
            return self.function(token.sub_action)
        elif isinstance(token, ACTION_SET_LIST_COMPOSITE):
            return self.function(token.sub_action)
        elif isinstance(token, PREDICATE):
            return parse_predicate
        elif isinstance(token, (ACTION_MAXIMIZE, ACTION_MINIMIZE)):
            return self.model.solve
        elif token==SUB_ACTION_TYPE.ATLEAST:
            return self.model.set_atleast
        elif token==SUB_ACTION_TYPE.ATMOST:
            return self.model.set_atmost
        elif token==SUB_ACTION_TYPE.AND:
            return self.model.set_and
        elif token==SUB_ACTION_TYPE.OR:
            return self.model.set_or
        elif token==SUB_ACTION_TYPE.XOR:
            return self.model.set_xor
        elif token==SUB_ACTION_TYPE.NOT:
            return self.model.set_not
        elif token==SUB_ACTION_TYPE.IMPLY:
            return self.model.set_imply
        elif token==SUB_ACTION_TYPE.EQUIV:
            return self.model.set_equal
        else:
            raise ValueError("Could not get function from token")
    
    def arguments(self, token):
        if isinstance(token, (ACTION_ASSUME, ACTION_PROPAGATE)):
            return [self.arguments(token.argument)[0]], {}
        elif isinstance(token, ACTION_SET_PRIMITIVE):
            return [self.parse(token.argument)], self.arguments(token.properties)[1] | {'bound': self.action(token.bound)}
        elif isinstance(token, ACTION_SET_PRIMITIVES):
            return [list(map(self.parse, token.arguments.items))], self.arguments(token.properties)[1] | {'bound': self.action(token.bound)}
        elif isinstance(token, ACTION_SET_VALUE_COMPOSITE):
            return [list(map(self.parse, token.arguments.items)), self.arguments(token.value)], self.arguments(token.properties)[1]
        elif isinstance(token, ACTION_SET_LIST_COMPOSITE):
            if isinstance(token.arguments, PREDICATE):
                return [self.parse(token.arguments)], self.arguments(token.properties)[1]
            elif token.sub_action==SUB_ACTION_TYPE.IMPLY:
                return [self.parse(token.arguments.items[0]), self.parse(token.arguments.items[1])], self.arguments(token.properties)[1]
            else:
                return [list(map(self.parse, token.arguments.items))], self.arguments(token.properties)[1]
        elif isinstance(token, (ACTION_MAXIMIZE, ACTION_MINIMIZE)):
            if isinstance(token.argument, LIST):
                return [self.arguments(token.argument)[0], self.parse(token.suchthat)], {}
            return [[self.arguments(token.argument)[0]], self.parse(token.suchthat)], {}
        elif "argument" in token.__dict__: # token is ACTION_GET, ACTION_DEL, ACTION_SUB, ACTION_CUT
            return [self.parse(token.argument)], {k: self.parse(v) for k,v in token.__dict__.items() if k != "argument"}
        elif isinstance(token, PROPERTIES):
            return {self.parse(k): self.parse(v) for k,v in token.properties.items()}, {self.parse(k): self.parse(v) for k,v in token.__dict__.items()}
        elif isinstance(token, LIST):
            return list(
                map(
                    self.parse,
                    token.items
                )
            )
        elif isinstance(token, PREDICATE):
            return [self.model.data, token.proposition], {}
        elif isinstance(token, BOUND):
            return [], {'real': token.lower, 'imag': token.upper}
        elif isinstance(token, (INT_VALUE, FLOAT_VALUE, VALUE)):
            return token.value
        elif isinstance(token, VARIABLE):
            return token.id
        elif isinstance(token, ASSIGNMENT):
            return {self.parse(token.lhs): self.parse(token.rhs)}
        else:
            return [], {self.parse(k): self.parse(v) for k,v in token.__dict__.items()}

def evaluate(ast):
    if isinstance(ast, list):
        return list(
            map(
                evaluate,
                ast
            )
        )
    elif isinstance(ast, dict):
        return {k: evaluate(v) for k,v in ast.items()}
    elif isinstance(ast, Node):
        return ast.func(*evaluate(ast.args), **evaluate(ast.kwargs))
    else:
        return ast
    
def evaluate_predicate(variable: str, properties: Dict, proposition: PROPOSITION) -> bool:
    if isinstance(proposition, VARIABLE):
        return properties.get(proposition.id)
    elif isinstance(proposition, (VALUE, INT_VALUE, FLOAT_VALUE)):
        return proposition.value
    elif isinstance(proposition, FUNCTION):
        return Node(func=proposition.name, args=evaluate_predicate(variable, properties, proposition.arg))
    fn_map = {
        OPERATION.GT: lambda x, y: x > y,
        OPERATION.LT: lambda x, y: x < y,
        OPERATION.EQ: lambda x, y: x == y,
        OPERATION.GTE: lambda x, y: x >= y,
        OPERATION.LTE: lambda x, y: x <= y,
        OPERATION.NEQ: lambda x, y: x != y,
        OPERATION.AND: lambda x, y: x and y,
        OPERATION.OR: lambda x, y: x or y,
        #OPERATION.NONE lambda x: x
    }
    try:
        return fn_map[proposition.operation](evaluate_predicate(variable, properties, proposition.lh), evaluate_predicate(variable, properties, proposition.rh))
    except:
        return False

def parse_predicate(data: Dict, predicate: PREDICATE) -> List[str]:
    return list(filter(lambda x: evaluate_predicate(x, data.get(x), predicate), data.keys())) 