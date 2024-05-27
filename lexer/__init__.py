# Usage: 
# import lexer
# inp = """SET x {price: 5.0, category: 'Model'}"""
# lexed = lexer.lex(inp)


# OPERATIONAL KEYWORDS:     SET, GET, DEL, SUB, CUT, ASSUME, REDUCE, PROPAGATE, MAXIMIZE, MINIMIZE
# ADDITIONAL KEYWORDS:      ATLEAST, ATMOST, AND, OR, XOR, NOT, EQUAL, IMPLY, EQUIV, SUCHTHAT

# 1) First token is always a operational keyword: SET, GET, DEL, SUB, CUT, ASSUME, REDUCE, PROPAGATE, MAXIMIZE, MINIMIZE
# 2) Second token can be: another keyword (ATLEAST), a variable (x), multiple variables (x,y,z), assigned variables (x=1), list comprehension
# 3) Third token can be: a keyword properties ({price: 400.0}), a variable, a list of variables and/or list comprehension
# 4) Fourth token can be: an integer, a variable bound (-2..3) or assigned variables 
# 5) Fifth token can only be properties (in the case of ATLEAST or ATMOST

# A COMMAND has at most 5 tokens and always returns something.

# Checking/Modifying commands:
# A GET-command always returns a list of variables (id, bound and properties)
# A SET-command always returns a list of variable IDs
# A DEL-command always returns a list of booleans

# Sub graph commands:
# A SUB-command always returns a new graph
# A CUT-command always returns a new graph
# A ASSUME-command always returns a new graph
# A REDUCE-command always returns a new graph

# Computational commands:
# A PROPAGATE-command always returns a list of variables
# A MAXIMIZE-command always returns a list of variables
# A MINIMIZE-command always returns a list of variables

# 1) Välja databas (rättigheter, anv lösen) <- huvudmodell
# 2) Göra operation / fokusera på delgraf
# 3) Göra operation på delgraf

# EXAMPLES
"""
    SET a                                       # Set boolean variable a with no attributes
    SET b {}                                    # Set boolean variable b with no attributes
    SET c {} -2..3                              # Set integer variable c with bounds -2 to 3
    SET [d,e,f]                                 # Set boolean variables d,e,f with no attributes

    SET x {price: 5.0, category: 'Model'}       # Set variable x with attributes price and category

    SET ATLEAST [x, y, z] 1 {}                  # set at least 1 of x, y, z
    SET ATMOST [x, y, z] 1 {}                   # set at most 1 of x, y, z
    
    SET AND [x : x.price > 5] {}                # set all x where price > 5
    # SET OR [...] {} 
    # SET XOR [...] {} 
    # SET NOT [...] {} 
    # SET EQUAL [...] {} 
    
    SET IMPLY [x, y] {}                          # Set if x then y
    SET EQUIV [x, y] {}                          # Set if x then y, if y then x

    SUB [x : x.price > 7.0]                     # Create's a sub graph with starting points where price > 7.0
    GET [x, y, z]                               # Get variables with ID x, y or z's properties and bounds
    SET AND [                                   # Set complex nested constraint (x OR y) AND (x OR z)
        x,
        SET OR [x, y],
        SET OR [x, z]
    ]

    GET x                                       # Get variable x's properties and bound
    GET [x : x.price > 5]                       # Get all x where price > 5
    GET [x : type(x) == PRIMITIVE]              # Get all x primitive variables


    DEL x                                       # Delete variable x
    DEL [x, y, z]                               # Delete variables with ID x, y or z
    DEL [x : (x.price > 10) && (x.type == "A")]  # Delete all x where price > 5 and type is A

    SUB x                                       # Create's a sub graph with x as root
    SUB [x, y, z]                               # Create's a sub graph with x, y, z as roots
    SUB [x : x.type == "COL"]                   # Create's a sub graph with all variables of type COL as roots

    SUB x
    SET AND [x : x.price > 5] 
    PROPAGATE {x: 1} 

    
    CUT A                                       # Cut away all nodes under A
    CUT [A, {B: "y"}]                           # Cut away all nodes under A, B
    CUT {A: "x", B: "y"}                        # Cut away all nodes under A and B. Rename A to x and B to y


    ASSUME {A: 1..2}                            # Assume A has tighter bound 1..2 and propagates the change
    REDUCE                                      # Reduces the graph by removing all nodes with constant bound


    PROPAGATE {x: 1..2}                         # Propagates the change of x=1..2 to all nodes
    MAXIMIZE {x:1} SUCHTHAT y=1                 # Finds a configuration that maximizes x=1 such that y is true
    MINIMIZE {X:1} SUCHTHAT y=0                 # Finds a configuration that minimizes x=1 such that y is false
    MINIMIZE {x:1} SUCHTHAT (SET AND [y,z])=1   # Finds a configuration that minimizes x=1 such that y and z are true
"""

import re
from ast import literal_eval
from maz import compose, invoke
from enum import Enum
from typing import Union, Any, List, Callable, Dict
from dataclasses import dataclass, field
from itertools import chain
import hashlib

# An action that will eventually return a string
class STRING_ACTION:
    pass

class ACTION_TYPE(Enum):
    SET = 1
    GET = 2
    DEL = 3
    SUB = 4
    CUT = 5
    ASSUME = 6
    REDUCE = 7
    PROPAGATE = 8
    MAXIMIZE = 9
    MINIMIZE = 10

class SUB_ACTION_TYPE(Enum):
    ATLEAST = 1
    ATMOST = 2
    AND = 3
    OR = 4
    XOR = 5
    NOT = 6
    EQUAL = 7
    IMPLY = 8
    EQUIV = 9

class KEYWORD(Enum):
    SUCHTHAT = 1

class DATATYPE(Enum):
    PRIMITIVE = 1
    COMPOSITE = 2

@dataclass
class PROPERTIES:
    properties: dict = field(default_factory=dict)

@dataclass
class LIST:
    items: list

class OPERATION(str, Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    AND = "&&"
    OR = "||"

@dataclass
class BOUND:
    lower: int
    upper: int

@dataclass
class VARIABLE:
    id: str

@dataclass
class VALUE:
    value: Union[int, float, str]

@dataclass
class INT_VALUE:
    value: int

@dataclass
class FLOAT_VALUE:
    value: float

@dataclass
class ASSIGNMENT:
    lhs: STRING_ACTION
    rhs: VALUE

@dataclass
class INT_ASSIGNMENT:
    lhs: STRING_ACTION
    rhs: int

@dataclass
class FUNCTION:
    name: str
    arg: Any

@dataclass
class PROPOSITION:
    operation: OPERATION
    lh: Union[VARIABLE, VALUE, "PROPOSITION"]
    rh: Union[VARIABLE, VALUE, "PROPOSITION"]

@dataclass
class PREDICATE:
    id: str
    proposition: PROPOSITION

LIST_PREDICATE = Union[PREDICATE, LIST]

@dataclass
class ACTION_SET_PRIMITIVE:
    argument: VARIABLE
    properties: PROPERTIES = field(default_factory=PROPERTIES)
    bound: BOUND = BOUND(0, 1)

@dataclass
class ACTION_SET_PRIMITIVES:
    arguments: LIST
    properties: PROPERTIES = field(default_factory=PROPERTIES)
    bound: BOUND = BOUND(0, 1)

@dataclass
class ACTION_SET_LIST_COMPOSITE:
    sub_action: SUB_ACTION_TYPE
    arguments: LIST_PREDICATE
    properties: PROPERTIES = field(default_factory=PROPERTIES)

@dataclass
class ACTION_SET_VALUE_COMPOSITE:
    sub_action: SUB_ACTION_TYPE
    arguments: LIST_PREDICATE
    value: VALUE
    properties: PROPERTIES = field(default_factory=PROPERTIES)

@dataclass
class ACTION_GET:
    argument: Union[VARIABLE, LIST_PREDICATE]

@dataclass
class ACTION_DEL:
    argument: Union[VARIABLE, LIST_PREDICATE]

@dataclass
class ACTION_SUB:
    argument: Union[VARIABLE, LIST_PREDICATE]

@dataclass
class ACTION_CUT:
    argument: Union[VARIABLE, PROPERTIES]

@dataclass
class ACTION_ASSUME:
    argument: PROPERTIES

@dataclass
class ACTION_REDUCE:
    pass

@dataclass
class ACTION_PROPAGATE:
    argument: PROPERTIES

@dataclass
class ACTION_MAXIMIZE:
    argument: PROPERTIES
    suchthat: INT_ASSIGNMENT

@dataclass
class ACTION_MINIMIZE:
    argument: PROPERTIES
    suchthat: INT_ASSIGNMENT

def prelex(inp):
    
    # Remove spaces before and after new lines
    return re.sub(r'\s*\n\s*', '\n',
        # Remove all comments
        re.sub(r'#.*', '', 
            # Remove all new lines and spaces in the beginning
            re.sub(r'^[\n\s]+', '', inp)
        )
    )

def lex_range(token):
    return BOUND(*map(literal_eval, token.split("..")))

def lex_prop(prop):
    key, value = prop.split(":")
    if ".." in value:
        return (key.strip(), lex_range(value.strip()))
    return (key.strip(), literal_eval(value.strip()))

def lex_props(token):
    inner = token[1:-1]
    props = list(filter(lambda x: ":" in x, inner.split(",")))
    return PROPERTIES(dict(map(lex_prop, props)))

def lex_assign(token):
    key, value = token.split("=")
    return ASSIGNMENT(lex_token(key.strip()), lex_token(value.strip()))

def lex_statement(token):
    if token[:3] in ["GET", "SET"]:
        return lex_action(token)
    else:
        return lex_token(token)

def split_outer(token, sep):
    # Splits a string by a separator, but only if the separator is not inside brackets
    bracket_count = 0
    i = 0
    splits = []
    _token = token
    while i < len(_token):
        if _token[i] == "[":
            bracket_count += 1
        elif _token[i] == "]":
            bracket_count -= 1
        elif _token[i] == sep and bracket_count == 0:
            splits.append(_token[:i].strip())
            _token = _token[i+1:]
            i = -1
        i += 1
    
    splits.append(_token.strip())
    return splits

def lex_list(token):
    return LIST(
        list(
            map(
                lex_statement,
                split_outer(token[1:-1], ",")
            )
        )
    )

def lex_function(token):
    call_start_index, call_end_index = token.find("("), token.find(")")
    function_name = token[:call_start_index]
    function_args = lex_token(token[call_start_index+1:call_end_index])
    return FUNCTION(function_name, function_args)

def drop_outer_brackets(token):
    if token[0] == "(" and token[-1] == ")":
        return token[1:-1]
    return token

def lex_proposition(token):
    if "&&" in token:
        return PROPOSITION(OPERATION.AND, *map(compose(lex_proposition, drop_outer_brackets), token.split("&&")))
    elif "||" in token:
        return PROPOSITION(OPERATION.OR, *map(compose(lex_proposition, drop_outer_brackets), token.split("||")))
    elif "==" in token:
        return PROPOSITION(OPERATION.EQ, *map(compose(lex_proposition, drop_outer_brackets), token.split("==")))
    elif "!=" in token:
        return PROPOSITION(OPERATION.NEQ, *map(compose(lex_proposition, drop_outer_brackets), token.split("==")))
    elif ">=" in token:
        return PROPOSITION(OPERATION.GTE, *map(compose(lex_proposition, drop_outer_brackets), token.split(">=")))
    elif "<=" in token:
        return PROPOSITION(OPERATION.LTE, *map(compose(lex_proposition, drop_outer_brackets), token.split("<=")))
    elif ">" in token:
        return PROPOSITION(OPERATION.GT, *map(compose(lex_proposition, drop_outer_brackets), token.split(">")))
    elif "<" in token:
        return PROPOSITION(OPERATION.LT, *map(compose(lex_proposition, drop_outer_brackets), token.split("<")))
    else:
        if token in DATATYPE.__members__:
            return DATATYPE[token]
        elif "(" in token and ")" in token:
            return lex_function(token)
        elif "." in token:
            l,r = token.split(".")
            if l.isnumeric():
                return VALUE(literal_eval(token))
            return VARIABLE(r)
        else:
            return VALUE(literal_eval(token))

def lex_variable_predicate(token):
    variable, predicate = token[1:-1].replace(" ", "").split(":")
    return PREDICATE(variable, lex_proposition(predicate))

def lex_list_predicate(token):
    # This is either a list or a predicate. If it has a colon, it's a predicate
    if ":" in token and not "{" in token:
        return lex_variable_predicate(token)
    else:
        return lex_list(token)

def lex_token(token):
    
    if token in SUB_ACTION_TYPE.__members__:
        return SUB_ACTION_TYPE[token]
    elif token in DATATYPE.__members__:
        return DATATYPE[token]
    elif token in KEYWORD.__members__:
        return KEYWORD[token]
    elif "(" == token[0] and ")" == token[-1]:
        return lex_statement(token[1:-1])
    elif "[" == token[0] and "]" == token[-1]:
        return lex_list_predicate(token)
    elif "{" == token[0] and "}" == token[-1]:
        return lex_props(token)
    elif "=" in token:
        return lex_assign(token)
    elif ".." in token:
        return lex_range(token)
    else:
        if token.isnumeric():
            return INT_VALUE(int(token))
        elif "." in token:
            try:
                return FLOAT_VALUE(token)
            except:
                pass
        else:
            return VARIABLE(token)
    
def lex_sub_commands(token):
    starting_brackets = []
    i = 0
    sub_command_idx_pairs = []
    while i < len(token):
        if token[i] == "(":
            starting_brackets.append(i)
        elif token[i] == ")":
            sub_command_idx_pairs.append((starting_brackets.pop(), i))
        i += 1
    return sub_command_idx_pairs

def lex_action(inp):

    first_space = inp.find(" ")
    action = inp[:first_space if first_space != -1 else len(inp)]

    if not action in ACTION_TYPE.__members__:
        raise ValueError(f"Invalid action: {action}")
    
    i = first_space + 1
    token = ""
    bracket_count = 0
    lexed = []
    while i < len(inp):
        token += inp[i]

        if inp[i] in ['(', '[', '{']:
            bracket_count += 1
        elif inp[i] in [')', ']', '}']:
            bracket_count -= 1
        elif (inp[i] == ' ') and bracket_count == 0:
            lexed.append(lex_token(token.strip()))
            token = ""

        i += 1

    lexed.append(lex_token(token.strip()))
    if action == "SET":
        if len(lexed) == 1:
            if type(lexed[0]) == LIST:
                return ACTION_SET_PRIMITIVES(lexed[0])
            elif type(lexed[0]) == VARIABLE:
                return ACTION_SET_PRIMITIVE(lexed[0])
            else:
                raise ValueError(f"Invalid SET action: First argument {lexed[0]} is invalid.")
        elif len(lexed) == 2:
            arg0, arg1 = lexed
            if type(arg0) == SUB_ACTION_TYPE:
                return ACTION_SET_LIST_COMPOSITE(arg0, arg1)
            elif type(arg0) in [VARIABLE, LIST]:
                if type(arg0) == LIST:
                    return ACTION_SET_PRIMITIVES(arg0, arg1)
                elif type(arg0) == VARIABLE:
                    return ACTION_SET_PRIMITIVE(arg0, arg1)
                else:
                    raise ValueError(f"Invalid SET action: First argument {arg0} is invalid.")
            else:
                raise ValueError(f"Invalid SET action: First argument {arg0} is invalid.")
        elif len(lexed) == 3:
            arg0, arg1, arg2 = lexed
            if type(arg0) == SUB_ACTION_TYPE:
                return ACTION_SET_LIST_COMPOSITE(arg0, arg1, arg2)
            elif type(arg0) in [VARIABLE, LIST]:
                if type(arg0) == LIST:
                    return ACTION_SET_PRIMITIVES(arg0, arg1, arg2)
                elif type(arg0) == VARIABLE:
                    return ACTION_SET_PRIMITIVE(arg0, arg1, arg2)
                else:
                    raise ValueError(f"Invalid SET action: First argument {arg0} is invalid.")
            else:
                raise ValueError(f"Invalid SET action: First argument {arg0} is invalid.")
        elif len(lexed) == 4:
            arg0, arg1, arg2, arg3 = lexed
            if type(arg0) == SUB_ACTION_TYPE:
                return ACTION_SET_VALUE_COMPOSITE(arg0, arg1, arg2, arg3)
            else:
                raise ValueError(f"Invalid SET action: First argument {arg0} is invalid.")
        else:
            raise ValueError(f"Invalid SET action: Too many arguments.")
        
    elif action == "GET":
        if len(lexed) == 1:
            return ACTION_GET(lexed[0])
        else:
            raise ValueError(f"Invalid GET action: Too many arguments.")
        
    elif action == "DEL":
        if len(lexed) == 1:
            return ACTION_DEL(lexed[0])
        else:
            raise ValueError(f"Invalid DEL action: Too many arguments.")
        
    elif action == "SUB":
        if len(lexed) == 1:
            return ACTION_SUB(lexed[0])
        else:
            raise ValueError(f"Invalid SUB action: Too many arguments.")
        
    elif action == "CUT":
        if len(lexed) == 1:
            return ACTION_CUT(lexed[0])
        else:
            raise ValueError(f"Invalid CUT action: Too many arguments.")
        
    elif action == "ASSUME":
        if len(lexed) == 1:
            return ACTION_ASSUME(lexed[0])
        else:
            raise ValueError(f"Invalid ASSUME action: Too many arguments.")
        
    elif action == "REDUCE":
        return ACTION_REDUCE()
        
    elif action == "PROPAGATE":
        if len(lexed) == 1:
            return ACTION_PROPAGATE(lexed[0])
        else:
            raise ValueError(f"Invalid PROPAGATE action: Too many arguments.")
        
    elif action == "MAXIMIZE":
        if len(lexed) == 1:
            if type(lexed[0]) != PROPERTIES:
                raise ValueError(f"Invalid MINIMIZE action: Expected properties, got {arg0}")
            return ACTION_MAXIMIZE(lexed[0])
        
        elif len(lexed) == 3:
            arg0, arg1, arg2 = lexed
            if type(arg0) != PROPERTIES:
                raise ValueError(f"Invalid MINIMIZE action: Expected properties, got {arg0}")
            if type(arg1) != KEYWORD:
                raise ValueError(f"Invalid MINIMIZE action: Expected SUCHTHAT keyword, got {arg2}")
            if type(arg2) != ASSIGNMENT:
                raise ValueError(f"Invalid MINIMIZE action: Expected assignment, got {arg3}")
            return ACTION_MAXIMIZE(arg0, arg2)

        else:
            raise ValueError(f"Invalid MINIMIZE action: Too few or too many arguments.")
        
    elif action == "MINIMIZE":
        if len(lexed) == 1:
            if type(lexed[0]) != PROPERTIES:
                raise ValueError(f"Invalid MINIMIZE action: Expected properties, got {arg0}")
            return ACTION_MINIMIZE(lexed[0])
        
        elif len(lexed) == 3:
            arg0, arg1, arg2 = lexed
            if type(arg0) != PROPERTIES:
                raise ValueError(f"Invalid MINIMIZE action: Expected properties, got {arg0}")
            if type(arg1) != KEYWORD:
                raise ValueError(f"Invalid MINIMIZE action: Expected SUCHTHAT keyword, got {arg2}")
            if type(arg2) != ASSIGNMENT:
                raise ValueError(f"Invalid MINIMIZE action: Expected assignment, got {arg3}")
            return ACTION_MINIMIZE(arg0, arg2)

        else:
            raise ValueError(f"Invalid MINIMIZE action: Too few or too many arguments.")
        
    else:
        raise ValueError(f"Invalid action: {action}")

def lex(inp):
    
    i = 0
    token = ""
    collected = []
    bracket_count = 0
    finp = prelex(inp)
    
    while i < len(finp):
        
        token += finp[i]
        if finp[i] == '\n':
            if bracket_count == 0:
                collected.append(lex_action(token.replace("\n", "").strip()))
                token = ""
        elif finp[i] in ['[', '{']:
            bracket_count += 1 
        elif finp[i] in [']', '}']:
            bracket_count -= 1
        
        i += 1 

    if token != "":
        collected.append(lex_action(token.replace("\n", "").strip()))

    return collected

@dataclass
class Node:
    func: Callable
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    def __call__(self):
        return self.func(*self.__evaluate(self.args), **self.__evaluate(self.kwargs))
    def __evaluate(self, n):
        if isinstance(n, list):
            return list(
                map(
                    self.__evaluate,
                    n
                )
            )
        elif isinstance(n, dict):
            return {self.__evaluate(k): self.__evaluate(v) for k,v in n.items()}
        elif isinstance(n, Node):
            return n()
        else:
            return n
    def __hash__(self) -> int:
        return hash(self()) #hashlib.sha256(self().encode()).hexdigest()



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
                    invoke,
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
                return [self.arguments(token.argument)[0], self.parse(token.suchthat), "glpk"], {}
            return [[self.arguments(token.argument)[0]], self.parse(token.suchthat), "glpk"], {}
        elif "argument" in token.__dict__: # token is ACTION_GET, ACTION_DEL, ACTION_SUB, ACTION_CUT
            return [self.parse(token.argument)], {k: self.parse(v) for k,v in token.__dict__.items() if k != "argument"}
        elif isinstance(token, PROPERTIES):
            return {self.parse(k): self.parse(v) for k,v in token.properties.items()}, {self.parse(k): self.parse(v) for k,v in token.__dict__.items()}
            #return [], {k: self.action(v) for k, v in token.properties.items()}
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


def get_predicate_expression(proposition):
    def _get_pred(prop):
        if isinstance(prop, VARIABLE):
            return lambda x, _: prop.id == x
        elif isinstance(prop, VALUE):
            if proposition.operation == OPERATION.GT:
                return lambda _, y: y > prop.value
    return lambda x, y: _get_pred(proposition.lh)(x,y) and _get_pred(proposition.rh)(x,y)
atomic_values = (VARIABLE, VALUE, INT_VALUE, FLOAT_VALUE)
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

    

    

