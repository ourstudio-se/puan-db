import lexer
import maz
from pldag import Puan
model = Puan()

def test_case1():

    case_1 = "SET a # Set boolean variable a with no attributes"
    expected_case_1 = lexer.ACTION_SET_PRIMITIVE(lexer.VARIABLE("a"))
    expected_ast_case_1 = lexer.Node(
            func=model.set_primitive,
            args=['a'],
            kwargs={'properties': {},  # 'properties' is empty because there are no attributes
                'bound': lexer.Node(
                func=complex,
                args=[],
                kwargs = {'real':0, 'imag': 1}
            )}
        )
    assert lexer.lex(case_1)[0] == expected_case_1
    assert lexer.Parser(model).parse(expected_case_1) == expected_ast_case_1
    
def test_case2():
    case_2 = "SET b {} # Set boolean variable b with no attributes"
    expected_case_2 = lexer.ACTION_SET_PRIMITIVE(lexer.VARIABLE("b"), bound=lexer.BOUND(0,1))
    expected_ast_case_2 = lexer.Node(
            func=model.set_primitive,
            args=['b'],
            kwargs={'properties': {}, 'bound': lexer.Node(
                func=complex,
                args=[],
                kwargs={'real': 0, 'imag':1}
            )}
        )
    assert lexer.lex(case_2)[0] == expected_case_2
    assert lexer.Parser(model).parse(expected_case_2) == expected_ast_case_2

def test_case3():
    case_3 = "SET c {} -2..3 # Set integer variable c with bounds -2 to 3"
    expected_case_3 = lexer.ACTION_SET_PRIMITIVE(lexer.VARIABLE("c"), bound=lexer.BOUND(-2, 3))
    expected_ast_case_3 = lexer.Node(
            func=model.set_primitive,
            args=['c'],
            kwargs={'properties': {}, 'bound': lexer.Node(
                func=complex,
                args=[],
                kwargs={'real': -2, 'imag':3}
            )}
        )
    assert lexer.lex(case_3)[0] == expected_case_3
    assert lexer.Parser(model).parse(expected_case_3) == expected_ast_case_3

def test_case4():
    case_4 = "SET [d,e,f] # Set boolean variables d,e,f with no attributes"
    expected_case_4 = lexer.ACTION_SET_PRIMITIVES(lexer.LIST([lexer.VARIABLE("d"), lexer.VARIABLE("e"), lexer.VARIABLE("f")]))
    expected_ast_case_4 = lexer.Node(
            func=model.set_primitives,
            args=[['d', 'e','f']],
            kwargs={'properties': {}, 'bound': lexer.Node(
                func=complex,
                args=[],
                kwargs={'real': 0, 'imag':1}
            )}
        )
    assert lexer.lex(case_4)[0] == expected_case_4
    assert lexer.Parser(model).parse(expected_case_4) == expected_ast_case_4

def test_case5():
    case_5 = "SET x {price: 5.0, category: 'Model'} # Set variable x with attributes price and category"
    expected_case_5 = lexer.ACTION_SET_PRIMITIVE(lexer.VARIABLE("x"), properties=lexer.PROPERTIES({"price": 5.0, "category": "Model"}))
    expected_ast_case_5 = lexer.Node(
            func=model.set_primitive,
            args=['x'],
            kwargs={'properties': {'price': 5.0, 'category': 'Model'}, 'bound': lexer.Node(
                func=complex,
                args=[],
                kwargs={'real': 0, 'imag':1}
            )}
        )
    assert lexer.lex(case_5)[0] == expected_case_5
    assert lexer.Parser(model).parse(expected_case_5) == expected_ast_case_5
    
def test_case6():
    case_6 = "SET ATLEAST [x, y, z] 1 {} # set at least 1 of x, y, z"
    expected_case_6 = lexer.ACTION_SET_VALUE_COMPOSITE(
        lexer.SUB_ACTION_TYPE.ATLEAST, 
        lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("y"), lexer.VARIABLE("z")]), 
        lexer.INT_VALUE(1),
    )
    expected_ast_case_6 = lexer.Node(
            func=model.set_atleast,
            args=[['x', 'y', 'z'], 1],
            kwargs={'properties': {}}
        )
    assert lexer.lex(case_6)[0] == expected_case_6
    assert lexer.Parser(model).parse(expected_case_6) == expected_ast_case_6

def test_case7():
    case_7 = "SET ATMOST [x, y, z] 1 {} # set at most 1 of x, y, z"
    expected_case_7 = lexer.ACTION_SET_VALUE_COMPOSITE(
        lexer.SUB_ACTION_TYPE.ATMOST, 
        lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("y"), lexer.VARIABLE("z")]),
        lexer.INT_VALUE(1),
    )
    expected_ast_case_7 = lexer.Node(
            func=model.set_atmost,
            args=[['x', 'y', 'z'], 1],
            kwargs={'properties': {}}
        )
    assert lexer.lex(case_7)[0] == expected_case_7
    assert lexer.Parser(model).parse(expected_case_7) == expected_ast_case_7

def test_case8():
    case_8 = "SET AND [x : x.price > 5] {} # set all x where price > 5"
    expected_case_8 = lexer.ACTION_SET_LIST_COMPOSITE(
        lexer.SUB_ACTION_TYPE.AND,  
        lexer.PREDICATE(
            id="x",
            proposition=lexer.PROPOSITION(
                lexer.OPERATION.GT,
                lexer.VARIABLE("price"),
                lexer.VALUE(5)
            )
        )
    )
    expected_ast_case_8 = lexer.Node(
        func=model.set_and,
        args=[lexer.Node(
                func=lexer.parse_predicate,
                args=[model.data, lexer.PROPOSITION(
                        lexer.OPERATION.GT,
                        lexer.VARIABLE("price"),
                        lexer.VALUE(5)
                    )
                ],
        )],
        kwargs={'properties': {}}
    )
    assert lexer.lex(case_8)[0] == expected_case_8
    assert lexer.Parser(model).parse(expected_case_8) == expected_ast_case_8


def test_case9():
    case_9 = "SET IMPLY [x, y] {} # Set if x then y"
    expected_case_9 = lexer.ACTION_SET_LIST_COMPOSITE(
        lexer.SUB_ACTION_TYPE.IMPLY,
        lexer.LIST([
            lexer.VARIABLE("x"),
            lexer.VARIABLE("y")
        ])
    )
    expected_ast_case_9 = lexer.Node(
            func=model.set_imply,
            args=['x', 'y'],
            kwargs={'properties': {}}
        )
    assert lexer.lex(case_9)[0] == expected_case_9
    assert lexer.Parser(model).parse(expected_case_9) == expected_ast_case_9

def test_case10():
    case_10 = "SET EQUIV [x, y] {} # Set if x then y, if y then x"
    expected_case_10 = lexer.ACTION_SET_LIST_COMPOSITE(
        lexer.SUB_ACTION_TYPE.EQUIV,
        lexer.LIST([
            lexer.VARIABLE("x"),
            lexer.VARIABLE("y")
        ])
    )
    expected_ast_case_10 = lexer.Node(
            func=model.set_equal,
            args=[['x', 'y']],
            kwargs={'properties': {}}
        )
    assert lexer.lex(case_10)[0] == expected_case_10
    assert lexer.Parser(model).parse(expected_case_10) == expected_ast_case_10

def test_case11():
    case_11 = "SUB [x : x.price > 7.0] # Create's a sub graph with starting points where price > 7.0"
    expected_case_11 = lexer.ACTION_SUB(
        lexer.PREDICATE(
            id="x",
            proposition=lexer.PROPOSITION(
                lexer.OPERATION.GT,
                lexer.VARIABLE("price"),
                lexer.VALUE(7.0)
            )
        )
    )
    expected_ast_case_11 = lexer.Node(
            func=model.sub,
            args=[lexer.Node(
                func=lexer.parse_predicate,
                args=[model.data, lexer.PROPOSITION(
                    lexer.OPERATION.GT,
                    lexer.VARIABLE("price"),
                    lexer.VALUE(7.0)
                )]
            )],
            kwargs={}
        )
    assert lexer.lex(case_11)[0] == expected_case_11
    assert lexer.Parser(model).parse(expected_case_11) == expected_ast_case_11

def test_case12():
    case_12 = "GET [x, y, z] # Get variables with ID x, y or z's properties and bounds"
    expected_case_12 = lexer.ACTION_GET(lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("y"), lexer.VARIABLE("z")]))
    expected_ast_case_12 = lexer.Node(
            func=model.get,
            args=[['x', 'y', 'z']],
        )
    assert lexer.lex(case_12)[0] == expected_case_12
    assert lexer.Parser(model).parse(expected_case_12) == expected_ast_case_12

def test_case13():
    case_13 = """
        SET AND [
            SET OR [x, y], 
            SET OR [x, z]
        ] # Set complex nested constraint (x OR y) AND (x OR z)
    """
    expected_case_13 = lexer.ACTION_SET_LIST_COMPOSITE(
        lexer.SUB_ACTION_TYPE.AND,
        lexer.LIST([
            lexer.ACTION_SET_LIST_COMPOSITE(
                lexer.SUB_ACTION_TYPE.OR,
                lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("y")])
            ),
            lexer.ACTION_SET_LIST_COMPOSITE(
                lexer.SUB_ACTION_TYPE.OR,
                lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("z")])
            )
        ])
    )
    expected_ast_case_13 = lexer.Node(
        func=model.set_and,
        args=[
            [
                lexer.Node(
                    func=model.set_or,
                    args=[['x', 'y']],
                    kwargs={'properties': {}}
                ),
                lexer.Node(
                    func=model.set_or,
                    args=[['x', 'z']],
                    kwargs={'properties': {}}
                )
            ]
        ],
        kwargs={'properties': {}}
    )
    assert lexer.lex(case_13)[0] == expected_case_13
    assert lexer.Parser(model).parse(expected_case_13) == expected_ast_case_13

def test_case14():

    case_14 = "GET x # Get variable x's properties and bound"
    expected_case_14 = lexer.ACTION_GET(lexer.VARIABLE("x"))
    expected_ast_case_14 = lexer.Node(
        func=model.get,
        args=['x'],
    )
    assert lexer.lex(case_14)[0] == expected_case_14
    assert lexer.Parser(model).parse(expected_case_14) == expected_ast_case_14

def test_case15():

    case_15 = "GET [x : x.price > 5] # Get all x where price > 5"
    expected_case_15 = lexer.ACTION_GET(
        lexer.PREDICATE(
            id="x",
            proposition=lexer.PROPOSITION(
                lexer.OPERATION.GT,
                lexer.VARIABLE("price"),
                lexer.VALUE(5)
            )
        )
    )
    expected_ast_case_15 = lexer.Node(
        func=model.get,
        args=[lexer.Node(
            func=lexer.parse_predicate,
            args=[model.data, lexer.PROPOSITION(
                lexer.OPERATION.GT,
                lexer.VARIABLE("price"),
                lexer.VALUE(5)
            )]
        )],
    )
    assert lexer.lex(case_15)[0] == expected_case_15
    assert lexer.Parser(model).parse(expected_case_15) == expected_ast_case_15

def test_case16():

    case_16 = "GET [x : type(x) == PRIMITIVE] # Get all x primitive variables"
    expected_case_16 = lexer.ACTION_GET(
        lexer.PREDICATE(
            id="x",
            proposition=lexer.PROPOSITION(
                lexer.OPERATION.EQ,
                lexer.FUNCTION("type", lexer.VARIABLE("x")),
                lexer.DATATYPE.PRIMITIVE,
            )
        )
    )
    expected_ast_case_16 = lexer.Node(
        func=model.get,
        args=[lexer.Node(
            func=lexer.parse_predicate,
            args=[model.data, lexer.PROPOSITION(
                lexer.OPERATION.EQ,
                lexer.FUNCTION("type", lexer.VARIABLE("x")),
                lexer.DATATYPE.PRIMITIVE,
            )]
        )],
    )
    assert lexer.lex(case_16)[0] == expected_case_16
    assert lexer.Parser(model).parse(expected_case_16) == expected_ast_case_16

def test_case17():
    case_17 = "DEL x # Delete variable x"
    expected_case_17 = lexer.ACTION_DEL(lexer.VARIABLE("x"))
    expected_ast_case_17 = lexer.Node(
        func=model.delete,
        args=['x'],
    )
    assert lexer.lex(case_17)[0] == expected_case_17
    assert lexer.Parser(model).parse(expected_case_17) == expected_ast_case_17

def test_case18():
    case_18 = "DEL [x, y, z] # Delete variables with ID x, y or z"
    expected_case_18 = lexer.ACTION_DEL(lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("y"), lexer.VARIABLE("z")]))
    expected_ast_case_18 = lexer.Node(
        func=model.delete,
        args=[['x', 'y', 'z']],
    )
    assert lexer.lex(case_18)[0] == expected_case_18
    assert lexer.Parser(model).parse(expected_case_18) == expected_ast_case_18

def test_case19():
    case_19 = "DEL [x : (x.price > 10) && (x.type == 'A')] # Delete all x where price > 5 and type is A"
    expected_case_19 = lexer.ACTION_DEL(
        lexer.PREDICATE(
            id="x",
            proposition=lexer.PROPOSITION(
                lexer.OPERATION.AND,
                lexer.PROPOSITION(
                    lexer.OPERATION.GT,
                    lexer.VARIABLE("price"),
                    lexer.VALUE(10)
                ),
                lexer.PROPOSITION(
                    lexer.OPERATION.EQ,
                    lexer.VARIABLE("type"),
                    lexer.VALUE("A")
                )
            )
        )
    )
    expected_ast_case_19 = lexer.Node(
        func=model.delete,
        args=[lexer.Node(
            func=lexer.parse_predicate,
            args=[model.data, lexer.PROPOSITION(
                lexer.OPERATION.AND,
                lexer.PROPOSITION(
                    lexer.OPERATION.GT,
                    lexer.VARIABLE("price"),
                    lexer.VALUE(10)
                ),
                lexer.PROPOSITION(
                    lexer.OPERATION.EQ,
                    lexer.VARIABLE("type"),
                    lexer.VALUE("A")
                )
            )]
        )],
    )
    assert lexer.lex(case_19)[0] == expected_case_19
    assert lexer.Parser(model).parse(expected_case_19) == expected_ast_case_19
def test_case20():
    case_20 = "SUB [x, y, z] # Create's a sub graph with x, y, z as roots"
    expected_case_20 = lexer.ACTION_SUB(lexer.LIST([lexer.VARIABLE("x"), lexer.VARIABLE("y"), lexer.VARIABLE("z")]))
    expected_ast_case_20 = lexer.Node(
        func=model.sub,
        args=[['x', 'y', 'z']],
    )
        
    assert lexer.lex(case_20)[0] == expected_case_20
    assert lexer.Parser(model).parse(expected_case_20) == expected_ast_case_20
def test_case21():
    case_21 = "CUT A # Cut away all nodes under A"
    expected_case_21 = lexer.ACTION_CUT(lexer.VARIABLE("A"))
    expected_ast_case_21 = lexer.Node(
        func=model.cut,
        args=['A'],
    )
    assert lexer.lex(case_21)[0] == expected_case_21
    assert lexer.Parser(model).parse(expected_case_21) == expected_ast_case_21
def test_case22():
    case_22 = "CUT [A, {B: 'y'}] # Cut away all nodes under A and B. Rename B to y"
    expected_case_22 = lexer.ACTION_CUT(lexer.LIST([lexer.VARIABLE("A"), lexer.PROPERTIES({"B": "y"})]))
    expected_ast_case_22 = lexer.Node(
        func=model.cut,
        args=[['A', {'B': 'y'}]],
    )
    assert lexer.lex(case_22)[0] == expected_case_22
    assert lexer.Parser(model).parse(expected_case_22) == expected_ast_case_22
def test_case23():
    case_23 = "CUT {A: 'x', B: 'y'} # Cut away all nodes under A and B. Rename A to x and B to y"
    expected_case_23 = lexer.ACTION_CUT(lexer.PROPERTIES({"A": "x", "B": "y"}))
    expected_ast_case_23 = lexer.Node(
        func=model.cut,
        args=[{'A': 'x', 'B': 'y'}],
    )
    assert lexer.lex(case_23)[0] == expected_case_23
    assert lexer.Parser(model).parse(expected_case_23) == expected_ast_case_23
def test_case24():
    case_24 = "ASSUME {A: 1..2} # Assume A has tighter bound 1..2 and propagates the change"
    expected_case_24 = lexer.ACTION_ASSUME(
        lexer.PROPERTIES({"A": lexer.BOUND(1, 2)}))
    expected_ast_case_24 = lexer.Node(
            func=model.propagate,
            args=[{'A': lexer.Node(func=complex, args=[], kwargs={'real': 1,'imag': 2})}],
        )
    assert lexer.lex(case_24)[0] == expected_case_24
    assert lexer.Parser(model).parse(expected_case_24) == expected_ast_case_24
def test_case25():
    case_25 = "REDUCE # Reduces the graph by removing all nodes with constant bound"
    expected_case_25 = lexer.ACTION_REDUCE()
    expected_ast_case_25 = lexer.Node(
        func=model.propagate,
        args=[],
        kwargs={}
    )

    assert lexer.lex(case_25)[0] == expected_case_25
    assert lexer.Parser(model).parse(expected_case_25) == expected_ast_case_25
def test_case26():
    case_26 = "PROPAGATE {x: 1..2} # Propagates the change of x=1..2 to all nodes"
    expected_case_26 = lexer.ACTION_PROPAGATE(lexer.PROPERTIES({"x": lexer.BOUND(1, 2)}))
    expected_ast_case_26 = lexer.Node(
        func=model.propagate,
        args=[{'x': lexer.Node(
            func=complex,
            kwargs={'real': 1, 'imag':2}
        )}],
    )
    assert lexer.lex(case_26)[0] == expected_case_26
    assert lexer.Parser(model).parse(expected_case_26) == expected_ast_case_26
def test_case27():
    case_27 = "MAXIMIZE {x:1} SUCHTHAT y=1 # Finds a configuration that maximizes x=1 such that y is 1"
    expected_case_27 = lexer.ACTION_MAXIMIZE(lexer.PROPERTIES({"x": 1}), lexer.ASSIGNMENT(lexer.VARIABLE("y"), lexer.INT_VALUE(1)))
    expected_ast_case_27 = lexer.Node(
        func=model.solve,
        args=[[{'x': 1}],
              {'y': 1}]
    )
    assert lexer.lex(case_27)[0] == expected_case_27
    assert lexer.Parser(model).parse(expected_case_27) == expected_ast_case_27
def test_case28():
    case_28 = "MINIMIZE {x:1} SUCHTHAT y=0 # Finds a configuration that minimizes x=1 such that y is 0"
    expected_case_28 = lexer.ACTION_MINIMIZE(
        lexer.PROPERTIES({"x": 1}),
        lexer.ASSIGNMENT(lexer.VARIABLE("y"), lexer.INT_VALUE(0)))
    expected_ast_case_28 = lexer.Node(
        func=model.solve,
        args=[[{'x': 1}],
              {'y': 0}]
    )
    assert lexer.lex(case_28)[0] == expected_case_28
    assert lexer.Parser(model).parse(expected_case_28) == expected_ast_case_28

def test_case29():
    case_29 = "MINIMIZE {x:1} SUCHTHAT (SET AND [y,z])=1 # Finds a configuration that minimizes x=1 such that y and z are true"
    model.set_primitives(['x', 'y', 'z'])
    expected_case_29 = lexer.ACTION_MINIMIZE(
        lexer.PROPERTIES({"x": 1}),
        lexer.ASSIGNMENT(
            lexer.ACTION_SET_LIST_COMPOSITE(
                lexer.SUB_ACTION_TYPE.AND,
                lexer.LIST([lexer.VARIABLE("y"), lexer.VARIABLE("z")]),
            ),
            lexer.INT_VALUE(1)
        )
    )
    expected_ast_case_29 = lexer.Node(
        func=model.solve,
        args=[[{'x': 1}],
              {lexer.Node(
                  func=model.set_and,
                  args=[['y', 'z']],
                  kwargs={'properties': {}}
              ): 1
              }]
    )
    assert lexer.lex(case_29)[0] == expected_case_29
    assert lexer.Parser(model).parse(expected_case_29) == expected_ast_case_29
def test_case30():
    full_input = """
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
    full_lexed = lexer.lex(full_input)
    assert len(full_lexed) == 34