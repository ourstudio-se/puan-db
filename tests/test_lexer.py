from lexer import lex
from pldag import Puan
from datatypes import *
model = Puan()

def test_case1():

    case_1 = "SET a # Set boolean variable a with no attributes"
    expected_case_1 = ACTION_SET_PRIMITIVE(VARIABLE("a"))
    assert lex(case_1)[0] == expected_case_1
    
def test_case2():
    case_2 = "SET b {} # Set boolean variable b with no attributes"
    expected_case_2 = ACTION_SET_PRIMITIVE(VARIABLE("b"), bound=BOUND(0,1))
    assert lex(case_2)[0] == expected_case_2

def test_case3():
    case_3 = "SET c {} -2..3 # Set integer variable c with bounds -2 to 3"
    expected_case_3 = ACTION_SET_PRIMITIVE(VARIABLE("c"), bound=BOUND(-2, 3))
    assert lex(case_3)[0] == expected_case_3

def test_case4():
    case_4 = "SET [d,e,f] # Set boolean variables d,e,f with no attributes"
    expected_case_4 = ACTION_SET_PRIMITIVES(LIST([VARIABLE("d"), VARIABLE("e"), VARIABLE("f")]))
    assert lex(case_4)[0] == expected_case_4

def test_case5():
    case_5 = "SET x {price: 5.0, category: 'Model'} # Set variable x with attributes price and category"
    expected_case_5 = ACTION_SET_PRIMITIVE(VARIABLE("x"), properties=PROPERTIES({"price": 5.0, "category": "Model"}))
    assert lex(case_5)[0] == expected_case_5
    
def test_case6():
    case_6 = "SET ATLEAST 1 [x, y, z] {} # set at least 1 of x, y, z"
    expected_case_6 = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATLEAST, 
        LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]), 
        INT_VALUE(1),
    )
    assert lex(case_6)[0] == expected_case_6

def test_case7():
    case_7 = "SET ATMOST 1 [x, y, z] {} # set at most 1 of x, y, z"
    expected_case_7 = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATMOST, 
        LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]),
        INT_VALUE(1),
    )
    assert lex(case_7)[0] == expected_case_7

def test_case8():
    case_8 = "SET AND [x : x.price > 5] {} # set all x where price > 5"
    expected_case_8 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.AND,  
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    assert lex(case_8)[0] == expected_case_8


def test_case9():
    case_9 = "SET IMPLY [x, y] {} # Set if x then y"
    expected_case_9 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.IMPLY,
        LIST([
            VARIABLE("x"),
            VARIABLE("y")
        ])
    )
    assert lex(case_9)[0] == expected_case_9

def test_case10():
    case_10 = "SET EQUIV [x, y] {} # Set if x then y, if y then x"
    expected_case_10 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.EQUIV,
        LIST([
            VARIABLE("x"),
            VARIABLE("y")
        ])
    )
    assert lex(case_10)[0] == expected_case_10

def test_case11():
    case_11 = "SUB [x : x.price > 7.0] # Create's a sub graph with starting points where price > 7.0"
    expected_case_11 = ACTION_SUB(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(7.0)
            )
        )
    )
    assert lex(case_11)[0] == expected_case_11

def test_case12():
    case_12 = "GET [x, y, z] # Get variables with ID x, y or z's properties and bounds"
    expected_case_12 = ACTION_GET(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    assert lex(case_12)[0] == expected_case_12

def test_case13():
    case_13 = """
        SET AND [
            SET OR [x, y], 
            SET OR [x, z]
        ] # Set complex nested constraint (x OR y) AND (x OR z)
    """
    expected_case_13 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.AND,
        LIST([
            ACTION_SET_LIST_COMPOSITE(
                SUB_ACTION_TYPE.OR,
                LIST([VARIABLE("x"), VARIABLE("y")])
            ),
            ACTION_SET_LIST_COMPOSITE(
                SUB_ACTION_TYPE.OR,
                LIST([VARIABLE("x"), VARIABLE("z")])
            )
        ])
    )
    assert lex(case_13)[0] == expected_case_13

def test_case14():

    case_14 = "GET x # Get variable x's properties and bound"
    expected_case_14 = ACTION_GET(VARIABLE("x"))
    assert lex(case_14)[0] == expected_case_14

def test_case15():

    case_15 = "GET [x : x.price > 5] # Get all x where price > 5"
    expected_case_15 = ACTION_GET(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    assert lex(case_15)[0] == expected_case_15

def test_case16():

    case_16 = "GET [x : type(x) == PRIMITIVE] # Get all x primitive variables"
    expected_case_16 = ACTION_GET(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.EQ,
                FUNCTION("type", VARIABLE("x")),
                DATATYPE.PRIMITIVE,
            )
        )
    )
    assert lex(case_16)[0] == expected_case_16

def test_case17():
    case_17 = "DEL x # Delete variable x"
    expected_case_17 = ACTION_DEL(VARIABLE("x"))
    assert lex(case_17)[0] == expected_case_17

def test_case18():
    case_18 = "DEL [x, y, z] # Delete variables with ID x, y or z"
    expected_case_18 = ACTION_DEL(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    assert lex(case_18)[0] == expected_case_18

def test_case19():
    case_19 = "DEL [x : (x.price > 10) && (x.type == 'A')] # Delete all x where price > 5 and type is A"
    expected_case_19 = ACTION_DEL(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.AND,
                PROPOSITION(
                    OPERATION.GT,
                    VARIABLE("price"),
                    VALUE(10)
                ),
                PROPOSITION(
                    OPERATION.EQ,
                    VARIABLE("type"),
                    VALUE("A")
                )
            )
        )
    )
    assert lex(case_19)[0] == expected_case_19
def test_case20():
    case_20 = "SUB [x, y, z] # Create's a sub graph with x, y, z as roots"
    expected_case_20 = ACTION_SUB(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    assert lex(case_20)[0] == expected_case_20
def test_case21():
    case_21 = "CUT A # Cut away all nodes under A"
    expected_case_21 = ACTION_CUT(VARIABLE("A"))
    assert lex(case_21)[0] == expected_case_21
def test_case22():
    case_22 = "CUT [A, {B: 'y'}] # Cut away all nodes under A and B. Rename B to y"
    expected_case_22 = ACTION_CUT(LIST([VARIABLE("A"), PROPERTIES({"B": "y"})]))
    assert lex(case_22)[0] == expected_case_22
def test_case23():
    case_23 = "CUT {A: 'x', B: 'y'} # Cut away all nodes under A and B. Rename A to x and B to y"
    expected_case_23 = ACTION_CUT(PROPERTIES({"A": "x", "B": "y"}))
    assert lex(case_23)[0] == expected_case_23
def test_case24():
    case_24 = "ASSUME {A: 1..2} # Assume A has tighter bound 1..2 and propagates the change"
    expected_case_24 = ACTION_ASSUME(
        PROPERTIES({"A": BOUND(1, 2)}))
    assert lex(case_24)[0] == expected_case_24
def test_case25():
    case_25 = "REDUCE # Reduces the graph by removing all nodes with constant bound"
    expected_case_25 = ACTION_REDUCE()
    assert lex(case_25)[0] == expected_case_25
def test_case26():
    case_26 = "PROPAGATE {x: 1..2} # Propagates the change of x=1..2 to all nodes"
    expected_case_26 = ACTION_PROPAGATE(PROPERTIES({"x": BOUND(1, 2)}))
    assert lex(case_26)[0] == expected_case_26
def test_case27():
    case_27 = "MAXIMIZE {x:1} SUCHTHAT y=1 # Finds a configuration that maximizes x=1 such that y is 1"
    expected_case_27 = ACTION_MAXIMIZE(PROPERTIES({"x": 1}), ASSIGNMENT(VARIABLE("y"), INT_VALUE(1)))
    assert lex(case_27)[0] == expected_case_27
def test_case28():
    case_28 = "MINIMIZE {x:1} SUCHTHAT y=0 # Finds a configuration that minimizes x=1 such that y is 0"
    expected_case_28 = ACTION_MINIMIZE(
        PROPERTIES({"x": 1}),
        ASSIGNMENT(VARIABLE("y"), INT_VALUE(0)))
    assert lex(case_28)[0] == expected_case_28

def test_case30():
    case_29 = "MINIMIZE {x:1} SUCHTHAT (SET AND [y,z])=1 # Finds a configuration that minimizes x=1 such that y and z are true"
    model.set_primitives(['x', 'y', 'z'])
    expected_case_29 = ACTION_MINIMIZE(
        PROPERTIES({"x": 1}),
        ASSIGNMENT(
            ACTION_SET_LIST_COMPOSITE(
                SUB_ACTION_TYPE.AND,
                LIST([VARIABLE("y"), VARIABLE("z")]),
            ),
            INT_VALUE(1)
        )
    )
    assert lex(case_29)[0] == expected_case_29
def test_case31():
    full_input = """
        SET a                                       # Set boolean variable a with no attributes
        SET b {}                                    # Set boolean variable b with no attributes
        SET c {} -2..3                              # Set integer variable c with bounds -2 to 3
        SET [d,e,f]                                 # Set boolean variables d,e,f with no attributes

        SET x {price: 5.0, category: 'Model'}       # Set variable x with attributes price and category

        SET ATLEAST 1 [x, y, z] {}                  # set at least 1 of x, y, z
        SET ATMOST 1 [x, y, z] {}                   # set at most 1 of x, y, z
        
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
        ASSUME {A: 1}                               # Assume A to be exaclty 1
        REDUCE                                      # Reduces the graph by removing all nodes with constant bound


        PROPAGATE {x: 1..2}                         # Propagates the change of x=1..2 to all nodes
        MAXIMIZE {x:1} SUCHTHAT y=1                 # Finds a configuration that maximizes x=1 such that y is true
        MINIMIZE {X:1} SUCHTHAT y=0                 # Finds a configuration that minimizes x=1 such that y is false
        MINIMIZE {x:1} SUCHTHAT (SET AND [y,z])=1   # Finds a configuration that minimizes x=1 such that y and z are true
    """
    full_lexed = lex(full_input)
    assert len(full_lexed) == 34

def test_case31():
    case_31 = """SET AND [ SET x, SET y, SET OR [ m, n] ]
    SET AND [ SET x, SET y, SET OR [ SET m, SET n] ] """
    expected_case_31 = [
        ACTION_SET_LIST_COMPOSITE(
            sub_action=SUB_ACTION_TYPE.AND,
            arguments=LIST(
                items=[
                    ACTION_SET_PRIMITIVE(argument=VARIABLE(id='x')),
                    ACTION_SET_PRIMITIVE(argument=VARIABLE(id='y')),
                    ACTION_SET_LIST_COMPOSITE(
                        sub_action=SUB_ACTION_TYPE.OR,
                        arguments=LIST(items=[VARIABLE(id='m'), VARIABLE(id='n')]))]),
        ),
        ACTION_SET_LIST_COMPOSITE(
            sub_action=SUB_ACTION_TYPE.AND,
            arguments=LIST(
                items=[
                    ACTION_SET_PRIMITIVE(argument=VARIABLE(id='x')),
                    ACTION_SET_PRIMITIVE(argument=VARIABLE(id='y')),
                    ACTION_SET_LIST_COMPOSITE(
                        sub_action=SUB_ACTION_TYPE.OR,
                        arguments=LIST(
                            items=[
                                ACTION_SET_PRIMITIVE(argument=VARIABLE(id='m')),
                                ACTION_SET_PRIMITIVE(argument=VARIABLE(id='n'))
                            ]),
                        )]),
        )
    ]
    assert lex(case_31) == expected_case_31

def test_atleast():
    query = "SET ATLEAST 2 [x, y, z] {} # set at least 2 of x, y, z"
    expected = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATLEAST,
        INT_VALUE(2),
        LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]),
    )
    assert lex(query)[0] == expected
    
def test_atmost():
    query = "SET ATMOST 2 [x, y, z] {} # set at most 2 of x, y, z"
    expected = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATMOST,
        INT_VALUE(2),
        LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]),
    )
    assert lex(query)[0] == expected

def test_imply():
    query = "SET IMPLY [x, y] {} # Set if x then y"
    expected = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.IMPLY,
        LIST([
            VARIABLE("x"),
            VARIABLE("y")
        ])
    )
    assert lex(query)[0] == expected

def test_equiv():
    query = "SET EQUIV [x, y] {} # Set if x then y, if y then x"
    expected = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.EQUIV,
        LIST([
            VARIABLE("x"),
            VARIABLE("y")
        ])
    )
    assert lex(query)[0] == expected

def test_and():
    query = "SET AND [x : x.price > 5] {} # set all x where price > 5"
    expected = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.AND,  
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    assert lex(query)[0] == expected

def test_or():
    query = "SET OR [x : x.price > 5] {} # set all x where price > 5"
    expected = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.OR,  
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    assert lex(query)[0] == expected

def test_xor():
    query = "SET XOR [x : x.price > 5] {} # set all x where price > 5"
    expected = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.XOR,  
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    assert lex(query)[0] == expected

def test_not():
    query = "SET NOT [x : x.price > 5] {} # set all x where price > 5"
    expected = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.NOT,  
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    assert lex(query)[0] == expected

def test_set_primitive():
    query = "SET x # Set boolean variable x with no attributes"
    expected = ACTION_SET_PRIMITIVE(VARIABLE("x"))
    assert lex(query)[0] == expected

    query = "SET x {} # Set boolean variable x with no attributes"
    expected = ACTION_SET_PRIMITIVE(VARIABLE("x"), bound=BOUND(0,1))
    assert lex(query)[0] == expected

    query = "SET x -2..3 # Set integer variable x with bounds -2 to 3"
    expected = ACTION_SET_PRIMITIVE(VARIABLE("x"), bound=BOUND(-2, 3))
    assert lex(query)[0] == expected

    query = "SET x {price: 5.0, category: 'Model'} # Set variable x with attributes price and category"
    expected = ACTION_SET_PRIMITIVE(VARIABLE("x"), properties=PROPERTIES({"price": 5.0, "category": "Model"}))
    assert lex(query)[0] == expected

    query = "SET x {price: 5.0, category: 'Model'} -2..3 # Set variable x with attributes price and category"
    expected = ACTION_SET_PRIMITIVE(VARIABLE("x"), properties=PROPERTIES({"price": 5.0, "category": "Model"}), bound=BOUND(-2, 3))
    assert lex(query)[0] == expected

    query = "SET x {text: 'hej : pa : dig'}"
    expected = ACTION_SET_PRIMITIVE(VARIABLE("x"), properties=PROPERTIES({"text": "hej : pa : dig"}))
    assert lex(query)[0] == expected

def test_set_primitives():
    query = "SET [x, y, z] # Set boolean variables x, y, z with no attributes"
    expected = ACTION_SET_PRIMITIVES(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    assert lex(query)[0] == expected
    
    query = "SET [x, y, z] {} # Set boolean variables x, y, z with no attributes"
    expected = ACTION_SET_PRIMITIVES(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    assert lex(query)[0] == expected

    query = "SET [x, y, z] -2..3 # Set boolean variables x, y, z with no attributes"
    expected = ACTION_SET_PRIMITIVES(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]), bound=BOUND(-2, 3))
    assert lex(query)[0] == expected

    query = "SET [x, y, z] {price: 5.0, category: 'Model'} # Set boolean variables x, y, z with no attributes"
    expected = ACTION_SET_PRIMITIVES(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]), properties=PROPERTIES({"price": 5.0, "category": "Model"}))
    assert lex(query)[0] == expected

    query = "SET [x, y, z] {price: 5.0, category: 'Model'} -2..3 # Set boolean variables x, y, z with no attributes"
    expected = ACTION_SET_PRIMITIVES(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]), properties=PROPERTIES({"price": 5.0, "category": "Model"}), bound=BOUND(-2, 3))
    assert lex(query)[0] == expected

def test_functions_as_property_values():
    query = """SET XOR [x : x.type == "MODEL"] {name: "Model", price: sum()}"""
    lexed = lex(query)[0]
    1