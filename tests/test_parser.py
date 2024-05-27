import puan_db_parser
import maz
from pldag import Puan
from datatypes import *
model = Puan()

def test_case1():
    case_1 = ACTION_SET_PRIMITIVE(VARIABLE("a"))
    expected_case_1 = puan_db_parser.Node(
            func=model.set_primitive,
            args=['a'],
            kwargs={'properties': {},  # 'properties' is empty because there are no attributes
                'bound': puan_db_parser.Node(
                func=complex,
                args=[],
                kwargs = {'real':0, 'imag': 1}
            )}
        )
    assert puan_db_parser.Parser(model).parse(case_1) == expected_case_1
    
def test_case2():
    case_2 = ACTION_SET_PRIMITIVE(VARIABLE("b"), bound=BOUND(0,1))
    expected_case_2 = puan_db_parser.Node(
            func=model.set_primitive,
            args=['b'],
            kwargs={'properties': {}, 'bound': puan_db_parser.Node(
                func=complex,
                args=[],
                kwargs={'real': 0, 'imag':1}
            )}
        )
    assert puan_db_parser.Parser(model).parse(case_2) == expected_case_2

def test_case3():
    case_3 = ACTION_SET_PRIMITIVE(VARIABLE("c"), bound=BOUND(-2, 3))
    expected_case_3 = puan_db_parser.Node(
            func=model.set_primitive,
            args=['c'],
            kwargs={'properties': {}, 'bound': puan_db_parser.Node(
                func=complex,
                args=[],
                kwargs={'real': -2, 'imag':3}
            )}
        )
    assert puan_db_parser.Parser(model).parse(case_3) == expected_case_3

def test_case4():
    case_4 = ACTION_SET_PRIMITIVES(LIST([VARIABLE("d"), VARIABLE("e"), VARIABLE("f")]))
    expected_case_4 = puan_db_parser.Node(
            func=model.set_primitives,
            args=[['d', 'e','f']],
            kwargs={'properties': {}, 'bound': puan_db_parser.Node(
                func=complex,
                args=[],
                kwargs={'real': 0, 'imag':1}
            )}
        )
    assert puan_db_parser.Parser(model).parse(case_4) == expected_case_4

def test_case5():
    case_5 = ACTION_SET_PRIMITIVE(VARIABLE("x"), properties=PROPERTIES({"price": 5.0, "category": "Model"}))
    expected_case_5 = puan_db_parser.Node(
            func=model.set_primitive,
            args=['x'],
            kwargs={'properties': {'price': 5.0, 'category': 'Model'}, 'bound': puan_db_parser.Node(
                func=complex,
                args=[],
                kwargs={'real': 0, 'imag':1}
            )}
        )
    assert puan_db_parser.Parser(model).parse(case_5) == expected_case_5
    
def test_case6():
    case_6 = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATLEAST, 
        LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]), 
        INT_VALUE(1),
    )
    expected_case_6 = puan_db_parser.Node(
            func=model.set_atleast,
            args=[['x', 'y', 'z'], 1],
            kwargs={'properties': {}}
        )
    assert puan_db_parser.Parser(model).parse(case_6) == expected_case_6

def test_case7():
    case_7 = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATMOST, 
        LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]),
        INT_VALUE(1),
    )
    expected_case_7 = puan_db_parser.Node(
            func=model.set_atmost,
            args=[['x', 'y', 'z'], 1],
            kwargs={'properties': {}}
        )
    assert puan_db_parser.Parser(model).parse(case_7) == expected_case_7

def test_case8():
    case_8 = ACTION_SET_LIST_COMPOSITE(
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
    expectedcase_case_8 = puan_db_parser.Node(
        func=model.set_and,
        args=[puan_db_parser.Node(
                func=puan_db_parser.parse_predicate,
                args=[model.data, PROPOSITION(
                        OPERATION.GT,
                        VARIABLE("price"),
                        VALUE(5)
                    )
                ],
        )],
        kwargs={'properties': {}}
    )
    assert puan_db_parser.Parser(model).parse(case_8) == expectedcase_case_8


def test_case9():
    case_9 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.IMPLY,
        LIST([
            VARIABLE("x"),
            VARIABLE("y")
        ])
    )
    expectedcase_case_9 = puan_db_parser.Node(
            func=model.set_imply,
            args=['x', 'y'],
            kwargs={'properties': {}}
        )
    assert puan_db_parser.Parser(model).parse(case_9) == expectedcase_case_9

def test_case10():
    case_10 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.EQUIV,
        LIST([
            VARIABLE("x"),
            VARIABLE("y")
        ])
    )
    expectedcase_case_10 = puan_db_parser.Node(
            func=model.set_equal,
            args=[['x', 'y']],
            kwargs={'properties': {}}
        )
    assert puan_db_parser.Parser(model).parse(case_10) == expectedcase_case_10

def test_case11():
    case_11 = ACTION_SUB(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(7.0)
            )
        )
    )
    expectedcase_case_11 = puan_db_parser.Node(
            func=model.sub,
            args=[puan_db_parser.Node(
                func=puan_db_parser.parse_predicate,
                args=[model.data, PROPOSITION(
                    OPERATION.GT,
                    VARIABLE("price"),
                    VALUE(7.0)
                )]
            )],
            kwargs={}
        )
    assert puan_db_parser.Parser(model).parse(case_11) == expectedcase_case_11

def test_case12():
    case_12 = ACTION_GET(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    expectedcase_case_12 = puan_db_parser.Node(
            func=model.get,
            args=[['x', 'y', 'z']],
        )
    assert puan_db_parser.Parser(model).parse(case_12) == expectedcase_case_12

def test_case13():
    """
        SET AND [
            SET OR [x, y], 
            SET OR [x, z]
        ] # Set complex nested constraint (x OR y) AND (x OR z)
    """
    case_13 = ACTION_SET_LIST_COMPOSITE(
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
    expectedcase_case_13 = puan_db_parser.Node(
        func=model.set_and,
        args=[
            [
                puan_db_parser.Node(
                    func=model.set_or,
                    args=[['x', 'y']],
                    kwargs={'properties': {}}
                ),
                puan_db_parser.Node(
                    func=model.set_or,
                    args=[['x', 'z']],
                    kwargs={'properties': {}}
                )
            ]
        ],
        kwargs={'properties': {}}
    )
    assert puan_db_parser.Parser(model).parse(case_13) == expectedcase_case_13

def test_case14():
    """GET x # Get variable x's properties and bound"""
    case_14 = ACTION_GET(VARIABLE("x"))
    expectedcase_case_14 = puan_db_parser.Node(
        func=model.get,
        args=['x'],
    )
    assert puan_db_parser.Parser(model).parse(case_14) == expectedcase_case_14

def test_case15():
    """GET [x : x.price > 5] # Get all x where price > 5"""
    case_15 = ACTION_GET(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )
        )
    )
    expectedcase_case_15 = puan_db_parser.Node(
        func=model.get,
        args=[puan_db_parser.Node(
            func=puan_db_parser.parse_predicate,
            args=[model.data, PROPOSITION(
                OPERATION.GT,
                VARIABLE("price"),
                VALUE(5)
            )]
        )],
    )
    assert puan_db_parser.Parser(model).parse(case_15) == expectedcase_case_15

def test_case16():
    """GET [x : type(x) == PRIMITIVE] # Get all x primitive variables"""
    case_16 = ACTION_GET(
        PREDICATE(
            id="x",
            proposition=PROPOSITION(
                OPERATION.EQ,
                FUNCTION("type", VARIABLE("x")),
                DATATYPE.PRIMITIVE,
            )
        )
    )
    expectedcase_case_16 = puan_db_parser.Node(
        func=model.get,
        args=[puan_db_parser.Node(
            func=puan_db_parser.parse_predicate,
            args=[model.data, PROPOSITION(
                OPERATION.EQ,
                FUNCTION("type", VARIABLE("x")),
                DATATYPE.PRIMITIVE,
            )]
        )],
    )
    assert puan_db_parser.Parser(model).parse(case_16) == expectedcase_case_16

def test_case17():
    """DEL x # Delete variable x"""
    case_17 = ACTION_DEL(VARIABLE("x"))
    expectedcase_case_17 =puan_db_parser.Node(
        func=model.delete,
        args=['x'],
    )
    assert puan_db_parser.Parser(model).parse(case_17) == expectedcase_case_17

def test_case18():
    """DEL [x, y, z] # Delete variables with ID x, y or z"""
    case_18 = ACTION_DEL(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    expectedcase_case_18 = puan_db_parser.Node(
        func=model.delete,
        args=[['x', 'y', 'z']],
    )
    assert puan_db_parser.Parser(model).parse(case_18) == expectedcase_case_18

def test_case19():
    """DEL [x : (x.price > 10) && (x.type == 'A')] # Delete all x where price > 5 and type is A"""
    case_19 = ACTION_DEL(
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
    expectedcase_case_19 = puan_db_parser.Node(
        func=model.delete,
        args=[puan_db_parser.Node(
            func=puan_db_parser.parse_predicate,
            args=[model.data, PROPOSITION(
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
            )]
        )],
    )
    assert puan_db_parser.Parser(model).parse(case_19) == expectedcase_case_19
def test_case20():
    case_20 = "SUB [x, y, z] # Create's a sub graph with x, y, z as roots"
    case_20 = ACTION_SUB(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    expectedcase_case_20 = puan_db_parser.Node(
        func=model.sub,
        args=[['x', 'y', 'z']],
    )
        
    assert puan_db_parser.Parser(model).parse(case_20) == expectedcase_case_20
def test_case21():
    """CUT A # Cut away all nodes under A"""
    case_21 = ACTION_CUT(VARIABLE("A"))
    expectedcase_case_21 = puan_db_parser.Node(
        func=model.cut,
        args=['A'],
    )
    assert puan_db_parser.Parser(model).parse(case_21) == expectedcase_case_21
def test_case22():
    """CUT [A, {B: 'y'}] # Cut away all nodes under A and B. Rename B to y"""
    case_22 = ACTION_CUT(LIST([VARIABLE("A"), PROPERTIES({"B": "y"})]))
    expectedcase_case_22 = puan_db_parser.Node(
        func=model.cut,
        args=[['A', {'B': 'y'}]],
    )
    assert puan_db_parser.Parser(model).parse(case_22) == expectedcase_case_22
def test_case23():
    """CUT {A: 'x', B: 'y'} # Cut away all nodes under A and B. Rename A to x and B to y"""
    case_23 = ACTION_CUT(PROPERTIES({"A": "x", "B": "y"}))
    expectedcase_case_23 = puan_db_parser.Node(
        func=model.cut,
        args=[{'A': 'x', 'B': 'y'}],
    )
    assert puan_db_parser.Parser(model).parse(case_23) == expectedcase_case_23
def test_case24():
    """ASSUME {A: 1..2} # Assume A has tighter bound 1..2 and propagates the change"""
    case_24 = ACTION_ASSUME(
        PROPERTIES({"A": BOUND(1, 2)}))
    expectedcase_case_24 = puan_db_parser.Node(
            func=model.propagate,
            args=[{'A': puan_db_parser.Node(func=complex, args=[], kwargs={'real': 1,'imag': 2})}],
        )
    assert puan_db_parser.Parser(model).parse(case_24) == expectedcase_case_24
def test_case25():
    """REDUCE # Reduces the graph by removing all nodes with constant bound"""
    case_25 = ACTION_REDUCE()
    expectedcase_case_25 = puan_db_parser.Node(
        func=model.propagate,
        args=[],
        kwargs={}
    )
    assert puan_db_parser.Parser(model).parse(case_25) == expectedcase_case_25
def test_case26():
    """PROPAGATE {x: 1..2} # Propagates the change of x=1..2 to all nodes"""
    case_26 = ACTION_PROPAGATE(PROPERTIES({"x": BOUND(1, 2)}))
    expectedcase_case_26 = puan_db_parser.Node(
        func=model.propagate,
        args=[{'x': puan_db_parser.Node(
            func=complex,
            kwargs={'real': 1, 'imag':2}
        )}],
    )
    assert puan_db_parser.Parser(model).parse(case_26) == expectedcase_case_26
def test_case27():
    """MAXIMIZE {x:1} SUCHTHAT y=1 # Finds a configuration that maximizes x=1 such that y is 1"""
    case_27 = ACTION_MAXIMIZE(PROPERTIES({"x": 1}), ASSIGNMENT(VARIABLE("y"), INT_VALUE(1)))
    expectedcase_case_27 = puan_db_parser.Node(
        func=model.solve,
        args=[[{'x': 1}],
              {'y': 1}]
    )
    assert puan_db_parser.Parser(model).parse(case_27) == expectedcase_case_27
def test_case28():
    """MINIMIZE {x:1} SUCHTHAT y=0 # Finds a configuration that minimizes x=1 such that y is 0"""
    case_28 = ACTION_MINIMIZE(
        PROPERTIES({"x": 1}),
        ASSIGNMENT(VARIABLE("y"), INT_VALUE(0)))
    expectedcase_case_28 = puan_db_parser.Node(
        func=model.solve,
        args=[[{'x': 1}],
              {'y': 0}]
    )
    assert puan_db_parser.Parser(model).parse(case_28) == expectedcase_case_28

def test_case29():
    """MINIMIZE {x:1} SUCHTHAT (SET AND [y,z])=1 # Finds a configuration that minimizes x=1 such that y and z are true"""
    model.set_primitives(['x', 'y', 'z'])
    case_29 = ACTION_MINIMIZE(
        PROPERTIES({"x": 1}),
        ASSIGNMENT(
            ACTION_SET_LIST_COMPOSITE(
                SUB_ACTION_TYPE.AND,
                LIST([VARIABLE("y"), VARIABLE("z")]),
            ),
            INT_VALUE(1)
        )
    )
    expectedcase_case_29 = puan_db_parser.Node(
        func=model.solve,
        args=[[{'x': 1}],
              {puan_db_parser.Node(
                  func=model.set_and,
                  args=[['y', 'z']],
                  kwargs={'properties': {}}
              ): 1
              }]
    )
    assert puan_db_parser.Parser(model).parse(case_29) == expectedcase_case_29
