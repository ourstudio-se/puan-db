import puan_db_parser
from pldag import Puan, Solver
from datatypes import *
import pytest
import numpy as np

def test_case1():
    model = Puan()
    case_1 = ACTION_SET_PRIMITIVE(VARIABLE("a"))
    m, s = puan_db_parser.Parser(model).evaluate(case_1)

    model.set_primitive('a')
    assert m == model and s == model.propagate({})
    
def test_case2():
    model = Puan()
    case_2 = ACTION_SET_PRIMITIVE(VARIABLE("b"), bound=BOUND(0,1))
    m, s = puan_db_parser.Parser(model).evaluate(case_2)
    model.set_primitive('b', bound=complex(0,1))
    assert m == model and s == model.propagate({})

def test_case3():
    model = Puan()
    case_3 = ACTION_SET_PRIMITIVE(VARIABLE("c"), bound=BOUND(-2, 3))
    m, s = puan_db_parser.Parser(model).evaluate(case_3)
    model.set_primitive('c', bound=complex(-2, 3))
    assert m == model and s == model.propagate({})

def test_case4():
    model = Puan()
    case_4 = ACTION_SET_PRIMITIVES(LIST([VARIABLE("d"), VARIABLE("e"), VARIABLE("f")]))
    m, s = puan_db_parser.Parser(model).evaluate(case_4)
    model.set_primitives(['d', 'e', 'f'])
    assert m == model and s == model.propagate({})

def test_case5():
    model = Puan()
    case_5 = ACTION_SET_PRIMITIVE(VARIABLE("x"), properties=PROPERTIES({"price": 5.0, "category": "Model"}))
    m, s = puan_db_parser.Parser(model).evaluate(case_5)
    model.set_primitive('x', properties={"price": 5.0, "category": "Model"})
    assert m == model and s == model.propagate({})
    
def test_case6():
    model = Puan()
    case_6 = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATLEAST, 
        INT_VALUE(1),
        LIST([ACTION_SET_PRIMITIVE(VARIABLE("x")), ACTION_SET_PRIMITIVE(VARIABLE("y")), ACTION_SET_PRIMITIVE(VARIABLE("z"))]), 
    )
    m, s = puan_db_parser.Parser(model).evaluate(case_6)
    model.set_atleast([model.set_primitive('x'), model.set_primitive('y'), model.set_primitive('z')], 1)
    assert m == model and s == model.propagate({})

def test_case7():
    model = Puan()
    case_7 = ACTION_SET_VALUE_COMPOSITE(
        SUB_ACTION_TYPE.ATMOST, 
        INT_VALUE(1),
        LIST([ACTION_SET_PRIMITIVE(VARIABLE("x")), ACTION_SET_PRIMITIVE(VARIABLE("y")), ACTION_SET_PRIMITIVE(VARIABLE("z"))]),
    )
    m, s = puan_db_parser.Parser(model).evaluate(case_7)
    model.set_atmost([model.set_primitive('x'), model.set_primitive('y'), model.set_primitive('z')], 1)
    assert m == model and s == model.propagate({})

def test_case8():
    model = Puan()
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
    m, s = puan_db_parser.Parser(model).evaluate(case_8)
    model.set_and(model.find(lambda k, v: k == 'price' and v > 5))
    assert m == model and s == model.propagate({})


def test_case9():
    model = Puan()
    case_9 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.IMPLY,
        LIST([
            ACTION_SET_PRIMITIVE(VARIABLE("x")),
            ACTION_SET_PRIMITIVE(VARIABLE("y"))
        ])
    )
    m, s = puan_db_parser.Parser(model).evaluate(case_9)
    model.set_imply(model.set_primitive('x'), model.set_primitive('y'))
    assert m == model and s == model.propagate({})
def test_case10():
    model = Puan()
    case_10 = ACTION_SET_LIST_COMPOSITE(
        SUB_ACTION_TYPE.EQUIV,
        LIST([
            ACTION_SET_PRIMITIVE(VARIABLE("x")),
            ACTION_SET_PRIMITIVE(VARIABLE("y"))
        ])
    )
    m, s = puan_db_parser.Parser(model).evaluate(case_10)
    model.set_equal([model.set_primitive("x"), model.set_primitive("y")])

    assert m == model and s == model.propagate({})

def test_case11():
    model = Puan()
    model.set_primitives(["x", "y", "z"], {'price': 10})
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
    m, s = puan_db_parser.Parser(model).evaluate(case_11)
    m_expected = model.sub(model.find(lambda k, v: k == 'price' and v > 7.0))
    assert m == m_expected and s == m_expected.propagate({})

def test_case12():
    model = Puan()
    model.set_primitives(["x", "y", "z"])
    case_12 = ACTION_GET(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    assert np.array_equal(puan_db_parser.Parser(model).parse(case_12)(), np.array([complex(0, 1), complex(0, 1), complex(0, 1)]))  
def test_case13():
    """
        SET AND [
            SET OR [x, y], 
            SET OR [x, z]
        ] # Set complex nested constraint (x OR y) AND (x OR z)
    """
    model = Puan()
    model.set_primitives(["x", "y", "z"])
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
    m, s = puan_db_parser.Parser(model).evaluate(case_13)
    model.set_and([model.set_or(['x', 'y']), model.set_or(['x', 'z'])])
    
    assert m == model and s == model.propagate({})

def test_case14():
    """GET x # Get variable x's properties and bound"""
    model = Puan()
    model.set_primitive("x")
    case_14 = ACTION_GET(VARIABLE("x"))
    assert np.array_equal(puan_db_parser.Parser(model).parse(case_14)(), np.array([complex(0, 1)]))

def test_case15():
    """GET [x : x.price > 5] # Get all x where price > 5"""
    model = Puan()
    model.set_primitives(["x", "y", "z"], {'price': 10})
    model.set_primitives(["a", "b", "c"], {'price': 5})
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
    assert np.array_equal(puan_db_parser.Parser(model).parse(case_15)(),model.get(*model.find(lambda k, v: k == 'price' and v > 5)))

def test_case16():
    """GET [x : type(x) == PRIMITIVE] # Get all x primitive variables"""
    model = Puan()
    model.set_primitives(["x", "y", "z"])
    model.set_and(["x", "y", "z"])
    
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
    assert np.array_equal(puan_db_parser.Parser(model).parse(case_16)(),model.get(*list(filter(lambda id: puan_db_parser.our_type(model, id)==puan_db_parser.DATATYPE.PRIMITIVE, model.ids))))

def test_case17():
    """DEL x # Delete variable x"""
    model = Puan()
    model.set_primitives(["x", "y", "z"])
    case_17 = ACTION_DEL(VARIABLE("x"))
    m, s = puan_db_parser.Parser(model).evaluate(case_17)
    model.delete('x')
    assert m == model and s == model.propagate({})
def test_case18():
    """DEL [x, y, z] # Delete variables with ID x, y or z"""
    model = Puan()
    model.set_primitives(["x", "y", "z"])
    case_18 = ACTION_DEL(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    m, s = puan_db_parser.Parser(model).evaluate(case_18)
    model.delete(*['x', 'y', 'z'])
    assert m == model and s == model.propagate({})
def test_case19():
    """DEL [x : (x.price > 10) && (x.type == 'A')] # Delete all x where price > 5 and type is A"""
    model = Puan()
    model.set_primitives(["x", "y", "z"], {'price': 11})
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
                    FUNCTION("type", VARIABLE("x")),
                    DATATYPE.PRIMITIVE
                )
            )
        )
    )
    m, s = puan_db_parser.Parser(model).evaluate(case_19)
    model.delete(*list(filter(lambda id: model.data.get(id).get('price', 0) > 10 and puan_db_parser.our_type(model, id) == DATATYPE.PRIMITIVE, model.ids)))
    assert m == model and s == model.propagate({})
def test_case20():
    """SUB [x, y, z] # Create's a sub graph with x, y, z as roots"""
    model = Puan()
    model.set_primitives(["x", "y", "z"])
    case_20 = ACTION_SUB(LIST([VARIABLE("x"), VARIABLE("y"), VARIABLE("z")]))
    m, s = puan_db_parser.Parser(model).evaluate(case_20)
    model.sub(['x', 'y', 'z'])
    assert m == model and s == model.propagate({})
def test_case21():
    """CUT A # Cut away all nodes under A"""
    model = Puan()
    model.set_primitives(["x", "y", "z"])
    model.set_and(["x", "y", "z"])
    case_21 = ACTION_CUT(VARIABLE('b7bf05ec0ed35b049bc6d22e20ce64aaa6014e22'))
    m, s = puan_db_parser.Parser(model).evaluate(case_21)
    m_expected = model.cut({model.set_and(["x", "y", "z"]): model.set_and(["x", "y", "z"])})
    assert m == m_expected and s == m_expected.propagate({})
def test_case22():
    """CUT [A, {B: 'y'}] # Cut away all nodes under A and B. Rename B to y"""
    model = Puan()
    model.set_primitives(["A", "B"])
    case_22 = ACTION_CUT(LIST([VARIABLE("A"), PROPERTIES({"B": "y"})]))
    m, s = puan_db_parser.Parser(model).evaluate(case_22)
    model.cut({'A': 'A', 'B': 'y'})
    assert m == model and s == model.propagate({})
def test_case23():
    """CUT {A: 'x', B: 'y'} # Cut away all nodes under A and B. Rename A to x and B to y"""
    model = Puan()
    model.set_primitives(["A", "B"])
    case_23 = ACTION_CUT(PROPERTIES({"A": "x", "B": "y"}))
    m, s = puan_db_parser.Parser(model).evaluate(case_23)
    m_expected = model.cut({'A': 'x', 'B': 'y'})
    assert m == m_expected and s == m_expected.propagate({})
def test_case24():
    """ASSUME {A: 1..2} # Assume A has tighter bound 1..2 and propagates the change"""
    model = Puan()
    model.set_primitive('A', bound=complex(0, 5))
    case_24 = ACTION_ASSUME(
        PROPERTIES({"A": BOUND(1, 2)}))
    m, s = puan_db_parser.Parser(model).evaluate(case_24)
    s_expected = model.propagate({'A': complex(1, 2)})
    assert m == model and s == s_expected
def test_case25():
    """REDUCE # Reduces the graph by removing all nodes with constant bound"""
    model = Puan()
    model.set_primitives(["x", "y", "z"], bound=complex(0, 1))
    model.set_primitives(["a", "b", "c"], bound=complex(1, 1))
    case_25 = ACTION_REDUCE()
    m, s = puan_db_parser.Parser(model).evaluate(case_25)
    s_expected = model.propagate({})
    assert m == model and s == s_expected
def test_case26():
    """PROPAGATE {x: 1..2} # Propagates the change of x=1..2 to all nodes"""
    model = Puan()
    model.set_primitives(["x", "y", "z"], bound=complex(0, 1))
    case_26 = ACTION_PROPAGATE(PROPERTIES({"x": BOUND(1, 2)}))
    m, s = puan_db_parser.Parser(model).evaluate(case_26)
    s_expected = model.propagate({'x': complex(1, 2)})
    assert m == model and s == s_expected
    
def test_case27():
    """MAXIMIZE {x:1} SUCHTHAT y=1 # Finds a configuration that maximizes x=1 such that y is 1"""
    model = Puan()
    model.set_primitives(["x", "y"], bound=complex(0, 1))
    model.set_or(["x", "y"])
    case_27 = ACTION_MAXIMIZE(PROPERTIES({"x": 1}), ASSIGNMENT(VARIABLE("y"), INT_VALUE(1)))
    _, s = puan_db_parser.Parser(model).evaluate(case_27)
    s_expected = model.solve([{'x': 1}], {'y': complex(1, 1)}, Solver.GLPK)[0]
    assert s == s_expected
def test_case28():
    """MINIMIZE {x:1} SUCHTHAT y=0 # Finds a configuration that minimizes x=1 such that y is 0"""
    model = Puan()
    model.set_primitives(["x", "y"], bound=complex(0, 1))
    model.set_or(["x", "y"])
    case_28 = ACTION_MINIMIZE(
        PROPERTIES({"x": 1}),
        ASSIGNMENT(VARIABLE("y"), INT_VALUE(0)))
    m, s = puan_db_parser.Parser(model).evaluate(case_28)
    s_expected = model.solve([{'x': 1}], {'y': complex(0, 0)}, Solver.GLPK)[0]
    assert s == s_expected

def test_case29():
    """MINIMIZE {x:1} SUCHTHAT (SET AND [y,z])=1 # Finds a configuration that minimizes x=1 such that y and z are true"""
    model = Puan()
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
    m, s = puan_db_parser.Parser(model).evaluate(case_29)
    s_expected = model.solve([{'x': 1}], {model.set_and(["y", "z"]): complex(1,1)}, Solver.GLPK)[0]
    assert s == s_expected

def test_case30():
    model = Puan()
    case_34 = [ACTION_SET_LIST_COMPOSITE(sub_action=SUB_ACTION_TYPE.OR, arguments=LIST(items=[ACTION_SET_PRIMITIVE(argument=VARIABLE(id='m')), ACTION_SET_PRIMITIVE(argument=VARIABLE(id='n'))])),
               ACTION_SET_LIST_COMPOSITE(sub_action=SUB_ACTION_TYPE.OR, arguments=LIST(items=[ACTION_SET_PRIMITIVE(argument=VARIABLE(id='x')), ACTION_SET_PRIMITIVE(argument=VARIABLE(id='y'))]))]
    puan_db_parser.Parser(model).evaluate(case_34)[0]

def test_case31():
    model = Puan()
    case_35 = [
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
    with pytest.raises(KeyError):
        model, solution = puan_db_parser.Parser(model).evaluate(case_35[0])
    model, solution = puan_db_parser.Parser(model).evaluate(case_35[1])
    assert model._imap == {'x': 0, 'y': 1, 'm': 2, 'n': 3, 'c8f3d9b100e0de8d1e2971cf78ec988d0bc2ed26': 4, '03ae5b8ca446c0fa36b2b4c44374482b9df1c061': 5}

def test_case32():
    model = Puan()
    model.set_primitive('x', properties={"price": 10})
    model.set_primitive('y', properties={"price": 5})
    case_32 = ACTION_SET_LIST_COMPOSITE(
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
    m, s = puan_db_parser.Parser(model).evaluate(case_32)
    model.set_not(model.find(lambda k, v: k == 'price' and v > 5))
    assert m == model and s == model.propagate({})
