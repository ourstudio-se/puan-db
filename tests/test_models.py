import api.models.schema as schema_models
import api.models.typed_model as typed_models

def test_define_property():
    p = schema_models.SchemaProperty(dtype="string")
    assert p.dtype == schema_models.SchemaPropertyDType.string
    
    p = schema_models.SchemaProperty(dtype="integer")
    assert p.dtype == schema_models.SchemaPropertyDType.integer
    
    p = schema_models.SchemaProperty(dtype="boolean")
    assert p.dtype == schema_models.SchemaPropertyDType.boolean
    
    p = schema_models.SchemaProperty(dtype="float")
    assert p.dtype == schema_models.SchemaPropertyDType.float

    p = schema_models.SchemaProperty(dtype="string")
    assert p.dtype == "string"

    p = schema_models.SchemaProperty(dtype="string", optional=True)
    assert p.dtype == "string"

    try:
        p = schema_models.SchemaProperty(dtype="abc")
        assert False
    except ValueError:
        assert True

def test_define_schema():
    
    schema_models.DatabaseSchema(
        properties={
            "a": schema_models.SchemaProperty(
                dtype=schema_models.SchemaPropertyDType.float
            )
        },
        primitives={
            "x": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean,
                properties=[
                    schema_models.SchemaPropertyReference(
                        property="a"
                    )
                ]
            )
        }
    )

    try:
        schema_models.DatabaseSchema(
            primitives={
                "x": schema_models.SchemaPrimitive(
                    ptype=schema_models.SchemaPrimitiveDtype.boolean,
                    properties=[
                        schema_models.SchemaPropertyReference(
                            property="string"
                        )
                    ]
                )
            }
        )
        assert False
    except ValueError:
        assert True

def test_define_model():

    schema = schema_models.DatabaseSchema(
        primitives={
            "SX": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean
            ),
        },
        composites={
            "SA": schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=schema_models.SchemaLogicOperator.and_,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="SX",
                            quantifier=schema_models.SchemaQuantifier.one_or_more
                        )
                    ]
                )
            )
        }
    )
    
    # Here propositions are defined before referenced
    typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    ptype="SX"
                ),
                "y": typed_models.Primitive(
                    ptype="SX"
                ),
            },
            composites={
                "A": typed_models.Composite(
                    ptype="SA",
                    inputs=["x", "y"]
                )
            }
        )
    )
        


    schema = schema_models.DatabaseSchema(
        primitives={
            "boolean": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean
            ),
            "integer": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean
            ),
        },
        composites={
            "and": schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=schema_models.SchemaLogicOperator.and_,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="boolean",
                            quantifier=schema_models.SchemaQuantifier.zero_or_more
                        ),
                        schema_models.SchemaQuantifiedVariable(
                            variable="integer",
                            quantifier=schema_models.SchemaQuantifier.zero_or_more
                        ),
                    ]
                )
            )
        }
    )

    model1 = typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    ptype="integer"
                ),
                "y": typed_models.Primitive(
                    ptype="boolean"
                ),
            },
            composites={
                "A": typed_models.Composite(
                    ptype="and",
                    inputs=["x", "y"]
                )
            }
        )
    )
    model2 = typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    ptype="boolean"
                ),
                "y": typed_models.Primitive(
                    ptype="boolean"
                ),
            },
            composites={
                "A": typed_models.Composite(
                    ptype="and",
                    inputs=["x", "y"]
                )
            }
        )
    )

    merged_model = model1.merge_data(model2.data)
    assert merged_model == typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    ptype="boolean"
                ),
                "y": typed_models.Primitive(
                    ptype="boolean"
                ),
            },
            composites={
                "A": typed_models.Composite(
                    ptype="and",
                    inputs=["x", "y"]
                )
            }
        )
    )

def test_schema_validation():
    schema = schema_models.DatabaseSchema(
        properties={
            "price": schema_models.SchemaProperty(
                dtype=schema_models.SchemaPropertyDType.integer,
                description="Price unit"
            )
        },
        primitives={
            "week": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaRangeType(lower=2400, upper=2600),
                quantifier=2,
            ),
            "option": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean,
                properties=[
                    schema_models.SchemaPropertyReference(
                        property="price"
                    )
                ]
            )
        },
        composites={
            "weekSpan": schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=schema_models.SchemaLogicOperator.geq,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="week",
                            quantifier=1
                        ),
                        schema_models.SchemaQuantifiedVariable(
                            variable="week",
                            quantifier=1
                        )
                    ]
                )
            ),
            "auth": schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=schema_models.SchemaLogicOperator.imply,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="weekSpan",
                            quantifier=1
                        ),
                        schema_models.SchemaQuantifiedVariable(
                            variable="option",
                            quantifier=1
                        )
                    ]
                ),
                quantifier=schema_models.SchemaQuantifier.one_or_more
            )
        }
    )

    model = typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    id="x",
                    ptype="week"
                ),
                "y": typed_models.Primitive(
                    ptype="week"
                ),
                "z": typed_models.Primitive(
                    ptype="option",
                    properties={
                        "price": 100
                    }
                )
            },
            composites={
                "ws1": typed_models.Composite(
                    ptype="weekSpan",
                    inputs=["x", "y"]
                ),
                "authOptZ": typed_models.Composite(
                    ptype="auth",
                    inputs=["ws1", "z"]
                )
            }
        )
    )

def test_define_composite_binary_relation_must_have_exactly_one_quantifiers():

    for operator in [
        schema_models.SchemaLogicOperator.equiv, 
        schema_models.SchemaLogicOperator.imply, 
        schema_models.SchemaLogicOperator.geq, 
        schema_models.SchemaLogicOperator.leq, 
    ]:
        try:
            schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=operator,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="week",
                            quantifier=1
                        ),
                        # Should fail because it has more than 2 items
                        schema_models.SchemaQuantifiedVariable(
                            variable="week",
                            quantifier=2
                        )
                    ]
                )
            )
            assert False
        except ValueError as e:
            assert True

def test_proposition_definition_order_in_model():
    schema = schema_models.DatabaseSchema(
        primitives={
            "SX": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean
            ),
        },
        composites={
            "SA": schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=schema_models.SchemaLogicOperator.and_,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="SX",
                            quantifier=schema_models.SchemaQuantifier.one_or_more
                        )
                    ]
                )
            )
        }
    )
    
    # Here propositions are defined before referenced
    typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    ptype="SX"
                ),
                "y": typed_models.Primitive(
                    ptype="SX"
                ),
            },
            composites={
                "A": typed_models.Composite(
                    ptype="SA",
                    inputs=["x", "y"]
                )
            }
        )
    )

    # But this one should fail
    # since the id is not defined at all before it is referenced
    try:
        db_model = typed_models.DatabaseModel(
            database_schema=schema,
            data=typed_models.SchemaData(
                primitives={
                    "x": typed_models.Primitive(
                        ptype="SX"
                    ),
                    "y": typed_models.Primitive(
                        ptype="SX"
                    ),
                },
                composites={
                    "A": typed_models.Composite(
                        ptype="SA",
                        inputs=["x", "y", "z"]
                    )
                }
            )
        )
        db_model.validate_all()
        assert False
    except ValueError:
        assert True

def test_schema_properties():

    schema = schema_models.DatabaseSchema(
        properties={
            "a": schema_models.SchemaProperty(
                dtype=schema_models.SchemaPropertyDType.integer,
                description="A property"
            )
        },
        primitives={
            "x": schema_models.SchemaPrimitive(
                ptype=schema_models.SchemaPrimitiveDtype.boolean,
                properties=[
                    schema_models.SchemaPropertyReference(
                        property="a"
                    )
                ]
            )
        },
        composites={
            "A": schema_models.SchemaComposite(
                relation=schema_models.SchemaLogicRelation(
                    operator=schema_models.SchemaLogicOperator.and_,
                    inputs=[
                        schema_models.SchemaQuantifiedVariable(
                            variable="x",
                            quantifier=schema_models.SchemaQuantifier.one_or_more
                        )
                    ]
                ),
                properties=[
                    schema_models.SchemaPropertyReference(
                        property="a"
                    )
                ]
            )
        }
    )

    model = typed_models.DatabaseModel(
        database_schema=schema,
        data=typed_models.SchemaData(
            primitives={
                "x": typed_models.Primitive(
                    id="x",
                    ptype="x",
                    properties={
                        "a": 100
                    }
                )
            },
            composites={
                "myA": typed_models.Composite(
                    ptype="A",
                    inputs=["x"],
                    properties={
                        "a": 110
                    }
                )
            }
        )
    )