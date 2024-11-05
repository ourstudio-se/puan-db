from api.models.schema import Schema
from api.models.typed_model import Model
from itertools import starmap

schema = Schema(**{
    "primitives": {
        "a": {
            "dtype": "boolean",
        }
    },
    "composites": {
        "A": {
            "relation": {
                "operator": "and",
                "items": [
                    {
                        "dtype": "a",
                        "quantifier": "+"
                    }
                ]
            }
        }
    }
})

model = Model(
    model_schema=schema, 
    **{
        "data": {
            "primitives": {
                "mya0": {
                    "dtype": "a"
                },
                "mya1": {
                    "dtype": "a"
                },
                "mya2": {
                    "dtype": "a"
                },
            },
            "composites": dict(
                map(
                    lambda x: (
                        str(x), {
                        "dtype": "A",
                        "arguments": [
                            "mya0",
                            "mya1",
                            "mya2"
                        ]
                    }),
                    range(1000)
                )
            )
        }
    }
)

from api.tools import timer
with timer("to_pldag"):
    model.to_pldag()