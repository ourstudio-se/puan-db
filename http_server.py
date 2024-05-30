import uvicorn
import os
import numpy as np
import lexer
from fastapi import FastAPI
from pldag import Puan, Solution
from puan_db_parser import Parser
from itertools import chain, starmap
from storage import AzureBlobModelHandler, ComputingDevice

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = AzureBlobModelHandler(
    salt="",
    connection_string=os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
    container=os.getenv('AZURE_STORAGE_CONTAINER')
)

def to_edges(model: Puan):
    return list(
        chain(
            starmap(
                lambda r,c: {
                    "source": model._icol(c),
                    "target": model._irow(r),
                },
                np.argwhere(model._amat)
            )
        )
    )

def to_nodes(model: Puan, solution: Solution):
    return list(
        map(
            lambda x: {
                "id": x, 
                "type": "primitive" if x in model.primitives else "composite",
                "label": x[:5], 
                "alias": model.id_to_alias(x),
                "bound": {
                    "lower": int(solution[x].bound.real),
                    "upper": int(solution[x].bound.imag),
                },
                "properties": model.data.get(x, {}),
                "bias": int(model._bvec[model._row(x)].real) if x in model.composites else None,
                "negated": bool(model._nvec[model._row(x)]) if x in model.composites else None,
            }, 
            model._imap
        )
    )

@app.get('/')
def home():
    return "Hello, World!"
@app.get('/api/models/{model_name}')
def get_model(model_name):
    model = handler.load_model(model_name)
    if model is None:
        return Response(content="Model not found", status_code=404)
    
    nodes = to_nodes(model, model.propagate({}))
    edges = to_edges(model)
    
    return {
        "model": model_name,
        "nodes": nodes,
        "edges": edges
    }

@app.get('/api/models')
def get_models():
    return handler.list_blobs()

@app.post('/api/models')
async def create_model(request: Request):
    model_name = await request.json().get("model", None)
    if model_name is None:
        raise HTTPException(status_code=400, detail="Model name was empty")

    existing_models = handler.list_blobs()

    if model_name in existing_models:
        raise HTTPException(status_code=400, detail="Model already exists")

    model = Puan()
    handler.save_model(model, model_name)

    return {
        "model": model_name,
        "nodes": [],
        "edges": []
    }

class QueryModel(BaseModel):
    query: Optional[str] = None
    model: Optional[str] = None

@app.post("/api/lex")
async def lex_query(query_model: QueryModel):
    query = query_model.query
    if query is None:
        raise HTTPException(status_code=400, detail="Query was empty")
    
    try:
        lexed = lexer.lex(query)[0]
        return {
            "lexed": {
                "type": lexed.__class__.__name__,
                "content": lexed
            },
            "error": None
        }
    except Exception as e:
        return {"tokens": [], "error": str(e)}

@app.post("/api/query")
async def post_data(query_model: QueryModel):
    try:
        model_name = query_model.model
        query = query_model.query
        if query is None:
            raise HTTPException(status_code=400, detail="Query was empty")
        
        if model_name is None:
            raise HTTPException(status_code=400, detail="No model set")
        
        comp_dev: ComputingDevice = ComputingDevice(model_name, handler)
        model: Puan = comp_dev.get()

        if not isinstance(model, Puan):
            raise HTTPException(status_code=400, detail="Invalid database model")

        model, solution = Parser(model).evaluate(lexer.lex(query))[-1]
        comp_dev.save(model)

        return {
            "model": model_name,
            "nodes": to_nodes(model, solution),
            "edges": to_edges(model),
            "error": None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('PORT', 8000))