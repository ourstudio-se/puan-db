import os
import numpy as np
import lexer
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from pldag import Puan
from puan_db_parser import Parser
from itertools import chain, starmap
from storage import AzureBlobModelHandler, ComputingDevice

handler = AzureBlobModelHandler(
    salt="",
    connection_string=os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
    container=os.getenv('AZURE_STORAGE_CONTAINER')
)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

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

def to_nodes(model: Puan, solution = None):

    def model_or_sol(model: Puan, solution, x):
        if solution:
            return int(solution[x].bound.real), int(solution[x].bound.imag)
        else:
            return (
                int(model._dvec[model._imap[x]].real),
                int(model._dvec[model._imap[x]].imag),
            )

    return list(
        map(
            lambda x: {
                "id": x, 
                "label": x[:5], 
                "alias": model.id_to_alias(x),
                "bound": dict(zip(["lower", "upper"], model_or_sol(model, solution, x)))
            }, 
            model._imap
        )
    )

@app.route('/')
@cross_origin()
def home():
    return "Hello, World!"

@app.route('/api/models/<model_name>', methods=['GET'])
@cross_origin()
def get_model(model_name):
    model = handler.load_model(model_name)
    if model is None:
        return jsonify({"error": "Model not found"}), 404
    
    nodes = to_nodes(model)
    edges = to_edges(model)
    
    return jsonify({
        "model": model_name,
        "nodes": nodes,
        "edges": edges
    })

@app.route('/api/models', methods=['GET'])
@cross_origin()
def get_models():
    return handler.list_blobs()

@app.route('/api/models', methods=['POST'])
@cross_origin()
def create_model():
    model_name = request.get_json().get("model", None)
    if model_name is None:
        return jsonify({"error": "Model name was empty"}), 400
    
    existing_models = handler.list_blobs()

    if model_name in existing_models:
        return jsonify({"error": "Model already exists"}), 400
    
    model = Puan()
    handler.save_model(model, model_name)

    return jsonify({
        "model": model_name,
        "nodes": [],
        "edges": []
    })

@app.route('/api/lex', methods=['POST'])
def lex_query():
    query = request.get_json().get("query", None)
    if query is None:
        return jsonify({"error": "Query was empty"}), 400
    
    try:
        lexed = lexer.lex(query)[0]
        return jsonify({
            "lexed": {
                "type": lexed.__class__.__name__,
                "content": lexed
            },
            "error": None
        })
    except Exception as e:
        return jsonify({"tokens": [], "error": str(e)}), 200

@app.route('/api/query', methods=['POST'])
@cross_origin()
def post_data():
    try:
        model_name = request.get_json().get("model", None)
        query = request.get_json().get("query", None)
        if query is None:
            return jsonify({"error": "Query was empty"}), 400
        
        if model_name is None:
            return jsonify({"error": "No model set"}), 400
        
        comp_dev : ComputingDevice = ComputingDevice(model_name, handler)
        model : Puan = comp_dev.get()

        if not type(model) == Puan:
            return jsonify({"error": "Invalid database model"}), 400

        model, solution = Parser(model).evaluate(lexer.lex(query))[-1]
        comp_dev.save(model)

        return jsonify({
            "model": model_name,
            "nodes": to_nodes(model, solution),
            "edges": to_edges(model),
            "error": None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)