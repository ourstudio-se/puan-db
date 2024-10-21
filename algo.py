# 0 1 2 3
# 1 4 7 10

from pldag import PLDAG, CompilationSetting
from itertools import starmap, chain, groupby
import networkx as nx

def under_node(node) -> str:
    return f"Ö_{node}"

def upper_node(node) -> str:
    return f"0_{node}"

# Load model from file
# model = PLDAG.from_file("db/256")
model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
model.set_primitives("xyza")
model.set_or([
    model.set_imply(
        model.set_and(["x", "y"]),
        model.set_or(["z", "x"])
    ),
    "a"
])
model.compile()

# Create a dictionary of nodes and their dependencies
node_deps = dict(zip(model.columns, map(model.dependencies, model.columns)))

reversed_deps = {node: set() for node in model.columns}
# Reverse the dependencies to get the dependents of each node
for node, deps in node_deps.items():
    for dep in deps:
        reversed_deps[dep].add(node)

# Create a topological sort and track the depth of each node
G = nx.DiGraph(reversed_deps)

# Perform a topological sort
topo_order = list(nx.topological_sort(G))

# Calculate the depth of each node (longest path to a leaf)
depths = {node: 0 for node in G.nodes}
for node in topo_order:
    for successor in G.successors(node):
        depths[successor] = max(depths[successor], depths[node] + 1)

# Create aux nodes
# First make room for new depths. Depths 0 1 2 3 4 becomes 1 4 7 10 13 -> [1, 3, 5, 7, 9]
depths = {node: 1 + depth*3 for node, depth in depths.items()}

# For each node, create two aux nodes (which later will be placed directly over and directly under the node)
depths_aux = {**depths, **dict(
    chain(
        *starmap(
            lambda node, i: [
                (under_node(node), i-1), 
                (upper_node(node), i+1)
            ], 
            depths.items()
        )
    )
)}

X_OFFSET = 600
Y_OFFSET = 100
NODE_HORIZONTAL_SPACING = 50
NODE_VERTICAL_SPACING = 50

# Set nodes init positions
nodes = {}
for i in range(max(depths_aux.values()) + 1):
    nds = sorted([node for node, depth in depths_aux.items() if depth == i])
    for j, node in enumerate(nds):
        nodes[node] = {
            "id": node,
            "x": j * NODE_HORIZONTAL_SPACING + X_OFFSET, 
            "y": i * NODE_VERTICAL_SPACING + Y_OFFSET,
            "depth": i,
            "aux": not node in depths,
            "primitive": node in model.primitives or node[2:] in model.primitives,
        }


# New edges to connect aux nodes
edges = list(
    chain(
        *map(
            lambda node: [
                {
                    "source": {"x": nodes[under_node(node)]['x'], "y": nodes[under_node(node)]['y'], "id": under_node(node)}, 
                    "target": {"x": nodes[node]['x'], "y": nodes[node]['y'], "id": node},
                },
                {
                    "source": {"x": nodes[node]['x'], "y": nodes[node]['y'], "id": node}, 
                    "target": {"x": nodes[upper_node(node)]['x'], "y": nodes[upper_node(node)]['y'], "id": upper_node(node)},
                }
            ],
            filter(
                lambda x: x in depths, 
                depths_aux.keys()
            )
        ),
        *starmap(
            lambda parent, children: map(
                lambda child: {
                    "source": {"x": nodes[upper_node(child)]['x'], "y": nodes[upper_node(child)]['y'], "id": upper_node(child)},
                    "target": {"x": nodes[under_node(parent)]['x'], "y": nodes[under_node(parent)]['y'], "id": under_node(parent)},
                },
                children
            ),
            node_deps.items()
        )
    )
)

graph = {
    "nodes": list(nodes.values()),
    "edges": edges,
}

import json
json.dump(graph, open("graph.json", "w"), indent=2)