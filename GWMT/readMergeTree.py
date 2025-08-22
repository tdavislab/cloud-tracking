import json
import os
import networkx as nx
import math
import numpy as np

def read_txt(node_filename, edge_filename, root_type="minimum", region=None, values=None, threshold=None):
    with open(node_filename, "r") as node_file:
        nodes_data = list(node_file.readlines())
    with open(edge_filename, "r") as edge_file:
        edges_data = list(edge_file.readlines())
    if not len(nodes_data) - 1 == len(edges_data):
        print("ERROR: ILLEGAL TREE!", node_filename)
        raise ValueError

    T = nx.Graph()
    min_val = float("inf")
    max_val = -min_val
    root = None

    for e, i in enumerate(nodes_data):
        node = i.split(" ")
        assert len(node) >= 4

        type_value = int(float(node[4].strip())) if len(node) >= 5 else 0
        if (type_value == 2) or (type_value == 4):
            type_value = 1
        elif type_value == 3:
            type_value = 2
        mark_value = int(float(node[5].strip())) if len(node) >= 6 else 1
        T.add_node(e, x=float(node[0]), y=float(node[1]), z=float(node[2]), height=float(node[3]), type=type_value,
                   color="r", mark=mark_value)

    # NOTE: This part needs DEBUGGING!
    if region is not None:
        print("Adding volume to the merge tree")
        
        if values is not None and threshold is not None:
            assert region.shape == values.shape
            region[values < threshold] = -1
        
        unique, counts = np.unique(region, return_counts=True)
        count_dict = dict(zip(unique, counts))
        
        for node in T.nodes():
            nodeX = int(T.nodes[node]["x"])
            nodeY = int(T.nodes[node]["y"])
            node_seg_id = region[nodeX, nodeY]
            if node_seg_id >= 0 and node_seg_id in count_dict:
                T.nodes[node]["volume"] = count_dict[node_seg_id]
            else:
                T.nodes[node]["volume"] = 0

    for i in edges_data:
        vertices = i.split(" ", 1)
        u = int(vertices[0])
        v = int(vertices[1])
        dist = math.fabs(T.nodes[u]['height'] - T.nodes[v]['height'])
        T.add_edge(u, v, weight=dist)
    
    for node in T.nodes():
        node_data = T.nodes[node]
        if float(node_data["height"]) < min_val:
            min_val = float(node_data["height"])    
        if float(node_data["height"]) > max_val:
            max_val = float(node_data["height"])
        
        if root_type == "minimum" and len(T.adj[node]) == 1 and np.isclose(node_data["height"], min_val):
            root = node
        elif root_type != "minimum" and len(T.adj[node]) == 1 and np.isclose(node_data["height"], max_val):
            root = node

    assert root is not None
    assert (T.nodes[root]["height"] == min_val and root_type == "minimum") or (T.nodes[root]["height"] == max_val and root_type == "maximum")

    T.root = root
    return T, root


def get_trees(*filenames, root_type="minimum", threshold=None):
    file_ids = []
    Ts = []
    roots = []

    for filename in filenames:
        if filename.find(".txt") != -1:
            if filename.find("Node") != -1:
                edge_filename = filename.replace("Node", "Edge")
                region_filename = filename.replace("treeNodes", "segmentation").replace(".txt", ".npy")
                value_filename = filename.replace("treeNodes_", "").replace(".txt", ".npy")
                assert os.path.exists(edge_filename)
                assert os.path.exists(region_filename)
                assert os.path.exists(value_filename)
                values = np.load(value_filename)
                region = np.load(region_filename)
                T, root = read_txt(filename, edge_filename, root_type=root_type, region=region, values=values, threshold=threshold)
                Ts.append(T)
                roots.append(root)
                file_ids.append(filename.split(".")[0].split("_")[-1])
    return Ts, roots


def get_regions(*filenames):
    regions = []
    values = []
    for filename in filenames:
        if filename.find(".txt") != -1:
            if filename.find("Node") != -1:
                region_filename = filename.replace("treeNodes", "segmentation").replace(".txt", ".npy")
                value_filename = filename.replace("treeNodes_", "").replace(".txt", ".npy")

                value = np.load(value_filename)
                region = np.load(region_filename)
                
                regions.append(region)
                values.append(value)
    return regions, values
                
