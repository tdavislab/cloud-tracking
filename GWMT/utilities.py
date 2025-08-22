import os
import math
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import networkx as nx
from weighted_hierarchy_pos import *
from os.path import join as pjoin
import pandas as pd
import scipy.ndimage
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def scale_normalization(hlist, pdist, region_scale, is_type_last=False):
    dimension = len(hlist[0][0])
    region_dimension = dimension - 1 if is_type_last else dimension
    scale = np.zeros((dimension + 1, ))
    N = len(hlist)
    assert len(hlist) == N

    scale[:region_dimension] = region_scale
    if region_dimension < dimension:
        scale[region_dimension:dimension] = 1.0

    assert len(pdist) == N
    max_pdist = None
    min_pdist = None
    for i in range(N):
        if max_pdist is None:
            max_pdist = np.max(pdist[i])
        else:
            max_pdist = max(max_pdist, np.max(pdist[i]))
        if min_pdist is None:
            min_pdist = np.min(pdist[i])
        else:
            min_pdist = min(min_pdist, np.min(pdist[i]))

    scale[-1] = max_pdist - min_pdist
    print("Scale:", scale)
    return scale


def djs_find(f, x):
    if f[x] != x:
        f[x] = djs_find(f, f[x])
    return f[x]


def coordinate_matrix(g, labels=None):
    if (labels is None) or len(labels) < 1:
        print("Warning! No label name for node labels is provided!")
        return None
    tmp = np.zeros((g.number_of_nodes(), len(labels)))
    for e, n in enumerate(g.nodes()):
        node = g.nodes[n]
        for ee, label in enumerate(labels):
            tmp[e][ee] = node[label]
    return tmp


class GraphColorMemory:
    max_idx = 0


def graph_color(T, T_ref=None, oc=None, maxima_only=False):
    # if there is no reference tree, this is the first tree
    # we assign an id to each point based on a certain order. Here we use the key order
    if T_ref is None:
        GraphColorMemory.max_idx = 0
        k = T.nodes.keys()
        pos = []
        for i in k:
            pos.append(i)

        pos.sort()
        order = dict()
        cnt = 0
        for i in range(len(pos)):
            if pos[i] not in order:
                order[pos[i]] = cnt
                cnt += 1

        diff_color = len(order)

        if GraphColorMemory.max_idx < diff_color:
            GraphColorMemory.max_idx = diff_color

        for i in k:
            assert i in order
            color_idx = order[i]
            if maxima_only and T.nodes[i]["type"] != 2:
                continue
            T.nodes[i]['color_value'] = color_idx
    else:
        # there is a reference tree. It should be the tree at the previous adjacent time step
        assert oc is not None

        N, M = oc.shape
        match_axis0 = np.argmax(oc, axis=0)  # len(match_axis0) == M
        match_axis1 = np.argmax(oc, axis=1)  # len(match_axis1) == N

        if not maxima_only:
            assert N == len(T_ref.nodes.keys())
            assert M == len(T.nodes.keys())

            for j in range(M):
                x = match_axis0[j]
                y = match_axis1[x]
                if j == y and T.nodes[j]["type"] == T_ref.nodes[x]["type"] and (oc[x][y] > 0):
                    T.nodes[j]["color_value"] = T_ref.nodes[x]["color_value"]
                else:
                    T.nodes[j]["color_value"] = GraphColorMemory.max_idx
                    GraphColorMemory.max_idx += 1
        else:
            maxima_ids_M = np.asarray([node for node in T.nodes() if T.nodes[node]["type"] == 2], dtype=int)
            maxima_ids_N = np.asarray([node for node in T_ref.nodes() if T_ref.nodes[node]["type"] == 2], dtype=int)
            assert N == len(maxima_ids_N)
            assert M == len(maxima_ids_M)
            
            for ej, j in enumerate(maxima_ids_M):
                ex = match_axis0[ej]
                ey = match_axis1[ex]
                if ej == ey and oc[ex][ey] > 0:
                    x = maxima_ids_N[ex]
                    T.nodes[j]["color_value"] = T_ref.nodes[x]["color_value"]
                else:
                    T.nodes[j]["color_value"] = GraphColorMemory.max_idx
                    GraphColorMemory.max_idx += 1
    return T


def get_dist2parent_distribution(T: nx.Graph, root: int, scalar_name: str):
    dist2parent = {}
    li = list()
    node = root
    dist2parent[node] = 0
    li.append(node)
    heights = []
    while len(li) > 0:
        node = li.pop(-1)
        heights.append(T.nodes[node][scalar_name])
        for u, vs in T.adjacency():
            if u != node:
                continue
            for v, e in vs.items():
                if v not in dist2parent:
                    dist2parent[v] = e['weight']
                    li.append(v)

    dists = np.asarray(list(dist2parent.values()), dtype=float)
    dist2parent[root] = np.max(heights) - np.min(heights)
    prob_weight = [dist2parent[key] for key in sorted(dist2parent.keys())]
    prob_weight /= np.sum(prob_weight)
    return np.asarray(prob_weight, dtype=float)


def get_volume_distribution(T: nx.Graph, vol_name: str):
    vols = {}
    for node in T.nodes():
        assert vol_name in T.nodes[node]
        vols[node] = T.nodes[node][vol_name]
    
    tot_vols = sum(vols.values())
    prob = np.array([vols[node] for node in T.nodes()], dtype=float)
    prob /= tot_vols
    return prob
    
    
def get_distance_and_distribution(tree, distribution="uniform", weight_mode="shortestpath", **params):
    """
    Required field for the strategy choice in **params
    ---------------------------------------------------
    distribution="uniform": None

    distribution="ancestor": params["root"], int, the id of the root node in the tree
                             params["scalar_name"], str, the name for the scalar function of nodes in the tree
                             
    distribution="volume": params["volume_name"], str, the name of volumes belong to the 

    weight_mode="shortestpath": params["edge_weight_name"], str, the name for the edge weight in the tree

    weight_mode="lca": params["root"], int, the id of the root node in the tree
                       params["scalar_name"], str, the name for the scalar function of nodes in the tree
    """
    num_of_nodes = tree.number_of_nodes()
    if distribution == "uniform":
        p = np.ones((num_of_nodes,)) / num_of_nodes
    elif distribution == "ancestor":
        assert "root" in params
        assert "scalar_name" in params
        p = get_dist2parent_distribution(tree, params["root"], params["scalar_name"])
    elif distribution == "volume":
        assert "volume_name" in params
        p = get_volume_distribution(tree, params["volume_name"])
    else:
        p = np.ones((num_of_nodes,)) / num_of_nodes

    if weight_mode == "shortestpath":
        assert "edge_weight_name" in params
        weight_str = params["edge_weight_name"]
    elif weight_mode == "lca":
        assert "scalar_name" in params
        assert "root" in params
        C = np.zeros((num_of_nodes, num_of_nodes))
        lca_matrix = lca(tree, params["root"])
        for node_a in tree.nodes:
            for node_b in tree.nodes:
                lca_node = lca_matrix[node_a, node_b]
                C[node_a][node_b] = tree.nodes[lca_node][params["scalar_name"]]
        return C, p
    elif weight_mode == "lca-threshold":
        assert "scalar_name" in params
        assert "root" in params
        assert "scalar_threshold" in params
        assert params["scalar_threshold"] is not None
        C = np.zeros((num_of_nodes, num_of_nodes))
        lca_matrix = lca(tree, params["root"])
        for node_a in tree.nodes:
            for node_b in tree.nodes:
                lca_node = lca_matrix[node_a, node_b]
                lca_val = tree.nodes[lca_node][params["scalar_name"]]
                if lca_val >= params["scalar_threshold"]:
                    C[node_a][node_b] = 1
                else:
                    C[node_a][node_b] = 0
        return C, p
    else:
        assert "edge_weight_name" in params
        weight_str = params["edge_weight_name"]

    D = list(nx.all_pairs_dijkstra_path_length(tree, weight=weight_str))
    C = np.zeros((num_of_nodes, num_of_nodes))
    for ii in range(num_of_nodes):
        dist_zip = zip(*sorted(D[ii][1].items()))
        dist = list(dist_zip)
        C[ii, :] = dist[1]
    return C, p


def lca(T, root):
    num = T.number_of_nodes()
    lca_mat = np.zeros((num, num), dtype=int) - 1
    ff = {}
    col = {}
    ancestor = {}
    for node in T.nodes():
        ff[node] = node
        col[node] = False
        ancestor[node] = node

    TarjanOLCA(T, root, None, ff, col, ancestor, lca_mat)
    # print(lca_mat)
    # exit()
    return lca_mat


def TarjanOLCA(T, u, parent, ff, col, ancestor, lca_mat):
    for neighbor in T.neighbors(u):
        if parent is not None and neighbor == parent:
            continue
        TarjanOLCA(T, neighbor, u, ff, col, ancestor, lca_mat)
        fu = djs_find(ff, u)
        fv = djs_find(ff, neighbor)
        if fu != fv:
            ff[fv] = fu
        fu = djs_find(ff, u)
        ancestor[fu] = u

    col[u] = True
    for node in T.nodes():
        if col[node] and lca_mat[u, node] < 0:
            fv = djs_find(ff, node)
            lca_mat[u, node] = lca_mat[node, u] = ancestor[fv]


def label_distance(a, b, metric="l2"):
    if metric == "l2" or (metric == "penaltyl2" and len(a) <= 3 and len(b) <= 3):
        assert len(a) == len(b)
        return np.linalg.norm(a-b)
    elif metric == "penaltyl2":
        assert len(a) == len(b)
        penalty = 1 if a[-1] != b[-1] else 0
        return np.linalg.norm(a[:-1] - b[:-1]) + penalty
    else:
        raise NotImplementedError
    
    
def get_leaves(subtree_leaves, T: nx.Graph, parents, node, parent):
    if node in subtree_leaves:
        return subtree_leaves
    
    parents[node] = parent
    
    children = []
    for nb in T.neighbors(node):
        if nb != parent:
            children.append(nb)
    if len(children) == 0:
        subtree_leaves[node] = [(node, node)]
        return [(node, node)]
    
    leaves = []
    for child in children:
        sub_leaves = get_leaves(subtree_leaves, T, parents, child, node)
        for sub_child, sub_leaf in sub_leaves:
            leaves.append((child, sub_leaf))
    
    subtree_leaves[node] = leaves
    return leaves
    

# Each non-leaf node pairs with the deepest leaf node in all of its subtrees
# Priority: from top to down
def mt_2_pd(pd_lst, occupied, subtree_leaves, T: nx.Graph, node, parent, maxima=True, val_name="height"):
    children = []
    for nb in T.neighbors(node):
        if nb != parent:
            children.append(nb)
    
    if len(children) == 0:
        return pd_lst
    
    leaves = subtree_leaves[node]
    leaves.sort(key=lambda x: T.nodes[x[1]][val_name], reverse=maxima)
    avoid_path = set()
    for child, leaf in leaves:
        if occupied[leaf]:
            avoid_path.add(child)
    
    found_leaves = 0
    for child, leaf in leaves:
        if child in avoid_path:
            continue
        occupied[leaf] = True
        avoid_path.add(child)
        pd_lst.append((node, leaf))
        found_leaves += 1
        if found_leaves >= len(children) - 1:
            break
    
    for child in children:
        pd_lst = mt_2_pd(pd_lst, occupied, subtree_leaves, T, child, node, maxima, val_name)
    return pd_lst


# Decompose a merge tree into a list of branches
# "height" indicates the function value
def pd_decomposition_mt(T: nx.Graph, root=None, maxima=True, val_name="height"):
    if root is None and "root" in T:
        root = T.root
    if root is None:
        extrema = None
        for node in T.nodes():
            if extrema is None or (maxima and T.nodes[node][val_name] < extrema) or (not maxima and T.nodes[node][val_name] > extrema):
                extrema = T.nodes[node][val_name]
                root = node
    
    # now, for each saddle, look for all subtrees and get their highest maxima
    subtree_leaves = {}
    parents = {}
    get_leaves(subtree_leaves, T, parents, root, -1)

    pd_lst = []
    occupied = [False for node in T.nodes()]
    pd_lst = mt_2_pd(pd_lst, occupied, subtree_leaves, T, root, -1, maxima, val_name)
    return pd_lst


def complete_dependent_vol(T: nx.Graph, dependent_vol, subtree_nodes, node, parent, vol_name):
    if node in dependent_vol:
        return

    children = []
    for nb in T.neighbors(node):
        if nb != parent:
            children.append(nb)
    
    assert len(children) > 0
    vol = T.nodes[node][vol_name]
    for child in children:
        complete_dependent_vol(T, dependent_vol, subtree_nodes, child, node, vol_name)
        vol += dependent_vol[child]
        subtree_nodes[node].extend(subtree_nodes[child])
    
    dependent_vol[node] = vol
    return


def mt_dependent_volume(T: nx.Graph, parents, vol_name="volume"):
    # now, for each saddle, look for all subtrees and get their maxima with the highest volume
    subtree_leaves = {}
    get_leaves(subtree_leaves, T, parents, T.root, -1)
    leaves = set()
    for _, leaf in subtree_leaves[T.root]:
        leaves.add(leaf)
    
    dependent_vol = {}
    subtree_nodes = {}
    
    for leaf in leaves:
        dependent_vol[leaf] = T.nodes[leaf][vol_name]

    for node in T.nodes():
        subtree_nodes[node] = [node]
    
    complete_dependent_vol(T, dependent_vol, subtree_nodes, T.root, -1, vol_name)
    return dependent_vol, subtree_nodes, subtree_leaves


# Find the best leaf node in the subtree of "node" based on volume
# Note: in this function, we only search for the best leaf node in the subtree.
#       We will label the path to the leaf node as occupied in a different function
# Returns - best_extrema: the best leaf node ID 
#         - best_chain: a list of nodes on the paths from the best leaf node to the ancestor
def vol_best_extrema(T: nx.Graph, dependent_vol, child_in_branch, node, parent):
    children = []
    for nb in T.neighbors(node):
        if nb != parent:
            children.append(nb)
    if len(children) < 1:
        return node, [node]
    
    best_vol = 0
    best_extrema = None
    best_chain = []
    for child in children:
        # we avoid re-selecting child nodes that already belong to other branches
        if child not in child_in_branch:
            if dependent_vol[child] > best_vol:
                best_vol = dependent_vol[child]
                best_extrema, best_chain = vol_best_extrema(T, dependent_vol, child_in_branch, child, node)
                
    if len(best_chain) == 0:
        return None, []
    best_chain.append(node)
    return best_extrema, best_chain    
    

# Merge tree branch decomposition based on volume
# Params: T (input) - the merge tree
#         dependent_vol (input) - a list of dependent volumes of nodes in the merge tree. 
#         child_selected (input/output) - a set to label all nodes that have been assigned to branches
#         vol_pd_lst (output) - a list of (ancestor_node, leaf_node, volume, leaf_value) for branches
#         first_branch_idx (output) - a dict of the first branch containing the node
# No Returns
def vol_decomposition_mt(T: nx.Graph, dependent_vol, child_in_branch,  vol_pd_lst, vol_pd_chain, first_branch_idx, node, parent, val_name="height"):
    children = []
    for nb in T.neighbors(node):
        if nb != parent:
            children.append(nb)
    if len(children) < 1:
        return # return dependent_vol[node], node
    
    while True:
        best_extrema, best_chain = vol_best_extrema(T, dependent_vol, child_in_branch, node, parent)
        if best_extrema is None:
            break
        
        # best_chain guarantees to have at least two items: leaf node and the ancestor node
        assert len(best_chain) >= 2
        # content in vol_pd_lst: (lower, upper, volume, upper_value)
        vol_pd_lst.append((node, best_extrema, dependent_vol[best_chain[-2]], T.nodes[best_extrema][val_name]))
        vol_pd_chain.append(best_chain)
        if node not in first_branch_idx:
            first_branch_idx[node] = len(vol_pd_lst) - 1
        
        # clearing up the path
        for chain_node in best_chain:
            if chain_node not in child_in_branch:
                child_in_branch[chain_node] = len(vol_pd_lst) - 1
        
    for child in children:
        vol_decomposition_mt(T, dependent_vol, child_in_branch, vol_pd_lst, vol_pd_chain, first_branch_idx, child, node, val_name)

    
# Compute the branch hierarchy
# Params: T (input) - the merge tree
#         vol_pd_lst (input) -  a list of (ancestor_node, leaf_node, volume, leaf_value) for branches
#         subtree_leaves (input) - a dict of lists of leaf nodes for all nodes
#         first_branch_idx (input) - a dict of the first branch containing the node
def find_parent_branches(T: nx.Graph, vol_pd_lst, subtree_leaves, first_branch_idx, pbranches, parents, node, parent):
    children = []
    for nb in T.neighbors(node):
        if nb != parent:
            children.append(nb)
    
    if len(children) < 1:
        return
    
    if node not in first_branch_idx:
        print("Warning! Cannot find the branch rooting at", node)
        return
    inc = first_branch_idx[node]
    
    while inc < len(vol_pd_lst) and vol_pd_lst[inc][0] == node:
        cur_parent = parents[node]
        while cur_parent != -1:
            assert cur_parent in first_branch_idx
            parent_inc = first_branch_idx[cur_parent]
            found = False
            while parent_inc < len(vol_pd_lst) and vol_pd_lst[parent_inc][0] == cur_parent:
                upper_parent_branch = vol_pd_lst[parent_inc]
                if upper_parent_branch in subtree_leaves[node]:
                    found = True
                    pbranches[inc] = parent_inc
                    break
                parent_inc += 1
            if found:
                break
            cur_parent = parents[cur_parent]
            
        inc += 1
        
    for child in children:
        find_parent_branches(T, vol_pd_lst, subtree_leaves, first_branch_idx, pbranches, parents, child, node)
    

# Adding nodes on a branch in the original tree to the simplified merge tree 
# Params: T_sim (input/output) - simplified merge tree
#         T (input) - original merge tree
#         subtree_leaves (input) - a dict of leaf nodes in the subtree of each node
#         parents (input) - a dict of parent node of each node
#         new_idx (input/output) - a dict recording the node ID of the original tree node in the simplified merge tree
def add_branch_nodes(T_sim : nx.Graph, T, subtree_leaves, parents, new_idx, node, leaf):
    if node not in new_idx:
        node_idx = node # T_sim.number_of_nodes()
        T_sim.add_node(node_idx, **T.nodes(data=True)[node])
        T_sim.nodes[node_idx]['old_idx'] = node
        # print("ADD T_{} to T_sim_{}".format(str(node), str(node_idx)))
        if node == T.root:
            T_sim.root = node_idx
        new_idx[node] = node_idx
        
    children = []
    for nb in T.neighbors(node):
        if nb != parents[node]:
            children.append(nb)
    
    if len(children) < 1:
        return
    
    for child in children:
        if (child, leaf) in subtree_leaves[node]:
            add_branch_nodes(T_sim, T, subtree_leaves, parents, new_idx, child, leaf)
            if new_idx[node] not in T_sim.adj[new_idx[child]]:
                T_sim.add_edge(new_idx[node], new_idx[child], weight=abs(T.nodes[node]["height"] - T.nodes[child]["height"]))
                
                
# Rename the ID of nodes to make them contiguous
def rename_node(T: nx.Graph, vol_mapping=None):
    T_sim = nx.Graph()
    k = 0
    corr = {}
    for node in T.nodes():
        corr[node] = k
        k += 1
        
    if vol_mapping is not None:
        for key in vol_mapping:
            assert vol_mapping[key] in corr
            vol_mapping[key] = corr[vol_mapping[key]]
    
    for node in T.nodes():
        T_sim.add_node(corr[node], **T.nodes(data=True)[node])
    for u, v in T.edges():
        T_sim.add_edge(corr[u], corr[v], weight=T.edges[u, v]["weight"])
    
    if hasattr(T, "root"):
        T_sim.root = corr[T.root]
    return T_sim, vol_mapping


# DEPRECATED. We compute the node volume at the very end of simplification instead
# Computing the simplified node volume is based on the volume mapping
# def compute_simplified_node_vol(T_sim: nx.Graph, T: nx.Graph, parent, node, vol_mapping, vol_name="volume"):
#     children = []
#     for nb in T_sim.neighbors(node):
#         if nb != parent:
#             children.append(nb)
            
#     # dp_vol = T_sim.nodes[node]["dependent_vol"]
#     # for child in children:
#     #     dp_vol -= T_sim.nodes[child]["dependent_vol"]
#     for n in vol_mapping:
#         if vol_mapping[n] == T_sim.nodes[node]["old_idx"]:
#             T_sim.nodes[node][vol_name] += T.nodes[n][vol_name]
            
#     for child in children:
#         compute_simplified_node_vol(T_sim, T, node, child, vol_mapping, vol_name)


# The master function to compute the volume-based simplification of merge trees.
# Step 1. Look for all leafs 
# Step 2. Iteratively pruning from leafs to the root. 
#         Every time two leaves hit on a saddle, compute the volume of the subtree and see if it should be pruned
# Step 3. Build up the simplified merge tree
# NOTE: We ignore the volume mapping for the simplified merge tree in this pipeline
def volume_simplify_mt(T: nx.Graph, vol_thres, disappear_vol_thres, vol_name="volume", val_name="height", stop_saddle_val=2.0):
    assert nx.is_tree(T)
    assert hasattr(T, "root")
    # leaves = set([node for node in T.nodes() if len(T.adj[node]) == 1 and node != T.root])
    leaves = set([node for node in T.nodes() if (T.nodes[node]["type"] == 2) and node != T.root])
    
    removed_nodes = set()
    for node in T.nodes():
        T.nodes[node]["dependent_vol"] = T.nodes[node][vol_name]
    
    # NOTE: This is a parallel process to the simplification step
    # The intention is to compute the branch decomposition based on volume,
    # which is necessary to have when we compute the volume assigned to each local maxima
    # def vol_branch_decomposition():    
    #     # we compute the dependent volume of each critical point (especially for saddles)
    #     # dependent volume: the sum volume of all nodes in the subtree
    #     parents = {}
    #     dependent_vol, subtree_nodes, subtree_leaves = mt_dependent_volume(T, parents, vol_name)
    #     assert parents[T.root] == -1
        
    #     # secondly, we compute the branch decomposition based on volume
    #     vol_pd_lst = []
    #     vol_pd_chain = []
    #     child_in_branch = {}
    #     first_branch_idx = {}
        
    #     # NOTE: the vol_pd_lst comes in a order by the volume/area, from the highest to the lowest
    #     vol_decomposition_mt(T, dependent_vol, child_in_branch, vol_pd_lst, vol_pd_chain, first_branch_idx, T.root, -1, val_name)
    #     # print("Branch Info:", vol_pd_lst)
    #     # print("Seniormost Branch ID of nodes:", child_in_branch)
    #     # print("Branch chain:", vol_pd_chain)
    #     # print("Volume-based Branch Decomposition Done.")
    #     return dependent_vol, vol_pd_lst, subtree_leaves, vol_pd_chain
    
    # full_dependent_vol, vol_pd_lst, subtree_leaves, vol_pd_chain = vol_branch_decomposition()
    
    # we only process the set of nodes to be removed
    current = set()
    for node in leaves:
        # if the volume of the leaf is too small, and its branch saddle is above the contour line
        if (T.nodes[node][vol_name] < disappear_vol_thres) or (T.nodes[node][val_name] < stop_saddle_val) or \
          (T.nodes[node][vol_name] < vol_thres and T.nodes[list(T.adj[node].keys())[0]][val_name] >= stop_saddle_val):
            # represent names the representative maxima of the subtree (the one with the highest volume)
            # this is very useful if the entire subtree should be pruned (volume is too small) but cannot (saddle_val < 2.0)
            T.nodes[node]["represent"] = node
            
            removed_nodes.add(node)
            current.add(node)
    print("Initially removing {} leaves.".format(str(len(removed_nodes))))
        
    checked_nodes = copy.deepcopy(leaves)
    while len(current) > 0:
        # for all current nodes that await to be pushed up
        # we find their parent nodes
        parents = []
        for node in current:
            for u in T.adj[node]:
                if u not in checked_nodes:
                    parents.append((u, node))
                    
        # sort by parent nodes
        parents.sort()
        
        # iterate through all parents to be processed
        this = 0
        parents_processed = False
        while True:
            that = this + 1
            # find all the entries of the parent
            while that < len(parents) and parents[that][0] == parents[this][0]:
                that += 1
            if that > this:
                parent = parents[this][0]
                # we only push up to the parents if all children are ready
                if len(T.adj[parent]) == that - this + 1:
                    assert that - this >= 2
                    parents_processed = True
                    max_dependent_u = None
                    max_dependent = None
                    for i in range(this, that):
                        u = parents[i][1]
                        T.nodes[parent]["dependent_vol"] += T.nodes[u]["dependent_vol"]
                        T.nodes[parent]["mark"] += T.nodes[u]["mark"]
                        if max_dependent is None or T.nodes[u]["dependent_vol"] > max_dependent:
                            max_dependent = T.nodes[u]["dependent_vol"]
                            max_dependent_u = T.nodes[u]["represent"]
                        current.remove(u)
                    T.nodes[parent]["represent"] = max_dependent_u
                    ancestor = [node for node in T.adj[parent] if node not in checked_nodes]
                    assert len(ancestor) == 1
                    if T.nodes[parent]["dependent_vol"] < disappear_vol_thres or (T.nodes[parent]["dependent_vol"] < vol_thres and T.nodes[ancestor[0]][val_name] >= stop_saddle_val):
                        removed_nodes.add(parent)
                        current.add(parent)
                    checked_nodes.add(parent)
            this = that
            if this >= len(parents):
                break
        
        if not parents_processed:
            break
    
    # Now, let's construct a simplified merge tree from the root
    # We build up the merge tree with all nodes that are not in removed_nodes
    # NOTE: This simplified merge tree may contain nodes with degree at 2
    # We will simplify it in a later step
    T_sim = nx.Graph()
    que = [(T.root, 0)]
    ptr = 0
    visited = set()
    T_sim.add_node(0, **T.nodes(data=True)[T.root])
    T_sim.root = 0
    sim_parents = {}
    # the parent node of root is -1
    sim_parents[0] = -1
    
    node_cnt = 1
    while ptr < len(que):
        node, new_tree_idx = que[ptr]
        visited.add(node)
        for next in T.adj[node]:
            children = []
            if next not in visited and next not in removed_nodes:
                children.append(next)
            else:
                visited.add(next)
            for child in children:
                # node_cnt becomes the new id for the child node
                if "represent" in T.nodes[child]:
                    T_sim.add_node(node_cnt, **T.nodes(data=True)[T.nodes[child]["represent"]])
                else:
                    T_sim.add_node(node_cnt, **T.nodes(data=True)[child])
                    que.append((child, node_cnt))
                
                sim_parents[node_cnt] = new_tree_idx
                T_sim.add_edge(new_tree_idx, node_cnt, weight=abs(T.nodes[node][val_name] - T_sim.nodes[node_cnt][val_name]))
                node_cnt += 1
        ptr += 1
    
    # for saddle_end, leaf_end, vol, leaf_val in vol_pd_lst:
    #     tot_vol = vol
    #     cur_node = saddle_end
        
    #     pass
      
    # Then, we simplify the nodes with degree at 2
    # The way we do this is by path compression
    real_parents = {0: -1}
    que = [0]
    ptr = 0
    while ptr < len(que):
        new_node = que[ptr]
        pp = sim_parents[new_node]
        if pp != -1 and len(T_sim.adj[pp]) == 2:
            real_parents[new_node] = real_parents[pp]
        else:
            real_parents[new_node] = pp
        for next in T_sim.adj[new_node]:
            if next != sim_parents[new_node]:
                que.append(next)
        ptr += 1
    
    T_sim2 = nx.Graph()
    T_sim2.root = 0
    for node in T_sim:
        if len(T_sim.adj[node]) != 2:
            T_sim2.add_node(node, **T_sim.nodes(data=True)[node])
            
    for each in real_parents:
        parent = real_parents[each]
        if len(T_sim.adj[each]) == 2:
            continue
        if parent != -1 and len(T_sim.adj[parent]) == 2:
            continue
        if parent != -1:
            T_sim2.add_edge(each, parent, weight=abs(T_sim.nodes[each][val_name] - T_sim.nodes[parent][val_name]))
    
    T_sim2, _ = rename_node(T_sim2)
    return T_sim, T_sim2


# The master function to compute the volume-based simplification of merge trees.
# Step 1. Compute the volume-based branch decomposition of merge trees
# Step 2. Record branches to be kept in the simplified tree
# Step 3. Build a new tree based on all recorded branches in Step 2
def volume_simplify_mt_dfs(T: nx.Graph, vol_thres, vol_name="volume", val_name="height", stop_saddle_val=2.0):
    assert nx.is_tree(T)
    assert hasattr(T, "root")
    
    # we compute the dependent volume of each critical point (especially for saddles)
    # dependent volume: the sum volume of all nodes in the subtree
    parents = {}
    dependent_vol, subtree_nodes, subtree_leaves = mt_dependent_volume(T, parents, vol_name)
    assert parents[T.root] == -1
    
    # print("Compute Dependent Volume Done.")
    
    # We can assign the dependent volume of each node to the node attribute
    # which helps us compute the node volume in the simplified tree
    for node in T.nodes():
        assert node in dependent_vol
        T.nodes[node]["dependent_vol"] = dependent_vol[node]
    
    # secondly, we compute the branch decomposition based on volume
    vol_pd_lst = []
    vol_pd_chain = []
    child_in_branch = {}
    first_branch_idx = {}
    
    # NOTE: the vol_pd_lst comes in a order by the volume/area, from the highest to the lowest
    vol_decomposition_mt(T, dependent_vol, child_in_branch, vol_pd_lst, vol_pd_chain, first_branch_idx, T.root, -1, val_name)
    # print("Branch Info:", vol_pd_lst)
    # print("Seniormost Branch ID of nodes:", child_in_branch)
    # print("Branch chain:", vol_pd_chain)
    # print("Volume-based Branch Decomposition Done.")
    
    vol_mapping = {}
    if vol_thres < 1:
        return T, vol_mapping
    
    T_sim = nx.Graph()
    
    # we set up the cascading relations between branches
    # parent_branches = [-1] * len(vol_pd_lst)
    # find_parent_branches(T, vol_pd_lst, subtree_leaves, first_branch_idx, parent_branches, parents, T.root, -1)
    # print("Parent branches:", parent_branches)
    
    # now, let's start to simplify the merge tree
    # we only keep all branches whose dependent volume >= vol_thres
    kept_branches = []
    kept_branch_idx = set()
    for ev, (lower, upper, vol, leaf_val) in enumerate(vol_pd_lst):
        if vol >= vol_thres or T.nodes[lower][val_name] <= stop_saddle_val:
            kept_branches.append((lower, upper, vol, leaf_val))
            kept_branch_idx.add(ev)
    
    # print("ALL BRANCHES:", vol_pd_lst)
    # print("KEPT BRANCHES:", kept_branches)
    
    # we create a shortcut to get the next (higher/closer-to-leaf) node of "node" in its highest-volume branch
    next_node_in_branch = {}
    for chain in vol_pd_chain:
        for i, node in enumerate(chain):
            if i > 0 and (node not in next_node_in_branch):
                next_node_in_branch[node] = chain[i-1]
                
    # we initiate volume mapping: finding the nearest ancestor node of which the branch is kept
    for node in T.nodes():
        vol_mapping[node] = node
        if child_in_branch[node] not in kept_branch_idx:
            nn = parents[node]
            while nn != -1:
                if child_in_branch[nn] in kept_branch_idx:
                    # the volume of "node" merges into the branch passing through "nn" (i.e., "nn" should NOT be the root)
                    # we map the volume to the branch of "nn", but, one node higher than "nn"
                    vol_mapping[node] = next_node_in_branch[nn]
                    break
                else:
                    nn = parents[nn]
    
    # print("Initial volume mapping:", vol_mapping)
    
    # we create a new tree based on all kept branches
    # we need to rework the node volume after simplification
    # The simplest way to achieve it is via the dependent volume
    
    # UPDATE: new_idx & old_idx are not IDENTITY mapping
    new_idx = {} 
    for lower, upper, vol, leaf_val in kept_branches:
        add_branch_nodes(T_sim, T, subtree_leaves, parents, new_idx, lower, upper)
    
    for node in vol_mapping:
        # if vol_mapping[node] not in T_sim.nodes:
        #     print("Error comes from:", node, vol_mapping[node])
        #     print("Simplified Tree:", T_sim.nodes)
        assert vol_mapping[node] in T_sim.nodes
    # print("Rebuild Simplified Tree Done.")
        
    # nx.draw_networkx(T_sim)
    # plt.show()
    # plt.close()
    assert nx.is_tree(T_sim)
    assert hasattr(T_sim, "root")
        
    ##### DEPRECATED #####
    # we need to recompute the node volume based on the dependent volume in the original tree
    # dependent_vol[node] = sum([dependent_vol[child] for child in children]) + node_vol[node]
    # dependent volumes of nodes are invariant from the tree simplification
    ######################
    
    # Instead of using dependent volume, we have volume mapping to help us update the new volume
    # This has to be done on the augmented simplified tree (i.e., saddles of removed branches still in the tree)
    # compute_simplified_node_vol(T_sim, T, -1, T_sim.root, vol_mapping, vol_name=vol_name)
    # print("Compute Volume in Simplified Tree Done.")
    
    # we remove nodes with a degree of two, and merge its edges together
    def add_remove_link(remove_links, st, tg, weight):
        if st in remove_links:
            assert remove_links[st] == (tg, weight)
        else:
            remove_links[st] = (tg, weight)
            
    def compress_mapping(mapping):
        while True:
            changed = False
            for key in mapping:
                if mapping[key] in mapping:
                    mapping[key] = mapping[mapping[key]]
                    changed = True
            if not changed:
                return
    
    while True:
        nodes_to_remove = set()
        edges_to_add = []
        vols_to_modify = []
        remove_links = {}
        update_mapping = {}
        for node in T_sim.nodes():
            adjacents = list(T_sim.adj[node])
            if len(adjacents) == 2:
                u, v = adjacents
                weights2u = T_sim.edges[node, u]['weight']
                weights2v = T_sim.edges[node, v]['weight']
                nodes_to_remove.add(node)
                
                # larger dependent volume is closer to the root
                # 1. we move volumes away from the root
                # 2. we prune edges toward the root (because the path is unique)
                
                # case 1: node is at lower level than u
                #   i.e.: root -> *** -> v -> node -> u
                if T_sim.nodes[node]["dependent_vol"] > T_sim.nodes[u]["dependent_vol"]:
                    # vols_to_modify.append([node, u, T_sim.nodes[node][vol_name]])
                    update_mapping[node] = u
                    add_remove_link(remove_links, u, node, weights2u)
                    add_remove_link(remove_links, node, v, weights2v)
                # case 2: node is at lower level than v
                #   i.e.: root -> *** -> u -> node -> v
                else:
                    assert T_sim.nodes[node]["dependent_vol"] > T_sim.nodes[v]["dependent_vol"]
                    update_mapping[node] = u
                    # vols_to_modify.append([node, v, T_sim.nodes[node][vol_name]])
                    add_remove_link(remove_links, v, node, weights2v)
                    add_remove_link(remove_links, node, u, weights2u)
        
        if len(nodes_to_remove) == 0:
            break
        
        # we compress update_mapping first
        compress_mapping(update_mapping)
        
        # we update volume mapping
        for node in vol_mapping:
            if vol_mapping[node] in update_mapping:
                vol_mapping[node] = update_mapping[vol_mapping[node]]
        
        for node in nodes_to_remove:
            # for ei, (_, target, _) in enumerate(vols_to_modify):
            #     if target == node:
            #         for st_search, target_search, _ in vols_to_modify:
            #             if st_search == target:
            #                 vols_to_modify[ei][1] = target_search
            T_sim.remove_node(node)
        
        # for st, target, vol in vols_to_modify:
        #     T_sim.nodes[target][vol_name] += vol
        
        for st_node in remove_links:
            if st_node in nodes_to_remove:
                continue
            tgt, weight = remove_links[st_node]
            while tgt in nodes_to_remove:
                assert tgt in remove_links
                next, next_weight = remove_links[tgt]
                weight += next_weight
                # print(tgt, "->", next)
                tgt = next
            edges_to_add.append((st_node, tgt, weight))
            
        # edges_to_add.sort()
        # head = 0
        # while head < len(edges_to_add) - 1:
        #     u, v, weight = edges_to_add[head]
        #     tail = head + 1
        #     while tail < len(edges_to_add) and edges_to_add[tail][0] < v:
        #         tail += 1
        #     if tail >= len(edges_to_add) or edges_to_add[tail][0] > v:
        #         head += 1
        #         continue
            
        #     while tail < len(edges_to_add) and edges_to_add[tail][0] == v:
        #         uu, vv, ww = edges_to_add[tail]
        #         edges_to_add[head] = [u, vv, weight + ww]
        #         edges_to_add.pop(tail)
        
        # head = 0
        # while head < len(edges_to_add):
        #     u, v, weight = edges_to_add[head]
        #     if u in nodes_to_remove and v in nodes_to_remove:
        #         edges_to_add.pop(head)
        #     else:
        #         assert u not in nodes_to_remove and v not in nodes_to_remove
        #         head += 1
            
        for u, v, weight in edges_to_add:
            T_sim.add_edge(u, v, weight=weight)
    
        assert nx.is_tree(T_sim)
    
    # print("Compressed Volume Mapping:", vol_mapping)
    
    T_return, vol_mapping = rename_node(T_sim, vol_mapping)
    # print("Renamed Volume Mapping:", vol_mapping)
    
    for node in T_return.nodes:
        T_return.nodes[node][vol_name] = 0
    
    for node in vol_mapping:
        T_return.nodes[vol_mapping[node]][vol_name] += T.nodes[node][vol_name]
    # print("Remove Redundant Nodes Done. Returning the tree...")
            
    return T_return, vol_mapping
    

# Return a stochastic matrix based on mat.
# stochastic matrix: sum of cols agree with p; sum of rows agree with q.
# This function can be used to generate a initialization of coupling matrix 
# for any given positive matrix input (0 values cannot be changed, so it is discouraged).
# However, try to avoid value that is too large or small to avoid weird float
# division / multiplication behavior.
def stochastic_matrix(mat: np.ndarray, p=None, q=None, eps=1e-12, max_iters=1000):
    N, M = mat.shape
    if p is None:
        p = np.ones((N, )) / N
    if q is None:
        q = np.ones((M, )) / M
    
    assert len(p) == N
    assert len(q) == M
        
    # L2 norm as the loss for probability sum
    def get_loss(a, p, q):
        x_sum = np.sum(a, axis=1)
        y_sum = np.sum(a, axis=0)
        x_loss = np.linalg.norm(x_sum - p)
        y_loss = np.linalg.norm(y_sum - q)
        return x_loss, y_loss
    
    for iter in range(max_iters):
        # fixing rows
        x_dev = p / np.sum(mat, axis=1)
        change_by_row = copy.deepcopy(np.broadcast_to(x_dev.reshape(N, 1), (N, M)))
        mat *= change_by_row
    
        # fixing columns
        y_dev = q / np.sum(mat, axis=0)
        change_by_col = copy.deepcopy(np.broadcast_to(y_dev.reshape(1, M), (N, M)))
        mat *= change_by_col

        x_loss, y_loss = get_loss(mat, p, q)
        if x_loss + y_loss < eps:
            break
    
    if x_loss + y_loss >= eps:
        print("Warning: G0 Matrix does not converge within {} iters. Consider adding more iterations.".format(str(max_iters)))
        print("Remaining loss:", x_loss + y_loss)
    
    return mat


def initialize_G0(h1, h2, p1, p2, max_dist, pairwiseDist):
    # can we initialize a coupling matrix based on the pairwise distance?
    if pairwiseDist is None:
        N = len(h1)
        M = len(h2)
        init_val = 1 / (N * M)
        better_val = max(N, M, 100) * init_val
        G0 = np.ones((N, M)) * init_val
        for i in range(N):
            for j in range(M):
                # for different types of cps, no initial probability
                if len(h1[i]) > 3 and (h1[i][-1] != h2[j][-1]):
                    continue
                # for maxima, discuss the matched distance
                if len(h1[i]) <= 3 or (h1[i][-1] == 2):
                    dist = label_distance(h1[i, :3], h2[j, :3], metric='l2')
                    if dist <= max_dist:
                        G0[i, j] = better_val
                # for saddles, allow all pairs of matching
                elif (len(h1[i]) > 3) and (h1[i][-1] == 1):
                    G0[i, j] = better_val
                # for global minimum, allow matching
                else:
                    G0[i, j] = better_val
    else:
        N, M = pairwiseDist.shape
        init_val = 1 / (N * M)
        better_val = max(N, M, 100) * init_val
        G0 = np.ones((N, M)) * init_val
        for i in range(N):
            for j in range(M):
                dist = pairwiseDist[i, j]
                if dist <= max_dist:
                    G0[i, j] = better_val

    G0 = stochastic_matrix(G0, p1, p2)
    return G0


# computing the max matched distance based on the coupling
# prob_rubric: "max": we consider the matched distance between critical points 
#                that are matched in one-to-one matching strategy, which requires 
#                the coupling probability to be the max within the row and column
#              "nonzero": all nonzero entries in the coupling will be considered as matching
# Note: we assume that the last dimension of h is always critical point type,
#       which we use to avoid computing the distance between saddles
def max_matched_distance_with_labels(h1, h2, oc, prob_rubric="max", pairwiseDist=None):
    if pairwiseDist is None:
        max_dist = 0

        N = len(h1)
        M = len(h2)
        for i in range(N):
            if (len(h1[i]) > 3) and (h1[i][-1] != 2):
                continue
            for j in range(M):
                if (len(h2[j]) > 3) and (h2[j][-1] != 2):
                    continue
                if prob_rubric == 'max':
                    if np.max(oc[i, :]) == oc[i, j] and np.max(oc[:, j]) == oc[i, j] and oc[i, j] > 1e-2 / (N*M):
                        max_dist = max(max_dist, label_distance(h1[i][:3],
                                                                h2[j][:3]))
                elif prob_rubric == 'nonzero':
                    if (oc[i, j] > 1e-3 * np.sum(oc[i, :])) or (oc[i, j] > 1e-3 * np.sum(oc[:, j])):
                        max_dist = max(max_dist, label_distance(h1[i][:3],
                                                                h2[j][:3]))
                else:
                    raise NotImplementedError
    else:
        max_dist = 0
        N, M = pairwiseDist.shape
        
        for i in range(N):
            for j in range(M):
                if prob_rubric == 'max':
                    if np.max(oc[i, :]) == oc[i, j] and np.max(oc[:, j]) == oc[i, j] and oc[i, j] > 1e-2 / (N*M):
                        max_dist = max(max_dist, pairwiseDist[i, j])
                elif prob_rubric == 'nonzero':
                    if (oc[i, j] > 1e-3 * np.sum(oc[i, :])) or (oc[i, j] > 1e-3 * np.sum(oc[:, j])):
                        max_dist = max(max_dist, pairwiseDist[i, j])
                else:
                    raise NotImplementedError
    
    return max_dist


# computing the max matched distance based on the coupling
# prob_rubric: "max": we consider the matched distance between critical points 
#                that are matched in one-to-one matching strategy, which requires 
#                the coupling probability to be the max within the row and column
#              "nonzero": all nonzero entries in the coupling will be considered as matching
def max_matched_distance_with_trees(tree1, tree2, h1, h2, oc, prob_rubric="max"):
    def crit_type_fail(node):
        if node['type'] != 2:
            return True
        return False

    max_dist = 0
    for node1 in tree1.nodes():
        if crit_type_fail(tree1.nodes[node1]):
            continue
        for node2 in tree2.nodes():
            if crit_type_fail(tree2.nodes[node2]):
                continue
            if prob_rubric == 'max':
                if np.max(oc[node1, :]) == oc[node1, node2] and np.max(oc[:, node2]) == oc[node1, node2]:
                    max_dist = max(max_dist, label_distance(h1[node1][:3],
                                                            h2[node2][:3]))
            elif prob_rubric == 'nonzero':
                if oc[node1, node2] > 1e-2 * np.sum(oc[node1, :]) and oc[node1, node2] > 1e-2 * np.sum(oc[:, node2]):
                    max_dist = max(max_dist, label_distance(h1[node1][:3],
                                                            h2[node2][:3]))
            else:
                raise NotImplementedError
    return max_dist


def save_binary_parameter_tuning(parameter_tuning_path, oc_path, alpha, best_ms, best_ocs, m_searchspace, dist_values):
    os.makedirs(parameter_tuning_path, exist_ok=True)
    alpha_str = str(round(alpha, 1))
    np.savetxt(pjoin(parameter_tuning_path, 
                     "best_ms_alpha_{}.txt".format(alpha_str)), 
               best_ms)
    for i, oc in enumerate(best_ocs):
        np.save(pjoin(oc_path, 
                      "alpha_{}_oc_{}.npy".format(alpha_str, str(i))), 
                oc)

    parameter_tuning_stats = {"m": m_searchspace, "max_matched_distance": dist_values}
    df = pd.DataFrame(parameter_tuning_stats)
    df.to_csv(pjoin(parameter_tuning_path, 
                    "m_tuning_stats_alpha_{}.csv".format(alpha_str)))

def load_binary_parameter_tuning(parameter_tuning_path, oc_path, alpha):
    alpha_str = str(round(alpha, 1))
    best_ms = np.atleast_1d(np.loadtxt(pjoin(parameter_tuning_path, 
                               "best_ms_alpha_{}.txt".format(alpha_str))))
    N_ocs = len(best_ms)
    best_ocs = []
    for i in range(N_ocs):
        oc = np.load(pjoin(oc_path, 
                           "alpha_{}_oc_{}.npy".format(alpha_str, str(i))), 
                     allow_pickle=True)
        best_ocs.append(oc)
    ds = pd.read_csv(pjoin(parameter_tuning_path, 
                           "m_tuning_stats_alpha_{}.csv".format(alpha_str)))
    df = pd.DataFrame(ds)
    parameter_tuning_stats = df.to_dict()
    m_searchspace = [float(x) for x in parameter_tuning_stats["m"]]
    dist_values = [float(x) for x in parameter_tuning_stats["max_matched_distance"]]
    return best_ms, best_ocs, m_searchspace, dist_values

    
def output_instance_debug(T, id, path: str):
    tree_filename = os.path.join(path, "treeNode_highlight_" + str(id).zfill(3) + ".txt")
    tree_file = open(tree_filename, "w")

    for node in T.nodes():
        ff = tree_file
        if T.nodes[node]["type"] != 2:
            continue
        T.nodes[node]["color_value"] = 0
        T.nodes[node]["color_length"] = 0
        T.nodes[node]["mark"] = 0
        print(T.nodes[node]["x"],
              T.nodes[node]["y"],
              T.nodes[node]["z"],
              T.nodes[node]["height"],
              T.nodes[node]["type"],
              T.nodes[node]["color_value"],
              T.nodes[node]["dependent_vol"],
              T.nodes[node]["volume"],
              file=ff)
        
    tree_file.close()
    
    
from functools import partial
from concurrent import futures

def concurrent_list_min_eucl_distance(list2, list1):
    return cdist(list1, list2, 'euclidean').min()

def concurrent_kd_min_eucl_distance(list2, kdT1):
    distances, _ = kdT1.query(list2, k=1)
    return np.min(distances)

def bounding_box_distance(bbox1, bbox2):
    """
    Compute the minimum distance between two bounding boxes.
    Each bounding box is represented as (min_row, min_col, max_row, max_col).
    """
    # Check if bounding boxes overlap
    if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
        bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
        # Bounding boxes do not overlap, compute distance between closest corners
        dx = max(bbox1[1] - bbox2[3], bbox2[1] - bbox1[3], 0)
        dy = max(bbox1[0] - bbox2[2], bbox2[0] - bbox1[2], 0)
        return np.sqrt(dx**2 + dy**2)
    else:
        # Bounding boxes overlap, minimum distance is 0
        return 0

def have_common_element(arr1, arr2):
    """
    Check if two arrays have at least one common element using sets.
    """
    set_arr1 = {tuple(each) for each in arr1}
    set_arr2 = {tuple(each) for each in arr2}
    return not set_arr1.isdisjoint(set_arr2)

def have_adjacent_element(arr1, arr2):
    """
    Check if two arrays have at least one adjacent element using sets.
    """
    set_arr1 = {tuple(each) for each in arr1}
    dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    for ex, ey in arr2:
        for dx, dy in dirs:
            nx = ex + dx
            ny = ey + dy
            if (nx, ny) in set_arr1:
                return True
    return False
    
def segmentation_distance(label_map1, label_map2, max_dist, normalize_factor=1, max_workers=7):
    N1, M1 = label_map1.shape
    N2, M2 = label_map2.shape
    assert (N1 == N2) and (M1 == M2)
    
    def get_boundary_pixels(candidates, mask):
        dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        positives = []
        for x, y in candidates:
            on_boundary = False
            for ndirx, ndiry in dirs:
                nx = x + ndirx
                ny = y + ndiry
                if (nx < 0) or (nx >= N1) or (ny < 0) or (ny >= M1):
                    continue
                if not mask[nx, ny]:
                    on_boundary = True
                    break
            if on_boundary:
                positives.append([x, y])
        return np.asarray(positives, dtype=int)
    
    label_index1, label_counts1 = np.unique(label_map1, return_counts=True)
    label_index2, label_counts2 = np.unique(label_map2, return_counts=True)
    
    label_num1 = len(label_counts1) - 1
    label_num2 = len(label_counts2) - 1
    
    seg_distance = np.ones((label_num1, label_num2)) * -1
    
    bboxes1 = []
    positives1 = []
    positives1_bounds = []
    for idx1 in range(label_num1):
        lab1 = idx1 + 1
        positives1_candidates = np.argwhere(label_map1==lab1)
        bbox1 = (np.min(positives1_candidates[:, 0]), np.min(positives1_candidates[:, 1]),
                 np.max(positives1_candidates[:, 0]), np.max(positives1_candidates[:, 1]))
        bboxes1.append(bbox1)
        positive_mask1 = label_map1==lab1
        positives1.append(positives1_candidates)
        positives1_bounds.append(get_boundary_pixels(positives1_candidates, positive_mask1))
        
    bboxes2 = []
    positives2 = []
    positives2_bounds = []
    for idx2 in range(label_num2):
        lab2 = idx2 + 1
        positives2_candidates = np.argwhere(label_map2==lab2)
        bbox2 = (np.min(positives2_candidates[:, 0]), np.min(positives2_candidates[:, 1]),
                 np.max(positives2_candidates[:, 0]), np.max(positives2_candidates[:, 1]))
        bboxes2.append(bbox2)
        positive_mask2 = label_map2==lab2
        positives2.append(positives2_candidates)
        positives2_bounds.append(get_boundary_pixels(positives2_candidates, positive_mask2))
    
    for idx1 in range(label_num1):
        positive1 = positives1[idx1]

        separating_idx2s = []
        overlapping_idx2s = []
        # First, bounding box distance. If too far, we skip computation
        for idx2 in range(label_num2):
            bbox_dist = bounding_box_distance(bboxes1[idx1], bboxes2[idx2])
            if bbox_dist > max_dist:
                seg_distance[idx1, idx2] = normalize_factor * 10
            elif bbox_dist == 0:
                if have_common_element(positive1, positives2[idx2]):
                    seg_distance[idx1, idx2] = 0
                elif have_adjacent_element(positive1, positives2_bounds[idx2]):
                    seg_distance[idx1, idx2] = 1
                else:
                    overlapping_idx2s.append(idx2)
            else:
                if bbox_dist < 2:
                    if have_adjacent_element(positive1, positives2_bounds[idx2]):
                        seg_distance[idx1, idx2] = 1
                        continue
                separating_idx2s.append(idx2)
            
        # print("Separating idx2s:", len(separating_idx2s))
        # print("Overlapping idx2s:", len(overlapping_idx2s))
        
        if max_workers <= 1:
            kdT1_bounds = cKDTree(positives1_bounds[idx1])
            for idx2 in separating_idx2s:
                positive2 = positives2_bounds[idx2]
                distances, _ = kdT1_bounds.query(positive2, k=1)
                seg_distance[idx1, idx2] = np.min(distances)
                
            kdT1 = cKDTree(positive1)
            for idx2 in overlapping_idx2s:
                positive2 = positives2[idx2]
                distances, _ = kdT1.query(positive2, k=1)
                seg_distance[idx1, idx2] = np.min(distances)
        else:
            kdT1_bounds = cKDTree(positives1_bounds[idx1])
            kdT1 = cKDTree(positive1)
            
            cc_kd_euclid_dist_bounds = partial(concurrent_kd_min_eucl_distance,
                                               kdT1=kdT1_bounds)
            cc_kd_euclid_dist = partial(concurrent_kd_min_eucl_distance,
                                        kdT1=kdT1)

            separating_positives2 = [positives2_bounds[each] for each in separating_idx2s]
            overlapping_positives2 = [positives2[each] for each in overlapping_idx2s]
            
            with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                for idx2, dist in zip(overlapping_idx2s, pool.map(cc_kd_euclid_dist, overlapping_positives2)):
                    seg_distance[idx1, idx2] = dist
                for idx2, dist in zip(separating_idx2s, pool.map(cc_kd_euclid_dist_bounds, separating_positives2)):
                    seg_distance[idx1, idx2] = dist
                    
        print("Finished {}/{}".format(str(idx1 + 1), str(label_num1)))        
        
    return seg_distance