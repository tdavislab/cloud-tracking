import numpy as np
import os
import sys
from os.path import join as pjoin
from scipy.optimize import linear_sum_assignment as lsa
import copy
import pickle
import time
from scipy.ndimage import label as lbl

class Node:
    def __init__(self, row_data, t: int) -> None:
        assert len(row_data) >= 5
        # ["x", "y", "z", "height", "type"]
        self.x = int(row_data[0])
        self.y = int(row_data[1])
        self.z = int(row_data[2])
        self.height = float(row_data[3])
        self.type = int(row_data[4])
        self.cluster = None
        self.time = t
        assert row_data[4] == 2
        
    def pos(self):
        return np.asarray([self.x, self.y, self.z])
    
    def __str__(self) -> str:
        return "({}, {}, {}) - height={}, time={}, cluster={}".format(str(self.x), str(self.y), str(self.z), str(self.height), str(self.time), str(self.cluster))


def load_ocs(path):
    oc_files = []
    for rr, dd, files in os.walk(path):
        for file in files:
            if "oc" in file and file.endswith("txt"):
                oc_files.append((rr, file))
    
    oc_files.sort(key=lambda x: int(x[1].split("_")[1]))
    ocs = []
    for rr, oc_file in oc_files:
        oc = np.loadtxt(pjoin(rr, oc_file))
        ocs.append(oc)
        
    return ocs

def load_nodes(path):
    node_files = []
    for rr, dd, files in os.walk(path):
        for file in files:
            if "treeNode_highlight" in file and file.endswith("txt"):
                node_files.append((rr, file))
    
    node_files.sort(key=lambda x: int(x[1].split("_")[2].split(".")[0]))
    nodes_list = []
    for i, (rr, node_file) in enumerate(node_files):
        nodes_data = np.loadtxt(pjoin(rr, node_file))
        nodes = []
        for row in nodes_data:
            nodes.append(Node(row, i))
        nodes_list.append(nodes)
        
    return nodes_list


def load_scalarfield(path):
    files = os.listdir(path)
    scalar_files = []
    for file in files:
        if file.endswith("txt") or file.endswith("npy"):
            scalar_files.append(file)
    
    try:
        scalar_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except:
        scalar_files.sort()
    
    scalars_list = []
    for scalar_file in scalar_files:
        if scalar_file.endswith("txt"):
            scalars = np.loadtxt(pjoin(path, scalar_file), dtype=float)
        else:
            assert scalar_file.endswith("npy")
            scalars = np.load(pjoin(path, scalar_file), allow_pickle=True)
        scalars_list.append(scalars)
        
    return scalars_list


def load_data(dataset, simplification, alpha):
    result_path = None
    results_folders = os.listdir("./initial-output/")
    for folder in results_folders:
        if dataset in folder and (simplification is None or (("_" + simplification) in folder)):
            result_path = pjoin("./initial-output/", folder)
            break
    result_path = pjoin(result_path, alpha)
    print("Loading output from", result_path)
    
    ocs = load_ocs(result_path)
    nodes = load_nodes(result_path)
    scalars = load_scalarfield(pjoin("../scalar/", dataset))

    # Sanity check: len(ocs) + 1 == len(nodes) == len(scalars)
    print(len(ocs), len(nodes), len(scalars))
    assert len(ocs) + 1 == len(nodes)
    assert len(nodes) <= len(scalars)
    
    # More sanity check: oc shape vs. node size
    for i in range(len(ocs)):
        N, M = ocs[i].shape
        assert len(nodes[i]) == N
        assert len(nodes[i+1]) == M
        
    return ocs, nodes, scalars


# Returns: clusters data
#          clusters[t]: all clusters of clouds at time step t
#          clusters[t][i]: the i-th cluster of clouds at time step t
#          clusters[t][i][k]: the k-th node in the i-th cluster of clouds at time step t
def compute_superlevel_cluster(all_nodes: list[list[Node]], all_scalars, thres, tType, report_sizes=False):
    # UPDATE: 08/29/2024
    # DC: if the distance between two cloud systems is more than 2-3 pixels (4-6 km), they can be considered independent systems. 
    # Therefore, we add a step to merge cloud systems if the Euclidean distance between their boundaries is less than 3 pixels (which searches the range [0~2, 0~2])
    if "system" in tType:
        cloud_merge_distance = 2
        print("NOTICE: This is computing cloud systems!")
    else:
        assert "object" in tType
        cloud_merge_distance = 0
        print("NOTICE: This is computing cloud objects!")
    
    structure = np.ones((3, 3))
    clusters = []
    clustered_sfs = []
    clustered_pixels = []
    
    # At each timestep, we compute the superlevel set component
    for t in range(len(all_nodes)):
        # scalar field at time t
        scalars = all_scalars[t]
        N, M = scalars.shape
        
        superlevels = copy.deepcopy(scalars)
        superlevels[scalars < thres] = 0
        
        # initial segmentation
        labels, nLabels = lbl(superlevels, structure=structure)
        visited = labels - 1
        
        remaining_labels = set()
        for ix, node in enumerate(all_nodes[t]):
            if node.height >= thres:
                assert visited[node.x, node.y] >= 0
                remaining_labels.add(visited[node.x, node.y])
        
        # After BFS, we start to check the boundaries of cloud systems and fuse systems close to each other
        # Algorithm: for each pixel, search its neighborhood. If found a different cluster ID, use a disjoint set to fuse them
        def djs_find(x, fa):
            if fa[x] != x:
                fa[x] = djs_find(fa[x], fa)
            return fa[x]

        cluster_fas = {key: key for key in remaining_labels}
        for i in range(N):
            for j in range(M):
                if visited[i, j] < 0:
                    continue
                if visited[i, j] not in remaining_labels:
                    visited[i, j] = -1
                    continue
                for di in range(cloud_merge_distance + 1):
                    for dj in range(cloud_merge_distance + 1):
                        if di ** 2 + dj ** 2 > cloud_merge_distance ** 2:
                            continue
                        if di + dj != 0:
                            next_x = int(i + di)
                            next_y = int(j + dj)
                            if next_x >= N:
                                continue
                            if next_y >= M:
                                continue
                            if visited[next_x, next_y] < 0:
                                continue
                            if visited[next_x, next_y] not in remaining_labels:
                                visited[next_x, next_y] = -1
                                continue
                            if visited[i, j] == visited[next_x, next_y]:
                                continue
                            fcur = djs_find(visited[i, j], cluster_fas)
                            fneighbor = djs_find(visited[next_x, next_y], cluster_fas)
                            if fcur != fneighbor:
                                if fcur < fneighbor:
                                    cluster_fas[fneighbor] = fcur
                                else:
                                    cluster_fas[fcur] = fneighbor
                                    
        new_cluster_ids = list(set([djs_find(old_key, cluster_fas) for old_key in cluster_fas]))
        new_cluster_ids.sort()
        new_id_map = {old_key: new_cluster_ids.index(djs_find(old_key, cluster_fas)) for old_key in cluster_fas}
        
        cluster_pixel = {}
        # Update cluster matrix
        for i in range(N):
            for j in range(M):
                if visited[i, j] < 0:
                    continue
                new_id = new_id_map[djs_find(visited[i, j], cluster_fas)]
                visited[i, j] = new_id
                if new_id in cluster_pixel:
                    cluster_pixel[new_id].append((i, j))
                else:
                    cluster_pixel[new_id] = [(i, j)]
        
        clustered_pixels.append(cluster_pixel)
        
        cluster_t = {}
        for ix, node in enumerate(all_nodes[t]):
            if node.height >= thres:
                assert visited[node.x, node.y] >= 0
                all_nodes[t][ix].cluster = visited[node.x, node.y]
                if visited[node.x, node.y] not in cluster_t:
                    cluster_t[visited[node.x][node.y]] = [ix]
                else:
                    cluster_t[visited[node.x][node.y]].append(ix)
                        
        assert len(np.unique(visited)) == len(np.unique(list(cluster_t.keys()))) + 1
            
        cluster_t_lst = []
        for key in cluster_t:
            cluster_t_lst.append(cluster_t[key])
        
        print("Time step, #clusters, #clusters after merge:", t, len(remaining_labels), len(cluster_t))
        clusters.append(cluster_t_lst)
        clustered_sfs.append(visited)
        
    if not report_sizes:
        return clusters, all_nodes, clustered_sfs
    else:
        return clusters, all_nodes, clustered_sfs, clustered_pixels



# we select the "centroid" critical point for each cloud system
# The instability is that critical points appear and disappear very often
def compute_cluster_centroid_critical_point(all_nodes: list[list[Node]], cluster_data: list[list[list[int]]]) -> list[list[int]]:
    cluster_centroids = []
    # we compute the centroid of clusters at each time step as the representative
    for t in range(len(all_nodes)):
        nodes = all_nodes[t]
        clusters = cluster_data[t]
        centroids = []
        for cluster in clusters:
            min_sum_dist = None
            centroid = None
            for node_x in cluster:
                sum_dist = 0
                for node_y in cluster:
                    if node_x != node_y:
                        sum_dist += np.linalg.norm(nodes[node_x].pos() - nodes[node_y].pos())
                if min_sum_dist is None or (sum_dist < min_sum_dist):
                    centroid = node_x
                    min_sum_dist = sum_dist
            
            centroids.append(centroid)
        
        cluster_centroids.append(centroids)
    
    return cluster_centroids


# we select the "centroid" pixel for each cloud system
# The instability is that the cloud system shape and size change often
def compute_cluster_centroid(clustered_pixels):
    cluster_centroids = []
    
    for t in range(len(clustered_pixels)):
        # clusters: a dictionary of "area ID: [pixels]"
        clusters = clustered_pixels[t]
        centroids = {}
        
        for cluster in clusters:
            cluster_info = clusters[cluster]
            centroid = np.mean(cluster_info, axis=0)
            assert len(centroid) == 2
            centroids[cluster] = centroid
        cluster_centroids.append(centroids)
    return cluster_centroids


def debug_clusters(dataset, all_nodes: list[list[Node]], clustered_sfs):
    output_dir = pjoin("./debug/", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    for t in range(len(all_nodes)):
        nodes = all_nodes[t]
        output_f = open(pjoin(output_dir, "clusters_{}.txt".format(str(t).zfill(3))), "w")
        for i in range(len(nodes)):
            node = nodes[i]
            print(node.x, node.y, node.z, node.height, node.type, node.cluster if node.cluster is not None else -1, 0, 0, file=output_f)
        
        output_f.close()
        
        clustered_sf = clustered_sfs[t]
        output_fname = pjoin(output_dir, "cluster_region_{}.npy".format(str(t).zfill(3)))
        np.save(output_fname, clustered_sf)


def debug_centroids(dataset, all_nodes: list[list[Node]], all_centroids):
    output_dir = pjoin("./debug/", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    for t in range(len(all_centroids)):
        nodes = all_nodes[t]
        centroids = all_centroids[t]
        output_f = open(pjoin(output_dir, "centroids_{}.txt".format(str(t).zfill(3))), "w")
        for i in range(len(centroids)):
            cent = centroids[i]
            print(nodes[cent].x, nodes[cent].y, nodes[cent].z, nodes[cent].height, nodes[cent].type, i, 0, 0, file=output_f)
        
        output_f.close()
        
        
def compute_matching_scores(all_clusters, all_nodes: list[list[Node]], all_ocs, all_diameters=None, diameter_weight=None):
    matching_scores = []
    for t in range(len(all_clusters) - 1):
        scores = np.zeros((len(all_clusters[t]), len(all_clusters[t+1])))
        oc = all_ocs[t]
        N, M = oc.shape
        # print(N, len(all_nodes[t]))
        # print(M, len(all_nodes[t+1]))
        for i in range(N):
            for j in range(M):
                if (all_nodes[t][i].cluster is not None) and (all_nodes[t+1][j].cluster is not None):
                # TODO: include diameter of cluster to remove probability of matching between faraway critical points
                    scores[all_nodes[t][i].cluster][all_nodes[t+1][j].cluster] += oc[i][j]
        
        matching_scores.append(scores)
    return matching_scores


def compute_one2one_matching(matching_scores, strategy="munkres", clustered_sfs=None, oc_thres=None):
    cloud_id_mapping = {}
    start_time = {}
    latest_time = {}
    cloud_ids = 0
    
    if strategy == "max-max":
        for t in range(len(matching_scores)):
            scores = matching_scores[t]
            N, M = scores.shape
            
            if t == 0:
                # We initiate the cloud_ids for all clouds at the first time step
                for i in range(N):
                    cloud_id_mapping[(t, i)] = cloud_ids
                    start_time[cloud_ids] = t
                    latest_time[cloud_ids] = t
                    cloud_ids += 1
                
            match_axis0 = np.argmax(scores, axis=0)  # len(match_axis0) == M
            match_axis1 = np.argmax(scores, axis=1)  # len(match_axis1) == N

            for j in range(M):
                x = match_axis0[j]
                y = match_axis1[x]
                if j == y and (scores[x][y] > 0):
                    # x -> y is a one-to-one trajectory
                    cloud_id_mapping[(t+1, j)] = cloud_id_mapping[(t, x)]
                    latest_time[cloud_id_mapping[(t, x)]] = t + 1
                else:
                    # j is a new trajectory
                    cloud_id_mapping[(t+1, j)] = cloud_ids
                    start_time[cloud_ids] = t + 1
                    latest_time[cloud_ids] = t + 1
                    cloud_ids += 1
    elif strategy == "munkres":
        for t in range(len(matching_scores)):
            neg_scores = -matching_scores[t]
            N, M = neg_scores.shape
            row_ind, col_ind = lsa(neg_scores)
            if t == 0:
                # We initiate the cloud_ids for all clouds at the first time step
                for i in range(N):
                    cloud_id_mapping[(t, i)] = cloud_ids
                    start_time[cloud_ids] = t
                    latest_time[cloud_ids] = t
                    cloud_ids += 1
                    
            checked_cols = set()
            for row, col in zip(row_ind, col_ind):
                checked_cols.add(col)
                if neg_scores[row][col] < 0:
                    cloud_id_mapping[(t+1, col)] = cloud_id_mapping[(t, row)]
                    latest_time[cloud_id_mapping[(t, row)]] = t + 1
                else:
                    cloud_id_mapping[(t+1, col)] = cloud_ids
                    start_time[cloud_ids] = t + 1
                    latest_time[cloud_ids] = t + 1
                    cloud_ids += 1
            
            for col in range(M):
                if col not in checked_cols:
                    cloud_id_mapping[(t+1, col)] = cloud_ids
                    start_time[cloud_ids] = t + 1
                    latest_time[cloud_ids] = t + 1
                    cloud_ids += 1
    elif strategy == "area-priority":
        print("Using area-priority strategy. The oc_threshold should be considerably high to avoid arbitrary large cloud matchings.")
        if clustered_sfs is None or oc_thres is None:
            raise ValueError("Need clustered_sfs and oc_thres for area-based matching")
        sortByAreas = []
        for t in range(len(matching_scores)):
            scores = matching_scores[t]
            N, M = scores.shape
            
            labels_c = clustered_sfs[t]
            temp = [(np.sum(labels_c == i), i) for i in range(N)]
            temp.sort(reverse=True)
            sortByAreas.append(temp)
            
            if t == len(matching_scores) - 1:
                labels_n = clustered_sfs[t + 1]
                temp = [(np.sum(labels_n == i), i) for i in range(M)]
                temp.sort(reverse=True)
                sortByAreas.append(temp)
                
        assert len(sortByAreas) == len(matching_scores) + 1
        print("Finished computing area sorting.")
        
        for t in range(len(matching_scores)):
            scores = matching_scores[t]
            
            N, M = scores.shape
            
            sortByArea_cur = sortByAreas[t]
            sortByArea_next = sortByAreas[t + 1]
            
            if t == 0:
                # We initiate the cloud_ids for all clouds at the first time step
                for i in range(N):
                    cloud_id_mapping[(t, i)] = cloud_ids
                    start_time[cloud_ids] = t
                    latest_time[cloud_ids] = t
                    cloud_ids += 1
            
            checked_cols = set()
            
            row_scores = np.sum(scores, axis=1)
            col_scores = np.sum(scores, axis=0)
            
            for i in range(N):
                row = sortByArea_cur[i][1]
                for j in range(M):
                    col = sortByArea_next[j][1]
                    if col in checked_cols:
                        continue
                    # as long as there is valid assignment
                    if (scores[row][col] > 0) and ((scores[row, col] == np.max(scores[row, :])) and (scores[row, col] == np.max(scores[:, col])) or 
                        (scores[row, col] >= row_scores[row] * oc_thres) and (scores[row, col] >= col_scores[col] * oc_thres)):
                        checked_cols.add(col)
                        cloud_id_mapping[(t+1, col)] = cloud_id_mapping[(t, row)]
                        latest_time[cloud_id_mapping[(t, row)]] = t + 1
                        break
            
            for col in range(M):
                if col not in checked_cols:
                    cloud_id_mapping[(t+1, col)] = cloud_ids
                    start_time[cloud_ids] = t + 1
                    latest_time[cloud_ids] = t + 1
                    cloud_ids += 1
    else:
        raise NotImplementedError
    
    durations = []
    for cloud_id in start_time:
        durations.append(latest_time[cloud_id] - start_time[cloud_id] + 1)
    return cloud_id_mapping, durations
    

def output_trajectories(dataset, all_centroids, clustered_sfs, cloud_id_mapping):
    mapping_lengths = {}
    for key in cloud_id_mapping:
        time, cloud = key
        cloud_id = cloud_id_mapping[key]
        if cloud_id not in mapping_lengths:
            mapping_lengths[cloud_id] = set()
        mapping_lengths[cloud_id].add(time)
    
    output_dir = pjoin("./postprocess-result/", dataset)
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = pjoin("./debug/", dataset)
    os.makedirs(debug_dir, exist_ok=True)
    
    for t in range(len(all_centroids)):
        centroids = all_centroids[t]
        output_f = open(pjoin(output_dir, "treeNode_highlight_{}.txt".format(str(t).zfill(3))), "w")
        
        for centroid in centroids:
            cent_info = centroids[centroid]
            print(cent_info[0], cent_info[1], 0, 0, 2, cloud_id_mapping[(t, centroid)], len(mapping_lengths[cloud_id_mapping[(t, centroid)]]), 0, file=output_f)
            # print(cent_info[0], cent_info[1], 0, 0, 2, centroid, len(mapping_lengths[cloud_id_mapping[centroid]]), 0, file=output_f)
        # for i in range(len(centroids)):
        #     cent = centroids[i]
        #     id_key = (nodes[cent].time, nodes[cent].cluster)
        #     print(nodes[cent].x, nodes[cent].y, nodes[cent].z, nodes[cent].height, nodes[cent].type, cloud_id_mapping[id_key], len(mapping_lengths[cloud_id_mapping[id_key]]), 0, file=output_f)
        
        output_f.close()
        print("save trajectory files done for", dataset)
        
        clustered_sf = clustered_sfs[t]
        N, M = clustered_sf.shape
        for i in range(N):
            for j in range(M):
                if clustered_sf[i][j] >= 0:
                    clustered_sf[i][j] = cloud_id_mapping[(t, clustered_sf[i][j])]
        output_fname = pjoin(debug_dir, "cluster_region_{}.npy".format(str(t).zfill(3)))
        np.save(output_fname, clustered_sf)
        print("save feature map done for", dataset)
        
        
# def save_statistics(dataset, durations):
#     output_dir = pjoin("./track-statistics/pFGW/", dataset)
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = pjoin(output_dir, "trajectory_durations.txt")
#     np.savetxt(output_path, durations)

def output_info(dataset, clustered_sfs,  centroids, matching_scores, feed_tType=None):
    output_dir = pjoin("./track-info/{}".format(feed_tType), dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(pjoin(output_dir, "clustered_sf.pkl"), "wb") as outfile:
        pickle.dump(clustered_sfs, outfile)
        outfile.close()
        
    with open(pjoin(output_dir, "centroids.pkl"), "wb") as outfile:
        pickle.dump(centroids, outfile)
        outfile.close()
    
    with open(pjoin(output_dir, "matching_scores.pkl"), "wb") as outfile:
        pickle.dump(matching_scores, outfile)
        outfile.close()
    
        
def supplementary_trajectories(dataset, all_centroids, matching_scores, oc_thresold):
    T = len(all_centroids)
    output_dir = pjoin("./postprocess-result/", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    for t in range(T):
        centroids = all_centroids[t]
    
        
def main(dataset, simplification, alpha, superlevel_thres, oc_threshold=0.0, feed_tType=None, size_thres=1):
    # Loading data
    ocs, nodes, scalars = load_data(dataset, simplification, alpha)
    if simplification is not None:
        dataset = "-".join([dataset, simplification, alpha])
    else:
        dataset = "-".join([dataset, "0", alpha])
    
    print("Superlevel Thres:", superlevel_thres)
    for start_step, end_step_raw, thres_for_step in superlevel_thres:
        # till the end
        if end_step_raw is None:
            end_step = len(nodes) + 1
        else:
            end_step = end_step_raw + 1
        
        print(start_step, end_step)
        print(len(nodes), len(scalars), len(ocs))
        
        # Compute the clusters of clouds based on the superlevel threshold
        start_time_cluster = time.perf_counter()
        cluster_data, step_nodes, clustered_sfs, clustered_pixels = compute_superlevel_cluster(nodes[start_step:end_step], scalars[start_step:end_step], thres_for_step, tType=feed_tType, report_sizes=True)
        end_time_cluster = time.perf_counter()
        print("Clustering anchor points by superlevel set:", end_time_cluster - start_time_cluster, "sec")
        # debug_clusters("{}-thres-{}".format(dataset, str(round(superlevel_thres, 1))), nodes, clustered_sfs)
        
        # Get the centroid (i.e., the node with the smallest sum of distance to all nodes within the cluster) node of the cluster
        # as the representative of the cluster
        # centroids = compute_cluster_centroid_critical_point(nodes, cluster_data)
        centroids = compute_cluster_centroid(clustered_pixels)
        # debug_centroids("{}-thres-{}".format(dataset, str(round(superlevel_thres, 1))), nodes, centroids)
        
        # (Optional) We also compute the diameter of clusters in order to remove some unreasonable pairings
        
        # We compute the matching scores between clusters at adjacent time steps
        start_time_matching = time.perf_counter()
        matching_scores = compute_matching_scores(cluster_data, step_nodes, ocs[start_step:end_step])    
        end_time_matching = time.perf_counter()
        
        print("Cloud system matching:", end_time_matching - start_time_matching, "sec")
        
        # We output the segmentation results, centroids, and matching scores (likely critical points are not useful)
        # for a unified pipeline that can be used to compute the statistics and visualization for all three techniques
        output_info("{}-{}-{}-thres-{}".format(dataset, str(start_step), str(end_step - 1), str(round(thres_for_step, 1))), clustered_sfs, centroids, matching_scores, feed_tType)
        
        # Here we compute one-to-one matching, which yields a consistent ID for a cloud across time steps
        # The default algorithm to get one2one matching is via Munkres (Hungarian) algorithm 
        #   to compute the bipartite matching with minimum cost (i.e., -maximum sum of probabilities)
        # cloud_id_mapping, durations = compute_one2one_matching(matching_scores)
        
        # clustered_sfs_copy = copy.deepcopy(clustered_sfs)
        # Now, we output one-to-one matching using only centroids and clustered scalarfields
        # output_trajectories("{}-thres-{}".format(dataset, str(round(superlevel_thres, 1))), centroids, clustered_sfs, cloud_id_mapping)
        
        # (Optional) We output some supplementary trajectories to show feature split/merge, which needs parameter tuning on the threshold of accepted scores
        # if oc_threshold > 0:
            # supplementary_trajectories("{}-thres-{}".format(dataset, str(round(superlevel_thres, 1))), centroids, matching_scores, oc_threshold)
        # pass

if __name__ == '__main__':
    tType = str(sys.argv[1])
    size_thres = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print("Using size threshold = {} !!! If need change, please use python postprocess_juelich.py [size_thres]".format(str(size_thres)))
    
    print("This code postprocess the tracking results and generate vtk files for visualization.")
    print("For easier usage of the code, please specify the request in postprocess_juelich.config.")
    
    if tType not in {'pFGW-system', 'pFGW-object'}:
        raise ValueError("tType must be pFGW-system or pFGW-object!")
    
    dataset = None
    superlevel_thres = None
    alpha = None
    simplification = None
    oc_threshold = 0.0
    print("Using info from postprocess_juelich.config...")
    if os.path.exists("postprocess_juelich.config"):
        with open("postprocess_juelich.config") as config_f:
            isReadingDataset = False
            isReadingSuperlevel = False
            isReadingSimplification = False
            isReadingAlpha = False
            isReadingOcThres = False
            config_lines = config_f.readlines()
            for i in range(len(config_lines)):
                line = config_lines[i].strip().replace("\\", "/")
                if line.startswith("#") and "dataset" in line:
                    isReadingDataset = True
                    isReadingSuperlevel = isReadingSimplification = isReadingAlpha = isReadingOcThres = False
                    continue
                elif line.startswith("#") and "superlevel" in line:
                    isReadingSuperlevel = True
                    isReadingAlpha = isReadingDataset = isReadingSimplification = isReadingOcThres =  False
                    continue
                elif line.startswith("#") and "simplification" in line:
                    isReadingSimplification = True
                    isReadingAlpha = isReadingDataset = isReadingSuperlevel = isReadingOcThres = False
                    continue
                elif line.startswith("#") and "alpha" in line:
                    isReadingAlpha = True
                    isReadingDataset = isReadingSimplification = isReadingSuperlevel = isReadingOcThres = False
                    continue
                elif line.startswith("#") and ("oc" in line or "coupling" in line):
                    isReadingOcThres = True
                    isReadingDataset = isReadingSimplification = isReadingSuperlevel = isReadingAlpha = False
                    continue
                elif line.startswith("#"):
                    continue
                
                if len(line) > 0:
                    if isReadingDataset:
                        dataset = line.strip()
                        isReadingDataset = False
                    elif isReadingSuperlevel:
                        # format: start_step-end_step: threshold
                        line = line.strip()
                        try:
                            steps, thres_for_step = line.split(":", 2)
                            if steps.endswith("-"):
                                start_step = steps.split("-")[0]
                                end_step = None
                            else:
                                start_step, end_step = steps.split("-", 2)
                                end_step = int(end_step)
                            start_step = int(start_step)
                            thres_for_step = float(thres_for_step)
                            if superlevel_thres is None:
                                superlevel_thres = [(start_step, end_step, thres_for_step)]
                            else:
                                superlevel_thres.append((start_step, end_step, thres_for_step))
                        except:
                            print("Incorrect line format for superlevel set threshold! Skipping the line:", line)
                    elif isReadingSimplification:
                        simplification = line.strip() + "percent"
                        isReadingSimplification = False
                    elif isReadingAlpha:
                        alpha = line.strip()
                        isReadingAlpha = False
                    elif isReadingOcThres:
                        oc_threshold = float(line.strip())
                        isReadingOcThres = False
                    else:
                        print("Unparsed line:", line)
    else:
        print("Please use postprocess_juelich.config to specify the command.")
        exit()
        
    main(dataset, simplification, alpha, superlevel_thres, oc_threshold, tType, size_thres)