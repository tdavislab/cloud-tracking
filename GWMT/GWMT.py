from utilities import *
from FGW import fgw_lp, fgw_partial_lp
from optim import NonConvergenceError
import time
from functools import partial
from concurrent import futures


def concurrent_m_tuning(adaptive_m, CList1, CList2, pList1, pList2, hList1, hList2, alpha, amijo, fgw_wrapper, G0, metric):
    oc, _ = fgw_wrapper(CList1, CList2,
                        pList1, pList2,
                        hList1, hList2,
                        alpha, m=adaptive_m, amijo=amijo, G0=G0, metric=metric)
    return oc


def concurrent_fgw(CList1, CList2, pList1, pList2, hList1, hList2, m, alpha, amijo, fgw_wrapper, G0):
    oc, _ = fgw_wrapper(CList1, CList2, pList1, pList2, hList1, hList2, alpha, m=m, amijo=amijo, G0=G0)
    return oc

def concurrent_initialize_steps(step, pfgw, benchmark):
    if pfgw.all_scales is None:
        raise ValueError("Need to set self.all_scales beforehand! Please use self.set_all_scales() first.")
    
    if benchmark:
        start_time = time.perf_counter()
        
    if pfgw.initializedLst[step]:
        print("Timestep {} is already initialized! Skipping...".format(str(step)))
        if benchmark:
            return pfgw.CList[step], pfgw.pList[step], pfgw.hList[step], 0
        
    T = pfgw.trees[step]
    C, p = get_distance_and_distribution(T,
                                         distribution=pfgw.prob_distribution,
                                         weight_mode=pfgw.weight_mode,
                                         maxima_only=pfgw.maxima_only,
                                         root=pfgw.roots[step],
                                         scalar_name=pfgw.scalar_name,
                                         edge_weight_name=pfgw.edge_weight_name,
                                         volume_name="volume",
                                         scalar_threshold=pfgw.scalar_threshold,
                                         scalar_field=None if pfgw.scalar_fields is None else pfgw.scalar_fields[step],
                                         binary_map=None if pfgw.binary_maps is None else pfgw.binary_maps[step]
                                         )
    C /= pfgw.all_scales[-1]
    h = coordinate_matrix(T, pfgw.labels)
    if pfgw.maxima_only:
        maxima_ids = np.asarray([node for node in T.nodes() if T.nodes[node]["type"] == 2], dtype=int)
        C = C[maxima_ids, maxima_ids]
        p = p[maxima_ids]
        p /= np.sum(p)
        h = h[maxima_ids, :]
    for i in range(len(h[0])):
        h[:, i] /= pfgw.all_scales[i]
    
    print("CList shape:", C.shape)
    print("pList shape:", p.shape)
    print("hList shape:", h.shape)
    
    if benchmark:
        return C, p, h, time.perf_counter() - start_time
    return C, p, h, 0

# binary search the highest m such that the matched distance is below max_dist
def concurrent_m_binary_search(C1, C2, p1, p2, h1, h2, M, max_m, min_m, max_dist, alpha, amijo, fgw_wrapper, metric, prob_rubric, benchmark):
    head = min_m
    tail = max_m
    eps = 1e-2
    
    if M is not None:
        assert C1.shape[0] == M.shape[0]
        assert C2.shape[0] == M.shape[1]

    G0 = initialize_G0(h1, h2, p1, p2, max_dist, M)
    best_m = None
    best_oc = None
    mList = []
    distList = []
    alert = False
    
    # For benchmark
    if benchmark:
        start_time = time.perf_counter()
        run_cnts = 0
    
    while tail - head > eps:
        if benchmark:
            run_cnts += 1
        mid_m = (head + tail) / 2
        if alert:
            raise ValueError("Already passed the best choice to the result. Should not loop again!")
        oc, _ = fgw_wrapper(C1, C2, p1, p2, h1, h2, alpha, mid_m, amijo, G0, metric, M)
        dist = max_matched_distance_with_labels(h1, h2, oc, prob_rubric, M)
        mList.append(mid_m)
        distList.append(dist)
        if dist > max_dist:
            tail = mid_m
            if tail - head <= eps and best_m is None:
                print("(ms, dists):", [(a, b) for a, b in zip(mList, distList)])
                print("Cannot find the possible m with the current max_dist!")
                best_m = mid_m
                best_oc = oc
                alert = True
        else:
            head = mid_m
            best_m = mid_m
            best_oc = oc
    
    if benchmark:
        end_time = time.perf_counter()
        
    if best_m is None:
        print("(ms, dists):", [(a, b) for a, b in zip(mList, distList)])
        print("Cannot find the possible m with the current max_dist!")
        raise ValueError

    if benchmark:
        return best_m, best_oc, mList, distList, (end_time - start_time) / run_cnts    
    return best_m, best_oc, mList, distList, 0


class GWMergeTree:
    def __init__(self, tree: nx.Graph, root: int):
        """ Initialization of a merge tree object

        Parameters
        ----------
        tree:  networkx.Graph
               the merge tree and its data
        root:  int
               the id of the root of the merge tree
        """
        self.tree = tree
        self.root = root

        try:
            assert nx.is_tree(self.tree)
            assert self.root in self.tree.nodes()
        except AssertionError:
            print("The tree data is incorrect! "
                  "Either the object is not a tree, or the root is not a valid point in the tree.")
            raise ValueError

    def label_validation(self, coordinate_labels: list, scalar_name="height", edge_weight_name="weight"):
        try:
            for node in self.tree.nodes():
                assert scalar_name in self.tree.nodes[node]
                for label in coordinate_labels:
                    assert label in self.tree.nodes[node]
        except AssertionError:
            print("The tree data is incorrect! "
                  "Either the scalar function name is not valid, or the label names for nodes are not valid.")
            raise KeyError

        try:
            for node in self.tree.nodes():
                assert "type" in self.tree.nodes[node]
        except AssertionError:
            print("You have to specify the node type (0: minima, 1: saddle, 2: maxima) for nodes in the tree!")
            raise KeyError

        try:
            for u, vs in self.tree.adjacency():
                for v, e in vs.items():
                    assert edge_weight_name in e
        except AssertionError:
            print("The edge weight name is incorrect!")
            raise KeyError


class TrackInfo:
    def __init__(self, t: int, node_id: int, color: int):
        """ Initialization of a TrackInfo object

        Parameters
        ----------
        t:       int
                 the time step of the node in the trajectory
        node_id: int
                 the node id of the node in the merge tree at time step t
        color:   int
                 the trajectory ID, which may not be consecutive
        """

        self.time = t
        self.node_id = node_id
        self.trajectory_id = color

    def __str__(self):
        return "T_{} - Node_{}".format(self.time, self.node_id)


class GWTracking:
    def __init__(self,
                 trees: list,
                 scale: float,
                 labels: list,
                 scalar_name="height",
                 edge_weight_name="weight",
                 weight_mode="shortestpath",
                 prob_distribution="uniform",
                 tracking_maxima_only=True,
                 maxima_only=False,
                 fully_initialized=False,
                 scalar_threshold=None,
                 **kwargs_initialize_step
                 ):
        """ Initialization for the pFGW feature tracking framework

            Parameters
            ----------
            trees : list[GWMergeTree],
                    a list of GWMergeTree for feature tracking
            scale : float,
                    the scale for coordinates of nodes. This term is used for normalization.
                    e.g. if the scalar field domain is a 2D rectangle, then scale=max(width,height) of the domain
            labels: list[str],
                    list for names of the coordinate dimensions of nodes in the GWMergeTree object
                    e.g. ["x", "y", "z"] identifying the coordinate of a 3D point
            scalar_name: str, default="height"
                         the name for the scalar field in GWMergeTree
            edge_weight_name: str, default="weight"
                         the name for the weight of edges in GWMergeTree
            weight_mode : str, default="shortestpath"
                          declare the strategy to generate the weight matrix W for the measure network.
                          Options are ["shortestpath", "lca"]
            prob_distribution: str, default="uniform"
                               declare the strategy to assign probability vector p to nodes
                               Options are ["uniform", "ancestor", "watershed_segmentation", "other_segmentation"]
            tracking_maxima_only: bool, default=True. 
                                  Setting this to True will only track local maxima, which is highly recommended for split trees.
                                  It needs minor editing to support tracking minima only.
            maxima_only: bool, default=False.
                         This is task dependent. Setting it to True will remove all saddles and local minima 
                         from the measure network, which can be beneficial to the computational efficiency.
            **kwargs_initialize_step: additional parameters passed to initialize measure networks
                                      available entries: binary_maps, scalar_fields

            References
            ----------
            .. [1]
            """

        self.trees = [x.tree for x in trees]
        self.roots = [x.root for x in trees]
        
        self.tracking_maxima_only = tracking_maxima_only
        self.maxima_only = maxima_only

        for tree in trees:
            tree.label_validation(labels, scalar_name, edge_weight_name)

        self.weight_mode = weight_mode
        if self.weight_mode not in {"shortestpath", "lca", "lca-threshold"}:
            print("Weight matrix mode undefined! Use function value difference by default.")
            self.weight_mode = "shortestpath"

        self.prob_distribution = prob_distribution
        if self.prob_distribution not in {"uniform", "ancestor", "watershed_segmentation"}:
            print("Probability Distribution of nodes has to be \'uniform\' or \'ancestor\' or \'watershed_segmentation\'! "
                  "Using uniform distribution by default.")
            self.prob_distribution = "uniform"

        self.scalar_name = scalar_name
        self.edge_weight_name = edge_weight_name

        self.labels = labels
        self.scale = scale

        self.num_trees = len(self.trees)
        self.CList = []
        self.pList = []
        self.hList = []
        self.initializedLst = []
        self.all_scales = None
        self.scalar_threshold = scalar_threshold
        self.scalar_fields = self.binary_maps = None
        
        if "scalar_fields" in kwargs_initialize_step and ("binary_maps" in kwargs_initialize_step):
            self.scalar_fields = kwargs_initialize_step["scalar_fields"]
            self.binary_maps = kwargs_initialize_step["binary_maps"]
        if fully_initialized:
            self.initialize_measure_network()
        else:
            self.simple_initialize_measure_network()
    
    def simple_initialize_measure_network(self):
        print("WARNING! We do not compute the scale for all time steps.")
        print("Please provide the scale for normalization when intialize steps.")
        for e in range(self.num_trees):
            self.CList.append(None)
            self.pList.append(None)
            self.hList.append(None)
            self.initializedLst.append(False)
            
    def set_all_scales(self, alls):
        assert len(alls) == len(self.labels) + 1
        self.all_scales = alls
            
    def initialize_step(self, timestep):
        if self.all_scales is None:
            raise ValueError("Need to set self.all_scales beforehand! Please use self.set_all_scales() first.")
        if self.initializedLst[timestep]:
            print("Timestep {} is already initialized! Skipping...".format(str(timestep)))
            return False
        T = self.trees[timestep]
        C, p = get_distance_and_distribution(T,
                                             distribution=self.prob_distribution,
                                             weight_mode=self.weight_mode,
                                             root=self.roots[timestep],
                                             scalar_name=self.scalar_name,
                                             edge_weight_name=self.edge_weight_name,
                                             volume_name="volume",
                                             scalar_threshold=self.scalar_threshold,
                                             scalar_field=None if self.scalar_fields is None else self.scalar_fields[timestep],
                                             binary_map=None if self.binary_maps is None else self.binary_maps[timestep])
        if self.maxima_only:
            maxima_ids = np.asarray([node for node in T.nodes() if T.nodes[node]["type"] == 2], dtype=int)
            C = C[maxima_ids, maxima_ids]
            p = p[maxima_ids]
            p /= np.sum(p)
        
        self.CList[timestep] = C / self.all_scales[-1]
        print("CList shape:", self.CList[timestep].shape)
        self.pList[timestep] = p
        
        self.hList[timestep] = coordinate_matrix(T, self.labels)
        if self.maxima_only:
            maxima_ids = np.asarray([node for node in T.nodes() if T.nodes[node]["type"] == 2], dtype=int)
            self.hList[timestep] = self.hList[timestep][maxima_ids, :]
            
        for i in range(len(self.hList[timestep][0])):
            self.hList[timestep][:, i] /= self.all_scales[i]
        self.initializedLst[timestep] = True
        print("Finished intializing timestep", timestep)
        return True
        
    def release_step(self, timestep):
        if not self.initializedLst[timestep]:
            print("Timestep {} is not initialized! But it's free to release it again so let's do it.".format(str(timestep)))
        self.initializedLst[timestep] = False
        self.CList[timestep] = None
        self.pList[timestep] = None
        self.hList[timestep] = None
        print("Finished releasing timetep", timestep)

    def initialize_measure_network(self):
        # Initialize the measure network intrinsic weight & probability distribution
        for e in range(self.num_trees):
            T = self.trees[e]

            # C refers to the measure network intrinsic weight matrix. shape: (N, N)
            # p refers to the probability distribution across nodes. shape: (N, )
            C, p = get_distance_and_distribution(T,
                                                 distribution=self.prob_distribution,
                                                 weight_mode=self.weight_mode,
                                                 maxima_only=self.maxima_only,
                                                 root=self.roots[e],
                                                 scalar_name=self.scalar_name,
                                                 edge_weight_name=self.edge_weight_name,
                                                 volume_name="volume",
                                                 scalar_field=None if self.scalar_fields is None else self.scalar_fields[e],
                                                 binary_map=None if self.binary_maps is None else self.binary_maps[e]
                                                 )
            h = coordinate_matrix(T, self.labels)
            
            if self.maxima_only:
                maxima_ids = np.asarray([node for node in T.nodes() if T.nodes[node]["type"] == 2], dtype=int)
                C = C[maxima_ids, maxima_ids]
                p = p[maxima_ids]
                p /= np.sum(p)
                h = h[maxima_ids, :]

            self.CList.append(C)
            self.pList.append(p)

            # h refers to the labels for nodes (extrinsic information). shape: (N, label_len)
            self.hList.append(h)
            self.initializedLst.append(True)

        # rescale both extrinsic and intrinsic information
        scale = scale_normalization(self.hList, self.CList, self.scale, is_type_last="type" in self.labels)
        for i in range(len(self.hList)):
            for j in range(len(self.hList[0][0])):
                self.hList[i][:, j] /= scale[j]
            self.CList[i] /= scale[-1]

    def fgw_wrapper(self, C1, C2, p, q, g1, g2, alpha, m, amijo=True, G0=None, metric='penaltyl2', M=None, max_dist=None):
        """ A wrapper that returns the coupling matrix for the partial OT problem

        Parameters
        ----------
        C1 : ndarray, shape (ns, ns)
             Metric cost matrix in the source space
        C2 : ndarray, shape (nt, nt)
             Metric cost matrix in the target space
        p : ndarray, shape (ns,)
             marginal probability restriction in the source space
        q : ndarray, shape (nt,)
             marginal probability restriction in the target space
        g1: ndarray, shape (ns, ds)
             labels for nodes in the source space, where "ds" is the dimension of labels for nodes in the source space
             NOTE: g1 can be None if M is provided
        g2: ndarray, shape (nt, dt)
             labels for nodes in the target space, where "dt" is the dimension of labels for nodes in the target space
             NOTE: g2 can be None if M is provided
        alpha: float, in range [0, 1]
             balancing parameter to control the ratio between Wasserstein and GW proportion
             alpha=0: pure Wasserstein distance
             alpha=1: pure GW distance
        m: float, in range (0, 1]
             probability mass to be preserved in the output coupling matrix
             m=1: FGW distance
        amijo: bool, optional
             If True the steps of the line-search is found via an amijo research. Else closed form is used.
             If there is convergence issues use False.
        G0: ndarray, shape (ns, nt), optional
             Initial state for the coupling matrix for gradient descent
             if G0 is None, it is set to be p[:,None]*q[None,:]
        M: ndarray, shape (ns, nt), optional
             The pairwise label distance between nodes from the source vs. target
             If M is None, it is computed as the pairwise label distance between g1 and g2
        max_dist: float, in range [0, 1], optional (default: None)
             The max Euclidean distance allowed between matched critical points.
             Any Euclidean distance above this threshold will be set to a large value (e.g., 10) to prevent mismatchings.
             Default is None (not applied).

        Returns
        -------
        C: ndarray, shape (ns, nt)
             the coupling matrix for the solution to compute pFGW distance
        """

        sizeP = C1.shape[0]
        sizeQ = C2.shape[0]
        
        if M is None:
            assert g1 is not None
            assert g2 is not None
            
            # loss should be squared
            M = np.zeros((sizeP, sizeQ))
            for i in range(sizeP):
                for j in range(sizeQ):
                    M[i, j] = label_distance(g1[i], g2[j], metric=metric) ** 2
        else:
            assert M.shape == (sizeP, sizeQ)
        
        # After testing, this trick is not needed.
        # if max_dist is not None:
            # NOTE: because we compute 2-pFGW distance, the cost matrix M is the squared Euclidean distance
            # M[M > (max_dist ** 2)] = max(2, 2 * np.max(M))

        if m < 1:
            try:
                return fgw_partial_lp((1 - alpha) * M, C1, C2, p, q, m=m, alpha=alpha, amijo=amijo, G0=G0)
            except (ValueError, NonConvergenceError):
                print("Fail to converge. Turning off amijo search. Using closed form.")
                return fgw_partial_lp((1 - alpha) * M, C1, C2, p, q, m=m, alpha=alpha, amijo=False, G0=G0)
        try:
            return fgw_lp((1 - alpha) * M, C1, C2, p, q, loss_fun='square_loss', alpha=alpha, amijo=amijo, G0=G0)
        except (ValueError, NonConvergenceError):
            print("Fail to converge. Turning off amijo search. Using closed form.")
            return fgw_lp((1 - alpha) * M, C1, C2, p, q, loss_fun='square_loss', alpha=alpha, amijo=False, G0=G0)

    def crit_type_fail(self, node):
        if (self.tracking_maxima_only and node['type'] != 2) or \
           (not self.tracking_maxima_only and node['type'] == 1):
            return True
        return False


    # we assume (ambitiously, but usually safe when alpha <= 0.4)
    def adaptive_m_tuning_binary_search(self,
                          timesteps: list,
                          alpha: float,
                          matched_dist_limit: float,
                          amijo=False,
                          m_tuning_range=np.arange(0.5, 1+1e-6, 0.01),
                          max_workers=5,
                          metric="penaltyl2",
                          prob_rubric='max',
                          Ms=None,
                          benchmark=False):
        if alpha > 0.2:
            if alpha > 0.5:
                print("Warning! Using high alpha value may make the matched distance not monotonous to m!")
                print("Too high alpha value! Quitting the function." )
                return None, None, None, None, None, None
        while timesteps[-1] >= self.num_trees:
            timesteps.pop(-1)
        if len(timesteps) < 2:
            print("No enough element to track. Aborting...")
            return None, None, None, None, None, None
        
        max_ms = []
        if self.binary_maps is not None:
            for i in range(len(timesteps) - 1):
                step_i = timesteps[i]
                step_j = timesteps[i + 1]
                space_i = np.sum(self.binary_maps[step_i])
                space_j = np.sum(self.binary_maps[step_j])
                if space_i < space_j:
                    max_ms.append(space_i / space_j)
                else:
                    max_ms.append(space_j / space_i)
            print("Possible max m according to cloud area:", max_ms)
        else:
            max_ms = [np.max(m_tuning_range)] * (len(timesteps) - 1)

        # Let's rework the tracking style
        # We (thread-)parallelize over the time steps
        # Because parallelizing binary-search can only yield to 2 threads in parallel
        cc_m_tuning_binary = partial(concurrent_m_binary_search, 
                                     max_m=np.max(m_tuning_range),
                                     min_m=np.min(m_tuning_range),
                                     max_dist=matched_dist_limit, 
                                     alpha=alpha, 
                                     amijo=amijo, 
                                     fgw_wrapper=self.fgw_wrapper,
                                     metric=metric,
                                     prob_rubric=prob_rubric,
                                     benchmark=benchmark)

        cc_initialize_step = partial(concurrent_initialize_steps,
                                     pfgw=self,
                                     benchmark=benchmark)
        
        best_ms = [None for i in timesteps[:-1]]
        best_ocs = [None for i in timesteps[:-1]]
        m_searchspace = [None for i in timesteps[:-1]]
        dist_values = [None for i in timesteps[:-1]]
        
        init_runtimes = [0 for i in timesteps]
        pFGW_runtimes = [0 for i in timesteps[:-1]]
        
        if max_workers > 1:
            # remaining parameters: CList1, CList2, pList1, pList2, hList1, hList2
            for parallel_start in range(0, len(timesteps) - 1, max_workers):
                parallel_end = parallel_start + max_workers + 1
                if parallel_end > len(timesteps):
                    parallel_end = len(timesteps)
                    
                prange = [timesteps[idx] for idx in range(parallel_start, parallel_end)]
                if len(prange) <= 1:
                    continue
                print("working from timestep {} to {}...".format(str(prange[0]), str(prange[-1])))
                
                # it's weird that the edit made in cc_initialize_step is not reflected in "self"
                with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                    for tmp, (C, p, h, runtime) in enumerate(pool.map(cc_initialize_step, prange)):
                        timestep = tmp + parallel_start
                        self.CList[timestep] = C
                        self.pList[timestep] = p
                        self.hList[timestep] = h
                        init_runtimes[timestep] = runtime
                        self.initializedLst[timestep] = True
                        print("Finished intializing timestep", timestep, ", CList shape:", C.shape)
                    
                CList_prev = [self.CList[i] for i in prange[:-1]]
                CList_next = [self.CList[i] for i in prange[1:]]
                pList_prev = [self.pList[i] for i in prange[:-1]]
                pList_next = [self.pList[i] for i in prange[1:]]
                hList_prev = [self.hList[i] for i in prange[:-1]]
                hList_next = [self.hList[i] for i in prange[1:]]
                Ms_sub = [None] * (len(prange) - 1) if Ms is None else [Ms[i] for i in prange[:-1]]
                
                with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                    for tmp, (best_m, best_oc, mList, distList, runtime) in enumerate(pool.map(cc_m_tuning_binary, 
                                                                                        CList_prev, 
                                                                                        CList_next, 
                                                                                        pList_prev, 
                                                                                        pList_next, 
                                                                                        hList_prev, 
                                                                                        hList_next,
                                                                                        Ms_sub)):
                        timestep = tmp + parallel_start
                        best_ms[timestep] = best_m
                        best_ocs[timestep] = best_oc
                        m_searchspace[timestep] = mList
                        dist_values[timestep] = distList
                        pFGW_runtimes[timestep] = runtime
                        print("The optimal m for timestep {} vs. {} is {}.".format(str(timesteps[timestep]), str(timesteps[timestep + 1]), best_m))
                        print("Max Matched Dist list:", distList)
                
                for idx in prange[:-1]:
                    # Can be changed depending on the max memory you have
                    if len(self.CList[idx]) ** 2 * len(timesteps) * 30 >= 1e8:
                        self.release_step(idx)
        else:
            for timestep in range(len(timesteps) - 1):
                id1 = timesteps[timestep]
                id2 = timesteps[timestep + 1]
                if not self.initializedLst[id1]:
                    start_time = time.perf_counter()
                    self.initialize_step(id1)
                    init_runtimes[id1] = time.perf_counter() - start_time
                if not self.initializedLst[id2]:
                    start_time = time.perf_counter()
                    self.initialize_step(id2)
                    init_runtimes[id2] = time.perf_counter() - start_time
                
                if benchmark:
                    print(f"Init Runtime #{id1}", init_runtimes[id1])
                    print(f"Init Runtime #{id2}", init_runtimes[id2])
                    
                best_m, best_oc, mList, distList, pfgw_time = cc_m_tuning_binary(self.CList[id1], self.CList[id2], 
                                                                      self.pList[id1], self.pList[id2],  
                                                                      self.hList[id1], self.hList[id2], None)
                best_ms[timestep] = best_m
                best_ocs[timestep] = best_oc # self.filtered_oc(timesteps[timestep], timesteps[timestep + 1], best_oc)
                m_searchspace[timestep] = mList
                dist_values[timestep] = distList
                pFGW_runtimes[timestep] = pfgw_time
                
                if benchmark:
                    print(f"pFGW runtime #{timestep}", pfgw_time)
                print("The optimal m for timestep {} vs. {} is {}.".format(str(timesteps[timestep]), str(timesteps[timestep + 1]), best_m))
                if len(self.CList[id1]) ** 2 * len(timesteps) * 30 >= 1e8:
                    self.release_step(id1)
        
        if benchmark:
            print("INIT runtime:", init_runtimes)
            print("pFGW runtime:", pFGW_runtimes)
            return best_ms, best_ocs, m_searchspace, dist_values, np.mean(init_runtimes), np.mean(pFGW_runtimes)
        return best_ms, best_ocs, m_searchspace, dist_values, None, None
    
    
    def filtered_oc(self, id1, id2, oc):
        assert len(self.trees[id1]) == oc.shape[0]
        assert len(self.trees[id2]) == oc.shape[1]
        id1_maxima = []
        id2_maxima = []
        for node in self.trees[id1]:
            if self.crit_type_fail(self.trees[id1].nodes[node]):
                continue
            id1_maxima.append(node)
        for node in self.trees[id2]:
            if self.crit_type_fail(self.trees[id2].nodes[node]):
                continue
            id2_maxima.append(node)
        id1_maxima = np.asarray(id1_maxima, dtype=int)
        id2_maxima = np.asarray(id2_maxima, dtype=int)
        oc_maxima = oc[id1_maxima]
        oc_maxima = oc_maxima[:, id2_maxima]
        return oc_maxima
    
    # WARNING: exist_ocs could be the filtered OC matrix
    def output_tracking(self, result_path, id_list=None, exist_ocs=None):
        while id_list[-1] >= self.num_trees:
            id_list.pop(-1)

        if len(id_list) < 2:
            print("No enough element to track. Exiting...")
            return

        self.color_lengths = {}
        self.accumulated = {}

        id1 = id_list[0]
        self.trees[id1] = graph_color(self.trees[id1])
        
        start_time = time.time()
        color_accumulation = {}
        
        for node in self.trees[id1]:
            color_val = self.trees[id1].nodes[node]["color_value"]
            if color_val not in color_accumulation:
                color_accumulation[color_val] = 1
            else:
                color_accumulation[color_val] += 1
        
        is_filtered = None
        filtered_ocs = []

        for i in range(1, len(id_list)):
            id1 = id_list[i - 1]
            id2 = id_list[i]
                
            print("OC load from file record between {} and {}...".format(str(id1), str(id2)))
            oc = exist_ocs[i - 1]
            
            # if we have removed all non-maxima critical points, we need to verify the OC matrix shape
            if self.maxima_only and (self.trees[id1].number_of_nodes() > oc.shape[0]):
                if is_filtered is not None and (not is_filtered):
                    raise ValueError("Mixture of filtered and unfiltered OCs is not supported!")
                is_filtered = True 
                maxima_ids = np.asarray([node for node in self.trees[id1].nodes() if self.trees[id1].nodes[node]["type"] == 2], dtype=int)
                assert len(maxima_ids) == oc.shape[0]
            else:
                if is_filtered:
                    raise ValueError("Mixture of filtered and unfiltered OCs is not supported!")
                is_filtered = False 
                if self.trees[id1].number_of_nodes() != oc.shape[0]:
                    raise ValueError("oc shape incorrect! Node size = {} vs. oc shape = {}".format(
                        str(self.trees[id1].number_of_nodes()), str(oc.shape[0])))
            
            if self.trees[id2].number_of_nodes() > oc.shape[1]:
                if is_filtered is not None and (not is_filtered):
                    raise ValueError("Mixture of filtered and unfiltered OCs is not supported!")
                is_filtered = True
                maxima_ids = np.asarray([node for node in self.trees[id2].nodes() if self.trees[id2].nodes[node]["type"] == 2], dtype=int)
                assert len(maxima_ids) == oc.shape[1]
            else:
                if is_filtered:
                    raise ValueError("Mixture of filtered and unfiltered OCs is not supported!")
                is_filtered = False
                if self.trees[id2].number_of_nodes() != oc.shape[1]:
                    raise ValueError("oc shape incorrect! Node size = {} vs. oc shape = {}".format(
                        str(self.trees[id2].number_of_nodes()), str(oc.shape[1])))
                
            # transferring the color from id1 to id2 based on OC
            self.trees[id2] = graph_color(self.trees[id2], self.trees[id1], oc, maxima_only=is_filtered)
            
            for node in self.trees[id2]:
                if "color_value" not in self.trees[id2].nodes[node]:
                    continue
                color_val = self.trees[id2].nodes[node]["color_value"]
                if color_val not in color_accumulation:
                    color_accumulation[color_val] = 1
                else:
                    color_accumulation[color_val] += 1
            
            # We filter out coupling information unrelated to maxima
            # Note: the filtered coupling matrix does not sum to m
            oc_maxima = oc 
            filtered_ocs.append(oc_maxima)

        for idx in id_list:
            for node in self.trees[idx].nodes():
                if "color_value" not in self.trees[idx].nodes[node]:
                    continue
                self.trees[idx].nodes[node]["color_length"] = color_accumulation[self.trees[idx].nodes[node]["color_value"]]

        for i in range(len(id_list)):
            idx = id_list[i]
            self.output_instance(idx, result_path)

        end_time = time.time()
        print("Total time spent: ", end_time - start_time)
        return filtered_ocs

    def output_instance(self, id, path: str):
        if id >= self.num_trees:
            print("ID out of boundary")
            return
        
        tree_filename = os.path.join(path, "treeNode_highlight_" + str(id).zfill(3) + ".txt")
        tree_file = open(tree_filename, "w")

        T = self.trees[id]
        for node in T.nodes():
            ff = tree_file
            if self.crit_type_fail(T.nodes[node]):
                continue
            print(T.nodes[node]["x"],
                  T.nodes[node]["y"],
                  T.nodes[node]["z"],
                  T.nodes[node]["height"],
                  T.nodes[node]["type"],
                  T.nodes[node]["color_value"],
                  T.nodes[node]["color_length"],
                  T.nodes[node]["mark"],
                  file=ff)
            
        tree_file.close()