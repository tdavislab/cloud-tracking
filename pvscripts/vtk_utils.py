from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
import numpy as np
import os


def extract_node_data(fieldsFileName, node_list, tolerance=1e-8):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fieldsFileName)
    reader.Update()
    ugrid = reader.GetOutput()

    all_points = vtk_to_numpy(ugrid.GetPoints().GetData())
    point_data = ugrid.GetPointData()

    matched_indices = []
    matched_nodes = []

    for node in node_list:
        distances = np.linalg.norm(all_points - node, axis=1)
        close_index = np.argmin(distances)

        if distances[close_index] < tolerance:
            matched_indices.append(close_index)
            matched_nodes.append(node)
        else:
            print(f"⚠️ No match for node: {node}")

    matched_indices = np.array(matched_indices)
    matched_nodes = np.array(matched_nodes)

    extracted_data = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        name = array.GetName()
        if name:
            data_np = vtk_to_numpy(array)
            extracted_data[name] = data_np[matched_indices]

    return extracted_data

    
def convertMergeTreesToTxt(data_file: str, tree_path: str, noSegmentation = False, *args):
    file_basename = os.path.basename(data_file)
    filename, _ = os.path.splitext(file_basename)
    vtk_subpath = os.path.join(tree_path, "vtk")
    
    critsFileName = os.path.join(vtk_subpath, "crits_" + filename + ".vtu")
    edgesFileName = os.path.join(vtk_subpath, "edges_" + filename + ".vtu")
    fieldsFileName = os.path.join(vtk_subpath, "fields_" + filename + ".vtu")
    
    critText_name = os.path.join(tree_path, "treeNodes_" + filename + ".txt")
    edgesText_name = os.path.join(tree_path, "treeEdges_" + filename + ".txt")
    
    if not noSegmentation:
        assert os.path.exists(fieldsFileName)
        segText_name = os.path.join(tree_path, "segmentation_" + filename + ".npy")
    
    # read edge file first
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(edgesFileName)
    reader.Update() 

    # extract data from Tree
    Tree = reader.GetOutput()
    nodes = vtk_to_numpy(Tree.GetPoints().GetData())
    edges = vtk_to_numpy(Tree.GetCells().GetData())
    scalar =  vtk_to_numpy(Tree.GetPointData().GetArray("Scalar"))
    
    array_len = edges[0]
    links = []
    ecnt = int(len(edges)/(array_len+1))
    for j in range(ecnt):
        links.append([edges[j*(array_len+1)+1], edges[j*(array_len+1)+2]])
    links = np.array(links)

    nodeReader = vtk.vtkXMLUnstructuredGridReader()
    nodeReader.SetFileName(critsFileName)
    nodeReader.Update()

    critsFile = nodeReader.GetOutput()
    crits = vtk_to_numpy(critsFile.GetPoints().GetData())
    critsType =  vtk_to_numpy(critsFile.GetPointData().GetArray("CriticalType"))

    nodeList = nodes.tolist()
    critsList = crits.tolist()

    nodeType = []
    for i in range(len(nodes)):
        nodeType.append(critsType[critsList.index(nodeList[i])])
    
    nodeType = np.array(nodeType)
    nodes = np.concatenate((nodes, np.array([scalar]).T), axis=1)
    nodes = np.concatenate((nodes, np.array([nodeType]).T), axis=1)
    nodes = nodes.astype(float)
    
    if len(args) > 0:
        infofields = extract_node_data(fieldsFileName, nodeList)
        for arg in args:
            if arg in infofields:
                nodes = np.concatenate((nodes, np.array([infofields[arg]]).T), axis=1)
                print(f"Added {arg} to the node information list.")
            else:
                print(f"{arg} is not available in critical point information.")

    # if os.path.exists(treeSeg):
        # print(treeSeg)
        # nodes = inRegion(nodes, treeSeg)

    np.savetxt(critText_name, nodes, fmt='%.6f')
    np.savetxt(edgesText_name, links, fmt='%d')
    if not noSegmentation:
        from vtx2mat import extract_scalar_from_vtx
        segmentation, _, _, _ = extract_scalar_from_vtx(fieldsFileName, "SegmentationId")
        np.save(segText_name, segmentation, allow_pickle=True)
    

def convertMorseComplexToTxt(data_file: str, graph_path: str, td_type: str, noSegmentation = True):
    file_basename = os.path.basename(data_file)
    filename, _ = os.path.splitext(file_basename)
    vtk_subpath = os.path.join(graph_path, "vtk")
    
    edgesFileName = os.path.join(vtk_subpath, "edges_" + filename + ".vtp")
    
    critText_name = os.path.join(graph_path, "treeNodes_" + filename + ".txt")
    edgesText_name = os.path.join(graph_path, "treeEdges_" + filename + ".txt")
    
    if not noSegmentation:
        print("Not involving segmentation for MC/MSC at the moment.")
        raise NotImplemented
        # assert os.path.exists(fieldsFileName)
        # segText_name = os.path.join(graph_path, "segmentation_" + filename + ".npy")
    
    # read edge file first
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(edgesFileName)
    reader.Update() 
    
    mGraph = reader.GetOutput()
    nodes = mGraph.GetPoints()
    nodes = vtk_to_numpy(nodes.GetData())
    masks = vtk_to_numpy(mGraph.GetPointData().GetArray("ttkMaskScalarField"))
    cellSeparatrixId = vtk_to_numpy(mGraph.GetCellData().GetArray("SeparatrixId"))
    cellSeparatrixType = vtk_to_numpy(mGraph.GetCellData().GetArray("SeparatrixType"))
    pointSeparatrixId = np.zeros((len(nodes), ), dtype=int)

    edges = []
    for i in range(mGraph.GetNumberOfCells()):
        vtkpoints = vtk.vtkIdList()
        mGraph.GetCellPoints(i, vtkpoints)
        points = []
        for j in range(vtkpoints.GetNumberOfIds()):
            idx = vtkpoints.GetId(j)
            points.append(idx)
            pointSeparatrixId[idx] = cellSeparatrixId[i]
        edges.append(points) 

    edges = np.asarray(edges)

    nodes = np.concatenate((nodes, masks.reshape(-1, 1)), axis=1)
    nodes = np.concatenate((nodes, pointSeparatrixId.reshape(-1, 1)), axis=1)

    edges = np.concatenate((edges, cellSeparatrixType.reshape(-1, 1)), axis=1)
    edges = np.concatenate((edges, cellSeparatrixId.reshape(-1, 1)), axis=1)

    np.savetxt(critText_name, nodes, fmt='%.6f')
    np.savetxt(edgesText_name, edges, fmt='%d')
