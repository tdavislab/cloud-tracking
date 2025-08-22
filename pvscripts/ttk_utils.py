from paraview.simple import *
import os
import numpy as np
from utilities import make_dir
import gc

PC_types = {
    "st": "saddle-max",
    "jt": "min-saddle",
    "ct": "all",
    "asc-mc": "min-saddle",
    "desc-mc": "saddle-max",
    "msc": "all"
}

class LoadTTK:
    isLoaded = False
    ttkPluginPath = "C:/Program Files/ParaView 5.13.3/bin/paraview-5.13/plugins/TopologyToolKit/TopologyToolKit.dll"
    
if not LoadTTK.isLoaded:
    try:
        print("Loading TTK from", LoadTTK.ttkPluginPath)
        LoadPlugin(LoadTTK.ttkPluginPath, False, globals())
        LoadTTK.isLoaded = True
    except:
        print("Cannot load TTK from", LoadTTK.ttkPluginPath)
        print("Please make sure the path to TTK plugin is correct!")
        exit()
        
def computePersistenceCurve(data_file: str, output_file: str, scalar_name: str, pc_type: str):
    # load data file
    _, ext = os.path.splitext(data_file)
    if "vtp" in ext:
        reader = XMLPolyDataReader(FileName=[data_file])
    elif "vti" in ext:
        reader = XMLImageDataReader(FileName=[data_file])
    elif "vts" in ext:
        reader = XMLStructuredGridReader(FileName=[data_file])
    elif "vtr" in ext:
        reader = XMLRectilinearGridReader(FileName=[data_file])
    elif "vtu" in ext:
        reader = XMLUnstructuredGridReader(FileName=[data_file])
    elif "vtk" in ext:
        reader = LegacyVTKReader(FileNames=[data_file])
        # raise NotImplementedError
    
    # tetrahedralization
    tetra = Tetrahedralize(Input=reader)
    
    # pass to persistence curve
    persDiagram = TTKPersistenceDiagram(Input=tetra)
    persDiagram.ScalarField = ['POINTS', scalar_name]
    persDiagram.InputOffsetField = ['POINTS', scalar_name]
    
    persCurve = TTKPersistenceCurve(Input=persDiagram)
    # persCurve = TTKPersistenceCurve(Input=tetra)
    # persCurve.ScalarField = ['POINTS', scalar_name]
    # persCurve.InputOffsetField = ['POINTS', scalar_name]
    
    # save output as csv files
    if pc_type == 'saddle-max':
        SaveData(output_file, OutputPort(persCurve, 2))
    else:
        if pc_type == 'min-saddle':
            SaveData(output_file, OutputPort(persCurve, 0))
        else:
            SaveData(output_file, OutputPort(persCurve, 3))
    
    Delete(reader)
    Delete(tetra)
    Delete(persCurve)
    del reader
    del tetra
    del persCurve
    gc.collect()


def getSimplifiedTetrahedralizedField(field, scalar_name: str, td_type:str, simplification_level:float):
     # tetrahedralization
    tetra = Tetrahedralize(Input=field)
    
    # persistence simplification
    persSimplifiedTetra = TTKTopologicalSimplificationByPersistence(Input=tetra)
    persSimplifiedTetra.InputArray = ['POINTS', scalar_name]
    if td_type == "jt":
        persSimplifiedTetra.PairType = 'Minimum-Saddle'
    elif td_type == "st":
        persSimplifiedTetra.PairType = 'Maximum-Saddle'
    elif td_type == "ct":
        persSimplifiedTetra.PairType = 'Extremum-Saddle'
    elif td_type == "asc-mc":
        # persSimplifiedTetra.PairType = 'Minimum-Saddle'
        persSimplifiedTetra.PairType = 'Extremum-Saddle'
    elif td_type == "desc-mc":
        # persSimplifiedTetra.PairType = 'Maximum-Saddle'
        persSimplifiedTetra.PairType = 'Extremum-Saddle'
    elif td_type == "msc":
        persSimplifiedTetra.PairType = 'Extremum-Saddle'
    else:
        print("Unsupported topological descriptor type:", td_type)
        raise TypeError
    persSimplifiedTetra.PersistenceThreshold = simplification_level
    persSimplifiedTetra.ThresholdIsAbsolute = 1

    return persSimplifiedTetra
    
    
# Note: simplification level: absolute persistence simplification level, not percentage
def computeMergeTrees(data_file: str, output_path: str, scalar_name: str, td_type: str, simplification_level: float, noSegmentation = False, saveSimplifiedField = False, pThres_percent = None):
    # load data file
    file_basename = os.path.basename(data_file)
    filename, ext = os.path.splitext(file_basename)
    if "vtp" in ext:
        reader = XMLPolyDataReader(FileName=[data_file])
    elif "vti" in ext:
        reader = XMLImageDataReader(FileName=[data_file])
    elif "vts" in ext:
        reader = XMLStructuredGridReader(FileName=[data_file])
    elif "vtr" in ext:
        reader = XMLRectilinearGridReader(FileName=[data_file])
    elif "vtu" in ext:
        reader = XMLUnstructuredGridReader(FileName=[data_file])
    elif "vtk" in ext:
        reader = LegacyVTKReader(FileNames=[data_file])
    
    persSimplifiedTetra = getSimplifiedTetrahedralizedField(reader, scalar_name, td_type, simplification_level)
    if saveSimplifiedField:
        assert pThres_percent is not None
        simplified_data_path = data_file.replace(file_basename, "field_simplified_{}".format(str(pThres_percent)))
        make_dir(simplified_data_path)
        from vtx2mat import extract_scalar_from_vtx_filter
        sc, xlist, ylist, zlist = extract_scalar_from_vtx_filter(persSimplifiedTetra, scalar_name, "vtu")
        outFilename = os.path.join(simplified_data_path, filename + ".npy")
        if zlist is None:
            np.save(outFilename, sc, allow_pickle=True)
        else:
            np.save(outFilename, sc, allow_pickle=True)

    # compute merge tree / contour tree using FTM
    mergeTree = TTKMergeandContourTreeFTM(Input=persSimplifiedTetra)
    mergeTree.ScalarField = ['POINTS', scalar_name]
    mergeTree.InputOffsetField = ['POINTS', scalar_name]
    if td_type == "jt":
        mergeTree.TreeType = 'Join Tree'
    elif td_type == "st":
        mergeTree.TreeType = 'Split Tree'
    else:
        assert (td_type == "ct")
        mergeTree.TreeType = 'Contour Tree'
    
    # save output vtk files in a subfolder
    vtk_subpath = os.path.join(output_path, "vtk")
    make_dir(vtk_subpath)
    outputCritsFileName = os.path.join(vtk_subpath, "crits_" + filename + ".vtu")
    outputEdgesFileName = os.path.join(vtk_subpath, "edges_" + filename + ".vtu")
    outputFieldsFileName = os.path.join(vtk_subpath, "fields_" + filename + ".vtu")
    
    SaveData(outputCritsFileName, OutputPort(mergeTree, 0))
    SaveData(outputEdgesFileName, OutputPort(mergeTree, 1))
    if not noSegmentation:
        SaveData(outputFieldsFileName, OutputPort(mergeTree, 2))
    
    Delete(reader)
    Delete(persSimplifiedTetra)
    del reader
    del persSimplifiedTetra
    if saveSimplifiedField:
        del sc
        del zlist
    Delete(mergeTree)
    del mergeTree
    gc.collect()

# Note: simplification level: absolute persistence simplification level, not percentage
def computeMorseComplex(data_file: str, output_path: str, scalar_name: str, td_type: str, simplification_level: float, noSegmentation = True, saveSimplifiedField = False, pThres_percent = None):
    # load data file
    file_basename = os.path.basename(data_file)
    filename, ext = os.path.splitext(file_basename)
    if "vtp" in ext:
        reader = XMLPolyDataReader(FileName=[data_file])
    elif "vti" in ext:
        reader = XMLImageDataReader(FileName=[data_file])
    elif "vts" in ext:
        reader = XMLStructuredGridReader(FileName=[data_file])
    elif "vtr" in ext:
        reader = RectilinearGridReader(FileName=[data_file])
    elif "vtu" in ext:
        reader = UnstructuredGridReader(FileName=[data_file])
    elif "vtk" in ext:
        reader = LegacyVTKReader(FileNames=[data_file])
    
    persSimplifiedTetra = getSimplifiedTetrahedralizedField(reader, scalar_name, td_type, simplification_level)
    
    # compute Morse complexes
    msc = TTKMorseSmaleComplex(Input=persSimplifiedTetra)
    msc.ScalarField = ['POINTS', scalar_name]
    msc.OffsetField = ['POINTS', scalar_name]

    # simplifying minimum <==> ascending separatrices + descending segmentation
    # simplifying maximum <==> descending separatrices + ascending segmentation
    # simplifying extremum <==> all separatrices + Morse-Smale segmentation
    # always include saddle connectors, and set the simplification level identical to persistence simplification
    if td_type == "asc-mc":
        msc.Ascending1Separatrices = 1
        msc.Descending1Separatrices = 0
        msc.AscendingSegmentation = 0
        msc.DescendingSegmentation = 1 if not noSegmentation else 0
        msc.MorseSmaleComplexSegmentation = 0
        msc.ReturnSaddleConnectors = 1
        msc.SaddleConnectorsPersistenceThreshold = simplification_level
    elif td_type == "desc-mc":
        msc.Ascending1Separatrices = 0
        msc.Descending1Separatrices = 1
        msc.AscendingSegmentation = 1 if not noSegmentation else 0
        msc.DescendingSegmentation = 0
        msc.MorseSmaleComplexSegmentation = 0
        msc.ReturnSaddleConnectors = 1
        msc.SaddleConnectorsPersistenceThreshold = simplification_level
    else:
        if td_type != "msc":
            raise TypeError
        msc.Ascending1Separatrices = 1
        msc.Descending1Separatrices = 1
        msc.AscendingSegmentation = 1 if not noSegmentation else 0
        msc.DescendingSegmentation = 1 if not noSegmentation else 0
        msc.MorseSmaleComplexSegmentation = 1 if not noSegmentation else 0
        msc.ReturnSaddleConnectors = 1
        msc.SaddleConnectorsPersistenceThreshold = simplification_level
    
    vtk_subpath = os.path.join(output_path, "vtk")
    make_dir(vtk_subpath)
    outputCritsFileName = os.path.join(vtk_subpath, "crits_" + filename + ".vtp")
    outputEdgesFileName = os.path.join(vtk_subpath, "edges_" + filename + ".vtp")
    outputFieldsFileName = os.path.join(vtk_subpath, "fields_" + filename + ".vtu")
    
    SaveData(outputCritsFileName, OutputPort(msc, 0))
    SaveData(outputEdgesFileName, OutputPort(msc, 1))
    if not noSegmentation:
        SaveData(outputFieldsFileName, OutputPort(msc, 3))
    
    del reader
    del persSimplifiedTetra
    del msc
    gc.collect()
