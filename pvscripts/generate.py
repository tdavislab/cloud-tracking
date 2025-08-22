from utilities import *
from vtk_utils import *
from ttk_utils import *
from matplotlib import pyplot as plt
import time
from paraview.servermanager import ProxyManager

if __name__ == '__main__':
    print("This code computes persistence curve (optional), then generates ST/JT/CT/ASC-MC/DESC-MC/MSC based on request.")
    print("For easier usage of the code, please specify the request in generate.config.")
    
    dataDirs = []
    tdTypes = []
    persThreshold = []
    isLocalPersThreshold = False
    scalarNames = []
    noSegmentation = False
    saveSimplifiedField = False
    benchmark = True
    
    print("Using info from generate.config...")
    if os.path.exists("generate.config"):
        with open("generate.config") as config_f:
            config_lines = config_f.readlines()
            isReadingDir = False
            isReadingType = False
            isReadingPersThres = False
            isReadingScalarNames = False
            for i in range(len(config_lines)):
                line = config_lines[i].strip().replace("\\", "/")
                if line.startswith("#") and "dir path for vtk files" in line:
                    isReadingDir = True
                    isReadingType= False
                    isReadingPersThres = False
                    isReadingScalarNames = False
                    continue
                elif line.startswith("#") and "topological descriptor type" in line:
                    isReadingDir = False
                    isReadingType = True
                    isReadingPersThres = False
                    isReadingScalarNames = False
                    continue
                elif line.startswith("#") and "persistence simplification level" in line:
                    isReadingDir = False
                    isReadingType = False
                    isReadingPersThres = True
                    isReadingScalarNames = False
                    continue
                elif line.startswith("#") and "scalar name" in line:
                    isReadingDir = False
                    isReadingType = False
                    isReadingPersThres = False
                    isReadingScalarNames = True
                    continue
                elif line.startswith("#") and "local persistence threshold" in line:
                    if "yes" in line.lower() or "true" in line.lower():
                        isLocalPersThreshold = True
                    isReadingDir = False
                    isReadingType = False
                    isReadingPersThres = False
                    isReadingScalarNames = False
                    continue
                elif line.startswith("#") and "simplified scalar field" in line:
                    if "yes" in line.lower() or "true" in line.lower():
                        saveSimplifiedField = True
                    isReadingDir = False
                    isReadingType = False
                    isReadingPersThres = False
                    isReadingScalarNames = False
                    continue
                elif line.startswith("#") and "segmentation" in line.lower():
                    info = line.split(":")[-1].lower()
                    if "no" in info or "false" in info:
                        noSegmentation = True
                    isReadingDir = False
                    isReadingType = False
                    isReadingPersThres = False
                    isReadingScalarNames = False
                    continue
                elif line.startswith("#"):
                    continue
                
                if isReadingDir:
                    assert not isReadingType
                    if len(line) > 0:
                        dataDirs.append(line)
                elif isReadingType:
                    if len(line) > 0:
                        tdTypes.append(line)
                elif isReadingPersThres:
                    if len(line) > 0:
                        persThreshold.append(float(line))
                elif isReadingScalarNames:
                    if len(line) > 0:
                        scalarNames.append(line)  
                else:
                    pass
    else:
        print("Please use generate.config to specify the command.")
        exit()
    
    # solve scalar name broadcast
    if len(scalarNames) < 1:
        print("Please specify the scalar name of the data.")
        exit()
    elif len(scalarNames) > 1:
        if len(scalarNames) != len(dataDirs):
            print("Please provide exactly one scalar name for each dataset.")
            exit()     
    elif len(dataDirs) > 1 and len(scalarNames) == 1:
        print("Broadcasting scalar names to all datasets...")
        scalarNames = [scalarNames[0]] * len(dataDirs)
    
    for dataDir, scalarName in zip(dataDirs, scalarNames):
        inputFiles = getInputFiles(dataDir, [".vts", ".vti", ".vtr", ".vtp", ".vtu", ".vtk"])
        for tdType in tdTypes:
            # let's create a folder to store the persistence curve results
            pc_root_path = os.path.join(dataDir, "PersistenceCurve")
            pc_type = PC_types[tdType.lower()]
            pc_path = os.path.join(pc_root_path, pc_type)
            make_dir(pc_path)
            
            existPersistenceCurve = detectMaxPersistence(pc_path)
            if len(persThreshold) == 0 or (not existPersistenceCurve and (len(persThreshold) > 1 or (persThreshold[0] != 0.0))):
                # compute the persistence curve
                for file in inputFiles:
                    file_basename = os.path.basename(file)
                    filename, ext = os.path.splitext(file_basename)
                    pc_filename = os.path.join(pc_path, "PC_" + filename + ".csv")
                    if os.path.exists(pc_filename):
                        continue
                    
                    if benchmark:
                        start_time_persistenceCurve = time.perf_counter()
                        
                    computePersistenceCurve(file, pc_filename, scalarName, pc_type)
                    
                    if benchmark:
                        end_time_persistenceCurve = time.perf_counter()
                        print("Persistence Curve runtime:", end_time_persistenceCurve - start_time_persistenceCurve)

                # draw and show / store the persistence curve
                max_persistence = createPersistenceCurvePlot(pc_path, pc_type, thres=None, isThresLocal=isLocalPersThreshold, fontsize=14)
                
                # fetch the input from users for desired persistence simplification
                print("Max persistence =", max_persistence)
                
                if len(persThreshold) == 0:
                    threshold = float(input("\nPlease enter percentage of persistence threshold: "))
                    threshold_str = str(threshold) + "percent"

                    threshold_val = threshold * max_persistence / 100
                    print("Max persistence =", max_persistence, ", max simplification level =", threshold_val)
                    
                    # draw and show / store the persistence curve again with proper persistence simplification
                    createPersistenceCurvePlot(pc_path, pc_type, thres=threshold, isThresLocal=isLocalPersThreshold, fontsize=14)
                    persThreshold.append(threshold)
                
            # note: each pThres is a percentage, not the absolute value
            for pThres in persThreshold:
                pThres_vals = []
                pc_root_path = os.path.join(dataDir, "PersistenceCurve")
                pc_type = PC_types[tdType.lower()]
                pc_path = os.path.join(pc_root_path, pc_type)
                
                if pThres != 0:
                    if isLocalPersThreshold:
                        pThres_vals = [pThres * x / 100 for x in loadMaxPersistence(pc_path)]
                    else:
                        pThres_vals = [pThres * max(loadMaxPersistence(pc_path)) / 100] * len(inputFiles)
                else:
                    pThres_vals = [pThres] * len(inputFiles)
                    
                # compute merge trees / contour trees
                if "st" in tdType.lower() or "jt" in tdType.lower() or "ct" in tdType.lower():
                    mt_output_root = os.path.join(dataDir, "_".join([tdType, str(pThres)]))
                    make_dir(mt_output_root)                
                    for file, pThres_val in zip(inputFiles, pThres_vals):
                        print("simplifying", file, "at persistence =", pThres_val)
                        
                        if benchmark:
                            start_time_mergeTree = time.perf_counter()
                        
                        computeMergeTrees(file, mt_output_root, scalarName, tdType, pThres_val, noSegmentation, saveSimplifiedField, pThres)
                        
                        if benchmark:
                            end_time_mergeTree = time.perf_counter()
                            print("Merge tree runtime:", end_time_mergeTree - start_time_mergeTree)
                        convertMergeTreesToTxt(file, mt_output_root, True, "latitude", "longitude", "lat_index", "lon_index")
                # compute Morse(-Smale) complexes
                # NOTICE: not involving segmentation for MC/MSC at the moment
                elif "mc" in tdType.lower() or "msc" in tdType.lower():
                    msc_output_root = os.path.join(dataDir, "_".join([tdType, str(pThres)]))
                    make_dir(msc_output_root)
                    for file, pThres_val in zip(inputFiles, pThres_vals):
                        print("simplifying", file, "at persistence =", pThres_val)
                        computeMorseComplex(file, msc_output_root, scalarName, tdType, pThres_val, True, saveSimplifiedField, pThres)
                        convertMorseComplexToTxt(file, msc_output_root, tdType, noSegmentation=True)
                else:
                    print("Unsupported topological descriptor type:", tdType)
                    
        # ResetSession()