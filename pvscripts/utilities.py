import sys
import os
import re
import numpy as np
from matplotlib import pyplot as plt
import csv

class VirtualEnv:
    setup = False

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

# default supported types: [".npy", ".txt", ".png"]
# returns FULL path of the input files
def getInputFiles(data_dir, supported_types=[".npy", ".txt", ".png"]):
    files = os.listdir(data_dir)
    fileList = []
    for file in files:
        isSupported = False
        for tp in supported_types:
            if file.endswith(tp):
                isSupported = True
                break
        if isSupported:
            fileList.append(file)

    if '.DS_Store' in fileList:
        fileList.remove('.DS_Store')
        
    try:
        order = []
        for i in range(len(fileList)):
            order.append(int(re.search(r'\d+', fileList[i]).group()))
        fileList = [os.path.join(data_dir, x) for _,x in sorted(zip(order,fileList))]
    except:
        fileList = [os.path.join(data_dir, x) for x in fileList]
        print("Cannot get proper order of the input files. Return fileList without sorting...")

    return fileList


def load2mat(inFile: str):
    if inFile.endswith("txt"):
        return np.loadtxt(inFile)
    elif inFile.endswith("npy"):
        return np.load(inFile, allow_pickle=True)
    elif inFile.endswith("png"):
        if not VirtualEnv.setup:
            VirtualEnv.setup = True
            virtualEnv = 'venv/Scripts/activate_this.py'
            exec(open(virtualEnv).read(), {'__file__': virtualEnv})
        print("Loading png files.")
        from PIL import Image
        image = Image.open(inFile)
        return np.array(image)
    else:
        print("Unsupported file format! Exit...")
        return None
    

def mat2png(mat, outFile: str):
    if not VirtualEnv.setup:
        VirtualEnv.setup = True
        virtualEnv = 'venv/Scripts/activate_this.py'
        exec(open(virtualEnv).read(), {'__file__': virtualEnv})
    from PIL import Image
    im = Image.fromarray(mat)
    im.save(outFile)


def getIdsFromVTKList(l):
    lcnt = l.GetNumberOfIds()
    ids = []
    for i in range(lcnt):
        ids.append(l.GetId(i))
    return ids

def createPersistenceCurvePlot(pc_path, pc_type, thres=None, isThresLocal=False, fontsize=14):
    files_pc_path = os.listdir(pc_path)
    pc_files = []
    for file in files_pc_path:
        if file.endswith(".csv"):
            pc_files.append(file)
            
    max_persistence = 0
    max_persistences = []
    for pc_file in pc_files:
        numPairs = []
        persistences = []
        csvFile = os.path.join(pc_path, pc_file)
        with open(csvFile, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                persistences.append(float(row[0]))
                numPairs.append(int(row[1]))
                
        max_persistences.append(max(persistences))
        if max(persistences) > max_persistence:
            max_persistence = max(persistences)

        plt.plot(persistences, numPairs, linewidth=1)
    
    plt.xlabel('p', fontsize=fontsize)
    if pc_type == 'saddle-max':
        plt.ylabel('Maxima Count', fontsize=fontsize)
    elif pc_type == 'min-saddle':
        plt.ylabel('Minima Count')
    else:
        plt.ylabel('Extrema Count')
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    
    xtck = list(np.arange(0,1,0.1) * max_persistence)
    xtck_str = []
    for tck in xtck:
        xtck_str.append("."+str(int(round(tck * 100 / max_persistence))).zfill(2))

    plt.xticks(xtck, xtck_str, fontsize=14)
    plt.yticks(fontsize=14)
    if thres is not None:
        print("Threshold: ", thres)
        if isThresLocal:
            for pers in max_persistences:
                plt.axvline(x=thres * pers / 100, color='k', linestyle='--', linewidth=0.8)
        else:
            plt.axvline(x=thres * max_persistence / 100, color='k', linestyle='--', linewidth=0.8)
        plt.xlim(-max_persistence * 0.02, max_persistence * 1.02)

    plot_path = os.path.join(pc_path, "plot")
    make_dir(plot_path)
    if thres is None:
        outFile = os.path.join(plot_path, "PersistenceCurve.png")
    else:
        if isThresLocal:
            outFile = os.path.join(plot_path, "PersistenceCurve_localthres_{}p.png".format(str(thres)))
        else:
            outFile = os.path.join(plot_path, "PersistenceCurve_globalthres_{}p.png".format(str(thres)))
    plt.savefig(outFile)
    if thres is None:
        plt.show()
    plt.close()

    return max_persistence


def detectMaxPersistence(pc_path):
    files_pc_path = os.listdir(pc_path)
    pc_files = []
    for file in files_pc_path:
        if file.endswith(".csv"):
            pc_files.append(file)
    
    if len(pc_files) < 1:
        print("No persistence curve file! Getting back...")
        return False
    return True
    

def loadMaxPersistence(pc_path):
    files_pc_path = os.listdir(pc_path)
    pc_files = []
    for file in files_pc_path:
        if file.endswith(".csv"):
            pc_files.append(file)
    
    if len(pc_files) < 1:
        print("No persistence curve file! Getting back...")
        return None
            
    max_persistences = []
    for pc_file in pc_files:
        numPairs = []
        persistences = []
        csvFile = os.path.join(pc_path, pc_file)
        with open(csvFile, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                persistences.append(float(row[0]))
                numPairs.append(int(row[1]))
                
        max_persistences.append(max(persistences))
    return max_persistences
