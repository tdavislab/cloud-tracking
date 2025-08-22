import os
import re
import sys

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def getInputFiles(dataSet):
    fileList = []
    for root, dirs, files in os.walk("./postprocess-result/"):
        if not dataSet in root:
            continue
        for file in files:
            if file.endswith(".txt") and file.find("highlight") != -1:
                id = int(re.search(r'\d+', file).group())
                fileList.append([root, file, id])

    return fileList


def createVisFile(p, outFile):
    if len(p) < 1:
        return

    if len(p.shape) == 1:
        p = np.asarray([p], dtype=float)
    points = vtk.vtkPoints()
    for i in range(len(p)):
        points.InsertPoint(i, p[i][0:3])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    scalars_ = vtk.vtkFloatArray()
    scalars_.SetNumberOfValues(len(p))
    for i in range(len(p)):
        scalars_.SetValue(i, p[i][3])

    polydata.GetPointData().SetScalars(scalars_)

    if len(p[0]) > 4:
        critType = numpy_to_vtk(p[:, 4])
        critType.SetName("Critical Type")
        polydata.GetPointData().AddArray(critType)

    if len(p[0]) > 5:
        if outFile.find("redundant") == -1:
            correspondence = numpy_to_vtk(p[:, 5])
            correspondence.SetName("Correspondence")
            polydata.GetPointData().AddArray(correspondence)
        else:
            fcorrespondence = numpy_to_vtk(p[:, 5])
            fcorrespondence.SetName("FCorrespondence")
            polydata.GetPointData().AddArray(fcorrespondence)

    if len(p[0]) > 6:
        color_length = numpy_to_vtk(p[:, 6])
        color_length.SetName("duration")
        polydata.GetPointData().AddArray(color_length)

    if len(p[0]) > 7:
        mark = numpy_to_vtk(p[:, 7])
        mark.SetName("Mark")
        polydata.GetPointData().AddArray(mark)

    if vtk.VTK_MAJOR_VERSION <= 5:
        polydata.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outFile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()
  

if __name__ == '__main__':

    dataSet = sys.argv[1]
    # DATA_DIR = "./data/"
    # OUTPUT_DIR = os.path.join(DATA_DIR, dataSet)
    # CP_DIR = os.path.join(OUTPUT_DIR, 'highLight')

    # make_dir(CP_DIR)

    inputFiles = getInputFiles(dataSet)
    for inputFile in inputFiles:
        inputDir = os.path.join(inputFile[0], inputFile[1])
        CP = np.loadtxt(inputDir)

        visFileName = "cp_" + str(inputFile[2]) + '.vtp'
        if inputDir.find("redundant") != -1:
            visFileName = "redundant_" + str(inputFile[2]) + '.vtp'
        visFile = os.path.join(inputFile[0], visFileName)
        createVisFile(CP, visFile)
