import sys
import os
import re
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from paraview.simple import *
from utilities import *

def saveRectData(scalarfield, scalarName, outFile):
    # first, try 3D
    try:
        x1, y1, z1 = scalarfield.shape
    except ValueError:
        # then, try 2D
        try:
            x1, y1 = scalarfield.shape
            z1 = 1
        except ValueError:
            x1 = scalarfield.shape[0]
            y1 = 1
            z1 = 1
    
    print("Scalar field shape: ({}, {}, {})".format(str(x1), str(y1), str(z1)))
    
    # TODO/FIXME: do we need transpose here??????
    if not isTranspose:
        scalarfield = scalarfield.reshape(-1, )
    else:
        scalarfield = scalarfield.T.reshape(-1, )

    xArray = list(range(x1))
    yArray = list(range(y1))
    zArray = list(range(z1))
    
    vtkX = numpy_to_vtk(xArray)
    vtkY = numpy_to_vtk(yArray)
    vtkZ = numpy_to_vtk(zArray)
    imagedata = vtk.vtkRectilinearGrid()
    imagedata.SetDimensions(x1, y1, z1)
    imagedata.SetXCoordinates(vtkX)
    imagedata.SetYCoordinates(vtkY)
    imagedata.SetZCoordinates(vtkZ)
    
    scalar = numpy_to_vtk(scalarfield)
    scalar.SetName(scalarName)
    imagedata.GetPointData().AddArray(scalar)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(imagedata)
    writer.Write()
    
def saveImageData(scalarfield, scalarName, outFile, isTranspose):
    x0 = y0 = z0 = 0
    # first, try 3D
    try:
        x1, y1, z1 = scalarfield.shape
        x1 -= 1
        y1 -= 1
        z1 -= 1
    except ValueError:
        # then, try 2D
        try:
            x1, y1 = scalarfield.shape
            x1 -= 1
            y1 -= 1
            z1 = 0
        except ValueError:
            x1 = scalarfield.shape[0]
            y1 = 0
            z1 = 0
    
    print("Scalar field shape: ({}, {}, {})".format(str(x1+1), str(y1+1), str(z1+1)))
    # TODO/FIXME: do we need transpose here??????
    if not isTranspose:
        scalarfield = scalarfield.reshape(-1, )
    else:
        scalarfield = scalarfield.T.reshape(-1, )

    imagedata = vtk.vtkImageData()
    imagedata.SetExtent(x0, x1, y0, y1, z0, z1)

    scalar = numpy_to_vtk(scalarfield)
    scalar.SetName(scalarName)
    imagedata.GetPointData().AddArray(scalar)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(imagedata)
    writer.Write()

def savePolyData(scalarfield, scalarName, outFile, isTranspose):
    print("Saving poly data (this can be slow, not recommended). Using Freudenthal tetrahedralization...")
    
    x0 = y0 = z0 = 0
    # first, try 3D
    try:
        x1, y1, z1 = scalarfield.shape
    except ValueError:
        # then, try 2D
        try:
            x1, y1 = scalarfield.shape
            z1 = 1
        except ValueError:
            x1 = scalarfield.shape[0]
            y1 = 1
            z1 = 1
    
    print("Scalar field shape: ({}, {}, {})".format(str(x1), str(y1), str(z1)))
    
        # TODO/FIXME: do we need transpose here??????
    if not isTranspose:
        scalarfield = scalarfield.reshape(-1, )
    else:
        scalarfield = scalarfield.T.reshape(-1, )
    
    def initializePostPocessingTracking():
        polyline_cell = vtk.vtkCellArray()
        return polyline_cell
    
    cell_type = None
    
    # if 2D
    if z1 == 1:
        N = x1
        M = y1
        
        def inField(x, y):
            if x < 0 or x >= N:
                return False
            if y < 0 or y >= M:
                return False
            return True
        
        def coordID(x, y):
            return x + y*N
        
        points = vtk.vtkPoints()
        for j in range(M):
            for i in range(N):
                points.InsertPoint(coordID(i, j), [i, j, 0])
        
        cell_array = initializePostPocessingTracking()
        for j in range(M):
            for i in range(N):
                if inField(i-1, j) and inField(i, j+1):
                    triangle = vtk.vtkTriangleStrip()
                    triangle_ids = triangle.GetPointIds()
                    triangle_ids.SetNumberOfIds(4)
                    triangle_ids.SetId(0, coordID(i, j))
                    triangle_ids.SetId(1, coordID(i - 1, j))
                    triangle_ids.SetId(2, coordID(i, j + 1))
                    triangle_ids.SetId(3, coordID(i - 1, j + 1))
                    cell_array.InsertNextCell(triangle)
        cell_type = "strip"
    # else, 3D
    else:
        N = x1
        M = y1
        Q = z1
        
        def inField(x, y, z):
            if x < 0 or x >= N:
                return False
            if y < 0 or y >= M:
                return False
            if z < 0 or z >= Q:
                return False
            return True
        
        def coordID(x, y, z):
            return x + y*N + z*N*M
        
        points = vtk.vtkPoints()
        for k in range(Q):
            for j in range(M):
                for i in range(N):
                    points.InsertPoint(coordID(i, j, k), [i, j, k])
        
        cell_array = initializePostPocessingTracking()
        coordinate_offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
        tetrahedra_points = [[0, 1, 2, 6], [0, 1, 5, 6], [0, 5, 4, 6], [0, 4, 7, 6], [0, 3, 7, 6], [0, 3, 2, 6]]
        
        for k in range(Q):
            print("k: {} / {}".format(str(k), str(Q)))
            for j in range(M):
                for i in range(N):
                    for tetra in tetrahedra_points:
                        nx = [None] * 4
                        ny = [None] * 4
                        nz = [None] * 4
                        outOfBound = False
                        for ept, pt in enumerate(tetra):
                            nx[ept] = i + coordinate_offsets[pt][0]
                            ny[ept] = j + coordinate_offsets[pt][1]
                            nz[ept] = k + coordinate_offsets[pt][2]
                            if not inField(nx[ept], ny[ept], nz[ept]):
                                outOfBound = True
                        if outOfBound:
                            continue
                        tetrah = vtk.vtkTetra()
                        tetra_pid = tetrah.GetPointIds()
                        tetra_pid.Reset()
                        
                        for pt in range(len(tetra)):
                            tetra_pid.InsertNextId(coordID(nx[pt], ny[pt], nz[pt]))
                        cell_array.InsertNextCell(tetrah)
        
        cell_type = "strip"
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    if cell_type == "strip":
        polydata.SetStrips(cell_array)
    elif cell_type is None:
        polydata.SetLines(cell_array)
    
    scalar2 = numpy_to_vtk(scalarfield)
    scalar2.SetName(scalarName)
    
    polydata.GetPointData().AddArray(scalar2)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(polydata)
    writer.Write()

def saveStructuredGridData(scalarfield, scalarName, outFile, isTranspose):
    x0 = y0 = z0 = 0
    # first, try 3D
    try:
        x1, y1, z1 = scalarfield.shape
    except ValueError:
        # then, try 2D
        try:
            x1, y1 = scalarfield.shape
            z1 = 1
        except ValueError:
            x1 = scalarfield.shape[0]
            y1 = 1
            z1 = 1
    
    print("Scalar field shape: ({}, {}, {})".format(str(x1), str(y1), str(z1)))
    
    def coordID(x, y, z):
        return x + y*x1 + z*x1*y1
    
    points = vtk.vtkPoints()
    for k in range(z1):
        for j in range(y1):
            for i in range(x1):
                points.InsertPoint(coordID(i, j, k), [i, j, k])
    
    # TODO/FIXME: do we need transpose here??????
    if not isTranspose:
        scalarfield = scalarfield.reshape(-1, )
    else:
        scalarfield = scalarfield.T.reshape(-1, )

    imagedata = vtk.vtkStructuredGrid()
    imagedata.SetDimensions(x1, y1, z1)
    imagedata.SetPoints(points)
    
    scalar = numpy_to_vtk(scalarfield)
    scalar.SetName(scalarName)
    imagedata.GetPointData().AddArray(scalar)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(imagedata)
    writer.Write()
    
    
if __name__ == '__main__':
    dataDirs = []
    vtkTypes = []
    isTranspose = False
    scalarName = "var"
    if os.path.exists("mat2vtx.config"):
        print("Using info from mat2vtx.config...")
        with open("mat2vtx.config") as config_f:
            config_lines = config_f.readlines()
            isReadingDir = False
            isReadingType = False
            for i in range(len(config_lines)):
                line = config_lines[i].strip().replace("\\", "/")
                if line.startswith("#") and "dir path" in line:
                    isReadingDir = True
                    isReadingType= False
                    continue
                elif line.startswith("#") and "vtk type" in line:
                    isReadingDir = False
                    isReadingType = True
                    continue
                elif line.startswith("#") and "need transpose" in line:
                    if "yes" in line.lower() or "true" in line.lower():
                        isTranspose = True
                    isReadingDir = False
                    isReadingType = False
                    continue
                elif line.startswith("#") and "scalar name" in line:
                    if i + 1 < len(config_lines):
                        tmpScalarName = config_lines[i+1].strip()
                        if len(tmpScalarName) > 0:
                            scalarName = tmpScalarName
                    print("Finish loading the scalar name config, loading complete...")
                    break
                elif line.startswith("#"):
                    continue
                
                if isReadingDir:
                    assert not isReadingType
                    if len(line) > 0:
                        dataDirs.append(line)
                elif isReadingType:
                    if len(line) > 0:
                        vtkTypes.append(line)
                else:
                    pass
    else:
        print("Loading config from command...")
        if len(sys.argv) < 3 or len(sys.argv) > 5:
            print("pvpython mat2vtx.py [dir path for .npy/.txt files] [vti/vts/vtp/vtr] [scalar name (default: var)]")
            exit()
        dataDirs = [sys.argv[1]]
        vtkTypes = [sys.argv[2]]
        if len(sys.argv) == 4:
            scalarName = sys.argv[3]
    
    for dataDir in dataDirs:
        for vtkType in vtkTypes:
            print("Processing {} to {} files...".format(dataDir, vtkType))
            outDir = os.path.join(dataDir, vtkType)
            make_dir(outDir)

            inputFiles = getInputFiles(dataDir)
            for file in inputFiles:
                suffix = file.split(".")[-1]
                scalarfield = load2mat(file)
                if scalarfield is None:
                    print("Cannot load scalar field! Skip...")
                    continue
                if isTranspose and len(scalarfield.shape) >= 3 and scalarfield.shape[2] > 1:
                    print("3D or higher dimensional data. Cannot transpose! Please modify the code for custom reshape.")
                    isTranspose = False
                filename = os.path.basename(file).replace("_juelich", "")
                output_filename = os.path.join(outDir, filename.replace(suffix, vtkType).replace("2t", "2_t_"))
                if vtkType == "vtr":
                    print("Data: safe; Triangulation: UNSAFE")
                    saveRectData(scalarfield, scalarName, output_filename, isTranspose)
                elif vtkType == "vti":
                    print("Data: safe; Triangulation: UNSAFE")
                    saveImageData(scalarfield, scalarName, output_filename, isTranspose)
                elif vtkType == "vtp":
                    print("Data: safe; Triangulation: safe")
                    savePolyData(scalarfield, scalarName, output_filename, isTranspose)
                elif vtkType == "vts":
                    print("Data: safe; Triangulation: safe")
                    saveStructuredGridData(scalarfield, scalarName, output_filename, isTranspose)
                else:
                    print("Unsupported VTK Type:", vtkType)
