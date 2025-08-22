import sys
import os
import re
import copy
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def getInputCPFiles(dataSet, vtkPath, cap=-1):
    fileList = []
    for root, dirs, files in os.walk("./postprocess-result/"):
        if not dataSet in root:
            continue
        ff = [x for x in files if x.startswith("cp_") and x.endswith(".vtp")]
        fl = (root, [])
        for file in ff:
            if file.startswith("cp_") and file.endswith(".vtp"):
                # print("FILES:", file)
                id = int(re.search(r'\d+', file).group())
                if (cap > 0) and (id > cap):
                    continue
                extra_file = "treeNode_extra_" + str(id).zfill(3) + ".txt"
                # simplification = re.search(r'\d+percent', root).group()
                # scalar_field_vtk_file = os.path.join(vtkPath, vtkfiles[id])
                fl[1].append([file, extra_file])
        if len(fl[1]) > 0:
            fileList.append(fl)
    return fileList

def getInputCPFiles_deprecated(data_dir):
    files = os.listdir(data_dir)
    fileList = []
    for i in files:
        if '.vtp' in i and "cp" in i:
            fileList.append(i)
    order = []
    if '.DS_Store' in fileList:
        fileList.remove('.DS_Store')
    for i in range(len(fileList)):
        order.append(int(re.search(r'\d+', fileList[i]).group()))
    fileList = [x for _,x in sorted(zip(order,fileList))]
    return fileList

def readTrackingFile(inFile, extraFile=None):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(inFile)
    reader.Update()
    ft = reader.GetOutput()
    
    ftID = vtk_to_numpy(ft.GetPointData().GetArray("Correspondence"))
    ftType = vtk_to_numpy(ft.GetPointData().GetArray("Critical Type"))
    scalars_ = vtk_to_numpy(ft.GetPointData().GetArray("Scalars_"))
    ncnt = ft.GetNumberOfPoints()
    mark = np.zeros((ncnt, ))
    try:
        mark = vtk_to_numpy(ft.GetPointData().GetArray("Mark"))
    except Exception as e:
        pass
    
    nodes = []
    for i in range(ncnt):
        nodes.append(ft.GetPoint(i))
    nodes = np.array(nodes)

    prev = []
    if extraFile is not None and os.path.exists(extraFile):
        with open(extraFile, "r") as ex_f:
            lines = ex_f.readlines()
            for line in lines:
                cor, prev_list_str = line.split(":")
                prev_list = prev_list_str.split(";")
                # print(prev_list)
                prev_list = [int(x.strip()) for x in prev_list]
                prev.append((int(cor.strip()), prev_list))

    return ftType, ftID, nodes, ncnt, prev, scalars_, mark

def getPolyLine(polyline_cell, l, times, idlengths, nodes, horizontal_move):
    if len(l)<2:
        return polyline_cell, idlengths, horizontal_move
    polyline = vtk.vtkPolyLine()
    polyline_pid = polyline.GetPointIds()
    
    ll = []
    for i in range(len(l)):
        if (i == 0) or abs(times[l[i-1]] - times[l[i]]) <= 1:
            polyline_pid.InsertNextId(l[i])
            if polyline_pid.GetNumberOfIds() >= 2:
                polyline_cell.InsertNextCell(polyline)
                polyline = vtk.vtkPolyLine()
                polyline_pid = polyline.GetPointIds()
                polyline_pid.InsertNextId(l[i])
            ll.append(l[i])
        elif i < len(l)-1:
            if len(ll) >= 2:
                if polyline_pid.GetNumberOfIds() >= 2:
                    polyline_cell.InsertNextCell(polyline)
                for lid in ll:
                    idlengths[lid] = len(ll)
            else:
                idlengths[ll[0]] = 1
            polyline = vtk.vtkPolyLine()
            polyline_pid = polyline.GetPointIds()
            polyline_pid.InsertNextId(l[i])
            ll = [l[i]]
            
    if polyline_pid.GetNumberOfIds() >= 2:
        polyline_cell.InsertNextCell(polyline)

    if len(ll) >= 2:
        for el, lid in enumerate(ll):
            idlengths[lid] = len(ll)
            if el > 0:
                horizontal_move[lid] = max(horizontal_move[lid], 
                                           ((nodes[ll[el - 1]][0] - nodes[lid][0]) ** 2 + (nodes[ll[el - 1]][1] - nodes[lid][1]) ** 2) ** 0.5)
    else:
        idlengths[ll[0]] = 1
        horizontal_move[ll[0]] = 0
    return polyline_cell, idlengths, horizontal_move

def getPolyLine_deprecated(polyline_cell, l, marks, idlengths, nodes, horizontal_move):
    if len(l)<2:
        return polyline_cell, idlengths, horizontal_move
    polyline = vtk.vtkPolyLine()
    polyline_pid = polyline.GetPointIds()
    
    ll = []
    for i in range(len(l)):
        if (i == 0) or (marks[l[i-1]] == marks[l[i]]):
            polyline_pid.InsertNextId(l[i])
            ll.append(l[i])
        elif i < len(l)-1:
            if len(ll) >= 2:
                polyline_cell.InsertNextCell(polyline)
                for lid in ll:
                    idlengths[lid] = len(ll)
            else:
                idlengths[ll[0]] = 1
            polyline = vtk.vtkPolyLine()
            polyline_pid = polyline.GetPointIds()
            polyline_pid.InsertNextId(l[i])
            ll = [l[i]]

    if len(ll) >= 2:
        polyline_cell.InsertNextCell(polyline)
        for el, lid in enumerate(ll):
            idlengths[lid] = len(ll)
            if el > 0:
                horizontal_move[lid] = max(horizontal_move[lid], 
                                           ((nodes[ll[el - 1]][0] - nodes[lid][0]) ** 2 + (nodes[ll[el - 1]][1] - nodes[lid][1]) ** 2) ** 0.5)
    else:
        idlengths[ll[0]] = 1
        horizontal_move[ll[0]] = 0
    return polyline_cell, idlengths, horizontal_move

def initializePostPocessingTracking():
    polyline_cell = vtk.vtkCellArray()
    return polyline_cell

def savePolyData(nodes, types ,ids , times, idlengths, scalars_, marks, markIdLengths, horizontal_movement, polyline_cell, outFile):
    points = vtk.vtkPoints()
    for i in range(0, len(nodes)):
        points.InsertPoint(i, [nodes[i,0], nodes[i,1], nodes[i,2]])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(polyline_cell)
        
    scalar2 = numpy_to_vtk(types)
    scalar2.SetName('Critical Type')
    scalar3 = numpy_to_vtk(ids)
    scalar3.SetName('Correspondence')
    scalar4 = numpy_to_vtk(times)
    scalar4.SetName('time')
    scalar5 = numpy_to_vtk(idlengths)
    scalar5.SetName('length')
    scalar6 = numpy_to_vtk(scalars_)
    scalar6.SetName('Scalars_')
    scalar7 = numpy_to_vtk(marks)
    scalar7.SetName('Mark')
    scalar8 = numpy_to_vtk(markIdLengths)
    scalar8.SetName('Mark_length')
    scalar8 = numpy_to_vtk(horizontal_movement)
    scalar8.SetName('Horizontal_Movement')

    polydata.GetPointData().AddArray(scalar2)
    polydata.GetPointData().AddArray(scalar3)
    polydata.GetPointData().AddArray(scalar4)
    polydata.GetPointData().AddArray(scalar5)
    polydata.GetPointData().AddArray(scalar6)
    polydata.GetPointData().AddArray(scalar7)
    polydata.GetPointData().AddArray(scalar8)
   
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(polydata)
    writer.Write()

def getTrajectory(inputFiles, dataDir, outDir):
    for root, flist in inputFiles:
        ooo = os.path.join(outDir, root)
        
        make_dir(ooo)
        outFile = os.path.join(ooo, "tracking.vtp")

        lengths = {}
        new_id = {}
        flist.sort(key=lambda x: int(x[0].replace("cp_", "").replace(".vtp","")))
        # print(flist)

        for i in range(len(flist)):
            cpfilename, extrafilename = flist[i]
            inputFile = os.path.join(root, cpfilename)
            # inFile = os.path.join(dataDir, inputFile)
            inFile = inputFile
            ftType, ftID, nodes, ncnt, _, scalars_, mark = readTrackingFile(inFile)

            for id in ftID:
                if id not in lengths:
                    lengths[id] = 1
                else:
                    lengths[id] += 1
                if lengths[id] > 1 and (id not in new_id):
                    new_id[id] = len(new_id) + 1

        prevAll = []
        for i in range(len(flist)):
            cpfilename, extrafilename = flist[i]
            inputFile = os.path.join(root, cpfilename)
            extraFile = os.path.join(root, extrafilename)
            inFile = inputFile
            ftType, ftID, nodes, ncnt, prev, scalars_, mark = readTrackingFile(inFile, extraFile)
            # for j in range(len(ftID)):
            #     ftID[j] = new_id[ftID[j]]

            idlength = []
            for id in ftID:
                idlength.append(lengths[id])

            if i == 0:
                nodesAll = np.array(nodes)
                typeAll = ftType
                idAll = ftID
                timeAll = np.ones(ncnt)*i
                idlengthAll = idlength
                scalarsAll = scalars_
                marksAll = mark
            else:
                idAll = np.concatenate((idAll, [ftID]), axis=None)
                typeAll = np.concatenate((typeAll, [ftType]), axis=None)
                timeAll = np.concatenate((timeAll, [np.ones(ncnt)*i]), axis=None)
                nodesAll = np.concatenate((nodesAll, nodes), axis=0)
                idlengthAll = np.concatenate((idlengthAll, idlength), axis=None)
                scalarsAll = np.concatenate((scalarsAll, scalars_), axis=None)
                marksAll = np.concatenate((marksAll, mark), axis=None)
                prevAll.append(prev)

        markIdLengthAll = copy.deepcopy(idlengthAll)
        horizontalMoveAll = np.zeros(idlengthAll.shape)

        polyline_cell = initializePostPocessingTracking()
        trIDs = list(set(idAll))
        for trID in trIDs:
            ft = np.where(np.isclose(idAll, trID))[0]
            if abs(trID - int(trID)) > 0.4:
                print(trID, ft)
            polyline_cell, markIdLengthAll, horizontalMoveAll = getPolyLine(polyline_cell, ft, timeAll, markIdLengthAll, nodesAll, horizontalMoveAll)

        for i in range(len(prevAll)):
            prev = prevAll[i]
            for cor, nodelist in prev:
                for ele in nodelist:
                    if ele < 0:
                        break
                    ft1 = np.where((timeAll == i) & (idAll == ele))[0]
                    ft2 = np.where((timeAll == i+1) & (idAll == cor))[0]
                    ft = np.concatenate((ft1, ft2), axis=None)
                    if len(ft) > 1:
                        polyline_cell, markIdLengthAll, horizontalMoveAll = getPolyLine(polyline_cell, ft, timeAll, markIdLengthAll, nodesAll, horizontalMoveAll)


        savePolyData(nodesAll, typeAll ,idAll , timeAll , idlengthAll, scalarsAll, marksAll, markIdLengthAll, horizontalMoveAll, polyline_cell, outFile)



if __name__ == '__main__':

    dataDir = sys.argv[1]
    cap = -1
    if len(sys.argv) > 2:
        cap = int(sys.argv[2])
    outDir = os.path.join(".", "fgw_results")
    make_dir(outDir)

    inputFiles = getInputCPFiles(dataDir, dataDir, cap)
    getTrajectory(inputFiles, dataDir, outDir)
