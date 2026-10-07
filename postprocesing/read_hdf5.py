#!/usr/bin/python
import sys, getopt
import os
import numpy as np
import h5py
from scipy import interpolate

def writefields2D(folder, destFolder):
    #Read mesh information
    f_sum = h5py.File(folder + '/summary.samrai', "r")
    nProcessors  = f_sum['/BASIC_INFO/number_processors']
    nPatches     = f_sum['/BASIC_INFO/number_patches_at_level']
    patchExtents = f_sum['/extents/patch_extents']
    patchMap     = f_sum['/extents/patch_map']
    varNames = f_sum['/BASIC_INFO/var_names']

    maximum = [0, 0]
    minimum = [999999, 999999]
    for i in range(len(patchExtents)):
        maximum[0] = max(maximum[0], patchExtents[i][1][0])
        maximum[1] = max(maximum[1], patchExtents[i][1][1])
        minimum[0] = min(minimum[0], patchExtents[i][0][0])
        minimum[1] = min(minimum[1], patchExtents[i][0][1])
    size = [(maximum[0] - minimum[0]) + 1 +1,(maximum[1] - minimum[1]) + 1+1]

    data = {}
    for iVar in range(len(varNames)):
        data[varNames[iVar]] = np.zeros(shape=(size[0], size[1]))

    for iPatch in range(nPatches[0]):
        iProc     = patchMap[iPatch][0]
        iProcStr = str(iProc).zfill(5)
        iPatchStr = str(iPatch).zfill(5)
        rangeX = (patchExtents[iPatch][0][0], patchExtents[iPatch][1][0]+1)
        rangeY = (patchExtents[iPatch][0][1], patchExtents[iPatch][1][1]+1)
        sizeX = patchExtents[iPatch][1][0] - patchExtents[iPatch][0][0] + 1+1;
        sizeY = patchExtents[iPatch][1][1] - patchExtents[iPatch][0][1] + 1+1;
        f_data = h5py.File(folder + '/processor_cluster.' + iProcStr + '.samrai', "r")
        for iVar in range(len(varNames)):
            tmp = f_data['/processor.' + iProcStr + '/level.00000/patch.' + iPatchStr + '/' + varNames[iVar]]
            #tmp = mapToCorrectDimensions(tmp, 2, (sizeX,sizeY))
            tmp = np.reshape(tmp, (sizeX,sizeY), order="F")
            data[varNames[iVar]][rangeX[0]:rangeX[1]+1,rangeY[0]:rangeY[1]+1] = tmp[:,:]
        f_data.close()

    data2 = data
    for iVar in range(len(varNames)):
        #Writing the variable to disk
        f = h5py.File(destFolder + "/" + varNames[iVar] + '.hdf5', "w")
        dset = f.create_dataset(varNames[iVar], data2[varNames[iVar]].shape, dtype='f')
        dset[:,:] = data2[varNames[iVar]][:,:]
        f.close()
    f_sum.close()
def main():

    for s in [0,1,2,3,4,5,6,7]:
# set xrange from 0 to the last number in dump.visit, with the output frequency    
        for t in xrange(0, 99840, 512):
            #print "Processing: t=",t;
            if not os.path.exists("outputDir_sphere" + str(s) + "/t" + str(t)):
                os.makedirs("data_outputDir_sphere" + str(s) + "/t" + str(t))
            writefields2D("outputDir_sphere" + str(s) + "/visit_dump." + ( "%05d" % t), "data_outputDir_sphere" + str(s) + "/t" + str(t))    
            
        for t in xrange(100352, 800256, 512):
            #print "Processing: t=",t;
            if not os.path.exists("outputDir_sphere" + str(s) + "/t" + str(t)):
                os.makedirs("data_outputDir_sphere" + str(s) + "/t" + str(t))
            writefields2D("outputDir_sphere" + str(s) + "/visit_dump." + ( "%06d" % t), "data_outputDir_sphere" + str(s) + "/t" + str(t))            

if __name__ == "__main__":
    main()
