# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:38:27 2022

@author: Administrator
"""

import numpy as np # numerical package for multi-dimension arrays
import hgrampy as hg


print('Make SURE big .npy dataset is OFFLINE(in local Drivew:), otherwise load too long from gDrive and fail as a reshape error!')# ; print('Loading clear objs...')
occludedNoisedObj=np.load("HOLODatasetfile.npy") # occluded 0-9 objects
print(hg.loadNpStrucArrAsDict("HOLOClassParas.npy") )#parameters used in hologram dataset generation

print("Occuled 0-9 Image size=", occludedNoisedObj.shape, " from .." +"HOLODatasetfile.npy" )
classNcount=hg.loadNpStrucArrAsDict("HOLOClassNcount.npy") #for cmp only
fileIndex=list(classNcount.keys()) # extract dictionary's keys of the 0-9 digits, convert to a list

nbImages, rows, columns =occludedNoisedObj.shape #nbImage=axis2=no. of images, image's: rows,columns=axis1,axis0 of the array. 27Dec2018 Rightmost is axis0
occluedLabelAdj, nbImages=hg.buildLabelsNcountImagesNEW(classNcount, fileIndex)
occludedNoisedObjAdj=occludedNoisedObj[0:nbImages,:,:]
hg.plotHeadTail('Objects', occludedNoisedObjAdj, mode=hg.MAGNITUDE_M); hg.plotHeadTail('Objects', occludedNoisedObjAdj, mode=hg.PHRASE_M)
hg.reconPlotHT('Reconstruct Objects', occludedNoisedObjAdj, hg.DEFAULT_OPT_CF['Z_DISTANCE']) #plot reconstructed obj