# -*- coding: utf-8 -*-
"""
Created on 11oct2021


@author: hoson
ver 1.0 12oct2021 - first version with great results 
ver 1.1 13oct2021 - able to handle twin frequency-spectrum's option, use .._t.npy files instead.
ver 2.0 20oct2021 - ver of loading each sample's objectWave & classCount from a npy file. 10 samples will load 10 dataset & 10 classNcount NPYs
"""
import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten #, concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical # plot_model must installed pydot, X.2.
import numpy as np; import time; import sklearn.model_selection as skm; import hgrampy as hg  # import self-made Hologram library
from subprocess import check_output; from multiprocessing import cpu_count # cpu & speed info
import sys; import pandas as pd; import os
from sklearn.metrics import confusion_matrix; import seaborn as sns; import matplotlib.pyplot as plt

NB_CLASSES=len(hg.BIO_FILENAME) #6, total classes of the biological samples
RAW_MODE, COS_MODE=20, 21; modeDict=dict( {20:"RAW_MODE", 21:"COS_MODE"}, default="None" )
smoothPhr=COS_MODE #RAW_MODE, COS_MODE
CONCATE_M=0 # CONCATE_M, MAX_M, MAX_SOFTMAX_M=0, 1, 2 #; ENSEMBLE_MODE=MAX_M
""" Common tuning network hyper-parameters """
NB_FILTER =3 #at least 16, 32 better than 48/64 12oct2021 
DELTA_E= 0.0001#0.0001 #0.00009 # error for training early-stop
PATIENCE_NB=5 #10 #5
NB_EPOCHS= 1000 # use earlyStopping to stop
SPILT_RATIO_4T=0.2  # Test Set's  ratio. e.g. 0.3=30% for test & 70% for train
VALID_SPLIT=0.2# reserve 0.1=10% fr trainSet for validation during Training

BATCH=32 # batch_size=BATCH, default=32. 32 means accumulate 32 uodate rounds before update the weights a round.
CONV_LAYERS=2 #2 max already. # no of convolution layers, 2=>0,1 two layers will be created
DROPOUT_RATE1=0.5 # at least 0.5, 0.25 no good. 12oct2021
DROPOUT_RATE2=0.5 # at least 0.5
STRIDES_1ST=(1,1) # pooling filter pixel move step, x= move-x-pixels per step.
STRIDES_2ND=(2,2) # (2,2). (1,1) is obviously better 12oct2021

""" Non-common tuning network hyper-parameters """
X_SILENCE=1 # 0=silence, 1=echo console
KERNAL_SIZE =(3,3) # convolution filter kernel size 3x3.
POOL_SIZE =(2,2) # size of pooling area for max pooling. Should be smaller than Kernal size
CHANNELS = 1 # image channel is one for gray image
OPTI_MODE= 'val_auc' #'val_auc' 'val_precision', 'auc'

METRICS_DEF = [metrics.CategoricalAccuracy(name='acc'), #debug 17aug2021, BinaryAcc to Cat..Acc
               metrics.AUC(name='auc') ] #AUC=0 => model's prediction is 100% wrong, AUC=1 => prediction 100% right.

SORT_KEY='fullScore'
LEFT, RIGHT=0,1 #indexs for left/right of Y-network

G_epochRan=[] #keep track epoch run in different training sessions.

def genConvDrpPooLayer(inputs, noOfLayers, noOfBaseFilter, kernalSize, strides:list, dropRates:list, poolSize ): # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS
    for i in range(noOfLayers): #Conv2D-Dropout-MaxPooling2D
        inputs  = Conv2D(filters= noOfBaseFilter*(i+1), #no-of-filters doubles after each layer (e.g. 32-64...)
				   kernel_size=kernalSize, strides=strides[i], #[0] is 1st stride size, [1] is 2nd etc.
				   padding='same', activation='relu')(inputs)
        inputs  = MaxPooling2D(pool_size=poolSize)(inputs) #default is (2, 2). debug output->input, 26aug2021
        inputs  = Dropout(dropRates[i])(inputs) #[0] is 1st dropout rate, [1] is 2nd etc.
        
    return inputs #debug output->input, 26aug2021

def concateNsoftMax(leftInp, rightInp, dropRate): # concatenateEnsemble-softmax
    """
    concate => flatten => Dropout => Dense(NB_CLASSES) => Softmax

    Parameters
    ----------
    leftInp : x input
    rightInp : y input

    Returns output the TF model
    -------
    """
    output = layers.concatenate([leftInp, rightInp]) # default concat at last axis e.g.(2,5)+(2,5)=>(2,10), A1
    output = Flatten()(output) # flat feature maps 
# CAN't add Dense layer here, all will be 5 !!. Can't explain YET. 6jun2021
    # output = Dense(NB_NEURON, activation='relu')(output)
    # output = Dropout(dropRate)(output) # output = layers.BatchNormalization()(output) #A4, worster performance and faster training
    output = Dense(NB_CLASSES,)(output)
    output = Dropout(dropRate)(output) # better performance just before Softmax, 28aug2021
    output = layers.Activation('softmax')(output)           
    
    return output

def createYcnn(noOfLayers, noOfBaseFilter, kernalSize, strides:list, dropRates:list, poolSize, rows, columns, channels): # 2x(Convolute and Pooling layers) + 2x Dense Layers model
# Y network, 2-input and 1-output, A2
	# left branch of Y network. 
    leftType = Input(shape=(rows, columns, channels)) #declare the Class object
    leftInp=leftType # creat the instance
    leftInp=genConvDrpPooLayer(leftInp, noOfLayers, noOfBaseFilter, kernalSize, strides, dropRates, poolSize) # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS

	# right branch of Y network
    rightType = Input(shape=(rows, columns, channels))  #declare the Class object
    rightInp=rightType # must difference to leftType debug28aug2021, Error: i/ps to model redundant, all i/p should only appear once
    rightInp=genConvDrpPooLayer(rightInp, noOfLayers, noOfBaseFilter, kernalSize, strides, dropRates, poolSize) # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS

    output = concateNsoftMax(leftInp, rightInp, DROPOUT_RATE2) #concate=> flatten=> Dropout=> Dense=> Softmax
    
	# build TF model
    model = Model([leftType, rightType], output) #leftType, rightType MUST differece debug28aug2021, Error: i/ps to model redundant, all i/p should only appear once
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS_DEF) #'sgd'. adam shall better&faster than sgd
    return model

def creatNprintEnModel(string, rows, columns): #e.g. string='modelSample.png'
    model = createYcnn(CONV_LAYERS, NB_FILTER, KERNAL_SIZE, (STRIDES_1ST, STRIDES_2ND),
                         (DROPOUT_RATE1, DROPOUT_RATE2), POOL_SIZE, rows, columns, CHANNELS) # MUST refresh after each loop
    model.summary(); plot_model(model, to_file=string, show_shapes=True, show_layer_names=True) #plot the model to *.png
    return model

def extractMagOrPhrase(dataset, magPhrase=hg.MAGNITUDE_M, phraseSmoothingMethod=RAW_MODE):
    nbImages,rows,columns = dataset.shape #e.g nbImages,rows,columns=(25000, 64, 64). For later reshape
    if magPhrase==hg.MAGNITUDE_M: dataset=np.absolute(dataset); print("Exacted Hologram Pixels Magnitudes") 

    elif magPhrase==hg.PHRASE_M:
        dataset=np.angle(dataset); print("\nExtracted Hologram Pixels Phrase-Angles.", end =' Applying..')
        if phraseSmoothingMethod==RAW_MODE:
            print("No phrase-smoothing")
        elif phraseSmoothingMethod==COS_MODE:
            print("Cos phrase-smoothing"); dataset=np.cos(dataset) #cos a bit better than sin
            
    else: print('Invalid Extraction Mode !. Quit the code now.'); quit()
            
    dataset=(dataset-dataset.min())/(dataset.max()-dataset.min()) #12oct2021, max => min-max Normalize
    dataset=dataset.reshape((nbImages,rows,columns,CHANNELS)); dataset=dataset.astype("float32") #!! astypes will discard complex part if place before absolute(). Try single precision for speed first.
    return dataset

def formatDataNlabel(dataset, labels, noOfClasses, magPhrase=hg.MAGNITUDE_M, phraseSmoothingMethod=RAW_MODE):
    dataset=extractMagOrPhrase(dataset, magPhrase, phraseSmoothingMethod); print("Processing without spliting")
    labels = to_categorical(labels, noOfClasses) # creat labels as a matrix table. 1 for the right class, 0 for others
    return dataset, labels # return corresponding Mag/Phrase info, categorized labels

def fitEval(msgString, model, inputs:list , label):
# Y-network example: https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-y-network-2.1.2.py
# validation_split: https://keras.io/api/models/model_training_apis/
    global G_epochRan #keep track epoch run in different training sessions.
    cList=[tf.keras.callbacks.EarlyStopping(monitor=OPTI_MODE, mode='max', restore_best_weights=True, #OPTI_MODE is 'auc'
                patience=PATIENCE_NB, min_delta=DELTA_E, verbose=1)] #early-stopping & echo console
    print("Starting Training...."); start=time.time()
    history=model.fit([inputs[LEFT],  inputs[RIGHT]], label, callbacks=cList, validation_split = VALID_SPLIT, # exclude some trainData for validation in earlyStopping. 
                batch_size=BATCH, epochs=NB_EPOCHS, verbose=X_SILENCE) 
    stop=time.time()-start; G_epochRan.append(len(history.history['loss']) - PATIENCE_NB) # minus no. of patience waited
    print(f"EBOR Training time:{stop}, this session's 'ACTUAL ran epochs:{G_epochRan[-1]}.")  # realtime used in model.fit

    print(f'Result {msgString}') #prompt use which old/Bio Ensemble method
    score=model.evaluate([inputs[LEFT],  inputs[RIGHT]], label, batch_size=BATCH, verbose=0)
    return history, score #del model, NO Need

def plotConfuMatrix(labels, predictions, string): #CONFUSION Matrix  
  if labels.ndim >1: labels=np.argmax(labels, axis=-1) # convert back matrix to vector for confusion_matrix()
  if predictions.ndim >1: predictions=np.argmax(predictions, axis=-1) #1-D array already, in fact not necessary. Design for change only!!
  
  cm=confusion_matrix(labels, predictions, labels=np.unique(labels), normalize=None) #KNOWN two classed ONLY
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title(string+" [Confusion matrix]")
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label') 
  
  print("\nCM Matrix from ",string, "dataset -" )
# =============================================================================
# Cij is number of observations known in group i and predicted in group j.
# count of true negatives=C00, false negatives=C10, true positives=C11 and false positives=C01:
# =============================================================================
  if np.size(cm,1)>1: #A2, some case axis1 size=1 instead of 2 when 0/0
      if cm[0][1] !=0: print("TP/FP=",cm[1][1]/cm[0][1], end=" ")
      elif  cm[1][1]+cm[0][1] !=0: print(" Precision=",cm[1][1]/(cm[1][1]+cm[0][1]), end=" " )
      elif cm[1][1]+cm[1][0] !=0: print(" Recall=",cm[1][1]/(cm[1][1]+cm[1][0]), " !!!!!!!!!!!!!")
      else: print("All three ratios divided by zeros, error") 
  else: print("Error as axis1 size=1 only and skip printings.")

  return cm

def reportParameters(string):
    print("TF ver:", tf.__version__, end =" "); print(" ,Keras ver=",tf.keras.__version__, end =' ,')
    print(check_output(["wmic","cpu","get", "name"])[46:87], end =' ,' ); print("No CPUs=",cpu_count())
    print("Python:", sys.version, end =' '); print("Ver:", sys.version_info)

    print(hg.loadNpStrucArrAsDict(string) )#parameters used in hologram dataset generation
    # parameters=hg.loadNpStrucArrAsDict(string+".npy")
    # print("Object at distance=", parameters.get("fzp distance"))
    return

def mergeDataNclass(waveBaseFilename, classCountBaseName):
    """
    Merge individual data (wave) and class files into a big dataset and a class count set
        for buildLabelsNcountImagesNEW() to build the numerical labels (0,1,2.....)
    Parameters
    ----------
    waveBaseFilename : basic format of an object wave file name
    classCountBaseName : basic format of the label npy file nmae

    Returns
    -------
    An object's wave dataset, the object's class name and how many sample for the class.
    """
    def loadWaveNClassCnt(bioFileName, waveBaseFilename, classCountBaseName):      
        nameStr=hg.file2SampleName(bioFileName); print(f"Processing {nameStr} ...", end='  ') #nameStr=HouseflyWing-1
        clsNCnt=hg.loadNpStrucArrAsDict(classCountBaseName+f"_{nameStr}"+ '.npy') #class & corresponding sample count  
        wave=np.load(waveBaseFilename+f"_{nameStr}"+ '.npy') # objectWaves (CGH)
        return wave, clsNCnt
    
    objsWave, classNcount=loadWaveNClassCnt(hg.BIO_FILENAME[0], waveBaseFilename, classCountBaseName) #handle 1st one
    for file in hg.BIO_FILENAME[1: :1]: # start:stop:increment, start 1, to end no stop, 1 each step 
        wave, clsNCnt=loadWaveNClassCnt(file, waveBaseFilename, classCountBaseName)
        objsWave=np.vstack((objsWave, wave)) #append(stack) array along axis-0
        classNcount.update(clsNCnt) #class & corresponding sample count
   
    print('Merged waves and classNCounts done.')
    return objsWave, classNcount # return all objects wave

        
# =============================================================================
# ###         Main Program start Here         ###
# =============================================================================
dataFilename, objClassNCount, paras = hg.getStoreFilenames()
path=os.getcwd() +"\\"; reportParameters(path+ paras+'.npy') # report all system and this Python's CNN parameters
print(f'Number of sample classes={NB_CLASSES}');results=pd.DataFrame() #init dummy dataframe

print('Make SURE big .npy dataset is OFFLINE(in local Drivew:), otherwise load too long from gDrive and fail as a reshape error!')
# objsWave=np.load(path+ dataFilename+ '.npy') # objectWaves (CGH)
# classNcount=hg.loadNpStrucArrAsDict(path+ objClassNCount+ '.npy') #class & corresponding sample count
objsWave, classNcount=mergeDataNclass(path+ dataFilename, path+ objClassNCount) # objectWaves (CGH) & each class and count
print("Bio sample objWave=", objsWave.shape, " from .." + dataFilename+ '_xxxx.npy' )
sampleName=list(classNcount.keys()) # extract dict's keys e.g. 'HoneyBeeWing', convert to a list

nbImages, rows, columns =objsWave.shape; assert nbImages ==hg.SIZE_OF_SAMPLE*len(hg.BIO_FILENAME), 'Size of samples WRONG!'
objsWaveLabel, nbImages=hg.buildLabelsNcountImagesNEW(classNcount, sampleName) #label=0,1,2....
hg.plotHeadTail('Objs_Mag', objsWave, mode=hg.MAGNITUDE_M); hg.plotHeadTail('Objs_Phase', objsWave, mode=hg.PHRASE_M)
hg.reconPlotHT('Reconstruct Bio objs', objsWave, hg.Z_HKU) #plot reconstructed obj

(objsWaveTrain, objsWaveTest, objsWaveLabelTrain, objsWaveLabelTest)=skm.train_test_split(objsWave, objsWaveLabel,  # shuffle and spilt the dataset, A.3.
                                                 test_size=SPILT_RATIO_4T, random_state=32)

print("!!Targeting trainset shape =", objsWaveTrain.shape, " !!\n")
print("\nNo of samples each class=", classNcount, "\n\n")

#########################################################################################################################
#                   Ensemble magnitude & phrases deep Learning Recognition Start Here                                    #
#########################################################################################################################
mString='Phr_'+modeDict.get(smoothPhr) # get Text description of smoothing methods
nnModel=creatNprintEnModel("modelBioSample"+"EnMode-"+str(CONCATE_M)+".png", rows, columns )

""" a) Trained. Then evaluate """
print("\nTraining...", end=" ")
trainMag, trainLabel=formatDataNlabel( #Magnitude
                        objsWaveTrain, objsWaveLabelTrain, NB_CLASSES, hg.MAGNITUDE_M) # MUST sync SMOOTHING_M with formatDataNlabel()
trainPhr, _             =formatDataNlabel( #Phrase. Label is same as Magnitude
                        objsWaveTrain, objsWaveLabelTrain, NB_CLASSES, hg.PHRASE_M, smoothPhr) # MUST sync SMOOTHING_M with formatDataNlabel()

fitHist, score=fitEval('By Concate Ensemble', nnModel, [trainMag,trainPhr], trainLabel) # MUST sync SMOOTHING_M with formatDataNlabel()

# =============================================================================
#         Evalute Bio Objs  TESTset (non-Trained)
# =============================================================================
objsWaveTestMag, objsWaveLabelTest = formatDataNlabel(objsWaveTest, objsWaveLabelTest, NB_CLASSES, hg.MAGNITUDE_M)
objsWaveTestPhr, _                = formatDataNlabel(objsWaveTest, objsWaveLabelTest, NB_CLASSES, hg.PHRASE_M, smoothPhr)
testScore = nnModel.evaluate([objsWaveTestMag, objsWaveTestPhr], objsWaveLabelTest, batch_size=BATCH, verbose=0) #return_dict=True
print(nnModel.metrics_names, end = ''); print(" Testset Score&Acc fr both:", testScore)

# =============================================================================
#         Evalute Bio Objs  Fullset(adjusted), (Fullset=Test+Train)
# =============================================================================
objsWaveMag, objsWaveLabel= formatDataNlabel(objsWave, objsWaveLabel, NB_CLASSES, hg.MAGNITUDE_M) # MUST sync SMOOTHING_M with convertNsplitData()
objsWavePhr, _               = formatDataNlabel(objsWave, objsWaveLabel, NB_CLASSES, hg.PHRASE_M, smoothPhr) # MUST sync SMOOTHING_M with convertNsplitData()
fullScore = nnModel.evaluate([objsWaveMag, objsWavePhr], objsWaveLabel, batch_size=BATCH, verbose=0)
print(nnModel.metrics_names, end = ''); print("\n Fullset Score&Acc fr both:", fullScore)

# Consoldiate results
hg.plot_history(fitHist, 'Train Accuracy by ', mString) # plot graphs
# nnModel.save('bioModelWgt'+mString+'.h5'); print("Saved network model, Phrase recognitizer's weights and all to disk as "+ "bioModelWgt"+mString+".h5") # Save entire model to a HDF5 file  # A.12

            # .argmax(axis=-1) == .predict_classes(), debug2Jun2021, No .predict_classes fr Keras functional API.
start=time.time(); objPredicted=nnModel.predict([objsWaveMag, objsWavePhr], verbose=0).argmax(axis=-1) #No ..ict_classes fr Keras functional API, A3
print(f'Bio Ensemble fullset predict time={time.time()-start}') #20sep2021
plotConfuMatrix(objsWaveLabel, objPredicted , "Bio Objs by concate Ensemble") #Predict from model trained by BOTH

results = results.append({'Phase Smoothing Mode':mString, # logging key results
                           # 'Phase Noise added in %':20, #ref to shiftImgOccRotoHolo.py 25aug2021
                          'testScore':testScore[1],                    
                          'fullScore':fullScore[1], # =SORT_KEY
                          }, ignore_index=True) 

results.sort_values(by = SORT_KEY, ascending=True, inplace=True) #SORT_KEY='occluedFullScoreFrBoth'. A9. sort Hi2Lo, save inplace
results=results.append({ "Para":[G_epochRan, NB_EPOCHS, BATCH, NB_FILTER, DROPOUT_RATE1, DROPOUT_RATE2, DELTA_E ] }, ignore_index=True)
results.to_excel("bioObjRecByConcateEnsemble.xlsx"); print("Results save into bioObjRecByConcateEnsemble.xlsx")

"""
Reference: A)
1) https://keras.io/api/layers/merging_layers/
2) https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-y-network-2.1.2.py
   https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
3) https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
4) Batch regulisation: https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
"""    
    