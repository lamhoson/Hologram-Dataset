# -*- coding: utf-8 -*-
"""
I = imread('sample58_157.png'); %!! Matlab index start with 1 instead 0
I = I(:,:,1); I = double(I);
I2 = (I - min(min(I)))/(max(max(I) - min(min(I))));
f=fft2(sqrt(I2)); f=fftshift(f);

%%system info constant
dx = 3.45e-6; la = 632.8e-9; ds = 0.1000; de = 0.1200;

Ver 1.0 7Oct2021 - 1st version
Ver 2.0 9oct2021 - use scipy for FFT2, "" replace "np.fft"
ver 2.1 9oct2021 - Add ifftshift(normHologram), !!pre-shift for correcting phase info, Kedar P.72
ver 3.0 10oct2021 - greatly cleaned and simplifed. Chop specturm out instead of mask out.
ver 3.1 11oct2021 - first all runs !!
ver 3.1.1 13oct2021 - add: keep twin frequency-spectrum image as option.
ver 4.0 20oct2021 - ver of saving each sample objectWave into a npy file. 10 samples will have 10 dataset and classNcount NPYs

"""
import cv2 as cv  # OpenCV for imaging operations
import numpy as np # numerical package for multi-dimension arrays
import hgrampy as hg
# from matplotlib import pyplot as plt #; plt.rcParams['font.sans-serif']=['SimHei'] #to handle CHINESE font
from skimage import io    # exposure.equalize_adapthist( (region-region.min())/(region.max()-region.min()) )
# import scipy.fft as spy #for fft2, "" replace "np.fft"
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import astropy.units as u
import copy; import pandas as pd
import gc #  garbage collection,

""" hg.TWIN_O_WAVES moved to hgrampy.py lib """
TOGGLE_LOGIC=False #True=wave, False=reconstructed Image
TWO, FOUR=2, 0 #rotations for extraction at quardant 4
QUARD= FOUR # Extract Quardant 2 or 4. Don't rotate if at Q4
# SAVE_OBJ_W_ONLY= True #False/True. Do'nt save DC cut, twin cut diagrams
DEBUG=True #print max spectrum value to console for observation/debug
DEBUG1=True #False/True , show all intermediate chops
DEBUG2=True #show only the final centered chop
WRKS=8 #max worker for scipy parallel processing.
SCALEUP_RATE=2 #scale-up rate to zero-pad during fzpFFT

LEFT, RIGHT=0,1; EQU, NOT_EQU=0,1 # equalisation or not during plot-save image files.

#     ### For DC cut ###
COLS=65; ROWS= COLS+50 # 150Cannot. ALL SAMPLE SAME !!.  # +360(will cover whole Y-axis)
DC_MASK_COL=np.arange(-COLS, COLS+1)
DC_MASK_ROW=np.arange(-ROWS, ROWS+1)
    ### For windowing out from RIGHT quardant
BOX= 98 #108 # 78. MUST <108,26oct2021. MUST <85,11oct2021(1st sampleSet)
OFFSET=26 #BOX must > OFFSET. magic window box-size & offset to chop out 1st-order

SRC_PATH="D:\\hkuHologram\\DH2(66laserChanged)\\" #2nd set of hologram
# SRC_PATH='D:\\hkuHologram\\DH1(not66&69)\\'
O_PATH="D:\\hkuHologram\\outputs\\"

G_fzpGenEd=False # set No fzp generated at program start.

def cutDC(shfEdSpm): #Cut FFT components close to (0,0) original
    row, col=shfEdSpm.shape[0]//2, shfEdSpm.shape[1]//2 #int divisions. locate center for cut DC
    # shfEdSpm[row, :]=0j #8oct2021, do'nt cut better. kill the x axis#
    tmp = copy.deepcopy(shfEdSpm) #debug, affect works afterward if NOT DEEPCOPY
    for i in DC_MASK_ROW: #Seem min&max magnitudes are all in center region, 6oct2021
        for j in DC_MASK_COL:
            tmp[row+i, col+j]=0j #locate center & cut low-frequencies DC region
            
    return tmp

def spatialFilter(shfEdSpm, cut=LEFT): #shifted Spectrum

    shfEdSpm=cutDC(shfEdSpm)
    leftRight=np.hsplit(shfEdSpm, 2) #crop into 2 halve [1,2,4,5]-> [1,2] [4,5]. https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html
    if cut==LEFT: #Cut left two quardrants        
        mask=np.zeros_like(leftRight[LEFT]) #left
        left=np.multiply(leftRight[LEFT], mask) #cut all -1 order spectrums
        maskedSpectrum= np.hstack((left, leftRight[RIGHT])) #merge back after cut spectrums
    elif cut==RIGHT: #Cut Right two quardrants      
        mask=np.zeros_like(leftRight[RIGHT]) #left
        right=np.multiply(leftRight[RIGHT], mask) #cut all -1 order spectrums
        maskedSpectrum= np.hstack((leftRight[LEFT], right)) #merge back after cut spectrums
            ## Cut twin (-1 order) away##            
            
    return maskedSpectrum

def chopNCtr1stOrder(shfEdSpm, filename): #i/p: shifted Spectrum
    shfEdSpm=spatialFilter(shfEdSpm, cut=LEFT) #clean all HIGH energy not necessary stuffs around DC
    if DEBUG1: dispNSave('Cut DC'+filename.split('\\')[1], filename+'_cutDcSpm', shfEdSpm, mode=NOT_EQU)

    ctrRow, ctrCol=shfEdSpm.shape[0]//2, shfEdSpm.shape[1]//2 #locate center's x-y coordinate. Top-Left=(0,0)
    buffer=copy.deepcopy(shfEdSpm[ctrRow -OFFSET: (ctrRow -OFFSET) +BOX, #move +1Order spectrum into the buffer
                                          ctrCol +2*OFFSET:ctrCol +(2*OFFSET+BOX)])

    if DEBUG1: dispNSave('Centered buffer'+filename.split('\\')[1], filename+'_ctrSpm', buffer, mode=NOT_EQU)
    
    padZeros=int(0.5*BOX) #pad zeros's size at 4 borders
    buffer=np.pad(buffer, (padZeros, padZeros)) #pad zeros to buffer
    if DEBUG1: dispNSave('Centered buffer'+filename.split('\\')[1], filename+'_padZeros', buffer, mode=NOT_EQU)
    r,c=buffer.shape[0]//2, buffer.shape[1]//2 #buffer center's x-y coordinate. Top-Left=(0,0)    
    maxRowCol= np.where(buffer == np.amax(buffer)) #locate max value location as center estimator
    print(f"Center Vs Max pos(center-Max): row at:{r}-{maxRowCol[0]}, col at:{c}-{maxRowCol[1]}") #Center should have max values

    assert abs(r-maxRowCol[0]) < padZeros and abs(c-maxRowCol[1]) < padZeros, 'Outbound Error' #can't shift more than padded zeros
    if DEBUG1: dispNSave('Centered buffer'+filename.split('\\')[1], filename+'_bfCtrAdjSpm', buffer, mode=NOT_EQU) #before center adjustment
    buffer=np.roll(buffer, maxRowCol[0]-r, axis=0) #assumed max is center, move to it. 0-axis:Row/y, 1-axis=Col/x
    buffer=np.roll(buffer, maxRowCol[1]-c, axis=1) # -1ve => rollUp/Left, +ve =>rollDn/Right
    if DEBUG2: dispNSave('Centered buffer'+filename.split('\\')[1], filename+'_afCtrAdjSpm', buffer, mode=NOT_EQU) #after center adjustment
    return buffer

def fftNctrSpm(normHologram): #Refer: fftShiftTests.py in cwd
    spectrum=fft2(np.sqrt(normHologram), workers=WRKS) #assumed square-law detector used    

    if DEBUG: print(f'         Spectrum values: min={np.abs(spectrum.min())}, max={np.abs(spectrum.max())}')
    return fftshift(spectrum) #Shift to center for display purpose. https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.fft.fft.html#scipy.fft.fft

def holo2Objwave(hologram, filename, twin=False):
    normHologram= (hologram - hologram.min())/ (hologram.max()-hologram.min()) #normalize
    
    shfEdSpm=fftNctrSpm(normHologram) #center-shifted spectrum for visualisation
    if DEBUG1: dispNSave('Full shifted Spectrum'+filename.split('\\')[1], filename+'_shfSpm', shfEdSpm, mode=EQU) #MUST EQU, 14oct2021
    shfEdSpm=np.rot90(shfEdSpm, QUARD) #counterClockwise rotate to select quardant
    if not hg.TWIN_O_WAVES:
        shiftChopSpm=chopNCtr1stOrder(shfEdSpm, filename) #auto centered
    else:
        shiftChopSpm=cutDC(shfEdSpm) #contains both real&imaginary spectrums, only cut low-frequencies around DC(0,0)
        shiftChopSpm=hg.chop2objSizeEle(shiftChopSpm,
                    int(shiftChopSpm.shape[0]//9), int(shiftChopSpm.shape[1]//9) )
        if DEBUG1: dispNSave('Twin Spectrum'+filename.split('\\')[1], filename+'_TwinSpm', shiftChopSpm, mode=NOT_EQU) #MUST NOT_EQU, 14oct2021
    
    chopSpm=ifftshift(shiftChopSpm) #UNDO center shift below iFFT calculation. Ref B1, P.D46, 20-23lines

    shfObjWave=ifft2(chopSpm) #INVERSE fft, ??center-shifted already??11oct2021
    if DEBUG:
        print(f'Shifted  Spectrum values: min={np.abs(shfEdSpm.min())}, max={np.abs(shfEdSpm.max())}')
        print(f'Chopped  Spectrum values: min={np.abs(chopSpm.min())}, max={np.abs(chopSpm.max())}')

    return shfObjWave 
    # return shfObjWave, chopSpm, shfEdSpm
    # return fftshift(shfObjWave), chopSpm, shfEdSpm #chopped !, if shift?

def dispNSave(titleStr, filename, signal, mode=EQU, savePh=False, plot=False):# plot=True, plot to Spyder plot window
    region=np.abs(signal)  # get magnitude of all complex values

    if mode==EQU: #Normalize and equalize for visual checks
        region=cv.normalize(region, None, 0, 1024, cv.NORM_MINMAX, dtype=cv.CV_8UC1) # 1024 or >512 is must
        region = cv.equalizeHist(region)

    if plot: hg.plotHolo(titleStr, region.max()-region) #reversed plot
    region=cv.normalize(region, None, 0, 1024, cv.NORM_MINMAX, dtype=cv.CV_8UC1) # 1024 or >512 is must
    io.imsave(filename+".jpg", region, check_contrast=False) #fr skimage module
    # plt.imsave(filename+".jpg", region)
    
    if savePh: #save phase diagram
        tmp=cv.normalize(np.angle(signal), None, 0, 1024, cv.NORM_MINMAX, dtype=cv.CV_8UC1) # 1024 or >512 is must
        if mode==EQU: tmp = cv.equalizeHist(tmp)
        io.imsave(filename+"_p.jpg", tmp, check_contrast=False) 
        # plt.imsave(filename+"_p.jpg", tmp)

def objWave2Image(sftEdObjWave): #i/p center shifted objectWave to match reconImageFFT()
    global G_fzp, G_fzpGenEd #GLOBAL variable
    
    (sftEdObjWave - sftEdObjWave.min())/ (sftEdObjWave.max()-sftEdObjWave.min()) #normalize
    # sftEdObjWave=sftEdObjWave/sftEdObjWave.max() #normalise
    if not G_fzpGenEd: #if did'nt gen Fzp, gen once.       
        print('Generating FZP, wait...')
        height=int(sftEdObjWave.shape[0]*SCALEUP_RATE); width=int(sftEdObjWave.shape[1]*SCALEUP_RATE) #mayNot need as shifted center, 7oct2021
        G_fzp=hg.genFzpFFT(height, width, hg.PSIZE_HKU, hg.LAMDA_HKU, hg.Z_HKU); G_fzpGenEd=True  
    recImage=hg.reconImageFFT(sftEdObjWave, G_fzp) #internally center-shifted for visualisation purpose
    return recImage

# =============================================================================
#                          ### Main program Start HERE ###
# =============================================================================
# 1) Object Wave Extraction
hg.checkRam('Prog start')
dataFile, classNCount, paras = hg.getStoreFilenames()
background=cv.imread(SRC_PATH+ 'sample0_1.png', cv.IMREAD_GRAYSCALE) #the BACKGROUND hologram without any sample&glass-plate
objectsWave=[]; parameters=dict(); objClassNCount=dict()  #empty dictation class to store object classes
for file in hg.BIO_FILENAME: #e.g: {'sample9_', 'sample54_'.....
    # sampleList=np.random.choice(np.arange(1, hg.SIZE_OF_SAMPLE + 1), 2) #pick 3 samples from 1-500
    sampleList=np.arange(1, 2) #hg.SIZE_OF_SAMPLE + 1)
    for s in sampleList: #501/201
        filename= file+str(s) #e.g: {'sample9_', 'sample54_'.....
        nameStr=hg.file2SampleName(filename)+'-'+str(s); print(f"Processing {nameStr} ...", end='  ') #e.g HouseflyWing-1
        hologram=cv.imread(SRC_PATH+filename+'.png', cv.IMREAD_GRAYSCALE) #(0,0)top-left-hand, (63,63)bottom-rigth-hand
        # hologram=np.true_divide(hologram, background) # hologram=hologram - background, Paper 1 normalise. Auto convert to float after divide()
        hologram=hologram.astype(np.float64)

        shfObjWave=holo2Objwave(hologram, O_PATH+nameStr) #cut x half and y half                     
        objectsWave.append(shfObjWave) #shfObjWave is center-shifted (visually readable) version
        dispNSave('ObjectWave '+nameStr, O_PATH+nameStr+'_objWav', shfObjWave, mode=NOT_EQU, savePh=True, plot=TOGGLE_LOGIC) # plot=True, plot to Spyder plot window
        
        if not hg.TWIN_O_WAVES:            
            recImage=objWave2Image(shfObjWave)
            recImage=hg.chop2objSizeEle(recImage, #reconstruct & display some images
                    int(recImage.shape[1]/(SCALEUP_RATE-0.15)), int(recImage.shape[1]/(SCALEUP_RATE-0.15)) ) # silightly scale dn to elimin border distorts
            dispNSave('Reconstr Image '+nameStr, O_PATH+nameStr+'_recIm', recImage, mode=NOT_EQU, plot=not TOGGLE_LOGIC)
        
    objClassNCount.update({hg.file2SampleName(filename): (s)}) #go next sample
    hg.checkRam(f'Just after: {filename}')
    print(f"\n Objects:{filename} wavefront & others save in {O_PATH} .npy(s), ...."); print(objClassNCount)
    np.save(O_PATH +dataFile+f"_{nameStr.split('-')[0]}", objectsWave) #auto convert to np's array from list
    del objectsWave; gc.collect(); objectsWave=[] #not keep for next class
    np.save(O_PATH +classNCount+f"_{nameStr.split('-')[0]}", objClassNCount) #keep for next class

parameters.update({"hImageW":hologram.shape[0], "hImageH":hologram.shape[1],  # save them as documentations
                   "waveW":shfObjWave.shape[0], "waveH":shfObjWave.shape[1], #"TwinMode": hg.TWIN_O_WAVES, 
                    "Twin Mode":hg.TWIN_O_WAVES, "SampleNames":hg.SAMPLE_NAME,
                   "psize":hg.PSIZE_HKU*u.um, "lambda":hg.LAMDA_HKU*u.um, "fzp distance":hg.Z_HKU*u.m, })
np.save(O_PATH +paras, parameters) # save them as documentations

df=pd.DataFrame.from_dict(parameters, orient='index') #; df=df.append(digClass, ignore_index=True) # A11 & A10
df.to_excel(O_PATH + paras+'.xlsx') #A9

if len(sampleList) < hg.SIZE_OF_SAMPLE: print(f'     Minor warning....., sample size smaller than {hg.SIZE_OF_SAMPLE}')

""" Ref:
https://pypi.org/project/pyoptica/

Joseph W. Goodman (2004) Introduction to Fourier Optics, W. H. Freeman",
Kedar Khare (2016) Fourier Optics and Computational Imaging, Wiley&Sons Ltd.",
David Voelz (2011) Computational Fourier Optics Matlab Tutorial, Spie Press"

Paper:
    1) AngularSpecturm algorithms for simulation and reconstruction of inline hologram.pdf
"""
