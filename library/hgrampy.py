# -*- coding: utf-8 -*-
"""  Python Module Name: hgrampy.py 
Ver 1.0   29 Nov 2018.  First release as holopy
Ver 2.07  15 Dec 2018.  Release as hgrampy.  convolve2() options changed to mode='same', boundary='fill'.
Ver 2.1   21 Dec 2018   Minor touch-ups
Ver 2.2   28 Dec 2018   Add holoImRadWrite() and touch-up some descriptions
Ver 3.1    6 Mar 2019   Add debugged normHoloFrImgOrWrpNoise() and restructure all crHologram**() functions
Ver 3.2   26 Jun 2019   Bug fix np.load, E13, allow_pickle=True is a MUST for lastest numpy library
Ver 3.3    8 Aug 2019   Add plot_history()
Ver 3.4   18 Feb 2020   Add plot_historyTFv1() delicated for TF Ver 1.x
Ver 3.5   27 Feb 2020   Add buildLabelsNcountImagesNEW() - build label. RESHAPED the label 27feb2020
ver 3.6   1  Jun2021    Add more cross papers constants:
                        R/X/Y/Z_OPTIONS and  FILE_IDX_IMG/OCC
ver 3.7   16jun2021     Added APIs for generate Hologram and reconstruct the object waves by FFT method
ver 3.7.1 19jun2021     TF FFT added but commented, not much faster than Anconada-Intel numpy in i7-10cores
ver 3.8   30jul2021     genHologramFFT() able to handel complex-number array(e.g. hologram)
ver 3.8.1 10sep2021     reconPlotHT(.., grams,..) internally convert input to np array if it is not np array. debug10sep2021
ver 3.8.2 14apr2022     change DATA_PATH to "c:\\genDataset\\cache\\"
                        add plotConfuMatrix() and 

Tested under:
1. Spyder, Scientific PYthon Development EnviRonment: https://www.anaconda.com/download/
2. Need to install OpenCV2 as not contained in Anaconda-Spyder IDE.
conda install -c conda-forge opencv 
    https://pypi.org/project/opencv-python/
    pip install opencv-python
    pip install opencv-contrib-python


Functions:
    crHologram()            - generate hologram object wave from a image file. Compatible to Matlab version
    normHoloFrImgOrWrp()      - direct take in numpy image array
    normHoloFrImgOrWrpNoise() - addition of speckle phrase-noise
    holoRecon()             - reconstruct hologram at specified depth plane. Compatible to Matlab version
    imshow()                - normalise reconstructed images from hologram to display computer screen as [0.255] gray image
    holoImShow()            - display complex pixel's magnitude
    holoImRadShow()         - display complex pixel's radian angle
    holoImWrite()           - save complex pixel's magnitude image diagram
    holoImRadWrite()        - save complex pixel's phrase radian angle image diagram
    guassianDreaming()      - direct augment complex numbers hologram by guassian distribution, 21Mar09
    

    loadNpStrucArrAsDict()  - quickly load numpy structural array to become simple Python dictionary
    buildLabelsNcountImages() - build label. RESHAPED the label 27feb2020
@author: hoson
"""
import cv2 as cv  # OpenCV for imaging operations
import numpy as np # numerical package for multi-dimension arrays
from scipy import signal as dsp # dsp functions
from scipy.stats import truncnorm #truncated normal distribution
from sklearn.metrics import confusion_matrix; import seaborn as sns
import matplotlib.pyplot as plt; import pandas as pd
import math as ma; import os
import psutil #; import gc #  garbage collection, clear memory by del 'variable' #for memory usage checking

__all__ = ( #Only listed APIs are imported when using "from mymodule import * "
            'genFzpFFT genHologramFFT ' #MUST leave a SPACE before '
            'rHologram genFzp ').split() 

TWIN_O_WAVES=False #False/ True # Phase MAY CANCEL OUT! Gen +1 & -1 order object waves, only cut 0-order DC. 
PLOT_HT = True #False # True to plot head & tail hologram/vdpGram
TRI, BI=3,2; MULTI_MODE=TRI; EXT = "_Tri.jpg" if MULTI_MODE==TRI else "_Bi.jpg"
THICKEST=16 #6 #in pixels, thickest object further from the center-digit
DATA_PATH="c:\\genDataset\\cache\\"  #faster C drive

DEFAULT_OPT_CF=dict( # Default Optical Configuration Parameters. E.g.: para=DEFAULT_CF.copy()
    PSIZE=6.4e-06, #pixel size in meter
    LAMBDA=5.4e-07, # wavelenght of the coherent light used. green look
    Z_DISTANCE = 0.016, # distance from Fresnel Plate to the object
    FZP_RATIO=1.5 #2 #scale-up ratio from image-size for fzp plane, e.g rows/cols*FZP_RATIO
    # H_HEIGHT=192,
    # H_WIDTH=192
                )
# =============================================================================
# LETO-3 : https://holoeye.com/wp-content/uploads/Spatial_Light_Modulators.pdf
# Resolution 1920 x 1080 Pixel
# Pixel Pitch 6.4 μm
# 420-650 nm 500nm=0.5um
# 420-800 nm
# 650-1100 nm
# =============================================================================

NORM_MAX, NORM_MINMAX=0,1 # for hologram normailsation methods
NOISE_PERCENT={'0%':0, '5%':0.3141, '10%':0.6282, '15%':0.9423, '20%':1.2564, '100%':3.141*2} ; NOISE=NOISE_PERCENT['15%'] #0/5/10/15/20%. 0=disable the noise adding process.

# SYSTEM CONSTANTS
# MAGNITUDE_M=1; PHRASE_M=0 # magnitude or phrase modes
PHRASE_M, MAGNITUDE_M, PHR_MAG_M=0, 1, 2 # magnitude or phrase modes
# for complex number hologram direct augmentation by phrase shifts
PHRASE_AUGMENT_BIG = 0.9423 # add 15% phrase variations, holoGram(N,M)
PHRASE_AUGMENT_SMALL = 0.6  # add 10%minus phrase variations, holoGram(N,M)

LOOK_UP_NUM ={"zero64x64":"0", "one64x64":"1", "two64x64":"2", "three64x64":"3", "four64x64":"4", "five64x64":"5",
           "six64x64":"6", "seven64x64":"7", "eight64x64":"8", "nine64x64":"9" }
LOOK_UP_STR ={0:"zero64x64", 1:"one64x64", 2:"two64x64", 3:"three64x64", 4:"four64x64", 5:"five64x64",
           6:"six64x64", 7:"seven64x64", 8:"eight64x64", 9:"nine64x64" }

FILE_IDX_IMG=("zero64x64", "one64x64", "two64x64", "three64x64", "four64x64", "five64x64",
           "six64x64", "seven64x64", "eight64x64", "nine64x64")
FILE_IDX_OCC=("occObjSame0", "occObjSame1", "occObjSame2", "occObjSame3", "occObjSame4", "occObjSame5",
            "occObjSame6", "occObjSame7", "occObjSame8", "occObjSame9")

# combinations of x-y-z shifts, rotate etc...
X_OPTIONS=(-3,-2,0,2,3)  # full x direction shift=[-3,-2,-1,0,1,2,3]
Y_OPTIONS=(-3,-2,0,2,3)   # full y direction sHift=[-3,-2,-1,0,1,2,3]
Z_OPTIONS=(-2,-1,0,1,2) # full z=[-2,-1,0,1,2]. One unit is 5%, 0.000xm=(0.0xm/5)/10/2
R_OPTIONS=(-30,-15,0,15,30) # rotate in degree [-30,-15,0,15,30], 30 used in TWO papers
    ### ONLY for the Occlude obj shift&Holo
X_OCC_OPTIONS=(-4, -7, 6, 5, 7) #xOptions=np.random.randint(-8, 8, size=5).flatten() #flat to tuple like
Y_OCC_OPTIONS=(-8,-4,0,4,8)   # found by trial&Error
X_OPT_Dict = dict(zip(X_OPTIONS, X_OCC_OPTIONS)) #1-to-1 Image shifts to OccludeObj Shifts
Y_OPT_Dict = dict(zip(Y_OPTIONS, Y_OCC_OPTIONS)) #1-to-1 Image shifts to OccludeObj Shifts

"""                         HKU STUFFS """
# BIO_FILENAME=('sample9_', 'sample54_', 'sample58_', 'sample59_', 'sample60_', 'sample65_', )
# SAMPLE_NAME={ 9:'ApiPlant',   54:'EugInsect',     58:'EarthWornCS', #avoid Chinese characters
#             59:'HouseflyWing', 60:'HoneyBeeWing', 65:'FlyLeg'}
# BIO_FILENAME=('sample0_', 'sample13_', 'sample18_', 'sample50_', 'sample59_', 'sample60_',  #xx_1-502.png
#    'sample69_', 'sample86_', 'sample91_', 'sample92_', 'sample100_',)# 'sample66_')
# SAMPLE_NAME={ 0:'background', 13:'StemCucurbita', 18:'PineStem', 50:'SeedZea', 59:'HouseflyWing', 60:'HoneyBeeWing', #Avoid CN for matplotlib. 9:'Apical Bud芽', 54:'Euglena虫'
#    69:'BirdFeather', 86:'CorpusVentriculi', 91:'LiverSection', 92:'LymphNodeSection', 100:'ChromosomeHuman',} #66:'HoneyBeeLeg', } #66's laser adjusted(brighter)
BIO_FILENAME=('sample13_', 'sample18_', 'sample50_', 'sample59_', 'sample60_',  #xx_1-502.png
   'sample69_', 'sample86_', 'sample91_', 'sample92_', 'sample100_',)# 'sample66_')
SAMPLE_NAME={13:'StemCucurbita', 18:'PineStem', 50:'SeedZea', 59:'HouseflyWing', 60:'HoneyBeeWing', #Avoid CN for matplotlib. 9:'Apical Bud芽', 54:'Euglena虫'
   69:'BirdFeather', 86:'CorpusVentriculi', 91:'LiverSection', 92:'LymphNodeSection', 100:'ChromosomeHuman',} #66:'HoneyBeeLeg', } #66's laser adjusted(brighter)


PSIZE_HKU = 3.45e-6 #dx=dy = 3.45e-6. My case=6.4e-06
Z_HKU= 0.105 #0.10#0.11 #ds = 0.1000; de = 0.1200;. My case=0.016
LAMDA_HKU = 632.8e-9 #la = 632.8e-9, red look. My case=5.4e-07(green). https://www.sciencelearn.org.nz/resources/47-colours-of-light
SIZE_OF_SAMPLE=500

def file2SampleName(filename): #get biological name from filename (samle??_.png)
    return SAMPLE_NAME[ int(filename.split('_')[0].split('sample')[1] ) ] #lookUp from dictionary

def getStoreFilenames():
    dataFile = 'readableWaveDataset'
    objClassNCount = "objClassNCount"
    paras='readableWaveParas'
    if TWIN_O_WAVES:
        dataFile = dataFile +'_t' #twin version
        objClassNCount = objClassNCount +'_t'
        paras= paras +'_t'
        print('In TWIN spectrum mode !!!!!')
    else: print('In normal Q4 spectrum non-twin mode !!!!!')
        
    return dataFile, objClassNCount, paras
"""                         HKU STUFFS """

""" Generate Hologram and reconstruct object by FFT methods """
def genFzpFFT(height, width, psize,lamda,z): #height=row, width=col
    """
    Generate spectrum of FZP by FFT

    Parameters
    ----------
    height : Height is index by row
    width  : Width is index by column
    psize : pixel size
    lamda : wavelength
    z : distance of FZP to the object

    Returns: fzp, the Fourier Transform of the Frensel Plate ( for Convolute(A,B)=iFFT(FFT(A)*FFT(b))

    """
    fr=np.array(np.zeros((height,width)))
    fi=np.array(np.zeros((height,width)))
    wn=2*ma.pi/lamda
    hDim=int(height/2); wDim=int(width/2)
    for row in range(0,height):
        for col in range(0,width):
            # x=(row-wDim)*psize
            # y=(col-hDim)*psize
            y=(row-wDim)*psize #x-y swapped, Hoson 17jun2021
            x=(col-hDim)*psize #x-y swapped, Hoson 17jun2021
            r=wn*ma.sqrt(x**2+y**2+z**2)
            vr=ma.cos(r)
            vi=ma.sin(r)
            fr[row][col]=vr
            fi[row][col]=vi

    fzpFFT=np.fft.fft2(fr+1j*fi)
    return fzpFFT

def myCopyMakeBorder(array2D, top, bottom, left, right): #add top,bottom with adjustment for np.pad()
    # array2D = array2D.astype(complex)
    if array2D.dtype !=complex: array2D = array2D.astype(complex)
    
    array2D=np.pad(array2D, (left, right), 'constant', constant_values=(0, 0)) # debug 6aug2021, .pad result is x-y symmetric&sqaure, can't long-tall rect
    top=top-left; bottom=bottom-right #np.pad will fill top with left, bottom with right, need to deduct them.
    if top >0: #debug10sep2021 for tall picture.
        upper=np.zeros((top, array2D.shape[1]) ); lower=np.zeros((bottom, array2D.shape[1]) )
        array2D=np.vstack((upper,array2D)) # (upper, lower)
        array2D=np.vstack((array2D, lower))
    elif top <0: # 2 x top (top-bottom both sides) overfilled, but left-right are right
        array2D=chop2objSizeEle(array2D, array2D.shape[1], array2D.shape[1]) #square center-chop according to left-right
    # top==0, then do nothing
    
    return array2D

def genHologramFFT(img, fzpFFT): #FFT instead of convolution method
    """
    Able to handle complex-number array, 30jul2021
    Able to handle both short (8,8,8) or tall (960x720 iPhone) picture
    FFT the image and multipy with fzpFFT, zero-center and return. 
    
    Parameters
    ----------
    fzpFFT : FFT of Fresnel Plate
    img : real-number image

    Returns: Hologram of 'img'. Same result as convolution by iFFT(fzpFFT * FFT(img))

    """
            ### adjust img to make sure SAME size with the fzpFFT array
    delta_w = fzpFFT.shape[1] - img.shape[1] #col
    delta_h = fzpFFT.shape[0] - img.shape[0] #row
    top, bottom = delta_h//2, delta_h-(delta_h//2) #fill length for top & bottom
    left, right = delta_w//2, delta_w-(delta_w//2) #fill length for left & right

    # image= cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0]) #pad zero, dark, https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
    image= myCopyMakeBorder(img, top, bottom, left, right) #this handle complex instead of just real.
    assert fzpFFT.shape==image.shape, 'Error! Fzp Vs Image sizes mismatch!'    
    
    A=np.fft.fft2(image)
    B=np.multiply(A,fzpFFT) # inputs MUST SAME sizes
    H=np.fft.ifft2(B) # inverse to get same result by convolution.
    return np.fft.fftshift(H) #Shift zero-frequency component to center of the spectrum

def reconImageFFT(H, fzpFFT): #FFT instead of convolution method
    """
    FFT the Hologram & multipy with conjugate of the fzpFFT, inverse FFT and centering

    Parameters
    ----------
    fzpFFT : FFT spectrum of a Frensel Plate
    H : input Hologram

    Returns: reconstracted image (but still complex pixels)

    """
            ### adjust Holo/VdpGram to make sure SAME size with the fzpFFT array
    delta_w = fzpFFT.shape[1] - H.shape[1] #col
    delta_h = fzpFFT.shape[0] - H.shape[0] #row
    top, bottom = delta_h//2, delta_h-(delta_h//2) #fill length for top & bottom
    left, right = delta_w//2, delta_w-(delta_w//2) #fill length for left & right

    H= myCopyMakeBorder(H, top, bottom, left, right) #this handle complex instead of just real.
    assert fzpFFT.shape==H.shape, 'Error! Fzp Vs Holo/VDPgram sizes mismatch!'  
    
    A=np.fft.fft2(H) #fft the Hologram
    B=np.multiply(A, np.conj(fzpFFT)) # times conjugate of fzpFFT
    I=np.fft.ifft2(B) #iFFT, has same result as convolution method.
    return np.fft.fftshift(I) #Shift zero-frequency component to center of the spectrum

""" Generate Hologram Object Waves by the convolution-based method
"""
def crHologram(nameString,z,psize,lamda, mode='same'):
    TMP=cv.imread(nameString ,cv.IMREAD_GRAYSCALE)     # Load image, BRG, as grey scale
    H=crHologramDirect(TMP,z,psize,lamda, mode)
    return H # return the Hologram object wave

def genFzp(N, M, z, psize, lamda, conjugate=1j):
    """
    Generate Fresnel Zone, or conjugate of Fresnel Zone Plate if conjugate=-1j

    Parameters
    ----------
    N : y-axis, 0 at top left hand corner. N,M =height,widht,31jul2021
    M : x-axis, 0 at top left hand corner

    z : distance from the hologram to the object, 16jun2021
    psize : pixel size of the hologram
    lamda : wavelength of the light
    conjugate :  The default is 1j, NOT complex-conjugate which for reconstruct/decode instead encode

    Returns: The fzp or conjugate if conjugate=-1j

    """
    wn=2*np.math.pi/lamda      #Wave number
    z2=z*z  # square of the depth value z
    x=np.arange(-M/2, M/2, 1) # start: -M/2,stop: M/2, step: 1. = [-M/2, -M/2)
    y=np.arange(-N/2, N/2, 1) # = [-N/2, N/2-1]
    kx, ky=np.meshgrid(x,y)   # =Matlab's (-M/2:M/2-1,-N/2:N/2-1)
    kx=kx*psize;    # Physical horizontal position
    ky=ky*psize;    # Physical vertical position
    kz=conjugate*wn* np.sqrt(kx**2+ky**2+z2) # 1j=0+ j. **= Matlab's ^ .
    fzp=np.exp(kz) # generate Fressel Zone Plane at the specific depth's plane.
        
    return fzp

""" Take in numpy image array directly instead of the image's file """
def crHologramDirect(imageArray,z,psize,lamda, noise=False, mode='same'):
    """
    Generate hologram by direct Input image array instead by image's filename

    Parameters
    ----------
    imageArray : image array
    z : z-depth distance
    psize : pixel size
    lamda : wavelength
    mode : convolution mode, same/valid/fill
        DESCRIPTION. The default is 'same'.

    Returns
    -------
    TYPE compkex number array
        DESCRIPTION. The generated hologram at the distance specified  by z

    """
    I=cv.normalize( imageArray.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX ) # copy as float if not float, normalize to [0.0,1.0]. E.7 & C.2. 
    N, M=I.shape  #  =Matlab size()  D.1. below

    """ Double check crHologramDirectNoise() if any changes in following. They MUST SAME """
    fzp=genFzp(N, M, z, psize, lamda, conjugate=1j)
    
    if noise: I=addGuassNoise1Sided(I, high=NOISE)
    H=dsp.convolve2d(I, fzp, mode= mode, boundary='fill') # D.2. below. mode='same'/'full', boundary='symm'/'fill'

    return H # return the Hologram object wave

def addGuassNoise1Sided(imageArray, high=0.6282): #0.6282 is 10% single-side swing
    """ i/p image can be complex number pixels (holograms) or real number pixel, 29jul2021
      Add 1-sided (+ve side) & truncated (range limited) guassian phrase noise to imageArray.
      e.g    5% of single-sided swing => 0.05*(2pi)=0.3141 radian 
            10% of single-sided swing => 0.1*(2pi)=0.6282 radian
            15% of single-sided swing => 0.15*(2pi)=0.9423 radian
            20% of single-sided swing => 0.2*(2pi)=1.2564 radian

    I(n,m)= i(n,m)*exp(i@), @(n,m) in [0,2pi)

    Parameters
    ----------
    imageArray : input imageArray
    low : 0 to force 1-side avoid -ve phrase, which is big (~2pi) and turn background dark pixels to bright pixels
    high : The default is 2*0.1*pi, +10% range [0, xx]

    Returns the noise added imageArray
    -------
    TYPE a complex-number array

    """
    np.random.seed() #reset the seed everytime to make o/p more random.
    assert high< 2*ma.pi, 'Noise MUST smaller then 2*pi'; low=0 #fix lower bound at zero
    phi = 1j * truncnorm.rvs(low, high, size=imageArray.shape) #A14, gen randomVariable

    noise = np.exp(phi)
    imageArray = imageArray * noise  # add unit-magnitude complex noise into the image, phrase only changes
    return imageArray # return noised imageArray

def crHologramDirectSized(imageArray, height, width,z,psize,lamda, noise=False, mode='same'):
    """
    Can specify generated hologram's size by flipped to (fzp,I) instead (I,fzp)

    Parameters
    ----------
    imageArray : real-pixel image
    height : 
    width : 
    z : distance from hologram to object
    psize : pixel size
    lamda : wavelength
    mode : default is 'same' size of 1st input imageArray

    Returns: the generated hologram

    """
    imageArray=cv.normalize( imageArray.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX ) # copy as float if not float, normalize to [0.0,1.0]. E.7 & C.2. 
    N, M=height, width  #  =Matlab size()  D.1. below

    """ Double check crHologramDirectNoise() if any changes in following. They MUST SAME """
    fzp=genFzp(N, M, z, psize, lamda, conjugate=1j)
    
    if noise: imageArray=addGuassNoise1Sided(imageArray, high=NOISE)
    H=dsp.convolve2d(fzp, imageArray, mode= mode, boundary='fill') # commutative: http://www.songho.ca/dsp/convolution/convolution_commutative.html

    return H # return the Hologram object wave    

""" Reconstructure a Hologram at specific distance """
def holoRecon(H,z,psize,lamda, mode='same'):
    """
    Reconstructure Hologram. mode MUST sync with crHologram/normHoloFrImgOrWrp

    Parameters
    ----------
    H : hologram
    z : FZP 
    psize : pixel size
    lamda : wavelength
    mode : convolution mode, same/valid/fill. The default is 'same'.

    Returns
    -------
    None.

    """
    N, M=H.shape # Size of hologram, D1. 
    fzp_c=genFzp(N, M, z, psize, lamda, conjugate=-1j) # Generate Conjugate of FZP 
    # fzp=genFzp(N, M, z, psize, lamda, conjugate=1j); fzp_c=np.conj(fzp) # Generate Conjugate of FZP 
# Compute reconstructed image 
    IR=dsp.convolve2d(H, fzp_c, mode=mode, boundary='fill') # D.2. above. mode='same'/'full', boundary='symm'/'fill'
    return IR # return the reconstructed hologram image at the specified depth plane

def vdp2HoloUnNormise(vdpGrams, z, psize, lamda, mode='same'):
    """
    Generate hologram by from vdpGrams

    Parameters
    ----------
    vdpGrams : image array
    z : z-depth distance
    psize : pixel size
    lamda : wavelength
    mode : convolution mode, same/valid/fill
        DESCRIPTION. The default is 'same'.

    Returns
    -------
    TYPE compkex number array
        DESCRIPTION. The generated hologram at the distance specified  by z

    """
    # I=cv.normalize( vdpGrams.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX ) # copy as float if not float, normalize to [0.0,1.0]. E.7 & C.2. 
    N, M=vdpGrams.shape  #  height,widht. =Matlab size()  D.1. below

    """ Double check crHologramDirectNoise() if any changes in following. They MUST SAME """
    fzp=genFzp(N, M, z, psize, lamda, conjugate=1j)
    H=dsp.convolve2d(vdpGrams, fzp, mode= mode, boundary='fill') # D.2. below. mode='same'/'full', boundary='symm'/'fill'

    return H # return the Hologram object wave

""" Show reconstructed images from hologram to computer screen's display. Pixels normalization to [0,255] on Computer Screen as unsigned 8-bit pixels"""
def imshow(WinTitleString, imageF): 
    TEMP=cv.normalize(imageF, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # squeeze to [0,255], unsigned 8-bit, E.8 & E.9.
    cv.imshow(WinTitleString, TEMP) # PLot to a pop-up window
    return TEMP # return displayed image. The HoloGReconRelpaceML used this TEMP return variable

""" Show Hologram pixels (complex numbers) magnitudes on Computer Screen as unsigned 8-bit pixels"""
def holoImShow(WinTitleString, H): 
    TEMP=np.absolute(H)  # get magnitude of all complex values
    TEMP=TEMP/TEMP.max()  # element-wise numpy array division. Pixel normalisation
    imshow(WinTitleString, TEMP) # it will handle cv.normalize before save file as above
    return TEMP # return float normalized info of the displayed image

def plotHolo(titleString, realNumPixel): #CAN be used by real-number-pixel, in addition to complex-pixel image
    # magImage=np.absolute(complexPixels)    #magnitude's image
    dis=cv.normalize(realNumPixel, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # squeeze to [0,255], unsigned 8-bit, E.8 & E.9.
    # realNumPixel=realNumPixel/np.max(realNumPixel)
    plt.figure(1)
    plt.imshow(dis, cmap='Greys')
    plt.title(titleString)
    plt.show()
    return dis # normalized magnitude image

def plotHeadTail(titleStr, gramsArray, mode=MAGNITUDE_M): # plot the first and last few holo/vdpGrams
    locations=(0, -1, int(len(gramsArray)/2) ) #head, tail & middle
    
    if PLOT_HT: #False to disable for speeding up stuffs.
        for i in locations: #show less
            if mode==MAGNITUDE_M:
                plotHolo(titleStr+f'_{i}', np.absolute(gramsArray[i]) )
            elif mode==PHRASE_M:
                plotHolo(titleStr+f'_{i}', np.angle(gramsArray[i]) )

def reconPlotHT(titleStr, holoVdpGrams, reconDistance=DEFAULT_OPT_CF['Z_DISTANCE']): #Plot head-tail reconstruced grams
    if isinstance(holoVdpGrams, np.ndarray) is False: holoVdpGrams=np.asarray(holoVdpGrams) #debug10sep2021, convert to numpy array if not
    
    locations=(0, -1, int(len(holoVdpGrams)/2) ) #head, tail & middle
    print('Reconstructing from holograms or vdpGrams.....')

    _, height, width=holoVdpGrams.shape #; reconstructedGrams=np.zeros_like(holoVdpGrams) #init buffer for vdpGrames
    fzpVdpFFT=genFzpFFT(height, width, DEFAULT_OPT_CF['PSIZE'], DEFAULT_OPT_CF['LAMBDA'], reconDistance) #assume H_HEIGHT==H_WIDTH
    for i in locations: #show less
        tmp=reconImageFFT(holoVdpGrams[i], fzpVdpFFT) #back propagate to vdp plane
        plotHolo(titleStr+f'_{i}', np.absolute(tmp) )

def reconPlot(titleStr, holoVdpGram, reconDistance=DEFAULT_OPT_CF['Z_DISTANCE']): #Plot ONE reconstruced grams
    if isinstance(holoVdpGram, np.ndarray) is False: holoVdpGram=np.asarray(holoVdpGram) #debug10sep2021, convert to numpy array if not
    print('Reconstructing from a holoGram or a vdpGram.....')

    height, width=holoVdpGram.shape #; reconstructedGram=np.zeros_like(holoVdpGram) #init buffer for vdpGrames
    fzpVdpFFT=genFzpFFT(height, width, DEFAULT_OPT_CF['PSIZE'], DEFAULT_OPT_CF['LAMBDA'], reconDistance) #assume H_HEIGHT==H_WIDTH
    tmp=reconImageFFT(holoVdpGram, fzpVdpFFT) #back propagate to vdp plane
    plotHolo(titleStr, np.absolute(tmp) )


""" Show Hologram pixels (complex numbers) phrase angle in radian on Computer Screen as unsigned 8-bit pixels"""
def holoImRadShow(WinTitleString,H):
    TEMP=np.angle(H) #get angle in radian
    TEMP=TEMP/TEMP.max()
    imshow(WinTitleString, TEMP) # it will handle cv.normalize before save file as above.
    return TEMP # return float normalized info of the displayed image

def holoImWrite(fileNameString, H): # save magnitude of Hologram as an image. Complex->Mag->float-normal->[0,255]
    IR=np.absolute(H)  # get magnitude of all complex values
    IR=IR/IR.max()  # element-wise numpy array division. Pixel normalisation
    TEMP=cv.normalize(IR, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # squeeze to [0,255], unsigned 8-bit, E.8 & E.9.
    assert cv.imwrite(fileNameString+".jpg", TEMP), 'API holoImWrite error!' #imwrite ret True if success.
    # cv.imwrite(fileNameString+".jpg", TEMP)
    return

def holoImRadWrite(fileNameString, H):  # save phrase angle of Hologram as an image. Complex->Phrase->float-normal->[0,255]
    IR=np.angle(H)  # get angle of all complex values
    IR=IR/IR.max()  # element-wise numpy array division. Pixel normalisation
    TEMP=cv.normalize(IR, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # squeeze to [0,255], unsigned 8-bit, E.8 & E.9.
    cv.imwrite(fileNameString+".jpg", TEMP)
    return

def loadNpStrucArrAsDict(fileString): #load numpy structural array, convert to simple Python dictation immediately.
    classNcount=np.load(fileString, allow_pickle=True); classNcount=classNcount.item() #A13 for pickle=True
    return classNcount

def plot_history(history, titleString, mString): # work for TF1&2 now, 6jun2021. plot accuracy Vs validation loss convergency trend 
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure() # Create a new figure. A bigger size plot with Accuracy ONLY
  plt.title(titleString + mString)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['acc'], label='Accuracy') # work for TF1&2 now, 6jun2021.  plt.plot(hist['epoch'], hist['accuracy'], label='Accuracy')
  plt.ylim([0,1.1]) # set y-axis largest limit
  plt.legend() # Place a legend on the axes
  plt.savefig(titleString + mString+"[0]")

  plt.figure() # Create another new figure.  cmp acc Vs loss
  plt.title(titleString + mString)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['acc'], label='Accuracy') # work for TF1&2 now, 6jun2021.  plt.plot(hist['epoch'], hist['accuracy'], label='Accuracy')
  plt.plot(hist['epoch'], hist['loss'],label = 'Loss')
  plt.ylim([0,1.1])
  plt.legend()
  plt.savefig(titleString + mString+"[1]")  

  if 'val_loss' in hist.columns: # plot only in validataion mode SET
      plt.figure() # cmp validation set's acc Vs loss
      plt.title("From Validation Set" + mString)
      plt.xlabel('Epoch')
      plt.ylabel('Validation Accuracy')
      # plt.plot(hist['epoch'], hist['val_accuracy'], label='Accuracy')  #A4
      # plt.plot(hist['epoch'], hist['val_loss'], label = 'Loss')
      plt.plot(hist['epoch'], hist['val_acc'], label='Accuracy')  # work for TF1&2 now, 6jun2021
      plt.plot(hist['epoch'], hist['val_loss'], label = 'Loss') # work for TF1&2 now, 6jun2021
      plt.ylim([0,1.1])
      plt.legend()
      plt.savefig(titleString + mString+"[2]")
  
  plt.show(); plt.close(fig='all') #Display, close all figure.
  return

def plot_historyTFv1(history, titleString, mString): # plot accuracy Vs validation loss convergency trend 
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure() # Create a new figure. A bigger size plot with Accuracy ONLY
  plt.title(titleString + mString)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['acc'], label='Accuracy')
  plt.ylim([0,1.1]) # set y-axis largest limit
  plt.legend() # Place a legend on the axes
  plt.savefig(titleString + mString+"[0]")

  plt.figure() # Create another new figure.  cmp acc Vs loss
  plt.title(titleString + mString)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['acc'], label='Accuracy')
  plt.plot(hist['epoch'], hist['loss'],label = 'Loss')
  plt.ylim([0,1.1])
  plt.legend()
  plt.savefig(titleString + mString+"[1]")  

  if 'val_loss' in hist.columns: # plot only in validataion mode SET
      plt.figure() # cmp validation set's acc Vs loss
      plt.title("From Validation Set" + mString)
      plt.xlabel('Epoch')
      plt.ylabel('Validation Accuracy')
      plt.plot(hist['epoch'], hist['val_acc'], label='Accuracy')  #A4
      plt.plot(hist['epoch'], hist['val_loss'], label = 'Loss')
      plt.ylim([0,1.1])
      plt.legend()
      plt.savefig(titleString + mString+"[2]")
  
  plt.show(); plt.close(fig='all') #Display, close all figure.
  return

def buildLabelsNcountImages(classNcount, fileIndex):
    labels=np.full((classNcount.get(fileIndex[0])),0) # init 1st labels for looping, ZERO, fill '0'
    count=classNcount.get(fileIndex[0])

    i=1  # loop start with one, '1'
    for file in fileIndex[1: :1]: # start:stop:increment, start 1, to end no stop, 1 each step 
        temp=np.full((classNcount.get(file)),i); i+=1  # digit 'i' for one, two...,nine
        labels=np.vstack((labels,temp))  # vertically stack up the array
        
        count +=classNcount.get(file) # count total numbver of images from the dictionary
        
    return labels, count

def buildLabelsNcountImagesNEW(classNcount, fileIndex):
    labels, count=buildLabelsNcountImages(classNcount, fileIndex)
        
    labels=labels.reshape(count,1) # count=nbImages e.g. (25000, 64, 64), 25k 64x64 images. 27feb2020
    return labels, count

def buildChoppedLabels(classLabelPairs:list): #this API NOT tested yet,28jul2021
    LABEL, COUNT=0,1 #help documentation below
    labels=np.full(classLabelPairs[0][COUNT], classLabelPairs[0][LABEL]) # init 1st labels for looping, ZERO, fill '0'
    count=classLabelPairs[0][COUNT] #init 1st count

    for pair in classLabelPairs[1: :1]: # start:stop:increment, start 1, to end no stop, 1 each step 
        temp=np.full(pair[COUNT], pair[LABEL]) #pair[1] is count of the label value. .full(shape, fill_value)
        labels=np.vstack((labels,temp))  # vertically stack up the array
        
        count +=pair[COUNT] # count total numbver of images from the dictionary
        
    return labels, count

def guassianDreaming(holoGram): #direct augment complex hologram by guassian distribution, 21Mar09
    np.random.seed() #E11&12. reset the seed everytime to make o/p more random.
    phi = 1j * np.random.normal(0, PHRASE_AUGMENT_BIG, holoGram.shape)  # add 15% phrase variations, holoGram(N,M)
    phraseShifts = np.exp(phi)
    tempHolo = holoGram * phraseShifts  # add unit-magnitude complex noise into the image, phrase only changes

    np.random.seed() #E11&12. reset the seed everytime to make o/p more random.
    phi = 1j * np.random.normal(0, PHRASE_AUGMENT_SMALL, holoGram.shape)  # add 10%- phrase variations, holoGram(N,M)
    phraseShifts = np.exp(phi)
    holoGram = holoGram * phraseShifts  # add unit-magnitude complex noise into the image, phrase only changes

    return tempHolo, holoGram # gen two augmented complex no. holograms




def cvImwrite2Dir(image, filename, directory): #change Dir, save image and restore current Dir
    """
    Problem chinese path need this ('G:/我的云端硬盘/1-FrDropbox...) for cv.imwrite save to target directory

    Parameters
    ----------
    image : image to save
    filename : string
    directory : string

    Returns:  None.

    """
    oldDirectory=os.getcwd() #save current directory info
    os.chdir(directory) #change to target directory
    cv.imwrite(filename, image)
    os.chdir(oldDirectory) #restore current directory before return to calling routine.

def scaleUp(sourceData, sourceLabel, newSize):
    """
    Scale up dataset size from source to a newSize by sample&repeat itself

    Parameters
    ----------
    sourceData : np array
    sourceLabel : np array
    newSize : int

    Returns
    -------
    sourceData : np array in bigger new size
    sourceLabel : np array in bigger new size

    """
    assert sourceData.shape[0]==sourceLabel.shape[0], "Datas and it's labels sizes NOT match."
    index= np.arange(sourceData.shape[0]) # creat indexs to the size of profit
    choosen= np.random.choice(index, newSize,  replace=True) #random choose wrt replacment from index to the size of vdpGramsMajor
    upRatio=int(choosen.shape[0]/sourceData.shape[0])
    sourceData= sourceData[choosen]; sourceLabel=sourceLabel[choosen]
    print("ScaledUp "+str(upRatio)+"x to balance source by the newSize.")   

    return sourceData, sourceLabel

def creatFileCode(filename, xShift, yShift, zShift=0, rotate=0, noise=0): #default No noise
    dict ={"zero64x64":"0", "one64x64":"1", "two64x64":"2", "three64x64":"3", "four64x64":"4", "five64x64":"5",
           "six64x64":"6", "seven64x64":"7", "eight64x64":"8", "nine64x64":"9" }
    if noise !=0: noise=1 # True, output 1, False output 0.
    sign=int( bool(xShift*yShift>0) ) # Same size True, output 1.     
    if (rotate<0): rotate +=360 # convert -ve degree into +ve [0,360] degrees
    string=dict.get(filename)+"S"+str(xShift)+str(yShift)+str(sign)+str(zShift)+str(noise)+"R"+str(rotate)
    return string

def chop2objSizeEle(gram, height, width): # for 1-Gram only
    """
    Chop a hologram/vdggram at center and output a gram

    Parameters
    ----------
    grams : input a complex pixel image 
    height, width : size of the out image

    Returns
    -------
    resized : size adjusted holograms/vdpgrams

    """
    # if CHOP_ENABLE: #disable chop and use the full hologram size  
    h, w= gram.shape # y,x. Gray/Mono 1-ch only
    startW = w//2 - width//2; startH = h//2 - height//2
        
    resized=gram[startH:startH +height, startW:startW +width] #A5

    return resized

def chop2objSize(grams, height, width):     #i/p vdpgrams/Holograms or else complex-field
    """
    Chop array of  holograms/vdggrams at center and output a grams

    Parameters
    ----------
    grams : input a complex pixel image 
    height, width : size of the out image

    Returns
    -------
    resized : size adjusted holograms/vdpgrams

    """
    # if CHOP_ENABLE: #disable chop and use the full hologram size  
    h, w= grams[0].shape # y,x. Gray/Mono 1-ch only
    startW = w//2 - width//2; startH = h//2 - height//2
    
    resized=grams[:, startH:startH +height, startW:startW +width] #A5
    # for gram in grams: # need creat buffer memory & more demanding on RAM, 5aug2021
    #     chop2objSizeEle(gram, h, w)
    # else: resized=grams # disabled
    
    return resized

def normHoloGrams(holoGrams, mode=NORM_MINMAX):
    tmps=[]
    if mode==NORM_MINMAX:        
        for holo in holoGrams: # INDIVIDUALLY normalize holograms
            minValue= holo.min(); maxValue=holo.max()
            tmps.append((holo - minValue) / (maxValue - minValue) )
            
    elif mode==NORM_MAX:
        for holo in holoGrams: # INDIVIDUALLY normalize holograms
            maxValue=holo.max()
            tmps.append(holo/holo.max() )

    return tmps

def invertHoloGrams(holoGrams):
    tmps=[]
    for holo in holoGrams: # INDIVIDUALLY invert holograms
        tmps.append(np.subtract(np.max(holo), holo) )

    return tmps
    
def checkRam(checkPoint):
    print(f"\n !!CheckPoint:{checkPoint}, used:", str(psutil.virtual_memory()[2])+'%') #total:[0], available:[1], %:[2] and used:[3]
    print(f"RAM used={'%.2f'%(psutil.virtual_memory()[3]/(1024*1024*1024) )}G", end=', ') 
    print(f"RAM available={round(psutil.virtual_memory()[1]/(1024*1024*1024),2)}G", '!!\n' )
    
def plotConfuMatrix(labels, predictions, string): #CONFUSION Matrix  
  if labels.ndim >1: labels=np.argmax(labels, axis=-1) # convert back matrix to vector for confusion_matrix()
  if predictions.ndim >1: predictions=np.argmax(predictions, axis=-1) #1-D array already, in fact not necessary. Design for change only!!
  
  cm=confusion_matrix(labels, predictions, labels=np.unique(labels), normalize=None) #KNOWN two classed ONLY
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title(string+" [Confusion matrix]")
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show(); plt.savefig('confusMatrix.png')
  
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
"""
Reference: 
A) Matlab Vs Python Commands:
1. http://mathesaurus.sourceforge.net/matlab-numpy.html
2. https://resources.sei.cmu.edu/asset_files/Presentation/2011_017_001_50519.pdf
3. ImageType Conv: https://www.mathworks.com/help/images/image-type-conversions.html
4. mat2gray(): https://www.mathworks.com/help/images/ref/mat2gray.html?searchHighlight=mat2gray&s_tid=doc_srchtitleNone
5. im2double(): https://www.mathworks.com/help/matlab/ref/im2double.html
   
B) Matlab RGB Vs Python OpenCV BGR:
1. https://www.mathworks.com/matlabcentral/answers/91036-how-do-i-split-a-color-image-into-its-3-rgb-channels

C) [lo hi], [0 1] format Matlab Vs OpenCV: 
1. https://www.mathworks.com/help/images/ref/imshow.html#bvmnrxi-1-DisplayRange
2. https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=imshow#imshow

D) Matlab Equivalent Functions:
1. size(): https://stackoverflow.com/questions/51388178/size-function-in-matlab-and-python
2. conv2()->convolve2d(): https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
3. max(): NumpyVsMatlab equivalence: https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html?highlight=numpy%20array%20matlab%20array
4. abs(): https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.absolute.html
5. im2double(): https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python

E) Matlab or Python Functions Reference:
1. Matlab Image DataType Conventions: https://www.mathworks.com/help/images/image-type-conversions.html
2. imshow(): https://www.mathworks.com/help/images/ref/imshow.html#bvmnrxi-1-DisplayRange
3. abs(): https://www.mathworks.com/help/matlab/ref/abs.html

6. np.save:  https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.save.html
   np.savez: (save multi): https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez.html
   np.savez_compressed (compressed)
   
7. np.astype(): https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.astype.html
8. CV_8U,_32F: https://stackoverflow.com/questions/8377091/what-are-the-differences-between-cv-8u-and-cv-32f-and-what-should-i-worry-about
9. cv.normalize(): https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#normalize
10. np.empty_alike(): https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.empty_like.html
11. random.seed(): https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
12. MIT's PyRandLib:  https://github.com/schmouk/PyRandLib
13. bug fix np.load(): https://www.numpy.org/devdocs/reference/generated/numpy.load.html
14. scipy truncate normal distribution: https://www.kite.com/python/docs/scipy.stats.truncnorm
15. fft2 zero padding: : https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
"""