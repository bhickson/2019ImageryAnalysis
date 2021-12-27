import chime

from joblib import Parallel

import os
import rasterio as rio
from rasterio.enums import ColorInterp
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

import numpy as np

from datetime import datetime
from skimage.segmentation import felzenszwalb, watershed, quickshift, slic

from skimage import measure
from skimage.measure import label

import geopandas as gpd

import gdal
import logging as logger

from joblib import Parallel, delayed
from skimage.exposure import adjust_sigmoid

import sys

from glob import glob
import geopandas as gpd

from skimage.filters import hessian, gaussian
from skimage.filters.rank import median, mean, mean_bilateral, equalize, entropy
from skimage.morphology import disk, ball, square, erosion, dilation, diamond, opening, closing
from skimage.exposure import adjust_log, rescale_intensity


from time import sleep

def normToUint8(a):
    a = (a + np.abs(np.nanmin(a))) * (255.0/(a.max() - a.min()))
    return a.astype(np.uint8)


def cross(num):
    if num%2 == 0 or num <= 2:
        print("Value must be an odd number greater than 2")
        return None
    shape = np.zeros((num,num))
    mid = int(num/2)
    for x in range(shape.shape[1]):
        shape[mid, x] = 1
    for y in range(shape.shape[0]):
        shape[y, mid] = 1
    return shape


def bilateralMean(array, size, s=20):
    nbands = min(array.shape)
    a = array.copy()
    for i in range(nbands):
        a[i] = opening(array[i], square(3))
        a[i] = mean_bilateral(a[i], disk(size), s0=s, s1=s)
    return a


def bilateralMeanDiff(array, sizeSmall, sizeLarge, nbands=4):
    largeMean = array.astype(np.float32)
    smallMean = array.astype(np.float32)
    for i in range(nbands):
        smallMean[i] = mean_bilateral(array[i], disk(sizeLarge), s0=20, s1=20)
        largeMean[i] = mean(array[i], disk(sizeLarge))
    diff = smallMean - largeMean
    return diff

    
def calcArrayMeanDiff(data, tval):
    baseVals = {"RED": data[0], "GREEN":data[1], "BLUE": data[2], "NIR": data[3]}
    val = baseVals[tval]
    del baseVals[tval]
    diffMean = val - np.mean([v for v in baseVals.values()],axis=0)
    return diffMean


def rescaleAndStretch(array, dtype, clipRange=(5,95)):
    array_scale = array.astype(dtype)
    nbands = min(array_scale.shape)
    for i in range(nbands):
        band = array_scale[i]
        fifthPerc = np.percentile(band, clipRange[0])
        nintyFifthPerc = np.percentile(band, clipRange[1])
        band[band<fifthPerc] = fifthPerc
        band[band>nintyFifthPerc] = nintyFifthPerc
        rescaled = rescale_intensity(band, out_range=dtype).astype(dtype)

        array_scale[i] = rescaled
        
    return array_scale


def getOrthoWindow(ortho_file, geometry):
    try:
        with rio.open(ortho_file) as src:
            orthoKwargs = src.profile
            nbands = src.count
            colorinterp = [i for i in src.colorinterp]

            win = from_bounds(*geometry.bounds, transform=src.transform)
            win_transform = src.window_transform(win)
            if src.res[0] > 0.6:
                orthoArray = src.read(window=win)
            else:
                print("Resampling on read")
                upscale_factor = 0.5
                win_transform = Affine(win_transform.a / upscale_factor, win_transform.b, win_transform.c,
                                       win_transform.d, win_transform.e / upscale_factor, win_transform.f)
                orthoArray = src.read(window=win, out_shape=(int(win.height * upscale_factor),
                                                             int(win.width * upscale_factor)),
                                      resampling=Resampling.nearest)

            width = orthoArray[0].shape[1]
            height = orthoArray[0].shape[0]

        orthoKwargs.update(
            driver = "GTiff",
            width = width,
            height = height,
            transform = win_transform,
            nodata = None
        )
    except:
        orthoArray = None
        orthoKwargs = None

    return orthoArray, orthoKwargs


def getRegionProperties(segments_file, outdir, overwrite=False):
    propertiesFile = os.path.join(outdir, os.path.basename(segments_file).replace(".tif","_regionProps.tif"))

    if os.path.exists(propertiesFile) and not overwrite:
        print(f"File exists skipping {os.path.basename(propertiesFile)}")
        return propertiesFile
    
    with rio.open(segments_file) as src:
        segments_array = src.read(1)
        kwargs = src.profile

    labels = label(segments_array, connectivity=1).astype(np.float32) + 1
    # Area: Number of pixels of the region.
    # Extent: Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
    # Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    # Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
    regionProperties = ["area", "extent", "eccentricity", "perimeter"]

    regions = measure.regionprops(labels.astype(np.int32), intensity_image=segments_array)

    properties_array_stack = []
    for prop in regionProperties:
        segs_property = calcSegmentProp(in_band=segments_array, regs=regions, labeled_array=labels, prop=prop)
        properties_array_stack.append(segs_property.astype(np.float32))    
    properties_array_stack = np.stack(properties_array_stack)#, axis=0)

    kwargs.update(count=len(regionProperties), dtype=np.float32)
    with rio.open(propertiesFile, "w", **kwargs) as dst:
        for n, tag in enumerate(regionProperties):
            dst.update_tags(n+1, NAME=tag)
        dst.write(properties_array_stack.astype(np.float32))

    return propertiesFile

        
def getPropertyOfSegments(array, segments, label_im, properties=["mean"]):

    output_array_stack = []

    # Iterate through R,G,B,NIR bands and use the segmented image to calculate mean in zones/labels
    for b in range(0, min(array.shape)):
        band_array = array[b]
        regions = measure.regionprops(label_im.astype(np.int32), intensity_image=band_array)
        for prop in properties:
            seg_mean_band = calcSegmentProp(in_band=band_array, regs=regions, labeled_array=label_im, prop=prop)
            output_array_stack.append(seg_mean_band.astype(np.uint16))
    out_array_stack_np = np.stack(output_array_stack)#, axis=0)

    return out_array_stack_np
        

def segmentArray(image_array, seg_type="felzenswalb"):
    #print("Beginning image segmentation on array")
    seg_start = datetime.now()
    if seg_type == "felzenswalb":
        segments = felzenszwalb(image_array, scale=50, sigma=0.5, min_size=12, multichannel=True)
    elif seg_type == "quickshift":
        #quickshift(image, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, random_seed=42)
        segments = quickshift(image_array[:,:,0], ratio=1, kernel_size=3, max_dist=7)
    else:
        print("Unknown segmentation algorithm. Exiting...")
        raise ValueError

    
    seg_end = datetime.now()
    #print("Felzenszwalb number of segments: {}. \n\t{} elapsed.".format(len(np.unique(segments_fz)), seg_end - seg_start))

    return segments


def segmentVIArray(image_array, seg_type="felzenswalb"):
    size = min(image_array.shape)
    #print("Beginning image segmentation on array")
    seg_start = datetime.now()
    if seg_type == "felzenswalb":
        segments = felzenszwalb(image_array, scale=10, sigma=0.1, min_size=4, multichannel=False)
    elif seg_type == "watershed":
        segments = watershed(image_array, connectivity=2, compactness=5)
    elif seg_type == "slic":
        segments = slic(image_array, compactness=1, sigma=1)
    elif seg_type == "quickshift":
        image_array = np.moveaxis(np.array([image_array,image_array,image_array]), 0, -1)
        segments = quickshift(image_array, ratio=0.5, kernel_size=3, max_dist=20)
    else:
        print("Unknown segmentation algorithm. Exiting...")
        raise ValueError

    
    seg_end = datetime.now()

    return segments


def calcSegmentProp(labeled_array, regs, in_band, prop="mean"):
    values_start = datetime.now()
    # label_im = label(segments, connectivity=1) + 1
    # regions = measure.regionprops(labeled_array.astype(np.int64), intensity_image=in_band)

    #print("\tBeginning mean calculation on segments...")
    #mean_array = np.copy(labeled_array)

    values_map = {}
    if prop=="mean":
        for i,r in enumerate(regs):
            values_map[r.label] = r.mean_intensity
    elif prop=="max":
        for i,r in enumerate(regs):
            values_map[r.label] = r.max_intensity
    elif prop=="min":
        for i,r in enumerate(regs):
            values_map[r.label] = r.min_intensity
    elif prop=="area":
        for i,r in enumerate(regs):
            values_map[r.label] = r.area
    elif prop=="extent":
        for i,r in enumerate(regs):
            values_map[r.label] = r.extent
    elif prop=="eccentricity":
        for i,r in enumerate(regs):
            values_map[r.label] = r.eccentricity
    elif prop=="perimeter":
        for i,r in enumerate(regs):
            values_map[r.label] = r.perimeter
    
    values_array = vec_translate(labeled_array, values_map)

    #print("\t...Mean calculation complete.\n\t{} elapsed.".format(datetime.now() - mean_start))

    return values_array


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


def segmentWindowV2(index, ortho, hag, geom, opath_segmented, overwrite=False):
        
    ofile_segments = os.path.basename(opath_segmented)
        
    testfile = opath_segmented[:]
    
    if os.path.exists(opath_segmented) and not overwrite:
        return opath_segmented
    
    try:
        wstart = datetime.now()
        
        ortho_array, kwargs = getOrthoWindow(ortho, geom)
        
        with rio.open(hag) as src:
            win = from_bounds(*geom.bounds, transform=src.transform)
            hag_array = src.read(window=win, out_shape=ortho_array[0].shape)
        
        #Cleanup HAG array. 2019 elevation data showed +- error of about 1.5 feet
        hag_array[hag_array<1.5] = 0

        """
        # VEG INDICIES
        bandRed = ortho_array[0]/255.0
        bandNIR = ortho_array[3]/255.0
        msavi_float = ((2 * bandNIR + 1) - np.sqrt(np.square(2 * bandNIR + 1) - (8 * (bandNIR - bandRed)))) / 2
        ndvi_float = (bandNIR - bandRed) / (bandNIR + bandRed)
        # lowest should be -1 so move to zero. highest (1) goes to 2
        msavi_norm = ((msavi_float+1)/2) * 255 
        ndvi_norm = ((ndvi_float+1)/2) * 255
        msavi_norm = msavi_norm.astype(np.uint8)
        ndvi_norm =  ndvi_norm.astype(np.uint8)
        """     

        ortho_array = bilateralMean(ortho_array, 20, s=20)
        


        # SEGMENTATION - boost contrast first, then run median (not mean) filter to edge smooth
        # contrast enhancement
        cut = 0.4
        gain = 9
        ortho_array_contAdj = adjust_sigmoid(ortho_array, cutoff=cut, gain=gain, inv=False)
        
        ortho_array_stack = []
        
        ortho_array_segs = segmentArray(ortho_array_contAdj)
        ortho_array_labels = label(ortho_array_segs, connectivity=1).astype(np.float32) + 1
        ortho_array_stack = getPropertyOfSegments(ortho_array, ortho_array_segs, ortho_array_labels, properties=["mean"])
        
        ortho_array_stack = np.stack(ortho_array_stack)#, axis=0)
        
        print("ortho_array_stack.shape", ortho_array_stack.shape)
        
        segmentsDir = "../EPCExtent_30cm/Segments"
        
        ofile_segments = os.path.basename(opath_segmented).replace(".tif","_Segments.tif")
        opath_segments = os.path.join(segmentsDir, ofile_segments)
        segsKwargs = kwargs.copy()
        #print("ortho_array_segs.max", np.max(ortho_array_segs), np.min(ortho_array_segs))
        
        segsKwargs.update(dtype=np.uint32)
        with rio.open(opath_segments, "w", **segsKwargs) as dst:
            dst.write(ortho_array_segs.astype(np.uint32), 1)
        
        hag_array_stack = getPropertyOfSegments(hag_array.astype(np.uint8),
                                                segments=ortho_array_segs[-1],
                                                label_im=ortho_array_labels,
                                                properties=["mean"])
        bandRed = ortho_array_stack[0]/255.0 # conversion to float
        bandNIR = ortho_array_stack[3]/255.0
        msavi_float = ((2 * bandNIR + 1) - np.sqrt(np.square(2 * bandNIR + 1) - (8 * (bandNIR - bandRed)))) / 2
        ndvi_float = (bandNIR - bandRed) / (bandNIR + bandRed)
        #Rescale veg indicies to uin8. The +1 moves all values to the positive integer space
        msavi = np.array([((msavi_float+1)/2) * 255]) 
        ndvi = np.array([((ndvi_float+1)/2) * 255])

        bands = ["RED", "GREEN", "BLUE", "NIR"]
        ortho_array_stack = np.append(ortho_array_stack, msavi.astype(np.uint8), axis=0)
        bands.append("MSAVI")
        print("ortho_array_stack.shape", ortho_array_stack.shape)

        colorinterp.append(ColorInterp.undefined)
        ortho_array_stack = np.append(ortho_array_stack, ndvi.astype(np.uint8), axis=0)
        bands.append("NDVI")
        print("ortho_array_stack.shape", ortho_array_stack.shape)

        colorinterp.append(ColorInterp.undefined)
        ortho_array_stack = np.append(ortho_array_stack, hag_array_stack.astype(np.uint8), axis=0)
        bands.append("HAG")

        colorinterp.append(ColorInterp.undefined)

        kwargs.update(count=min(ortho_array_stack.shape))
        print("ortho_array_stack.shape", ortho_array_stack.shape)
        with rio.open(opath_segmented, 'w', **kwargs) as oras:
            oras.write(ortho_array_stack.astype(np.uint8))
            oras.colorinterp = colorinterp
            for i, band in enumerate(bands):
                oras.set_band_description(i+1, band)

        wend = datetime.now()
        print(f"Finished for window #{index} - \n\t{opath_segmented} in {wend-wstart}")
    except Exception as e:
        print(f"FAILED for #{index} - {opath_segmented}\n{e}")
        opath_segmented = opath_segmented.replace(".tif", "_FAILED.tif")
        
    return opath_segmented


def calcNDPI(array):
    """ Normalized Difference Pool Index Calculation 
    Uses difference of BLUE band to mean of NIR and RED over sum of BLUE band and mean of NIR and RED bands
    """
    red_nir_mean = np.nanmean([array[0], array[3]],axis=0)
    ndpi = (array[2] - red_nir_mean)/(array[2] + red_nir_mean)
    
    return ndpi


def getHAGWindow(hagFile, geometry, windowShape):
    try:
        with rio.open(hagFile) as src:
            win = from_bounds(*geometry.bounds, transform=src.transform)
            hagWin_array = src.read(1, window=win, out_shape=windowShape)

        #Cleanup HAG array. 2019 elevation data showed +- error of about 1.5 feet
        hagWin_array[hagWin_array<1.5] = 0
    except:
        hagWin_array = None
        
    return hagWin_array


def findFile(path, row, directory=None, files_dict=None):
    """ finds file in directory containing the given path and row in the file name.
    Either the directory to search can be passed, or a dictionary for file name and
    file path. Turns out it's much faster to earch a dictionary keys than a list of
    file locations. This is a little clunky and assumes inconsistent file naming. 
    Easier would be just to glob(directory + "/f{path}_{row}*.tif")"""
    if files_dict == None and directory == None:
        raise ValueError("One of 'files' or 'directory' must be specified for findFind function")
    elif files_dict == None and type(directory) == str:
        files = glob(directory + "/*.tif")
        files_dict = {os.path.basename(file):file for file in files}
    elif type(files_dict) != dict and directory == None:
        raise ValueError("If no 'directory' value passed, files must be list of files")
    
    finds = [v for k,v in files_dict.items() if path in k and row in k]
  
    if len(finds) == 1:
        return finds[0]
    elif len(finds) > 1:
        raise ValueError(f"To many file matches for {path}_{row}")
    else:
        #raise ValueError(f"No file matches for {path}_{row}")
        return None
    
    
def getRoadsDistance(path, row, geometry, windowShape, roadFileDir="../EPCExtent_30cm/RoadDistances"):
    roads_file = findFile(path, row, roadFileDir)
    if roads_file:
        with rio.open(roads_file) as src:
            win = from_bounds(*geometry.bounds, transform=src.transform)
            roads_distance = src.read(1, window=win, out_shape=windowShape)
    else:
        print(f"Couldn't find necessary roads file for {path}_{row}. Creating array with distance to roads of 1000 ft.")
        roads_distance = np.full(windowShape, 1000)
        
    return roads_distance


def getNessStack(a):
    redness = calcArrayMeanDiff(a, "RED")
    greenness = calcArrayMeanDiff(a, "GREEN")
    blueness = calcArrayMeanDiff(a, "BLUE")
    nirness = calcArrayMeanDiff(a, "NIR")
    mean_vals = np.mean(a[:4,:,:], axis=0)
    ness_stack = np.vstack([[redness], [greenness], [blueness], [nirness], [mean_vals]])
    
    return ness_stack


def rescaleToRange(a, vmin, vmax, dtype, omin=None, omax=None):
    if not isinstance(a, np.ndarray):
        a  = np.array(a)
        
    if not omin and not omax:
        omin = np.iinfo(dtype).min
        omax = np.iinfo(dtype).max
        
    indiff = abs(vmax-vmin)
    outdiff = abs(omax-omin)
    
    a += abs(vmin) # bring whole array to a positive integer space
    inFraction = np.round(a/indiff, 4)
    newout = (inFraction*outdiff)-abs(omin)
    
    return newout.astype(dtype)


def getGaussian(array, sigma, channels=0):
    if channels!=0:
        array = np.moveaxis(array,0,-1)
        out_gaussian = gaussian(array, sigma=sigma, multichannel=True)
        out_gaussian = np.moveaxis(out_gaussian, -1,-0)
    else:
        out_gaussian = gaussian(array, sigma=sigma, multichannel=False)
    
    out_gaussian = rescaleToRange(out_gaussian, 0, 255, np.uint16)
    
    return out_gaussian


def calculateMSAVI(red_band, nir_band, dtype):
    if red_band.dtype != dtype:
        raise ValueError(f"Error in MSAVI calculation. Expected input bands to be of type {dtype}")
    
    zeros = (red_band == 0) & (nir_band == 0)
    # MSAVI2 expects values to be reflectance numbers between 0 and 1. Divide by max value of dtype (max value of uint8 is 255, uint16 is 65535).
    red_band = red_band/np.iinfo(dtype).max
    nir_band = nir_band/np.iinfo(dtype).max
    msaviFloat = ((2 * nir_band + 1) - np.sqrt(np.square(2 * nir_band + 1) - (8 * (nir_band - red_band)))) / 2
    
    # although this is a vegetation index of band ratios, we can assume that if both input bands have zero
    #  values, the result should be absolute minimum, rather than zero, which is the middle value the
    #  range of possible values (-1 to 1)
    msaviFloat[zeros] = -1
    
    return msaviFloat


def calculateNDVI(red_band, nir_band, dtype):
    if red_band.dtype != dtype:
        raise ValueError(f"Error in NDVI calculation. Expected input bands to be of type {dtype}")
    
    zeros = (red_band == 0) & (nir_band == 0)
    # although NDVI ratio shouldn't matter. At the extremes of values (dtype minimum and dtype maximum)
    #  the calculation breaks down with overflows in the numerator and denominator. Bring into float from 0 to 1
    red_band = red_band/np.iinfo(dtype).max
    nir_band = nir_band/np.iinfo(dtype).max
    
    ndviFloat = (nir_band - red_band) / (nir_band + red_band)
    
    # although this is a vegetation index of band ratios, we can assume that if both input bands have zero
    #  values, the result should be absolute minimum, rather than zero, which is the middle value the
    #  range of possible values (-1 to 1)
    ndviFloat[zeros] = -1
    # in cases where
    
    return ndviFloat


def segmentWindowV3(dfrow, outdir, ortho, hag, writeOutStack=True, returnArray=True, overwrite=False):
    
    version_suffix = "TrainingStackV3"
    ofile = f"{dfrow.path}_{dfrow.row}_{version_suffix}.tif"
    ofile_path = os.path.join(outdir, ofile)
    
    if os.path.exists(ofile_path) and not returnArray and not overwrite:
        # If the file exists and we don't want the array, return path
        return ofile_path
    elif os.path.exists(ofile_path) and returnArray and not overwrite:
        # File exists and we want the array, return the data
        with rio.open(ofile_path) as src:
            return src.read()  
    
    local_ortho, kwargs = getOrthoWindow(ortho, dfrow.geometry)
    hag_array = getHAGWindow(hag, dfrow.geometry, local_ortho[0].shape)
    
    if not isinstance(local_ortho, np.ndarray):
        print(f"Requested window is out of extent of {ortho}. Returning None")
        return None
    if not isinstance(hag_array, np.ndarray):
        print(f"Requested window is out of extent of {hag}. Returning None")
        return None
    
    bmness_stack = getNessStack(bilateralMean(local_ortho,10,10))
    dtpr_array = getRoadsDistance(dfrow.path, dfrow.row, dfrow.geometry, local_ortho[0].shape) # Distance To Paved Roads

    local_ortho = rescaleToRange(local_ortho, 0, 255, np.uint16)
    bmness_stack = rescaleToRange(bmness_stack, -255, 255, np.uint16)
    scale=25
    sigma=0.5
    min_size=15
    ness_segs = felzenszwalb(np.moveaxis(bmness_stack,0,-1), scale=scale, sigma=sigma, min_size=min_size, multichannel=True)
    ness_labels = label(ness_segs, connectivity=1).astype(np.float32) + 1
    local_stack_ness = getPropertyOfSegments(bmness_stack, ness_segs, ness_labels, properties=["mean"])
    local_stack_ness = np.stack(local_stack_ness)
    local_stack_ortho = getPropertyOfSegments(local_ortho, ness_segs, ness_labels, properties=["mean"])
    local_stack_ortho = np.stack(local_stack_ortho)
    
    msavi_float = calculateMSAVI(red_band=local_stack_ortho[0], nir_band=local_stack_ortho[3], dtype=np.uint16)
    ndvi_float = calculateNDVI(red_band=local_stack_ortho[0], nir_band=local_stack_ortho[3], dtype=np.uint16)
    gaussianSig2 = getGaussian(local_ortho, 2, channels=len(local_ortho)) # Gaussian Blur with Sigma of 2
    gaussianSig5 = getGaussian(local_ortho, 5, channels=len(local_ortho)) # Gaussian Blur with Sigma of 5
    ndpi = calcNDPI(local_stack_ortho)

    
    bands_dict = {}
    bands_dict["RED"] = local_stack_ortho[0]
    bands_dict["GREEN"] = local_stack_ortho[1]
    bands_dict["BLUE"] = local_stack_ortho[2]
    bands_dict["NIR"] = local_stack_ortho[3]
    bands_dict["MSAVI"] = rescaleToRange(msavi_float, -1, 1, np.uint16)
    bands_dict["NDVI"] = rescaleToRange(ndvi_float, -1, 1, np.uint16)
    bands_dict["NDPI"] =  rescaleToRange(ndpi, -1, 1, np.uint16)
    bands_dict["REDness"] = local_stack_ness[0].astype(np.uint16)
    bands_dict["GREENness"] = local_stack_ness[1].astype(np.uint16)
    bands_dict["BLUEness"] = local_stack_ness[2].astype(np.uint16)
    bands_dict["NIRness"] = local_stack_ness[3].astype(np.uint16)
    bands_dict["HAG"] = getPropertyOfSegments(np.array([hag_array]), ness_segs, ness_labels, properties=["mean"])[0].astype(np.uint16)
    bands_dict["DPR"] = dtpr_array.astype(np.uint16)
    bands_dict["GaussianSigma2_RED"] = gaussianSig2[0].astype(np.uint16)
    bands_dict["GaussianSigma2_GREEN"] = gaussianSig2[1].astype(np.uint16)
    bands_dict["GaussianSigma2_BLUE"] = gaussianSig2[2].astype(np.uint16)
    bands_dict["GaussianSigma2_NIR"] = gaussianSig2[3].astype(np.uint16)
    bands_dict["GaussianSigma5_RED"] = gaussianSig5[0].astype(np.uint16)
    bands_dict["GaussianSigma5_GREEN"] = gaussianSig5[1].astype(np.uint16)
    bands_dict["GaussianSigma5_BLUE"] = gaussianSig5[2].astype(np.uint16)
    bands_dict["GaussianSigma5_NIR"] = gaussianSig5[3].astype(np.uint16)
    
    
    regionProperties = ["area", "extent", "eccentricity", "perimeter"]
    regions = measure.regionprops(ness_labels.astype(np.int32), intensity_image=ness_segs)
    properties_array_stack = []
    for prop in regionProperties:
        segs_property = calcSegmentProp(in_band=ness_segs, regs=regions, labeled_array=ness_labels, prop=prop)
        if prop=="extent" or prop=="eccentricity":
            segs_property = rescaleToRange(segs_property, 0, 1, np.uint16)
        else:
            segs_property = segs_property.astype(np.uint16)
        bands_dict[f"Segment_{prop}"] = segs_property

    full_stack = np.stack([v for k,v in bands_dict.items()])
    bands = [k for k,v in bands_dict.items()]
    colorinterps = [ColorInterp.red,ColorInterp.green,ColorInterp.blue]
    
    for i in range(len(full_stack) - len(colorinterps)):
        colorinterps.append(ColorInterp.undefined)

    if writeOutStack:
        kwargs.update(count=len(full_stack), dtype=full_stack.dtype, compress="LZW")
        with rio.open(ofile_path, "w", **kwargs) as oras:
            oras.write(full_stack)
            oras.colorinterp = colorinterps
            for i, band in enumerate(bands):
                oras.set_band_description(i+1, band)        
        print(f"Wrote training stack out to {ofile_path}")
        
    if returnArray:
        return full_stack
    else: 
        return ofile_path


def buildVRT(directory, outfile):
    outfile = os.path.join(os.path.abspath(directory), outfile)
    vrt = gdal.BuildVRT(outfile, glob(os.path.abspath(directory) + "/*.tif"))
    del vrt
    print(f"Created {outfile}")
    
    
def finished():
    chime.theme("zelda")
    for i in range(3):
        chime.info()
        sleep(0.5)
    chime.theme("material")

    
def getSubSample(data, maxSampSize, features_cols, class_column):
    #data_sub = data.groupby(class_column, group_keys=False).apply(lambda x: x.sample(sampSize)).reset_index(drop=True)
    data_sub = data.groupby(class_column, group_keys=False).apply(lambda x: x.sample(maxSampSize if len(x) > maxSampSize else len(x))).reset_index(drop=True)

    #data_sub, features_cols = calcMeanDiffs(data_sub, features_cols)
    data_sub = data_sub[features_cols + [class_column]]
    
    return data_sub


#1 get entropy mask
#ortho_copy = ortho_array.copy()
def localHistogramOverEntropy(ortho_array, nbands=4, entropyDisk=10, histoDisk=150):
    orthoMB = bilateralMean(ortho_array, 20, s=10)

    for i in range(nbands):
        band_entropy = entropy(ortho_array[i], disk(10))
        entropy_mask = np.where(band_entropy>4, 1, 0)
        #2 equalize nonmasked areas
        orthoMB[i]  = equalize(orthoMB[i], selem=disk(150), mask=entropy_mask)

    return orthoMB


def createLocalHistogramOrtho(dfrow, outdir, ortho, overwrite=False):
    version_suffix = "TrainingStackV3"
    ofile = f"{dfrow.path}_{dfrow.row}_{version_suffix}.tif"
    ofile_path = os.path.join(outdir, ofile)
    
    if os.path.exists(ofile_path) and not overwrite:
        # If the file exists and we don't want the array, return path
        return ofile_path
    
    local_ortho, kwargs = getOrthoWindow(ortho, dfrow.geometry)

    if not isinstance(local_ortho, np.ndarray):
        print(f"Requested window is out of extent of {ortho}. Returning None")
        return None
    
    local_ortho = localHistogramOverEntropy(local_ortho)
    
    bands = ["RED", "GREEN", "BLUE", "NIR"]
            
    with rio.open(ofile_path, "w", **kwargs) as dst:
        dst.write(local_ortho)
        for i, band in enumerate(bands):
                dst.set_band_description(i+1, band + "_LHE")
    print(f"Finished with {ofile_path}")
    
    return ofile_path