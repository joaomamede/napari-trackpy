import scipy.ndimage as ndi
#import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import pims
#import libtiff
#import time
#import csv
_8bit = float(2**8-1)
_16bit = float(2**16-1)
ratio = _8bit /_16bit


def _get_open_filename(self):
    from napari.utils import history
    _last_folder = history.get_open_history()[0]
    for i in range(len(self.viewer.layers)-1,-1,-1):
        if self.viewer.layers[i]._type_string == 'image':
            _filename = self.viewer.layers[i].name.split(" :: ")[0]
            _filename = _last_folder +"/"+ _filename
            break
    return _filename

def points_in_mask(coords,mask):
    '''
    Input: 
    coords: list of coordinates (numpy)
    mask: masked image with numbered labels (numpy)

    Return:
    values_at_coords: with boolean array reporting which mask does the coordinate belongs to
    '''
    import numpy as np
    # coords_int = np.round(coords).astype(int)  # or np.floor, depends
    coords_int = np.floor(coords).astype(int)  # or np.floor, depends
    try:
        values_at_coords = mask[tuple(coords_int.T)]
    except: values_at_coords = []
    # .astype(np.int)
    # values_at_coords = mask[tuple(coords_int)].astype(np.int)
    print(values_at_coords)
    return values_at_coords


def multiply(real_events, image):
    import numpy as np
    image = real_events*image
    return image

##keep for now for the #make masks section
def make_labels_trackpy(image,mass,size=9,_separation=3,_numba=True,max_mass=0,_round=True):
    import trackpy as tp
    import scipy.ndimage as ndi
    from scipy.ndimage.morphology import binary_dilation

    if image.ndim == 2:
        _size = size
    elif image.ndim == 3:
        _size = (3, size, size)
        # _size = (9, size, size)
     # ~ dotrack(ficheiro, plotgraph=True,_numba=True,massa=1500,tamanho=13,dist=250,memoria=1,stub=3,frame=19,colourIDX=0):
    if image.ndim == 2:
        if _numba:
            f = tp.locate(image,diameter=size,separation=_separation,minmass=mass,engine='numba')
        else:
            f = tp.locate(image,diameter=size,separation=_separation, minmass=mass)
    elif image.ndim == 3:
        if _numba:
            f = tp.locate(image,diameter=_size,separation = (3, 3, 3),
                minmass=mass,engine='numba')
        else:
            f = tp.locate(image,diameter=_size,separation = (3, 3, 3),
                minmass=mass)
            # size = (11, 13, 13)

    if max_mass > 0:
        f = f[f['mass'] <= max_mass]
    #outputsomehow is 3D, we want 2
    if image.ndim == 2:
        coords = np.dstack((round(f.y),round(f.x)))[0].astype(int)
    elif image.ndim == 3:
        coords = np.dstack((round(f.z),round(f.y),round(f.x)))[0].astype(int)



    #this is super slow
    # ~ masks = tp.masks.mask_image(coords,np.ones(image.shape),size/2)

    #This is faster
    if image.ndim == 2:
        r = (size-1)/2 # Radius of circles
        #make 3D compat
        disk_mask = tp.masks.binary_mask(r,image.ndim)
        # Initialize output array and set the maskcenters as 1s
        out = np.zeros(image.shape,dtype=bool)
        #check if there's a problem with subpixel masking
        out[coords[:,0],coords[:,1]] = 1
        # Use binary dilation to get the desired output
        out = binary_dilation(out,disk_mask)
        labels, nb = ndi.label(out)

        if _round:
            return labels, coords
        else:
            if image.ndim == 2:
                coords = np.dstack((f.y,f.x))[0]
                return labels, coords
    elif image.ndim == 3:
            coords = np.dstack((f.z,f.y,f.x))[0]
            return None, coords


def make_labels_trackpy_links(image,j,size=5,_round=False):
    import trackpy as tp
    import scipy.ndimage as ndi
    from scipy.ndimage import binary_dilation


    #outputsomehow is 3D, we want 2
    coords = np.dstack((round(j.y),round(j.x)))[0].astype(int)

    #this is super slow
    # ~ masks = tp.masks.mask_image(coords,np.ones(image.shape),size/2)

    #This is faster
    r = (size-1)/2 # Radius of circles
    #make 3D compat
    disk_mask = tp.masks.binary_mask(r,image.ndim)
    # Initialize output array and set the maskcenters as 1s
    out = np.zeros(image.shape,dtype=bool)
    #check if there's a problem with subpixel masking
    out[coords[:,0],coords[:,1]] = 1
    # Use binary dilation to get the desired output
    out = binary_dilation(out,disk_mask)


    labels, nb = ndi.label(out)
    if _round:
        return labels, coords
    else:
        if image.ndim == 2:
            # coords = j.loc[:,['particle','frame','y','x']]
            coords = j.loc[:,['frame','y','x']]
#             coords = np.dstack((j.particle,j.y,j.x))[0]
            return labels, coords
#     return labels, coords

def simple_labels(image):
    labels, nb = ndi.label(image)
    return labels

def mean_calc(values,value_max):
    import numpy as np
    #Only the ones above threshold will be averaged don't worry about
    #the b>value_max because the threshold is the same.
    a = np.asarray([value['MeanIntensity'] for value in values])
    b = np.asarray([value['MaxIntensity'] for value in values])
    return np.mean(a[b > value_max])

def mean_calc_total(values):
    import numpy as np
    return np.mean([value['MeanIntensity'] for value in values])

def max_calc_total(values):
    import numpy as np
    return np.mean([value['MaxIntensity'] for value in values])

def listify(data,datatype):
    if datatype == 'max': datatype = 'MaxIntensity'
    elif datatype == 'mean': datatype = 'MeanIntensity'
    return np.asarray([value[datatype] for value in data])

def rebin(arr, new_shape):
    from PIL import Image
    return np.array(Image.fromarray(arr).resize(new_shape,resample=Image.NEAREST))

#~ @v.parallel(block=True)
def othercolor(colour,labels):
    import numpy as np
    import scipy.ndimage as ndi
    #This gives an array with max mean and stdev values per each label

    data = {}
    data['max'] = ndi.maximum(colour,labels,np.arange(1,labels.max()+1))
    data['mean'] = ndi.mean(colour,labels,np.arange(1,labels.max()+1))
    data['stdev'] = ndi.standard_deviation(colour,labels,np.arange(1,labels.max()+1))
    data['median'] = ndi.median(colour,labels,np.arange(1,labels.max()+1))
    try:
        temp = np.array(ndi.measurements.center_of_mass(
            colour,labels,np.arange(1,labels.max()+1)))
        # ~ print(temp[:,0])
        data['COMY'] = temp[:,0]
        data['COMX'] = temp[:,1]
    except: 'Something went wrong with centroids'
    return data


def othercolor2(colour,labels,treshold):
    import numpy as np
    import scipy.ndimage as ndi
    '''Output: A dict() with max mean and stdev values per each label'''

    data = {}
    data['max'] = ndi.maximum(colour,labels,np.arange(1,labels.max()+1))
    data['mean'] = ndi.mean(colour,labels,np.arange(1,labels.max()+1))
    data['stdev'] = ndi.standard_deviation(colour,labels,np.arange(1,labels.max()+1))
    data['median'] = ndi.median(colour,labels,np.arange(1,labels.max()+1))
    data['mean_pos'] = data['mean'][data['max'] > treshold]
    data['median_pos'] = data['median'][data['max'] > treshold]
    data['stdev_pos'] = data['stdev'][data['max'] > treshold]
    data['max_pos'] = data['max'][data['max'] > treshold]

    return data

def contrast_img(img,min_,max_ ):
    img[img>max_]=max_
    img[img<min_]=min_
    img -= min_
    img = img * (_16bit/float(max_-min_))
    return img


def convert16to8bits_gpu(x,display_min=0,display_max=2**16-1):
    import cupy as cp
    def display(image, display_min, display_max): # copied from Bi Rico
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
        # image = cp.array(image, copy=FalseTrue)
        image.clip(display_min, display_max, out=image)
        image -= display_min
        cp.floor_divide(image, (display_max - display_min + 1) / 256,
                        out=image, casting='unsafe')
        return image.astype(cp.uint8)

    lut = cp.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return cp.asnumpy(cp.take(lut, x))

def convert16to8bits(x,display_min=0,display_max=2**16-1):
    def display(image, display_min, display_max): # copied from Bi Rico
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
        # image = cp.array(image, copy=FalseTrue)
        image.clip(display_min, display_max, out=image)
        image -= display_min
        np.floor_divide(image, (display_max - display_min + 1) / 256,
                        out=image, casting='unsafe')
        return image.astype(np.uint8)

    lut = np.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, x)
