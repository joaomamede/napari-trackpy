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

def make_labels_links_alternative(shape, j, radius=5):
    if 'z' in j:
        pos = j[['z', 'y', 'x']].values
    else:
        pos = j[['y', 'x']].values

    ndim = len(shape)
    inarr = np.indices(shape).T
    a = np.sum(inarr, axis=-1, keepdims=True)
    a2 = np.sum(inarr ** 2, axis=-1, keepdims=True)

    b = pos
    for _ in range(len(shape)):
        b = np.expand_dims(b, axis=-2)
    b2 = b ** 2

    sq_sum = a2 - a * 2 * b + b2 * len(shape)

    mask_total = np.any(sq_sum <= radius ** 2, axis=-1).T
    labels, _ = label(mask_total)

    return labels, positions


def make_labels_links_original(shape, j, radius=5):
    import numpy as np
    if 'z' in j:
        pos = np.dstack((j.z, j.y, j.x))[0]
    else:
        pos = np.dstack((j.y, j.x))[0]

    ndim = len(shape)
    in_mask = np.array([np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
                        for p in pos])
    mask_total = np.any(in_mask, axis=0).T
    labels, _ = label(mask_total)

    return labels, pos

def make_labels_links_cupy(shape, j, radius=5):
    import cupy as cp
    if 'z' in j:
        pos = np.dstack((j.z, j.y, j.x))[0]
    else:
        pos = np.dstack((j.y, j.x))[0]

    ndim = len(shape)
    in_mask = np.array([np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
                        for p in pos])
    mask_total = np.any(in_mask, axis=0).T
    labels, _ = label(mask_total)

    return labels, pos
import numpy as np
from scipy.ndimage import label
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fill_mask_numba(mask, positions, radius):
    #only 3D like this
    shape = mask.shape
    print("positions numba shape", positions.shape)
    for i in prange(len(positions)):
        pz, py, px = positions[i]
        for x in range(shape[2]):
            for y in range(shape[1]):
                for z in range(shape[0]):
                    distance = np.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
                    if distance <= radius:
                        mask[z, y, x] = 1

def fill_mask_cupy(mask, positions, radius):
    shape = mask.shape
    for i in range(len(positions)):
        pz, py, px = positions[i]
        x, y, z = cp.meshgrid(cp.arange(shape[2]), cp.arange(shape[1]), cp.arange(shape[0]))
        distance = cp.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
        mask[distance <= radius] = 1
        
def make_labels_links_cupy(shape, j, radius=5):
    if 'z' in j.columns:
        positions = cp.asarray(np.dstack((j.z, j.y, j.x))[0])
        print("3D", j)
    else:
        positions = cp.asarray(np.dstack((j.y, j.x))[0])
        print("2D", j)
    
    # Prepare data
    mask = cp.zeros(shape, dtype=cp.uint8)
    
    # Fill mask using CuPy
    fill_mask_cupy(mask, positions, radius)
    
    # Use label function from scipy to identify connected components
    # Note: As of my last update, CuPy arrays must be transferred back to NumPy arrays for scipy's label function
    mask_np = cp.asnumpy(mask)
    labels, _ = label(mask_np)

    return labels, positions
def make_labels_links_numba(shape, j, radius=5):
    if 'z' in j.columns:
        positions = np.dstack((j.z, j.y, j.x))[0]
        print("3D", j)
    else:
        positions = np.dstack((j.y, j.x))[0]
        print("2D", j)
    
    # Prepare data
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Fill mask using Numba
    fill_mask_numba(mask, positions, radius)
    
    # Use label function from scipy to identify connected components
    labels, _ = label(mask)

    return labels, positions



def make_labels_links_opencl(shape, j, radius=5):
    """
    Creates binary masks around given positions with a specified radius in a 3D space using PyOpenCL.

    :param shape: Tuple of the output volume shape (Z, Y, X).
    :param positions: NumPy array of positions (Z, Y, X).
    :param radius: The radius around each position to fill in the mask.
    :return: A 3D NumPy array representing the binary mask.
    """
    import numpy as np
    import pyopencl as cl
    from scipy.ndimage import label
    if 'z' in j:
        # "Need to loop each t and do one at a time"
        positions = np.dstack((j.z,j.y,j.x))[0]#.astype(int)
        print("3D",j)
    else:
        positions = np.dstack((j.y,j.x))[0]#.astype(int)
        print("2D",j)
    # Prepare data
    mask = np.zeros(shape, dtype=np.uint8)
    positions_flat = positions.flatten().astype(np.float32)
    radius = np.float32(radius)
    
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)

    # PyOpenCL setup
#     ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Create buffers
    mask_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=mask)
    positions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions_flat)
    
    # Kernel code
    if 'z' in j:
        kernel_code = """
        __kernel void fill_mask(__global uchar *mask, __global const float *positions, const float radius, const int num_positions) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int z = get_global_id(2);
            int width = get_global_size(0);
            int height = get_global_size(1);
            int depth = get_global_size(2);
            int idx = x + y * width + z * width * height;
            
            for (int i = 0; i < num_positions; ++i) {
                float pz = positions[i * 3];
                float py = positions[i * 3 + 1];
                float px = positions[i * 3 + 2];
                
                float distance = sqrt(pow(px - x, 2) + pow(py - y, 2) + pow(pz - z, 2));
                if (distance <= radius) {
                    mask[idx] = 1;
                    break;
                }
            }
        }
        """
    else:
        kernel_code = """
        __kernel void fill_mask(__global uchar *mask, __global const float *positions, const float radius, const int num_positions) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int width = get_global_size(0);
            int height = get_global_size(1);
            int idx = x + y * width;
            
            for (int i = 0; i < num_positions; ++i) {
                float py = positions[i * 3];
                float px = positions[i * 3 + 1];
                
                float distance = sqrt(pow(px - x, 2) + pow(py - y, 2));
                if (distance <= radius) {
                    mask[idx] = 1;
                    break;
                }
            }
        }
        """
    
    # Build kernel
    prg = cl.Program(ctx, kernel_code).build()
    
    # Execute kernel
    global_size = shape[::-1]  # Note: PyOpenCL uses column-major order, so we reverse the dimensions
    prg.fill_mask(queue, global_size, None, mask_buf, positions_buf, radius, np.int32(len(positions)))
    
    # Read back the results
    cl.enqueue_copy(queue, mask, mask_buf)
    labels,_=label(mask)
    return labels,positions

def make_labels_trackpy_links(shape,j,radius=5,_algo="GPU"):
    """
    Creates binary masks around given positions with a specified radius in a 3D space using PyOpenCL.

    :param shape: Tuple of the output volume shape (Z, Y, X).
    :param positions: NumPy array of positions (Z, Y, X).
    :param radius: The radius around each position to fill in the mask.
    :return: A 3D NumPy array representing the labeled masks. The positions as array
    """
    import trackpy as tp
    import scipy.ndimage as ndi
    from scipy.ndimage import binary_dilation
    if 'z' in j:
        # "Need to loop each t and do one at a time"
        pos = np.dstack((j.z,j.y,j.x))[0]#.astype(int)
        print("3D",j)
    else:
        pos = np.dstack((j.y,j.x))[0]#.astype(int)
        print("2D",j)
    if _algo == "GPU":
        import cupy as cp
        pos_cp = cp.asarray(pos)

        ##this is what tp.masks.mask_image does maybe put a cupy here to make if faster.
        ndim = len(shape)
        # radius = validate_tuple(radius, ndim)
        pos_cp = cp.atleast_2d(pos_cp)

        # if include_edge:
        in_mask = cp.array([cp.sum(((cp.indices(shape).T - p) / radius)**2, -1) <= 1
                    for p in pos_cp])
        # else:
        #     in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
        #                for p in pos]
        mask_total = cp.any(in_mask, axis=0).T
        
        ##if they overlap the labels won't match the points
        #we can make np.ones * ID of the point and then np.max(axis=-1)
        labels, nb = ndi.label(cp.asnumpy(mask_total))
        # image * mask_cluster.astype(np.uint8)
        
        #this is super slow
        # ~ masks = tp.masks.mask_image(coords,np.ones(image.shape),size/2)
    elif _algo=='CPU':


        ##this is what tp.masks.mask_image does maybe put a cupy here to make if faster.
        ndim = len(shape)
        # radius = validate_tuple(radius, ndim)
        pos = np.atleast_2d(pos)
        # if include_edge:
        in_mask = np.array([np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
                    for p in pos])
        # else:
        #     in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
        #                for p in pos]
        mask_total = np.any(in_mask, axis=0).T
        ##if they overlap the labels won't match the points
        #we can make np.ones * ID of the point and then np.max(axis=-1)
        labels, nb = ndi.label(mask_total)
    elif _algo=='fast':
    #This is faster
        
        # r = (radius-1)/2 # Radius of circles
        # print(radius,r)
    #     #make 3D compat
        disk_mask = tp.masks.binary_mask(radius,len(shape))
        # print(disk_mask)
    #     # Initialize output array and set the maskcenters as 1s
        out = np.zeros(shape,dtype=bool)

        if 'z' in j:
            pos = np.dstack((j.z,j.y,j.x))[0].astype(int)
            pos = np.atleast_2d(pos)
            print(pos)
            out[pos[:,0],pos[:,1],pos[:,2]] = 1            

        else:
            pos = np.dstack((j.y,j.x))[0].astype(int)
            pos = np.atleast_2d(pos)
            print(pos)
            out[pos[:,0],pos[:,1]] = 1
    #     # Use binary dilation to get the desired output
    
        out = binary_dilation(out,disk_mask)

        labels, nb = ndi.label(out)
        print("Number of labels:",nb)
        # if _round:
        #     return labels, coords
        # else:
        #     if image.ndim == 2:
        #         # coords = j.loc[:,['particle','frame','y','x']]
        #         coords = j.loc[:,['frame','y','x']]
        #         # coords = np.dstack((j.particle,j.y,j.x))[0]
        #         return labels, coords
    elif _algo == 'openCL':
            # Prepare data
        mask = np.zeros(shape, dtype=np.uint8)
        positions_flat = positions.flatten().astype(np.float32)
        radius = np.float32(radius)
        
        # PyOpenCL setup
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # Create buffers
        mask_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=mask)
        positions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions_flat)
        
        # Kernel code
        kernel_code = """
        __kernel void fill_mask(__global uchar *mask, __global const float *positions, const float radius, const int num_positions) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int z = get_global_id(2);
            int width = get_global_size(0);
            int height = get_global_size(1);
            int depth = get_global_size(2);
            int idx = x + y * width + z * width * height;
            
            for (int i = 0; i < num_positions; ++i) {
                float pz = positions[i * 3];
                float py = positions[i * 3 + 1];
                float px = positions[i * 3 + 2];
                
                float distance = sqrt(pow(px - x, 2) + pow(py - y, 2) + pow(pz - z, 2));
                if (distance <= radius) {
                    mask[idx] = 1;
                    break;
                }
            }
        }
        """
        
        # Build kernel
        prg = cl.Program(ctx, kernel_code).build()
        
        # Execute kernel
        global_size = shape[::-1]  # Note: PyOpenCL uses column-major order, so we reverse the dimensions
        prg.fill_mask(queue, global_size, None, mask_buf, positions_buf, radius, np.int32(len(positions)))
        
        # Read back the results
        cl.enqueue_copy(queue, mask, mask_buf)
        labels, nb = ndi.label(mask)

    return labels, pos

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
