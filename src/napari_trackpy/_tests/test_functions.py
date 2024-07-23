# coding: utf-8
import pandas as pd
import numpy as np
import cupy as cp

import time
from scipy.ndimage import label
# Original function

def make_labels_links_original(shape, j, radius=5):
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

# Alternative optimized function


import numpy as np
from scipy.ndimage import label
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fill_mask_numba(mask, positions, radius):
    shape = mask.shape
    for i in prange(len(positions)):
        pz, py, px = positions[i]
        for x in range(shape[2]):
            for y in range(shape[1]):
                for z in range(shape[0]):
                    distance = np.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
                    if distance <= radius:
                        mask[z, y, x] = 1
                        


        
        
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
    
    
def fill_mask_cupy(mask, positions, radius):
    shape = mask.shape
    print(positions[0])
    print(shape)
    for i in range(len(positions)):
        pz, py, px = positions[i]
        print(pz)
#        x, y, z = cp.meshgrid(cp.arange(shape[2]), cp.arange(shape[1]), cp.arange(shape[0]))
        z, y, x = cp.meshgrid(cp.arange(shape[0]), cp.arange(shape[1]), cp.arange(shape[2]), 
                              indexing='ij'#'xy'
                              )
        print(z.shape,y.shape,x.shape)
        distance = cp.sqrt( (pz - z) ** 2 + (py - y) ** 2  + (px - x) ** 2)
        print(distance.shape)
        print(mask.shape)
        mask[distance <= radius] = 1
        
        return mask
        
def make_labels_links_cupy(shape, j, radius=5):
    if 'z' in j.columns:
        positions = cp.asarray(np.vstack((j.z, j.y, j.x)).T)
        print("3D", j)
    else:
        positions = cp.asarray(np.vstack((j.y, j.x)).T)
        print("2D", j)
    
    # Prepare data
    mask = cp.zeros(shape, dtype=cp.uint8)
    
    # Fill mask using CuPy
    mask = fill_mask_cupy(mask, positions, radius)
    print(mask.max())
    # Use label function from scipy to identify connected components
    # Note: As of my last update, CuPy arrays must be transferred back to NumPy arrays for scipy's label function
    mask_np = cp.asnumpy(mask)
    labels, _ = label(mask_np)

    return labels, positions

def fill_mask_numpy(mask, positions, radius):
    shape = mask.shape
    for i in range(len(positions)):
        pz, py, px = positions[i]
        #np.meshgrid is not keeping the right shapes, I need to do 1,0,2.... problem is indexing default flag
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                                indexing='ij'#'xy'
                              )
        distance = np.sqrt( (pz - z) ** 2 + (py - y) ** 2  + (px - x) ** 2)
        mask[distance <= radius] = 1
        
      
def make_labels_links_numpy(shape, j, radius=5):
    if 'z' in j.columns:
        positions = np.dsack((j.z, j.y, j.x))[0]
        print("3D", j)
    else:
        positions = np.vstack((j.y, j.x))[0]
        print("2D", j)
    
    # Prepare data
    mask = np.zeros(shape, dtype=cp.uint8)
    
    # Fill mask using CuPy
    fill_mask_numpy(mask, positions, radius)
    
    # Use label function from scipy to identify connected components
    # Note: As of my last update, CuPy arrays must be transferred back to NumPy arrays for scipy's label function
    # mask_np = cp.asnumpy(mask)
    labels, _ = label(mask)

    return labels, positions
# Generate a sample dataset with 20 points with decimals
np.random.seed(42)
sample_data = pd.DataFrame({
    'x': np.random.rand(20)*1024,
    'y': np.random.rand(20)*1024,
    'z': np.random.rand(20)*20,
})


# Test the speed of the original function
start_time = time.time()
result_original = make_labels_links_original((20, 1024, 1024), sample_data)
original_time = time.time() - start_time

# Test the speed of the alternative optimized function
start_time = time.time()
result_cupy= make_labels_links_cupy((20, 1024, 1024), sample_data)
cupy_time = time.time() - start_time

# Test the speed of the corrected NumPy optimized function with numba
start_time = time.time()
result_numba = make_labels_links_numba((20, 1024, 1024), sample_data)
numba_time = time.time() - start_time

# Test the speed of the corrected NumPy optimized function with numba
start_time = time.time()
result_numpy = make_labels_links_numpy((20, 1024, 1024), sample_data)
numpy_time = time.time() - start_time

# Check if the results are the same
assert np.array_equal(result_original[0], result_cupy[0])
assert np.array_equal(result_original[0], result_numba[0])
assert np.array_equal(result_original[0], result_numpy[0])

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(np.max(result_original[0],axis=0))
plt.imshow(np.max(result_alternative[0],axis=0))
plt.imshow(np.max(result_numba[0],axis=0))
plt.imshow(np.max(result_numpy[0],axis=0))
print(f"Original Function Time: {original_time:.5f} seconds")
print(f"cupy Optimized Function Time: {cupy_time:.5f} seconds")
print(f"Numba Optimized Function Time: {numba_time:.5f} seconds")
print(f"Numpy Optimized Function Time: {numba_time:.5f} seconds")




# adding comments to https://stackoverflow.com/a/25553970/5006740
# here we go functional
# from functools import partial
# import multiprocessing


# # we will FREEZE args later for a and b
# def f(a, b, c):
#     print("{} {} {}".format(a, b, c))


# def main():
#     iterable = [1, 2, 3, 4, 5]
#     pool = multiprocessing.Pool()
#     a = "hi"
#     b = "there"
#     # FREEZE goes here
#     func = partial(f, a, b)

#     # f's c arg (iterable in this case) is unfreezed, we map it
#     pool.map(func, iterable)
#     pool.close()
#     pool.join()


import trackpy as tp
import pandas as pd
from multiprocessing import Pool
import os
from functools import partial
multiproc_locate = partial(tp.locate, diameter=5,
                    minmass=10000,engine="numba",
                    # **kwargs
                    )
# def multi_run_wrapper(args):
#    return multi_locate(*args)
num_items = img.shape[0]
num_done = 0
def handle_result(res):
    global num_done
    num_done += 1
    print('finished item {} of {}.'.format(num_done, num_items))

with Pool(os.cpu_count()) as p:
    # temp_f = p.map(multiproc_locate,img)
    temp_f = p.apply_async(multiproc_locate,img,callback=handle_result)
#                   _diam=self.diameter_input.value(),
#                   _minmass=self.mass_slider.value()),zip(range(img.shape[0]),img[:]))
    
print(temp_f)# dummy data and function
all_inputs = list(zip(range(10), range(20,30)))
def name_of_function(a, b):
    return a+b

num_items = len(all_inputs)
num_done = 0
def handle_result(res):
    global num_done
    num_done += 1
    print('finished item {} of {}.'.format(num_done, num_items))

p = Pool(5)
for args in all_inputs:
    p.apply_async(name_of_function, args, callback=handle_result)
p.close()
