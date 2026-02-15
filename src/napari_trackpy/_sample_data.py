"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    return [(numpy.random.rand(512, 512), {})]




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
    labels,_=label(mask)
    return labels,positions
