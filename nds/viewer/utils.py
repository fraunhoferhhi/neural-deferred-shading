import numpy as np
import pycuda.gl
from OpenGL.GL import GL_RENDERBUFFER

def ray_plane_intersection(p0, n, l0, l):
    return np.dot((p0 - l0), n) / np.dot(l, n)

class CudaBuffer:
    def __init__(self, id, target=GL_RENDERBUFFER, flags=pycuda.gl.graphics_map_flags.NONE):
        self.cuda_buffer = pycuda.gl.RegisteredImage(id, target, flags)
        self.cuda_buffer_map = self.cuda_buffer.map()

    def copy_to_tensor(self, tensor):
        '''
        Copy to a tensor with shape HxWxC
        '''
        h,w,c = tensor.shape
        copy_op = pycuda.driver.Memcpy2D()
        copy_op.set_src_array(self.cuda_buffer_map.array(0, 0))
        copy_op.set_dst_device(tensor.data_ptr())
        copy_op.width_in_bytes = w * c * tensor.element_size()
        copy_op.src_pitch = copy_op.width_in_bytes
        copy_op.dst_pitch = copy_op.width_in_bytes
        copy_op.height = tensor.shape[0]
        copy_op(aligned=False)

    def copy_from_tensor(self, tensor):
        '''
        Copy from a tensor with shape HxWxC
        '''
        h,w,c = tensor.shape
        copy_op = pycuda.driver.Memcpy2D()
        copy_op.set_src_device(tensor.data_ptr())
        copy_op.set_dst_array(self.cuda_buffer_map.array(0, 0))
        copy_op.width_in_bytes = w * c * tensor.element_size()
        copy_op.src_pitch = copy_op.width_in_bytes
        copy_op.dst_pitch = copy_op.width_in_bytes
        copy_op.height = h
        copy_op(aligned=False)

    def destroy(self):
        self.cuda_buffer_map.unmap()
        self.cuda_buffer.unregister()