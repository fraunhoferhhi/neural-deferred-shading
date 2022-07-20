import glfw
from queue import Queue
import moderngl
import numpy as np
from OpenGL.GL import GL_TEXTURE_2D
import pycuda.driver
import pycuda.gl
from pycuda.tools import make_default_context
from pyrr import Matrix44
import threading
import torch

from .camera import PerspectiveCamera
from .controller import OrbitControl
from .primitives import Quad, CoordinateSystem
from .shaders import *
from .utils import CudaBuffer

class MeshViewer:
    def __init__(self, width=600, height=600, name="OpenGL Window", neural_shader=None, device='cuda:0'):
        self.width = width
        self.height = height
        self.name = name
        self.command_queue = Queue()
        self.clear_color = [1, 1, 1]

        self.drag_point_left = None
        self.drag_point_right = None

        self.user_mouse_scroll_callback = None

        self.neural_shader = neural_shader
        self.device = device

        self.render_thread = threading.Thread(target=self.run)
        self.render_thread.start()

    def run(self):
        self.create_window()

        # Create the camera and its controller
        self.viewport = (0, 0, self.width, self.height)
        self.context.viewport = self.viewport
        self.camera = PerspectiveCamera(self.viewport)
        self.camera_controller = OrbitControl(self.camera)
        self.model_matrix = np.eye(4)
        self.inverse_model_matrix = np.eye(4)

        self.coordinate_system = CoordinateSystem(self.context)

        # Create the default program for triangle mesh rendering
        self.mesh_program_name = 'face'
        self.mesh_programs = {
            'face': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_color_face),
            'smooth': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_color_smooth),
            'normal': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_normal),
            'neural': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_position_normal),
        }
        self.mesh_program = self.mesh_programs['face']
        self.vao = None

        # Rendering with the neural shader is a little more involved and we need CUDA-OpenGL interop
        if self.neural_shader:
            # Initialize cuda context
            assert torch.cuda.is_available()
            self.cuda_context = make_default_context(lambda dev: pycuda.gl.make_context(dev))
            self.quad = Quad(self.context)

            # initialize pytorch tensors
            self.positions = torch.empty((self.height, self.width, 4), dtype=torch.float32, device=self.device, requires_grad=False).contiguous()
            self.normals = torch.empty((self.height, self.width, 4), dtype=torch.float32, device=self.device, requires_grad=False).contiguous()
            self.model_output = torch.empty((self.height, self.width, 4), dtype=torch.uint8, device=self.device, requires_grad=False).contiguous()

            # Create shared buffers
            rbo_position = self.context.renderbuffer((self.width, self.height), components=3, dtype='f4')
            rbo_normal = self.context.renderbuffer((self.width, self.height), components=3, dtype='f4')
            texture = self.context.texture((self.width, self.height), components=3, dtype='f1')

            self.cuda_rbo_position = CudaBuffer(rbo_position.glo)
            self.cuda_rbo_normal = CudaBuffer(rbo_normal.glo)
            self.cuda_texture = CudaBuffer(texture.glo, target=GL_TEXTURE_2D)

            # Create the frame buffer object for off-screen rendering
            depth_attachment = self.context.depth_renderbuffer((self.width, self.height))
            self.fbo = self.context.framebuffer([rbo_position, rbo_normal], depth_attachment=depth_attachment)

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            glfw.make_context_current(self.window)

            # Execute all queued commands
            while not self.command_queue.empty():
                try:
                    command = self.command_queue.get_nowait()
                    command()
                except RuntimeError as e:
                    print(e)

            self.context.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            self.context.clear(*self.clear_color)

            # Update shader data
            model_view_matrix = self.model_matrix @ self.camera.view_matrix
            self.mesh_program['model_view_matrix'].write(model_view_matrix.astype('f4'))
            self.mesh_program['projection_matrix'].write(self.camera.projection_matrix.astype('f4'))

            if self.vao:
                if self.mesh_program_name == 'neural':
                    # 1st pass: render the mesh to a frame buffer
                    self.fbo.clear(*self.clear_color, depth=1.0)
                    self.fbo.use()

                    self.vao.render()

                    self.cuda_rbo_position.copy_to_tensor(self.positions)
                    self.cuda_rbo_normal.copy_to_tensor(self.normals)

                    # The positions are in model coordinates so we need the camera position in model coordinates, too
                    camera_position_in_model_space = (np.concatenate([self.camera.eye, np.array([1])]) @ self.inverse_model_matrix)[:3]
                    view_direction = torch.from_numpy(camera_position_in_model_space).to(dtype=torch.float32, device=self.device) - self.positions[...,:3]
                    view_direction = torch.nn.functional.normalize(view_direction, dim=-1)
                    mask = (self.positions[...,2] != 1).unsqueeze(-1)

                    if self.neural_shader:
                        color = self.neural_shader(self.positions[...,:3], self.normals[...,:3], view_direction)
                        self.model_output[...,:3] = ((color.detach() * mask + ~mask).clamp_(0,1) * 255).byte()
                    else:
                        # Fallback
                        self.model_output[..., :3] = 0.5*(self.positions[..., :3] + 1)*255 + 0.5

                    # Copy the model output to a texture
                    self.cuda_texture.copy_from_tensor(self.model_output)
                    torch.cuda.synchronize()

                    # 2nd pass: transfer the mesh colors to the default frame buffer (of the screen) 
                    #           by rendering a quad without depth testing.
                    self.context.screen.use()

                    self.context.disable(moderngl.DEPTH_TEST)
                    self.quad.render(texture)
                    self.context.enable(moderngl.DEPTH_TEST)

                    # 3rd pass: a depth-only pass to populate the default depth buffer with the object depths
                    #           This is necessary so that geometry in the following render passes (e.g. coordinate axes)
                    #           is properly occluded by the object.
                    # FIXME: How can we directly blit the FBO depths to the default depth buffer?
                    self.context.screen.color_mask = False, False, False, False
                    self.vao.render()
                    self.context.screen.color_mask = True, True, True, True
                else:
                    self.vao.render()

            # Render the coordinate system 
            self.coordinate_system.render(self.context, self.camera)

            glfw.swap_buffers(self.window)
    
        glfw.make_context_current(self.window)

        if self.neural_shader is not None:
            self.cuda_rbo_position.destroy()
            self.cuda_rbo_normal.destroy()
            self.cuda_texture.destroy()
            self.cuda_context.pop()
        glfw.destroy_window(self.window)
        #glfw.terminate()

    def create_window(self):
        if not glfw.init():
            return
            
        glfw.window_hint(glfw.SRGB_CAPABLE, 1)
        glfw.window_hint(glfw.FLOATING, 1)
    
        self.window = glfw.create_window(self.width, self.height, self.name, None, None)
    
        if not self.window:
            raise RuntimeError("Unable to create window.")
    
        glfw.make_context_current(self.window)
        self.context = moderngl.create_context()

        glfw.set_cursor_pos_callback(self.window, self.mouse_event_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self.window_resize_callback)
        glfw.set_key_callback(self.window, self.key_callback)

    def mouse_event_callback(self, window, xpos, ypos):
        if self.drag_point_left:
            self.camera_controller.handle_drag(self.drag_point_left[0], xpos, self.drag_point_left[1], ypos, 0)
            self.drag_point_left = (xpos, ypos)
        elif self.drag_point_right:
            self.camera_controller.handle_drag(self.drag_point_right[0], xpos, self.drag_point_right[1], ypos, 1)
            self.drag_point_right = (xpos, ypos)

    def mouse_button_callback(self, window, button, action, mods):
        # Detect drag start/end event
        if action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)

            if button == glfw.MOUSE_BUTTON_LEFT:
                self.drag_point_left = (xpos, ypos)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.drag_point_right = (xpos, ypos)
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.drag_point_left = None
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.drag_point_right = None

    def mouse_scroll_callback(self, window, x_offset: float, y_offset: float):
        self.camera_controller.handle_scroll(x_offset, y_offset)

        if self.user_mouse_scroll_callback:
            self.user_mouse_scroll_callback(x_offset, y_offset)

    def window_resize_callback(self, window, width, height):
        if width > 0 and height > 0:
            self.viewport = (0, 0, width, height)
            self.context.viewport = self.viewport
            self.camera.viewport = self.viewport

    def key_callback(self, window, key, scancode, action, mods):
        if self.user_key_callback:
            self.user_key_callback(key, scancode, action, mods)

    def set_mesh(self, v, f, n=None, c=None):
        self.__enqueue_command(lambda: self.__set_mesh(v, f, n, c))

    def __set_mesh(self, v, f, n, c):
        v_flat = v.ravel().astype('f4')
        if c is None:
            c = 0.85*np.ones((v.shape[0], 3), dtype='f4')
        c = np.asarray(c)
        if len(c.shape) == 1 and c.shape[0] == 3:
            c = np.tile(c[None, :], (len(v), 1))
        c_flat = c.ravel().astype('f4')
        f_flat = f.ravel().astype('i4')

        if n is not None:
            n_flat = n.ravel().astype('f4')
            self.vnbo = self.context.buffer(n_flat)
        else:
            self.vnbo = None

        self.vbo = self.context.buffer(v_flat)
        self.vcbo = self.context.buffer(c_flat)
        self.ibo = self.context.buffer(f_flat)
        self.__update_vao()


    def set_model_matrix(self, model_matrix):
        self.__enqueue_command(lambda: self.__set_model_matrix(model_matrix))

    def __set_model_matrix(self, model_matrix):
        self.model_matrix = Matrix44(model_matrix.T)
        self.inverse_model_matrix = self.model_matrix.inverse

    def set_shading_mode(self, mode):
        self.__enqueue_command(lambda: self.__set_shading_mode(mode))

    def __set_shading_mode(self, mode):
        if mode == 'neural' and self.neural_shader is None:
            raise RuntimeError("Neural shading mode requires a neural shader")

        self.mesh_program_name = mode
        self.mesh_program = self.mesh_programs[self.mesh_program_name]
        if self.vao:
            self.__update_vao()

    def __enqueue_command(self, command, wait=False):
        if not wait:
            self.command_queue.put(command)
        else:
            event = threading.Event()
            def execute_and_set():
                command()
                event.set()
            self.command_queue.put(execute_and_set)
            event.wait()

    def __update_vao(self):
        content = [(self.vbo, '3f', 'position')]

        if self.vnbo and self.mesh_program.get('normal', None):
            content += [(self.vnbo, '3f', 'normal')]

        if self.vcbo and self.mesh_program.get('color', None):  
            content += [(self.vcbo, '3f', 'color')]

        # We control the 'in_vert' and `in_color' variables
        self.vao = self.context.vertex_array(
            self.mesh_program,
            # [
            #     # Map in_vert to the first 2 floats
            #     # Map in_color to the next 3 floats
            #     #(self.vbo, '2f 3f', 'in_vert', 'in_color'),
            #     (self.vbo, '3f', 'position'),
            #     #(self.vnbo, '3f', 'normal'),
            #     (self.vcbo, '3f', 'color'),
            # ],
            content,
            index_buffer=self.ibo,
            index_element_size=4
        )
