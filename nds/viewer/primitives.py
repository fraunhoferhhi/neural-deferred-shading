import moderngl
import numpy as np

from .shaders import mesh_vertex_shader, fragment_shader_flat

class Quad:
    def __init__(self, context: moderngl.Context):

        self.prog = context.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_uv;
                out vec2 uv;

                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    uv = in_uv;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D image;
                in vec2 uv;
                out vec3 out_color;

                void main() {
                    out_color = texture(image, uv).rgb;
                }
            '''
        )

        # (2) positions + (2) uvs
        vertices = np.array([
            [-1,  1, 0, 1],
            [-1, -1, 0, 0],
            [ 1,  1, 1, 1],
            [ 1, -1, 1, 0]
        ], dtype='f4')

        self.vbo = context.buffer(vertices)

        self.vao = context.vertex_array(
            self.prog,
            [
                # Map in_position to the first 2 floats
                # Map in_uvs to the next 2 floats
                (self.vbo, '2f 2f', 'in_position', 'in_uv')
            ],
        )

    def render(self, texture):
        texture.use(0)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)

class CoordinateSystem():
    def __init__(self, context: moderngl.Context):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
        ])
        self.vbo = context.buffer(vertices.ravel().astype('f4'))

        colors = np.array([
            [255/255, 105/255, 97/255],
            [255/255, 105/255, 97/255],
            [97/255, 255/255, 184/255],
            [97/255, 255/255, 184/255],
            [97/255, 168/255, 255/255],
            [97/255, 168/255, 255/255],
        ])
        self.vcbo = context.buffer(colors.ravel().astype('f4'))

        self.program = context.program(
            vertex_shader=mesh_vertex_shader,
            fragment_shader=fragment_shader_flat
        )

        self.vao = context.vertex_array(
            self.program,
            [
                (self.vbo, '3f', 'position'),
                (self.vcbo, '3f', 'color'),
            ]
        )

    def render(self, context, camera):
        self.program['model_view_matrix'].write(camera.view_matrix.astype('f4'))
        self.program['projection_matrix'].write(camera.projection_matrix.astype('f4'))
        self.vao.render(moderngl.LINES)
