from argparse import ArgumentParser
import functools
import glfw
import math
from pathlib import Path
import torch

from nds.modules import SpaceNormalization, NeuralShader
from nds.utils import AABB, read_mesh
from nds.viewer import MeshViewer

def get_rotation(angle, axis='X'):
    R = torch.eye(4)
    if axis == 'X':
        R[1, 1] = math.cos(angle)
        R[1, 2] = math.sin(angle)
        R[2, 1] = -math.sin(angle)
        R[2, 2] = math.cos(angle)
    elif axis == 'Y':
        R[0, 0] = math.cos(angle)
        R[0, 2] = math.sin(angle)
        R[2, 0] = -math.sin(angle)
        R[2, 2] = math.cos(angle)
    elif axis == 'Z':
        R[0, 0] = math.cos(angle)
        R[0, 1] = math.sin(angle)
        R[1, 0] = -math.sin(angle)
        R[1, 1] = math.cos(angle)
    return R

def key_callback(userdata, key, scancode, action, mods):   
    if action == glfw.RELEASE:
        if key == glfw.KEY_1:
            userdata['shading_mode'] = 'face'
            userdata['viewer'].set_shading_mode(userdata['shading_mode'])
        elif key == glfw.KEY_2:
            userdata['shading_mode'] = 'smooth'
            userdata['viewer'].set_shading_mode(userdata['shading_mode'])
        elif key == glfw.KEY_3:
            userdata['shading_mode'] = 'normal'
            userdata['viewer'].set_shading_mode(userdata['shading_mode'])
        elif key == glfw.KEY_4 and userdata['shader'] is not None:
            userdata['shading_mode'] = 'neural'
            userdata['viewer'].set_shading_mode(userdata['shading_mode'])

    if action == glfw.REPEAT or action == glfw.RELEASE:
        if key == glfw.KEY_LEFT or key == glfw.KEY_RIGHT or key == glfw.KEY_UP or key == glfw.KEY_DOWN:
            # Rotation
            R = torch.eye(4)
            if key == glfw.KEY_LEFT or key == glfw.KEY_RIGHT:
                R = get_rotation(0.1 if key == glfw.KEY_LEFT else -0.1, axis='Y') @ R
            elif key == glfw.KEY_UP or key == glfw.KEY_DOWN:
                R = get_rotation(0.1 if key == glfw.KEY_UP else -0.1, axis='Z') @ R

            userdata['model_matrix'] = R @ userdata['model_matrix']
            userdata['viewer'].set_model_matrix(userdata['model_matrix'])

if __name__ == '__main__':
    parser = ArgumentParser("Interactive Mesh Viewer with Neural Deferred Shading")
    parser.add_argument("--mesh", type=Path, required=True, help="Path to the triangle mesh")
    parser.add_argument("--shader", type=Path, default=None, help="Path to the neural shader")
    parser.add_argument("--bbox", type=Path, required=True, help="Path to the bounding box")
    args = parser.parse_args()

    mesh = read_mesh(args.mesh, device='cpu')
    aabb = AABB.load(args.bbox)

    normalization = SpaceNormalization(aabb.corners)
    mesh = normalization.normalize_mesh(mesh)
    
    device = 'cuda:0'
    shader = NeuralShader.load(args.shader, device=device) if args.shader is not None else None

    viewer = MeshViewer(neural_shader=shader, device=device)

    userdata = {
        'viewer': viewer,
        'mesh': mesh,
        'shader': shader,
        'aabb': aabb,
        'model_matrix': torch.eye(4),
        'shading_mode': 'face'
    }
    viewer.user_key_callback = functools.partial(key_callback, userdata)
    viewer.set_shading_mode(userdata['shading_mode'])
    viewer.set_mesh(mesh.vertices.numpy(), mesh.indices.numpy(), n=mesh.vertex_normals.numpy())

    print("Keyboard Controls:")
    print("(Left/Right Arrow): Rotate around y axis")
    print("(Up/Down Arrow): Rotate around z axis")
    print("-- Shading Modes")
    print("(1) Face shading")
    print("(2) Smooth shading")
    print("(3) Normals")
    if args.shader is not None:
        print("(4) Neural Shader")