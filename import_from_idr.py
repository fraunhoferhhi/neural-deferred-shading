from argparse import ArgumentParser
from PIL import Image
import numpy as np
from pathlib import Path
import torch
import imageio
from nds.core import Camera, View
import cv2

from nds.utils.io import read_views
from nds.utils.geometry import AABB, normalize_aabb

def decompose(P):
    K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    c = c[:3, 0] / c[3]
    t = - R @ c
    # ensure unique scaling of K matrix
    K = K / K[2,2]
    return K, R, t


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True, help="Path to the directory with the IDR scan.")
    parser.add_argument("--bbox", type=Path, required=True, help="Path to the bounding box file.")
    parser.add_argument("--view_dir", type=Path, required=True, help="Path to the directory containing the views.")
    args = parser.parse_args()
    # Read images from folder
    image_path = args.scan / 'image'
    mask_path = args.scan / 'mask'

    image_paths = sorted([path for path in image_path.iterdir() if (path.is_file() and path.suffix == '.png' and not str(path.stem).startswith('.'))])
    mask_paths = sorted([path for path in mask_path.iterdir() if (path.is_file() and path.suffix == '.png' and not str(path.stem).startswith('.'))])

    # Read cameras
    cameras = np.load(args.scan / 'cameras.npz')

    # Make a unit aabb 
    #! WARNING! This assumes a symmetric bbox (aka cube) if the bbox is not a cube 
    #! this will not give the bbox back, but something similar
    bbox = np.array([[-.5, -.5, -.5],
                    [.5,  .5,  .5]], dtype=np.float32)

    args.view_dir.mkdir(parents=True, exist_ok=True)
    for i, view in enumerate(image_paths):
        K, R, t = decompose(cameras[f"world_mat_{i}"][:3,:])
        np.savetxt(args.view_dir / f'cam{i:06d}_r.txt', R, fmt='%.20f')
        np.savetxt(args.view_dir / f'cam{i:06d}_t.txt', t, fmt='%.20f')
        np.savetxt(args.view_dir / f'cam{i:06d}_k.txt', K, fmt='%.20f')

        A_inv = cameras[f"scale_mat_{i}"]
        bbox_denormalized = (bbox @ A_inv[:3, :3].T) + A_inv[:3, 3][np.newaxis, :]


    aabb = AABB(bbox_denormalized)
    aabb.save(args.bbox)
    
    for idx, (image, mask) in enumerate(zip(image_paths, mask_paths)):

        color = np.concatenate((imageio.imread(image),
                                imageio.imread(mask, pilmode='L')[..., np.newaxis]), 
                                axis=-1)
        imageio.imwrite(args.view_dir / f'cam{idx:06d}.png', color)

