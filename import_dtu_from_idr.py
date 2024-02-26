from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import imageio
import cv2
from tqdm import tqdm

from nds.utils.geometry import AABB

def decompose(P):
    K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    c = c[:3, 0] / c[3]
    t = - R @ c
    # ensure unique scaling of K matrix
    K = K / K[2,2]
    return K, R, t

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Directory containing the DTU data in the IDR format.")
    parser.add_argument("output_dir", type=Path, help="Output directory for the DTU data in NDS format.")
    parser.add_argument("--skip_existing", default=False, action='store_true', help="Skip conversion of scans that already exist in the output directory")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    for scan_dir in [directory for directory in input_dir.iterdir() if directory.is_dir()]:
        print(f"-- Converting {scan_dir.name}")

        scan_output_dir = output_dir / scan_dir.name
        if args.skip_existing and scan_output_dir.exists():
            print("-- Skipping, because output directory already exists")
            continue

        scan_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect the image and mask paths
        image_dir = scan_dir / 'image'
        mask_dir  = scan_dir / 'mask'

        image_paths = sorted([path for path in image_dir.iterdir() if (path.is_file() and path.suffix == '.png' and not str(path.stem).startswith('.'))])
        mask_paths  = sorted([path for path in mask_dir.iterdir() if (path.is_file() and path.suffix == '.png' and not str(path.stem).startswith('.'))])

        # Read cameras
        cameras = np.load(scan_dir / 'cameras.npz')

        # Make a unit aabb 
        #! WARNING! This assumes a symmetric bbox (aka cube) if the bbox is not a cube 
        #! this will not give the bbox back, but something similar
        bbox = np.array([[-.5, -.5, -.5],
                         [.5,  .5,  .5]], dtype=np.float32)

        views_output_dir = scan_output_dir / "views"
        views_output_dir.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(image_paths):
            K, R, t = decompose(cameras[f"world_mat_{i}"][:3,:])
            np.savetxt(views_output_dir / f'cam{i:06d}_r.txt', R, fmt='%.20f')
            np.savetxt(views_output_dir / f'cam{i:06d}_t.txt', t, fmt='%.20f')
            np.savetxt(views_output_dir / f'cam{i:06d}_k.txt', K, fmt='%.20f')

            A_inv = cameras[f"scale_mat_{i}"]
            bbox_denormalized = (bbox @ A_inv[:3, :3].T) + A_inv[:3, 3][np.newaxis, :]

        # Save the bounding box
        # NOTE (MW): This assumes that all scale_mats are equal for a scan (which seems to be the case)
        #            Otherwise we would choose the last one, which seems a bit arbitrary
        aabb = AABB(bbox_denormalized)
        aabb.save(scan_output_dir / "bbox.txt")
        
        # Convert the images and masks to a single RGBA PNG
        for idx, (image, mask) in tqdm(enumerate(zip(image_paths, mask_paths)), leave=False):
            color = np.concatenate((imageio.imread(image),
                                    imageio.imread(mask, pilmode='L')[..., np.newaxis]), 
                                    axis=-1)
            imageio.imwrite(views_output_dir / f'cam{idx:06d}.png', color)