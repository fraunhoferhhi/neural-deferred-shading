from pathlib import Path
import numpy as np
import torch

from nds.core import Mesh
from nds.utils.io import read_mesh
from nds.utils.geometry import find_connected_faces

TEST_DATA_DIR = Path(__file__).parent / "data"

def _sort_face_pairs(face_pairs: torch.Tensor):
    face_pairs_ = face_pairs.cpu().numpy()
    face_pairs_ = np.sort(face_pairs_, axis=1)
    return torch.from_numpy(face_pairs_[np.lexsort((face_pairs_[:, 1], face_pairs_[:, 0]))]).to(device=face_pairs.device)

def test_find_connected_faces():
    # Create a simple mesh with 4 faces and 5 vertices
    f = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.int64)

    # Find connected faces
    connected_faces = find_connected_faces(f)

    # Check that the connected faces are correct
    connected_faces_expected = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64)
    
    connected_faces = _sort_face_pairs(connected_faces)
    connected_faces_expected = _sort_face_pairs(connected_faces_expected)
    assert torch.equal(connected_faces, connected_faces_expected)

def test_find_connected_faces_vectorized():
    # Compare legacy and vectorized implementations on a real mesh

    mesh: Mesh = read_mesh(TEST_DATA_DIR / "skull.obj")

    connected_faces_legacy = find_connected_faces(mesh.indices, use_legacy_impl=True)
    connected_faces_vectorized = find_connected_faces(mesh.indices, use_legacy_impl=False)

    connected_faces_legacy = _sort_face_pairs(connected_faces_legacy)
    connected_faces_vectorized = _sort_face_pairs(connected_faces_vectorized)
    assert torch.equal(connected_faces_legacy, connected_faces_vectorized)