import torch

def get_coordinate_norm(coords, width, height):
    # For coords shape (Bx)HxWx2 the norm will have shape ((1, )1, 1, 2)
    dims = [1]*(len(coords.shape)-1) + [-1]
    return torch.tensor([width, height], dtype=torch.float32, device=coords.device).view(*dims)

def normalize_coords(coords, width, height):
    norm = get_coordinate_norm(coords, width, height)
    return (coords / norm) * 2 - 1

def denormalize_coords(coords, width, height):
    norm = get_coordinate_norm(coords, width, height)
    # Select the integer pixel coordinate closest to the float coordinate
    return ((coords + 1) / 2 * norm + 0.5).to(dtype=torch.int64)

def sample(image, pixel_coords):
    coordinates = normalize_coords(pixel_coords[..., :2], image.shape[1], image.shape[0]).to(image.device)
    samples = torch.nn.functional.grid_sample(image.permute(2, 0, 1).unsqueeze(0), coordinates.unsqueeze(0))
    return samples.squeeze(0).permute(1, 2, 0)