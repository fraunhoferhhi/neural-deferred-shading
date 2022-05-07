from pathlib import Path
import matplotlib.pyplot as plt
import math

def create_mosaic_figure(images):
    num_images = len(images)
    num_rows = int(math.ceil(math.sqrt(num_images)))
    num_cols = num_rows

    fig, axs = plt.subplots(num_rows, num_rows, figsize=(2.5*num_cols, 2.5*num_rows))

    for i, image in enumerate(images):
        row = i // num_rows
        col = i % num_cols
        axs[row][col].imshow(image)

    for i in range(num_images, num_rows*num_cols):
        ax = axs.flatten()[i]
        ax.axis('off')

    return fig, axs

def visualize_views(views, highlight_silhouette=False, show=True, save_path: Path=None):
    """ Visualize a list of views by plotting their color images as a mosaic.

    Args:
        views: The views to visualize.
        show: Indicator whether to display the create figure or not.
        save_path (optional): Path to save the figure to.
    """
    if highlight_silhouette:
        images = [((v.mask + 1.0) * v.color).clamp_(min=0.0, max=1.0).cpu() for v in views]
    else:
        images = [v.color.cpu() for v in views]

    fig, axs = create_mosaic_figure(images) 

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    if not show:
        plt.close(fig)

def visualize_mesh_as_overlay(renderer, views, mesh, show=True, save_path: Path=None):
    """ Visualize a mesh rendered as overlay to the given views. 
    The individual images are aranged as mosaic.

    Args:
        views: The views to use for rendering.
        mesh: The mesh to visualize.
        show: Indicator whether to display the create figure or not.
        save_path (optional): Path to save the figure to.
    """

    gbuffers = renderer.render(views, mesh, channels=["mask"])

    overlay_images = []
    for view, gbuffer in zip(views, gbuffers):
        color_overlay = (gbuffer["mask"] + 1.0) * view.color
        overlay_images += [color_overlay.clamp_(min=0.0, max=1.0).cpu()]

    fig, axs = create_mosaic_figure(overlay_images)

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    if not show:
        plt.close(fig)