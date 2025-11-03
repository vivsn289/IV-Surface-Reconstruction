import numpy as np
from scipy.interpolate import griddata


def bicubic_interpolation(surface: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Performs bicubic interpolation to inpaint a surface.

    Args:
        surface (np.ndarray): The (H, W) surface with missing values (zeros).
        mask (np.ndarray): The (H, W) boolean mask where True indicates observed pixels.

    Returns:
        np.ndarray: The inpainted (H, W) surface.
    """
    h, w = surface.shape

    # Create coordinates for the grid
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    # Get the coordinates and values of the observed points
    points = np.array([yy[mask], xx[mask]]).T
    values = surface[mask]

    # The coordinates where we want to interpolate
    grid_x, grid_y = np.mgrid[0:h, 0:w]

    # Perform interpolation
    inpainted_surface = griddata(
        points, values, (grid_y, grid_x), method="cubic", fill_value=np.mean(values)
    )

    return inpainted_surface.astype(np.float32)
