import numpy as np


def compute_steering_matrix(N=10, M=10, dx=0.5, dy=0.5,
                             theta_range=(0, 90, 19),
                             phi_range=(0, 360, 37)):
 
    N, M : int
        Number of elements along each axis of the array.
    dx, dy : float
        Element spacing (in wavelengths) along x and y.
    theta_range, phi_range : tuple(start, stop, num)
        Passed to np.linspace to build the theta/phi angle grids (degrees).

    Returns
    -------
    steering : ndarray, shape (N*M, n_theta, n_phi)
        Complex steering matrix.
    theta_deg, phi_deg : ndarray
        Angle grids in degrees.
    """
    theta_deg = np.linspace(*theta_range)
    phi_deg = np.linspace(*phi_range)

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    m = np.arange(M)
    n = np.arange(N)

    m_grid, n_grid = np.meshgrid(m, n, indexing='ij')
    m_flat = m_grid.flatten()
    n_flat = n_grid.flatten()

    k_x = 2 * np.pi * dx * np.outer(np.sin(theta), np.cos(phi))
    k_y = 2 * np.pi * dy * np.outer(np.sin(theta), np.sin(phi))

    phase = (
        m_flat[:, None, None] * k_x[None, :, :] +
        n_flat[:, None, None] * k_y[None, :, :]
    )

    steering = np.exp(1j * phase)

    return steering, theta_deg, phi_deg
