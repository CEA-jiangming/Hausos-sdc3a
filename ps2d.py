import numpy as np
from scipy import stats
from scipy import signal
from numpy import linalg as LA 


def calculate_2d_power_spectrum(cube, pixel_length, nbin=10, key='std', bin_edges_klos=None, bin_edges_kper=None, apply_window=False):


    """
    Calculate the 2D cylindrical mean power spectrum from a 3D image spectral cube.

    Parameters
    ----------
    cube : numpy.ndarray
        3D image spectral cube.
    pixel_length[1]:  size_perp_cMpc/pixel : float
        Size of the cube in comoving megaparsecs perpendicular to the line of sight.
    length[0]: size_LOS_cMpc/pixel : float
        Size of the cube in comoving megaparsecs along the line of sight.
    nbin : int
        Number of perpendicular bins or line-of-sight bins. Must be the same for statistic_2d.
    apply_window : bool, optional
        Whether to apply a window function along the frequency axis (default is False).
    bin_edges_klos : numpy.ndarray or None, optional
        Bin edges for line-of-sight wavenumbers. If None, they will be computed.
    bin_edges_kper : numpy.ndarray or None, optional
        Bin edges for perpendicular wavenumbers. If None, they will be computed.

    Returns
    -------
    ps2d : numpy.ndarray
        2D cylindrical mean power spectrum. units = Mpc^3 [klos, kperp]
    bin_central_kper : numpy.ndarray
        Central bin values for perpendicular wavenumbers.
    bin_central_klos : numpy.ndarray
        Central bin values for line-of-sight wavenumbers.
    """

    # Create a copy of the cube for windowing if required
    if apply_window:
        cube_windowed = cube.copy()
        window = signal.windows.nuttall(cube_windowed.shape[0])
        window /= window.sum()
        cube_windowed = cube_windowed * window[:, np.newaxis, np.newaxis]
    else:
        cube_windowed = cube

    # Perform 3D FFT
    nz, nx, ny = cube_windowed.shape
    
    size_LOS_cMpc  = nz*pixel_length[0] 
    size_perp_cMpc = nx*pixel_length[1]

    
    volume = size_perp_cMpc ** 2 * size_LOS_cMpc
    voxsize = volume / (nx * ny * nz)

    cube_fft = np.fft.fftn(cube_windowed.astype('float64'))

    # Calculate power spectrum
    ps3d = np.abs(cube_fft) ** 2  # dimensionless
    ps3d *= voxsize ** 2 / volume  # Mpc^3

    # Calculate cylindrical wave numbers
    kz = np.abs(2 * np.pi * np.fft.fftfreq(nz, d= pixel_length[0]))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=pixel_length[1])
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=pixel_length[1])
    
    
    
    kz3d, kx3d, ky3d = np.meshgrid(kz,kx, ky, indexing='ij') # not change the order as cube being same order
    kper3d = np.sqrt(kx3d ** 2 + ky3d ** 2)

    
    # Flatten the arrays
    kper_flat = kper3d.flatten()
    klos_flat = kz3d.flatten()
    ps3d_flat = ps3d.flatten()

    # Calculate the bin edges for binning statistics if not provided
    if bin_edges_klos is None:
        bin_edges_klos = np.linspace(1e-4, klos_flat.max(), nbin + 1)
    if bin_edges_kper is None:
        bin_edges_kper = np.linspace(1e-4, kper_flat.max(), nbin + 1)

    # Calculate binned statistics of the power spectrum
    ps2d,los_edge,per_edge,_ = stats.binned_statistic_2d(klos_flat,kper_flat, values=ps3d_flat, statistic=key,
                                                     bins=[bin_edges_klos,bin_edges_kper])
    
    del cube_windowed, cube_fft, ps3d, kz3d, kx3d, ky3d, kper3d, kper_flat, klos_flat, ps3d_flat


    return ps2d,los_edge,per_edge






