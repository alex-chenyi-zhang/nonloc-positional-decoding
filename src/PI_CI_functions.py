import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.special import xlogy

def exponential_mean_function(x_positions, lamb):
    """
    Computes the mean function of a Gaussian Process with exponential decay.

    Parameters:
    x_positions (array-like): Vector of x positions.
    lamb (float): Decay constant of the exponential function.

    Returns:
    np.ndarray: Mean function of the Gaussian Process.
    """
    return np.exp(-x_positions / lamb)

def sample_from_gp(x_positions, covariance_kernel, corr_length, delta_x, sigma_ext, sigma_int, mu_S, lamb):
    """
    Samples from a Gaussian Process with the given mean function and covariance kernel with varying correlation length 
    scale. Then a spatially independent noise of amplitude sigma_int is added to the samples. 
    (For now we just use expoenential decay as mean function)

    Parameters:
    x_positions (2D array): Vector of x positions.
    covariance_kernel (string): Type of kernel to use in the Gaussian Process.
    corr_length (float): Correlation length scale of the covariance kernel, expressed in number of neighbors.
    delta_x (float): Typical spacing between cells
    sigma_int (float): Standard deviation of the spatially independent component of the noise.
    sigma_ext (float): Standard deviation of the spatially correlated noise.
    mu_S (float): Mean value scaling factor.
    lamb (float): Decay constant of the exponential function

    Returns:
    np.ndarray: Samples from the Gaussian Process.
    """
    n_embryos = x_positions.shape[0]
    n_cells = x_positions.shape[1]
    mean_function = exponential_mean_function
    # Compute the mean values evaluated at the x_positions
    mean_values = mean_function(x_positions, lamb)
    
    if covariance_kernel == 'SquaredExponential':
        kernel = sigma_ext**2 * RBF(length_scale=corr_length * delta_x)
    elif covariance_kernel == 'SimpleExponential':
        kernel = sigma_ext**2 * Matern(length_scale=corr_length * delta_x, nu=0.5)
    else:
        raise ValueError('Invalid covariance kernel. Supported kernels are: "Squared Exponential" and "Simple Exponential"')

    # Create a Gaussian process regressor with the selected kernel
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0)

    samples = np.zeros((n_embryos, n_cells))
    # Sample from the Gaussian Process, i.e. the spatially correlated noise
    for i_embryo in range(n_embryos):
        seed = np.random.randint(0, 2**32 - 1)
        samples[i_embryo,:] = gp.sample_y(x_positions[i_embryo].reshape(-1, 1), random_state=seed).flatten()


    # Rescale the fluctuations by the mean function
    samples *= mean_values

    # Add mean function and spatially uncorrelated fluctuations
    if sigma_int != 0:
        samples_with_mean = mu_S * mean_values + samples + np.random.normal(0, sigma_int, (n_embryos, n_cells))
    else:    
        samples_with_mean = mu_S * mean_values + samples

    return samples_with_mean
