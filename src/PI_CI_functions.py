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

def sample_from_gp(x_positions, mean_function, covariance_kernel, corr_length, delta_x, sigma_S, sigma_int, mu_S):
    """
    Samples from a Gaussian Process with the given mean function and covariance kernel with varying correlation length 
    scale. Then a spatially independent noise of amplitude sigma_int is added to the samples.

    Parameters:
    x_positions (2D array): Vector of x positions.
    mean_function (callable): Mean function of the Gaussian Process.
    covariance_kernel (string): Type of kernel to use in the Gaussian Process.
    corr_length (float): Correlation length scale of the covariance kernel, expressed in number of neighbors.
    delta_x (float): Typical spacing between cells
    sigma_int (float): Standard deviation of the spatially independent component of the noise.
    sigma_S (float): Standard deviation of the spatially correlated noise.
    mu_S (float): Mean value scaling factor.

    Returns:
    np.ndarray: Samples from the Gaussian Process.
    """
    n_embryos = x_positions.shape[0]
    n_cells = x_positions.shape[1]
    
    # Compute the mean values evaluated at the x_positions
    mean_values = mean_function(x_positions)
    
    if covariance_kernel == 'Squared Exponential':
        kernel = sigma_S**2 * RBF(length_scale=corr_length)
    elif covariance_kernel == 'Simple Exponential':
        kernel = sigma_S**2 * Matern(length_scale=corr_length, nu=0.5)
    else:
        raise ValueError('Invalid covariance kernel. Supported kernels are: "Squared Exponential" and "Simple Exponential"')

    # Create a Gaussian process regressor with the selected kernel
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0)

    # Sample from the Gaussian Process, i.e. the spatially correlated noise
    samples = gp.sample_y(x_positions, n_samples=n_embryos)

    # Rescale the fluctuations by the mean function
    samples *= mean_values

    # Add mean function and spatially uncorrelated fluctuations
    if sigma_int != 0:
        samples_with_mean = mu_S * mean_values + samples + np.random.normal(0, sigma_int, (n_embryos, n_cells))
    else:    
        samples_with_mean = mu_S * mean_values + samples

    return samples_with_mean.T
