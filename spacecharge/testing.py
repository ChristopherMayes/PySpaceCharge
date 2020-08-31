import numpy as np


def test_particles(n_particle=1000000,
               sigma_x = 0.001,
               sigma_y = 0.001,
               sigma_z = 0.0001,
               total_charge = 1e-9,
               mean_x = 0,
               mean_y = 0,
               mean_z = 0):
    """
    Test Gaussian bunch to compare with OpenSpaceCharge
    
    Returns tuple:
        particles_xyz, weights
    """
    
    mean = [mean_x, mean_y, mean_z]
    cov = np.diag([sigma_x**2, sigma_y**2, sigma_z**2])
    
    particles_xyz = np.random.multivariate_normal(mean, cov, n_particle)
    weights = np.full(n_particle, total_charge/n_particle)
    
    return particles_xyz, weights

