import numpy as np


def deposit_particles(xyz, weights=None, bins=(32,32,32), range=None):
    """
    Simple histogramming of particles.
    
    Parameters
    ----------
    xyz : np.array 
        Array of particle positions in [m[
        with shape (n_particles, 3) 
    
    weights: np.array
        Array of weights in [C]
    
    Returns
    -------
    tupule of
    
    rho_mesh : np.array 
            Charge mesh
            rho_mesh.sum() will equal weights.sum(), the total charge in [C]
    
    deltas : (float, float, float)
        Mesh spacing in [m]
    
    coord_vecs : [np.array, np.array, np.array]
        coordinate vectors to create coordinate meshes, e.g. 
            X, Y, Z = np.meshgrid(*coord_vecs, indexing='ij')  # Must have this indexing!!!
    
    """
    rho_mesh, edges = np.histogramdd(xyz, bins=bins, range = range, weights=weights )
    
    # Get deltas
    deltas = np.array([np.mean(np.diff(e)) for e in edges])

    coord_vecs = np.array([np.linspace(e.min(), e.max(), bins[i]) for i, e in enumerate(edges)])
    
    #return dict(rho_mesh=rho_mesh, deltas=deltas, coord_vecs=coord_vecs)
    return rho_mesh, deltas, coord_vecs
