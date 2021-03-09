import scipy.constants
from scipy.signal import fftconvolve, oaconvolve

import numpy as np
from numpy import sqrt, arctan, arctan2, log


def lafun(x,y,z):
    """
    Indefinite integral for the Coulomb potential
    
    \int 1/r dx dy dz
    
    """

    r=sqrt(x**2+y**2+z**2)
    
    res = -0.5*z**2*arctan(x*y/(z*r)) - 0.5*y**2*arctan(x*z/(y*r)) - 0.5*x**2*arctan(y*z/(x*r)) \
          + y*z*log(x+r) + x*z*log(y+r) + x*y*log(z+r)
    return res

def xlafun(x, y, z):  
    """
    
    Indefinite integral for the x component of the electric field 
    
    \int x/r^3 dx dy dz
    
    
    Integrals for Ey, Ez can be evaluated by calling:
        Ey: xlafun(y, z, x)   
        Ez: xlafun(z, x, y)  
        
    Should not be evaluated exactly on the coordinate axes. 
    
    """
    r=np.sqrt(x**2+y**2+z**2)

    #return x*arctan2(z, x)+x*arctan2(y*z, (x*r)) - z*log(y+r) - y*log(z+r) # Don't use because of branch cut
    return x *(arctan(z/x) + arctan(y*z/(x*r))) - z*log(y+r) - y*log(z+r)

    # ylafun is the same with y, z, x arguments
    #res= y*arctan2(x/y)+y*arctan2(z*x/(y*r))-x*log(z+r)-z*log(x+r)
    

def offset_symmetric_vec(n, delta):
    return np.arange(-n,n,1)*delta + delta/2

def igf_mesh3(rho_shape, deltas, gamma=1, component=None):
    """
    Returns the integrated Green function mesh appropriate to be convolved with a 
    charge mesh of shape rho_shape
    
    Parameters
    ----------
    shape : tuple(int, int, int)
        Shape of the charge mesh
    
    deltas : tuple(float, float, float)
        mesh spacing corresonding to dx, dy, dz
        
    gamma : float
        relativistic gamma
        
    component:
        'coulomb'
        'x'
        'y'
        'z'
        
    Returns
    -------
    
    GF : np.array
        Green function array of shape (2*rho_shape -1)
        
    
    """
    
    dx, dy, dz = tuple(deltas) # Convenience
    
    # Make an offset grid
    D = [dx, dy, dz]
    # Get the correct relativistic scaling
    if component == 'z':
        D[2]*=gamma 
    
    vecs = [offset_symmetric_vec(n, delta) for n, delta in zip(rho_shape, D)] 

    meshes = np.meshgrid(*vecs, indexing='ij')
    
    if component == 'coulomb':
        func = lafun 
    elif component == 'x':
        func = lambda x, y, z: xlafun(x, y, z)
    elif component == 'y':
        func = lambda x, y, z: xlafun(y, z, x)    
    elif component == 'z':
        func = lambda x, y, z: xlafun(z, x, y)      
    else:
        raise ValueError(f'Invalid component: {component}')    
    
    # Evaluate on the offset grid
    GG = func(*meshes)
    
    # Evaluate the indefinite integral over the cube 
    #       (x2,y2,z2) -    (x1,y2,z2) -    (x2,y1,z2) -    (x2,y2,z1) -    (x1,y1,z1)   +    (x1,y1,z2)   +    (x1,y2,z1)  +    (x2,y1,z1)
    res = GG[1:,1:,1:] - GG[:-1,1:,1:] - GG[1:,:-1,1:] - GG[1:,1:,:-1] - GG[:-1,:-1,:-1] + GG[:-1,:-1,1:] + GG[:-1,1:,:-1]  + GG[1:,:-1,:-1]

    if component in ['z', 'coulomb']:
        factor = 1/(dx*dy*dz*gamma)
    else:
        factor = 1/(dx*dy*dz)    
    
    return res*factor



def spacecharge_mesh(rho_mesh, deltas, gamma=1, component=None):
    
    # Green gunction
    green_mesh = igf_mesh3(rho_mesh.shape, deltas, gamma=gamma, component=component)
    
    # Convolution
    field_mesh = fftconvolve(rho_mesh, green_mesh, mode='same')
    
    # Factor to convert to V/m
    factor = 1/(4*np.pi*scipy.constants.epsilon_0)
    
    return factor*field_mesh