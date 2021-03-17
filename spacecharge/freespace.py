import scipy.constants
from scipy.signal import fftconvolve, oaconvolve

import numpy as np
from numpy import sqrt, arctan ,arctanh, arctan2, log


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
    
    = x*arctan((y*z)/(r*x)) -z*log(r+y) + y*log((r-z)/(r+z))/2
    
    Integrals for Ey, Ez can be evaluated by calling:
        Ey: xlafun(y, z, x)   
        Ez: xlafun(z, x, y)  
        
    Should not be evaluated exactly on the coordinate axes. 
    
    """
    r=np.sqrt(x**2+y**2+z**2)

    #return x*arctan2(z, x)+x*arctan2(y*z, (x*r)) - z*log(y+r) - y*log(z+r) # Don't use because of branch cut

    #Form 0 (original)
    #return x *(arctan(z/x) + arctan(y*z/(x*r))) - z*log(y+r) - y*log(z+r)

    # Form 4 (Slightly faster)
    return x*arctan((y*z)/(r*x)) -z*log(r+y) + y*log((r-z)/(r+z))/2
        
def offset_symmetric_vec(n, delta):
    return np.arange(-n,n+1,1)*delta + delta/2

def igf_mesh3(rho_shape, deltas, gamma=1, offset=(0,0,0), component=None):
    """
    Returns the integrated Green function (IGF) mesh appropriate to be convolved with a 
    charge mesh of shape rho_shape.
    
    Parameters
    ----------
    shape : tuple(int, int, int)
        Shape of the charge mesh
    
    deltas : tuple(float, float, float)
        mesh spacing corresonding to dx, dy, dz
        
    gamma : float
        relativistic gamma
        
    offset : tuple(float, float, float)
        Offset coordinates for the center of the grid in [m]. Default: (0,0,0)
        For example, an offset of (0,0,10) can be used to compute the field at z=+10 m relative to the rho_mesh center. 
        
    component:
        'coulomb'
        'x'
        'y'
        'z'
        
    Returns
    -------
    
    GF : np.array
        Green function array of shape (2*rho_shape)
        
        The origin will be at index rho_shape-1, and should be zero by symmetry for x, y, z components
        
    
    """
    
    dx, dy, dz = tuple(deltas) # Convenience
    
    # Boost to the rest frame
    D = [dx, dy, dz*gamma]
    offset = offset[0], offset[1], offset[2]*gamma # Note that this is an overall offset
    
    # Make an offset grid
    vecs = [offset_symmetric_vec(n, delta)+o for n, delta, o in zip(rho_shape, D, offset)] 
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



def spacecharge_mesh(rho_mesh, deltas, gamma=1, offset=(0,0,0), component=None):
    
    # Green gunction
    green_mesh = igf_mesh3(rho_mesh.shape, deltas, gamma=gamma, offset=offset, component=component)
    
    # Convolution
    field_mesh = fftconvolve(rho_mesh, green_mesh, mode='same')
    
    # Factor to convert to V/m
    factor = 1/(4*np.pi*scipy.constants.epsilon_0)
    
    return factor*field_mesh