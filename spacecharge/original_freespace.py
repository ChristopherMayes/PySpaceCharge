import numpy as np
from numpy import sqrt, arctan, arctan2, log




def lafun(x,y,z):
# lafun is the function involving log and atan in the PRSTAB paper (I should find a better name for this function)

    r=sqrt(x**2+y**2+z**2)
    res=-0.5*z**2*arctan(x*y/(z*r))-0.5*y**2*arctan(x*z/(y*r))-0.5*x**2*arctan(y*z/(x*r)) \
     +y*z*log(x+r)+x*z*log(y+r)+x*y*log(z+r)
    return res

def igfcoulombfun(u,v,w,gam,dx,dy,dz):

    x1=u-dx/2
    x2=u+dx/2
    y1=v-dy/2
    y2=v+dy/2
    z1=(w-dz/2)*gam
    z2=(w+dz/2)*gam
    ##     res=1.d0/sqrt(u**2+v**2+w**2)  # coulomb
    ##     res=u/(u**2+v**2+w**2)**1.5d0  # x-electric field
    res=lafun(x2,y2,z2)-lafun(x1,y2,z2)-lafun(x2,y1,z2)-lafun(x2,y2,z1)-lafun(x1,y1,z1)+ \
        lafun(x1,y1,z2)+lafun(x1,y2,z1)+lafun(x2,y1,z1)
    res=res/(dx*dy*dz*gam)
    
    return res


def xlafun(x, y, z):  
    """
    
    Indefinite integral over x, y, z of 
    
    """
    r=sqrt(x**2+y**2+z**2)

    res=z-x*arctan2(z, x)+x*arctan2(y*z, (x*r))

    # Handle special cases
    # Hint from: https://stackoverflow.com/a/52209380
    ypr = y+r
    res -= z*log(ypr, out=np.zeros_like(ypr), where=(ypr != 0))
    
    zpr = z+r
    res -= y*log(zpr, out=np.zeros_like(zpr), where=(zpr != 0))
    
    return res

def ylafun(x,y,z):
    """
    
    """
    r=sqrt(x**2+y**2+z**2)

    res=x-y*arctan2(x, y)+y*arctan2(z*x, (y*r))
    
    # Handle special cases
    zpr = z+r
    res -= x*log(zpr, out=np.zeros_like(zpr), where=(zpr != 0))
    
    xpr = x+r
    res -= z*log(xpr, out=np.zeros_like(xpr), where=(xpr != 0 ))

    return res

def zlafun(x,y,z):
    """
    
    """
    r=sqrt(x**2+y**2+z**2)
    
    res=y-z*arctan2(y, z)+z*arctan2(x*y,(z*r))  
    
    # Handle special cases
    xpr = x+r
    res -= y*log(xpr, out=np.zeros_like(xpr), where=(xpr != 0 ))    
    
    ypr = y+r
    res -= x*log(ypr, out=np.zeros_like(ypr), where=(ypr != 0))    
    
    return res

    

def igfun_efield(u, v, w, gamma, dx, dy ,dz, component=None):
    """
    General integrated Green function to compute an electric field component
    
    Original 'Safe' version
    """
    ep=  1e-10 #-13 causes NaN's
    em= -1e-10
    x1=u-dx/2
    x2=u+dx/2
    y1=v-dy/2
    y2=v+dy/2
    z1=(w-dz/2)*gamma
    z2=(w+dz/2)*gamma
    
    # Look for center points
    is_x_center = np.logical_and(x1<0, x2>0)
    is_y_center = np.logical_and(y1<0, y2>0)
    is_z_center = np.logical_and(z1<0, z2>0)
    # is center ix
    ix = np.logical_and(np.logical_and(is_x_center, is_y_center), is_z_center)
  
    funcs = {'x':xlafun, 'y':ylafun, 'z':zlafun}
    func = funcs[component]

    # Make empty array to fill
    res = np.empty(u.shape)
    
    # center calc
    res[ix] = igf_center_general(x1[ix], x2[ix], y1[ix], y2[ix], z1[ix], z2[ix], em, ep, func)
    
    # Noncenter calc
    ix = np.logical_not(ix)
    res[ix] = igf_noncenter_general(x1[ix], x2[ix], y1[ix], y2[ix], z1[ix], z2[ix], func)
    
    if component == 'z':
        factor = 1/(dx*dy*dz*gamma)
    else:
        factor = 1/(dx*dy*dz)
    
    return res*factor
    
def igf_noncenter_general(x1, x2, y1, y2, z1, z2, func):
    """
    Evaluate the 3D integral over cubic domain
    """
    
    return func(x2,y2,z2)-func(x1,y2,z2)-func(x2,y1,z2)-func(x2,y2,z1) \
            -func(x1,y1,z1)+func(x1,y1,z2)+func(x1,y2,z1)+func(x2,y1,z1)

def igf_center_general(x1, x2, y1, y2, z1, z2, em, ep, func):  
    """
    Special case when surrounding the singularity. 
    
    Ask Robert Ryne for details. 
    """
    
    # Send full arrays for vectorization to work
    em = np.full(len(x1), em)
    ep = np.full(len(x1), ep)
    
    return func(em,em,em) \
         -func(x1,em,em)-func(em,y1,em)-func(em,em,z1)-func(x1,y1,z1) \
         +func(x1,y1,em)+func(x1,em,z1)+func(em,y1,z1)+func(x2,em,em) \
         -func(ep,em,em)-func(x2,y1,em)-func(x2,em,z1)-func(ep,y1,z1) \
         +func(ep,y1,em)+func(ep,em,z1)+func(x2,y1,z1)+func(ep,y2,ep) \
         -func(x1,y2,ep)-func(ep,ep,ep)-func(ep,y2,z1)-func(x1,ep,z1) \
         +func(x1,ep,ep)+func(x1,y2,z1)+func(ep,ep,z1)+func(ep,ep,z2) \
         -func(x1,ep,z2)-func(ep,y1,z2)-func(ep,ep,ep)-func(x1,y1,ep) \
         +func(x1,y1,z2)+func(x1,ep,ep)+func(ep,y1,ep)+func(x2,y2,ep) \
         -func(ep,y2,ep)-func(x2,ep,ep)-func(x2,y2,z1)-func(ep,ep,z1) \
         +func(ep,ep,ep)+func(ep,y2,z1)+func(x2,ep,z1)+func(x2,ep,z2) \
         -func(ep,ep,z2)-func(x2,y1,z2)-func(x2,ep,ep)-func(ep,y1,ep) \
         +func(ep,y1,z2)+func(ep,ep,ep)+func(x2,y1,ep)+func(ep,y2,z2) \
         -func(x1,y2,z2)-func(ep,ep,z2)-func(ep,y2,ep)-func(x1,ep,ep) \
         +func(x1,ep,z2)+func(x1,y2,ep)+func(ep,ep,ep)+func(x2,y2,z2) \
         -func(ep,y2,z2)-func(x2,ep,z2)-func(x2,y2,ep)-func(ep,ep,ep) \
         +func(ep,ep,z2)+func(ep,y2,ep)+func(x2,ep,ep)




#------------------
# Method 2

def xlafun2(x, y, z):  
    """
    
    Indefinite integral over x, y, z of 
    
    Should not be evaluated exactly on the axes
    
    """
    r=np.sqrt(x**2+y**2+z**2)

    #return x*arctan2(z, x)+x*arctan2(y*z, (x*r)) - z*log(y+r) - y*log(z+r)
    return x*arctan(z/x)+x*arctan(y*z/(x*r)) - z*log(y+r) - y*log(z+r)

    # ylafun is the same with y, z, x arguments
    #res= y*arctan2(x/y)+y*arctan2(z*x/(y*r))-x*log(z+r)-z*log(x+r)
    



def igfun_efield2(u, v, w, gamma, dx, dy ,dz, component=None):
    """
    General integrated Green function to compute an electric field component
   
    Evaluates the intefinite integral on the cube surrounding points u, w, v
    
    
    """

    x1=u-dx/2
    x2=u+dx/2
    y1=v-dy/2
    y2=v+dy/2
    z1=(w-dz/2)*gamma
    z2=(w+dz/2)*gamma
  
    if component == 'coulomb':
        func = lafun 
    elif component == 'x':
        func = lambda x, y, z: xlafun2(x, y, z)
    elif component == 'y':
        func = lambda x, y, z: xlafun2(y, z, x)    
    elif component == 'z':
        func = lambda x, y, z: xlafun2(z, x, y)      
    else:
        raise ValueError(f'Invalid component: {component}')
    
    res = igf_noncenter_general(x1, x2, y1, y2, z1, z2, func)
    
    if component in ['z', 'coulomb']:
        factor = 1/(dx*dy*dz*gamma)
    else:
        factor = 1/(dx*dy*dz)
    
    return res*factor

def symmetric_vec(n, delta):
    """
    
    Returns a symmetric coordinate vector of size 2n+1 with spacing delta
    Element [n] is the origin = 0
    
    """
    return np.arange(-(n-1),n,1)*delta

def igf_mesh2(shape, deltas, gamma=1, component=None):
    """
    Returns the integrated Green function mesh with new shape 2*shape+1
    
    """
    
    vecs = [symmetric_vec(n, delta) for n, delta in zip(shape, deltas)]
    #print(vecs)
    X2, Y2, Z2 = np.meshgrid(*vecs, indexing='ij')
    
    G2 = igfun_efield2(X2, Y2, Z2, gamma, *deltas, component=component)
    
    return G2

