import numpy as np
from scipy import sparse
import warnings
from time import time

class Index:
    """
    Holds information about the location of various boundary nodes in a node array.
    Essentially a dictionary.
    """
    def __init__(self, x0, x1, y0, y1, z0, z1, gh):
        self.z0 = z0
        self.z1 = z1
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.gh = gh
    def __str__(self):
        return f'x0:{len(self.x0)}, x1:{len(self.x1)}, y0:{len(self.y0)}, y1:{len(self.y1)}, z0:{len(self.z0)}, z1:{len(self.z1)}'

def zero_rows(A, rows):
    if isinstance(A,sparse.lil_matrix):
        A.rows[rows] = [[] for j in rows]
        A.data[rows] = [[] for j in rows]
    elif isinstance(A, sparse.csr_matrix):
        r, c = A.shape
        mask = np.ones((r,), dtype=bool)
        mask[rows] = False
        nnz_per_row = np.diff(A.indptr)

        mask = np.repeat(mask, nnz_per_row)
        nnz_per_row[rows] = 0
        A.data = A.data[mask]
        A.indices = A.indices[mask]
        A.indptr[1:] = np.cumsum(nnz_per_row)
    else:
        raise ValueError("zero_rows only works for lil. Convert with tolil() first.")
    return A
def node_drop_3d(box, ninit, dotmax, radius):
    """
    Per van der Sande and Fornberg, adapted from https://github.com/kierav/node_generation
    INPUT:
        box    = [xmin, xmax, ymin, ymax, zmin, zmax]
        ninit  = [numx, numy] initial node layout at lower boundary
        dotmax = upper bound of dots to place
        radius = function radius(xyz) takes nx3 array of locations and returns desired node
                 radius there
    OUTPUT:
        xyz    = nx3 numpy array of node locations, guaranteed n < dotmax
    """
    dotmax = int(dotmax)
    
    dotnr = 0
    np.random.seed(0)
    xyz = np.zeros((dotmax,3))
    excess_height = 0.1
    dx = (box[1]-box[0])/(ninit[0]-1)
    dy = (box[3]-box[2])/(ninit[1]-1)
    xx = np.linspace(box[0],box[1],ninit[0])
    yy = np.linspace(box[2],box[3],ninit[1])
    XX,YY = np.meshgrid(xx,yy, indexing='ij')
    r = radius(np.array(
        [XX.ravel(),YY.ravel(),box[0]*np.ones(XX.ravel().shape)]
    ).transpose())
    pdp = box[4]+0.01*min(r)*np.array(np.random.rand(ninit[0],ninit[1]))
    nodeindices = np.zeros(pdp.shape)
    idx = np.argmin(pdp.ravel())
    zm = pdp.ravel()[idx]
    i1,i2 = np.unravel_index(idx,pdp.shape)
    
    while zm<=(1+excess_height)*box[5] and dotnr < dotmax:
        # --- Add new nodes to generated nodes
        xyz[dotnr,:] = [box[0]+dx*i1,box[2]+dy*i2,pdp[i1,i2]]
        nodeindices[i1,i2] = dotnr
        
        r = radius(xyz[dotnr,:])
        r = r[0] # should be length 1, but numpy array to float conversion is depreciated.
        
        # --- Find PDPs inside the new circle
        ileft =   int(max(0,i1 - np.floor(r/dx)))
        iright =  int(min(ninit[0],i1+np.floor(r/dx)+1))
        ibottom = int(max(0,i2-np.floor(r/dy)))
        itop =    int(min(ninit[1],i2+np.floor(r/dy)+1))
        
        xx = np.arange(ileft,iright)
        yy = np.arange(ibottom,itop)
        X,Y = np.meshgrid(xx,yy, indexing='ij')
        
        # --- Update heights of PDPs within radius
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="invalid value encountered in sqrt")
            height = (r**2 - (dx*(X-i1))**2 - (dy*(Y-i2))**2)**0.5
        height[r**2 < (dx*(X-i1))**2 + (dy*(Y-i2))**2] = 0
        pdp[ileft:iright, ibottom:itop] = np.max(np.stack((pdp[ileft:iright,ibottom:itop],pdp[i1,i2]+height),axis=0),axis=0)
        
        # --- Identify next node location as a local minimum of the PDPs
        ix = np.argmin(pdp[ileft:iright,ibottom:itop], axis=0,keepdims=True)
        zm = np.take_along_axis(pdp[ileft:iright,ibottom:itop],ix,0)
        iy = np.argmin(zm.ravel())
        zm = zm.ravel()[iy]
        i1 = ileft + ix.ravel()[iy]
        i2 = ibottom + iy
        
        searchr = min(2*np.ceil(r/dx)-1,np.floor(ninit[0]/2))
        
        while True:
            if i1-searchr < 0:
                xsearch = np.concatenate(
                    [np.arange(ninit[0]+i1-searchr,ninit[0],dtype='int'),
                     np.arange(0,i1+searchr,dtype='int')]
                )
            elif i1+searchr > ninit[0]:
                xsearch = np.concatenate(
                    [np.arange(i1-searchr,ninit[0],dtype='int'),
                     np.arange(0,i1+searchr-ninit[0],dtype='int')]
                )
            else:
                xsearch = np.arange(i1-searchr,i1+searchr,dtype='int')
            
            if i2-searchr < 0:
                ysearch = np.concatenate(
                    [np.arange(ninit[1]+i2-searchr,ninit[1],dtype='int'),
                     np.arange(0,i2+searchr,dtype='int')]
                )
            elif i2+searchr > ninit[1]:
                ysearch = np.concatenate(
                    [np.arange(i2-searchr,ninit[1],dtype='int'),
                     np.arange(0,i2+searchr-ninit[1],dtype='int')]
                )
            else:
                ysearch = np.arange(i2-searchr,i2+searchr,dtype='int')
            
            ix = np.argmin(pdp[xsearch[:,np.newaxis],ysearch], axis=0,keepdims=True)
            zm = np.take_along_axis(pdp[xsearch[:,np.newaxis],ysearch],ix,0)
            iy = np.argmin(zm)
            ix = ix.ravel()[iy]
            i1 = xsearch[ix]
            i2 = ysearch[iy]
            zm = pdp[i1,i2]
            
            # Stop once a local min has been found within the search radius
            if ix > searchr/2 and ix < len(xsearch)-searchr/2 and iy > searchr/2 and iy < len(ysearch)-searchr/2:
                break
                
        dotnr = dotnr+1
                
    xyz = xyz[1:dotnr+1,:]
    xyz = xyz[xyz[:,2]<=box[5],:]
    
    return xyz
    
    
def potfield(nodes, Bz0, Dx, Dy, Dz, Dxx, Dyy, Dzz, index, config={}):
    """
    Uses finite differences to generate a potential field on a (potentially) scattered domain
    INPUT:
        nodes   = nx3 node locations
        Bz0     = z=0 boundary values on the perpendicular component of B
        Dx..Dzz = Differentiation matrices for the node layout (eg, with rbf-fd)
        index   = mhs Index object
        config  = Dictionary of configuration values. Used keys are:
                - gpu      [default False] boolean to use CUDA for matrix solution
                - lsqr_tol [default 1e-6]  tolerance for matrix solution
                - lsqr_lim [default 1e4]   maximum number of iterations for matrix solution
                - verbose  [default False] Whether to produce printed output on lsqr success
                           (only works with cpu configuration per cupyx limitations)
    OUTPUT:
        B = nx3 magnetic field vector corresponding to the points in nodes
    """
    config.setdefault('gpu',False)
    config.setdefault('lsqr_tol',1e-6)
    config.setdefault('lsqr_lim',1e4)
    config.setdefault('verbose',False)
    n = nodes.shape[0]
    lap = (Dx.dot(Dx)+Dy.dot(Dy)+Dz.dot(Dz))
    I = sparse.eye(n,format='csr')
    lap[index.z0,:] = Dz[index.z0,:]
    lap[index.z1,:] = I[index.z1,:]
    lap[index.x0,:] = I[index.x0,:]
    lap[index.y0,:] = I[index.y0,:]
    lap[index.x1,:] = I[index.x1,:]
    lap[index.y1,:] = I[index.y1,:]
    rhs = np.zeros((n,1))
    rhs[index.z0] = Bz0
    lap = lap.tocsr()
    
    if config['gpu']:
        from cupyx.scipy import sparse as cusp
        import cupy
        lapgpu = cusp.csr_matrix(lap)
        rhsgpu = cupy.asarray(rhs.ravel())
        A, flag, lsiter, lsres = cusp.linalg.lsqr(lapgpu,rhsgpu)[0:4]
    else:
        A, flag, lsiter, lsres = sparse.linalg.lsqr(lap,rhs,atol=config['lsqr_tol'],btol=config['lsqr_tol'],iter_lim=config['lsqr_lim'])[0:4]
    if config['verbose']:
        print(f'Potential solution done: {flag}. Iteration {lsiter} returned residual {lsres}')
    B = np.zeros((n,3))
    B[:,0] = Dx.dot(A)
    B[:,1] = Dy.dot(A)
    B[:,2] = Dz.dot(A)
    
    return B

def resB(B,dens,pres,Bx0,By0,Bz0,Dx,Dy,Dz,Dxx,Dyy,Dzz,index,config={}):
    """
    Provides the mhs residual for a given magnetic field and plasma
    INPUTS:
        B       = nx3 magnetic field vector
        dens    = length n density
        pres    = length n pressure OR nx3 pressure gradient
        Bx0     = dirichlet photospheric boundary value on Bx
        By0     = dirichlet photospheric boundary value on By
        Bz0     = dirichlet photospheric boundary value on Bz
        Dx..Dzz = Differentiation matrices for the node layout (eg, with rbf-fd)
        index   = mhs index object
        config  = Dictionary of configuration values. Used keys are:
                - gamma     [default 1e-4]   Hyperviscocity weight
                - g         [default 1]      Gravitational constant
                - weight_z0 [default 1]      Weighting on dirichlet boundary
                - BCz1      [default 'Dz+I'] See num_mhs documentation for details
                - BCx0      [default 'Dx']   See num_mhs documentation for details
                - BCx1      [default 'Dx']   See num_mhs documentation for details
                - BCy0      [default 'Dy']   See num_mhs documentation for details
                - BCy1      [default 'Dy']   See num_mhs documentation for details
    OUTPUTS:
        r = nx3 residual vector
    """
    config.setdefault('gamma', 1e-4)
    config.setdefault('g',1)
    config.setdefault('BCz1','Dz+I')
    config.setdefault('BCx0','Dx')
    config.setdefault('BCx1','Dx')
    config.setdefault('BCy0','Dy')
    config.setdefault('BCy1','Dy')
    config.setdefault('weight_z0',1)
    
    n = B.shape[0]
    
    zeromat = sparse.csr_matrix((n,n))
    I = sparse.eye(n, format='csr')
    
    OpDict = {'Dx':locals()['Dx'],'Dy':locals()['Dy'],'Dz':locals()['Dx'],
              'Dxx':locals()['Dxx'],'Dyy':locals()['Dyy'],'Dzz':locals()['Dzz'],
              'Dxy':locals()['Dx'].dot(locals()['Dy']),'Dxz':locals()['Dx'].dot(locals()['Dz']),
              'Dyz':locals()['Dy'].dot(locals()['Dz']),'I':locals()['I']
             }
    
    curler = sparse.hstack((sparse.vstack((zeromat,-Dz,Dy)),
                            sparse.vstack((Dz,zeromat,-Dx)),
                            sparse.vstack((-Dy,Dx,zeromat))))
    R1 = curler.dot(B.ravel('F'))/(4*np.pi)
    R1 = R1.reshape((n,3),order='F')
    forcemat = sparse.hstack((sparse.vstack((zeromat,-sparse.diags(R1[:,2]),sparse.diags(R1[:,1]))),
                              sparse.vstack((sparse.diags(R1[:,2]),zeromat,-sparse.diags(R1[:,0]))),
                              sparse.vstack((-sparse.diags(R1[:,1]),sparse.diags(R1[:,0]),zeromat))))
    lap = Dxx+Dyy+Dzz
    hyperv = sparse.hstack((sparse.vstack((lap,zeromat,zeromat)),
                            sparse.vstack((zeromat,lap,zeromat)),
                            sparse.vstack((zeromat,zeromat,lap))))
    r = forcemat.dot(B.ravel('F')) - config['gamma']*hyperv.dot(B.ravel('F'))
    
    if pres.shape[1] == 3:
        r = r - np.vstack((pres[:,0],
                          pres[:,1],
                          pres[:,2]+config['g']*dens)).ravel()
    else:
        r = r - np.vstack((Dx.dot(pres),
                           Dy.dot(pres),
                           Dz.dot(pres)+config['g']*dens)).ravel()
    
    r = r.reshape((n,3),order='F')
    
    r[index.x0,0] = eval(config['BCx0'],OpDict)[index.x0,:].dot(B[:,0])
    r[index.x1,0] = eval(config['BCx1'],OpDict)[index.x1,:].dot(B[:,0])
    r[index.y0,1] = eval(config['BCy0'],OpDict)[index.y0,:].dot(B[:,1])
    r[index.y1,1] = eval(config['BCy1'],OpDict)[index.y1,:].dot(B[:,1])
    r[index.z1,2] = eval(config['BCz1'],OpDict)[index.z1,:].dot(B[:,2])
    r[index.z0,:] = config['weight_z0']*(B[index.z0,:] - np.hstack((Bx0,By0,Bz0)))
    
    return r

def resBJ(B, Dx, Dy, Dz, Dxx, Dyy, Dzz, index, config={}):
    """
    Provides the mhs jacobian for a given magnetic field and plasma, that is, the
    derivative of resB(B,...) with respect to each component of B.
    INPUTS:
        B       = nx3 magnetic field vector
        Dx..Dzz = Differentiation matrices for the node layout (eg, with rbf-fd)
        index   = mhs index object
        config  = Dictionary of configuration values. Used keys are:
                - gamma     [default 1e-4]   Hyperviscocity weight
                - weight_z0 [default 1]      Weighting on dirichlet boundary
                - BCz1      [default 'Dz+I'] See num_mhs documentation for details
                - BCx0      [default 'Dx']   See num_mhs documentation for details
                - BCx1      [default 'Dx']   See num_mhs documentation for details
                - BCy0      [default 'Dy']   See num_mhs documentation for details
                - BCy1      [default 'Dy']   See num_mhs documentation for details
    OUTPUTS:
        J = 3nx3n Jacobian matrix (unwraps r in Fortran order, to produce a long vector
            ordered like [[r_x],[r_y],[r_z]])
    """
    config.setdefault('gamma', 1e-4)
    config.setdefault('BCz1','Dz+I')
    config.setdefault('BCx0','Dx')
    config.setdefault('BCx1','Dx')
    config.setdefault('BCy0','Dy')
    config.setdefault('BCy1','Dy')
    config.setdefault('weight_z0',1)
    
    n = B.shape[0]
    zeromat = sparse.csr_matrix((n,n))
    I = sparse.eye(n, format='csr')
    
    OpDict = {'Dx':locals()['Dx'],'Dy':locals()['Dy'],'Dz':locals()['Dx'],
              'Dxx':locals()['Dxx'],'Dyy':locals()['Dyy'],'Dzz':locals()['Dzz'],
              'Dxy':locals()['Dx'].dot(locals()['Dy']),'Dxz':locals()['Dx'].dot(locals()['Dz']),
              'Dyz':locals()['Dy'].dot(locals()['Dz']),'I':locals()['I']
             }
    
    current = sparse.bmat([[None,-Dz,Dy],[Dz,None,-Dx],[-Dy,Dx,None]],format='csr').dot(B.ravel('F'))
    current = current.reshape((n,3),order='F')
    
    lap = Dxx+Dyy+Dzz
    
    J = sparse.bmat([[Dy.multiply(B[:,1]) + Dz.multiply(B[:,2]),
                      sparse.diags(-current[:,2]) - Dx.multiply(B[:,1]),
                      sparse.diags( current[:,1]) - Dx.multiply(B[:,2])],
                     [sparse.diags( current[:,2]) - Dy.multiply(B[:,0]),
                      Dx.multiply(B[:,0]) + Dz.multiply(B[:,2]),
                      sparse.diags(-current[:,0]) - Dx.multiply(B[:,2])],
                     [sparse.diags(-current[:,1]) - Dz.multiply(B[:,0]),
                      sparse.diags( current[:,0]) - Dz.multiply(B[:,1]),
                      Dx.multiply(B[:,0]) + Dy.multiply(B[:,1])]
                    ],format='csr')
    
    J = J/(4*np.pi) - config['gamma'] * sparse.hstack((sparse.vstack((lap,zeromat,zeromat)),
                                                       sparse.vstack((zeromat,lap,zeromat)),
                                                       sparse.vstack((zeromat,zeromat,lap))))
    
    zero_rows(J,index.x0)
    J[index.x0,0:n] = eval(config['BCx0'],OpDict)[index.x0,:]
    zero_rows(J,index.x1)
    J[index.x1,0:n] = eval(config['BCx1'],OpDict)[index.x1,:]
    zero_rows(J,n+index.y0)
    J[n+index.y0,0:n] = eval(config['BCy0'],OpDict)[index.y0,:]
    zero_rows(J,n+index.y1)
    J[n+index.y1,0:n] = eval(config['BCy1'],OpDict)[index.y1,:]
    zero_rows(J,2*n+index.z1)
    J[2*n+index.z1,0:n] = eval(config['BCz1'],OpDict)[index.z1,:]
    zero_rows(J,index.z0)
    J[np.ix_(index.z0,index.z0)] = config['weight_z0']*sparse.eye(len(index.z0))
    zero_rows(J,n+index.z0)
    J[np.ix_(n+index.z0,n+index.z0)] = config['weight_z0']*sparse.eye(len(index.z0))
    zero_rows(J,2*n+index.z0)
    J[np.ix_(2*n+index.z0,2*n+index.z0)] = config['weight_z0']*sparse.eye(len(index.z0))
    
    return J
    
    
def num_mhs(dens,pres, Bx0,By0,Bz0, nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, index, config={}, Binit=None):
    """
    Provides the mhs jacobian for a given magnetic field and plasma, that is, the
    derivative of resB(B,...) with respect to each component of B.
    INPUTS:
        dens    = length n density
        pres    = length n pressure OR nx3 pressure gradient
        Bx0     = dirichlet photospheric boundary value on Bx
        By0     = dirichlet photospheric boundary value on By
        Bz0     = dirichlet photospheric boundary value on Bz
        nodes   = nx3 numpy array of node locations
        Dx..Dzz = Differentiation matrices for the node layout (eg, with rbf-fd)
        index   = mhs index object
        config  = Dictionary of configuration values. Possible keys are:
                - gpu          [default False] Boolean to use CUDA for matrix solution
                - gamma        [default 1e-4]  Hyperviscocity weight
                - g            [default 1]     Gravitational constant
                - residual_tol [default 1e-4]  Stopping criterion on L2 norm of mhs residual
                - relative_tol [default 5e-3]  Stopping criterion on ratio of current mhs
                                               residual to previous. Proxy for stagnation.
                - maxiters     [default 10]    Stopping criterion on number of iterations of
                                               Quasi-Newton descent
                - lseps        [default 1]     Weighting for divergence regularization in lsqr
                - lsqr_tol     [default 1e-6]  tolerance for matrix solution
                - lsqr_lim     [default 1e4]   maximum number of iterations for matrix solution
                - weight_z0    [default 1]     Weighting on dirichlet boundary
                - verbose      [default False] Whether to produce printed output on progress
                Additionally, the config dictionary can specify the boundary conditions
                for the system as a string representation of operator algebra, to be applied
                to the component of magnetic field perpendicular to that boundary. Provided
                below are the defaults, as well as recommendations of other common values.
                - BCz1 : 'Dz+I' (radiative exponential decay). Consider also:
                       - constant DzBz ('Dzz'),
                       - closed top ('I')
                - BCx0 : 'Dx' (zero gradient)
                       - 'I' (zero flux)
                       - 'Dxx-Dx' (open)
                - BCx1 : 'Dx' (zero gradient)
                       - 'I' (zero flux)
                       - 'Dxx+Dx' (open)
                - BCy0 : 'Dy' (zero gradient)
                       - 'I' (zero flux)
                       - 'Dyy-Dy' (open)
                - BCy1 : 'Dy' (zero gradient)
                       - 'I' (zero flux)
                       - 'Dyy+Dy' (open)
    OUTPUTS:
        B = nx3 magnetic field vector corresponding to the points in nodes
        r = list of the L2 norms of residuals computed for each step in the descent
    """
    config.setdefault('gamma', 1e-4)
    config.setdefault('g',1)
    config.setdefault('residual_tol', 1e-4)
    config.setdefault('maxiters',10)
    config.setdefault('relative_tol',5e-3)
    config.setdefault('lseps',1)
    config.setdefault('BCz1','Dz+I')
    config.setdefault('BCx0','Dx')
    config.setdefault('BCx1','Dx')
    config.setdefault('BCy0','Dy')
    config.setdefault('BCy1','Dy')
    config.setdefault('weight_z0',1)
    config.setdefault('gpu',False)
    config.setdefault('lsqr_tol',1e-6)
    config.setdefault('lsqr_lim',1e4)
    config.setdefault('verbose',False)
    
    n = nodes.shape[0]
    
    if Binit is None:
        B = potfield(nodes,index,Bz0,Dx,Dy,Dz,Dxx,Dyy,Dzz)
    else:
        B = Binit
    
    def res_fn(Bn):
        return resB(Bn, dens, pres, Bx0, By0, Bz0, Dx, Dy, Dz, Dxx, Dyy, Dzz, index, config=config).ravel('F')
    def res_fn_norm(Bn):
        return np.sum(res_fn(Bn)**2)**0.5
    
    rs = [res_fn_norm(B)]
    
    iters = 0
    if config['verbose']:
        print(f"Starting iterations -- initial residual {res_fn_norm(B):e}")
        tic = time()
    while rs[-1] > config['residual_tol'] and iters < config['maxiters'] and (len(rs) <= 1 or abs(rs[-2]-rs[-1])/rs[-1] > config['relative_tol']):
        iters += 1
        
        r = res_fn(B).ravel('F')
        print("building J")
        J = resBJ(B,Dx,Dy,Dz,Dxx,Dyy,Dzz,index,config=config)
        print("cuthill...")
        divmat = sparse.hstack((Dx,Dy,Dz))
        cuthill = sparse.csgraph.reverse_cuthill_mckee(J)
        rOrdered = np.expand_dims(r[cuthill],1)
        J = sparse.vstack((J[np.ix_(cuthill,cuthill)], 
                           config['lseps']*divmat[:,cuthill]
                         ))
        rOrdered = np.vstack((rOrdered, np.zeros((n,1))))
        print("J built. Starting lsqr")
        if config['gpu']:
            from cupyx.scipy import sparse as cusp
            import cupy
            Jgpu = cusp.csr_matrix(J)
            rgpu = cupy.asarray(rOrdered)
            update, flag, lsiter, resvec = cusp.linalg.lsqr(Jgpu, rgpu, atol=config['lsqr_tol'],btol=config['lsqr_tol'],iter_lim=config['lsqr_lim'])[0:4]
        else:
            update, flag, lsiter, resvec = sparse.linalg.lsqr(J, rOrdered, atol=config['lsqr_tol'],btol=config['lsqr_tol'],iter_lim=config['lsqr_lim'])[0:4]
        print("lsqr done")
        s = np.zeros(cuthill.shape,dtype='int')
        s[cuthill] = np.arange(0,3*n,dtype='int')
        update = update[s].reshape((n,3),order='F')
        
        beta = 1.
        while res_fn_norm(B) < res_fn_norm(B-beta*update):
            beta = 0.5*beta
            if abs(beta) < 1e-6:
                beta = 0
        B = B - beta*update
        
        I = sparse.eye(n, format='csr')
        
        div = divmat.dot(B.ravel('F'))
        div[index.z0] = 0
        div[index.z1] = 0
        
#         lap = (Dxx+Dyy+Dzz)
#         lap[index.z0,:] = Dz[index.z0,:]
#         lap[index.z1,:] = I[index.z1,:]
        
#         if config['gpu']:
#             phiup, pflag, plsiters, presvec = cupyx.scipy.sparse.linalg.lsqr(cupyx.scipy.sparse.csr_matrix(lap),cupy.ndarray(div),atol=config['lsqr_tol'],btol=config['lsqr_tol'],iter_lim=config['lsqr_lim'])[0:4]
#         else:
#             phiup, pflag, plsiters, presvec = sparse.linalg.lsqr(lap,div,atol=config['lsqr_tol'],btol=config['lsqr_tol'],iter_lim=config['lsqr_lim'])[0:4]
        
#         phiup = np.expand_dims(phiup,1)
#         B = B - np.hstack((Dx.dot(phiup),Dy.dot(phiup),Dz.dot(phiup)))
        
        
        if config['verbose']:
            toc = time()
            print(f'step {iters}, beta = {beta}, res = {res_fn_norm(B):e}, {(toc-tic):.2f} seconds.')
            tic = toc
        
        rs.append(res_fn_norm(B))
        
        
    return (B, rs)

