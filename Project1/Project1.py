#=============================================================================#
# README
#=============================================================================#

"""
Starting at line 825, there are several commented function call. To run the
simulation and recreate figures in the report, just uncomment the 
corresponding function and run the code.
"""

#=============================================================================#
# Import library
#=============================================================================#

import scipy.sparse as sp
import scipy.sparse.linalg as sppla
import numpy as np
import matplotlib.pyplot as plt
import math as math
import time

from IPython import get_ipython;   
get_ipython().magic('reset -sf')

#=============================================================================#
# Function definition
#=============================================================================#

# Central differnce operator for second order derivative in 1D
def Central1D(n,dx):
    cd_coef = 1/(dx**2)
    Lh = sp.lil_matrix((n,n))
    for i in range(1,n-1):
        Lh[i,i] = -2*cd_coef
        Lh[i,i+1] = cd_coef
        Lh[i,i-1] = cd_coef
    Lh[0,0] = -2*cd_coef
    Lh[0,1] = cd_coef
    Lh[-1,-1] = -2*cd_coef
    Lh[-1,-2] = cd_coef
    return Lh

# Convergence test for 1D problem
def Conv1Dss():
    size_x = 1
    conv_grid = np.array([5,11,101,1001,2001])
    err = np.zeros(conv_grid.size)
    hx = np.zeros(conv_grid.size)
    for k, n in enumerate(conv_grid):
        x = np.linspace(0,size_x,n)
        dx = size_x/(n-1)
        # Central differnece operator
        Lh = Central1D(n,dx)
        # heat source
        r = 240*(x**2)
        # left boundary, Neumann
        cd_coef = 1/(dx**2)
        Nbc = 0
        Lh[Nbc,:] = 0
        Lh[Nbc,Nbc] = -2*cd_coef
        Lh[Nbc,Nbc+1] = 2*cd_coef
        # Right boundary, Dirichlet
        Lh[-1,:] = 0
        Lh[-1,-1] = 1
        r[-1] = 0
        u = sppla.spsolve(-Lh, r)
        # True solution
        u_true = 20*(1-x**4)
        # Norm of error
        err[k] = np.linalg.norm(u-u_true,np.inf)
        hx[k] = dx
    # First and second order function for comparison
    t = np.linspace(0.00034691,0.3,2)
    f1 = t*900*0.1
    f2 = t**2*80*0.01
    # Plot
    ft = 15
    ft2 = 12
    ss1D = plt.figure()
    ss1D.set_size_inches(12,4)
    # Plot solution
    plt.subplot(1, 2, 1)
    plt.plot(x,u,x,u_true,'-.')
    plt.legend(('Finite Diff.','True Sol.'), fontsize=ft2)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$u$', fontsize=ft)
    # Plot error
    plt.subplot(1, 2, 2)
    plt.loglog(hx,err, t,f1,'--',t,f2,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error', fontsize=ft)
    plt.legend(('Error','$O(h)$','$O(h^2)$'), fontsize=ft2)
    ss1D.savefig('ss1D.png',dpi=300)

def Central2D_BC(n,h,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC):
    temp = Central1D(n,h)
    Imat = sp.identity(n)
    Lh = sp.kron(Imat,temp) + sp.kron(temp,Imat)
    Lh = Lh.tolil()
    # Modify finite difference operator to include Neumann BC
    for i in range(0,n*n):
        if NBC_xl[i]:
            if (i!=0):
                Lh[i,i-1] = 0
            Lh[i,i+1] = 2*Lh[i,i+1]
        if NBC_xr[i]:
            if ((i+1)%n!=0):
                Lh[i,i+1] = 0
            Lh[i,i-1] = 2*Lh[i,i-1]
        if NBC_yb[i]:
            if ((i-n)>-1):
                Lh[i,i-n] = 0
            Lh[i,i+n] = 2*Lh[i,i+n]
        if NBC_yt[i]:
            if ((i+n)<n**2):
                Lh[i,i+n] = 0
            Lh[i,i-n] = 2*Lh[i,i-n]
    # Modify finite difference operator to include Dirichlet BC
    for i in range(0,n**2):
        if DBC[i]:
            Lh[i,:] = 0
            Lh[i,i] = 1
    return Lh

def Conv2Dss():
    size_xy = 1
    conv_grid = np.array([5,11,101])
    err = np.zeros(conv_grid.size)
    hv = np.zeros(conv_grid.size)
    for k, n in enumerate(conv_grid):
        # Mesh setup
        x = np.linspace(0,size_xy,n)
        y = x
        h = size_xy/(n-1)
        # Define Neumann boundary conditions
        NBC_xl = np.kron( (y==y), (x==-1) )
        NBC_xr = np.kron( (y==y), (x==size_xy) )
        NBC_yb = np.kron( (y==-1), (x==x) )
        NBC_yt = np.kron( (y==size_xy), (x==x) )
        # Define Dirichlet boundary conditions
        DBC = np.kron( (y==0), (x==x) ) | np.kron( (y==y), (x==0) )
        # Heat source
        r = 240*( np.kron( (4*y-y**4), (x**2) ) + np.kron( (y**2), (4*x-x**4) ) )
        # Modify r to include Dirichlet BC
        r = r*(1-DBC)
        # Operator for second derivative in x and y
        Lh = Central2D_BC(n,h,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC)
        # Solve u
        u = sppla.spsolve(-Lh, r)
        # True solution
        u_true = 20*( np.kron( (4*y-y**4), (4*x-x**4) )  )
        # Norm of error
        err[k] = np.linalg.norm(u-u_true,np.inf)
        hv[k] = h
    # First and second order function for comparison
    t = np.linspace(0.0104691,0.3,2)
    f1 = t*2000*0.1
    f2 = t**2*1000*0.01
    # Plot 
    xlist, ylist = np.meshgrid(x, y)
    z = np.reshape(u,(n,n))
    # z_true = np.reshape(u_true,(nx,ny))
    ss2D = plt.figure()
    ss2D.set_size_inches(12,4)
    # Plot solution
    ft = 15
    ft2 = 12
    plt.subplot(1, 2, 1)
    cp1=plt.contourf(xlist,ylist,z,30)
    ss2D.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    # Plot error
    plt.subplot(1, 2, 2)
    plt.loglog(hv,err, t,f1,'--',t,f2,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error', fontsize=ft)
    plt.legend(('Error','$O(h)$','$O(h^2)$'), fontsize=ft2)
    ss2D.savefig('ss2D.png',dpi=300)

def Conv2Dtrans():
    size_xy = 1
    # conv_grid = np.array([5,11,101])
    conv_grid = np.array([11,51,101])
    err = np.zeros(conv_grid.size)
    hv = np.zeros(conv_grid.size)
    for k, n in enumerate(conv_grid):
            # Mesh setup
            x = np.linspace(0,size_xy,n)
            y = x
            h = size_xy/(n-1)
            u = np.zeros(x.size**2)
            # Define Neumann boundary conditions
            NBC_xl = np.kron( (y==y), (x==-1) )
            NBC_xr = np.kron( (y==y), (x==size_xy) )
            NBC_yb = np.kron( (y==-1), (x==x) )
            NBC_yt = np.kron( (y==size_xy), (x==x) )
            # Define Dirichlet boundary conditions
            DBC = np.kron( (y==0), (x==x) ) | np.kron( (y==y), (x==0) )
            # Operator for second derivative in x and y and modified heat source
            Lh = Central2D_BC(n,h,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC)
            Lh = sp.csr_matrix(Lh)
            # Second order Runge-Kutta
            t = 0
            ht = (h**2)/4
            t_end = 0.5
            i = 0
            # Second Order Runge-Kutta
            while t < t_end:
                i = i+1
                # First stage
                r = 240*(1-np.exp(-t))*( np.kron( (4*y-y**4), (x**2) ) + np.kron( (y**2), (4*x-x**4) ) ) \
                    + 20*np.exp(-t)*( np.kron( (4*y-y**4), (4*x-x**4) ) )
                r = r*(1-DBC)
                k1 = ht*(Lh.dot(u)+r)
                # Second stage
                t = t+ht
                r = 240*(1-np.exp(-t))*( np.kron( (4*y-y**4), (x**2) ) + np.kron( (y**2), (4*x-x**4) ) ) \
                    + 20*np.exp(-t)*( np.kron( (4*y-y**4), (4*x-x**4) ) )
                r = r*(1-DBC)
                k2 = ht*(Lh.dot(u+k1)+r)
                # Final solution
                u = u+(k1+k2)/2     
            # True solution
            u_true = 20*(1-np.exp(-t))*( np.kron( (4*y-y**4), (4*x-x**4) )  )
            # Norm of error
            err[k] = np.linalg.norm(u-u_true,np.inf)
            hv[k] = h
    # Plot 
    t = np.linspace(0.0104691,0.1,2)
    f1 = t*800*0.1
    f2 = t**2*800*0.01
    xlist, ylist = np.meshgrid(x, y)
    z = np.reshape(u,(n,n))
    trans2D = plt.figure()
    trans2D.set_size_inches(12,4)
    # Plot solution
    ft = 15
    ft2 = 12
    plt.subplot(1, 2, 1)
    cp1=plt.contourf(xlist,ylist,z,30)
    trans2D.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    # Plot error
    plt.subplot(1, 2, 2)
    plt.loglog(hv,err, t,f1,'--',t,f2,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error', fontsize=ft)
    plt.legend(('Error','$O(h)$','$O(h^2)$'), fontsize=ft2)
    trans2D.savefig('trans2D.png',dpi=300)

def geomRoom():
    ft = 15
    room = plt.figure()
    room.set_size_inches(4,4)
    plt.plot([0,1.5],[3.6,3.6],'k',[0,1.5],[3.4,3.4],'k',[1.5,1.5],[3.4,3.6],'k')
    plt.plot([3.0,5.0],[3.6,3.6],'k',[3.0,5.0],[3.4,3.4],'k',[3.0,3.0],[3.4,3.6],'k')
    plt.plot([0.0,2.5],[2.1,2.1],'k',[0.0,2.5],[1.9,1.9],'k',[2.5,2.5],[1.9,2.1],'k')
    plt.plot([0.1,0.1],[0.6,1.4],'b',[0.1,0.1],[2.5,3.0],'b',[0.1,0.1],[4.0,4.5],'b')
    plt.plot([0.0,0.1],[0.6,0.6],'b',[0.0,0.1],[1.4,1.4],'b')
    plt.plot([0.0,0.1],[2.5,2.5],'b',[0.0,0.1],[3.0,3.0],'b')
    plt.plot([0.0,0.1],[4.0,4.0],'b',[0.0,0.1],[4.5,4.5],'b')
    plt.plot([4.9,4.9],[1.0,2.0],'b',[4.9,5.0],[1.0,1.0],'b',[4.9,5.0],[2.0,2.0],'b')
    plt.plot([4.0,4.5],[4.9,4.9],'b',[4.0,4.0],[4.9,5.0],'b',[4.5,4.5],[4.9,5.0],'b')
    plt.plot([1.4,1.8],[1.4,1.4],'r',[1.4,1.8],[1.8,1.8],'r',[1.4,1.4],[1.4,1.8],'r',[1.8,1.8],[1.4,1.8],'r')
    plt.plot([1.0,1.4],[3.7,3.7],'r',[1.0,1.4],[4.1,4.1],'r',[1.0,1.0],[3.7,4.1],'r',[1.4,1.4],[3.7,4.1],'r')
    plt.plot([3.2,3.6],[3.7,3.7],'r',[3.2,3.6],[4.1,4.1],'r',[3.2,3.2],[3.7,4.1],'r',[3.6,3.6],[3.7,4.1],'r')
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    plt.xlim((0,5))
    plt.ylim((0,5))
    room.savefig('room',dpi=300)

def Room_BC_heatsource(x,y):
    size_xy = 5
    # Define Neumann boundary conditions
    NBC_xl = np.kron( (y==y), (x==0) ) | np.kron( (y>=3.39999)&(y<=3.600001), (x>1.49999)&(x<=1.500001) ) \
                                       | np.kron( (y>=1.89999)&(y<=2.100001), (x>2.49999)&(x<=2.500001) )  
    # print (np.sum(NBC_xl))
    NBC_xr = np.kron( (y==y), (x==size_xy) ) | np.kron( (y>=3.39999)&(y<=3.600001),(x>3.99999)&(x<=3.000001) )
    NBC_yb = np.kron( (y==0), (x==x) )  | np.kron( (y>3.59999)&(y<=3.600001), (x<=1.50001)|(x>=2.99999) ) \
                                        | np.kron( (y>2.09999)&(y<=2.100001), (x<=2.50001) ) 
    NBC_yt = np.kron( (y==size_xy), (x==x) )  | np.kron( (y>3.39999)&(y<=3.400001), (x<=1.50001)|(x>=2.99999) ) \
                                              | np.kron( (y>1.80999)&(y<=1.900001), (x<=2.50001) )    
    # Define Dirichlet boundary conditions
    DBC = np.kron( (y>=0.6)&(y<=1.4), (x==0) ) | np.kron( (y>=1.0)&(y<=2.0), (x==5.0) ) \
                                               | np.kron( (y>=4.0)&(y<=4.5), (x==0) ) \
                                               | np.kron( (y>=2.5)&(y<=3.0), (x==0) ) \
                                               | np.kron( (y==5.0), (x>=4.0)&(x<=4.5 ) )
    # Heat source
    mag = 120
    r =  mag*np.kron( (y>=3.7)&(y<=4.1), (x>=3.2)&(x<=3.6) ) \
        |mag*np.kron( (y>=1.4)&(y<=1.8), (x>=1.4)&(x<=1.8) ) \
        |mag*np.kron( (y>=3.7)&(y<=4.1), (x>=1.0)&(x<=1.4) )
    # Modify r to include Dirichlet BC
    r = r*(1-DBC)
    return NBC_xl, NBC_xr, NBC_yb, NBC_yt, DBC, r
        
def setupRoom(n=201):
    size_xy = 5
    # Mesh setup
    x = np.linspace(0,size_xy,n)
    y = x
    h = size_xy/(n-1)
    NBC_xl, NBC_xr, NBC_yb, NBC_yt, DBC, r = Room_BC_heatsource(x,y)
    # Operator for second derivative in x and y
    Lh = Central2D_BC(n,h,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC)
    Lh = sp.csr_matrix(Lh)
    return n, h, x, y, Lh, r

def plotRoom(x,y,n,u,name):
    # Plot 
    xlist, ylist = np.meshgrid(x, y)
    z = np.reshape(u,(n,n))
    ft = 15
    ssRoom = plt.figure()
    ssRoom.set_size_inches(12,4)
    # Plot solution
    plt.subplot(1, 2, 1)
    cp1=plt.contourf(xlist,ylist,z,25)
    ssRoom.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    # Draw the wall, window, and heater
    plt.subplot(1, 2, 2)
    cp1=plt.contourf(xlist,ylist,z,25)
    ssRoom.colorbar(cp1)
    plt.plot([0,1.5],[3.6,3.6],'k',[0,1.5],[3.4,3.4],'k',[1.5,1.5],[3.4,3.6],'k')
    plt.plot([3.0,5.0],[3.6,3.6],'k',[3.0,5.0],[3.4,3.4],'k',[3.0,3.0],[3.4,3.6],'k')
    plt.plot([0.0,2.5],[2.1,2.1],'k',[0.0,2.5],[1.9,1.9],'k',[2.5,2.5],[1.9,2.1],'k')
    plt.plot([0.1,0.1],[0.6,1.4],'b',[0.1,0.1],[2.5,3.0],'b',[0.1,0.1],[4.0,4.5],'b')
    plt.plot([0.0,0.1],[0.6,0.6],'b',[0.0,0.1],[1.4,1.4],'b')
    plt.plot([0.0,0.1],[2.5,2.5],'b',[0.0,0.1],[3.0,3.0],'b')
    plt.plot([0.0,0.1],[4.0,4.0],'b',[0.0,0.1],[4.5,4.5],'b')
    plt.plot([4.9,4.9],[1.0,2.0],'b',[4.9,5.0],[1.0,1.0],'b',[4.9,5.0],[2.0,2.0],'b')
    plt.plot([4.0,4.5],[4.9,4.9],'b',[4.0,4.0],[4.9,5.0],'b',[4.5,4.5],[4.9,5.0],'b')
    plt.plot([1.4,1.8],[1.4,1.4],'r',[1.4,1.8],[1.8,1.8],'r',[1.4,1.4],[1.4,1.8],'r',[1.8,1.8],[1.4,1.8],'r')
    plt.plot([1.0,1.4],[3.7,3.7],'r',[1.0,1.4],[4.1,4.1],'r',[1.0,1.0],[3.7,4.1],'r',[1.4,1.4],[3.7,4.1],'r')
    plt.plot([3.2,3.6],[3.7,3.7],'r',[3.2,3.6],[4.1,4.1],'r',[3.2,3.2],[3.7,4.1],'r',[3.6,3.6],[3.7,4.1],'r')
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    plt.xlim((0,5))
    plt.ylim((0,5))
    ssRoom.savefig(name,dpi=300)

def Room_trans_ex():
    n, h, x, y, Lh, r = setupRoom()
    u = np.zeros(x.size**2)
    # Second order Runge-Kutta
    t = 0
    ht = (h**2)/4
    # print(ht)
    t_end = 32
    # Save at this step
    # saveu = np.array([100,3000,6000,10000,30000,60000,math.floor(t_end/ht)-1])
    saveu = np.array([100,6000,30000,math.floor(t_end/ht)-1])
    # Second Order Runge-Kutta
    i = 0
    while t < t_end:
        i = i+1
        # First stage
        k1 = ht*(Lh.dot(u)+r)
        # Second stage
        t = t+ht
        k2 = ht*(Lh.dot(u+k1)+r)
        # Final solution
        u = u+(k1+k2)/2
        if i in saveu:
            times = str(t)
            name = 'transRoom_ex_'+times[0:8]+'.png'
            # Plot u
            plotRoom(x,y,n,u,name)

def Room_trans_im():
    n, h, x, y, Lh, r = setupRoom()
    u = np.zeros(x.size**2)
    # Diagonally implicit Runge-Kutta
    t = 0
    t_end = 32
    mult = 100
    ht = (h**2)/4*mult
    # Factorized implicit operator
    Lhk = sp.identity(n**2)-1/4*ht*Lh
    solve_k = sp.linalg.factorized(Lhk.tocsc())
    # Save at this step
    # saveu = np.array([100,3000,6000,10000,30000,60000,math.floor(t_end/ht)-1])
    saveu = np.array([100/mult,6000/mult,30000/mult,math.floor(t_end/ht)-1])
    # Second Order Runge-Kutta
    i = 0
    while t < t_end:
        i = i+1
        t = t+ht
        rhs1 = Lh.dot(u)+r
        # First stage
        k1 = solve_k(rhs1)
        # Second stage
        k2 = solve_k(rhs1+ht/2*Lh.dot(k1))
        # Final solution
        u = u+(k1+k2)*ht/2
        if i in saveu:
            times = str(t)
            name = 'transRoom_im_'+times[0:8]+'.png'
            # Plot u
            plotRoom(x,y,n,u,name)

def Room_ss():
    # Setup geometry, heat source, and boundary conditions
    n, h, x, y, Lh, r = setupRoom()
    # Solve u
    u = sppla.spsolve(-Lh, r)
    # Plot u
    plotRoom(x,y,n,u,'ssRoom.png')

def Room_jacobi(nx=101, maxiter=300):
    n, h, x, y, Lh, r = setupRoom(n=nx)
    u = np.random.rand(n**2)
    err = np.array([])
    diag = np.diag(Lh.toarray())
    C = -r/diag
    diag = sp.csr_matrix(np.diag(1/diag))
    T = diag.dot(Lh)
    T = T.tolil()
    for i in range(u.size):
        T[i,i] = 0
    T = -sp.csr_matrix(T)
    for i in range(maxiter):
        u = T.dot(u) + C
        error = np.linalg.norm(-Lh.dot(u)-r,np.infty)
        err = np.append(err,error)
        if error < 1e-8:
            print('Jacobi converged at iteration ',i)
            break
    # Plot setup
    perr = Lh.dot(u)+r
    xlist, ylist = np.meshgrid(x, y)
    rperr = np.reshape(perr,(n,n))
    z = np.reshape(u,(n,n))
    ft = 15
    # Plot residual    
    ResPlot = plt.figure()
    ResPlot.set_size_inches(12,4)
    plt.subplot(1, 2, 1)
    cp2=plt.contourf(xlist,ylist,rperr,25)
    ResPlot.colorbar(cp2)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    plt.subplot(1, 2, 2)
    plt.plot(err)
    plt.xlabel('iteration', fontsize=ft)
    plt.ylabel('residual', fontsize=ft)
    ResPlot.savefig('plot_res_jacobi',dpi=300)
    # Plot solution
    Jplot = plt.figure()
    cp1=plt.contourf(xlist,ylist,z,25)
    Jplot.set_size_inches(6,4)
    Jplot.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    Jplot.savefig('plot_sol_jacobi',dpi=300)

def Room_damped_jacobi(alpha=0.05, nx=101, maxiter=300):
    n, h, x, y, Lh, r = setupRoom(n=nx)
    u = np.random.rand(n**2)
    err = np.array([])
    diag = np.diag(Lh.toarray())
    C = -r/diag
    diag = sp.csr_matrix(np.diag(1/diag))
    T = diag.dot(Lh)
    T = T.tolil()
    del diag
    for i in range(u.size):
        T[i,i] = 0
    T = -sp.csr_matrix(T)
    for i in range(maxiter):
        u = alpha*u + (1-alpha)*(T.dot(u) + C)
        error = np.linalg.norm(-Lh.dot(u)-r,np.infty)
        err = np.append(err,error)
        if error < 1e-8:
            print('Jacobi converged at iteration ',i)
            break
    # Plot setup
    perr = Lh.dot(u)+r
    xlist, ylist = np.meshgrid(x, y)
    rperr = np.reshape(perr,(n,n))
    z = np.reshape(u,(n,n))
    ft = 15
    # Plot residual    
    ResPlot = plt.figure()
    ResPlot.set_size_inches(12,4)
    plt.subplot(1, 2, 1)
    cp2=plt.contourf(xlist,ylist,rperr,25)
    ResPlot.colorbar(cp2)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    plt.subplot(1, 2, 2)
    plt.plot(err)
    plt.xlabel('iteration', fontsize=ft)
    plt.ylabel('residual', fontsize=ft)
    ResPlot.savefig('plot_res_damped_jacobi',dpi=300)
    # Plot solution
    Jplot = plt.figure()
    cp1=plt.contourf(xlist,ylist,z,25)
    Jplot.set_size_inches(6,4)
    Jplot.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    Jplot.savefig('plot_sol_damped_jacobi',dpi=300)

def restrict(u):
    n = int(np.sqrt(u.size))
    u = np.reshape(u,(n,n))
    u = u[::2,::2]
    n1 = int(np.sqrt(u.size))
    return np.reshape(u,int(n1**2))

def convRestrict():
    size_xy = 1
    conv_grid = np.array([5,11,101])
    err = np.zeros(conv_grid.size)
    hv = np.zeros(conv_grid.size)
    for k, n in enumerate(conv_grid):
        # Mesh setup
        x = np.linspace(0,size_xy,n)
        y = x
        h = size_xy/(n-1)
        # Define Neumann boundary conditions
        NBC_xl = np.kron( (y==y), (x==-1) )
        NBC_xr = np.kron( (y==y), (x==size_xy) )
        NBC_yb = np.kron( (y==-1), (x==x) )
        NBC_yt = np.kron( (y==size_xy), (x==x) )
        # Define Dirichlet boundary conditions
        DBC = np.kron( (y==0), (x==x) ) | np.kron( (y==y), (x==0) )
        # Heat source
        r = 240*( np.kron( (4*y-y**4), (x**2) ) + np.kron( (y**2), (4*x-x**4) ) )
        # Modify r to include Dirichlet BC
        r = r*(1-DBC)
        # Operator for second derivative in x and y
        Lh = Central2D_BC(n,h,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC)
        # Solve u
        u = sppla.spsolve(-Lh, r)
        # Restrict
        u = restrict(u)
        x = x[::2]
        y = y[::2]
        n = x.size
        # True solution
        u_true = 20*( np.kron( (4*y-y**4), (4*x-x**4) )  )
        # Norm of error
        err[k] = np.linalg.norm(u-u_true,np.inf)
        hv[k] = h
    # First and second order function for comparison
    t = np.linspace(0.0104691,0.3,2)
    f1 = t*2000*0.1
    f2 = t**2*1000*0.01
    # Plot 
    xlist, ylist = np.meshgrid(x, y)
    z = np.reshape(u,(n,n))
    # z_true = np.reshape(u_true,(nx,ny))
    ss2D = plt.figure()
    ss2D.set_size_inches(12,4)
    # Plot solution
    ft = 15
    ft2 = 12
    plt.subplot(1, 2, 1)
    cp1=plt.contourf(xlist,ylist,z,30)
    ss2D.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    # Plot error
    plt.subplot(1, 2, 2)
    plt.loglog(hv,err, t,f1,'--',t,f2,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error', fontsize=ft)
    plt.legend(('Error','$O(h)$','$O(h^2)$'), fontsize=ft2)
    ss2D.savefig('plot_conv_restrict.png',dpi=300)

def ConvProlong():
    size_xy = 1
    conv_grid = np.array([3,51,101])
    err = np.zeros(conv_grid.size)
    hv = np.zeros(conv_grid.size)
    for k, n in enumerate(conv_grid):
        # Mesh setup
        x = np.linspace(0,size_xy,n)
        y = x
        h = size_xy/(n-1)
        # Define Neumann boundary conditions
        NBC_xl = np.kron( (y==y), (x==-1) )
        NBC_xr = np.kron( (y==y), (x==size_xy) )
        NBC_yb = np.kron( (y==-1), (x==x) )
        NBC_yt = np.kron( (y==size_xy), (x==x) )
        # Define Dirichlet boundary conditions
        DBC = np.kron( (y==0), (x==x) ) | np.kron( (y==y), (x==0) )
        # Heat source
        r = 240*( np.kron( (4*y-y**4), (x**2) ) + np.kron( (y**2), (4*x-x**4) ) )
        # Modify r to include Dirichlet BC
        r = r*(1-DBC)
        # Operator for second derivative in x and y
        Lh = Central2D_BC(n,h,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC)
        # Solve u
        u = sppla.spsolve(-Lh, r)
        # Prolong
        n2 = int(2*(n-1)+1) 
        temp = np.reshape(u,(n,n))
        u = np.zeros((n2,n2))
        u[::2,::2] = temp
        u = np.reshape(u,n2**2)
        x = np.linspace(0,size_xy,n2)
        y = x
        h = size_xy/(n2-1)
        # For heat source selection
        prolg = (h**2)/4*(u==0)
        # Define Neumann boundary conditions
        NBC_xl = np.kron( (y==y), (x==-1) )
        NBC_xr = np.kron( (y==y), (x==size_xy) )
        NBC_yb = np.kron( (y==-1), (x==x) )
        NBC_yt = np.kron( (y==size_xy), (x==x) )
        # Define Dirichlet boundary conditions
        DBC = np.kron( (y==0), (x==x) ) | np.kron( (y==y), (x==0) )
        # Heat source
        r = 240*( np.kron( (4*y-y**4), (x**2) ) + np.kron( (y**2), (4*x-x**4) ) )
        # Modify r to include Dirichlet BC
        r = r*(1-DBC)
        n2 = int(2*(n-1)+1)    
        Imat = sp.eye(n2)
        m1 = np.eye(n2)
        m1[1::2,1::2]=0
        m1 = sp.kron(Imat,m1)
        Imat = np.zeros((n2,n2))
        m2 = np.zeros((n2,n2))
        for i in range(1,n2,2):
            Imat[i,i+1] = 1
            Imat[i,i-1] = 1
            m2[i,i+1] = 0.25
            m2[i,i-1] = 0.25
        m2 = sp.kron(Imat,m2)
        m1 = m1+m2
        m2 = sp.lil_matrix((n2**2,n2**2))
        for i in range(1,n2**2,2):
            m2[i,i+1] = 0.25
            m2[i,i-1] = 0.25
            if (i-n2>-1):
                m2[i,i-n2] = 0.25
            if (i+n2<n2**2):
                m2[i,i+n2] = 0.25
        for i in range(0,n2**2,2):
            m2[i,i] = 1 
        for i in range(0,n2*n2):
            if NBC_xl[i]:
                m2[i,:] = m2[i+1,:]
            if NBC_xr[i]:
                m2[i,:] = m2[i-1,:]
            if NBC_yb[i]:
                m2[i,:] = m2[i+n2,:]
            if NBC_yt[i]:
                m2[i,:] = m2[i-n2,:]
            if DBC[i]:
                m2[i,:] = 0
        u = m2.dot(m1.dot(u)) + (r*prolg)
        # True solution
        u_true = 20*( np.kron( (4*y-y**4), (4*x-x**4) )  )
        # Norm of error
        err[k] = np.linalg.norm(u-u_true,np.inf)
        hv[k] = h
    # First and second order function for comparison
    t = np.linspace(0.00504691,0.3,2)
    f1 = t*20000*0.1
    f2 = t**2*1000*0.01
    # Plot 
    xlist, ylist = np.meshgrid(x, y)
    z = np.reshape(u,(n2,n2))
    # z_true = np.reshape(u_true,(nx,ny))
    ss2D = plt.figure()
    ss2D.set_size_inches(12,4)
    # Plot solution
    ft = 15
    ft2 = 12
    plt.subplot(1, 2, 1)
    cp1=plt.contourf(xlist,ylist,z,30)
    ss2D.colorbar(cp1)
    plt.xlabel('$x$', fontsize=ft)
    plt.ylabel('$y$', fontsize=ft)
    # Plot error
    plt.subplot(1, 2, 2)
    plt.loglog(hv,err, t,f1,'--',t,f2,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error', fontsize=ft)
    plt.legend(('Error','$O(h)$','$O(h^2)$'), fontsize=ft2)
    ss2D.savefig('plot_conv_prolong.png',dpi=300)

def prolong(n,NBC_xl,NBC_xr,NBC_yb,NBC_yt,DBC):
    size_xy = 5
    # Prolong
    n2 = int(2*(n-1)+1)
    temp = np.ones(n**2)
    temp = np.reshape(temp,(n,n))
    u = np.zeros((n2,n2))
    u[::2,::2] = temp
    u = np.reshape(u,n2**2)
    h = size_xy/(n2-1)
    # For heat source selection
    prolg = (h**2)/4*(u==0)
    # Compute prolongation matrix   
    Imat = sp.eye(n2)
    m1 = np.eye(n2)
    m1[1::2,1::2]=0
    m1 = sp.kron(Imat,m1)
    Imat = np.zeros((n2,n2))
    m2 = np.zeros((n2,n2))
    for i in range(1,n2,2):
        Imat[i,i+1] = 1
        Imat[i,i-1] = 1
        m2[i,i+1] = 0.25
        m2[i,i-1] = 0.25
    m2 = sp.kron(Imat,m2)
    m1 = m1+m2
    m2 = sp.lil_matrix((n2**2,n2**2))
    for i in range(1,n2**2,2):
        m2[i,i+1] = 0.25
        m2[i,i-1] = 0.25
        if (i-n2>-1):
            m2[i,i-n2] = 0.25
        if (i+n2<n2**2):
            m2[i,i+n2] = 0.25
    for i in range(0,n2**2,2):
        m2[i,i] = 1 
    for i in range(0,n2*n2):
        if NBC_xl[i]:
            m2[i,:] = m2[i+1,:]
        if NBC_xr[i]:
            m2[i,:] = m2[i-1,:]
        if NBC_yb[i]:
            m2[i,:] = m2[i+n2,:]
        if NBC_yt[i]:
            m2[i,:] = m2[i-n2,:]
        if DBC[i]:
            m2[i,:] = 0
    m = m2.dot(m1)
    m = m.tocsr()
    return m, prolg

def Multigrid():
    # Setup for first level
    n1, h1, x1, y1, Lh1, r1 = setupRoom(n=201)
    NBC_xl1, NBC_xr1, NBC_yb1, NBC_yt1, DBC1, r1 = Room_BC_heatsource(x1,y1)
    diag1 = np.diag(Lh1.toarray())
    C1 = -r1/diag1
    T1 = sp.csr_matrix(np.diag(1/diag1)).dot(Lh1)
    T1 = T1.tolil()
    for i in range(n1**2):
        T1[i,i] = 0
    T1 = -sp.csr_matrix(T1)
    # Setup for second level
    n2, h2, x2, y2, Lh2, r2 = setupRoom(n=101)
    NBC_xl2, NBC_xr2, NBC_yb2, NBC_yt2, DBC2, r2 = Room_BC_heatsource(x2,y2)
    m2, prolg2 = prolong(n2,NBC_xl1,NBC_xr1,NBC_yb1,NBC_yt1,DBC1)
    diag2 = np.diag(Lh2.toarray())
    C2 = -r2/diag2
    T2 = sp.csr_matrix(np.diag(1/diag2)).dot(Lh2)
    T2 = T2.tolil()
    for i in range(n2**2):
        T2[i,i] = 0
    T2 = -sp.csr_matrix(T2)
    # Setup for third level
    n3, h3, x3, y3, Lh3, r3 = setupRoom(n=51)
    NBC_xl3, NBC_xr3, NBC_yb3, NBC_yt3, DBC3, r3 = Room_BC_heatsource(x3,y3)
    m3, prolg3 = prolong(n3,NBC_xl2,NBC_xr2,NBC_yb2,NBC_yt2,DBC2)
    # Start multigrid method
    alpha = 0.05
    u1 = np.zeros(n1**2)
    err = np.array([])
    # V-Cycle iteration
    for k in range(500):
        # First level
        for i in range(3):
            u1 = alpha*u1 + (1-alpha)*(T1.dot(u1) + C1)
        res1 = -Lh1.dot(u1)-r1
        # Second level
        r2 = restrict(res1)
        C2 = r2/diag2*0
        u2 = np.zeros(n2**2)
        for i in range(3):
            u2 = alpha*u2 + (1-alpha)*(T2.dot(u2) + C2)
        res2 = -Lh2.dot(u2)+r2
        # Third level
        r3 = restrict(res2)
        u3 = sppla.spsolve(Lh3, r3)
        # Prolongate to second level
        u3 = np.reshape(u3,(n3,n3))
        u3b = np.zeros((n2,n2))
        u3b[::2,::2] = u3
        u3b = np.reshape(u3b,n2**2)
        u2 = u2 + m3.dot(u3b) 
        for i in range(3):
            u2 = alpha*u2 + (1-alpha)*(T2.dot(u2) + C2)
        # Prolongate to first level
        u2 = np.reshape(u2,(n2,n2))
        u2b = np.zeros((n1,n1))
        u2b[::2,::2] = u2
        u2b = np.reshape(u2b,n1**2)
        u1 = u1 + m2.dot(u2b) 
        for i in range(3):
            u1 = alpha*u1 + (1-alpha)*(T1.dot(u1) + C1)
        # Compute error
        error = np.linalg.norm(-Lh1.dot(u1)-r1,np.infty)
        err = np.append(err,error)
        if (k==0):
            error0 = error
        if error/error0 < 1e-12:
            print('Multigrid method converged at iteration ',k)
            break
    # Plot setup
    ft = 15
    # Plot residual    
    ResPlot = plt.figure()
    plt.semilogy(err)
    plt.xlabel('number of V-Cycle iteration', fontsize=ft)
    plt.ylabel('residual', fontsize=ft)
    ResPlot.savefig('plot_res_multigrid',dpi=300)
    # Plot solutoin
    plotRoom(x1,y1,n1,u1,'plot_sol_multigrid')

#=============================================================================#
# Task 1
#=============================================================================#

# Start measuring time needed to do simulation
tic = time.perf_counter()
    
# Convergence test for 1D finite difference, steady state
# Conv1Dss()

# Convergence test for 2D finite difference, steady state
# Conv2Dss()

# Convergence test for 2D finite difference, transient
# Conv2Dtrans()

# Plot geometry of the room, including wall, window, and heater
# geomRoom()

# Room, transient, explicit method
# Room_trans_ex()

#=============================================================================#
# Task 2
#=============================================================================#

# Room, transient, implicit method
# Room_trans_im()

# Room, steady state
# Room_ss()

# Room, steady state, Jacobi method
# Room_jacobi(nx=201,maxiter=300)

# Room, steady state, damped Jacobi method
# Room_damped_jacobi(alpha=0.05, nx=201, maxiter=300)
  
#=============================================================================#
# Task 3
#=============================================================================#

# Convergence test for restriction
# convRestrict()

# Convergence test for prolongation
# ConvProlong()

# Room, multigrid method
Multigrid()

###############################################################################

# Print time needed to do simulation
toc = time.perf_counter()
print(f"\nTotal time = {toc - tic:0.4f} seconds")






















