#!/usr/bin/env python3
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

import time

import matplotlib.pyplot as plt
import meshpy.triangle as triangle
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from mpl_toolkits.mplot3d import Axes3D

# from IPython import get_ipython;
# get_ipython().magic('reset -sf')

np.set_printoptions(linewidth=1000)

#=============================================================================#
# Function for meshing
#=============================================================================#

def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

def make_mesh(control_max_area):
    points = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    facets = round_trip_connect(0, len(points)-1)

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0)/3
        a1 = control_max_area
        a2 = control_max_area
        max_area = a1 + la.norm(bary, np.inf)*a2
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    built_mesh = triangle.build(info, refinement_func=needs_refinement)
    return np.array(built_mesh.points), np.array(built_mesh.elements)

def make_mesh_circ(control_max_area,npcirc):
    points = [(0,-1),(0,0),(1,0)]
    facets = round_trip_connect(0, len(points)-1)
    facets = np.array(facets)
    facets = facets[0:-1]

    circ_start = len(points)
    points.extend(
            (np.cos(angle), np.sin(angle))
            for angle in np.linspace(0, 1.5*np.pi, npcirc, endpoint=False))

    new_facets = round_trip_connect(circ_start, len(points)-1)
    facets = np.vstack((facets,new_facets))
    facets[facets.shape[0]-1,1] = 0

    facets = tuple(map(tuple, facets))

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0)/3
        a1 = control_max_area
        a2 = control_max_area
        max_area = a1 + la.norm(bary, np.inf)*a2
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    built_mesh = triangle.build(info, refinement_func=needs_refinement)
    return np.array(built_mesh.points), np.array(built_mesh.elements)

def make_mesh_corner(control_max_area):
    points = [(-1, -1), (1, -1), (1, 1), (0.2,1), (0,0), (-0.2,1), (0, 1), (-1,1)]
    facets = round_trip_connect(0, len(points)-1)

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0)/3
        a1 = control_max_area
        a2 = control_max_area
        max_area = a1 + la.norm(bary, np.inf)*a2
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    built_mesh = triangle.build(info, refinement_func=needs_refinement)
    return np.array(built_mesh.points), np.array(built_mesh.elements)

def longest_edge(V,E):
    hmax = 0
    iemax = 0
    for ie in range (0,E.shape[0]):
        ix = E[ie,:]
        xl = V[ix]
        h = max(la.norm(xl[0,:]-xl[1,:],2),la.norm(xl[1,:]-xl[2,:],2),la.norm(xl[2,:]-xl[0,:],2))
        if (hmax<h):
            hmax = h
            iemax = ie
    return hmax, iemax

#=============================================================================#
# Finite Element Solver: second order triangular
#=============================================================================#

# Quadrature points
ref_quad_weights = np.array(
         [0.44676317935602256, 0.44676317935602256, 0.44676317935602256,
          0.21990348731064327, 0.21990348731064327, 0.21990348731064327])
ref_quad_nodes = np.array([
    [-0.10810301816807008, -0.78379396366385990],
    [-0.10810301816806966, -0.10810301816807061],
    [-0.78379396366386020, -0.10810301816806944],
    [-0.81684757298045740, -0.81684757298045920],
    [0.63369514596091700, -0.81684757298045810],
    [-0.81684757298045870, 0.63369514596091750]]).T
ref_quad_nodes = (ref_quad_nodes + 1)/2
ref_quad_weights = ref_quad_weights * 0.5

class MatrixBuilder:
    def __init__(self):
        self.rows = []
        self.cols = []
        self.vals = []

    def add(self, rows, cols, submat):
        for i, ri in enumerate(rows):
            for j, cj in enumerate(cols):
                self.rows.append(ri)
                self.cols.append(cj)
                self.vals.append(submat[i, j])

    def coo_matrix(self):
        return sparse.coo_matrix((self.vals, (self.rows, self.cols)))

# Shape function
def ShapeFunc(ksi,eta,N):
    N[0] = (1-ksi-eta)*(1-2*ksi-2*eta)
    N[4] = 4*ksi*(1-ksi-eta)
    N[1] = ksi*(2*ksi-1)
    N[3] = 4*eta*(1-ksi-eta)
    N[5] = 4*ksi*eta
    N[2] = eta*(2*eta-1)
    return N

# Gradient of shape function
def GradShapeFunc(ksi,eta,Bmat):
    Bmat[0,0] = -(1-2*ksi-2*eta) - 2*(1-ksi-eta)
    Bmat[0,4] = 4*(1-ksi-eta) - 4*ksi
    Bmat[0,1] = 4*ksi-1
    Bmat[0,3] = -4*eta
    Bmat[0,5] = 4*eta
    Bmat[0,2] = 0
    Bmat[1,0] = -(1-2*ksi-2*eta) - 2*(1-ksi-eta)
    Bmat[1,4] = -4*ksi
    Bmat[1,1] = 0
    Bmat[1,3] = 4*(1-ksi-eta) - 4*eta
    Bmat[1,5] = 4*ksi
    Bmat[1,2] = 4*eta-1
    return Bmat

def make_quadratic(V,E):
    nv = len(V)-1
    ne = len(E)
    NE = np.zeros([ne,6],dtype=int)
    for ie in range (0,ne):
            ix = E[ie,:]
            xl = V[ix]
            for i in range (0,len(xl)):
                node_new = (xl[i,:]+xl[i-1,:])/2
                inx = np.where((V[:,0] == node_new[0]) & (V[:,1]==node_new[1]))[0]
                if inx.size!=0:
                    ix = np.append(ix,inx)
                else:
                    nv = nv+1
                    V = np.vstack((V,node_new))
                    ix = np.append(ix,nv)
            NE[ie,:] = ix
    return V, NE, nv

def FEM_solver(V,E,BVP):
    # Add nodes to make quadratic triangular element
    nv0 = len(V)
    nv = len(V)-1
    V, NE, nv = make_quadratic(V,E)

    # Build stiffness matrix and force vector
    N = np.zeros(6)
    Bmat = np.zeros([2,6])
    F_global = np.zeros(len(V))
    nint = int(ref_quad_nodes.size/2)

    a_builder = MatrixBuilder()

    for ie in range (0,len(NE)):
        Kmat = np.zeros([6,6])
        Fmat = np.zeros(6)
        ix = NE[ie,:]
        xl = V[ix,:]
        for i in range (0,nint):
            ksi = ref_quad_nodes[0,i]
            eta = ref_quad_nodes[1,i]
            W = ref_quad_weights[i]
            # Shape function
            N = ShapeFunc(ksi,eta,N)
            # Gradient of shape function reference coordinate
            Bmat = GradShapeFunc(ksi,eta,Bmat)
            # Jacobian
            J = Bmat @ xl
            Jdet = np.linalg.det(J)
            Jinv = np.linalg.inv(J)
            # Location of integration point in the physical coordinate
            x = N @ xl
            # Gradient of shape function in physical coordinate
            Bmat = Jinv @ Bmat
            # Stiffness matrix and force vector
            Kmat = Kmat + BVP['kappa'](x)*(Bmat.transpose() @ Bmat)*W*Jdet
            Fmat = Fmat + BVP['f'](x)*N*W*Jdet
        a_builder.add(ix, ix, Kmat)
        F_global[ix] = F_global[ix] + Fmat
    K_global = a_builder.coo_matrix().tocsr().tocoo()

    is_boundary =  BVP["h_dir_boundary"](V)
    is_g_boundary = BVP["nh_dir_boundary"](V.T)

    u0 = np.zeros(len(V))
    u0[is_g_boundary] = BVP['g'](V[is_g_boundary].T)
    rhs = F_global - K_global @ u0
    rhs[is_g_boundary] = 0.0
    rhs[is_boundary] = 0.0

    K_global = K_global.tolil()
    for i in range(0,nv+1):
        if is_boundary[i]:
            K_global[i,:] = 0
            K_global[i,i] = 1
        if is_g_boundary[i]:
            K_global[i,:] = 0
            K_global[i,i] = 1

    # Solve
    uhat = sla.spsolve(K_global.tocsr(), rhs)
    u = uhat + u0
    # Compute error
    uexact = BVP['uexact'](V.T)
    err = la.norm(u-uexact,2)/la.norm(uexact,2)
    # err = la.norm(u-uexact,2)
    return u[0:nv0], err, u


#=============================================================================#
# Collection of Boundary Value Problem
#=============================================================================#

def make_convtest1():
        def kappa(xvec):
            x, y = xvec
            return np.ones(x.size)

        def uexact(xvec):
            x, y = xvec
            return -(y**3) + 3*y + 2

        def f(xvec):
            x, y = xvec
            return 6*y

        def h_dir_boundary(V):
            tol = 1e-12
            is_boundary = (
                (np.abs(V[:,1]+1) < tol))
            return is_boundary

        def nh_dir_boundary(V):
            tol = 1e-12
            V = V.T
            is_boundary = (
                (np.abs(V[:,1]+1) < tol))
            return is_boundary

        def g(xvec):
            x, y = xvec
            return 0*x

        return {"kappa": kappa, "f": f, "g": g, "uexact": uexact, "h_dir_boundary": h_dir_boundary, \
                "nh_dir_boundary": nh_dir_boundary}

def make_convtest2():
        def kappa(xvec):
            x, y = xvec
            return np.ones(x.size)

        def uexact(xvec):
            x, y = xvec
            z = x+y*1j
            phi = np.angle(z)
            for i in range(0,phi.size):
                if phi[i] < 0:
                    phi[i] = phi[i] + np.pi*2
            r = np.sqrt(x**2 + y**2)
            return r**(2/3)*np.sin(2/3*phi)

        def f(xvec):
            x, y = xvec
            return 0*y

        def h_dir_boundary(V):
            tol = 1e-12
            is_boundary = (
                  ( (np.abs(V[:,0]) < tol) & (((V[:,1]) < tol) & (np.abs(V[:,1]+1) > tol))  )
                | ( (np.abs(V[:,1]) < tol) & (((V[:,0]) > tol) & (np.abs(V[:,0]-1) > tol))  )
                )
            return is_boundary

        def nh_dir_boundary(V):
            x, y = V
            tol = 1e-12
            is_g_boundary = (np.abs(x**2 + y**2 - 1) < tol)
            return is_g_boundary

        def g(xvec):
            x, y = xvec
            z = x+y*1j
            phi = np.angle(z)
            for i in range(0,phi.size):
                if phi[i] < 0:
                    phi[i] = phi[i] + np.pi*2
            return np.sin(2/3*phi)

        return {"kappa": kappa, "f": f, "g": g, "uexact": uexact, "h_dir_boundary": h_dir_boundary, \
                "nh_dir_boundary": nh_dir_boundary}

def make_convtest3():
        def kappa(xvec):
            x, y = xvec
            return np.ones(x.size)

        def uexact(xvec):
            x, y = xvec
            return np.ones(x.size)

        def f(xvec):
            x, y = xvec
            return 0*y

        def h_dir_boundary(V):
            tol = 1e-12
            is_boundary = (
                (np.abs(V[:,1]+1) < tol))
            return is_boundary

        def nh_dir_boundary(V):
            V = V.T
            tol = 1e-12
            is_g_boundary = (
                (np.abs(V[:,1]-1) < tol))
            return is_g_boundary

        def g(xvec):
            x, y = xvec
            gbound = np.zeros(x.size)
            for i in range (0,x.size):
                if (x[i]<0):
                    gbound[i] = 0.5
                else:
                    gbound[i] = 0
            return gbound

        return {"kappa": kappa, "f": f, "g": g, "uexact": uexact, "h_dir_boundary": h_dir_boundary, \
                "nh_dir_boundary": nh_dir_boundary}

#=============================================================================#
# Error Estimator
#=============================================================================#

quad_nodes_line_12 = np.array([
    [0.211324865405187, 0.0],
    [0.788675134594813, 0.0]]).T
quad_weight_line_12 = np.array([0.5, 0.5])

quad_nodes_line_23 = np.array([
    [0.0, 0.211324865405187],
    [0.0, 0.788675134594813]]).T
quad_weight_line_23 = np.array([0.5, 0.5])

quad_nodes_line_31 = np.array([
    [0.211324865405187,   0.788675134594813],
    [0.788675134594813,   0.211324865405187]]).T
quad_weight_line_31 = np.array([0.707106781186548, 0.707106781186548])

def find_edge_connection(V0,E0,ix,ix2,ie):
    # Element and node connectivities with neighbors
    # nv = len(V0)
    ne = len(E0)
    element_ids = np.empty((ne, 3), dtype=np.intp)
    element_ids[:] = np.arange(ne).reshape(-1, 1)
    V2E = sparse.coo_matrix(
        (np.ones((ne*3,), dtype=np.intp),
         (E0.ravel(),
          element_ids.ravel(),)))
    elnum = np.arange(0,ne)
    # E2E = V2E.T @ V2E
    # V2V = V2E @ V2E.T
    vcon = V2E!=0
    vcon = vcon.toarray()
    # Element connected to vertex
    iv1 = ix2[0]
    iv2 = ix2[1]
    iv3 = ix2[2]
    elcon1 = elnum[vcon[iv1]]   # list of element connected to vertex iv1
    elcon2 = elnum[vcon[iv2]]   # list of element connected to vertex iv2
    elcon3 = elnum[vcon[iv3]]   # list of element connected to vertex iv3
    # remove current vertex
    elcon1 = elcon1[elcon1!=ie]
    elcon2 = elcon2[elcon2!=ie]
    elcon3 = elcon3[elcon3!=ie]
    # Find element that connect through the edge
    elcon12 = np.intersect1d(elcon1,elcon2)
    elcon23 = np.intersect1d(elcon2,elcon3)
    elcon31 = np.intersect1d(elcon3,elcon1)
    return elcon12, elcon23, elcon31

def GradShapeFunc2(ksi,eta,xl,J,Bmat,Bmat2):
    # Matrix of second derivative of shape function
    Bmat2[0,0] = 4
    Bmat2[0,4] = -8
    Bmat2[0,1] = 4
    Bmat2[0,3] = 0
    Bmat2[0,5] = 0
    Bmat2[0,2] = 0
    Bmat2[1,0] = 4
    Bmat2[1,4] = 0
    Bmat2[1,1] = 0
    Bmat2[1,3] = -8
    Bmat2[1,5] = 0
    Bmat2[1,2] = 4
    Bmat2[2,0] = 4
    Bmat2[2,4] = -4
    Bmat2[2,1] = 0
    Bmat2[2,3] = -4
    Bmat2[2,5] = 4
    Bmat2[2,2] = 0

    # Matrix of second derivative of shape function in the physical coordinate
    T1 = np.zeros([3,3])
    T2 = Bmat2@xl
    d2N_bar = Bmat2 - (T2@Bmat)
    T1[0,0] = J[0,0]**2
    T1[0,1] = J[0,1]**2
    T1[0,2] = 2*J[0,0]*J[0,1]
    T1[1,0] = J[1,0]**2
    T1[1,1] = J[1,1]**2
    T1[1,2] = 2*J[1,0]*J[1,1]
    T1[2,0] = J[0,0]*J[1,0]
    T1[2,1] = J[0,1]*J[1,1]
    T1[2,2] = J[0,0]*J[1,1] + J[1,0]*J[0,1]
    invT1 = np.linalg.inv(T1)
    Bmat2 = invT1 @ d2N_bar

    return Bmat2

def compute_edge_jump_error(n1,n2,V,E,E0,u,ul,ix2,xl,xl2,nint_l,elcon,quad_nodes_line,quad_weight_line):
    N = np.zeros(6)
    N_c = np.zeros(6)
    Bmat = np.zeros([2,6])
    Bmat_c = np.zeros([2,6])
    nvec = np.zeros(2)
    normRe = 0
    ixc = E[elcon[0],:]
    ixc2 = E0[elcon[0],:]
    xlc = V[ixc]
    ulc = u[ixc]
    # Find number of nodes in the corresponding elemnet
    elnum = np.arange(0,3)
    node1 = elnum[ixc2 == ix2[n1-1]][0]
    node2 = elnum[ixc2 == ix2[n2-1]][0]
    # Gauss points
    quad_nodes_line = quad_nodes_line
    quad_weight_line = quad_weight_line
    if ( node1+node2+2 == 3):
        quad_nodes_line_c = quad_nodes_line_12
    elif( node1+node2+2 == 5):
        quad_nodes_line_c = quad_nodes_line_23
    elif( node1+node2+2 == 4):
        quad_nodes_line_c = quad_nodes_line_31
    # Normal vector of the node
    vec = xl2[node2]-xl2[node1]
    nvec[0] = -vec[1]
    nvec[1] = vec[0]
    nvec = nvec/(la.norm(nvec,2))
    he = la.norm(vec,2)
    # Integrate along the edge
    for i in range (0,nint_l):
        # i = 1
        # Gaussian quadrature point
        ksi = quad_nodes_line[0,i]
        eta = quad_nodes_line[1,i]
        ksi_c = quad_nodes_line_c[0,i]
        eta_c = quad_nodes_line_c[1,i]
        W = quad_weight_line[i]
        # Shape function
        N = ShapeFunc(ksi,eta,N)
        N_c = ShapeFunc(ksi_c,eta_c,N_c)
        # Gradient of shape function reference coordinate
        Bmat = GradShapeFunc(ksi,eta,Bmat)
        Bmat_c = GradShapeFunc(ksi_c,eta_c,Bmat_c)
        # Jacobian
        J = Bmat @ xl
        Jdet = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        J_c = Bmat_c @ xlc
        Jinv_c = np.linalg.inv(J_c)
        # Gradient of shape function in physical coordinate
        Bmat = Jinv @ Bmat
        Bmat_c = Jinv_c @ Bmat_c
        # Compute jump on gradient in the perpendicular direction
        grad1 = (Bmat @ ul) @ nvec
        grad2 = (Bmat_c @ ulc) @ nvec
        RE = (grad1-grad2)
        normRe = normRe + he/2*(RE**2)*W*Jdet
    return normRe

def error_estimate(V0,E0,BVP):
    u3, err0, u = FEM_solver(V0,E0,BVP)
    V, E, nv = make_quadratic(V0,E0)
    u_exact = BVP['uexact'](V.T)
    e = u_exact-u
    e = e
    nint = int(ref_quad_nodes.size/2)
    nint_l = int(quad_nodes_line_12.size/2)
    N = np.zeros(6)
    Bmat = np.zeros([2,6])
    Bmat2 = np.zeros([3,6])
    eta_TR = np.zeros(E.shape[0])
    errnorm_T = np.zeros(E.shape[0])
    ferrnorm_T = np.zeros(E.shape[0])

    for ie in range(0,E.shape[0]):
        # ie = 0
        ix = E[ie,:]
        ix2 = E0[ie,:]
        xl = V[ix]
        xl2 = V0[ix2]
        ul = u[ix]
        el = e[ix]
        fl = BVP['f'](xl.T)
        h = max(la.norm(xl2[0,:]-xl2[1,:],2),la.norm(xl2[1,:]-xl2[2,:],2),la.norm(xl2[2,:]-xl2[0,:],2))
        error_H1_norm = 0
        error_f_norm = 0
        normRT = 0
        normRe = 0
        for i in range (0,nint):
            # Gaussian quadrature point
            ksi = ref_quad_nodes[0,i]
            eta = ref_quad_nodes[1,i]
            W = ref_quad_weights[i]
            # Shape function
            N = ShapeFunc(ksi,eta,N)
            # Gradient of shape function reference coordinate
            Bmat = GradShapeFunc(ksi,eta,Bmat)
            # Jacobian
            J = Bmat @ xl
            Jdet = np.linalg.det(J)
            Jinv = np.linalg.inv(J)
            # Location of integration point in the physical coordinate
            x = N @ xl
            # Gradient of shape function in physical coordinate
            Bmat = Jinv @ Bmat
            # Matrix of second derivative of shape function in the physical coordinate
            Bmat2 = GradShapeFunc2(ksi,eta,xl,J,Bmat,Bmat2)
            # Laplacian of FEM solution
            Lap_uh = Bmat2 @ ul
            # Body force and its representation
            f_exact = BVP['f'](x)
            f_interp = N @ fl
            # Residual from PDE
            RT = (Lap_uh[0] + Lap_uh[1] + BVP['f'](x))
            normRT = normRT + (h**2)*(RT**2)*W*Jdet
            # ErrX, Y = V0[:, 0], V0[:, 1]or norm with respect to analytical solution
            error_H1_norm = error_H1_norm + ((N@el)**2+ np.sum(Bmat@el)**2)*W*Jdet
            # Error in representation of body force
            error_f_norm = error_f_norm + (h**2)*((f_exact-f_interp)**2)*W*Jdet
        # Residual from jump in gradient between element
        elcon12, elcon23, elcon31 = find_edge_connection(V0,E0,ix,ix2,ie)
        if (elcon12.size!=0):
            normRe = normRe + compute_edge_jump_error(1,2,V,E,E0,u,ul,ix2,xl,xl2,nint_l,elcon12,quad_nodes_line_12,quad_weight_line_12)
        if (elcon23.size!=0):
            normRe = normRe + compute_edge_jump_error(2,3,V,E,E0,u,ul,ix2,xl,xl2,nint_l,elcon23,quad_nodes_line_23,quad_weight_line_23)
        if (elcon31.size!=0):
            normRe = normRe + compute_edge_jump_error(3,1,V,E,E0,u,ul,ix2,xl,xl2,nint_l,elcon31,quad_nodes_line_31,quad_weight_line_31)

        # \eta_{T,R}
        eta_TR[ie] = np.sqrt(normRT + normRe)
        # \Vert u-u_h \Vert_{1,T}
        errnorm_T[ie] = np.sqrt(error_H1_norm)
        # \Vert f-f_h \Vert_{0,T'}^2
        ferrnorm_T[ie] = np.sqrt(error_f_norm)
        # For efficiency
    err_norm_elem = np.zeros(E0.shape[0])
    for ie in range (0,E0.shape[0]):
        elcon12, elcon23, elcon31 = find_edge_connection(V0,E0,ix,ix2,ie)
        if (elcon12.size!=0):
            es1 = errnorm_T[elcon12[0]]
            fs1 = ferrnorm_T[elcon12[0]]
        else:
            es1 = 0
            fs1 = 0
        if (elcon23.size!=0):
            es2 = errnorm_T[elcon23[0]]
            fs2 = ferrnorm_T[elcon23[0]]
        else:
            es2 = 0
            fs2 = 0
        if (elcon31.size!=0):
            es3 = errnorm_T[elcon31[0]]
            fs3 = ferrnorm_T[elcon31[0]]
        else:
            es3 = 0
            fs3 = 0
        err_norm_elem[ie] = err_norm_elem[ie] + ( errnorm_T[ie] + es1 + es2 + es3 )**2
        err_norm_elem[ie] = err_norm_elem[ie] + ( ferrnorm_T[ie] + fs1 + fs2 + fs3 )**2

    return eta_TR, errnorm_T, ferrnorm_T, np.sqrt(err_norm_elem), u3

def plot_errormap(X,Y,E,errnorm_T,eta_TR,name):
    ft = 23
    errormap_rec1 = plt.figure()
    errormap_rec1.set_size_inches(19,5)
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect("equal")
    plt.triplot(X, Y, E)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('mesh', fontsize=ft)
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X, Y, triangles=E, facecolors=errnorm_T)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X, Y, triangles=E, facecolors=eta_TR)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('$\eta_{T,R}$', fontsize=ft)
    plt.colorbar()
    plt.tight_layout()
    errormap_rec1.savefig(name,dpi=300)

#=============================================================================#
# Adaptive FEM
#=============================================================================#

def plot_errormap2(X,Y,E,eta_TR,name):
    ft = 23
    errormap_rec1 = plt.figure()
    errormap_rec1.set_size_inches(12,5)
    plt.subplot(1, 2, 1)
    plt.gca().set_aspect("equal")
    plt.triplot(X, Y, E)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('mesh', fontsize=ft)
    plt.subplot(1, 2, 2)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X, Y, triangles=E, facecolors=eta_TR)
    # plt.triplot(X, Y, E)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('$\eta_{T,R}$', fontsize=ft)
    plt.colorbar()
    plt.tight_layout()
    errormap_rec1.savefig(name,dpi=300)

def plot_to_be_refined(X,Y,E,to_be_refined):
    ft = 18
    hmesh = plt.figure()
    hmesh.set_size_inches(6,6)
    plt.gca().set_aspect("equal")
    plt.triplot(X, Y, E)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.tripcolor(X, Y, triangles=E, facecolors=to_be_refined)
    plt.colorbar()
    plt.tight_layout()

def plot_refine(X,Y,E,eta_TR,to_be_refined,name):
    ft = 23
    errormap_rec1 = plt.figure()
    errormap_rec1.set_size_inches(18,5)
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect("equal")
    plt.triplot(X, Y, E)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('mesh', fontsize=ft)
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X, Y, triangles=E, facecolors=eta_TR)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.title('$\eta_{T,R}$', fontsize=ft)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect("equal")
    plt.triplot(X, Y, E)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.tripcolor(X, Y, triangles=E, facecolors=to_be_refined)
    plt.title('Element to be refined', fontsize=ft)
    plt.colorbar()
    plt.tight_layout()
    errormap_rec1.savefig(name,dpi=300)

def bisect(V,E,nv,ne,to_be_refined):
    elnum = np.arange(0,ne)
    el_idx = elnum[to_be_refined]
    nv = nv-1
    for i in range(0,el_idx.size):
        ie = el_idx[i]
        ix = E[ie]
        xl = V[ix]
        len12 = np.linalg.norm(xl[0,:]-xl[1,:],2)
        len23 = np.linalg.norm(xl[1,:]-xl[2,:],2)
        len31 = np.linalg.norm(xl[2,:]-xl[0,:],2)
        len_max = max(len12,len23,len31)
        if (len_max == len12):
            P = (xl[0,:]+xl[1,:])/2
            inx = np.where((V[:,0] == P[0]) & (V[:,1]==P[1]))[0]
            if inx.size!=0:
                E = np.vstack((E,np.array([ix[0],inx[0],ix[2]])))
                E[ie,0] = inx
            else:
                nv = nv+1
                V = np.vstack((V,P))
                E = np.vstack((E,np.array([ix[0],nv,ix[2]])))
                E[ie,0] = nv
        elif (len_max == len23):
            P = (xl[1,:]+xl[2,:])/2
            inx = np.where((V[:,0] == P[0]) & (V[:,1]==P[1]))[0]
            if inx.size!=0:
                E = np.vstack((E,np.array([ix[0],inx[0],ix[2]])))
                E[ie,2] = inx
            else:
                nv = nv+1
                V = np.vstack((V,P))
                E = np.vstack((E,np.array([ix[0],nv,ix[2]])))
                E[ie,2] = nv
        elif (len_max == len31):
            P = (xl[2,:]+xl[0,:])/2
            inx = np.where((V[:,0] == P[0]) & (V[:,1]==P[1]))[0]
            if inx.size!=0:
                E = np.vstack((E,np.array([inx[0],ix[1],ix[2]])))
                E[ie,2] = inx
            else:
                nv = nv+1
                V = np.vstack((V,P))
                E = np.vstack((E,np.array([nv,ix[1],ix[2]])))
                E[ie,2] = nv
        to_be_refined = np.append(to_be_refined,True)
    nv = nv+1
    ne = E.shape[0]
    return V, E, ne, nv, to_be_refined

def make_conform(V,E,nv,ne):
    nv = nv-1
    nconf_count = 0

    # If longest edge of nonconforming element if bisected
    for ie in range (0,E.shape[0]):
        ix = E[ie,:]
        xl = V[ix]
        # Compute Length
        len12 = np.linalg.norm(xl[0,:]-xl[1,:],2)
        len23 = np.linalg.norm(xl[1,:]-xl[2,:],2)
        len31 = np.linalg.norm(xl[2,:]-xl[0,:],2)
        len_max = max(len12,len23,len31)

        if (len_max == len12):
            P = (xl[0,:]+xl[1,:])/2
            inx = np.where((V[:,0] == P[0]) & (V[:,1]==P[1]))[0]
            if inx.size!=0:
                E = np.vstack((E,np.array([ix[0],inx[0],ix[2]])))
                E[ie,0] = inx[0]
        elif (len_max == len23):
            P = (xl[1,:]+xl[2,:])/2
            inx = np.where((V[:,0] == P[0]) & (V[:,1]==P[1]))[0]
            if inx.size!=0:
                E = np.vstack((E,np.array([ix[0],inx[0],ix[2]])))
                E[ie,2] = inx[0]
        elif (len_max == len31):
            P = (xl[2,:]+xl[0,:])/2
            inx = np.where((V[:,0] == P[0]) & (V[:,1]==P[1]))[0]
            if inx.size!=0:
                E = np.vstack((E,np.array([inx[0],ix[1],ix[2]])))
                E[ie,2] = inx[0]

    # Otherwise
    for ie in range (0,E.shape[0]):
        ix = E[ie,:]
        xl = V[ix]

        len12 = np.linalg.norm(xl[0,:]-xl[1,:],2)
        len23 = np.linalg.norm(xl[1,:]-xl[2,:],2)
        len31 = np.linalg.norm(xl[2,:]-xl[0,:],2)
        len_max = max(len12,len23,len31)

        Q12 = (xl[0,:]+xl[1,:])/2
        Q23 = (xl[1,:]+xl[2,:])/2
        Q31 = (xl[2,:]+xl[0,:])/2

        if (len_max == len12):
            P = (xl[0,:]+xl[1,:])/2
        elif (len_max == len23):
            P = (xl[1,:]+xl[2,:])/2
        elif (len_max == len31):
            P = (xl[2,:]+xl[0,:])/2

        inx12 = np.where((V[:,0] == Q12[0]) & (V[:,1]==Q12[1]))[0]
        inx23 = np.where((V[:,0] == Q23[0]) & (V[:,1]==Q23[1]))[0]
        inx31 = np.where((V[:,0] == Q31[0]) & (V[:,1]==Q31[1]))[0]

        if inx12.size!=0:
            nconf_count = nconf_count + 1
            inx = inx12
        elif inx23.size!=0:
            nconf_count = nconf_count + 1
            inx = inx23
        elif inx31.size!=0:
            nconf_count = nconf_count + 1
            inx = inx31
        else:
            continue

        inx = inx[0]
        if ( (inx12.size!=0) & (len_max == len31) ):
            nv = nv+1
            V = np.vstack((V,P))
            E = np.vstack((E,np.array([ix[0],inx,nv])))
            E = np.vstack((E,np.array([inx,ix[1],nv])))
            E[ie,0] = nv
        elif ( (inx12.size!=0) & (len_max == len23) ):
            nv = nv+1
            V = np.vstack((V,P))
            E = np.vstack((E,np.array([inx,ix[1],nv])))
            E = np.vstack((E,np.array([inx,nv,ix[2]])))
            E[ie,1] = inx
        elif ( (inx23.size!=0) & (len_max == len12) ):
            nv = nv+1
            V = np.vstack((V,P))
            E = np.vstack((E,np.array([nv,ix[1],inx])))
            E = np.vstack((E,np.array([nv,inx,ix[2]])))
            E[ie,1] = nv
        elif ( (inx23.size!=0) & (len_max == len31) ):
            nv = nv+1
            V = np.vstack((V,P))
            E = np.vstack((E,np.array([ix[0],inx,nv])))
            E = np.vstack((E,np.array([nv,inx,ix[2]])))
            E[ie,2] = inx
        elif ( (inx31.size!=0) & (len_max == len12) ):
            nv = nv+1
            V = np.vstack((V,P))
            E = np.vstack((E,np.array([ix[0],nv,inx])))
            E = np.vstack((E,np.array([inx,nv,ix[2]])))
            E[ie,0] = nv
        elif ( (inx31.size!=0) & (len_max == len23) ):
            nv = nv+1
            V = np.vstack((V,P))
            E = np.vstack((E,np.array([ix[0],nv,inx])))
            E = np.vstack((E,np.array([inx,nv,ix[2]])))
            E[ie,2] = nv

    return V, E, nconf_count

def refine(V,E,nv,ne,to_be_refined):
    V, E, ne, nv, to_be_refined = bisect(V,E,nv,ne,to_be_refined)
    nconf_count = 1
    while (nconf_count != 0):
        V, E, nconf_count = make_conform(V,E,nv,ne)
    return V, E

#=============================================================================#
# Test Case for Investigation
#=============================================================================#

def testFEM():
    convtest1 = make_convtest1()

    V, E = make_mesh(0.1)
    h0, iemax0 = longest_edge(V,E)
    u, err0, _ = FEM_solver(V,E,convtest1)

    V1, E1 = make_mesh(0.01)
    h1, iemax1 = longest_edge(V1,E1)
    # X1, Y1 = V1[:, 0], V1[:, 1]
    u1, err1, _ = FEM_solver(V1,E1,convtest1)

    ft = 15
    ft2 = 12

    V, E = make_mesh(0.001)
    h2, iemax2 = longest_edge(V,E)
    u, err2, _ = FEM_solver(V,E,convtest1)

    # fig = plt.figure(figsize=(7,7))
    # ax = Axes3D(fig)
    # ax.plot_trisurf(X1, Y1, u1, triangles=E1, cmap=plt.cm.jet, linewidth=0.2)
    # plt.xlabel('$X$', fontsize=ft)
    # plt.ylabel('$Y$', fontsize=ft)
    # plt.show()

    rec_conv = plt.figure()
    rec_conv.set_size_inches(6,4)
    # plt.subplot(1, 2, 1)
    # plt.gca().set_aspect("equal")
    # plt.triplot(X1, Y1, E1)
    # plt.xlabel('$X$', fontsize=ft)
    # plt.ylabel('$Y$', fontsize=ft)
    # plt.subplot(1, 2, 2)
    t = np.linspace(1,0.1,2)
    f1 = (t**3)*30*0.001
    plt.loglog([h0,h1,h2],[err0,err1,err2],t,f1,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error norm', fontsize=ft)
    plt.legend(('$ \Vert u-u_h \Vert_{0} \; / \; \Vert u \Vert_{0}$','$O(h^3)$'), fontsize=ft2)
    plt.show()
    rec_conv.savefig('plot_rec_conv.png',dpi=300)


def testFEM_corner():
    ft = 15
    ft2 = 12

    convtest2 = make_convtest2()

    V1, E1 = make_mesh_circ(0.1,700)
    h1, iemax1 = longest_edge(V1,E1)
    u1, err1, _ = FEM_solver(V1,E1,convtest2)
    # X1, Y1 = V1[:, 0], V1[:, 1]

    V2, E2 = make_mesh_circ(0.01,700)
    h2, iemax2 = longest_edge(V2,E2)
    # X2, Y2 = V2[:, 0], V2[:, 1]
    u2, err2, _ = FEM_solver(V2,E2,convtest2)

    V3, E3 = make_mesh_circ(0.005,700)
    h3, iemax3 = longest_edge(V3,E3)
    u3, err3, _ = FEM_solver(V3,E3,convtest2)
    # X3, Y3 = V3[:, 0], V3[:, 1]

    # mesh = plt.figure()
    # mesh.set_size_inches(16,4)
    # plt.subplot(1,3,1)
    # plt.gca().set_aspect("equal")
    # plt.triplot(X1, Y1, E1)
    # plt.xlabel('$X$', fontsize=ft)
    # plt.ylabel('$Y$', fontsize=ft)
    # plt.subplot(1,3,2)
    # plt.gca().set_aspect("equal")
    # plt.triplot(X2, Y2, E2)
    # plt.xlabel('$X$', fontsize=ft)
    # plt.ylabel('$Y$', fontsize=ft)
    # plt.subplot(1,3,3)
    # plt.gca().set_aspect("equal")
    # plt.triplot(X3, Y3, E3)
    # plt.xlabel('$X$', fontsize=ft)
    # plt.ylabel('$Y$', fontsize=ft)
    # mesh.savefig('plot_circ_mesh.png',dpi=300)

    # ctest = plt.figure(figsize=(7,7))
    # ax = Axes3D(ctest)
    # ax.plot_trisurf(X2, Y2, u2, triangles=E2, cmap=plt.cm.jet, linewidth=0.2)
    # plt.xlabel('$X$', fontsize=ft)
    # plt.ylabel('$Y$', fontsize=ft)
    # plt.show()
    # ctest.savefig('plot_circ_sol.png',dpi=300)

    konv = 2/3/2
    t = np.linspace(0.7,0.2,2)
    f1 = (t**konv)*30*0.0001
    cconv = plt.figure()
    cconv.set_size_inches(7,4)
    plt.loglog([h1,h2,h3],[err1,err2,err3],t,f1,'--')
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error norm', fontsize=ft)
    plt.legend(('$ \Vert u-u_h \Vert_{0} \; / \; \Vert u \Vert_{0}$','$O(h^{1/3})$'), fontsize=ft2)
    plt.show()
    cconv.savefig('plot_circ_conv.png',dpi=300)
    plt.tight_layout()

def error_est_test_smooth():
    # Investigate reliable bound
    c = 200
    err_norm_total = np.zeros(3)
    eta_R = np.zeros(3)
    # Set up BVP
    convtest1 = make_convtest1()
    # Mesh 1
    V1, E1 = make_mesh(0.1)
    X1, Y1 = V1[:, 0], V1[:, 1]
    h1, iemax1 = longest_edge(V1,E1)
    eta_TR1, errnorm_T1, ferrnorm_T1, err_norm_elem1, u1 = error_estimate(V1,E1,convtest1)
    err_norm_total[0] = np.sum(errnorm_T1)
    eta_R[0] = np.sqrt(np.sum(eta_TR1**2))
    # Mesh 2
    V2, E2 = make_mesh(0.01)
    X2, Y2 = V2[:, 0], V2[:, 1]
    h2, iemax2 = longest_edge(V2,E2)
    eta_TR2, errnorm_T2, ferrnorm_T2, err_norm_elem2, u2 = error_estimate(V2,E2,convtest1)
    err_norm_total[1] = np.sum(errnorm_T2)
    eta_R[1] = np.sqrt(np.sum(eta_TR2**2))
    # Mesh 3
    V3, E3 = make_mesh(0.001)
    X3, Y3 = V3[:, 0], V3[:, 1]
    h3, iemax3 = longest_edge(V3,E3)
    eta_TR3, errnorm_T3, ferrnorm_T3, err_norm_elem3, u3 = error_estimate(V3,E3,convtest1)
    err_norm_total[2] = np.sum(errnorm_T3)
    eta_R[2] = np.sqrt(np.sum(eta_TR3**2))

    # Plot efficiency check
    ft = 23
    rec_eff_check = plt.figure()
    rec_eff_check.set_size_inches(19,5)
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X1, Y1, triangles=E1, facecolors=c*err_norm_elem1-eta_TR1)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    # plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X2, Y2, triangles=E2, facecolors=c*err_norm_elem2-eta_TR2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    # plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X3, Y3, triangles=E3, facecolors=c*err_norm_elem3-eta_TR3)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    # plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.tight_layout()
    rec_eff_check.savefig('plot_rec_eff_check.png',dpi=300)

    # Plot Solution
    ft = 15
    ft2 = 12
    rec_sol = plt.figure()
    rec_sol.set_size_inches(15,5)
    ax = rec_sol.add_subplot(1, 3, 1, projection='3d')
    ax.plot_trisurf(X1, Y1, u1, triangles=E1, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    ax = rec_sol.add_subplot(1, 3, 2, projection='3d')
    ax.plot_trisurf(X2, Y2, u2, triangles=E2, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    ax = rec_sol.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(X3, Y3, u3, triangles=E3, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.tight_layout()
    rec_sol.savefig('plot_rec_sol',dpi=300)

    # Plot error element-wise
    plot_errormap(X1,Y1,E1,errnorm_T1,eta_TR1,'plot_rec_errormap1.png')
    plot_errormap(X2,Y2,E2,errnorm_T2,eta_TR2,'plot_rec_errormap2.png')
    plot_errormap(X3,Y3,E3,errnorm_T3,eta_TR3,'plot_rec_errormap3.png')

    # Plot error evolution
    konv = 2
    t = np.linspace(1.2,0.1,2)
    f1 = (t**konv)*30
    glob_error = plt.figure()
    glob_error.set_size_inches(12,4)
    plt.subplot(1, 2, 1)
    plt.loglog([h1,h2,h3],err_norm_total)
    plt.loglog([h1,h2,h3],eta_R)
    plt.loglog(t,f1,'--')
    plt.legend(('$ \Vert u-u_h \Vert_{1,\Omega} $','$\eta_R$','$O(h^{2})$'), loc='upper left', fontsize=ft2+2)
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error norm', fontsize=ft)
    glob_error.savefig('plot_rec_error.png',dpi=300)
    plt.subplot(1, 2, 2)
    plt.loglog([h1,h2,h3],err_norm_total)
    plt.loglog([h1,h2,h3],eta_R*c)
    plt.loglog(t,f1,'--')
    plt.legend(('$ \Vert u-u_h \Vert_{1,\Omega} $','$c \eta_R$','$O(h^{2})$'), loc='upper left', fontsize=ft2+2)
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error norm', fontsize=ft)
    glob_error.savefig('plot_rec_error.png',dpi=300)


def error_est_test_corner():
    # Investigate reliable bound
    c = 200
    err_norm_total = np.zeros(3)
    eta_R = np.zeros(3)
    # Set up BVP
    convtest1 = make_convtest2()
    # Mesh 1
    V1, E1 = make_mesh_circ(0.1,700)
    X1, Y1 = V1[:, 0], V1[:, 1]
    h1, iemax1 = longest_edge(V1,E1)
    eta_TR1, errnorm_T1, ferrnorm_T1, err_norm_elem1, u1 = error_estimate(V1,E1,convtest1)
    err_norm_total[0] = np.sum(errnorm_T1)
    eta_R[0] = np.sqrt(np.sum(eta_TR1**2))
    # Mesh 2
    V2, E2 = make_mesh_circ(0.01,700)
    X2, Y2 = V2[:, 0], V2[:, 1]
    h2, iemax2 = longest_edge(V2,E2)
    eta_TR2, errnorm_T2, ferrnorm_T2, err_norm_elem2, u2 = error_estimate(V2,E2,convtest1)
    err_norm_total[1] = np.sum(errnorm_T2)
    eta_R[1] = np.sqrt(np.sum(eta_TR2**2))
    # Mesh 3
    V3, E3 = make_mesh_circ(0.005,700)
    X3, Y3 = V3[:, 0], V3[:, 1]
    h3, iemax3 = longest_edge(V3,E3)
    eta_TR3, errnorm_T3, ferrnorm_T3, err_norm_elem3, u3 = error_estimate(V3,E3,convtest1)
    err_norm_total[2] = np.sum(errnorm_T3)
    eta_R[2] = np.sqrt(np.sum(eta_TR3**2))


    # Plot efficiency check
    ft = 23
    rec_eff_check = plt.figure()
    rec_eff_check.set_size_inches(19,5)
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X1, Y1, triangles=E1, facecolors=c*err_norm_elem1-eta_TR1)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    # plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X2, Y2, triangles=E2, facecolors=c*err_norm_elem2-eta_TR2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    # plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect("equal")
    plt.tripcolor(X3, Y3, triangles=E3, facecolors=c*err_norm_elem3-eta_TR3)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    # plt.title('$ \Vert u-u_h \Vert_{1,T} $', fontsize=ft)
    plt.colorbar()
    plt.tight_layout()
    rec_eff_check.savefig('plot_circ_eff_check.png',dpi=300)

    # Plot Solution
    ft = 15
    ft2 = 12
    rec_sol = plt.figure()
    rec_sol.set_size_inches(15,5)
    ax = rec_sol.add_subplot(1, 3, 1, projection='3d')
    ax.plot_trisurf(X1, Y1, u1, triangles=E1, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    ax = rec_sol.add_subplot(1, 3, 2, projection='3d')
    ax.plot_trisurf(X2, Y2, u2, triangles=E2, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    ax = rec_sol.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(X3, Y3, u3, triangles=E3, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.tight_layout()
    rec_sol.savefig('plot_circ_sol',dpi=300)

    # Plot error element-wise
    plot_errormap(X1,Y1,E1,errnorm_T1,eta_TR1,'plot_circ_errormap1.png')
    plot_errormap(X2,Y2,E2,errnorm_T2,eta_TR2,'plot_circ_errormap2.png')
    plot_errormap(X3,Y3,E3,errnorm_T3,eta_TR3,'plot_circ_errormap3.png')

    # Plot error evolution
    glob_error = plt.figure()
    glob_error.set_size_inches(12,4)
    plt.subplot(1, 2, 1)
    plt.loglog([h1,h2,h3],err_norm_total)
    plt.loglog([h1,h2,h3],eta_R)
    plt.legend(('$ \Vert u-u_h \Vert_{1,\Omega} $','$\eta_R$'), loc='best', fontsize=ft2+2)
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error norm', fontsize=ft)

    plt.subplot(1, 2, 2)
    plt.loglog([h1,h2,h3],err_norm_total)
    plt.loglog([h1,h2,h3],eta_R*c)
    plt.legend(('$ \Vert u-u_h \Vert_{1,\Omega} $','$c \eta_R$'), loc='best', fontsize=ft2+2)
    plt.xlabel('$h$', fontsize=ft)
    plt.ylabel('error norm', fontsize=ft)
    glob_error.savefig('plot_circ_error.png',dpi=300)


def adaptive_test():
    # Error threshold for mesh to be refined
    th = 50
    # Set initial mesh
    V, E = make_mesh_corner(0.1)
    # Set up variable for plotting
    err_norm_total = np.zeros(20)
    plot_dof = np.zeros(20)
    eta_R = np.zeros(20)
    numiter = np.zeros(20)
    # Set up BVP
    convtest3 = make_convtest3()
    # Solve
    eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
    err_norm_total[0] = np.sum(errnorm_T)
    eta_R[0] = np.sqrt(np.sum(eta_TR**2))
    plot_dof[0] = V.shape[0]
    # For plotting later
    X, Y = V[:, 0], V[:, 1]
    X0 = X
    Y0 = Y
    u0 = u
    E0 = E
    # Element to be refined
    th_eta_TR = np.max(eta_TR)*th/100
    to_be_refined = eta_TR > th_eta_TR
    # Plot error element-wise0
    plot_refine(X,Y,E,eta_TR,to_be_refined,'plot_adapt_err_00.png')
    # Adaptive FEM
    for i in range (1,20):
        if (i<10):
            name = 'plot_adapt_err_0'+str(i)+'.png'
        else:
            name = 'plot_adapt_err_'+str(i)+'.png'
        # Refine using Rivara method
        V, E = refine(V,E,V.shape[0],E.shape[0],to_be_refined)
        # Solve
        eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
        err_norm_total[i] = np.sum(errnorm_T)
        eta_R[i] = np.sqrt(np.sum(eta_TR**2))
        plot_dof[i] = V.shape[0]
        numiter[i] = i
        # Element to be refined
        th = 50
        th_eta_TR = np.max(eta_TR)*th/100
        to_be_refined = eta_TR > th_eta_TR
        X, Y = V[:, 0], V[:, 1]
        plot_refine(X,Y,E,eta_TR,to_be_refined,name)
        # Save for plot
        if (i==10):
            u10 = u
            X10 = X
            Y10 = Y
            E10 = E
    # Plot Solution
    ft = 15
    ft2 = 12
    adp_sol = plt.figure()
    adp_sol.set_size_inches(15,5)
    ax = adp_sol.add_subplot(1, 3, 1, projection='3d')
    ax.plot_trisurf(X0, Y0, u0, triangles=E0, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    ax = adp_sol.add_subplot(1, 3, 2, projection='3d')
    ax.plot_trisurf(X10, Y10, u10, triangles=E10, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    ax = adp_sol.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(X, Y, u, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
    plt.xlabel('$X$', fontsize=ft)
    plt.ylabel('$Y$', fontsize=ft)
    plt.tight_layout()
    adp_sol.savefig('plot_adapt_sol.png',dpi=300)
    # Run non-adaptive FEM solver for comparison
    # Set up variable for plotting
    err_norm_total_na = np.zeros(3)
    plot_dof_na = np.zeros(3)
    eta_R_na = np.zeros(3)
    # Create mesh
    V, E = make_mesh_corner(0.1)
    # Solve
    eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
    err_norm_total_na[0] = np.sum(errnorm_T)
    eta_R_na[0] = np.sqrt(np.sum(eta_TR**2))
    plot_dof_na[0] = V.shape[0]
    # Create mesh
    V, E = make_mesh_corner(0.02)
    # Solve
    eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
    err_norm_total_na[1] = np.sum(errnorm_T)
    eta_R_na[1] = np.sqrt(np.sum(eta_TR**2))
    plot_dof_na[1] = V.shape[0]
    # Create mesh
    V, E = make_mesh_corner(0.004)
    # Solve
    eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
    err_norm_total_na[2] = np.sum(errnorm_T)
    eta_R_na[2] = np.sqrt(np.sum(eta_TR**2))
    plot_dof_na[2] = V.shape[0]
    # Plot convergence
    t = np.linspace(50,400,2)
    f1 = (t**(-1))*5
    errev1 = plt.figure()
    errev1.set_size_inches(6,4)
    plt.loglog(plot_dof,eta_R)
    plt.loglog(plot_dof_na,eta_R_na)
    plt.loglog(t,f1,'--')
    plt.xlabel('number of dof', fontsize=ft)
    plt.ylabel('$\eta_R$', fontsize=ft)
    plt.legend(('adaptive FEM','non-adaptive FEM','$O(N_{dof}^{-1})$'), loc='best', fontsize=ft2)
    errev1.savefig('plot_adapt_conv.png',dpi=300)
    # Plot convergence
    t = np.linspace(1,20,2)
    f1 = (t**(-1))*5
    eier = plt.figure()
    eier.set_size_inches(6,4)
    plt.loglog(numiter,eta_R)
    plt.loglog(t,f1,'--')
    plt.xlabel('number of iteration', fontsize=ft)
    plt.ylabel('$\eta_R$', fontsize=ft)
    plt.legend(('adaptive FEM','$O(N_{iter}^{-1})$'), loc='best', fontsize=ft2)
    eier.savefig('plot_adapt_iter.png',dpi=300)

def time_complexity1():
    # Initial mesh 1
    thv = np.array([0.1, 0.2, 0.3, 0.4, 1, 1.5, 3, 4, 5])
    # Set up variable for plotting
    num_of_ref = np.zeros(thv.size)
    ctime = np.zeros(thv.size)
    for i in range (0,thv.size):
        th = thv[i]
        # Set initial mesh
        V, E = make_mesh_corner(0.01)
        # Set up BVP
        convtest3 = make_convtest3()
        # Solve
        eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
        # Determine element to be meshed
        th_eta_TR = np.max(eta_TR)*th/100
        to_be_refined = eta_TR > th_eta_TR
        # Total number of element to be meshed
        nel_ref = np.sum(to_be_refined)
        # Time needed to refine
        start_refine = time.perf_counter()
        V, E = refine(V,E,V.shape[0],E.shape[0],to_be_refined)
        end_refine = time.perf_counter()
        # Save for plotting
        num_of_ref[i] = nel_ref
        ctime[i] = end_refine-start_refine

    # Initial mesh 2
    thv = np.array([0.1, 0.2, 0.3, 0.4, 1, 1.5, 3, 4, 5])
    # Set up variable for plotting
    num_of_ref2 = np.zeros(thv.size)
    ctime2 = np.zeros(thv.size)

    for i in range (0,thv.size):
        th = thv[i]
        # Set initial mesh
        V, E = make_mesh_corner(0.005)
        # Set up BVP
        convtest3 = make_convtest3()
        # Solve
        eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
        # Determine element to be meshed
        th_eta_TR = np.max(eta_TR)*th/100
        to_be_refined = eta_TR > th_eta_TR
        # Total number of element to be meshed
        nel_ref = np.sum(to_be_refined)
        # Time needed to refine
        start_refine = time.perf_counter()
        V, E = refine(V,E,V.shape[0],E.shape[0],to_be_refined)
        end_refine = time.perf_counter()
        # Save for plotting
        num_of_ref2[i] = nel_ref
        ctime2[i] = end_refine-start_refine

    # Initial mesh 3
    thv = np.array([0.1, 0.2, 0.3, 0.4, 1, 2])
    # Set up variable for plotting
    num_of_ref3 = np.zeros(thv.size)
    ctime3 = np.zeros(thv.size)

    for i in range (0,thv.size):
        th = thv[i]
        # Set initial mesh
        V, E = make_mesh_corner(0.003)
        # Set up BVP
        convtest3 = make_convtest3()
        # Solve
        eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
        # Determine element to be meshed
        th_eta_TR = np.max(eta_TR)*th/100
        to_be_refined = eta_TR > th_eta_TR
        # Total number of element to be meshed
        nel_ref = np.sum(to_be_refined)
        # Time needed to refine
        start_refine = time.perf_counter()
        V, E = refine(V,E,V.shape[0],E.shape[0],to_be_refined)
        end_refine = time.perf_counter()
        # Save for plotting
        num_of_ref3[i] = nel_ref
        ctime3[i] = end_refine-start_refine

    ft = 15
    ft2 = 13
    t = np.linspace(15,1200,2)
    f1 = (t**(0.2))*0.15
    errev1 = plt.figure()
    errev1.set_size_inches(6,4)
    plt.loglog(num_of_ref,ctime)
    plt.loglog(num_of_ref2,ctime2)
    plt.loglog(num_of_ref3,ctime3)
    plt.loglog(t,f1,'--')
    plt.xlabel('number of element to be refined', fontsize=ft)
    plt.ylabel('time needed for refinement (s)', fontsize=ft)
    plt.legend(('total Element = 400','total Element = 803','total Element = 1212','$O(N_{refine}^{0.3})$'), loc='best', fontsize=ft2)
    errev1.savefig('plot_complexity_const_nel.png',dpi=300)

def time_complexity2():
    # Initial mesh 1
    sizev = np.array([0.02, 0.01, 0.006])
    # Set up variable for plotting
    nel = np.zeros(sizev.size)
    ctime = np.zeros(sizev.size)
    for i in range (0,sizev.size):
        size = sizev[i]
        # Set initial mesh
        V, E = make_mesh_corner(size)
        # Set up BVP
        convtest3 = make_convtest3()
        # Solve
        eta_TR, errnorm_T, ferrnorm_T, err_norm_elem, u = error_estimate(V,E,convtest3)
        # Determine element to be meshed
        to_be_refined = np.zeros(E.shape[0], dtype=bool)
        to_be_refined[50:150] = True
        # Total number of element to be meshed
        # nel_ref = np.sum(to_be_refined)
        # Time needed to refine
        start_refine = time.perf_counter()
        V, E = refine(V,E,V.shape[0],E.shape[0],to_be_refined)
        end_refine = time.perf_counter()
        # Save for plotting
        nel[i] = E.shape[0]
        ctime[i] = end_refine-start_refine

    ft = 15
    ft2 = 13
    t = np.linspace(300,1200,2)
    f1 = (t**(2))*0.0006
    errev1 = plt.figure()
    errev1.set_size_inches(6,4)
    plt.loglog(nel,ctime)
    plt.loglog(t,f1,'--')
    plt.xlabel('total Element', fontsize=ft)
    plt.ylabel('time needed for refinement (s)', fontsize=ft)
    plt.legend(('100 element to be refined','$O(N_{elem}^{2})$'), loc='best', fontsize=ft2)
    errev1.savefig('plot_complexity_const_ref.png',dpi=300)

#=============================================================================#
# Run Test and Simulation
#=============================================================================#

# Start measuring time needed to do simulation
tic = time.perf_counter()


# Test residual-based error estimator for smooth problem
# error_est_test_smooth()

# Test convergece of the FEM solver
# testFEM()

# Test residual-based error estimator for problem with re-entrant corner
# error_est_test_corner()

# Test convergece of the FEM solver, re-entrant corner
# testFEM_corner()

# Test adaptive FEM using Rivara refinement
adaptive_test()

# Time complexity, constant number of total element
# time_complexity1()

# Time complexity, constant number of element to be refined
# time_complexity2()

#=============================================================================#
# Workshop
#=============================================================================#




toc = time.perf_counter()
print(f"\nTotal time = {toc - tic:0.4f} seconds")











