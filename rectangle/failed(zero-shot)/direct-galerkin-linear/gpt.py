import numpy as np
import time
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import gmres

# ============================================================
# Exact solution
# ============================================================

def u_exact(x, y):
    return (np.sinh(np.pi * x) / np.sinh(2*np.pi)) * np.cos(np.pi * y)

def grad_u_exact(x, y):
    ux = (np.pi * np.cosh(np.pi * x) / np.sinh(2*np.pi)) * np.cos(np.pi * y)
    uy = -(np.sinh(np.pi * x) / np.sinh(2*np.pi)) * np.pi * np.sin(np.pi * y)
    return ux, uy

# ============================================================
# Quadrature (order 8)
# ============================================================

nq = 8
gp, gw = leggauss(nq)
phi1 = (1 - gp)/2
phi2 = (1 + gp)/2

# ============================================================
# Mesh generation (linear boundary elements)
# ============================================================

def build_mesh(M):
    nodes = []
    elems = []
    btypes = []

    # bottom (Neumann)
    xs = np.linspace(0,2,M+1)
    for i in range(M):
        nodes.append([xs[i],0])
    nodes.append([2,0])

    # right (Dirichlet)
    ys = np.linspace(0,1,M+1)
    for i in range(1,M+1):
        nodes.append([2,ys[i]])

    # top (Neumann)
    xs = np.linspace(2,0,M+1)
    for i in range(1,M+1):
        nodes.append([xs[i],1])

    # left (Dirichlet)
    ys = np.linspace(1,0,M+1)
    for i in range(1,M):
        nodes.append([0,ys[i]])

    nodes = np.array(nodes)
    Nn = len(nodes)

    for i in range(Nn):
        elems.append((i,(i+1)%Nn))

    for i in range(Nn):
        x,y = nodes[i]
        if np.isclose(y,0) or np.isclose(y,1):
            btypes.append('N')
        else:
            btypes.append('D')

    return nodes, elems, btypes

# ============================================================
# Kernels
# ============================================================

def G(r):
    return -(1/(2*np.pi))*np.log(r)

def dGdn(dx,dy,r,nx,ny):
    return -(1/(2*np.pi))*(dx*nx+dy*ny)/(r**2)

# ============================================================
# BEM Galerkin Solver
# ============================================================

def solve_bem(M):

    t0 = time.time()
    nodes, elems, btypes = build_mesh(M)
    Nn = len(nodes)

    node_type = np.zeros(Nn,dtype=int)  # 0=D,1=N
    u_known = np.zeros(Nn)
    q_known = np.zeros(Nn)

    for i,(x,y) in enumerate(nodes):
        if btypes[i]=='D':
            node_type[i]=0
            u_known[i]=u_exact(x,y)
        else:
            node_type[i]=1
            q_known[i]=0.0

    A = np.zeros((Nn,Nn))
    rhs = np.zeros(Nn)

    # Precompute element geometry
    elem_data=[]
    for (n1,n2) in elems:
        x1,y1=nodes[n1]
        x2,y2=nodes[n2]
        tx=x2-x1; ty=y2-y1
        L=np.hypot(tx,ty)
        nx=ty/L; ny=-tx/L
        elem_data.append((n1,n2,x1,y1,x2,y2,nx,ny,L))

    # Double integration (Galerkin)
    for e_i,(i1,i2,x1i,y1i,x2i,y2i,nxi,nyi,Li) in enumerate(elem_data):

        xi_q = x1i*phi1 + x2i*phi2
        yi_q = y1i*phi1 + y2i*phi2
        Ji = Li/2

        for e_j,(j1,j2,x1j,y1j,x2j,y2j,nxj,nyj,Lj) in enumerate(elem_data):

            xj_q = x1j*phi1 + x2j*phi2
            yj_q = y1j*phi1 + y2j*phi2
            Jj = Lj/2

            for a,(inode,phi_i) in enumerate([(i1,phi1),(i2,phi2)]):
                for b,(jnode,phi_j) in enumerate([(j1,phi1),(j2,phi2)]):

                    valH = 0.0
                    valG = 0.0

                    for qi in range(nq):
                        for qj in range(nq):
                            dx = xi_q[qi]-xj_q[qj]
                            dy = yi_q[qi]-yj_q[qj]
                            r = np.hypot(dx,dy)
                            if r<1e-14:
                                continue
                            w = gw[qi]*gw[qj]*Ji*Jj
                            valH += phi_i[qi]*phi_j[qj]*dGdn(dx,dy,r,nxj,nyj)*w
                            valG += phi_i[qi]*phi_j[qj]*G(r)*w

                    if node_type[jnode]==0:
                        A[inode,jnode] += -valG
                        rhs[inode] += valH*u_known[jnode]
                    else:
                        A[inode,jnode] += valH
                        rhs[inode] += -valG*q_known[jnode]

        # jump term
        A[i1,i1]+=0.5*Li/2
        A[i2,i2]+=0.5*Li/2

    t_asm = time.time()

    sol,_ = gmres(A,rhs,atol=1e-10, callback_type='legacy')
    t_solve = time.time()

    u=np.zeros(Nn); q=np.zeros(Nn)
    for i in range(Nn):
        if node_type[i]==0:
            u[i]=u_known[i]
            q[i]=sol[i]
        else:
            u[i]=sol[i]
            q[i]=q_known[i]

    # Interior evaluation
    nxg,nyg=30,15
    xs=np.linspace(0.05,1.95,nxg)
    ys=np.linspace(0.05,0.95,nyg)
    XX,YY=np.meshgrid(xs,ys)
    u_num=np.zeros_like(XX)

    for e,(n1,n2,x1,y1,x2,y2,nxj,nyj,Lj) in enumerate(elem_data):
        xq = x1*phi1 + x2*phi2
        yq = y1*phi1 + y2*phi2
        J=Lj/2
        for qi in range(nq):
            for i in range(nxg):
                for j in range(nyg):
                    dx=XX[j,i]-xq[qi]
                    dy=YY[j,i]-yq[qi]
                    r=np.hypot(dx,dy)
                    if r<1e-14: continue
                    Gv=G(r); Hv=dGdn(dx,dy,r,nxj,nyj)
                    u_loc=u[n1]*phi1[qi]+u[n2]*phi2[qi]
                    q_loc=q[n1]*phi1[qi]+q[n2]*phi2[qi]
                    u_num[j,i]+= (Hv*u_loc - Gv*q_loc)*gw[qi]*J

    u_ex=u_exact(XX,YY)
    relL2=np.linalg.norm(u_num-u_ex)/np.linalg.norm(u_ex)

    t_eval=time.time()

    return Nn, t_asm-t0, t_solve-t_asm, t_eval-t_solve, t_eval-t0, relL2


# ============================================================
# Refinement study
# ============================================================

Ms=[10,20,40, 80, 160]
prev=None
print("M  DOFs  Asm(s) Solve(s) Eval(s) Total(s) RelL2   Rate")
for M in Ms:
    dof,ta,ts,te,tt,err=solve_bem(M)
    rate=np.log(prev/err)/np.log(2) if prev is not None else np.nan
    print(f"{M:<3}{dof:<6}{ta:7.3f}{ts:9.3f}{te:9.3f}{tt:9.3f}  {err:8.2e}{rate:7.2f}")
    prev=err